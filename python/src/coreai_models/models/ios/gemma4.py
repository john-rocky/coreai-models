# Gemma 4 E2B decode in Apple's OFFICIAL iOS STATIC-shape contract.
#
# Community port — NOT an Apple model. This is the RELEASABLE on-device decode path
# (fixed shape / no per-step memory growth) that replaces the DYNAMIC static-q1 core
# (ondevice/export_gemma4_decode.py). The dynamic core is numerically CORRECT (HF greedy
# 8/8) but NOT device-releasable: its `position_ids` length grows each step, so the runtime
# compiles + caches a new shape specialization per step -> ~200 MB/step -> jetsam OOM
# (see ondevice/CoreAIChat/IOS_FIXED_SHAPE_HANDOFF.md). Fixed shape stops that growth.
#
# It conforms to the iOS static-shape trick used by `export/ios.py` + `models/ios/{mistral,qwen3_5}.py`:
#
#   * FIXED-capacity KV cache, written at `in_step`, read WHOLE every step; an additive
#     `causal_mask` (0 / large-neg) masks the unwritten / future / out-of-window columns.
#     Every tensor shape is CONSTANT across steps -> no recompile -> the ANE static fast path.
#   * `in_step` (int32 scalar) selects the write column.
#   * `position_ids[1, 1]` carries ONLY the current decode token's absolute position (for RoPE).
#
# Gemma 4 specifics preserved verbatim from the HF-verified macOS port (`models/macos/gemma4_text.py`,
# 8/8 vs HF): DUAL head_dim (sliding 256 / full 512) -> DUAL KV caches + DUAL causal masks;
# DUAL RoPE (sliding theta=1e4 full-rotation / full theta=1e6 partial = 64 real + 192 zero freqs);
# per-head weighted Q/K RMSNorm + scale-free V RMSNorm; attention scale = 1.0; RMSNorm multiplies
# by weight directly; KV-sharing L15-34 (read producers L13 sliding / L14 full); the PLE per-layer
# skip + per-layer scalar; final softcap stays in the (separate) head bundle.
#
# This reuses the proven macOS gemma4 submodules (q/k/v projections + norms, the composite RoPE
# with its partial-rotation `freqs`, the gelu-tanh MLP, the PLE block) and changes ONLY the two
# things that made the dynamic graph dynamic: the KV fetch (dynamic narrow -> static fixed-capacity
# whole-read) and the position handling (growing position_ids -> in_step + causal-mask inputs) —
# exactly the strategy `models/ios/qwen3_5.py` used.
#
# v1 sliding cache is LINEAR (fixed capacity `ctx`, written at `in_step`) + a windowed causal mask,
# which is numerically identical to the verified non-ring `Gemma4DecodeStateful`. (A `ctx`-wide
# linear sliding cache costs only ~37 MB more than the W=512 ring; the ring is a later memory-only
# optimisation — the flat-memory property comes from the FIXED shape, not the cache width.)
#
# The giant embedding / PLE / lm_head tables stay OUT of this graph: the core consumes pre-gathered
# `inputs_embeds` + `per_layer_inputs` (supplied on device by the mmap front-end, Gemma4Gather.swift),
# and the tied lm_head + softcap is the separate head bundle.
from __future__ import annotations

import os

import torch
import torch.nn as nn

from coreai_models.models.macos.gemma4_text import (
    MLP,
    Gemma4TextConfig,
    ScaleFreeRMSNorm,
    _full_inv_freq,
    _sliding_inv_freq,
)
from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA

# Large finite negative for the additive attention mask (fp16-safe; avoids the NaN a true -inf can
# create in softmax and the -inf constants the device lowering rejects). Matches models/ios/qwen3_5.py.
MASK_NEG = -1.0e4

# Attention lowering: "sdpa" = the composite (fused) SDPA op that lowers on the MPSGraph runtime
# (the proven-on-GPU gemma4 path); "manual" = the explicit matmul/softmax decomposition. The device
# (ANE/GPU) runs through MPSGraph, which rejects the manual decomposition ("MLIR pass manager failed"),
# so the composite op is required on-device. Toggle via GEMMA4_IOS_ATTN for bisection.
_ATTN_IMPL = os.environ.get("GEMMA4_IOS_ATTN", "sdpa")


# --------------------------------------------------------------------------- #
# Fixed-capacity DUAL KV cache (iOS static trick), macOS [.,1,nkv,seq,head_dim] layout.
# --------------------------------------------------------------------------- #
class StaticDualKVCache:
    """Two fixed-capacity KV caches — SLIDING (head_dim 256) + FULL (head_dim 512) — kept in the
    macOS layout ``[n_slots, 1, n_kv_heads, ctx, head_dim]`` so the proven gemma4 attention math is
    unchanged. Unlike the macOS ``KVCache`` it does NOT narrow to ``seq_len``: the new column is
    written at ``in_step`` and the WHOLE fixed-length cache is read back; the ``causal_mask`` handles
    masking. Constant shapes everywhere -> no per-step recompile."""

    def __init__(
        self,
        sliding_k: torch.Tensor,
        sliding_v: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
    ) -> None:
        self._sk, self._sv, self._fk, self._fv = sliding_k, sliding_v, full_k, full_v

    @staticmethod
    def _write(cache: torch.Tensor, slot: int, in_step: torch.Tensor, col: torch.Tensor) -> None:
        """Write ``col`` ([1, n_kv, 1, head_dim], a single decode column) into ``cache``
        ([n_slots, 1, n_kv, ctx, head_dim]) at sequence position ``in_step`` for ``slot``."""
        device = cache.device
        z = torch.tensor((0,), dtype=torch.int32, device=device)
        s0 = torch.tensor((slot,), dtype=torch.int32, device=device)
        s1 = torch.tensor((slot + 1,), dtype=torch.int32, device=device)
        in_step = in_step.to(torch.int32).reshape(1)
        # begin = [slot, 0, 0, in_step, 0]; end = [slot+1, 1, n_kv, in_step+1, head_dim]
        begin = torch.cat([s0, z, z, in_step, z])
        end = torch.cat(
            [
                s1,
                torch.tensor((cache.size(1),), dtype=torch.int32, device=device),
                torch.tensor((cache.size(2),), dtype=torch.int32, device=device),
                in_step + 1,
                torch.tensor((cache.size(4),), dtype=torch.int32, device=device),
            ]
        )
        mutable_slice_update(x=cache, update=col.unsqueeze(0), begin=begin, end=end)

    @staticmethod
    def _slot(cache: torch.Tensor, slot: int) -> torch.Tensor:
        """Read one slot -> [1, n_kv, ctx, head_dim] (the whole fixed cache for that slot)."""
        return cache[slot]

    def write_and_read(
        self, is_full: bool, slot: int, in_step: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Producer / own-cache layers: write the new k/v column at ``in_step``, read the WHOLE
        fixed cache for ``slot`` ([1, n_kv, ctx, head_dim])."""
        kc, vc = (self._fk, self._fv) if is_full else (self._sk, self._sv)
        self._write(kc, slot, in_step, k)
        self._write(vc, slot, in_step, v)
        return self._slot(kc, slot), self._slot(vc, slot)

    def read(self, is_full: bool, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """KV-shared layers: read the producer ``slot`` (written earlier this step), no write."""
        kc, vc = (self._fk, self._fv) if is_full else (self._sk, self._sv)
        return self._slot(kc, slot), self._slot(vc, slot)


# --------------------------------------------------------------------------- #
# Static gemma4 attention (q=1 decode): per-head Q/K/V norms + dual RoPE + masked SDPA over the
# WHOLE fixed cache. Submodule names match models/macos/gemma4_text.py:Attention so the proven
# state-dict loads directly.
# --------------------------------------------------------------------------- #
class Gemma4StaticAttention(nn.Module):
    def __init__(self, cfg: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full = cfg.is_full(layer_idx)
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = hd = cfg.head_dim_of(layer_idx)

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, cfg.hidden_size, bias=False)

        self.q_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.v_norm = ScaleFreeRMSNorm(eps=cfg.rms_norm_eps)

        # Dual RoPE: same composite op + precomputed (type-specific) inv_freq as the macOS path.
        # Kept fp32 (NOT a buffer) so model.half() can't underflow the small full-rotation freqs.
        self.inv_freq = (_full_inv_freq(cfg) if self.is_full else _sliding_inv_freq(cfg)).float()
        self.rope = RoPE(scale=1.0)
        self.scale = 1.0  # gemma4 attention scale is 1.0 (QK-norm bounds the magnitudes)
        # Composite SDPA (fused, MPSGraph-friendly) — scale 1.0, no built-in causal/window mask
        # (we pass an explicit per-step mask). enable_gqa expands the single KV head internally.
        self.sdpa = SDPA(scale=1.0)

    def _project_q(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        return self.rope(q, position_ids=position_ids, freqs=self.inv_freq)

    def _project_kv(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, _ = x.shape
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_norm(k)
        k = self.rope(k, position_ids=position_ids, freqs=self.inv_freq)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_norm(v)
        return k, v

    def forward(
        self,
        x: torch.Tensor,                # [1, 1, hidden]
        position_ids: torch.Tensor,     # [1, 1] absolute position of the decode token
        in_step: torch.Tensor,          # int32 scalar (== position for a contiguous decode)
        kv: StaticDualKVCache,
        slot: int,
        write: bool,
        mask: torch.Tensor,             # [1, 1, 1, ctx] additive (0 / MASK_NEG), full OR sliding
    ) -> torch.Tensor:
        b, s, _ = x.shape  # s == 1
        H, nkv, hd = self.n_heads, self.n_kv_heads, self.head_dim

        q = self._project_q(x, position_ids)                  # [b, H, 1, hd]
        if write:
            k, v = self._project_kv(x, position_ids)          # [b, nkv, 1, hd]
            k_full, v_full = kv.write_and_read(self.is_full, slot, in_step, k, v)
        else:
            # KV-shared layer: read the producer's WHOLE cache (it ran earlier this step).
            k_full, v_full = kv.read(self.is_full, slot)
        # k_full / v_full: [b=1, nkv, ctx, hd].

        if _ATTN_IMPL == "sdpa":
            # Composite (fused) SDPA — the MPSGraph-friendly path (required on device). enable_gqa
            # expands the single KV head internally; the additive mask becomes a boolean attend mask
            # (0 == attend). q [b,H,1,hd], k/v [b,nkv,ctx,hd] -> out [b,H,1,hd].
            out = self.sdpa(query=q, key=k_full, value=v_full, attn_mask=(mask == 0))
        else:
            # Manual decomposition (does NOT lower on MPSGraph — kept for CPU/debug numerical parity).
            ctx = k_full.shape[-2]
            rep = H // nkv
            ke = k_full.unsqueeze(2).expand(b, nkv, rep, ctx, hd).reshape(b, H, ctx, hd)
            ve = v_full.unsqueeze(2).expand(b, nkv, rep, ctx, hd).reshape(b, H, ctx, hd)
            scores = (q @ ke.transpose(-1, -2)) * self.scale + mask   # [b,H,1,ctx]
            out = scores.softmax(dim=-1) @ ve                          # [b,H,1,hd]

        out = out.permute(0, 2, 1, 3).reshape(b, s, H * hd)    # [b, 1, H*hd]
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Static gemma4 decoder layer (q=1): mirrors models/macos/gemma4_text.py:DecoderLayer.forward_stateful
# exactly (norms, MLP, PLE skip, per-layer scalar) — only the attention internals are the static path.
# --------------------------------------------------------------------------- #
class Gemma4StaticDecoderLayer(nn.Module):
    def __init__(self, cfg: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        h = cfg.hidden_size
        eps = cfg.rms_norm_eps

        self.self_attn = Gemma4StaticAttention(cfg, layer_idx)
        self.mlp = MLP(h, cfg.intermediate_of(layer_idx))

        self.input_layernorm = RMSNorm(h, eps=eps)
        self.post_attention_layernorm = RMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps=eps)

        self.ple_dim = cfg.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Linear(h, self.ple_dim, bias=False)
        self.per_layer_projection = nn.Linear(self.ple_dim, h, bias=False)
        self.post_per_layer_input_norm = RMSNorm(h, eps=eps)
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        in_step: torch.Tensor,
        per_layer_input: torch.Tensor,
        kv: StaticDualKVCache,
        slot: int,
        write: bool,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_ids, in_step, kv, slot, write, mask)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        # PLE: gated per-layer skip connection.
        residual = x
        x = self.per_layer_input_gate(x)
        x = nn.functional.gelu(x, approximate="tanh")
        x = x * per_layer_input
        x = self.per_layer_projection(x)
        x = self.post_per_layer_input_norm(x)
        x = residual + x

        return x * self.layer_scalar


# --------------------------------------------------------------------------- #
# Static gemma4 decode model (q=1): dual KV routing + dual masks, final norm.
# --------------------------------------------------------------------------- #
class Gemma4StaticModel(nn.Module):
    def __init__(self, cfg: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = cfg
        self.layers = nn.ModuleList(
            [Gemma4StaticDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        # (is_full, slot, write) per layer — same routing as the macOS stateful decode.
        self._route, _, _ = cfg.stateful_routing()

    def forward(
        self,
        inputs_embeds: torch.Tensor,        # [1, 1, hidden]
        per_layer_inputs: torch.Tensor,     # [1, 1, num_layers, ple_dim]
        position_ids: torch.Tensor,         # [1, 1]
        in_step: torch.Tensor,              # int32 scalar
        causal_mask_full: torch.Tensor,     # [1, 1, 1, ctx]
        causal_mask_sliding: torch.Tensor,  # [1, 1, 1, ctx]
        kv: StaticDualKVCache,
    ) -> torch.Tensor:
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            is_full, slot, write = self._route[i]
            mask = causal_mask_full if is_full else causal_mask_sliding
            x = layer(
                x, position_ids, in_step, per_layer_inputs[:, :, i, :], kv, slot, write, mask
            )
        return self.norm(x)


# --------------------------------------------------------------------------- #
# Decode core (inputs_embeds -> hidden) for the head-split export, mirroring the macOS
# Gemma4DecodeStateful: the giant embed/PLE/lm_head tables stay on the device front-end; the
# converted core holds only the 35-layer transformer.
# --------------------------------------------------------------------------- #
class Gemma4StaticDecodeCore(nn.Module):
    """Static q=1 dual-KV decode core. forward: ``inputs_embeds`` [1,1,hidden],
    ``per_layer_inputs`` [1,1,num_layers,ple_dim], ``position_ids`` [1,1], ``in_step`` scalar,
    ``causal_mask_full`` [1,1,1,ctx], ``causal_mask_sliding`` [1,1,1,ctx] + 4 dual-KV states
    (sliding/full k/v) -> final-norm hidden [1,1,hidden]. All shapes static."""

    # Opt OUT of composite externalization (same as macOS Gemma4DecodeStateful): the decode graph
    # lowers fine without externalize and there is no separate PLE front-end inside this wrapper.
    coreai_externalize_specs: tuple = ()

    def __init__(self, cfg: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = cfg
        self.model = Gemma4StaticModel(cfg)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        position_ids: torch.Tensor,
        in_step: torch.Tensor,
        causal_mask_full: torch.Tensor,
        causal_mask_sliding: torch.Tensor,
        sliding_k: torch.Tensor,
        sliding_v: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
    ) -> torch.Tensor:
        kv = StaticDualKVCache(sliding_k, sliding_v, full_k, full_v)
        return self.model(
            inputs_embeds, per_layer_inputs, position_ids, in_step,
            causal_mask_full, causal_mask_sliding, kv,
        )


# State names surfaced via export_to_coreai(state_names=...) and the Swift runtime.
STATIC_DECODE_STATE_NAMES = ("slidingKeyCache", "slidingValueCache", "fullKeyCache", "fullValueCache")

# Input names (order matches Gemma4StaticDecodeCore.forward's non-state args).
STATIC_DECODE_INPUT_NAMES = (
    "inputs_embeds",
    "per_layer_inputs",
    "position_ids",
    "in_step",
    "causal_mask_full",
    "causal_mask_sliding",
)


def build_static_decode_state(
    cfg: Gemma4TextConfig, max_ctx: int, dtype: torch.dtype = torch.float16
) -> dict[str, torch.Tensor]:
    """Allocate the FIXED-capacity dual-KV state tensors (all zero).

    Sliding cache (12 slots, head_dim 256) + full cache (3 slots, head_dim 512), each
    ``[n_slots, 1, n_kv_heads, max_ctx, head_dim]`` (LINEAR, written at ``in_step``)."""
    _, n_sliding, n_full = cfg.stateful_routing()
    nkv = cfg.num_key_value_heads
    return {
        "sliding_k": torch.zeros(n_sliding, 1, nkv, max_ctx, cfg.head_dim, dtype=dtype),
        "sliding_v": torch.zeros(n_sliding, 1, nkv, max_ctx, cfg.head_dim, dtype=dtype),
        "full_k": torch.zeros(n_full, 1, nkv, max_ctx, cfg.global_head_dim, dtype=dtype),
        "full_v": torch.zeros(n_full, 1, nkv, max_ctx, cfg.global_head_dim, dtype=dtype),
    }


def build_causal_mask_full(
    in_step: int, max_ctx: int, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Additive FULL-attention decode mask ``[1,1,1,max_ctx]``: 0 for cache columns
    ``[0 .. in_step]`` (causal, already written), MASK_NEG for unwritten / future columns."""
    mask = torch.full((1, 1, 1, max_ctx), MASK_NEG, dtype=dtype)
    mask[..., : in_step + 1] = 0.0
    return mask


def build_causal_mask_sliding(
    in_step: int, max_ctx: int, window: int, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Additive SLIDING-attention decode mask ``[1,1,1,max_ctx]`` for a LINEAR sliding cache:
    0 for columns ``(in_step - window, in_step]`` (causal AND inside the window of ``window``),
    MASK_NEG elsewhere. Matches HF gemma sliding window [pos-W+1, pos]."""
    mask = torch.full((1, 1, 1, max_ctx), MASK_NEG, dtype=dtype)
    lo = max(0, in_step - window + 1)
    mask[..., lo : in_step + 1] = 0.0
    return mask
