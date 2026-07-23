# Gemma 4 E2B decode re-authored in Apple's OFFICIAL iOS static-shape (ANE) contract.
#
# Community port — NOT an Apple model. This is the PROPER iOS/ANE path (Phase 2): the Phase-1
# shortcut (models/ios/gemma4.py) reused the macOS composite ops + macOS converter and runs on the
# iPhone GPU only — the ANE compiler REJECTS the fused composite SDPA. This file conforms to the iOS
# contract that LOWERS ON ANE, the same one used by `export/ios.py` + `models/ios/{mistral,qwen3_5_ios}.py`
# and proven (qwen sibling) to load+specialize through the official `set_static_shape_config` + KV
# `HardwareConstraints` (IOSurface) pipeline:
#
#   * Channels-first `[B, C, 1, L]` tensors everywhere; ALL projections are `Conv2d` 1x1.
#   * Fixed-capacity iOS KV cache `[n_slots, 1, n_kv*head_dim, 1, max_ctx]` (seq LAST, update on
#     dim 4 via `KVCacheHandler`); each step writes the new k/v column at `in_step` and the per-head
#     iOS `SDPA` reads the WHOLE cache, masked by an additive `causal_mask` `[1, max_ctx, 1, q]`.
#     Every shape is constant across steps — the static config the ANE wants.
#
# Gemma 4's NEW-to-the-iOS-path feature is the DUAL head_dim (sliding 256 / full 512), which a single
# uniform `KVCacheHandler` can't hold. So this port carries TWO KV caches as Core AI state — a SLIDING
# cache (head_dim 256) and a FULL cache (head_dim 512), each its own `KVCacheHandler` with the official
# IOSurface constraints — plus DUAL causal masks and DUAL RoPE. Both states are KV caches (the kind
# the official mistral/qwen3 path already puts in ANE state), so this is a lower-risk extension than
# qwen3.5's conv/rec SSM state. (De-risk slice _iso_gemma4_ane_slice.py confirmed the dual-KV + dual
# mask + PLE + gemma norms lower+specialize on the macOS GPU delegate via this pipeline.)
#
# RoPE cos/sin are GRAPH INPUTS (a "sidecar"), looked up by position on the Swift side from precomputed
# tables — NOT baked constant tables gathered in-graph. This matches CoreML-LLM's 34 tok/s gemma4 recipe
# and the handoff brief, and it is REQUIRED here: baking gemma4's RoPE tables as in-graph constants (the
# mistral-style `RoPECache`) trips the macOS-27 MPSGraph `CanonicalizeCopyWithConstraints` pass (SIGSEGV
# at specialization) once the RoPE'd key flows into the IOSurface-constrained KV cache — dropping the
# constant tables in favour of runtime cos/sin inputs avoids the constant-folding that crashes the pass.
# Build the per-position cos/sin with `build_rope_tables` (host/Swift precompute) + `rope_cos_sin_at`.
#
# Other gemma 4 specifics preserved from the HF-verified macOS port (`models/macos/gemma4_text.py`,
# 8/8 vs HF): per-head weighted Q/K RMSNorm + scale-free V RMSNorm (before RoPE); attention scale = 1.0;
# RMSNorm multiplies by weight directly (the iOS `RMSNorm` primitive, NOT `RMSNormPlusOne`); DUAL RoPE
# (sliding theta=1e4 full-rotation over hd 256 / full theta=1e6 "proportional" = 64 real + 192 zero
# freqs over hd 512) seeded from the SAME `_sliding_inv_freq` / `_full_inv_freq` the macOS port uses;
# KV-sharing L15-34 (read the producers' caches); gelu-tanh gated MLP (double-wide on the shared
# layers); PLE per-layer gated skip + per-layer scalar; final softcap stays in the (separate) head
# bundle. The macOS composite RoPE is `interleaved=False` = GPT-NeoX half-split, exactly what the iOS
# `apply_rope` / `rotate_half` implement, so the rotation matches.
#
# The transformer core consumes pre-gathered `inputs_embeds` + `per_layer_inputs` (supplied on device
# by the mmap front-end, Gemma4Gather.swift) and emits `hidden`; the giant embedding / PLE / lm_head
# tables stay OUT of this graph (head + softcap = the separate head bundle).
from __future__ import annotations

import os

import torch
import torch.nn as nn

from coreai_models.models.macos.gemma4_text import (
    Gemma4TextConfig,
    _full_inv_freq,
    _sliding_inv_freq,
)
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import apply_rope

# Large finite negative for the additive attention mask (fp16-safe; avoids the NaN a true -inf can
# create in softmax and the -inf constants the device lowering rejects). Matches models/ios/qwen3_5_ios.py.
MASK_NEG = -1.0e4

# The device ANE (MPSGraph) REJECTS the native softmax op ("MLIR pass manager failed" at load on the
# iOS-27 beta) — the same lesson CoreML-LLM's `ane_ops.ane_softmax` encodes ("avoids the native softmax
# op because earlier iOS/chip combinations rejected it on ANE"). So gemma4's attention uses a DECOMPOSED
# softmax (max/sub/exp/sum/div) by default. Set GEMMA4_ANE_NATIVE_SOFTMAX=1 to A/B the native op.
_NATIVE_SOFTMAX = os.environ.get("GEMMA4_ANE_NATIVE_SOFTMAX", "0") == "1"


# --------------------------------------------------------------------------- #
# Scale-free RMSNorm (gemma4 V-norm): normalize over the last dim, no learnable scale.
# (The iOS `RMSNorm` primitive always multiplies by a weight; gemma4's V-norm has none.)
# --------------------------------------------------------------------------- #
class ScaleFreeRMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        with torch.device("cpu"):
            self._eps = nn.Buffer(torch.tensor(eps), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        inv = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + self._eps)
        return (xf * inv).to(dtype)


# --------------------------------------------------------------------------- #
# iOS-contract MLP: channels-first Conv2d, gelu-tanh gated (gemma4 uses gelu_pytorch_tanh,
# NOT the silu of primitives/ios/mlp.py).
# --------------------------------------------------------------------------- #
class Gemma4ANEMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.up_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.down_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, q, _, dim = x.shape
        x = x.reshape(b * q, dim, 1, 1)
        gate = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        down = self.down_proj(gate * self.up_proj(x))
        return down.reshape(b, q, 1, dim)


# --------------------------------------------------------------------------- #
# Per-head iOS SDPA with a DECOMPOSED softmax. Mirrors primitives/ios/sdpa.SDPA's head-split layout
# (each head computed individually = the ANE-friendly structure) but replaces the native `.softmax`
# op — which the device ANE/MPSGraph rejects ("MLIR pass manager failed") — with the max/sub/exp/sum/div
# form CoreML-LLM uses on this exact device (`ane_ops.ane_softmax`). scale is applied to the key.
# --------------------------------------------------------------------------- #
class Gemma4ANESDPA(nn.Module):
    def __init__(self, head_dim: int, scale: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        with torch.device("cpu"):
            self._scale = nn.Buffer(torch.tensor(scale), persistent=False)

    def forward(
        self,
        query: torch.Tensor,        # [b, H*hd, 1, s]
        key: torch.Tensor,          # [b, HKV*hd, 1, L]
        value: torch.Tensor,        # [b, HKV*hd, 1, L]
        causal_mask: torch.Tensor,  # [1, L, 1, s]  additive
    ) -> torch.Tensor:
        key = key.transpose(-3, -1) * self._scale          # [b, L, 1, HKV*hd]
        queries = query.split(self.head_dim, dim=1)        # H x [b, hd, 1, s]
        keys = list(key.split(self.head_dim, dim=-1))      # HKV x [b, L, 1, hd]
        n_heads = len(queries)
        for i in range(len(keys)):
            keys[i] = keys[i].permute(0, 2, 3, 1)          # [b, 1, hd, L]
        kv_group = n_heads // len(keys)

        scores = []
        for h in range(n_heads):
            q = queries[h].permute(0, 2, 3, 1)             # [b, 1, s, hd]
            attn = q @ keys[h // kv_group]                 # [b, 1, s, L]
            scores.append(attn.permute(0, 3, 1, 2))        # [b, L, 1, s]
        full = torch.cat(scores, dim=2)                    # [b, L, H, s]
        masked = full + torch.cat([causal_mask] * n_heads, dim=2)

        # Softmax over the key axis (dim 1). DECOMPOSED (ANE-friendly) unless toggled to native.
        if _NATIVE_SOFTMAX:
            full = masked.softmax(1)
        else:
            ms = masked.to(torch.float16)
            mx = ms.max(dim=1, keepdim=True).values
            e = torch.exp(ms - mx)
            full = e / e.sum(dim=1, keepdim=True)

        scores = full.split(1, dim=2)
        values = list(value.split(self.head_dim, dim=1))   # HKV x [b, hd, 1, L]
        for i in range(len(values)):
            values[i] = values[i].permute(0, 2, 3, 1).squeeze(1)   # [b, L, hd]
        weights = []
        for h in range(n_heads):
            s = scores[h].permute(0, 2, 3, 1).squeeze(1)   # [b, s, L]
            w = (s @ values[h // kv_group]).unsqueeze(1)   # [b, 1, s, hd]
            weights.append(w.permute(0, 3, 1, 2))          # [b, hd, 1, s]
        return torch.cat(weights, dim=1)                   # [b, H*hd, 1, s]


# --------------------------------------------------------------------------- #
# iOS-contract gemma4 attention: Conv2d q/k/v/o, per-head Q/K RMSNorm + scale-free V-norm,
# dual RoPE (cos/sin supplied as inputs by layer type), per-head iOS SDPA (scale 1.0) over the WHOLE
# fixed cache masked by the additive causal_mask. Writes its k/v to the cache at `in_step` (own-cache)
# or reads a producer slot whole (KV-shared layer).
# --------------------------------------------------------------------------- #
class Gemma4ANEAttention(nn.Module):
    def __init__(self, cfg: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full = cfg.is_full(layer_idx)
        self.n_heads = H = cfg.num_attention_heads
        self.n_kv_heads = HKV = cfg.num_key_value_heads
        self.head_dim = hd = cfg.head_dim_of(layer_idx)
        d = cfg.hidden_size

        self.q_proj = nn.Conv2d(d, H * hd, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(d, HKV * hd, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(d, HKV * hd, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(H * hd, d, kernel_size=1, bias=False)

        self.q_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.v_norm = ScaleFreeRMSNorm(eps=cfg.rms_norm_eps)

        # gemma4 attention scale is 1.0 (QK-norm bounds the magnitudes), not 1/sqrt(hd).
        self.sdpa = Gemma4ANESDPA(head_dim=hd, scale=1.0)

    def forward(
        self,
        x: torch.Tensor,            # [b, s, 1, hidden]
        rope_cos: torch.Tensor,     # [b, s, head_dim]   (this layer's RoPE type, supplied as input)
        rope_sin: torch.Tensor,     # [b, s, head_dim]
        in_step: torch.IntTensor,   # int32 scalar (write column)
        causal_mask: torch.Tensor,  # [1, max_ctx, 1, s]  (full OR sliding)
        cache: KVCacheHandler,
        slot: int,
        write: bool,
    ) -> torch.Tensor:
        b, s, _, _ = x.shape
        H, HKV, hd = self.n_heads, self.n_kv_heads, self.head_dim

        x = x.transpose(-3, -1)             # [b, hidden, 1, s]
        query = self.q_proj(x)              # [b, H*hd, 1, s]

        query = query.transpose(-3, -1).reshape(b, s, H, hd)
        query = self.q_norm(query).transpose(-2, -3)          # [b, H, s, hd]
        query = apply_rope(query, rope_cos, rope_sin)

        if write:
            key = self.k_proj(x)            # [b, HKV*hd, 1, s]
            value = self.v_proj(x)
            key = key.transpose(-3, -1).reshape(b, s, HKV, hd)
            key = self.k_norm(key).transpose(-2, -3)          # [b, HKV, s, hd]
            key = apply_rope(key, rope_cos, rope_sin)
            value = value.transpose(-3, -1).reshape(b, s, HKV, hd)
            value = self.v_norm(value).transpose(-2, -3)      # [b, HKV, s, hd]

            # Back to channels-first [b, n*hd, 1, s] for the cache + per-head SDPA.
            key = key.transpose(-2, -3).reshape(b, s, 1, HKV * hd).transpose(-3, -1)
            value = value.transpose(-2, -3).reshape(b, s, 1, HKV * hd).transpose(-3, -1)
            key, value = cache.update_and_fetch(slot, in_step, key, value, s)
        else:
            # KV-shared layer: read the producer slot's WHOLE cache (written earlier this step).
            key, value = cache.k_cache[slot], cache.v_cache[slot]

        # Channels-first query for the official per-head SDPA.
        query = query.transpose(-2, -3).reshape(b, s, 1, H * hd).transpose(-3, -1)
        out = self.sdpa(query, key, value, causal_mask)       # [b, H*hd, 1, s]
        out = self.o_proj(out)
        return out.transpose(-3, -1)        # [b, s, 1, hidden]


# --------------------------------------------------------------------------- #
# iOS-contract gemma4 decoder layer: mirrors macOS DecoderLayer.forward_stateful (norms, MLP,
# PLE gated skip, per-layer scalar) with the attention internals on the iOS static path.
# --------------------------------------------------------------------------- #
class Gemma4ANEDecoderLayer(nn.Module):
    def __init__(self, cfg: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full = cfg.is_full(layer_idx)
        d = cfg.hidden_size
        eps = cfg.rms_norm_eps

        self.self_attn = Gemma4ANEAttention(cfg, layer_idx)
        self.mlp = Gemma4ANEMLP(d, cfg.intermediate_of(layer_idx))

        self.input_layernorm = RMSNorm(d, eps=eps)
        self.post_attention_layernorm = RMSNorm(d, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(d, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(d, eps=eps)

        self.ple_dim = cfg.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Conv2d(d, self.ple_dim, kernel_size=1, bias=False)
        self.per_layer_projection = nn.Conv2d(self.ple_dim, d, kernel_size=1, bias=False)
        self.post_per_layer_input_norm = RMSNorm(d, eps=eps)
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

    def forward(
        self,
        x: torch.Tensor,                 # [b, s, 1, hidden]
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        per_layer_input: torch.Tensor,   # [b, s, 1, ple_dim]
        cache: KVCacheHandler,
        slot: int,
        write: bool,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, rope_cos, rope_sin, in_step, causal_mask, cache, slot, write)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        # PLE: gated per-layer skip connection (channels-first Conv2d gate + projection).
        b, s, _, d = x.shape
        residual = x
        g = self.per_layer_input_gate(x.reshape(b * s, d, 1, 1))      # [b*s, ple_dim, 1, 1]
        g = nn.functional.gelu(g, approximate="tanh").reshape(b, s, 1, self.ple_dim)
        g = g * per_layer_input
        g = self.per_layer_projection(g.reshape(b * s, self.ple_dim, 1, 1)).reshape(b, s, 1, d)
        g = self.post_per_layer_input_norm(g)
        x = residual + g

        return x * self.layer_scalar


# --------------------------------------------------------------------------- #
# iOS-contract gemma4 decode model: dual KV caches (sliding 256 / full 512) carried as Core AI state,
# dual RoPE supplied as inputs, dual causal masks, KV-share routing, final norm.
# --------------------------------------------------------------------------- #
class Gemma4ANEModel(nn.Module):
    def __init__(self, cfg: Gemma4TextConfig, max_ctx: int) -> None:
        super().__init__()
        self.config = cfg
        self.max_ctx = max_ctx
        self.layers = nn.ModuleList(
            [Gemma4ANEDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

        # (is_full, slot, write) per layer — same routing as the macOS stateful decode.
        self._route, n_sliding, n_full = cfg.stateful_routing()
        nkv = cfg.num_key_value_heads
        # Two KV caches: sliding (head_dim 256) + full (head_dim 512), official iOS layout.
        self.sliding_cache = KVCacheHandler(max(n_sliding, 1), nkv * cfg.head_dim)
        self.full_cache = KVCacheHandler(max(n_full, 1), nkv * cfg.global_head_dim)

    def forward(
        self,
        inputs_embeds: torch.Tensor,        # [b, s, 1, hidden]
        per_layer_inputs: torch.Tensor,     # [b, s, num_layers, ple_dim]
        in_step: torch.IntTensor,           # int32 scalar
        causal_mask_full: torch.Tensor,     # [1, max_ctx, 1, s]
        causal_mask_sliding: torch.Tensor,  # [1, max_ctx, 1, s]
        rope_cos_sliding: torch.Tensor,     # [b, s, head_dim]
        rope_sin_sliding: torch.Tensor,
        rope_cos_full: torch.Tensor,        # [b, s, global_head_dim]
        rope_sin_full: torch.Tensor,
        sliding_key_cache: torch.Tensor,
        sliding_value_cache: torch.Tensor,
        full_key_cache: torch.Tensor,
        full_value_cache: torch.Tensor,
    ) -> torch.Tensor:
        self.sliding_cache.register_kv_cache(sliding_key_cache, sliding_value_cache)
        self.full_cache.register_kv_cache(full_key_cache, full_value_cache)

        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            is_full, slot, write = self._route[i]
            cache = self.full_cache if is_full else self.sliding_cache
            mask = causal_mask_full if is_full else causal_mask_sliding
            cos = rope_cos_full if is_full else rope_cos_sliding
            sin = rope_sin_full if is_full else rope_sin_sliding
            x = layer(
                x, cos, sin, in_step, mask,
                per_layer_inputs[:, :, i, :].unsqueeze(-2),   # [b, s, 1, ple_dim]
                cache, slot, write,
            )
        return self.norm(x)


# --------------------------------------------------------------------------- #
# Decode core (inputs_embeds -> hidden) for the head-split export — the giant embed/PLE/lm_head tables
# stay on the device front-end (Gemma4Gather.swift); this converted graph is the transformer only.
# --------------------------------------------------------------------------- #
class Gemma4ANEDecodeCore(nn.Module):
    # Opt OUT of composite externalization (no separate PLE front-end inside this wrapper).
    coreai_externalize_specs: tuple = ()

    def __init__(self, cfg: Gemma4TextConfig, max_ctx: int) -> None:
        super().__init__()
        self.config = cfg
        self.max_ctx = max_ctx
        self.model = Gemma4ANEModel(cfg, max_ctx)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask_full: torch.Tensor,
        causal_mask_sliding: torch.Tensor,
        rope_cos_sliding: torch.Tensor,
        rope_sin_sliding: torch.Tensor,
        rope_cos_full: torch.Tensor,
        rope_sin_full: torch.Tensor,
        sliding_key_cache: torch.Tensor,
        sliding_value_cache: torch.Tensor,
        full_key_cache: torch.Tensor,
        full_value_cache: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            inputs_embeds, per_layer_inputs, in_step,
            causal_mask_full, causal_mask_sliding,
            rope_cos_sliding, rope_sin_sliding, rope_cos_full, rope_sin_full,
            sliding_key_cache, sliding_value_cache, full_key_cache, full_value_cache,
        )


# State names surfaced via the converter (state_names=...) and the Swift runtime. Kept identical to
# the forward parameter names (snake_case) so torch.export kwargs / dynamic_shapes bind directly and
# the converter assigns these names to the mutated inputs in order.
STATIC_DECODE_STATE_NAMES = (
    "sliding_key_cache",
    "sliding_value_cache",
    "full_key_cache",
    "full_value_cache",
)

# Non-state input names (order matches Gemma4ANEDecodeCore.forward's non-state args).
STATIC_DECODE_INPUT_NAMES = (
    "inputs_embeds",
    "per_layer_inputs",
    "in_step",
    "causal_mask_full",
    "causal_mask_sliding",
    "rope_cos_sliding",
    "rope_sin_sliding",
    "rope_cos_full",
    "rope_sin_full",
)


def build_static_decode_state(
    cfg: Gemma4TextConfig, max_ctx: int, dtype: torch.dtype = torch.float16
) -> dict[str, torch.Tensor]:
    """Allocate the FIXED-capacity dual-KV state tensors (all zero), official iOS layout
    `[n_slots, 1, n_kv*head_dim, 1, max_ctx]` (seq LAST, update on dim 4)."""
    _, n_sliding, n_full = cfg.stateful_routing()
    nkv = cfg.num_key_value_heads
    sk = nkv * cfg.head_dim
    fk = nkv * cfg.global_head_dim
    return {
        "sliding_key_cache": torch.zeros(max(n_sliding, 1), 1, sk, 1, max_ctx, dtype=dtype),
        "sliding_value_cache": torch.zeros(max(n_sliding, 1), 1, sk, 1, max_ctx, dtype=dtype),
        "full_key_cache": torch.zeros(max(n_full, 1), 1, fk, 1, max_ctx, dtype=dtype),
        "full_value_cache": torch.zeros(max(n_full, 1), 1, fk, 1, max_ctx, dtype=dtype),
    }


def build_rope_tables(
    cfg: Gemma4TextConfig, max_ctx: int, dtype: torch.dtype = torch.float16
) -> dict[str, torch.Tensor]:
    """Precompute the per-position RoPE cos/sin tables (sliding head_dim 256 / full head_dim 512),
    GPT-NeoX half-split: `emb = cat((freqs, freqs))`. The Swift sidecar (or the export reference)
    indexes these by position to supply the `rope_cos/sin_{sliding,full}` graph INPUTS."""
    def table(inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = torch.arange(max_ctx, dtype=torch.float32)
        freqs = seq[:, None] * inv_freq.float()[None, :]   # [max_ctx, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)            # [max_ctx, dim]
        return torch.cos(emb).to(dtype), torch.sin(emb).to(dtype)

    cos_s, sin_s = table(_sliding_inv_freq(cfg))
    cos_f, sin_f = table(_full_inv_freq(cfg))
    return {"cos_sliding": cos_s, "sin_sliding": sin_s, "cos_full": cos_f, "sin_full": sin_f}


def rope_cos_sin_at(
    tables: dict[str, torch.Tensor], position: int
) -> dict[str, torch.Tensor]:
    """Look up the 4 RoPE inputs for a single decode position -> each `[1, 1, head_dim]`."""
    return {
        "rope_cos_sliding": tables["cos_sliding"][position].reshape(1, 1, -1),
        "rope_sin_sliding": tables["sin_sliding"][position].reshape(1, 1, -1),
        "rope_cos_full": tables["cos_full"][position].reshape(1, 1, -1),
        "rope_sin_full": tables["sin_full"][position].reshape(1, 1, -1),
    }


def build_causal_mask_full(
    in_step: int, max_ctx: int, q: int = 1, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Additive FULL-attention iOS-SDPA mask `[1, max_ctx, 1, q]`: 0 for cache columns
    `[0 .. in_step]` (causal, already written), MASK_NEG for unwritten / future columns."""
    mask = torch.full((1, max_ctx, 1, q), MASK_NEG, dtype=dtype)
    mask[:, : in_step + 1] = 0.0
    return mask


def build_causal_mask_sliding(
    in_step: int, max_ctx: int, window: int, q: int = 1, dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Additive SLIDING-attention iOS-SDPA mask `[1, max_ctx, 1, q]` for a LINEAR sliding cache:
    0 for columns `(in_step - window, in_step]` (causal AND inside the window), MASK_NEG else.
    Matches HF gemma sliding window [pos-W+1, pos]."""
    mask = torch.full((1, max_ctx, 1, q), MASK_NEG, dtype=dtype)
    lo = max(0, in_step - window + 1)
    mask[:, lo : in_step + 1] = 0.0
    return mask
