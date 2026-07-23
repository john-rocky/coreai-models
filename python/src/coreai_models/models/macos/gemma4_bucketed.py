# Community port — NOT an Apple model.
"""Gemma 4 E2B BUCKETED-DYNAMIC decode core (Session B / macOS speed) — ADDITIVE.

The macOS decode is OVERHEAD-bound: the dynamic `Gemma4DecodeStateful` re-specializes (recompiles)
the graph every step because `position_ids` length grows (a new shape per step ≈ the whole per-step
cost; see `ondevice/_prof_decode_gemma4.py`). Driving the SAME graph with a FIXED shape collapses it
to one compile reused (~69× in Step 0). This core makes that correct for incremental decode.

It is the SAME math as `Gemma4DecodeStateful` (it reuses the very same `Gemma4TextModel` submodules and
weights — q/k/v projections + norms, the composite RoPE with partial-rotation freqs, the gelu-tanh MLP,
the PLE block, the dual head_dim KV routing) and changes ONLY the two things that made the graph dynamic:
  * KV fetch: a FIXED-capacity (bucket B) dual cache, written at `in_step`, read as a fixed-length
    `narrow(0, B)` slice (NOT the growing `narrow(0, seq_len)`),
  * position handling: the query carries `position_ids[1, 1]` (its absolute position, for RoPE) and the
    write column / mask are derived from it — so the only varying quantity is a VALUE, not a shape.
A fixed-length read includes unwritten / future / out-of-window columns, so explicit additive masks
(`causal_mask_full` / `causal_mask_sliding`) zero them out through the composite SDPA's `attn_mask`.

⚠️ macOS-27 beta Mac-GPU constraint (Session B finding, `ondevice/_macos_speed_RESULTS.md`): a FULLY
STATIC graph that contains an in-graph KV write SIGTRAPs on the beta delegate. So this core is exported
with a DYNAMIC `Dim` on the cache/mask seq, then DRIVEN at a fixed bucket value B: the dynamic lowering
dodges the SIGTRAP while the fixed feed gives 1-compile-reuse. (The masks let the read be fixed-length;
the Dim keeps the graph dynamic-shaped.)

Boundary: this file is purely additive. It imports and REUSES `gemma4_text.py` submodules but does not
modify `Gemma4DecodeStateful.forward` / `GatherEmbeds` / `Gemma4Head` numerics (Session A reads those as
its oracle).
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from coreai_models.models.macos.gemma4_text import (
    DECODE_STATE_NAMES,
    Attention,
    DecoderLayer,
    Gemma4Head,
    Gemma4TextConfig,
    Gemma4TextModel,
)
from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.sdpa import SDPA

# Large finite negative for the additive masks (fp16-safe; matches models/ios/gemma4.py MASK_NEG).
MASK_NEG = -1.0e4

# Diagnostic: GEMMA4_BUCKETED_NOWRITE=1 skips the in-graph KV write (read-only) to isolate whether the
# data-dependent-index mutable_slice_update is the macOS-27-beta GPU crash trigger. RUN test only.
_NOWRITE = os.environ.get("GEMMA4_BUCKETED_NOWRITE", "0") == "1"

BUCKETED_DECODE_STATE_NAMES = DECODE_STATE_NAMES  # (slidingKeyCache, slidingValueCache, fullKeyCache, fullValueCache)
BUCKETED_INPUT_NAMES = ("inputs_embeds", "per_layer_inputs", "position_ids",
                        "causal_mask_full", "causal_mask_sliding")


# --------------------------------------------------------------------------- #
# Fixed-capacity (bucket B) dual KV cache: tensor `in_step` write, fixed `narrow(0, B)` read.
# Same macOS [n_slots, 1, nkv, B, head_dim] layout as KVCache, so the reused attention math is intact.
# --------------------------------------------------------------------------- #
class BucketedDualKVCache:
    def __init__(self, sk: torch.Tensor, sv: torch.Tensor, fk: torch.Tensor, fv: torch.Tensor) -> None:
        self._sk, self._sv, self._fk, self._fv = sk, sv, fk, fv

    @staticmethod
    def _write(cache: torch.Tensor, slot: int, in_step: torch.Tensor, col: torch.Tensor) -> None:
        """Write one decode column ``col`` ([1, nkv, 1, hd]) into ``cache``
        ([n_slots, 1, nkv, B, hd]) at sequence position ``in_step`` for ``slot``."""
        device = cache.device
        z = torch.tensor((0,), dtype=torch.int32, device=device)
        s0 = torch.tensor((slot,), dtype=torch.int32, device=device)
        s1 = torch.tensor((slot + 1,), dtype=torch.int32, device=device)
        step = in_step.to(torch.int32).reshape(1)
        begin = torch.cat([s0, z, z, step, z])
        end = torch.cat([
            s1,
            torch.tensor((cache.size(1),), dtype=torch.int32, device=device),
            torch.tensor((cache.size(2),), dtype=torch.int32, device=device),
            step + 1,
            torch.tensor((cache.size(4),), dtype=torch.int32, device=device),
        ])
        mutable_slice_update(x=cache, update=col.unsqueeze(0), begin=begin, end=end)

    @staticmethod
    def _read(cache: torch.Tensor, slot: int, B: int) -> torch.Tensor:
        """Fixed-length read of ``slot`` -> [1, nkv, B, hd] (the whole bucket)."""
        return cache.narrow(0, slot, 1).narrow(-2, 0, B).squeeze(0)

    def write_and_read(self, is_full: bool, slot: int, in_step: torch.Tensor,
                       k: torch.Tensor, v: torch.Tensor, B: int):
        kc, vc = (self._fk, self._fv) if is_full else (self._sk, self._sv)
        self._write(kc, slot, in_step, k)
        self._write(vc, slot, in_step, v)
        return self._read(kc, slot, B), self._read(vc, slot, B)

    def read(self, is_full: bool, slot: int, B: int):
        kc, vc = (self._fk, self._fv) if is_full else (self._sk, self._sv)
        return self._read(kc, slot, B), self._read(vc, slot, B)


def _bucketed_attention(attn: Attention, sdpa: SDPA, x: torch.Tensor, position_ids: torch.Tensor,
                        in_step: torch.Tensor, kv: BucketedDualKVCache, slot: int, write: bool,
                        mask: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """Reuse ``attn``'s proven submodules; route attention through the bucketed cache + masked SDPA.

    ``causal`` (diagnostic only, GEMMA4_BUCKETED_ATTN=causal): use the layer's own is_causal+window SDPA
    (the proven dynamic-core SDPA) and ignore the mask — isolates whether the explicit-mask path is what
    the macOS-27 beta GPU rejects, vs the fixed-shape write itself. NOT numerically correct (the pad is
    unmasked); it is a pure RUN test."""
    b, s, _ = x.shape  # s == 1
    H, hd = attn.n_heads, attn.head_dim
    B = mask.shape[-1]

    q = attn.q_proj(x).view(b, s, H, hd).permute(0, 2, 1, 3)
    q = attn.q_norm(q)
    q = attn.rope(q, position_ids=position_ids, freqs=attn.inv_freq)

    if write and not _NOWRITE:
        k, v = attn._project_kv(x, position_ids)             # [1, nkv, 1, hd] each, rope'd
        k, v = kv.write_and_read(attn.is_full, slot, in_step, k, v, B)
    else:
        # _NOWRITE (diagnostic): skip the in-graph mutable_slice_update entirely; read the cache only.
        # Isolates whether the data-dependent-index KV WRITE is the GPU-crash trigger (RUN test only).
        k, v = kv.read(attn.is_full, slot, B)                # KV-shared: producer ran earlier this step
    if causal:
        out = attn.sdpa(query=q, key=k, value=v)             # layer's is_causal+window SDPA (no mask)
    else:
        # masked composite SDPA (scale 1.0, NO is_causal — the explicit mask handles causal/window/pad).
        out = sdpa(query=q, key=k, value=v, attn_mask=(mask == 0))
    out = out.permute(0, 2, 1, 3).reshape(b, s, H * hd)
    return attn.o_proj(out)


def _bucketed_layer(layer: DecoderLayer, sdpa: SDPA, x: torch.Tensor, position_ids: torch.Tensor,
                    in_step: torch.Tensor, per_layer_input: torch.Tensor, kv: BucketedDualKVCache,
                    slot: int, write: bool, mask: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """Mirror DecoderLayer.forward_stateful (norms / MLP / PLE / layer_scalar) — bucketed attention only."""
    residual = x
    x = layer.input_layernorm(x)
    x = _bucketed_attention(layer.self_attn, sdpa, x, position_ids, in_step, kv, slot, write, mask, causal)
    x = layer.post_attention_layernorm(x)
    x = residual + x

    residual = x
    x = layer.pre_feedforward_layernorm(x)
    x = layer.mlp(x)
    x = layer.post_feedforward_layernorm(x)
    x = residual + x

    residual = x
    x = layer.per_layer_input_gate(x)
    x = nn.functional.gelu(x, approximate="tanh")
    x = x * per_layer_input
    x = layer.per_layer_projection(x)
    x = layer.post_per_layer_input_norm(x)
    x = residual + x

    return x * layer.layer_scalar


class Gemma4DecodeBucketed(nn.Module):
    """Fixed-shape (bucket B) dual-KV decode core. Reuses a ``Gemma4TextModel``'s weights; only the
    attention KV-fetch + position handling differ. forward:

      ``inputs_embeds`` [1,1,h], ``per_layer_inputs`` [1,1,num_layers,ld], ``position_ids`` [1,1],
      ``causal_mask_full`` [1,1,1,B], ``causal_mask_sliding`` [1,1,1,B] + the 4 dual-KV states
      (sliding/full k/v, each [n_slots,1,nkv,B,head_dim]) -> final-norm ``hidden`` [1,1,h].

    ``in_step`` (the write column) is derived from ``position_ids`` (== the decode position), so the
    signature has no extra scalar. Export with a DYNAMIC ``Dim`` on the cache/mask seq, drive at fixed B.
    """

    coreai_externalize_specs: tuple = ()  # same opt-out as Gemma4DecodeStateful

    def __init__(self, text_model: Gemma4TextModel) -> None:
        super().__init__()
        self.text_model = text_model
        self._route, _, _ = text_model.config.stateful_routing()
        self.sdpa = SDPA(scale=1.0)  # shared masked SDPA (stateless config)
        # Diagnostic toggle (GEMMA4_BUCKETED_ATTN=causal): use the proven is_causal+window SDPA and
        # ignore the masks — isolates the explicit-mask path as the GPU-crash trigger (RUN test only).
        self._causal = os.environ.get("GEMMA4_BUCKETED_ATTN", "masked") == "causal"

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        position_ids: torch.Tensor,        # [1, 1] absolute position of the decode token
        causal_mask_full: torch.Tensor,    # [1, 1, 1, B] additive (0 / MASK_NEG)
        causal_mask_sliding: torch.Tensor,
        sliding_k: torch.Tensor,
        sliding_v: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
    ) -> torch.Tensor:
        kv = BucketedDualKVCache(sliding_k, sliding_v, full_k, full_v)
        in_step = position_ids.reshape(-1)[0]  # scalar == decode position; write column index
        x = inputs_embeds
        for i, layer in enumerate(self.text_model.layers):
            is_full, slot, write = self._route[i]
            mask = causal_mask_full if is_full else causal_mask_sliding
            x = _bucketed_layer(layer, self.sdpa, x, position_ids, in_step,
                                per_layer_inputs[:, :, i, :], kv, slot, write, mask, self._causal)
        return self.text_model.norm(x)

    def build_macos_export_spec(self, target_dtype: torch.dtype, bucket: int, trace_bucket: int,
                                offset: int = 8) -> dict:
        """Reference inputs + dynamic shapes for the bucketed export. The cache/mask seq is a DYNAMIC
        ``Dim`` (so the graph dodges the beta static-write SIGTRAP); drive it at the fixed ``bucket``.

        ``trace_bucket`` is the example seq length to trace at (use != the runtime bucket so the dim is
        genuinely dynamic). ``offset`` is just a non-trivial example decode position."""
        cfg = self.text_model.config
        h, ld, nl = cfg.hidden_size, cfg.hidden_size_per_layer_input, cfg.num_hidden_layers
        _, n_sliding, n_full = cfg.stateful_routing()
        nkv = cfg.num_key_value_heads
        T = trace_bucket
        ref = {
            "inputs_embeds": torch.zeros(1, 1, h, dtype=target_dtype),
            "per_layer_inputs": torch.zeros(1, 1, nl, ld, dtype=target_dtype),
            "position_ids": torch.tensor([[offset]], dtype=torch.int32),
            "causal_mask_full": torch.zeros(1, 1, 1, T, dtype=target_dtype),
            "causal_mask_sliding": torch.zeros(1, 1, 1, T, dtype=target_dtype),
            "sliding_k": torch.zeros(n_sliding, 1, nkv, T, cfg.head_dim, dtype=target_dtype),
            "sliding_v": torch.zeros(n_sliding, 1, nkv, T, cfg.head_dim, dtype=target_dtype),
            "full_k": torch.zeros(n_full, 1, nkv, T, cfg.global_head_dim, dtype=target_dtype),
            "full_v": torch.zeros(n_full, 1, nkv, T, cfg.global_head_dim, dtype=target_dtype),
        }
        S = torch.export.Dim("S", min=2, max=4096)
        dynamic_shapes = {
            "inputs_embeds": {}, "per_layer_inputs": {}, "position_ids": {},
            "causal_mask_full": {3: S}, "causal_mask_sliding": {3: S},
            "sliding_k": {3: S}, "sliding_v": {3: S}, "full_k": {3: S}, "full_v": {3: S},
        }
        return {
            "reference_inputs": ref,
            "dynamic_shapes": dynamic_shapes,
            "input_names": BUCKETED_INPUT_NAMES,
            "output_names": ("hidden",),
            "state_names": BUCKETED_DECODE_STATE_NAMES,
        }


def build_bucketed_state(cfg: Gemma4TextConfig, bucket: int,
                         dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
    """Zero dual-KV state at fixed capacity ``bucket`` (LINEAR sliding + full; written at in_step)."""
    _, n_sliding, n_full = cfg.stateful_routing()
    nkv = cfg.num_key_value_heads
    return {
        "sliding_k": torch.zeros(n_sliding, 1, nkv, bucket, cfg.head_dim, dtype=dtype),
        "sliding_v": torch.zeros(n_sliding, 1, nkv, bucket, cfg.head_dim, dtype=dtype),
        "full_k": torch.zeros(n_full, 1, nkv, bucket, cfg.global_head_dim, dtype=dtype),
        "full_v": torch.zeros(n_full, 1, nkv, bucket, cfg.global_head_dim, dtype=dtype),
    }


def build_causal_mask_full(in_step: int, bucket: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Additive FULL-attention mask [1,1,1,bucket]: 0 for columns [0..in_step], MASK_NEG beyond."""
    m = torch.full((1, 1, 1, bucket), MASK_NEG, dtype=dtype)
    m[..., : in_step + 1] = 0.0
    return m


def build_causal_mask_sliding(in_step: int, bucket: int, window: int,
                              dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Additive SLIDING mask [1,1,1,bucket] for a LINEAR sliding cache: 0 for (in_step-window, in_step]."""
    m = torch.full((1, 1, 1, bucket), MASK_NEG, dtype=dtype)
    lo = max(0, in_step - window + 1)
    m[..., lo : in_step + 1] = 0.0
    return m


# =========================================================================== #
# HOST-MANAGED cache variant — the macOS-27-beta-GPU-SAFE fixed-shape decode.
#
# The canonical bucketed core (above) writes the new KV column in-graph at a DATA-indexed offset
# (`mutable_slice_update` at a tensor `in_step`). The macOS-27 beta Mac GPU (Core AI -> MPSGraph) cannot
# lower that op (SIGSEGV) — yet it runs masked SDPA over PLAIN inputs and `cat` fine (verified). So this
# variant removes BOTH the Core AI state and the indexed in-graph write:
#   * the fixed-size KV cache is a PLAIN input/output (the HOST writes the new column at position p between
#     steps — a cheap numpy assignment, no in-graph op),
#   * the current token's K/V is appended to the read with `torch.cat` (fixed length B+1),
#   * an explicit mask (length B+1) marks the valid past columns + the current one.
# Fully STATIC shapes, NO state, NO in-graph indexed write -> only MPSGraph-friendly ops -> runs on the
# beta Mac GPU while staying entirely within Core AI (no raw Metal). Numerically identical to the canonical
# bucketed / stateful cores.
# =========================================================================== #

HOSTCACHE_INPUT_NAMES = ("inputs_embeds", "per_layer_inputs", "position_ids",
                         "causal_mask_full", "causal_mask_sliding",
                         "sliding_k", "sliding_v", "full_k", "full_v")
HOSTCACHE_OUTPUT_NAMES = ("hidden", "sliding_k_cur", "sliding_v_cur", "full_k_cur", "full_v_cur")


def _hostcache_attention(attn: Attention, sdpa: SDPA, x: torch.Tensor, position_ids: torch.Tensor,
                         past_k: torch.Tensor, past_v: torch.Tensor, mask: torch.Tensor,
                         cur_kv: tuple[torch.Tensor, torch.Tensor] | None = None):
    """Attention over [host-written past cache ++ the current token] via `cat` (no state, no write).

    ``past_k``/``past_v``: this slot's history [1, nkv, B, hd] (host-written positions 0..p-1).
    ``cur_kv``: None for an own-cache (non-shared) layer -> it computes its current K/V and RETURNS them
    (so the host can store them) as ``(out, cur_k, cur_v)``; given (the producer's current) for a KV-shared
    layer -> returns ``out`` only."""
    b, s, _ = x.shape  # s == 1
    H, hd = attn.n_heads, attn.head_dim
    q = attn.q_proj(x).view(b, s, H, hd).permute(0, 2, 1, 3)
    q = attn.q_norm(q)
    q = attn.rope(q, position_ids=position_ids, freqs=attn.inv_freq)

    own = cur_kv is None
    if own:
        cur_k, cur_v = attn._project_kv(x, position_ids)   # [1, nkv, 1, hd]
    else:
        cur_k, cur_v = cur_kv
    k_full = torch.cat([past_k, cur_k], dim=-2)            # [1, nkv, B+1, hd]
    v_full = torch.cat([past_v, cur_v], dim=-2)
    out = sdpa(query=q, key=k_full, value=v_full, attn_mask=(mask == 0))
    out = out.permute(0, 2, 1, 3).reshape(b, s, H * hd)
    out = attn.o_proj(out)
    return (out, cur_k, cur_v) if own else out


def _hostcache_layer(layer: DecoderLayer, sdpa: SDPA, x: torch.Tensor, position_ids: torch.Tensor,
                     per_layer_input: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor,
                     mask: torch.Tensor, cur_kv=None):
    """DecoderLayer.forward_stateful's norms/MLP/PLE around the host-cache attention."""
    own = cur_kv is None
    residual = x
    h = layer.input_layernorm(x)
    a = _hostcache_attention(layer.self_attn, sdpa, h, position_ids, past_k, past_v, mask, cur_kv)
    if own:
        h, cur_k, cur_v = a
    else:
        h = a
    h = layer.post_attention_layernorm(h)
    x = residual + h

    residual = x
    h = layer.pre_feedforward_layernorm(x)
    h = layer.mlp(h)
    h = layer.post_feedforward_layernorm(h)
    x = residual + h

    residual = x
    h = layer.per_layer_input_gate(x)
    h = nn.functional.gelu(h, approximate="tanh")
    h = h * per_layer_input
    h = layer.per_layer_projection(h)
    h = layer.post_per_layer_input_norm(h)
    x = residual + h
    x = x * layer.layer_scalar
    return (x, cur_k, cur_v) if own else x


class Gemma4DecodeHostCache(nn.Module):
    """Fixed-shape decode core with HOST-managed KV (no Core AI state, no in-graph indexed write).

    forward(inputs_embeds[1,1,h], per_layer_inputs[1,1,nl,ld], position_ids[1,1],
            causal_mask_full[1,1,1,B+1], causal_mask_sliding[1,1,1,B+1],
            sliding_k[n_s,1,nkv,B,256], sliding_v, full_k[n_f,1,nkv,B,512], full_v)
      -> hidden[1,1,h], sliding_k_cur[n_s,1,nkv,1,256], sliding_v_cur, full_k_cur[n_f,1,nkv,1,512], full_v_cur

    The host passes the caches holding positions 0..p-1, runs the step, then writes the returned *_cur
    columns into the caches at position p for the next step. All shapes static."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, text_model: Gemma4TextModel) -> None:
        super().__init__()
        self.text_model = text_model
        self._route, _, _ = text_model.config.stateful_routing()
        self.sdpa = SDPA(scale=1.0)

    def forward(self, inputs_embeds, per_layer_inputs, position_ids,
                causal_mask_full, causal_mask_sliding,
                sliding_k, sliding_v, full_k, full_v):
        x = inputs_embeds
        producer_cur: dict[str, tuple] = {}
        out_sk, out_sv, out_fk, out_fv = [], [], [], []
        for i, layer in enumerate(self.text_model.layers):
            is_full, slot, write = self._route[i]
            mask = causal_mask_full if is_full else causal_mask_sliding
            pk_cache, pv_cache = (full_k, full_v) if is_full else (sliding_k, sliding_v)
            past_k = pk_cache[slot]          # [1, nkv, B, hd]
            past_v = pv_cache[slot]
            ple_i = per_layer_inputs[:, :, i, :]
            if write:                        # non-shared: owns this slot, computes current K/V
                x, cur_k, cur_v = _hostcache_layer(layer, self.sdpa, x, position_ids, ple_i,
                                                   past_k, past_v, mask)
                (out_fk if is_full else out_sk).append(cur_k)
                (out_fv if is_full else out_sv).append(cur_v)
                if layer.is_producer:
                    producer_cur[layer.layer_type] = (cur_k, cur_v)
            else:                            # KV-shared: reuse the producer's current + its slot history
                x = _hostcache_layer(layer, self.sdpa, x, position_ids, ple_i, past_k, past_v, mask,
                                     cur_kv=producer_cur[layer.layer_type])
        hidden = self.text_model.norm(x)
        sk_cur = torch.stack(out_sk, dim=0)  # [n_s, 1, nkv, 1, hd]
        sv_cur = torch.stack(out_sv, dim=0)
        fk_cur = torch.stack(out_fk, dim=0)  # [n_f, 1, nkv, 1, hd]
        fv_cur = torch.stack(out_fv, dim=0)
        return hidden, sk_cur, sv_cur, fk_cur, fv_cur

    def build_macos_export_spec(self, target_dtype: torch.dtype, bucket: int, query_len: int = 1) -> dict:
        """Reference inputs for a FULLY STATIC export (no state, no Dim). ``query_len``=1 is the decode
        core; ``query_len``=chunk is the CHUNKED-PREFILL core (process Q prompt tokens per pass -> fewer
        passes -> lower TTFT). The same forward serves both (cat appends Q columns; masks are [1,1,Q,B+Q])."""
        cfg = self.text_model.config
        h, ld, nl = cfg.hidden_size, cfg.hidden_size_per_layer_input, cfg.num_hidden_layers
        _, n_sliding, n_full = cfg.stateful_routing()
        nkv, Q, B = cfg.num_key_value_heads, query_len, bucket
        ref = {
            "inputs_embeds": torch.zeros(1, Q, h, dtype=target_dtype),
            "per_layer_inputs": torch.zeros(1, Q, nl, ld, dtype=target_dtype),
            "position_ids": torch.arange(Q, dtype=torch.int32).unsqueeze(0),
            "causal_mask_full": torch.zeros(1, 1, Q, B + Q, dtype=target_dtype),
            "causal_mask_sliding": torch.zeros(1, 1, Q, B + Q, dtype=target_dtype),
            "sliding_k": torch.zeros(n_sliding, 1, nkv, B, cfg.head_dim, dtype=target_dtype),
            "sliding_v": torch.zeros(n_sliding, 1, nkv, B, cfg.head_dim, dtype=target_dtype),
            "full_k": torch.zeros(n_full, 1, nkv, B, cfg.global_head_dim, dtype=target_dtype),
            "full_v": torch.zeros(n_full, 1, nkv, B, cfg.global_head_dim, dtype=target_dtype),
        }
        return {
            "reference_inputs": ref,
            "dynamic_shapes": None,  # FULLY STATIC -> one compile, reused every pass
            "input_names": HOSTCACHE_INPUT_NAMES,
            "output_names": HOSTCACHE_OUTPUT_NAMES,
            "state_names": None,
        }


# =========================================================================== #
# FUSED-ATTENTION host-cache variant — Lever #4 (ATTENTION_KERNEL_HANDOFF §B).
# Identical math/IO to Gemma4DecodeHostCache, but the per-layer `cat(past,cur)` + composite SDPA are
# replaced by ONE custom flash-decode Metal kernel (the `flash` module, injected so this file stays free
# of the coreai_torch kernel dependency). GPU-only specialization (the kernel is GPU MSL); the plain
# Gemma4DecodeHostCache remains the ANE / fallback path. q/k/v/o projections + norms + RoPE are UNCHANGED
# (still composite MPSGraph) — this increment fuses only the SDPA subgraph + the two KV cats.
# =========================================================================== #
def _hostcache_attention_fused(attn: Attention, flash: nn.Module, x: torch.Tensor,
                               position_ids: torch.Tensor, past_k: torch.Tensor, past_v: torch.Tensor,
                               mask: torch.Tensor,
                               cur_kv: tuple[torch.Tensor, torch.Tensor] | None = None):
    """`_hostcache_attention` with cat+SDPA -> `flash(q, past_k, cur_k, past_v, cur_v, mask)` (one kernel).

    ``flash`` returns context [1, 1, H*hd] (head-major), so the caller skips the permute/reshape and
    goes straight to o_proj. Same own-cache vs KV-shared contract as the composite path."""
    b, s, _ = x.shape  # s == 1
    H, hd = attn.n_heads, attn.head_dim
    q = attn.q_proj(x).view(b, s, H, hd).permute(0, 2, 1, 3)
    q = attn.q_norm(q)
    q = attn.rope(q, position_ids=position_ids, freqs=attn.inv_freq)  # [1, H, 1, hd]

    own = cur_kv is None
    if own:
        cur_k, cur_v = attn._project_kv(x, position_ids)   # [1, nkv=1, 1, hd]
    else:
        cur_k, cur_v = cur_kv
    out = flash(q, past_k, cur_k, past_v, cur_v, mask)      # [1, 1, H*hd]
    out = attn.o_proj(out)
    return (out, cur_k, cur_v) if own else out


def _hostcache_layer_fused(layer: DecoderLayer, flash: nn.Module, x: torch.Tensor,
                           position_ids: torch.Tensor, per_layer_input: torch.Tensor,
                           past_k: torch.Tensor, past_v: torch.Tensor, mask: torch.Tensor, cur_kv=None):
    """`_hostcache_layer` around the FUSED attention (norms / MLP / PLE identical)."""
    own = cur_kv is None
    residual = x
    h = layer.input_layernorm(x)
    a = _hostcache_attention_fused(layer.self_attn, flash, h, position_ids, past_k, past_v, mask, cur_kv)
    if own:
        h, cur_k, cur_v = a
    else:
        h = a
    h = layer.post_attention_layernorm(h)
    x = residual + h

    residual = x
    h = layer.pre_feedforward_layernorm(x)
    h = layer.mlp(h)
    h = layer.post_feedforward_layernorm(h)
    x = residual + h

    residual = x
    h = layer.per_layer_input_gate(x)
    h = nn.functional.gelu(h, approximate="tanh")
    h = h * per_layer_input
    h = layer.per_layer_projection(h)
    h = layer.post_per_layer_input_norm(h)
    x = residual + h
    x = x * layer.layer_scalar
    return (x, cur_k, cur_v) if own else x


class Gemma4DecodeHostCacheFusedAttn(Gemma4DecodeHostCache):
    """Host-cache decode core with the attention SDPA + KV-cat fused into ONE Metal kernel (GPU path).

    Same forward signature / outputs as ``Gemma4DecodeHostCache``. ``flash_sdpa`` is an nn.Module with
    ``forward(q[1,H,1,hd], past_k[1,nkv,B,hd], cur_k[1,nkv,1,hd], past_v, cur_v, mask[1,1,1,B+1]) ->
    context[1,1,H*hd]`` (e.g. ``gemma4_metal_mlp.MetalFlashSDPA``). Register that kernel with the
    converter alongside the FFN kernel when exporting."""

    def __init__(self, text_model: Gemma4TextModel, flash_sdpa: nn.Module) -> None:
        super().__init__(text_model)
        self.flash = flash_sdpa

    def forward(self, inputs_embeds, per_layer_inputs, position_ids,
                causal_mask_full, causal_mask_sliding,
                sliding_k, sliding_v, full_k, full_v):
        x = inputs_embeds
        producer_cur: dict[str, tuple] = {}
        out_sk, out_sv, out_fk, out_fv = [], [], [], []
        for i, layer in enumerate(self.text_model.layers):
            is_full, slot, write = self._route[i]
            mask = causal_mask_full if is_full else causal_mask_sliding
            pk_cache, pv_cache = (full_k, full_v) if is_full else (sliding_k, sliding_v)
            past_k = pk_cache[slot]
            past_v = pv_cache[slot]
            ple_i = per_layer_inputs[:, :, i, :]
            if write:
                x, cur_k, cur_v = _hostcache_layer_fused(layer, self.flash, x, position_ids, ple_i,
                                                         past_k, past_v, mask)
                (out_fk if is_full else out_sk).append(cur_k)
                (out_fv if is_full else out_sv).append(cur_v)
                if layer.is_producer:
                    producer_cur[layer.layer_type] = (cur_k, cur_v)
            else:
                x = _hostcache_layer_fused(layer, self.flash, x, position_ids, ple_i, past_k, past_v,
                                           mask, cur_kv=producer_cur[layer.layer_type])
        hidden = self.text_model.norm(x)
        sk_cur = torch.stack(out_sk, dim=0)
        sv_cur = torch.stack(out_sv, dim=0)
        fk_cur = torch.stack(out_fk, dim=0)
        fv_cur = torch.stack(out_fv, dim=0)
        return hidden, sk_cur, sv_cur, fk_cur, fv_cur


# =========================================================================== #
# FUSED-NORM-BRIDGE host-cache variant — Lever #4 glue fusion (the dispatch-bound "rest").
# Keeps the COMPOSITE SDPA (the flash fold regressed — see _macos_speed_RESULTS.md) and replaces only the
# per-layer post_attention_layernorm -> +residual -> pre_feedforward_layernorm bridge with ONE custom kernel
# (``bridge`` module, injected). Probes whether collapsing the dispatch-bound elementwise glue recovers time.
# =========================================================================== #
def _hostcache_layer_fused_norm(layer: DecoderLayer, sdpa: SDPA, bridge: nn.Module, x: torch.Tensor,
                                position_ids: torch.Tensor, per_layer_input: torch.Tensor,
                                past_k: torch.Tensor, past_v: torch.Tensor, mask: torch.Tensor,
                                cur_kv=None):
    """`_hostcache_layer` with post_attn_ln + residual + pre_ffn_ln fused into ``bridge`` (composite SDPA)."""
    own = cur_kv is None
    residual = x
    h = layer.input_layernorm(x)
    a = _hostcache_attention(layer.self_attn, sdpa, h, position_ids, past_k, past_v, mask, cur_kv)
    attn_out = a[0] if own else a
    # FUSED: x2 = residual + post_attention_layernorm(attn_out); ffn_in = pre_feedforward_layernorm(x2)
    x2, ffn_in = bridge(attn_out, residual)
    h = layer.mlp(ffn_in)
    h = layer.post_feedforward_layernorm(h)
    x = x2 + h

    residual = x
    h = layer.per_layer_input_gate(x)
    h = nn.functional.gelu(h, approximate="tanh")
    h = h * per_layer_input
    h = layer.per_layer_projection(h)
    h = layer.post_per_layer_input_norm(h)
    x = residual + h
    x = x * layer.layer_scalar
    if own:
        _, cur_k, cur_v = a
        return x, cur_k, cur_v
    return x


class Gemma4DecodeHostCacheFusedNorm(Gemma4DecodeHostCache):
    """Host-cache decode core with the post-attention norm bridge fused into one Metal kernel per layer.

    Same IO as ``Gemma4DecodeHostCache`` (composite SDPA retained). ``norm_bridges`` is a per-layer list of
    nn.Modules with ``forward(attn_out[1,1,h], residual[1,1,h]) -> (x2[1,1,h], ffn_in[1,1,h])`` (e.g.
    ``gemma4_metal_mlp.MetalNormBridge``); register that kernel with the converter alongside the FFN kernel."""

    def __init__(self, text_model: Gemma4TextModel, norm_bridges: list[nn.Module]) -> None:
        super().__init__(text_model)
        self.norm_bridges = nn.ModuleList(norm_bridges)

    def forward(self, inputs_embeds, per_layer_inputs, position_ids,
                causal_mask_full, causal_mask_sliding,
                sliding_k, sliding_v, full_k, full_v):
        x = inputs_embeds
        producer_cur: dict[str, tuple] = {}
        out_sk, out_sv, out_fk, out_fv = [], [], [], []
        for i, layer in enumerate(self.text_model.layers):
            is_full, slot, write = self._route[i]
            mask = causal_mask_full if is_full else causal_mask_sliding
            pk_cache, pv_cache = (full_k, full_v) if is_full else (sliding_k, sliding_v)
            past_k, past_v = pk_cache[slot], pv_cache[slot]
            ple_i = per_layer_inputs[:, :, i, :]
            bridge = self.norm_bridges[i]
            if write:
                x, cur_k, cur_v = _hostcache_layer_fused_norm(layer, self.sdpa, bridge, x, position_ids,
                                                              ple_i, past_k, past_v, mask)
                (out_fk if is_full else out_sk).append(cur_k)
                (out_fv if is_full else out_sv).append(cur_v)
                if layer.is_producer:
                    producer_cur[layer.layer_type] = (cur_k, cur_v)
            else:
                x = _hostcache_layer_fused_norm(layer, self.sdpa, bridge, x, position_ids, ple_i,
                                                past_k, past_v, mask, cur_kv=producer_cur[layer.layer_type])
        hidden = self.text_model.norm(x)
        return (hidden, torch.stack(out_sk, 0), torch.stack(out_sv, 0),
                torch.stack(out_fk, 0), torch.stack(out_fv, 0))


def build_hostcache_caches(cfg: Gemma4TextConfig, bucket: int,
                           dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
    """Zero host-managed KV caches at fixed capacity ``bucket`` (plain tensors, NOT Core AI states)."""
    _, n_sliding, n_full = cfg.stateful_routing()
    nkv = cfg.num_key_value_heads
    return {
        "sliding_k": torch.zeros(n_sliding, 1, nkv, bucket, cfg.head_dim, dtype=dtype),
        "sliding_v": torch.zeros(n_sliding, 1, nkv, bucket, cfg.head_dim, dtype=dtype),
        "full_k": torch.zeros(n_full, 1, nkv, bucket, cfg.global_head_dim, dtype=dtype),
        "full_v": torch.zeros(n_full, 1, nkv, bucket, cfg.global_head_dim, dtype=dtype),
    }


def build_hostcache_mask_full(p: int, bucket: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Additive FULL mask [1,1,1,bucket+1] over [past slots 0..B-1 ++ current at B]: past 0..p-1 valid
    (host-written, causal) + the current token (index B) valid; everything else MASK_NEG."""
    m = torch.full((1, 1, 1, bucket + 1), MASK_NEG, dtype=dtype)
    m[..., :p] = 0.0        # past positions 0..p-1
    m[..., bucket] = 0.0    # current token (position p)
    return m


def build_hostcache_mask_sliding(p: int, bucket: int, window: int,
                                 dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Additive SLIDING mask [1,1,1,bucket+1]: past slots in (p-window, p-1] valid + current (index B)."""
    m = torch.full((1, 1, 1, bucket + 1), MASK_NEG, dtype=dtype)
    lo = max(0, p - window + 1)
    m[..., lo:p] = 0.0      # past positions in the window, < p
    m[..., bucket] = 0.0    # current token (position p, always in its own window)
    return m


# ---- CHUNKED PREFILL masks (Q queries at positions [start..start+Q-1], cat layout [past B ++ Q new]) ---
def build_prefill_mask_full(start: int, Q: int, bucket: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """FULL block-causal mask [1,1,Q,B+Q]. Query i (position start+i) attends to past cache slots 0..start-1
    and chunk tokens k<=i (block-causal); MASK_NEG elsewhere."""
    m = torch.full((1, 1, Q, bucket + Q), MASK_NEG, dtype=dtype)
    for i in range(Q):
        m[0, 0, i, :start] = 0.0                 # host-written past (positions 0..start-1)
        m[0, 0, i, bucket : bucket + i + 1] = 0.0  # chunk tokens 0..i (causal within the chunk)
    return m


def build_prefill_mask_sliding(start: int, Q: int, bucket: int, window: int,
                               dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """SLIDING block-causal mask [1,1,Q,B+Q]: query i (position start+i) attends to in-window
    (> start+i-window) past slots + in-window chunk tokens k<=i."""
    m = torch.full((1, 1, Q, bucket + Q), MASK_NEG, dtype=dtype)
    for i in range(Q):
        pos = start + i
        lo_past = max(0, pos - window + 1)
        if lo_past < start:
            m[0, 0, i, lo_past:start] = 0.0       # past slots inside the window
        lo_chunk = max(0, i - window + 1)          # chunk tokens k in (i-window, i]
        m[0, 0, i, bucket + lo_chunk : bucket + i + 1] = 0.0
    return m


# =========================================================================== #
# GPU head + in-graph argmax — Lever #3 (avoid the full-vocab CPU readback).
# The decode head is the tied lm_head (1536 -> 262144) + Gemma 4 softcap. For GREEDY decode the softcap
# is monotonic (argmax(tanh(z/c)*c) == argmax(z)), so the top-1 token is just argmax(lm_head(hidden)).
# Doing the argmax IN-GRAPH returns ONE int instead of 262144 logits -> no big readback/transfer, and the
# matmul runs on the GPU. (Sampling, which needs the real logits, would use the plain Gemma4Head instead.)
# =========================================================================== #
class Gemma4HeadArgmax(nn.Module):
    """``hidden [1,1,hidden] -> token_id [1,1] (int32)`` = argmax over the tied LM head. Greedy only."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, head: Gemma4Head) -> None:
        super().__init__()
        self.lm_head = head.lm_head  # tied embed weight (shared, no copy)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)               # [1, 1, vocab]
        return torch.argmax(logits, dim=-1).to(torch.int32)  # [1, 1]
