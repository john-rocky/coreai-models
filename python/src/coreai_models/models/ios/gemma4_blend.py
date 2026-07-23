# Gemma 4 E2B iOS STATIC decode core with a BLEND-INPUT KV write — the FB23024751 escape.
#
# Community port — NOT an Apple model. Identical to models/ios/gemma4.py (the static dual-KV
# core, PyTorch 8/8 vs HF) in EVERY op except the KV-cache write:
#
#   gemma4.py (crashes the beta runtimes):  mutable_slice_update at the RUNTIME-DATA `in_step`
#       -> Mac GPU SIGTRAP / device GPU SIGSEGV / device ANE MLIR failure (FB23024751,
#          apple/coreai-models#5) — the write COLUMN is derived in-graph from a data tensor.
#
#   THIS file (runs):  cache[slot] <- cache[slot]*(1-m) + col*m   with  m = `write_mask` INPUT
#       -> the graph never sees an integer step; the write region is the (compile-time) slot.
#          Isolated + state-verified EXACT on the beta Mac GPU incl. the production slot shape
#          (ondevice/_iso_kvwrite_escapes_gpu.py modes blend-inputmask / blend-slotview).
#
# The host builds the 2 KB one-hot `write_mask[1,1,1,ctx]` per step (1.0 at the write column,
# 0.0 elsewhere) exactly where it already builds `causal_mask_full`/`causal_mask_sliding`.
# Numerics are BIT-exact vs the slice_update write: with m one-hot, x*(1-m)+col*m is an exact
# fp16 select (multiplies by 0.0/1.0, adds of 0.0).
#
# Everything else (dual head_dim KV, dual RoPE, per-head norms, PLE, KV-share routing, composite
# SDPA + additive mask inputs, fixed shapes) is REUSED from models/ios/gemma4.py via subclassing —
# `in_step` threads Model -> Layer -> Attention untouched and is consumed ONLY by the cache write,
# so the same plumbing carries `write_mask` with zero math duplication.
#
# Why: fixed shapes (no per-step respecialization / jetsam growth) + Core AI states (no host
# KV round-trip per token) WITHOUT waiting for Apple's fix. See ondevice/_kvwrite_escapes_RESULTS.md.
from __future__ import annotations

import torch

from coreai_models.models.ios.gemma4 import (
    Gemma4StaticDecodeCore,
    StaticDualKVCache,
    build_causal_mask_full,       # noqa: F401  (re-exported for drivers)
    build_causal_mask_sliding,    # noqa: F401
    build_static_decode_state,    # noqa: F401
)

STATIC_DECODE_STATE_NAMES = ("slidingKeyCache", "slidingValueCache", "fullKeyCache", "fullValueCache")

# Input names (order matches Gemma4BlendDecodeCore.forward's non-state args): `write_mask`
# replaces gemma4.py's `in_step`.
BLEND_DECODE_INPUT_NAMES = (
    "inputs_embeds",
    "per_layer_inputs",
    "position_ids",
    "write_mask",
    "causal_mask_full",
    "causal_mask_sliding",
)


class BlendDualKVCache(StaticDualKVCache):
    """StaticDualKVCache whose write is the blend-input formulation: the third write arg is the
    one-hot ``write_mask`` tensor (NOT an integer step). Reads are inherited unchanged."""

    @staticmethod
    def _write(cache: torch.Tensor, slot: int, write_mask: torch.Tensor, col: torch.Tensor) -> None:
        """Blend ``col`` ([1, n_kv, 1, head_dim]) into ``cache`` ([n_slots, 1, n_kv, ctx, head_dim])
        at the column marked by ``write_mask`` ([1, 1, 1, ctx], one-hot), for the compile-time
        ``slot``. No data-tensor index anywhere -> lowers on the beta MPSGraph backends."""
        sl = cache[slot]                                                   # [1, n_kv, ctx, hd] view
        m = write_mask.reshape(1, 1, cache.size(3), 1).to(cache.dtype)     # [1, 1, ctx, 1]
        colb = col.reshape(1, cache.size(2), 1, cache.size(4))             # [1, n_kv, 1, hd]
        sl.copy_(sl * (1 - m) + colb * m)


class Gemma4BlendDecodeCore(Gemma4StaticDecodeCore):
    """Static q=1 dual-KV decode core with the blend-input KV write. forward: ``inputs_embeds``
    [1,1,hidden], ``per_layer_inputs`` [1,1,num_layers,ple_dim], ``position_ids`` [1,1],
    ``write_mask`` [1,1,1,ctx] (one-hot at the write column), ``causal_mask_full`` /
    ``causal_mask_sliding`` [1,1,1,ctx] + 4 dual-KV states -> final-norm hidden [1,1,hidden].
    All shapes static; the graph contains NO data-tensor-derived write index."""

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        position_ids: torch.Tensor,
        write_mask: torch.Tensor,
        causal_mask_full: torch.Tensor,
        causal_mask_sliding: torch.Tensor,
        sliding_k: torch.Tensor,
        sliding_v: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
    ) -> torch.Tensor:
        kv = BlendDualKVCache(sliding_k, sliding_v, full_k, full_v)
        # `write_mask` rides the model's `in_step` plumbing — its only consumer is the KV write.
        return self.model(
            inputs_embeds, per_layer_inputs, position_ids, write_mask,
            causal_mask_full, causal_mask_sliding, kv,
        )


def build_write_mask(in_step: int, max_ctx: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """One-hot write mask ``[1,1,1,max_ctx]``: 1.0 at the write column ``in_step``, 0.0 elsewhere.
    Built on the HOST per step (like the causal masks) so the graph never sees the step index."""
    mask = torch.zeros((1, 1, 1, max_ctx), dtype=dtype)
    mask[..., in_step] = 1.0
    return mask
