# Community port — NOT an Apple model.
"""Absorbed-MLA decode form of Youtu-LLM-2B.

Youtu shares GLM-4.7-Flash's DeepSeek-MLA attention (same kv_lora 512 / qk_rope 64,
scale 192**-0.5 — the DeepSeek-V2-Lite shape the flash-decode kernel already bakes),
so the absorbed-MLA machinery is config-driven and reused VERBATIM from
``glm4_moe_lite_absorbed``: the latent ``[512]++[64]`` KV cache (two equal [288]
halves), the ``W_UK``/``W_UV`` lifts sliced from ``kv_b_proj``, the stateful wrapper,
and the ``mla_metal_sdpa`` flash-decode kernel. This module just wires those onto the
(dense-MLP) Youtu model: load naive -> absorbify in place -> wrap for stateful decode.

The only Youtu-specific pieces (dense MLP every layer, tied head, rope_theta 1.6e6,
rms_norm_eps 1e-6) live in ``youtu.py``; the MLA cache/kernel are architecture-shared."""
from __future__ import annotations

import torch

from coreai_models.models.macos.glm4_moe_lite_absorbed import (
    DECODE_STATE_NAMES_ABSORBED,
    Glm4MoeLiteAbsorbedStatefulForCausalLM,
    Glm4MoeLiteMLAAbsorbed,
    build_absorbed_decode_state,
)
from coreai_models.models.macos.youtu import YoutuStatefulForCausalLM, youtu_from_hf

# Re-export the shared names under the youtu module for a self-documenting export.
DECODE_STATE_NAMES_ABSORBED = DECODE_STATE_NAMES_ABSORBED
build_absorbed_decode_state = build_absorbed_decode_state
YoutuAbsorbedStatefulForCausalLM = Glm4MoeLiteAbsorbedStatefulForCausalLM


def absorbify_youtu(causal_lm: YoutuStatefulForCausalLM) -> YoutuStatefulForCausalLM:
    """Swap every layer's naive ``YoutuMLA`` for the absorbed form, in place (the
    naive ``kv_b_proj`` is dropped -> no extra peak memory). Returns the same object,
    now driven by ``YoutuAbsorbedStatefulForCausalLM`` for the latent cache."""
    cfg = causal_lm.config
    for layer in causal_lm.model.layers:
        layer.self_attn = Glm4MoeLiteMLAAbsorbed.from_naive(layer.self_attn, cfg)
    return causal_lm


def youtu_absorbed_from_hf(
    huggingface_model_id: str, target_dtype: torch.dtype = torch.float16
) -> YoutuStatefulForCausalLM:
    """Load Youtu-LLM via the naive loader, then absorbify in place. Wrap the result
    with ``YoutuAbsorbedStatefulForCausalLM.from_causal_lm`` for stateful decode."""
    model = youtu_from_hf(huggingface_model_id, target_dtype=target_dtype)
    return absorbify_youtu(model)
