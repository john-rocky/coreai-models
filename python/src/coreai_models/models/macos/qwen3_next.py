# Qwen3-Next text decoder (HF model_type `qwen3_next`, e.g. Qwen/Qwen3-Coder-Next 80B-A3B)
# for the Core AI authoring path.
#
# Community port — NOT an Apple model.  The architecture is the SAME family as Qwen3.6's
# `qwen3_5_moe` (hybrid GatedDeltaNet / gated full attention 3:1, GVA 32 value heads over 16 key
# heads, partial-RoPE, sparse-MoE FFN with a sigmoid-gated shared expert).  The model classes are
# reused VERBATIM from qwen3_5_moe.py; only the checkpoint flavour differs, so this overlay adds
# just the things that differ:
#
#   1. FLAT config.json (no nested `text_config`) -> read top-level keys directly.
#   2. `layer_types` is NOT in the config -> derived from `full_attention_interval`
#      (a layer is full attention iff (idx + 1) % interval == 0; verified against the 80B
#      checkpoint: full @ [3, 7, 11, ..., 47], the other 36 are linear_attention/GatedDeltaNet).
#   3. weight prefix `model.` (Coder-Next is text-only — no `model.language_model.` vision wrapper).
#   4. PER-EXPERT routed weights `...mlp.experts.{e}.{gate,up,down}_proj.weight` (NOT Qwen3.6's
#      packed `gate_up_proj`); the per-expert -> [E, …] stacking lives in
#      moe_metal_streaming.from_hf_streaming_metal_moe (the 159 GB-fp16 model must be streamed,
#      it does not fit the 128 GB host as a full fp16 state-dict).
#
# Config (Qwen/Qwen3-Coder-Next): hidden 2048, 48 layers (all MoE, decoder_sparse_step=1),
# 16 attn / 2 KV heads, head_dim 256, vocab 151936 (untied lm_head), 512 experts top-10,
# moe_intermediate_size 512, shared_expert_intermediate_size 512, partial_rotary_factor 0.25,
# rope_theta 5e6, GatedDeltaNet linear_num_value_heads 32 / linear_num_key_heads 16.
from __future__ import annotations

from coreai_models.models.macos.qwen3_5_moe import (
    DECODE_STATE_NAMES,  # noqa: F401  (re-exported for export scripts)
    Qwen3_5MoeConfig,
    Qwen3_5MoeStatefulForCausalLM,  # noqa: F401  (model class reused verbatim)
    build_decode_state,  # noqa: F401  (re-exported for export scripts)
)

# Coder-Next is text-only: routed/shared/attn weights live directly under `model.`.
QWEN3_NEXT_PREFIX = "model."

# The qwen3_next model body == the qwen3_5_moe body (only the loader/config differ).
Qwen3NextStatefulForCausalLM = Qwen3_5MoeStatefulForCausalLM


def qwen3_next_config_from_json(
    cfg: dict,
    num_layers: int | None = None,
) -> Qwen3_5MoeConfig:
    """Build the authoring config straight from a FLAT qwen3_next config.json dict.

    Unlike Qwen3.6's `qwen3_5_moe_config_from_json` (which reads a nested `text_config` and an
    explicit `layer_types`), qwen3_next has neither: the config is flat and the per-layer mixer
    type is implied by `full_attention_interval`. Everything else maps 1:1 onto Qwen3_5MoeConfig.
    """
    n_layers = num_layers if num_layers is not None else cfg["num_hidden_layers"]
    interval = cfg["full_attention_interval"]
    # full attention every `interval`-th layer (1-indexed), GatedDeltaNet otherwise.
    layer_types = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n_layers)
    ]
    return Qwen3_5MoeConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=n_layers,
        vocab_size=cfg["vocab_size"],
        # no dense FFN in this arch (all layers MoE); keep the replaced base MLP construction tiny.
        intermediate_size=cfg.get("intermediate_size", cfg["moe_intermediate_size"]),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        head_dim=cfg["head_dim"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        partial_rotary_factor=cfg.get("partial_rotary_factor", 0.25),
        rope_theta=cfg.get("rope_theta", 1e7),
        linear_num_key_heads=cfg["linear_num_key_heads"],
        linear_num_value_heads=cfg["linear_num_value_heads"],
        linear_key_head_dim=cfg["linear_key_head_dim"],
        linear_value_head_dim=cfg["linear_value_head_dim"],
        linear_conv_kernel_dim=cfg["linear_conv_kernel_dim"],
        full_attention_interval=interval,
        layer_types=layer_types,
        moe_intermediate_size=cfg["moe_intermediate_size"],
        shared_expert_intermediate_size=cfg["shared_expert_intermediate_size"],
        num_experts=cfg["num_experts"],
        num_experts_per_tok=cfg["num_experts_per_tok"],
    )
