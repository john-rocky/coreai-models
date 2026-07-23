# Ministral-3-3B (`mistral3` multimodal checkpoint, text decoder) for Core AI.
#
# Community port — NOT an Apple model.  The shipped checkpoint
# `mistralai/Ministral-3-3B-Instruct-2512` is a Mistral3 multimodal model whose
# text decoder is:
#   * a standard Mistral decoder (GQA + RoPE + RMSNorm + SwiGLU), with
#   * **YARN** rope scaling carried in `text_config.rope_parameters` (base 1e6),
#   * **FP8-quantized** linear weights (per-tensor: `weight` is float8_e4m3fn,
#     `weight_scale_inv` is the scalar dequant multiplier; `activation_scale` is
#     for activation quant and unused for fp16 inference).
#
# The text weights nest under `language_model.` and the config under
# `text_config` (handled by the registry entry). The reauthored math is exactly
# the shared Mistral decoder; this subclass only adds the FP8 -> fp16 dequant in
# `_mutate_state_dict` (before the parent fuses q/k/v). YARN is picked up by the
# shared `_make_rope` from `rope_parameters`. The `ministral3` text config type
# is registered with AutoConfig by `ministral3_config_shim`.

import torch
from transformers.models.mistral.modeling_mistral import (
    MistralForCausalLM as HFMistralForCausalLM,
)
from typing_extensions import override

# Importing the shim registers the inner `ministral3` text config type with
# transformers AutoConfig (otherwise Mistral3Config parsing raises KeyError).
from coreai_models.models.macos import ministral3_config_shim as _ministral3_config_shim  # noqa: F401
from coreai_models.models.macos.mistral import MistralForCausalLM


class Ministral3ForCausalLM(MistralForCausalLM):
    # Unused on the macOS streaming (from_hf_memory_efficient) path, which reads
    # raw safetensors via AutoConfig; set so the base contract is satisfied.
    _HF_MODEL_CLASS = HFMistralForCausalLM

    @override
    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        # Per-tensor FP8 dequant: w_fp16 = w_fp8.to(fp16) * weight_scale_inv.
        # The streaming loader already cast the float8_e4m3fn weight to the
        # target dtype (fp16); applying the scalar scale recovers the real
        # weight. Drop the scale / activation_scale bookkeeping tensors.
        for key in list(state_dict.keys()):
            if key.endswith(".weight_scale_inv"):
                weight_key = key[: -len(".weight_scale_inv")] + ".weight"
                scale = state_dict.pop(key)
                if weight_key in state_dict:
                    w = state_dict[weight_key].to(torch.float16)
                    state_dict[weight_key] = w * scale.to(torch.float16)
            elif key.endswith(".activation_scale"):
                state_dict.pop(key)

        # Now fuse q/k/v exactly like the base Mistral decoder.
        super()._mutate_state_dict(state_dict)
