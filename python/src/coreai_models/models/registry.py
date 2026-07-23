# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Model registry mapping HuggingFace model_type to model classes."""

from dataclasses import dataclass
from functools import lru_cache

import torch.nn as nn

# Importing the config shim registers the Qwen3.5 configs with transformers
# AutoConfig (the 4.57 export env has no built-in qwen3_5), so the export
# pipeline's `AutoConfig.from_pretrained` can parse the checkpoint.
from coreai_models.models.macos import qwen3_5_config as _qwen3_5_config  # noqa: F401

# Likewise register gemma4 / gemma4_text with AutoConfig on import (4.57 has no gemma4).
from coreai_models.models.macos import gemma4_config_shim as _gemma4_config_shim  # noqa: F401

# Register the inner `ministral3` text config type so AutoConfig can parse the
# Ministral-3 (mistral3 multimodal) checkpoint.
from coreai_models.models.macos import ministral3_config_shim as _ministral3_config_shim  # noqa: F401


@dataclass
class ModelEntry:
    """Registry entry for a model family."""

    macos_class: type[nn.Module] | None = None
    ios_class: type[nn.Module] | None = None
    # Multimodal-checkpoint hooks consumed by `from_hf_memory_efficient` and the
    # `text_config` unwrap in the export pipeline.
    # `hf_config_attr`: attribute on the top-level HF config holding the
    #     per-modality sub-config (e.g. "text_config" for Gemma-3).
    # `hf_state_dict_prefix`: prefix on safetensors keys for this modality
    #     (e.g. "language_model." for Gemma-3). Stripped before assignment.
    hf_config_attr: str | None = None
    hf_state_dict_prefix: str = ""


@lru_cache(maxsize=1)
def _get_registry() -> dict[str, ModelEntry]:
    """Build the model registry (cached singleton). Lazy imports to avoid circular deps."""
    from coreai_models.models.ios.mistral import MistralForCausalLMForiOS
    from coreai_models.models.ios.olmo2 import Olmo2ForCausalLMForiOS
    from coreai_models.models.ios.phi3 import Phi3ForCausalLMForiOS
    from coreai_models.models.ios.qwen2 import Qwen2ForCausalLMForiOS
    from coreai_models.models.ios.qwen3 import Qwen3ForCausalLMForiOS
    from coreai_models.models.ios.smollm3 import SmolLM3ForCausalLMForiOS
    from coreai_models.models.macos.gemma3_text import Gemma3ForCausalLM
    from coreai_models.models.macos.gemma4_text import Gemma4TextForCausalLM
    from coreai_models.models.macos.gpt_oss import GptOssForCausalLM
    from coreai_models.models.macos.ministral3 import Ministral3ForCausalLM
    from coreai_models.models.macos.mistral import MistralForCausalLM
    from coreai_models.models.macos.mixtral import MixtralForCausalLM
    from coreai_models.models.macos.olmo2 import Olmo2ForCausalLM
    from coreai_models.models.macos.phi3 import Phi3ForCausalLM
    from coreai_models.models.macos.qwen2 import Qwen2ForCausalLM
    from coreai_models.models.macos.qwen3 import Qwen3ForCausalLM
    from coreai_models.models.macos.qwen3_5 import Qwen3_5StatefulForCausalLM
    from coreai_models.models.macos.qwen3_moe import Qwen3MoeForCausalLM
    from coreai_models.models.macos.qwen3_vl import (
        Qwen3VLForCausalLM,
    )
    from coreai_models.models.macos.smollm3 import SmolLM3ForCausalLM

    return {
        "gemma3_text": ModelEntry(
            macos_class=Gemma3ForCausalLM,
            hf_config_attr="text_config",
            hf_state_dict_prefix="language_model.",
        ),
        # Gemma 4 E2B/E4B text decoder (community port). Multimodal checkpoint:
        # text weights nest under "model.language_model."; dual head_dim + KV
        # sharing + PLE handled in the re-authored module. AutoConfig needs the
        # gemma4 shim (transformers 4.57 predates gemma4) — registered on lookup.
        "gemma4_text": ModelEntry(
            macos_class=Gemma4TextForCausalLM,
            hf_config_attr="text_config",
            hf_state_dict_prefix="model.language_model.",
        ),
        "gpt_oss": ModelEntry(
            macos_class=GptOssForCausalLM,
        ),
        "mistral": ModelEntry(
            macos_class=MistralForCausalLM,
            ios_class=MistralForCausalLMForiOS,
        ),
        "mixtral": ModelEntry(
            macos_class=MixtralForCausalLM,
        ),
        # Ministral-3 multimodal (mistral3) checkpoint: export the FP8 text
        # decoder nested under text_config / language_model.
        "mistral3": ModelEntry(
            macos_class=Ministral3ForCausalLM,
            hf_config_attr="text_config",
            hf_state_dict_prefix="language_model.",
        ),
        "olmo2": ModelEntry(
            macos_class=Olmo2ForCausalLM,
            ios_class=Olmo2ForCausalLMForiOS,
        ),
        "phi3": ModelEntry(
            macos_class=Phi3ForCausalLM,
            ios_class=Phi3ForCausalLMForiOS,
        ),
        "smollm3": ModelEntry(
            macos_class=SmolLM3ForCausalLM,
            ios_class=SmolLM3ForCausalLMForiOS,
        ),
        "qwen2": ModelEntry(
            macos_class=Qwen2ForCausalLM,
            ios_class=Qwen2ForCausalLMForiOS,
        ),
        "qwen3": ModelEntry(
            macos_class=Qwen3ForCausalLM,
            ios_class=Qwen3ForCausalLMForiOS,
        ),
        "qwen3_moe": ModelEntry(
            macos_class=Qwen3MoeForCausalLM,
        ),
        # Qwen3.5 hybrid linear/full-attention text decoder (community port).
        # Text weights nest under "model.language_model." in the multimodal ckpt.
        "qwen3_5_text": ModelEntry(
            macos_class=Qwen3_5StatefulForCausalLM,
            hf_config_attr="text_config",
            hf_state_dict_prefix="model.language_model.",
        ),
        # Qwen3-VL: vision-language model.
        # macos_class = standard text decoder (input_ids, for text-only use).
        # Qwen3VLForCausalLMEmbeddings is used by the VLM export script (takes inputs_embeds).
        "qwen3_vl": ModelEntry(
            macos_class=Qwen3VLForCausalLM,
            hf_config_attr="text_config",
            hf_state_dict_prefix="model.language_model.",
        ),
    }


# Type alias for the remapping dict
MODEL_TYPE_REMAPPING: dict[str, str] = {
    "gemma3": "gemma3_text",
    "qwen2_5": "qwen2",
    # Plain LlamaForCausalLM (e.g. MiniCPM5-1B) is architecturally identical to
    # Mistral minus the sliding window: GQA + RoPE + RMSNorm + SiLU, no qkv bias,
    # no qk-norm, explicit head_dim honored by the Mistral builder.
    "llama": "mistral",
    # The Qwen3.5 checkpoint's top-level model_type is the VLM "qwen3_5"; we
    # export its text decoder.
    "qwen3_5": "qwen3_5_text",
    # Gemma 4's top-level multimodal model_type -> text decoder.
    "gemma4": "gemma4_text",
}


def get_model_entry(model_type: str) -> ModelEntry:
    """Look up a model by HuggingFace model_type."""
    registry = _get_registry()
    remapped = MODEL_TYPE_REMAPPING.get(model_type, model_type)
    if remapped not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown model type '{model_type}'. Available: {available}")
    return registry[remapped]


def list_models() -> list[str]:
    """List all supported model types."""
    return sorted(_get_registry().keys())
