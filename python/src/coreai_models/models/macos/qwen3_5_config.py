# Config shim for Qwen3.5 (community port — NOT an Apple model).
#
# transformers 4.57 (the coreai-models export env) has no `qwen3_5` model, so
# AutoConfig.from_pretrained rejects the checkpoint. We register minimal typed
# PretrainedConfig classes for `qwen3_5` (the VLM wrapper) and its `text_config`
# / `vision_config` sub-configs so AutoConfig can parse config.json. Only the
# fields the text-decoder export reads are consumed downstream; everything else
# rides along as plain attributes.
#
# Importing this module registers the configs (idempotently). The model registry
# imports it so registration happens before the export pipeline calls AutoConfig.
from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class Qwen3_5VisionConfig(PretrainedConfig):
    model_type = "qwen3_5_vision"


class Qwen3_5TextConfig(PretrainedConfig):
    model_type = "qwen3_5_text"


class Qwen3_5VLConfig(PretrainedConfig):
    model_type = "qwen3_5"
    sub_configs = {"text_config": Qwen3_5TextConfig, "vision_config": Qwen3_5VisionConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs) -> None:
        # Materialise the sub-configs as typed objects (transformers' diff/repr
        # machinery calls `.to_dict()` on them, which a plain dict lacks).
        if isinstance(text_config, dict):
            text_config = Qwen3_5TextConfig(**text_config)
        if isinstance(vision_config, dict):
            vision_config = Qwen3_5VisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        super().__init__(**kwargs)


def register_qwen3_5_configs() -> None:
    """Register the Qwen3.5 configs with transformers AutoConfig (idempotent)."""
    for model_type, cls in (
        ("qwen3_5_vision", Qwen3_5VisionConfig),
        ("qwen3_5_text", Qwen3_5TextConfig),
        ("qwen3_5", Qwen3_5VLConfig),
    ):
        try:
            AutoConfig.register(model_type, cls)
        except ValueError:
            # Already registered (re-import) — fine.
            pass


register_qwen3_5_configs()
