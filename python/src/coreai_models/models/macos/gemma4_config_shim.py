# Community port — NOT an Apple model. BSD-3-Clause (see LICENSE).

"""Transformers ``AutoConfig`` shim for Gemma 4 checkpoints.

The pinned transformers (4.57.x) predates Gemma 4, so
``AutoConfig.from_pretrained("google/gemma-4-*")`` raises
``ValueError: model type 'gemma4' not recognized``. The registry / CLI export
path reads the config through ``AutoConfig``, so without this it can't load.

We register two permissive ``PretrainedConfig`` subclasses that just carry the
checkpoint's JSON fields as attributes — enough for the re-authored
``Gemma4TextConfig.from_hf_config`` to consume. No modeling classes are
registered (the decoder is our own re-authored module). Drop this once the
venv's transformers ships native Gemma 4 support.
"""
from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class Gemma4TextHFConfig(PretrainedConfig):
    model_type = "gemma4_text"


class Gemma4HFConfig(PretrainedConfig):
    model_type = "gemma4"
    # Parse the nested ``text_config`` dict as a typed sub-config.
    sub_configs = {"text_config": Gemma4TextHFConfig}

    def __init__(self, text_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = Gemma4TextHFConfig(**text_config)
        self.text_config = text_config
        super().__init__(**kwargs)


def register_gemma4_configs() -> None:
    """Register gemma4 / gemma4_text with ``AutoConfig`` (idempotent)."""
    for model_type, cls in (("gemma4_text", Gemma4TextHFConfig), ("gemma4", Gemma4HFConfig)):
        try:
            AutoConfig.register(model_type, cls)
        except ValueError:
            pass  # already registered in this interpreter


# Register on import so a module-level `import ...gemma4_config_shim` in the
# registry makes AutoConfig.from_pretrained work before get_model_entry runs.
register_gemma4_configs()
