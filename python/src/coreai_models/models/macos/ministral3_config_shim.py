# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Transformers ``AutoConfig`` shim for Ministral-3 checkpoints.

The Ministral-3 multimodal checkpoint (`mistralai/Ministral-3-3B-Instruct-2512`)
has top-level ``model_type: "mistral3"`` (recognized by transformers 4.57), but
its nested ``text_config`` carries ``model_type: "ministral3"`` which the pinned
transformers does not know — so ``Mistral3Config.__init__`` raises
``KeyError: 'ministral3'`` when it tries
``CONFIG_MAPPING["ministral3"](**text_config)``.

The Ministral-3 text decoder is architecturally a Mistral decoder (GQA + RoPE +
RMSNorm + SwiGLU) with YARN rope scaling carried in ``rope_parameters``, so we
register ``ministral3`` as a thin ``MistralConfig`` subclass. Drop once the venv
ships native Ministral-3 support.
"""

from __future__ import annotations

from transformers import AutoConfig
from transformers.models.mistral.configuration_mistral import MistralConfig


class Ministral3TextConfig(MistralConfig):
    model_type = "ministral3"


def register_ministral3_configs() -> None:
    """Register ``ministral3`` with ``AutoConfig`` (idempotent)."""
    try:
        AutoConfig.register("ministral3", Ministral3TextConfig)
    except ValueError:
        pass  # already registered in this interpreter


# Register on import so the registry's module-level import makes
# AutoConfig.from_pretrained work before get_model_entry runs.
register_ministral3_configs()
