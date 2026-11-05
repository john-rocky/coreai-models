# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""A text-only checkpoint exports through the registry entry of its multimodal family."""

import torch
from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config, Gemma3TextConfig

from coreai_models.export import pipeline as export_pipeline
from coreai_models.models import base as models_base
from coreai_models.models.macos.gemma3_text import Gemma3ForCausalLM
from coreai_models.models.registry import get_model_entry
from coreai_models.primitives.macos.cache import KVCache


def _text_config() -> Gemma3TextConfig:
    return Gemma3TextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
        head_dim=16,
        query_pre_attn_scalar=16,
        sliding_window=16,
        sliding_window_pattern=2,
        rope_theta=10000.0,
        rope_local_base_freq=10000.0,
        pad_token_id=0,
    )


def _resolve(monkeypatch, hf_config):
    monkeypatch.setattr(export_pipeline.AutoConfig, "from_pretrained", lambda *_, **__: hf_config)
    config = export_pipeline.ExportConfig(
        hf_model_id="google/gemma-3-1b-it", compute_precision="bfloat16"
    )
    return export_pipeline.resolve_model(config).hf_config


def test_resolve_model_keeps_text_only_config(monkeypatch):
    text_config = _text_config()
    assert _resolve(monkeypatch, text_config) is text_config


def test_resolve_model_still_unwraps_multimodal_config(monkeypatch):
    text_config = _text_config()
    wrapped = Gemma3Config(text_config=text_config)
    assert isinstance(_resolve(monkeypatch, wrapped), Gemma3TextConfig)


def test_memory_efficient_load_reads_text_only_weights(monkeypatch, tmp_path):
    torch.manual_seed(0)
    reference = HFGemma3ForCausalLM(_text_config()).to(torch.float32).eval()
    reference.save_pretrained(tmp_path, safe_serialization=True)
    monkeypatch.setattr(models_base, "snapshot_download", lambda *_, **__: str(tmp_path))

    entry = get_model_entry("gemma3_text")
    model = Gemma3ForCausalLM.from_hf_memory_efficient(
        "google/gemma-3-1b-it",
        target_dtype=torch.float32,
        hf_config_attr=entry.hf_config_attr,
        hf_state_dict_prefix=entry.hf_state_dict_prefix,
    )

    assert not any(p.is_meta for p in model.parameters())

    # A prefix filter that dropped every key would leave nothing the logits could match.
    input_ids = torch.randint(1, 100, (1, 4))
    position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
    k_cache, v_cache = KVCache.create_cache_tensors(_text_config(), dtype=torch.float32)
    with torch.no_grad():
        logits = model.eval()(input_ids, position_ids, k_cache, v_cache)
        expected = reference(input_ids=input_ids, position_ids=position_ids.long()).logits
    torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)
