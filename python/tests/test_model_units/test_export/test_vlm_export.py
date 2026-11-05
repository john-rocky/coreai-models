# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the VLM specs, the Qwen3-VL decoder loader and StaticVisionEncoder."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.models.macos.qwen3_vl import Qwen3VLForCausalLMEmbeddings
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.vlm.export import (
    SUPPORTED_MODELS,
    StaticVisionEncoder,
    load_model_from_safetensors,
)


class _FakeVisualModel(nn.Module):
    """Minimal stand-in for the HF visual backbone."""

    def __init__(self, hidden: int = 32, patch_dim: int = 768):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, hidden, bias=False)
        self.blocks = nn.ModuleList()
        self.merger = nn.Linear(hidden, hidden, bias=False)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        t, h, w = grid_thw[0].tolist()
        return torch.zeros(t * h * w, 32)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        t, h, w = grid_thw[0].tolist()
        return torch.zeros(t * h * w, 8)


_COMMON = dict(image_size=32, patch_size=16, spatial_merge_size=1, temporal_patch_size=2)


class TestStaticVisionEncoderGridT:
    def test_single_image_default(self):
        enc = StaticVisionEncoder(_FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON)
        assert enc.grid_t == 1
        assert enc.num_frames == 1

    def test_single_image_explicit(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=1
        )
        assert enc.grid_t == 1

    def test_multi_frame_divisible(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=4
        )
        assert enc.grid_t == 2
        assert enc.num_patches == 2 * 2 * 2  # grid_t * grid_h * grid_w

    def test_multi_frame_not_divisible_raises(self):
        with pytest.raises(ValueError, match="must be divisible"):
            StaticVisionEncoder(
                _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=3
            )

    def test_patchify_single_image_shape(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=1
        )
        pixels = torch.randn(1, 3, 32, 32)
        patches = enc._patchify(pixels)
        assert patches.shape == (enc.num_patches, enc.patch_dim)

    def test_patchify_multi_frame_shape(self):
        enc = StaticVisionEncoder(
            _FakeVisualModel(patch_dim=2 * 3 * 16 * 16), **_COMMON, num_frames=4
        )
        pixels = torch.randn(1, 3 * 4, 32, 32)
        patches = enc._patchify(pixels)
        assert patches.shape == (enc.num_patches, enc.patch_dim)


class TestQwen3VLSpecs:
    @pytest.mark.parametrize(
        ("short_name", "hf_model_id", "output_name"),
        [
            ("qwen3-vl", "Qwen/Qwen3-VL-2B-Instruct", "qwen3_vl_2b"),
            ("qwen3-vl-4b", "Qwen/Qwen3-VL-4B-Instruct", "qwen3_vl_4b"),
            ("qwen3-vl-8b", "Qwen/Qwen3-VL-8B-Instruct", "qwen3_vl_8b"),
        ],
    )
    def test_entry(self, short_name: str, hf_model_id: str, output_name: str):
        spec = SUPPORTED_MODELS[short_name]
        assert spec.short_name == short_name
        assert spec.hf_model_id == hf_model_id
        assert spec.output_name == output_name
        # The family shares one vision geometry and the image placeholder token.
        assert spec.image_token_id == 151655
        assert spec.num_visual_tokens == 196
        assert spec.image_mean == spec.image_std == (0.5, 0.5, 0.5)
        assert build_aimodel_metadata(hf_model_id).license == "Apache-2.0"


class TestQwen3VLDecoderLoader:
    """A checkpoint shaped like 4B and 8B: head_dim is not hidden / heads, the head tied or not."""

    @staticmethod
    def _checkpoint(path: Path, tie_word_embeddings: bool) -> Qwen3VLForConditionalGeneration:
        text = Qwen3VLTextConfig(
            vocab_size=64,
            hidden_size=40,
            intermediate_size=80,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=64,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 5_000_000,
                "mrope_section": [4, 2, 2],
                "mrope_interleaved": True,
            },
        )
        vision = Qwen3VLVisionConfig(
            hidden_size=32, intermediate_size=64, depth=1, num_heads=2, out_hidden_size=40
        )
        config = Qwen3VLConfig(
            text_config=text, vision_config=vision, tie_word_embeddings=tie_word_embeddings
        )
        torch.manual_seed(0)
        model = Qwen3VLForConditionalGeneration(config).eval()
        model.save_pretrained(path)
        return model

    @pytest.mark.parametrize("tie_word_embeddings", [True, False])
    def test_head_and_logits_match_the_checkpoint(self, tmp_path: Path, tie_word_embeddings: bool):
        reference = self._checkpoint(tmp_path, tie_word_embeddings)
        model = load_model_from_safetensors(
            Qwen3VLForCausalLMEmbeddings,
            AutoConfig.from_pretrained(tmp_path),
            str(tmp_path),
            max_ctx=32,
            num_layers=None,
            dtype=torch.float32,
        ).eval()

        embeddings = reference.model.language_model.embed_tokens
        assert torch.equal(model.lm_head.weight, reference.lm_head.weight)
        assert torch.equal(model.lm_head.weight, embeddings.weight) is tie_word_embeddings

        input_ids = torch.arange(1, 9).unsqueeze(0)
        with torch.no_grad():
            inputs_embeds = embeddings(input_ids)
            expected = reference.lm_head(
                reference.model.language_model(inputs_embeds=inputs_embeds).last_hidden_state
            )
            k_cache, v_cache = KVCache.create_cache_tensors(model.config, seq_len=32)
            position_ids = torch.arange(8, dtype=torch.int32).unsqueeze(0)
            logits = model(inputs_embeds, position_ids, k_cache, v_cache)
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=0)
