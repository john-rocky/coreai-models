# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for StaticVisionEncoder grid_t / num_frames handling and the VLM specs."""

import jinja2
import pytest
import torch
import torch.nn as nn

from coreai_models.vlm.export import (
    SUPPORTED_MODELS,
    StaticVisionEncoder,
    _wrap_image_token_in_chat_template,
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


class TestMiniCPMV46Spec:
    _IMAGE_TOKEN = "<|image_pad|>"
    # The shape of the checkpoint's template around string content.
    _TEMPLATE = "{%- if content is string -%}{{- content -}}{%- endif -%}"

    def _render(self, template: str, content: str) -> str:
        return jinja2.Environment().from_string(template).render(content=content)

    def test_two_merges_give_64_tokens_at_448(self):
        assert SUPPORTED_MODELS["minicpm-v-4.6"].num_visual_tokens == 64

    def test_wrapper_surrounds_the_placeholder(self):
        spec = SUPPORTED_MODELS["minicpm-v-4.6"]
        template = _wrap_image_token_in_chat_template(self._TEMPLATE, self._IMAGE_TOKEN, spec)
        rendered = self._render(template, f"{self._IMAGE_TOKEN}\nWhat is this?")
        assert rendered == "<image_id>0</image_id><image><|image_pad|></image>\nWhat is this?"

    def test_text_only_content_is_unchanged(self):
        spec = SUPPORTED_MODELS["minicpm-v-4.6"]
        template = _wrap_image_token_in_chat_template(self._TEMPLATE, self._IMAGE_TOKEN, spec)
        assert self._render(template, "What is this?") == "What is this?"

    @pytest.mark.parametrize(
        "template",
        [
            pytest.param("{{ content }}", id="missing"),
            pytest.param("{{- content -}}{{- content -}}", id="ambiguous"),
        ],
    )
    def test_refuses_a_template_it_does_not_recognize(self, template):
        spec = SUPPORTED_MODELS["minicpm-v-4.6"]
        with pytest.raises(ValueError, match="chat template"):
            _wrap_image_token_in_chat_template(template, self._IMAGE_TOKEN, spec)
