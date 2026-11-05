# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the MiniCPM-V-4.6 VLM components: the vision encoder and the embeddings decoder.

All tests use small configs with random weights -- no HF downloads.
"""

import pytest
import torch
from transformers.models.minicpmv4_6.configuration_minicpmv4_6 import MiniCPMV4_6Config
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import MiniCPMV4_6Model
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from coreai_models._constants import (
    CONV_STATES_NAME,
    KEY_CACHE_NAME,
    MAIN_GRAPH_NAME,
    RECURRENT_STATES_NAME,
    VALUE_CACHE_NAME,
)
from coreai_models.models.base import TraceSpec
from coreai_models.models.macos.minicpmv4_6_vision import MiniCPMV46VisionModel
from coreai_models.models.macos.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForCausalLMEmbeddings

_PATCH = 14
_PREFIXES = ("vision_tower.", "merger.")


def _make_hf_vision(table_size: int) -> MiniCPMV4_6Model:
    """A tiny HF model. ``table_size`` sets the position table, as 980 does on the checkpoint."""
    config = MiniCPMV4_6Config(
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "image_size": table_size,
            "patch_size": _PATCH,
            "layer_norm_eps": 1e-6,
        },
        text_config={
            "model_type": "qwen3_5_text",
            "hidden_size": 24,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 12,
            "intermediate_size": 48,
            "vocab_size": 50,
            "layer_types": ["full_attention"],
        },
        insert_layer_id=1,
    )
    torch.manual_seed(0)
    model = MiniCPMV4_6Model(config).eval()
    # Large enough that the tower's tanh GELU and the merger's exact GELU differ measurably.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.5)
    return model


def _make_vision(hf_model: MiniCPMV4_6Model, image_size: int) -> MiniCPMV46VisionModel:
    """The re-authored encoder, loaded with ``hf_model``'s weights the production way."""
    config = hf_model.config
    vision = config.vision_config
    model = MiniCPMV46VisionModel(
        image_size=image_size,
        hidden_size=vision.hidden_size,
        intermediate_size=vision.intermediate_size,
        num_hidden_layers=vision.num_hidden_layers,
        num_attention_heads=vision.num_attention_heads,
        patch_size=vision.patch_size,
        layer_norm_eps=vision.layer_norm_eps,
        insert_layer_id=config.insert_layer_id,
        text_hidden_size=config.text_config.hidden_size,
    )
    tensors = {
        f"model.{k}": v.clone() for k, v in hf_model.state_dict().items() if k.startswith(_PREFIXES)
    }
    model.load_checkpoint_tensors(tensors, table_side=vision.image_size // vision.patch_size)
    return model.eval()


def _navit_pack(pixel_values: torch.Tensor) -> torch.Tensor:
    """``[1, 3, H, W]`` -> the ``[1, 3, patch, H*W/patch]`` packing of the HF image processor."""
    patches = torch.nn.functional.unfold(pixel_values, (_PATCH, _PATCH), stride=(_PATCH, _PATCH))
    patches = patches.reshape(3, _PATCH, _PATCH, -1).permute(0, 1, 3, 2)
    return patches.reshape(1, 3, _PATCH, -1)


class TestMiniCPMV46VisionModel:
    @pytest.mark.parametrize(
        ("table_size", "image_size"),
        [
            pytest.param(112, 112, id="grid_is_the_position_table"),
            # The checkpoint's case: the 448 input reads a 980 table sparsely.
            pytest.param(980, 448, id="grid_smaller_than_the_position_table"),
        ],
    )
    def test_matches_hf_image_features(self, table_size: int, image_size: int) -> None:
        hf_model = _make_hf_vision(table_size)
        model = _make_vision(hf_model, image_size)
        pixel_values = torch.randn(1, 3, image_size, image_size)
        grid = image_size // _PATCH

        with torch.no_grad():
            ours = model(pixel_values)
            theirs = hf_model.get_image_features(
                _navit_pack(pixel_values), torch.tensor([[grid, grid]])
            ).pooler_output[0]

        assert ours.shape == (1, (grid // 4) ** 2, hf_model.config.text_config.hidden_size)
        assert torch.allclose(ours[0], theirs, atol=1e-5, rtol=1e-4)

    def test_output_follows_the_weights_dtype_for_float32_pixels(self) -> None:
        model = _make_vision(_make_hf_vision(112), 112).to(torch.float16)
        out = model(torch.randn(1, 3, 112, 112, dtype=torch.float32))
        assert out.dtype == torch.float16

    def test_checkpoint_keys_must_cover_the_module(self) -> None:
        hf_model = _make_hf_vision(112)
        model = _make_vision(hf_model, 112)
        tensors = {
            f"model.{k}": v.clone()
            for k, v in hf_model.state_dict().items()
            if k.startswith(_PREFIXES) and "vit_merger.linear_1" not in k
        }
        with pytest.raises(RuntimeError, match="vit_merger.linear_1"):
            model.load_checkpoint_tensors(tensors, table_side=8)

    def test_position_table_is_sliced_to_the_grid(self) -> None:
        model = _make_vision(_make_hf_vision(980), 448)
        assert model.vision_tower.embeddings.position_embedding.shape == (32 * 32, 32)

    @pytest.mark.parametrize("image_size", [112 + 14, 100])
    def test_rejects_a_size_the_merges_cannot_divide(self, image_size: int) -> None:
        with pytest.raises(ValueError, match="divisible"):
            MiniCPMV46VisionModel(image_size=image_size)


def _make_decoder_config() -> Qwen3_5TextConfig:
    config = Qwen3_5TextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=100,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=True,
    )
    config.layer_types = ["linear_attention", "full_attention"]
    return config


class TestQwen3_5ForCausalLMEmbeddings:
    def test_matches_the_token_decoder_on_the_same_weights(self) -> None:
        config = _make_decoder_config()
        torch.manual_seed(0)
        tokens = Qwen3_5ForCausalLM(config).eval()
        embeds = Qwen3_5ForCausalLMEmbeddings(config).eval()
        embeds.load_state_dict(dict(tokens.state_dict()))

        input_ids = torch.randint(1, config.vocab_size, (1, 6), dtype=torch.int32)
        position_ids = torch.arange(6, dtype=torch.int32).unsqueeze(0)
        with torch.no_grad():
            expected = tokens(
                input_ids,
                position_ids,
                *tokens.new_eager_state(dtype=torch.float32, device="cpu").tensors,
            )
            actual = embeds(
                tokens.model.embed_tokens(input_ids),
                position_ids,
                *embeds.new_eager_state(dtype=torch.float32, device="cpu").tensors,
            )
        assert torch.equal(actual, expected)

    def test_head_stays_tied_to_the_table(self) -> None:
        model = Qwen3_5ForCausalLMEmbeddings(_make_decoder_config())
        assert model.lm_head.weight is model.model.embed_tokens.weight

    def test_export_contract(self) -> None:
        config = _make_decoder_config()
        model = Qwen3_5ForCausalLMEmbeddings(config)
        spec = TraceSpec(max_context_length=32, cache_seq_len=16, query_len=4, offset=2)
        reference = model.build_reference_inputs(config, torch.float16, spec)
        shapes = model.build_dynamic_shapes(config, spec)

        model.validate_export_contract(reference, shapes)
        assert model.export_input_names()[MAIN_GRAPH_NAME] == ("inputs_embeds", "position_ids")
        assert model.export_state_names()[MAIN_GRAPH_NAME] == (
            KEY_CACHE_NAME,
            VALUE_CACHE_NAME,
            CONV_STATES_NAME,
            RECURRENT_STATES_NAME,
        )

        graph = reference[MAIN_GRAPH_NAME]
        assert list(graph) == [
            "inputs_embeds",
            "position_ids",
            "k_cache",
            "v_cache",
            "conv_states",
            "recurrent_states",
        ]
        model.reference_inputs_as_args(graph)
        assert graph["inputs_embeds"].shape == (1, 4, config.hidden_size)
        assert graph["inputs_embeds"].dtype == torch.float16
        # One KV row per full-attention layer, one conv/recurrent row per linear-attention layer.
        assert graph["k_cache"].shape[0] == 1
        assert graph["conv_states"].shape[0] == 1

    def test_exports_no_prefill_graph(self) -> None:
        assert not Qwen3_5ForCausalLMEmbeddings.exports_prefill_graph

    def test_config_comes_from_the_nested_text_config(self) -> None:
        text_config = _make_decoder_config()
        hf_config = type("Wrapper", (), {"text_config": text_config})()
        config = Qwen3_5ForCausalLMEmbeddings._get_reauthored_config(hf_config, 32)
        assert config is text_config
        assert config.max_position_embeddings == 32

    def test_rejects_a_layer_count_cut(self) -> None:
        with pytest.raises(ValueError, match="hybrid"):
            Qwen3_5ForCausalLMEmbeddings._get_reauthored_config(_make_decoder_config(), 32, 1)
