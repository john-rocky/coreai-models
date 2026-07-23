# Qwen3.6 MoE text decoder (HF model_type `qwen3_5_moe`, e.g. Qwen/Qwen3.6-35B-A3B)
# for the Core AI authoring path.
#
# Community port — NOT an Apple model.  The token mixers (hybrid GatedDeltaNet /
# gated full attention, 3:1 interleave, mRoPE-as-partial-RoPE) are exactly Qwen3.5
# and are reused from qwen3_5.py, including the GVA query/key head repeat
# (35B-A3B runs 32 value heads over 16 key heads).  The only new block is the
# sparse-MoE FFN, mirrored from transformers' Qwen3_5MoeSparseMoeBlock:
#
#   router  : Linear(d, E) -> fp32 softmax over ALL E -> top-k -> renormalize
#             by the top-k sum (norm_topk_prob is unconditional in HF)
#   experts : per-expert GLU via SwitchGLU (GatherMM composite — data-dependent
#             expert gather, same primitive Apple's qwen3_moe/gpt_oss use)
#   shared  : an always-on MLP(shared_expert_intermediate_size) scaled by a
#             per-token sigmoid scalar gate, added to the sparse sum
#
# Checkpoint mapping (transformers >= 5.x packed-expert layout):
#   model.language_model.*                   -> model.*            (text decoder)
#   lm_head.weight                           -> lm_head.weight     (untied)
#   ...mlp.gate.weight            [E, d]     -> ...mlp.gate.weight (router)
#   ...mlp.experts.gate_up_proj   [E, 2I, d] -> ...mlp.switch_mlp.{gate,up}_proj.weight  [1,E,I,d]
#   ...mlp.experts.down_proj      [E, d, I]  -> ...mlp.switch_mlp.down_proj.weight       [1,E,d,I]
#   ...mlp.shared_expert.{gate,up,down}_proj -> same names
#   ...mlp.shared_expert_gate.weight [1, d]  -> same name
#   model.visual.* / mtp.*                   -> skipped (vision tower, MTP head)
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.switch import SwitchGLU

from .qwen3_5 import (
    DECODE_STATE_NAMES,  # noqa: F401  (re-exported for export scripts)
    Qwen3_5Config,
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5StatefulForCausalLM,
    build_decode_state,  # noqa: F401  (re-exported for export scripts)
)


@dataclass
class Qwen3_5MoeConfig(Qwen3_5Config):
    """Qwen3_5Config + the sparse-MoE fields (values from 35B-A3B text_config)."""

    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts: int = 256
    num_experts_per_tok: int = 8


class Qwen3_5MoeSparseBlock(nn.Module):
    """HF Qwen3_5MoeSparseMoeBlock re-authored on the SwitchGLU/GatherMM composite.

    Scores stay fp32 through the weighted sum (the broadcast multiply promotes),
    matching Apple's qwen3_moe authoring; HF accumulates per-expert in the model
    dtype, so this is equal-or-better precision against the fp32 oracle.
    """

    def __init__(self, config: Qwen3_5MoeConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(d, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(d, config.moe_intermediate_size,
                                    config.num_experts, bias=False)
        self.shared_expert = MLP(d, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.gate(x).float(), dim=-1)  # [b,s,E] fp32
        scores, indices = torch.topk(probs, self.top_k, dim=-1)  # [b,s,k]
        scores = scores / scores.sum(dim=-1, keepdim=True)
        y = self.switch_mlp(x, indices.to(torch.uint16))  # [b,s,k,d] (x.dtype)
        y = (y * scores.unsqueeze(-1)).sum(dim=-2)  # fp32 weighted accumulate
        shared = self.shared_expert(x) * torch.sigmoid(self.shared_expert_gate(x))
        return (y + shared.to(torch.float32)).to(x.dtype)


class Qwen3_5MoeDecoderLayer(Qwen3_5DecoderLayer):
    """Dense decoder layer with the MLP swapped for the sparse-MoE block.
    Mixer dispatch / residual wiring / norms come from the base class."""

    def __init__(self, config: Qwen3_5MoeConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.mlp = Qwen3_5MoeSparseBlock(config)


class Qwen3_5MoeModel(Qwen3_5Model):
    def _make_layer(self, config: Qwen3_5MoeConfig, layer_idx: int) -> nn.Module:
        return Qwen3_5MoeDecoderLayer(config, layer_idx)


def qwen3_5_moe_config_from_json(
    text_cfg: dict,
    max_context_length: int | None = None,
    num_layers: int | None = None,
) -> Qwen3_5MoeConfig:
    """Build the authoring config straight from config.json's ``text_config`` dict
    (no transformers AutoConfig — the export env's transformers predates
    qwen3_5_moe)."""
    del max_context_length  # context length is an export-time concern
    n_layers = num_layers if num_layers is not None else text_cfg["num_hidden_layers"]
    layer_types = list(text_cfg.get("layer_types") or [])[:n_layers]
    rope = text_cfg.get("rope_parameters") or {}
    return Qwen3_5MoeConfig(
        hidden_size=text_cfg["hidden_size"],
        num_hidden_layers=n_layers,
        vocab_size=text_cfg["vocab_size"],
        # no dense FFN in this arch; keep the (replaced) base MLP construction tiny
        intermediate_size=text_cfg.get("intermediate_size",
                                       text_cfg["moe_intermediate_size"]),
        rms_norm_eps=text_cfg.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=bool(text_cfg.get("tie_word_embeddings", False)),
        head_dim=text_cfg["head_dim"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_key_value_heads=text_cfg["num_key_value_heads"],
        partial_rotary_factor=rope.get("partial_rotary_factor", 0.25),
        rope_theta=rope.get("rope_theta", 1e7),
        linear_num_key_heads=text_cfg["linear_num_key_heads"],
        linear_num_value_heads=text_cfg["linear_num_value_heads"],
        linear_key_head_dim=text_cfg["linear_key_head_dim"],
        linear_value_head_dim=text_cfg["linear_value_head_dim"],
        linear_conv_kernel_dim=text_cfg["linear_conv_kernel_dim"],
        full_attention_interval=text_cfg["full_attention_interval"],
        layer_types=layer_types,
        moe_intermediate_size=text_cfg["moe_intermediate_size"],
        shared_expert_intermediate_size=text_cfg["shared_expert_intermediate_size"],
        num_experts=text_cfg["num_experts"],
        num_experts_per_tok=text_cfg["num_experts_per_tok"],
    )


QWEN3_5_TEXT_PREFIX = "model.language_model."


class Qwen3_5MoeStatefulForCausalLM(Qwen3_5StatefulForCausalLM):
    """Stateful prefill+decode graph for the MoE family. forward /
    build_macos_export_spec are inherited (they are config-generic); only the
    model body and the checkpoint loader differ."""

    def _init_model(self, config: Qwen3_5MoeConfig) -> None:
        self.model = Qwen3_5MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.last_token_only = False

    @classmethod
    def from_hf_memory_efficient(
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        hf_config_attr: str | None = "text_config",
        hf_state_dict_prefix: str = QWEN3_5_TEXT_PREFIX,
    ):
        """Load the MoE text decoder from the multimodal checkpoint: unpack the
        packed expert tensors into SwitchLinear layout, pick up the untied
        ``lm_head.weight`` from the checkpoint root, skip vision/MTP weights."""
        del mmap_path  # 128 GB host: keep weights in RAM (disk is the constraint)
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        from coreai_models.models.base import _is_layer_key_beyond

        model_dir = snapshot_download(
            huggingface_model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
        with open(os.path.join(model_dir, "config.json")) as f:
            raw = json.load(f)
        text_cfg = raw[hf_config_attr] if hf_config_attr else raw
        config = qwen3_5_moe_config_from_json(text_cfg, max_context_length, num_layers)

        model = cls(config, model_device="meta")
        model.to(dtype=target_dtype)

        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors files in {model_dir}")
        prefix = hf_state_dict_prefix
        inter = config.moe_intermediate_size
        sd: dict[str, torch.Tensor] = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key == "lm_head.weight" and not config.tie_word_embeddings:
                        sd[key] = f.get_tensor(key).to(target_dtype)
                        continue
                    if not key.startswith(prefix):
                        continue  # skips model.visual.* and mtp.*
                    local = "model." + key[len(prefix):]
                    if num_layers is not None and _is_layer_key_beyond(local, num_layers):
                        continue
                    tensor = f.get_tensor(key)
                    if tensor.dtype != target_dtype and tensor.is_floating_point():
                        tensor = tensor.to(target_dtype)
                    if local.endswith(".mlp.experts.gate_up_proj"):
                        base = local[: -len("experts.gate_up_proj")] + "switch_mlp"
                        sd[f"{base}.gate_proj.weight"] = (
                            tensor[:, :inter, :].unsqueeze(0).contiguous()
                        )
                        sd[f"{base}.up_proj.weight"] = (
                            tensor[:, inter:, :].unsqueeze(0).contiguous()
                        )
                        continue
                    if local.endswith(".mlp.experts.down_proj"):
                        base = local[: -len("experts.down_proj")] + "switch_mlp"
                        sd[f"{base}.down_proj.weight"] = tensor.unsqueeze(0).contiguous()
                        continue
                    sd[local] = tensor

        model.load_state_dict(sd, assign=True, strict=False)
        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        model.model.reset_buffers()

        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(f"Parameters not loaded: {meta_params}")
        return model
