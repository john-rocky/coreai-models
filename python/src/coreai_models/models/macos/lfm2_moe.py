# LFM2.5-MoE text decoder (HF model_type `lfm2_moe`, e.g.
# LiquidAI/LFM2.5-8B-A1B) for the Core AI authoring path.
#
# Community port — NOT an Apple model. LFM2.5-MoE = the LFM2.5 conv +
# full-attention hybrid (reused verbatim from lfm2.py: short-conv mixer,
# per-head q/k RMSNorm GQA attention, fused conv-state write) with the dense
# feed-forward replaced by a sparse-MoE block on every layer at or beyond
# `num_dense_layers`. The first `num_dense_layers` layers keep a plain dense
# MLP at the raw `intermediate_size` (NOT the 2/3 block-auto-adjusted ff_dim —
# LFM2.5-MoE ships the dense width directly).
#
# The sparse-MoE block follows transformers' Lfm2MoeSparseMoeBlock EXACTLY,
# which differs from Qwen3.5-MoE in three ways that matter for numerics:
#   1. SIGMOID routing (not softmax): routing_weights = sigmoid(router_logits)
#   2. expert_bias added ONLY to the selection scores — the top-k indices come
#      from (sigmoid + expert_bias) but the gathered weights come from the raw
#      sigmoid. (HF `route_tokens_to_experts`.)
#   3. NO shared expert (Qwen3.5-MoE has one; LFM2.5-MoE does not).
# Normalization is norm_topk_prob with a +1e-6 denominator guard, then a
# routed_scaling_factor multiply (1.0 for 8B-A1B). Experts run through the
# SwitchGLU/GatherMM composite — the same data-dependent expert gather Apple's
# qwen3_moe/gpt_oss use; the gather is lowered, never a Python expert loop.
#
# Checkpoint mapping (LiquidAI original layout — experts stored UNPACKED, one
# tensor per expert, unlike Qwen3.5-MoE's packed gate_up_proj):
#   model.embed_tokens / embedding_norm        -> same
#   ...layers.N.{operator_norm,ffn_norm}       -> same
#   ...layers.N.conv.{in_proj,out_proj,conv}   -> same (conv layers)
#   ...layers.N.self_attn.{q,k,v,out}_proj/...norm -> same (attn layers)
#   dense (N < num_dense_layers):
#     ...feed_forward.{w1,w3,w2}.weight        -> ...feed_forward.{gate,up,down}_proj.weight
#   moe (N >= num_dense_layers):
#     ...feed_forward.gate.weight   [E, d]     -> ...feed_forward.gate.weight (router)
#     ...feed_forward.expert_bias   [E]        -> ...feed_forward.expert_bias (buffer)
#     ...feed_forward.experts.{e}.w1.weight [I,d] -> stack_e -> ...switch_mlp.gate_proj.weight [1,E,I,d]
#     ...feed_forward.experts.{e}.w3.weight [I,d] -> stack_e -> ...switch_mlp.up_proj.weight   [1,E,I,d]
#     ...feed_forward.experts.{e}.w2.weight [d,I] -> stack_e -> ...switch_mlp.down_proj.weight [1,E,d,I]
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass

import torch
import torch.nn as nn

from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.switch import SwitchGLU

from .lfm2 import (
    DECODE_STATE_NAMES,  # noqa: F401  (re-exported for export scripts)
    Lfm2Config,
    Lfm2DecoderLayer,
    Lfm2ForCausalLMStateful,
    Lfm2Model,
    _MLP_KEY_MAP,
    build_decode_state,  # noqa: F401  (re-exported for export scripts)
)


@dataclass
class Lfm2MoeConfig(Lfm2Config):
    """Lfm2Config + the sparse-MoE fields (values from LFM2.5-8B-A1B config).

    ``block_auto_adjust_ff_dim`` defaults False here: the dense layers use the
    raw ``intermediate_size`` (LFM2.5-MoE ships 7168 directly; the 2/3 adjust
    that LFM2.5-dense applies would give the wrong width)."""

    block_auto_adjust_ff_dim: bool = False
    moe_intermediate_size: int = 1792
    num_experts: int = 32
    num_experts_per_tok: int = 4
    num_dense_layers: int = 2
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0
    use_expert_bias: bool = True

    def is_moe(self, layer_idx: int) -> bool:
        return layer_idx >= self.num_dense_layers


# --------------------------------------------------------------------------- #
# Sparse-MoE FFN — transformers Lfm2MoeSparseMoeBlock on the SwitchGLU composite
# --------------------------------------------------------------------------- #
class Lfm2MoeSparseBlock(nn.Module):
    """Sigmoid-routed top-k MoE with optional selection-only expert bias and
    NO shared expert. Scores stay fp32 through the weighted accumulate (matches
    Apple's qwen3_moe authoring; HF accumulates per-expert in the model dtype,
    so this is equal-or-better against the fp32 oracle)."""

    def __init__(self, config: Lfm2MoeConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.use_expert_bias = config.use_expert_bias
        self.gate = nn.Linear(d, config.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(d, config.moe_intermediate_size,
                                    config.num_experts, bias=False)
        if self.use_expert_bias:
            self.register_buffer(
                "expert_bias",
                torch.zeros(config.num_experts, dtype=torch.float32),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing_weights = torch.sigmoid(self.gate(x).float())  # [b,s,E] fp32
        if self.use_expert_bias:
            scores_for_routing = routing_weights + self.expert_bias
            _, indices = torch.topk(scores_for_routing, self.top_k, dim=-1)  # [b,s,k]
            scores = torch.gather(routing_weights, -1, indices)  # weights from raw sigmoid
        else:
            scores, indices = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-6)
        scores = scores * self.routed_scaling_factor
        y = self.switch_mlp(x, indices.to(torch.uint16))  # [b,s,k,d] (x.dtype)
        y = (y * scores.unsqueeze(-1)).sum(dim=-2)  # fp32 weighted accumulate
        return y.to(x.dtype)


# --------------------------------------------------------------------------- #
# Decoder layer — dense MLP for the first num_dense_layers, sparse MoE after
# --------------------------------------------------------------------------- #
class Lfm2MoeDecoderLayer(Lfm2DecoderLayer):
    """Mixer dispatch (conv vs full attention) / residual wiring / norms come
    from the base; only the feed-forward differs by layer index."""

    def __init__(self, config: Lfm2MoeConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        if config.is_moe(layer_idx):
            self.feed_forward = Lfm2MoeSparseBlock(config)


class Lfm2MoeModel(Lfm2Model):
    def __init__(self, config: Lfm2MoeConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Lfm2MoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )


class Lfm2MoeForCausalLMStateful(Lfm2ForCausalLMStateful):
    """Stateful prefill+decode graph for the MoE family. forward /
    build_macos_export_spec are inherited (config-generic); only the model body
    and the checkpoint loader differ."""

    def __init__(self, config: Lfm2MoeConfig) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.model = Lfm2MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embedding:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.last_token_only = False


# --------------------------------------------------------------------------- #
# Config + weight loading (safetensors-direct; no transformers lfm2_moe needed)
# --------------------------------------------------------------------------- #
def lfm2_moe_config_from_dict(raw: dict) -> Lfm2MoeConfig:
    layer_types = raw.get("layer_types")
    if not layer_types:
        full_idxs = set(raw.get("full_attn_idxs") or [])
        layer_types = [
            "full_attention" if i in full_idxs else "conv"
            for i in range(raw["num_hidden_layers"])
        ]
    return Lfm2MoeConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        vocab_size=raw["vocab_size"],
        intermediate_size=raw["intermediate_size"],
        block_auto_adjust_ff_dim=False,  # dense layers use raw intermediate_size
        norm_eps=raw.get("norm_eps", 1e-5),
        tie_embedding=raw.get("tie_word_embeddings", raw.get("tie_embedding", True)),
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        conv_L_cache=raw.get("conv_L_cache", 3),
        conv_bias=raw.get("conv_bias", False),
        rope_theta=(raw.get("rope_parameters") or {}).get("rope_theta",
                                                          raw.get("rope_theta", 1e6)),
        max_position_embeddings=raw.get("max_position_embeddings", 128000),
        layer_types=list(layer_types),
        moe_intermediate_size=raw["moe_intermediate_size"],
        num_experts=raw["num_experts"],
        num_experts_per_tok=raw["num_experts_per_tok"],
        num_dense_layers=raw.get("num_dense_layers", 0),
        norm_topk_prob=raw.get("norm_topk_prob", True),
        routed_scaling_factor=raw.get("routed_scaling_factor", 1.0),
        use_expert_bias=raw.get("use_expert_bias", True),
    )


# experts.{e}.wK -> the SwitchGLU projection it feeds (w1=gate, w3=up, w2=down)
_EXPERT_PROJ_MAP = {"w1": "gate_proj", "w3": "up_proj", "w2": "down_proj"}
_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+\.feed_forward)\.experts\.(\d+)\.(w1|w2|w3)\.weight$"
)


def lfm2_moe_from_hf(
    huggingface_model_id: str,
    target_dtype: torch.dtype = torch.float16,
    fp32_attn_proj: bool = True,
):
    """Load LFM2.5-MoE from the HF checkpoint into the authored module.

    Experts are stored one tensor per expert in the checkpoint; they are
    stacked into the [1, E, out, in] SwitchLinear layout. As in lfm2.py the
    attention q/k/v/out projections stay fp32 on an fp16 load (the GPU-delegate
    precision-critical path — LFM2.5's large q/k-norm gains amplify fp16 matmul
    noise)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    model_dir = snapshot_download(
        huggingface_model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )
    with open(os.path.join(model_dir, "config.json")) as f:
        config = lfm2_moe_config_from_dict(json.load(f))

    with torch.device("meta"):
        model = Lfm2MoeForCausalLMStateful(config)
    model.to(dtype=target_dtype)

    attn_proj_re = re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.weight$")
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    # Accumulate per-expert tensors, then stack into SwitchLinear params.
    # expert_stack[(ff_prefix, proj)] = {expert_idx: tensor}
    expert_stack: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
    sd: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                m = _EXPERT_RE.match(key)
                if m is not None:
                    ff_prefix, e_idx, proj = m.group(1), int(m.group(2)), m.group(3)
                    t = f.get_tensor(key)
                    if t.dtype != target_dtype:
                        t = t.to(target_dtype)
                    expert_stack.setdefault((ff_prefix, _EXPERT_PROJ_MAP[proj]), {})[
                        e_idx
                    ] = t
                    continue
                # expert_bias stays fp32 (matches the registered buffer dtype)
                if key.endswith(".feed_forward.expert_bias"):
                    sd[key] = f.get_tensor(key).to(torch.float32)
                    continue
                # dense MLP rename w1/w3/w2 -> gate/up/down
                local = key
                for old, new in _MLP_KEY_MAP.items():
                    if old in local:
                        local = local.replace(old, new)
                        break
                t = f.get_tensor(key)
                dtype = target_dtype
                if fp32_attn_proj and attn_proj_re.search(local):
                    dtype = torch.float32
                if t.dtype != dtype and t.is_floating_point():
                    t = t.to(dtype)
                sd[local] = t

    E = config.num_experts
    for (ff_prefix, proj), per_expert in expert_stack.items():
        if len(per_expert) != E:
            raise RuntimeError(
                f"{ff_prefix}.switch_mlp.{proj}: got {len(per_expert)} experts, expected {E}"
            )
        stacked = torch.stack([per_expert[e] for e in range(E)], dim=0)  # [E,out,in]
        sd[f"{ff_prefix}.switch_mlp.{proj}.weight"] = stacked.unsqueeze(0).contiguous()

    model.load_state_dict(sd, assign=True, strict=False)
    if config.tie_embedding:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.reset_buffers()

    meta_params = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_params:
        raise RuntimeError(f"Parameters not loaded: {meta_params}")
    model.eval()
    return model
