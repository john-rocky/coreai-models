# Phi-3 / Phi-4-mini (`Phi3ForCausalLM`) text decoder for the Core AI path.
#
# Community port — NOT an Apple model.  A standard pre-norm dense GQA decoder
# with two Phi-isms, both already in the HF checkpoint layout (so no state-dict
# fusion is needed):
#   * **Fused QKV** — `qkv_proj` packs [all-q | all-k | all-v] (bias-free).
#   * **Fused gate/up MLP** — `gate_up_proj` packs [gate | up]; the block is
#     `down(up * silu(gate))`.
# Phi-4-mini additionally uses **partial rotary** (`partial_rotary_factor=0.75`
# → only the first 96 of 128 head dims are rotated) and **LongRoPE** frequency
# rescaling; both are handled by `LongRoPE` (short-factor regime).

import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import (
    Phi3Config,
)
from transformers.models.phi3.modeling_phi3 import (
    Phi3ForCausalLM as HFPhi3ForCausalLM,
)
from typing_extensions import Self, override

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import (
    LongRoPE,
    compute_longrope_attention_scaling,
    initialize_rope,
)
from coreai_models.primitives.macos.sdpa import SDPA


def _rotary_dim(config: Phi3Config, head_dim: int) -> int:
    return int(head_dim * getattr(config, "partial_rotary_factor", 1.0))


def _make_phi3_rope(config: Phi3Config, head_dim: int) -> nn.Module:
    base = resolve_rope_theta(config)
    rotary_dim = _rotary_dim(config, head_dim)
    scaling = getattr(config, "rope_scaling", None)
    rope_type = None
    if isinstance(scaling, dict):
        rope_type = scaling.get("rope_type") or scaling.get("type")
    if isinstance(scaling, dict) and rope_type == "longrope":
        original_max = (
            getattr(config, "original_max_position_embeddings", None)
            or config.max_position_embeddings
        )
        # `attention_scaling` is a fixed model property derived from the DESIGNED
        # context (e.g. 131072), not the export-capped KV context. `_get_reauthored_config`
        # stashes the designed max before capping so a 4096-context ANE export
        # still gets the correct factor.
        designed_max = getattr(config, "_longrope_designed_max", config.max_position_embeddings)
        attention_scaling = compute_longrope_attention_scaling(designed_max, original_max)
        return LongRoPE(
            head_dim,
            rotary_dim,
            base,
            scaling["short_factor"],
            attention_scaling=attention_scaling,
        )
    # Plain Phi-3 (no longrope): default RoPE, partial if rotary_dim < head_dim.
    return initialize_rope(dims=(rotary_dim if rotary_dim < head_dim else None), base=base)


class Phi3MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up_states = self.gate_up_proj(x)
        gate, up_states = up_states.chunk(2, dim=-1)
        return self.down_proj(up_states * nn.functional.silu(gate))


class Attention(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", None) or dim // n_heads

        # Phi packs qkv (and gate/up) — bias-free; matches HF key names directly.
        self.qkv_proj = nn.Linear(
            dim,
            n_heads * head_dim + n_kv_heads * head_dim + n_kv_heads * head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.sdpa = SDPA(is_causal=True)
        self.rope = _make_phi3_rope(config, head_dim)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        qkv = (
            self.qkv_proj(x)
            .reshape(batch_size, query_len, n_heads + 2 * n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        query_key = qkv.narrow(1, 0, n_heads + n_kv_heads)
        value = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        query_key = self.rope(query_key, position_ids=rope_positions)
        query = query_key.narrow(1, 0, n_heads)
        key = query_key.narrow(1, n_heads, n_kv_heads)

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
            )

        output = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = Phi3MLP(hidden_size, config.intermediate_size)

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Phi3Model(nn.Module):
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class Phi3ForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFPhi3ForCausalLM

    @classmethod
    def _get_reauthored_config(cls, hf_config, max_context_length=None, num_layers=None):
        # Stash the DESIGNED context length before it is capped to the export
        # KV context, so longrope attention scaling is computed from the model's
        # true ratio (e.g. 131072/4096) rather than the capped one.
        if not hasattr(hf_config, "_longrope_designed_max"):
            hf_config._longrope_designed_max = hf_config.max_position_embeddings
        return super()._get_reauthored_config(hf_config, max_context_length, num_layers=num_layers)

    @override
    def _init_model(self, config: Phi3Config) -> None:
        self.model = Phi3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, cache)
        return self.lm_head(out)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # No fusion: HF Phi-3 already ships fused qkv_proj / gate_up_proj and the
        # reauthored module names mirror the HF keys 1:1.
        return

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
