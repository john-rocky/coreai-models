# OLMo-2 (`Olmo2ForCausalLM`) text decoder for the Core AI authoring path.
#
# Community port — NOT an Apple model.  OLMo-2 differs from a plain Llama/Qwen
# decoder in two ways, both replicated here:
#   1. **Post-norm placement.**  The attention/MLP sub-layers read the *raw*
#      residual (no input/pre-feedforward norm); their outputs are normalized
#      and then added back (`x + post_attention_layernorm(attn(x))`).
#   2. **QK-norm on the flat projection.**  `q_norm`/`k_norm` are RMSNorms over
#      the whole `num_heads * head_dim` (resp. `num_kv_heads * head_dim`) vector,
#      applied BEFORE the head split — so the projections are NOT fused into a
#      single QKV (the per-head fused norm used by qwen3 would change the
#      normalization denominator).
# Otherwise standard dense GQA + RoPE (default) + SwiGLU + RMSNorm.

import torch
import torch.nn as nn
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2Config,
)
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2ForCausalLM as HFOlmo2ForCausalLM,
)
from typing_extensions import Self, override

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA


class Attention(nn.Module):
    def __init__(self, config: Olmo2Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", None) or dim // n_heads

        bias = getattr(config, "attention_bias", False)
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

        # OLMo-2 normalizes the full (all-heads) q/k projection, not per-head.
        self.q_norm = RMSNorm(n_heads * head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(n_kv_heads * head_dim, eps=config.rms_norm_eps)

        self.sdpa = SDPA(is_causal=True)
        self.rope = initialize_rope(base=resolve_rope_theta(config))

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        # Flat QK-norm before splitting into heads.
        query = self.q_norm(self.q_proj(x))
        key = self.k_norm(self.k_proj(x))
        value = self.v_proj(x)

        query = query.reshape(batch_size, query_len, n_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.reshape(batch_size, query_len, n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.reshape(batch_size, query_len, n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        query = self.rope(query, position_ids=rope_positions)
        key = self.rope(key, position_ids=rope_positions)

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
    def __init__(self, config: Olmo2Config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(hidden_size, config.intermediate_size)

        # OLMo-2 post-norm: normalize the sub-layer OUTPUT, not the input.
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = x + self.post_attention_layernorm(self.self_attn(x, position_ids, cache))
        return h + self.post_feedforward_layernorm(self.mlp(h))


class Olmo2Model(nn.Module):
    def __init__(self, config: Olmo2Config) -> None:
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


class Olmo2ForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFOlmo2ForCausalLM

    @override
    def _init_model(self, config: Olmo2Config) -> None:
        self.model = Olmo2Model(config)
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
        # No fusion: reauthored module names mirror the HF keys 1:1
        # (q/k/v/o_proj, q_norm/k_norm, post_attention_layernorm,
        # post_feedforward_layernorm, mlp.{gate,up,down}_proj, norm, lm_head).
        return

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
