# Phi-3 / Phi-4-mini (`Phi3ForCausalLM`) iOS (static-shape, palettizable).
#
# Mirrors `models/ios/mistral.py` (Conv2d projections + per-head decode SDPA) but
# handles the Phi specifics:
#   * HF ships FUSED `qkv_proj` / `gate_up_proj`; `_mutate_state_dict` SPLITS them
#     back into the separate q/k/v and gate/up Conv2d weights the iOS path uses.
#   * Phi-4-mini partial rotary (0.75) + LongRoPE: the RoPE cache is sized to the
#     rotary sub-dimension and `apply_rope_partial` rotates only that prefix.

import torch
import torch.nn as nn
from transformers.models.phi3.modeling_phi3 import (
    Phi3Config,
)
from transformers.models.phi3.modeling_phi3 import (
    Phi3ForCausalLM as HFPhi3ForCausalLM,
)

from coreai_models._hf import resolve_rope_theta
from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.mlp import MLP
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import (
    RoPECache,
    apply_rope,
    apply_rope_partial,
    compute_longrope_attention_scaling,
    compute_longrope_inv_freq,
)
from coreai_models.primitives.ios.sdpa import SDPA


def _rotary_dim(config: Phi3Config, head_dim: int) -> int:
    return int(head_dim * getattr(config, "partial_rotary_factor", 1.0))


def _make_rope_cache(config: Phi3Config, head_dim: int, max_cache_size: int) -> RoPECache:
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
        designed_max = getattr(config, "_longrope_designed_max", config.max_position_embeddings)
        attention_scaling = compute_longrope_attention_scaling(designed_max, original_max)
        inv_freq = compute_longrope_inv_freq(rotary_dim, base, scaling["short_factor"])
        return RoPECache(
            rotary_dim,
            max_cache_size,
            base,
            inv_freq=inv_freq,
            attention_scaling=attention_scaling,
        )
    # Plain Phi-3: default rope over the (possibly partial) rotary dim.
    return RoPECache(rotary_dim, max_cache_size, base)


class Attention(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", None) or dim // n_heads
        self.rotary_dim = _rotary_dim(config, head_dim)

        self.q_proj = nn.Conv2d(dim, n_heads * head_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim, n_kv_heads * head_dim, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(n_heads * head_dim, dim, kernel_size=1, bias=False)

        self.sdpa = SDPA(head_dim=self.head_dim)

    def _rope(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        if self.rotary_dim < self.head_dim:
            return apply_rope_partial(x, rope_cos, rope_sin, self.rotary_dim)
        return apply_rope(x, rope_cos, rope_sin)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _, hidden_size = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        x = x.transpose(-3, -1)
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        query = (
            query.transpose(-3, -1)
            .reshape(batch_size, query_len, n_heads, self.head_dim)
            .transpose(-2, -3)
        )
        key = (
            key.transpose(-3, -1)
            .reshape(batch_size, query_len, n_kv_heads, self.head_dim)
            .transpose(-2, -3)
        )

        seq_len = rope_cos.shape[1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)

        query = self._rope(query, rope_cos, rope_sin)
        key = self._rope(key, rope_cos, rope_sin)

        query = (
            query.transpose(-2, -3)
            .reshape(batch_size, query_len, 1, n_heads * self.head_dim)
            .transpose(-3, -1)
        )
        key = (
            key.transpose(-3, -2)
            .reshape(batch_size, query_len, 1, n_kv_heads * self.head_dim)
            .transpose(-3, -1)
        )

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx,
                in_step,
                key,
                value,
                query_len,
            )

        output = self.sdpa(query, key, value, causal_mask)
        output = self.o_proj(output)
        return output.transpose(-3, -1)


class TransformerBlock(nn.Module):
    def __init__(self, config: Phi3Config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(dim=hidden_size, hidden_dim=config.intermediate_size)

        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(
            self.input_layernorm(x),
            rope_cos,
            rope_sin,
            in_step,
            causal_mask,
            cache,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Phi3Model(nn.Module):
    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            token_embeddings = layer(
                token_embeddings,
                rope_cos,
                rope_sin,
                in_step,
                causal_mask,
                cache,
            )
        return self.norm(token_embeddings)


class Phi3Extend(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.model = Phi3Model(config)
        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)

        self.prefill_mode = False

        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.lm_head = None

        self.kv_cache = KVCacheHandler(config.num_hidden_layers, config.hidden_size)

        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        self.rope = _make_rope_cache(config, head_dim, config.max_position_embeddings)

    def forward(
        self,
        transformer_input: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        embedding_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.kv_cache.register_kv_cache(key_cache, value_cache)
        rope_cos, rope_sin = self.rope.gather_cos_sin(position_ids)

        batch_size, seq_len, _, hidden_dim = transformer_input.shape
        out = self.model(
            transformer_input,
            rope_cos,
            rope_sin,
            in_step,
            causal_mask,
            self.kv_cache,
        )
        if self.prefill_mode:
            return self.kv_cache.k_cache[0, 0, 0, 0, 0] + self.kv_cache.v_cache[0, 0, 0, 0, 0]

        if self.lm_head is not None:
            return self.lm_head(out.transpose(-2, -3))

        if embedding_table.dtype == torch.int8:
            embedding_table = dequantize_per_tensor(
                embedding_table,
                self.emb_scale,
                self.emb_zero_point,
                out.dtype,
            )

        embedding_table = embedding_table.reshape(
            embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2]
        )

        out = out.transpose(-3, -1).reshape(batch_size, 1, hidden_dim, seq_len)
        return (embedding_table @ out).transpose(-2, -1)


class Phi3ForCausalLMForiOS(BaseForCausalLMForiOS):
    _HF_MODEL_CLASS = HFPhi3ForCausalLM

    @classmethod
    def _get_reauthored_config(cls, hf_config, max_context_length=None, num_layers=None):
        if not hasattr(hf_config, "_longrope_designed_max"):
            hf_config._longrope_designed_max = hf_config.max_position_embeddings
        return super()._get_reauthored_config(hf_config, max_context_length, num_layers=num_layers)

    def _init_model(self, config: Phi3Config) -> None:
        self.extend = Phi3Extend(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.gather_embeddings(input_ids, self.load_embeddings.embedding_table)
        return self.extend(
            token_embeddings,
            position_ids,
            in_step,
            causal_mask,
            key_cache,
            value_cache,
            self.load_embeddings.embedding_table,
        )

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        max_layer = -1
        for k in state_dict:
            name_split = k.split(".")
            if len(name_split) != 6:
                continue
            if not k.startswith("model.layers."):
                continue
            max_layer = max(max_layer, int(name_split[2]))

        if max_layer < 0:
            err = "invalid state_dict"
            raise ValueError(err)

        cfg = self.config
        n_heads = cfg.num_attention_heads
        n_kv = cfg.num_key_value_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
        inter = cfg.intermediate_size
        q_size, kv_size = n_heads * head_dim, n_kv * head_dim

        for i in range(max_layer + 1):
            # Split fused qkv_proj -> q/k/v, reshape for Conv2d.
            qkv = state_dict.pop(f"model.layers.{i}.self_attn.qkv_proj.weight")
            q = qkv[:q_size]
            k = qkv[q_size : q_size + kv_size]
            v = qkv[q_size + kv_size :]
            for proj, w in (("q_proj", q), ("k_proj", k), ("v_proj", v)):
                state_dict[f"model.layers.{i}.self_attn.{proj}.weight"] = (
                    w.unsqueeze(-1).unsqueeze(-1)
                )
            o_key = f"model.layers.{i}.self_attn.o_proj.weight"
            state_dict[o_key] = state_dict[o_key].unsqueeze(-1).unsqueeze(-1)

            # Split fused gate_up_proj -> gate/up, reshape for Conv2d.
            gu = state_dict.pop(f"model.layers.{i}.mlp.gate_up_proj.weight")
            gate = gu[:inter]
            up = gu[inter:]
            state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = gate.unsqueeze(-1).unsqueeze(-1)
            state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = up.unsqueeze(-1).unsqueeze(-1)
            dp_key = f"model.layers.{i}.mlp.down_proj.weight"
            state_dict[dp_key] = state_dict[dp_key].unsqueeze(-1).unsqueeze(-1)

        # Handle embeddings
        embedding_table = state_dict["model.embed_tokens.weight"].unsqueeze(1)
        if not self.disable_embedding_quantization:
            embedding_table, scale, zero_point = quantize_per_tensor(
                embedding_table, nbits=8, symmetric=True
            )
        else:
            scale = torch.tensor(1.0, dtype=embedding_table.dtype)
            zero_point = torch.tensor(0, dtype=torch.int8)

        state_dict["load_embeddings.embedding_table"] = embedding_table
        state_dict["gather_embeddings.scale"] = scale
        state_dict["gather_embeddings.zero_point"] = zero_point
        state_dict["extend.emb_scale"] = scale
        state_dict["extend.emb_zero_point"] = zero_point

        state_dict.pop("model.embed_tokens.weight")

        # Phi3Model is held inside Phi3Extend — add "extend." prefix
        new_state_dict = {}
        keys_to_pop = set()

        for k, _v in state_dict.items():
            if k.startswith("model.") and "gather_embeddings" not in k:
                new_key = f"extend.{k}"
                new_state_dict[new_key] = state_dict[k]
                keys_to_pop.add(k)

        for k in keys_to_pop:
            state_dict.pop(k)
        state_dict.update(new_state_dict)

        if not self.config.tie_word_embeddings:
            state_dict["extend.lm_head.weight"] = state_dict["lm_head.weight"]

        state_dict.pop("lm_head.weight", None)
