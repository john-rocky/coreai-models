# Chatterbox T3 (Resemble AI) — the AR speech-token model, Core AI authoring path.
#
# Community port — NOT an Apple model. T3 is a plain dense Llama backbone
# (`Llama_520M`: 30 layers · hidden 1024 · 16 heads · head_dim 64 · intermediate
# 4096 · SwiGLU · RMSNorm · **llama3-scaled RoPE** θ=500k factor 8) driven by
# CUSTOM input embeddings — the host assembles `[cond prefix | text_emb+text_pos |
# speech_emb+speech_pos]` (see chatterbox `prepare_input_embeds`) and feeds the
# result as `inputs_embeds`; the graph runs the transformer + a speech head over
# the 8194-token speech vocab. So this overlay is `llama.py` MINUS the token
# embedding (embeds-in) and with `lm_head -> speech_head`. The learned absolute
# position embeddings are added host-side; the backbone still applies its own
# llama3 RoPE (both are load-bearing — verified vs the fp32 chatterbox T3).
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA


@dataclass
class ChatterboxT3Config:
    hidden_size: int = 1024
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 64
    intermediate_size: int = 4096
    rms_norm_eps: float = 1e-5
    speech_vocab_size: int = 8194
    max_position_embeddings: int = 4096  # for KVCache.create_cache_tensors (max speech tokens)
    vocab_size: int = 8194  # metadata alias (== speech_vocab_size)
    # llama3-scaled RoPE (from the Llama_520M tfmr config)
    rope_theta: float = 500000.0
    rope_factor: float = 8.0
    rope_low_freq_factor: float = 1.0
    rope_high_freq_factor: float = 4.0
    rope_original_max_position_embeddings: int = 8192

    @property
    def rope_scaling(self) -> dict:
        return {
            "rope_type": "llama3",
            "factor": self.rope_factor,
            "low_freq_factor": self.rope_low_freq_factor,
            "high_freq_factor": self.rope_high_freq_factor,
            "original_max_position_embeddings": self.rope_original_max_position_embeddings,
        }


class T3Attention(nn.Module):
    """Plain-Llama attention (fused QKV, no bias, no q/k-norm) with llama3 RoPE."""

    def __init__(self, config: ChatterboxT3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.qkv_proj = nn.Linear(
            dim, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.sdpa = SDPA(is_causal=True)
        self.rope = initialize_rope(
            dims=self.head_dim, base=config.rope_theta, scaling_config=config.rope_scaling
        )

    def forward(self, x: torch.Tensor, position_ids: torch.IntTensor,
                cache: KVCache | None = None) -> torch.Tensor:
        b, q, _ = x.shape
        nh, nkv = self.n_heads, self.n_kv_heads
        qkv = self.qkv_proj(x).reshape(b, q, nh + 2 * nkv, self.head_dim).permute(0, 2, 1, 3)
        query_key = qkv.narrow(1, 0, nh + nkv)
        value = qkv.narrow(1, nh + nkv, nkv)
        seq_len = position_ids.shape[-1]
        torch._check_is_size(q)
        torch._check_is_size(seq_len)
        offset = seq_len - q
        torch._check_is_size(offset)
        query_key = self.rope(query_key, position_ids=position_ids.narrow(-1, offset, q))
        query = query_key.narrow(1, 0, nh)
        key = query_key.narrow(1, nh, nkv)
        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx, offset, key, value, seq_len=seq_len, query_len=q
            )
        out = self.sdpa(query, key, value).permute(0, 2, 1, 3).reshape(b, q, nh * self.head_dim)
        return self.o_proj(out)


class T3Block(nn.Module):
    def __init__(self, config: ChatterboxT3Config, layer_idx: int) -> None:
        super().__init__()
        d = config.hidden_size
        self.self_attn = T3Attention(config, layer_idx)
        self.mlp = MLP(d, config.intermediate_size)
        self.input_layernorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(d, eps=config.rms_norm_eps)

    def forward(self, x, position_ids, cache=None):
        h = x + self.self_attn(self.input_layernorm(x), position_ids, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class T3Transformer(nn.Module):
    """The tfmr backbone — embeds-in (no token embedding; the host assembles embeds)."""

    def __init__(self, config: ChatterboxT3Config) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [T3Block(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, inputs_embeds, position_ids, cache=None):
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class ChatterboxT3(nn.Module):
    """forward(inputs_embeds [b,seq,1024], position_ids [b,seq], k_cache, v_cache)
    -> speech logits [b,seq,8194]. Prefill uses cache=None."""

    def __init__(self, config: ChatterboxT3Config) -> None:
        super().__init__()
        self.config = config
        self.model = T3Transformer(config)
        self.speech_head = nn.Linear(config.hidden_size, config.speech_vocab_size, bias=True)

    def forward(self, inputs_embeds, position_ids, k_cache=None, v_cache=None):
        cache = KVCache(k_cache, v_cache) if k_cache is not None else None
        h = self.model(inputs_embeds, position_ids, cache)
        return self.speech_head(h)


def _load_t3_state(snapshot_dir: str, target_dtype: torch.dtype):
    """Read tfmr.* + speech_head.* from the chatterbox t3 safetensors, fuse q/k/v."""
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(snapshot_dir, "t3_cfg.safetensors")))
    if not files:
        files = sorted(glob.glob(os.path.join(snapshot_dir, "t3*.safetensors")))
    raw: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():  # noqa: SIM118
                if k.startswith("tfmr.") or k.startswith("speech_head."):
                    raw[k] = f.get_tensor(k)
        break
    sd: dict[str, torch.Tensor] = {}
    # transformer layers: tfmr.layers.N.* -> model.layers.N.* with fused qkv
    import re
    n_layers = 1 + max(
        (int(m.group(1)) for k in raw if (m := re.match(r"tfmr\.layers\.(\d+)\.", k))), default=-1
    )
    for i in range(n_layers):
        q = raw.pop(f"tfmr.layers.{i}.self_attn.q_proj.weight")
        k_ = raw.pop(f"tfmr.layers.{i}.self_attn.k_proj.weight")
        v = raw.pop(f"tfmr.layers.{i}.self_attn.v_proj.weight")
        sd[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.cat([q, k_, v], dim=0)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = raw.pop(
            f"tfmr.layers.{i}.self_attn.o_proj.weight"
        )
        for proj in ("gate_proj", "up_proj", "down_proj"):
            sd[f"model.layers.{i}.mlp.{proj}.weight"] = raw.pop(
                f"tfmr.layers.{i}.mlp.{proj}.weight"
            )
        sd[f"model.layers.{i}.input_layernorm.weight"] = raw.pop(
            f"tfmr.layers.{i}.input_layernorm.weight"
        )
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = raw.pop(
            f"tfmr.layers.{i}.post_attention_layernorm.weight"
        )
    sd["model.norm.weight"] = raw.pop("tfmr.norm.weight")
    sd["speech_head.weight"] = raw.pop("speech_head.weight")
    if "speech_head.bias" in raw:
        sd["speech_head.bias"] = raw.pop("speech_head.bias")
    return sd, n_layers


def chatterbox_t3_from_pretrained(
    snapshot_dir: str, target_dtype: torch.dtype = torch.float32
) -> ChatterboxT3:
    sd, n_layers = _load_t3_state(snapshot_dir, target_dtype)
    config = ChatterboxT3Config(num_hidden_layers=n_layers)
    model = ChatterboxT3(config)
    has_bias = "speech_head.bias" in sd
    if not has_bias:
        model.speech_head = nn.Linear(config.hidden_size, config.speech_vocab_size, bias=False)
    model = model.to(dtype=target_dtype)
    sd = {k: v.to(target_dtype) if v.is_floating_point() else v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model
