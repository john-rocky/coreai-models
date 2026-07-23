# Youtu-LLM-2B text decoder (HF model_type `youtu`, i.e. tencent/Youtu-LLM-2B)
# for the Core AI authoring path.
#
# Community port — NOT an Apple model. A DENSE Multi-head Latent Attention (MLA,
# DeepSeek-V2/V3 style) decoder: the same MLA that GLM-4.7-Flash (glm4_moe_lite)
# uses, but with a plain gated MLP on EVERY layer instead of a sparse MoE FFN,
# and a weight-tied lm_head. Authored decoupled from transformers (weights load
# straight from the HF safetensors, config is a local dataclass) so the export
# env needs no upstream `youtu` support.
#
# Architecture (config.json, tencent/Youtu-LLM-2B):
#   hidden 2048 · 32 layers · 16 heads (full MHA) · q_lora 1536 · kv_lora 512
#   qk_nope 128 + qk_rope 64 = qk_head_dim 192 · v_head_dim 128 (asymmetric,
#   DeepSeek-V2-Lite shape) · dense MLP intermediate 6144 (silu) · rope_theta
#   1.6e6, interleaved decoupled RoPE on the 64-dim slice · rms_norm_eps 1e-6 ·
#   vocab 128256 (Llama-3 tokenizer) · tie_word_embeddings=True.
#
# The MLA math + interleaved RoPE are bit-identical to glm4_moe_lite's (verified:
# HF `apply_rotary_pos_emb_interleave` == glm4_moe_lite `apple_rope_interleave`);
# only the FFN (dense here) and the tied head differ. The absorbed-MLA decode
# path (glm4_moe_lite_absorbed) + the MLA flash-decode Metal kernel
# (mla_metal_sdpa, which already bakes the kv_lora 512 / qk_rope 64 / scale
# 192**-0.5 DeepSeek-V2-Lite config) are reused unchanged — this module supplies
# the naive/materialized form used for the fp32 parity gate.
#
# Checkpoint mapping (tencent/Youtu-LLM-2B — authored names == HF names):
#   model.embed_tokens.weight                                              -> same
#   lm_head.weight (TIED to embed_tokens; may be absent in the ckpt)       -> tie
#   ...self_attn.{q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj}.weight -> same
#   ...self_attn.{q_a_layernorm,kv_a_layernorm}.weight                     -> same
#   ...mlp.{gate_proj,up_proj,down_proj}.weight                            -> same
#   ...{input,post_attention}_layernorm.weight                            -> same
#   model.norm.weight                                                      -> same
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA


@dataclass
class YoutuConfig:
    """Subset of Youtu-LLM config needed for authoring (values from config.json).

    Field names mirror the HF `YoutuConfig` (and glm4_moe_lite's) so the shared
    absorbed-MLA module / flash-decode kernel read them without adaptation."""

    hidden_size: int = 2048
    num_hidden_layers: int = 32
    vocab_size: int = 128256
    intermediate_size: int = 6144  # dense FFN width (every layer)
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    # MLA attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 16  # materialized full MHA (== num_attention_heads)
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    attention_bias: bool = False
    rope_theta: float = 1.6e6
    rope_interleave: bool = True
    max_position_embeddings: int = 131072

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim  # 192

    @property
    def head_dim(self) -> int:
        # RoPE dim == qk_rope_head_dim (HF sets config.head_dim = qk_rope_head_dim).
        return self.qk_rope_head_dim


# --------------------------------------------------------------------------- #
# Decoupled interleaved RoPE (manual, from precomputed cos/sin) — matches HF
# Youtu apply_rotary_pos_emb_interleave bit-for-bit (identical to glm4_moe_lite's
# apply_rope_interleave: de-interleave the rope slice, then rotate-half RoPE).
# --------------------------------------------------------------------------- #
def _deinterleave(x: torch.Tensor) -> torch.Tensor:
    """[..., a0,a1,a2,a3,...] -> [..., a0,a2,...,a1,a3,...] (even dims, then odd)."""
    return torch.cat([x[..., 0::2], x[..., 1::2]], dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope_interleave(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    """q: [b,H,s,64], k: [b,1,s,64]; cos,sin: [b,s,64]. Interleaved decoupled RoPE.

    De-interleave the rope slice, then standard rotate-half RoPE. cos/sin are
    computed in fp32 and cast to the query dtype (HF casts the rotary tables to
    the activation dtype)."""
    cos = cos.unsqueeze(1).to(q.dtype)  # [b,1,s,64]
    sin = sin.unsqueeze(1).to(q.dtype)
    q = _deinterleave(q)
    k = _deinterleave(k)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


# --------------------------------------------------------------------------- #
# MLA attention (naive / materialized): latent down/up projections -> per-head
# q/k/v -> decoupled RoPE on the 64-dim slice -> SDPA over qk=192 with v=128.
# Identical to glm4_moe_lite's Glm4MoeLiteMLA (the absorbed decode form lives in
# youtu_absorbed / glm4_moe_lite_absorbed).
# --------------------------------------------------------------------------- #
class YoutuMLA(nn.Module):
    def __init__(self, config: YoutuConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope = config.qk_nope_head_dim
        self.qk_rope = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim  # 192
        self.v_head_dim = config.v_head_dim  # 128
        bias = config.attention_bias

        self.q_a_proj = nn.Linear(d, self.q_lora_rank, bias=bias)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(d, self.kv_lora_rank + self.qk_rope, bias=bias)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, d, bias=bias)
        self.sdpa = SDPA(scale=self.qk_head_dim**-0.5, is_causal=True)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        H = self.num_heads

        # Query: down to q_lora, norm, up to per-head qk_head_dim, split nope/rope.
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(b, s, H, self.qk_head_dim).transpose(1, 2)  # [b,H,s,192]
        q_nope, q_rope = torch.split(q, [self.qk_nope, self.qk_rope], dim=-1)

        # Key/Value: shared compressed latent + a single decoupled rope head.
        compressed = self.kv_a_proj_with_mqa(x)  # [b,s,512+64]
        kv_c, k_rope = torch.split(compressed, [self.kv_lora_rank, self.qk_rope], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_c))
        kv = kv.view(b, s, H, self.qk_nope + self.v_head_dim).transpose(1, 2)  # [b,H,s,256]
        k_nope, value = torch.split(kv, [self.qk_nope, self.v_head_dim], dim=-1)
        k_rope = k_rope.view(b, 1, s, self.qk_rope)  # single shared rope head

        q_rope, k_rope = apply_rope_interleave(q_rope, k_rope, cos, sin)
        k_rope = k_rope.expand(b, H, s, self.qk_rope)  # broadcast to all heads

        query = torch.cat([q_nope, q_rope], dim=-1)  # [b,H,s,192]
        key = torch.cat([k_nope, k_rope], dim=-1)  # [b,H,s,192]

        out = self.sdpa(query, key, value)  # [b,H,s,128]
        out = out.transpose(1, 2).reshape(b, s, H * self.v_head_dim)  # [b,s,2048]
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Decoder layer — MLA + dense gated MLP (every layer).
# --------------------------------------------------------------------------- #
class YoutuDecoderLayer(nn.Module):
    def __init__(self, config: YoutuConfig, layer_idx: int) -> None:
        super().__init__()
        d = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = YoutuMLA(config)
        self.input_layernorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.mlp = MLP(d, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache=None,
        offset: int = 0,
        seq_len: int | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        normed = self.input_layernorm(x)
        # Prefill (no cache) -> naive YoutuMLA(x,cos,sin); stateful decode -> the
        # absorbed attention (swapped in by youtu_absorbed) with the latent cache.
        if kv_cache is None:
            r = self.self_attn(normed, cos, sin)
        else:
            r = self.self_attn(normed, cos, sin, kv_cache, offset, seq_len, layer_idx)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))


# --------------------------------------------------------------------------- #
# Decoder stack (prefill; the fp32 parity reference).
# --------------------------------------------------------------------------- #
class YoutuModel(nn.Module):
    def __init__(self, config: YoutuConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [YoutuDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rd = config.qk_rope_head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_buffers(self, device: str = "cpu") -> None:
        rd = self.config.qk_rope_head_dim
        self.inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, rd, 2, device=device).float() / rd)
        )

    def rope_cos_sin(self, position_ids: torch.Tensor):
        # position_ids [b,s] -> cos/sin [b,s,qk_rope_head_dim].
        freqs = position_ids[..., None].float() * self.inv_freq  # [b,s,rd/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [b,s,rd]
        return emb.cos(), emb.sin()

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Prefill (no KV cache; causal SDPA over the query block)."""
        h = self.embed_tokens(input_ids)
        cos, sin = self.rope_cos_sin(position_ids)
        for layer in self.layers:
            h = layer(h, cos, sin)
        return self.norm(h)

    def forward_stateful(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor, kv_cache
    ) -> torch.Tensor:
        return self.forward_stateful_core(
            self.embed_tokens(input_ids), position_ids, kv_cache
        )

    def forward_stateful_core(
        self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, kv_cache
    ) -> torch.Tensor:
        """Stateful prefill/decode on pre-computed embeddings. ``inputs_embeds``
        carries the query tokens ([b, query_len]); ``position_ids`` carries the
        full positions ([b, seq_len]) so offset = seq_len - query_len."""
        query_len = inputs_embeds.shape[1]
        seq_len = position_ids.shape[1]
        offset = seq_len - query_len
        q_pos = position_ids.narrow(1, offset, query_len)
        cos, sin = self.rope_cos_sin(q_pos)
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, kv_cache=kv_cache, offset=offset,
                      seq_len=seq_len, layer_idx=i)
        return self.norm(h)


class YoutuStatefulForCausalLM(nn.Module):
    """Prefill text decoder (the fp32 parity gate reference). forward(input_ids
    [b,s], position_ids [b,s]) -> logits [b,s,vocab]. Decode/export uses the
    absorbed-MLA form (youtu_absorbed)."""

    def __init__(self, config: YoutuConfig) -> None:
        super().__init__()
        self.config = config
        self.model = YoutuModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.last_token_only = False

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        h = self.model(input_ids, position_ids)
        if self.last_token_only:
            h = h[:, -1:, :]
        return self.lm_head(h)


# --------------------------------------------------------------------------- #
# Config + weight loading (safetensors-direct; no transformers `youtu`).
# --------------------------------------------------------------------------- #
def youtu_config_from_dict(raw: dict) -> YoutuConfig:
    rope = raw.get("rope_parameters") or {}
    return YoutuConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        vocab_size=raw["vocab_size"],
        intermediate_size=raw["intermediate_size"],
        rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", True)),
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
        q_lora_rank=raw["q_lora_rank"],
        kv_lora_rank=raw["kv_lora_rank"],
        qk_nope_head_dim=raw["qk_nope_head_dim"],
        qk_rope_head_dim=raw["qk_rope_head_dim"],
        v_head_dim=raw["v_head_dim"],
        attention_bias=bool(raw.get("attention_bias", False)),
        rope_theta=rope.get("rope_theta", raw.get("rope_theta", 1.6e6)),
        rope_interleave=bool(raw.get("rope_interleave", True)),
        max_position_embeddings=raw.get("max_position_embeddings", 131072),
    )


def youtu_from_hf(
    huggingface_model_id: str,
    target_dtype: torch.dtype = torch.float16,
) -> YoutuStatefulForCausalLM:
    """Load Youtu-LLM from the HF checkpoint into the authored module. Authored
    module names == HF names, so the state dict loads directly; the tied lm_head
    is re-pointed at embed_tokens (the ckpt may omit lm_head.weight)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    model_dir = snapshot_download(
        huggingface_model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )
    with open(os.path.join(model_dir, "config.json")) as f:
        config = youtu_config_from_dict(json.load(f))

    with torch.device("meta"):
        model = YoutuStatefulForCausalLM(config)
    model.to(dtype=target_dtype)

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    sd: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                t = f.get_tensor(key)
                if t.dtype != target_dtype and t.is_floating_point():
                    t = t.to(target_dtype)
                sd[key] = t

    if config.tie_word_embeddings and "lm_head.weight" not in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]

    model.load_state_dict(sd, assign=True, strict=False)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.reset_buffers()

    meta_params = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_params:
        raise RuntimeError(f"Parameters not loaded: {meta_params[:8]} ... ({len(meta_params)} total)")
    model.eval()
    return model
