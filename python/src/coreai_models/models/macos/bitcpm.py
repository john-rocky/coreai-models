# Community port — NOT an Apple model.
"""BitCPM-CANN-8B (MiniCPM4-8B arch, 1.58-bit ternary) decode graph for the Core AI engine.

Dense GQA + RMSNorm + SwiGLU + LongRoPE decoder with MiniCPM **mup** scalars (embed x12,
residual x scale_depth/sqrt(L), logits / (hidden/dim_model_base)) and the 7 per-layer linears
(q/k/v/o, gate/up/down) on the fused 2-bit ternary matvec kernel (``bitcpm_ternary_metal``).
Embedding (Q4_K) + untied head (Q6_K) stay higher-precision fp16. Mirrors ``llama.py``'s KV-only
decode contract — ``forward(input_ids, position_ids, k_cache, v_cache) -> logits`` — so it exports
through the standard pure-attention pipelined path; the one delta is baked LongRoPE (the RoPE
primitive rejects non-default scaling; VoxCPM proved baked rope exports for minicpm4) applied by
hand before the KV write.

Weights load from the TQ2_0 gguf (``gguf.dequantize`` -> fp; the ternary linears re-derive
{-1,0,1}+per-256-block scale exactly). mup scalars are read from the gguf metadata.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.models.macos.bitcpm_ternary_metal import (
    MetalTernaryLinear,
    build_tern_kernel,
)

ROPE_THETA = 10000.0


class BitCPMConfig:
    """Minimal config carrying the fields KVCache.create_cache_tensors + the graph need."""

    model_type = "minicpm"

    def __init__(self, *, hidden_size=4096, intermediate_size=16384, num_hidden_layers=32,
                 num_attention_heads=32, num_key_value_heads=2, head_dim=128, vocab_size=73448,
                 rms_norm_eps=1e-6, max_position_embeddings=32768, scale_emb=12.0,
                 residual_scale=0.2474873661994934, logit_div=16.0):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.scale_emb = scale_emb
        self.residual_scale = residual_scale
        self.logit_div = logit_div


def _rope_tables(short_factor: torch.Tensor, head_dim: int, buf: int, dtype=torch.float32):
    """Baked LongRoPE cos/sin [buf, head_dim] (scaling_factor==1 since orig==max==32768)."""
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, head_dim, 2).float() / head_dim))  # [hd/2]
    eff = inv / short_factor.float()
    t = torch.arange(buf).float()
    fr = torch.outer(t, eff)
    emb = torch.cat((fr, fr), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        f = x.float()
        f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + self.eps)
        return f.to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, cfg: BitCPMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)
        self.sdpa = SDPA(is_causal=True)

    def forward(self, x, position_ids, cos, sin, cache: KVCache):
        b, q, _ = x.shape
        nh, nkv, hd = self.nh, self.nkv, self.hd
        query = self.q_proj(x).reshape(b, q, nh, hd).permute(0, 2, 1, 3)
        key = self.k_proj(x).reshape(b, q, nkv, hd).permute(0, 2, 1, 3)
        value = self.v_proj(x).reshape(b, q, nkv, hd).permute(0, 2, 1, 3)

        seq_len = position_ids.shape[-1]
        offset = seq_len - q
        rpos = position_ids.narrow(-1, offset, q).reshape(-1)
        c = cos.index_select(0, rpos).reshape(1, 1, q, hd)
        s = sin.index_select(0, rpos).reshape(1, 1, q, hd)
        qf, kf = query.float(), key.float()
        query = (qf * c + _rotate_half(qf) * s).to(x.dtype)
        key = (kf * c + _rotate_half(kf) * s).to(x.dtype)

        key, value = cache.update_and_fetch(self.layer_idx, offset, key, value,
                                            seq_len=seq_len, query_len=q)
        out = self.sdpa(query, key, value).permute(0, 2, 1, 3).reshape(b, q, nh * hd)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: BitCPMConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rs = cfg.residual_scale

    def forward(self, x, position_ids, cos, sin, cache):
        r = self.self_attn(self.input_layernorm(x), position_ids, cos, sin, cache)
        h = x + r * self.rs                                   # mup residual scale
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r * self.rs


class BitCPM8B(nn.Module):
    def __init__(self, cfg: BitCPMConfig):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Block(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        cos, sin = _rope_tables(torch.ones(cfg.head_dim // 2), cfg.head_dim,
                                cfg.max_position_embeddings, dtype=torch.float16)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)
        self.scale_emb = cfg.scale_emb
        self.logit_div = cfg.logit_div

    def forward(self, input_ids, position_ids, k_cache, v_cache):
        cache = KVCache(k_cache, v_cache)
        x = self.embed_tokens(input_ids) * self.scale_emb
        for layer in self.layers:
            x = layer(x, position_ids, self.cos_table, self.sin_table, cache)
        x = self.norm(x)
        return self.lm_head(x / self.logit_div)


# --------------------------------------------------------------------------- #
# gguf loader: dequant -> fp16 module, then swap the 7 linears to ternary kernel
# --------------------------------------------------------------------------- #
def load_bitcpm8b_from_gguf(gguf_path: str, *, num_layers: int | None = None,
                            dtype=torch.float16):
    """Build BitCPM8B with ternary linears from the TQ2_0 gguf. Returns (model, tern_kernel)."""
    import gguf
    from gguf.quants import dequantize

    r = gguf.GGUFReader(gguf_path)
    ten = {t.name: t for t in r.tensors}

    def deq(name):
        t = ten[name]
        return torch.from_numpy(np.asarray(dequantize(t.data, t.tensor_type)).copy())

    def meta(k, default):
        f = r.get_field(k)
        try:
            return f.contents() if f else default
        except Exception:
            return default

    nl = num_layers or int(meta("minicpm.block_count", 32))
    cfg = BitCPMConfig(
        num_hidden_layers=nl,
        scale_emb=float(meta("minicpm.embedding_scale", 12.0)),
        residual_scale=float(meta("minicpm.residual_scale", 0.2474873661994934)),
        logit_div=float(meta("minicpm.logit_scale", 16.0)),
    )
    kernel = build_tern_kernel()
    model = BitCPM8B(cfg).to(dtype).eval()

    with torch.no_grad():
        model.embed_tokens.weight.copy_(deq("token_embd.weight").to(dtype))
        model.norm.weight.copy_(deq("output_norm.weight").to(dtype))
        model.lm_head.weight.copy_(deq("output.weight").to(dtype))
        # baked rope from the real short_factor
        sf = deq("rope_factors_short.weight")
        cos, sin = _rope_tables(sf, cfg.head_dim, cfg.max_position_embeddings, dtype=dtype)
        model.cos_table, model.sin_table = cos, sin
        for i in range(nl):
            blk, p = model.layers[i], f"blk.{i}."
            blk.input_layernorm.weight.copy_(deq(p + "attn_norm.weight").to(dtype))
            blk.post_attention_layernorm.weight.copy_(deq(p + "ffn_norm.weight").to(dtype))
            a, m = blk.self_attn, blk.mlp
            a.q_proj = MetalTernaryLinear(deq(p + "attn_q.weight"), kernel)
            a.k_proj = MetalTernaryLinear(deq(p + "attn_k.weight"), kernel)
            a.v_proj = MetalTernaryLinear(deq(p + "attn_v.weight"), kernel)
            a.o_proj = MetalTernaryLinear(deq(p + "attn_output.weight"), kernel)
            m.gate_proj = MetalTernaryLinear(deq(p + "ffn_gate.weight"), kernel)
            m.up_proj = MetalTernaryLinear(deq(p + "ffn_up.weight"), kernel)
            m.down_proj = MetalTernaryLinear(deq(p + "ffn_down.weight"), kernel)
    return model, kernel
