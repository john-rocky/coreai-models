# Community port — NOT an Apple model.
"""BitVLA language model (BitNet b1.58 2B4T, 1.58-bit ternary) decode graph for the Core AI engine.

The autoregressive action-token generator of BitVLA (lxsy/bitvla-bf16). Dense GQA + RMSNorm +
ReLU2 FFN + **SubLN** (attn_sub_norm before o_proj, ffn_sub_norm before down_proj) + RoPE theta
500000, untied lm_head, NO mup scalars. The 7 per-layer linears (q/k/v/o, gate/up/down) run the
generalized fused 2-bit ternary matvec (``bitnet_ternary_metal.BitLinearMetal`` = W1.58 weight +
A8 per-token int8 activation, arbitrary K%16, M=1 decode). embed_tokens is HOST-side (text-token
lookup + spliced 256 vision embeds) so the graph takes **inputs_embeds**, letting us inject the
projected image features. lm_head stays fp16.

Mirrors BitCPM's KV-only decode contract but with inputs_embeds:
``forward(inputs_embeds, position_ids, k_cache, v_cache) -> logits``. Baked RoPE tables (plain
theta-500000 rope; baking keeps the graph primitive-clean, like BitCPM/VoxCPM). M=1 decode-only
(static-ids S=1 export); prefill = loop the graph one position at a time (image embeds then text).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.models.macos.bitnet_ternary_metal import (
    BitLinearMetal,
    build_tern_kernel,
)

ROPE_THETA = 500000.0


# Action tokens occupy the vocab tail [ACT_LO, vocab) = 256 rows; the action model only ever emits
# these (fixed 7-token generation), so the lm_head can be sliced to them -> 656MB(fp16) -> 1.3MB,
# and decode argmax is constrained to valid action bins. token_id = ACT_LO + argmax(logits[256]).
ACT_LO = 128012
N_ACTION_BINS = 256


class BitVLALLMConfig:
    model_type = "bitvla_llm"

    def __init__(self, *, hidden_size=2560, intermediate_size=6912, num_hidden_layers=30,
                 num_attention_heads=20, num_key_value_heads=5, head_dim=128, vocab_size=128268,
                 rms_norm_eps=1e-5, max_position_embeddings=4096, action_head=False):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.action_head = action_head        # slice lm_head to the 256 action rows


def _rope_tables(head_dim, buf, dtype=torch.float32):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, head_dim, 2).float() / head_dim))
    fr = torch.outer(torch.arange(buf).float(), inv)
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
        return (f * self.weight).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: BitVLALLMConfig, layer_idx: int, kernel):
        super().__init__()
        self.layer_idx = layer_idx
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        h = cfg.hidden_size
        self.q_proj = BitLinearMetal(torch.zeros(self.nh * self.hd, h), kernel)
        self.k_proj = BitLinearMetal(torch.zeros(self.nkv * self.hd, h), kernel)
        self.v_proj = BitLinearMetal(torch.zeros(self.nkv * self.hd, h), kernel)
        self.o_proj = BitLinearMetal(torch.zeros(h, self.nh * self.hd), kernel)
        self.attn_sub_norm = RMSNorm(self.nh * self.hd, cfg.rms_norm_eps)   # SubLN before o_proj
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
        return self.o_proj(self.attn_sub_norm(out))                        # SubLN -> o_proj


class MLP(nn.Module):
    def __init__(self, h, inter, eps, kernel):
        super().__init__()
        self.gate_proj = BitLinearMetal(torch.zeros(inter, h), kernel)
        self.up_proj = BitLinearMetal(torch.zeros(inter, h), kernel)
        self.down_proj = BitLinearMetal(torch.zeros(h, inter), kernel)
        self.ffn_sub_norm = RMSNorm(inter, eps)                            # SubLN before down_proj

    def forward(self, x):
        h = F.relu(self.gate_proj(x)) ** 2 * self.up_proj(x)               # ReLU2
        return self.down_proj(self.ffn_sub_norm(h))


class Block(nn.Module):
    def __init__(self, cfg: BitVLALLMConfig, layer_idx: int, kernel):
        super().__init__()
        self.self_attn = Attention(cfg, layer_idx, kernel)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size, cfg.rms_norm_eps, kernel)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, position_ids, cos, sin, cache):
        x = x + self.self_attn(self.input_layernorm(x), position_ids, cos, sin, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class BitVLALLM(nn.Module):
    """inputs_embeds [1,1,H] decode graph -> logits [1,1,vocab]. embed lookup is host-side."""

    def __init__(self, cfg: BitVLALLMConfig, kernel):
        super().__init__()
        self.config = cfg
        self.layers = nn.ModuleList([Block(cfg, i, kernel) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        head_out = N_ACTION_BINS if cfg.action_head else cfg.vocab_size
        self.lm_head = nn.Linear(cfg.hidden_size, head_out, bias=False)
        cos, sin = _rope_tables(cfg.head_dim, cfg.max_position_embeddings, dtype=torch.float16)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def forward(self, inputs_embeds, position_ids, k_cache, v_cache):
        cache = KVCache(k_cache, v_cache)
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, position_ids, self.cos_table, self.sin_table, cache)
        x = self.norm(x)
        return self.lm_head(x)


def load_bitvla_llm(safetensors_path: str, *, num_layers: int | None = None, dtype=torch.float16,
                    action_head: bool = False):
    """Build BitVLALLM with ternary linears from the bf16 master. Returns (model, kernel, embed_w).
    action_head=True slices lm_head to the 256 action rows [ACT_LO:vocab]."""
    from safetensors.torch import load_file
    sd = load_file(safetensors_path)
    p = "language_model.model."
    nl = num_layers or 30
    cfg = BitVLALLMConfig(num_hidden_layers=nl, action_head=action_head)
    kernel = build_tern_kernel()
    model = BitVLALLM(cfg, kernel).to(dtype).eval()

    with torch.no_grad():
        model.norm.weight.copy_(sd[p + "norm.weight"].to(dtype))
        head_w = sd["language_model.lm_head.weight"]
        if action_head:
            head_w = head_w[ACT_LO:ACT_LO + N_ACTION_BINS]
        model.lm_head.weight.copy_(head_w.to(dtype))
        for i in range(nl):
            blk, s = model.layers[i], f"{p}layers.{i}."
            blk.input_layernorm.weight.copy_(sd[s + "input_layernorm.weight"].to(dtype))
            blk.post_attention_layernorm.weight.copy_(sd[s + "post_attention_layernorm.weight"].to(dtype))
            blk.self_attn.attn_sub_norm.weight.copy_(sd[s + "self_attn.attn_sub_norm.weight"].to(dtype))
            blk.mlp.ffn_sub_norm.weight.copy_(sd[s + "mlp.ffn_sub_norm.weight"].to(dtype))
            a, m = blk.self_attn, blk.mlp
            a.q_proj = BitLinearMetal(sd[s + "self_attn.q_proj.weight"], kernel)
            a.k_proj = BitLinearMetal(sd[s + "self_attn.k_proj.weight"], kernel)
            a.v_proj = BitLinearMetal(sd[s + "self_attn.v_proj.weight"], kernel)
            a.o_proj = BitLinearMetal(sd[s + "self_attn.o_proj.weight"], kernel)
            m.gate_proj = BitLinearMetal(sd[s + "mlp.gate_proj.weight"], kernel)
            m.up_proj = BitLinearMetal(sd[s + "mlp.up_proj.weight"], kernel)
            m.down_proj = BitLinearMetal(sd[s + "mlp.down_proj.weight"], kernel)
    embed_w = sd[p + "embed_tokens.weight"].to(dtype)                      # host-side embed lookup
    return model, kernel, embed_w
