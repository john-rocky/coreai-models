# ZAYA1-8B text decoder (HF model_type `zaya`, Zyphra/ZAYA1-8B) for the Core AI
# authoring path.
#
# Community port — NOT an Apple model. The zoo's first model with Compressed
# Convolutional Attention (CCA) and the zoo's first Exponential-Depth-Averaging
# (EDA) MoE router. Authored to be exportable via
# coreai_models.export.macos.export_to_coreai; decoupled from transformers
# (weights load straight from the HF safetensors, config is a local dataclass)
# so the export env needs no upstream `zaya` support.
#
# Architecture (80 layers, STRICTLY ALTERNATING):
#   even layers (0,2,..,78) = ZayaDecoderATTLayer  (CCA attention)
#   odd  layers (1,3,..,79) = ZayaDecoderMLPLayer  (top-1/16 MoE + EDA router)
# 8.4B total / 760M active (1 expert per MoE layer).
#
# Three novel pieces vs the existing zoo decoders:
#
#  1. CCA attention (every even layer). GQA (8 q heads / 2 kv heads, head_dim
#     128) with FOUR extra twists vs a plain attention block:
#       a. a depthwise (k=2) then grouped (k=2, groups=10) CAUSAL Conv1d
#          "preconditioner" on the packed [q;k] channels (1280 = 1024+256),
#          left-padded by 2; its output is added to a residual qk-mean stream.
#       b. QK L2-normalisation per head, scaled by sqrt(head_dim), with a learned
#          per-kv-head temperature on the key.
#       c. a time-shifted value: value = concat(val_proj1(h), val_proj2(h_{t-1})).
#       d. partial RoPE (factor 0.5 -> rotary_dim 64), standard rotate_half.
#     Decode time-state beyond the KV cache: a kernel-2 conv_state (the previous
#     packed-qk column) and a 1-step prev_hs (the previous hidden for v2) — the
#     lfm2.py conv-state "loop-free step" pattern. The KV cache stores the
#     post-conv per-head K/V (full GQA size).
#
#  2. EDA MoE router. down_proj(2048->256) -> (+ prev MoE layer's router state *
#     learned scale, the "Exponential Depth Averaging" depth stream — threaded
#     layer->layer, NOT a time cache) -> RMSNorm -> 3-layer GELU MLP -> softmax
#     -> top-1 with selection-only balancing biases. MoD: a 17th "skip" expert
#     (identity pass-through). At top-1 this is a static per-token select, NOT
#     dynamic depth (the layer-skip code path is gated on top-k>1).
#
#  3. Learned residual scaling. Each layer applies a per-channel (scale, bias) to
#     the running residual and to the sub-layer output before merging; the
#     residual accumulator is kept in fp32.
#
# Checkpoint mapping (Zyphra original layout — experts UNPACKED, one tensor per
# expert; gate+up are PACKED inside linear_fc1, swiglu splits them):
#   model.embed_tokens.weight                              -> same (TIED to head)
#   model.final_norm.weight                                -> same
#   model.res_scale.{hidden_states,residual}_{scale,bias}  -> final res_scale
#   att layers (even N):
#     ...input_norm.weight                                 -> same
#     ...res_scale.* (layer 0: hidden_* only)              -> same
#     ...self_attn.qkv.{linear_q,linear_k,val_proj1,val_proj2}.weight -> same
#     ...self_attn.qkv.conv_qk.{0,1}.{weight,bias}         -> same
#     ...self_attn.qkv.temp                                -> same ([n_kv_heads])
#     ...self_attn.o_proj.weight                           -> same
#   moe layers (odd N):
#     ...input_norm.weight / ...res_scale.*                -> same
#     ...zaya_block.router.down_proj.{weight,bias}         -> same
#     ...zaya_block.router.rmsnorm_eda.weight              -> same
#     ...zaya_block.router.router_mlp.{0,2,4}.{weight,bias}-> same
#     ...zaya_block.router.balancing_biases   [E+1]        -> same (buffer)
#     ...zaya_block.router.router_states_scale [256]       -> same (absent on 1st MoE layer)
#     ...zaya_block.experts.local_experts.{e}.linear_fc1.weight [2*I,d]
#         -> split rows -> switch_mlp.gate_proj/up_proj.weight [1,E,I,d]
#     ...zaya_block.experts.local_experts.{e}.linear_fc2.weight [d,I]
#         -> stack_e -> switch_mlp.down_proj.weight [1,E,d,I]
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU


@dataclass
class ZayaConfig:
    """Subset of the ZAYA1-8B config needed for authoring (from config.json)."""

    hidden_size: int = 2048
    num_hidden_layers: int = 80
    vocab_size: int = 262272
    ffn_hidden_size: int = 4096  # gated -> moe_intermediate = ffn_hidden_size // 2
    norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = True
    # CCA attention
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 128
    cca_time0: int = 2  # depthwise conv kernel
    cca_time1: int = 2  # grouped conv kernel
    attention_bias: bool = False
    rope_theta: float = 5_000_000.0
    partial_rotary_factor: float = 0.5
    # MoE
    num_experts: int = 16
    moe_router_topk: int = 1
    zaya_mlp_expansion: int = 256  # router hidden dim
    zaya_use_mod: bool = True      # adds a 17th "skip" expert
    zaya_use_eda: bool = True
    scale_residual_merge: bool = True
    residual_in_fp32: bool = True
    max_position_embeddings: int = 131072

    @property
    def latent_q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim  # 1024

    @property
    def latent_k_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim  # 256

    @property
    def gqa_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads  # 4

    @property
    def moe_intermediate_size(self) -> int:
        return self.ffn_hidden_size // 2  # 2048

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)  # 64

    @property
    def conv_in_out_ch(self) -> int:
        return self.latent_q_dim + self.latent_k_dim  # 1280

    @property
    def conv_state_width(self) -> int:
        # kernel-1 columns of the packed-qk stream the two k=2 convs need.
        return (self.cca_time0 - 1) + (self.cca_time1 - 1)  # 2

    @property
    def router_num_experts(self) -> int:
        return self.num_experts + (1 if self.zaya_use_mod else 0)  # 17

    @property
    def num_att_layers(self) -> int:
        return sum(1 for i in range(self.num_hidden_layers) if i % 2 == 0)

    @property
    def num_moe_layers(self) -> int:
        return self.num_hidden_layers - self.num_att_layers

    def is_moe(self, layer_idx: int) -> bool:
        return layer_idx % 2 == 1


# --------------------------------------------------------------------------- #
# Partial RoPE (manual, from precomputed cos/sin) — standard rotate_half on the
# first rotary_dim dims, matches HF modeling_zaya apply_rotary_pos_emb.
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    """q: [b,Hq,s,D], k: [b,Hk,s,D]; cos,sin: [b,s,rotary_dim] (rotary_dim<=D).

    Rotate only the first rotary_dim dims, pass the rest through. cos/sin are
    computed fp32 and cast to the query dtype (HF casts the rotary tables to the
    activation dtype)."""
    rd = cos.shape[-1]
    cos = cos.unsqueeze(1).to(q.dtype)  # [b,1,s,rd]
    sin = sin.unsqueeze(1).to(q.dtype)
    q_rot, q_pass = q[..., :rd], q[..., rd:]
    k_rot, k_pass = k[..., :rd], k[..., rd:]
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


# --------------------------------------------------------------------------- #
# CCA: produces per-head query/key/value from hidden_states. Conv preconditioner
# + qk-mean residual + QK L2-norm + learned temp + time-shifted value.
# --------------------------------------------------------------------------- #
class ZayaCCA(nn.Module):
    def __init__(self, config: ZayaConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.nq = config.num_attention_heads
        self.nkv = config.num_key_value_heads
        self.hd = config.head_dim
        self.groups = config.gqa_groups
        self.lq = config.latent_q_dim
        self.lk = config.latent_k_dim
        self.sqrt_hd = float(self.hd ** 0.5)
        self.pad0 = config.cca_time0 - 1
        self.pad1 = config.cca_time1 - 1
        self.total_pad = self.pad0 + self.pad1
        bias = config.attention_bias

        self.linear_q = nn.Linear(d, self.lq, bias=bias)
        self.linear_k = nn.Linear(d, self.lk, bias=bias)
        self.val_proj1 = nn.Linear(d, self.lk // 2, bias=bias)
        self.val_proj2 = nn.Linear(d, self.lk // 2, bias=bias)
        ch = config.conv_in_out_ch
        self.conv_qk = nn.Sequential(
            nn.Conv1d(ch, ch, config.cca_time0, groups=ch, padding=0),
            nn.Conv1d(ch, ch, config.cca_time1, groups=self.nq + self.nkv, padding=0),
        )
        self.temp = nn.Parameter(torch.zeros(self.nkv))

    def _qk_means(self, q: torch.Tensor, k: torch.Tensor):
        """q:[b,s,lq], k:[b,s,lk] -> qk_mean_q [b,s,nq,hd], qk_mean_k [b,s,nkv,hd]."""
        b, s, _ = q.shape
        query_pre = q.view(b, s, self.nq, self.hd)
        key_pre = (
            k.view(b, s, self.nkv, 1, self.hd)
            .expand(b, s, self.nkv, self.groups, self.hd)
            .reshape(b, s, self.nq, self.hd)
        )
        qk_mean_q = (query_pre + key_pre) / 2
        qk_mean_k = qk_mean_q.view(b, s, self.nkv, self.groups, self.hd).mean(dim=-2)
        return qk_mean_q, qk_mean_k

    def _finish(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """QK L2-norm + temp; returns per-head [b,*,head_dim] q/k/v (NOT flattened)."""
        q_norm = query.norm(p=2, dim=-1, keepdim=True)
        k_norm = key.norm(p=2, dim=-1, keepdim=True)
        key = (key * (self.sqrt_hd / k_norm)) * self.temp[None, None, :, None]
        query = query * (self.sqrt_hd / q_norm)
        return query, key, value

    def forward(self, hs: torch.Tensor):
        """Prefill (no cache). hs:[b,s,d] -> q[b,s,nq,hd], k/v[b,s,nkv,hd].

        CCA math runs in the projection weight dtype (fp32 on the fp32_attn
        path): the QK L2-norm sums head_dim squares and overflows fp16 on this
        bf16-trained model — the lfm2 q/k-norm lesson, sharper. q/k/v are returned
        in the activation dtype so RoPE/KV/SDPA stay fp16."""
        b, s, _ = hs.shape
        dt = hs.dtype
        hs = hs.to(self.linear_q.weight.dtype)
        hs_d = F.pad(hs[:, :-1, :], (0, 0, 1, 0))  # previous-token stream for v2
        q = self.linear_q(hs)
        k = self.linear_k(hs)
        qk = torch.cat([q, k], dim=-1)  # [b,s,1280]
        qk_mean_q, qk_mean_k = self._qk_means(q, k)

        x = qk.transpose(1, 2)  # [b,1280,s]
        x = F.pad(x, (self.total_pad, 0))
        x = self.conv_qk(x).transpose(1, 2)  # [b,s,1280]
        query = x[..., : self.lq].view(b, s, self.nq, self.hd) + qk_mean_q
        key = x[..., self.lq :].view(b, s, self.nkv, self.hd) + qk_mean_k

        v1 = self.val_proj1(hs)
        v2 = self.val_proj2(hs_d)
        value = torch.cat([v1, v2], dim=-1).view(b, s, self.nkv, self.hd)
        q, k, v = self._finish(query, key, value)
        return q.to(dt), k.to(dt), v.to(dt)

    def forward_step(self, hs: torch.Tensor, conv_in: torch.Tensor, prev_hs: torch.Tensor):
        """Stateful step. hs:[b,s,d] (s>=1), conv_in:[b,1280,total_pad] previous
        packed-qk columns, prev_hs:[b,d] previous hidden (for v2 first column).
        Returns (q,k,v, new_conv[b,1280,total_pad], new_prev_hs[b,d])."""
        b, s, _ = hs.shape
        dt = hs.dtype
        wdt = self.linear_q.weight.dtype
        hs = hs.to(wdt)
        conv_in = conv_in.to(wdt)
        prev_hs = prev_hs.to(wdt)
        q = self.linear_q(hs)
        k = self.linear_k(hs)
        qk = torch.cat([q, k], dim=-1)  # [b,s,1280]
        qk_mean_q, qk_mean_k = self._qk_means(q, k)

        cols = qk.transpose(1, 2)  # [b,1280,s]
        w = torch.cat([conv_in, cols], dim=-1)  # [b,1280,total_pad+s]
        new_conv = w[..., -self.total_pad :]
        # two stacked valid convs (each k=2): a length-(total_pad+s) input gives
        # length-s output, the same as the prefill left-pad path.
        x = F.conv1d(w, self.conv_qk[0].weight, self.conv_qk[0].bias,
                     groups=self.conv_qk[0].groups)
        x = F.conv1d(x, self.conv_qk[1].weight, self.conv_qk[1].bias,
                     groups=self.conv_qk[1].groups)
        x = x.transpose(1, 2)  # [b,s,1280]
        query = x[..., : self.lq].view(b, s, self.nq, self.hd) + qk_mean_q
        key = x[..., self.lq :].view(b, s, self.nkv, self.hd) + qk_mean_k

        # v2 uses the previous token: prepend prev_hs, drop the last.
        hs_d = torch.cat([prev_hs.unsqueeze(1), hs[:, :-1, :]], dim=1)
        v1 = self.val_proj1(hs)
        v2 = self.val_proj2(hs_d)
        value = torch.cat([v1, v2], dim=-1).view(b, s, self.nkv, self.hd)
        new_prev_hs = hs[:, -1, :]
        q, k, v = self._finish(query, key, value)
        return (q.to(dt), k.to(dt), v.to(dt),
                new_conv.to(dt), new_prev_hs.to(dt))


# --------------------------------------------------------------------------- #
# CCA attention layer wrapper: per-head q/k/v -> RoPE -> GQA SDPA -> o_proj.
# --------------------------------------------------------------------------- #
class ZayaAttention(nn.Module):
    def __init__(self, config: ZayaConfig) -> None:
        super().__init__()
        self.config = config
        self.nq = config.num_attention_heads
        self.nkv = config.num_key_value_heads
        self.hd = config.head_dim
        self.groups = config.gqa_groups
        self.qkv = ZayaCCA(config)
        self.o_proj = nn.Linear(self.nq * self.hd, config.hidden_size,
                                bias=config.attention_bias)
        self.sdpa = SDPA(scale=self.hd ** -0.5, is_causal=True)

    def _attend(self, q, k, v, cos, sin, kv_cache, offset, seq_len, layer_idx):
        b, s = q.shape[0], q.shape[1]
        q = q.transpose(1, 2)  # [b,nq,s,hd]
        k = k.transpose(1, 2)  # [b,nkv,s,hd]
        v = v.transpose(1, 2)
        q, k = apply_partial_rope(q, k, cos, sin)
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(
                layer_idx, offset, k, v, seq_len=seq_len, query_len=s
            )
        out = self.sdpa(q, k, v)  # GQA handled internally; [b,nq,s,hd]
        out = out.transpose(1, 2).reshape(b, s, self.nq * self.hd)
        odt = out.dtype
        return self.o_proj(out.to(self.o_proj.weight.dtype)).to(odt)

    def forward(self, hs, cos, sin):
        q, k, v = self.qkv(hs)
        return self._attend(q, k, v, cos, sin, None, 0, None, 0)

    def forward_step(self, hs, cos, sin, kv_cache, conv_in, prev_hs,
                     offset, seq_len, layer_idx):
        q, k, v, new_conv, new_prev = self.qkv.forward_step(hs, conv_in, prev_hs)
        out = self._attend(q, k, v, cos, sin, kv_cache, offset, seq_len, layer_idx)
        return out, new_conv, new_prev


# --------------------------------------------------------------------------- #
# EDA router + top-1/16 MoE (with MoD skip expert).
# --------------------------------------------------------------------------- #
class ZayaRouter(nn.Module):
    def __init__(self, config: ZayaConfig, use_eda: bool) -> None:
        super().__init__()
        self.config = config
        self.use_eda = use_eda
        D = config.zaya_mlp_expansion
        E = config.router_num_experts
        self.down_proj = nn.Linear(config.hidden_size, D, bias=True)
        self.rmsnorm_eda = RMSNorm(D, eps=config.norm_epsilon)
        if use_eda:
            self.router_states_scale = nn.Parameter(torch.ones(D))
        self.router_mlp = nn.Sequential(
            nn.Linear(D, D, bias=True),
            nn.GELU(),
            nn.Linear(D, D, bias=True),
            nn.GELU(),
            nn.Linear(D, E, bias=False),
        )
        self.register_buffer("balancing_biases", torch.zeros(E, dtype=torch.float32))

    def forward(self, hs: torch.Tensor, router_states: torch.Tensor | None):
        """hs:[b,s,d] -> (route_prob[b,s], choice[b,s] int64, router_next[b,s,D])."""
        x = self.down_proj(hs)
        if self.use_eda and router_states is not None:
            x = x + router_states * self.router_states_scale
        router_next = x
        logits = self.router_mlp(self.rmsnorm_eda(x))
        prob = torch.softmax(logits.float(), dim=-1)
        biased = prob + self.balancing_biases  # selection-only bias
        choice = biased.argmax(dim=-1)  # [b,s] top-1
        route_prob = torch.gather(prob, -1, choice.unsqueeze(-1)).squeeze(-1)
        return route_prob, choice, router_next


class ZayaMoE(nn.Module):
    def __init__(self, config: ZayaConfig, use_eda: bool) -> None:
        super().__init__()
        self.config = config
        self.skip_idx = config.num_experts  # 16 = MoD identity expert
        self.use_mod = config.zaya_use_mod
        self.router = ZayaRouter(config, use_eda)
        self.switch_mlp = SwitchGLU(
            config.hidden_size, config.moe_intermediate_size, config.num_experts,
            bias=False,
        )

    def forward(self, hs: torch.Tensor, router_states: torch.Tensor | None):
        route_prob, choice, router_next = self.router(hs, router_states)
        # Clamp the (possibly skip) index into the real-expert range for the gather.
        safe = choice.clamp(max=self.config.num_experts - 1)
        y = self.switch_mlp(hs, safe.unsqueeze(-1).to(torch.uint16))[:, :, 0, :]
        if self.use_mod:
            is_skip = (choice == self.skip_idx).unsqueeze(-1)
            y = torch.where(is_skip, hs, y)  # skip = identity pass-through
        out = y * route_prob.to(y.dtype).unsqueeze(-1)
        return out, router_next


# --------------------------------------------------------------------------- #
# Learned residual scaling.
# --------------------------------------------------------------------------- #
class ResidualScaling(nn.Module):
    def __init__(self, config: ZayaConfig, not_first_layer: bool) -> None:
        super().__init__()
        d = config.hidden_size
        self.not_first_layer = not_first_layer
        self.hidden_states_scale = nn.Parameter(torch.ones(d))
        self.hidden_states_bias = nn.Parameter(torch.zeros(d))
        if not_first_layer:
            self.residual_scale = nn.Parameter(torch.ones(d))
            self.residual_bias = nn.Parameter(torch.zeros(d))

    def forward(self, residual: torch.Tensor | None, hidden_states: torch.Tensor):
        hidden_states = (hidden_states + self.hidden_states_bias) * self.hidden_states_scale
        if self.not_first_layer and residual is not None:
            residual = (residual + self.residual_bias) * self.residual_scale
        return residual, hidden_states


# --------------------------------------------------------------------------- #
# Decoder layers (att / moe) — explicit residual accumulator (fp32).
# --------------------------------------------------------------------------- #
class ZayaDecoderLayer(nn.Module):
    def __init__(self, config: ZayaConfig, layer_n: int) -> None:
        super().__init__()
        self.config = config
        self.layer_n = layer_n
        self.is_moe = config.is_moe(layer_n)
        self.input_norm = RMSNorm(config.hidden_size, eps=config.norm_epsilon)
        self.res_scale = ResidualScaling(config, not_first_layer=(layer_n != 0))
        if self.is_moe:
            # EDA is OFF for the first MoE layer (layer_n == 1).
            self.mlp = ZayaMoE(config, use_eda=(config.zaya_use_eda and layer_n != 1))
        else:
            self.self_attn = ZayaAttention(config)

    def _merge(self, hidden_states, residual):
        if self.config.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        if residual is None:
            residual = hidden_states.to(torch.float32) if self.config.residual_in_fp32 \
                else hidden_states
        else:
            residual = hidden_states + residual
        normed = self.input_norm(residual.to(self.input_norm.weight.dtype))
        return normed, residual

    def forward(self, hidden_states, residual, cos, sin, prev_router):
        normed, residual = self._merge(hidden_states, residual)
        if self.is_moe:
            out, prev_router = self.mlp(normed, prev_router)
        else:
            out = self.self_attn(normed, cos, sin)
        return out, residual, prev_router

    def forward_step(self, hidden_states, residual, cos, sin, prev_router,
                     kv_cache, conv_in, prev_hs, offset, seq_len, att_idx):
        normed, residual = self._merge(hidden_states, residual)
        new_conv = new_prev = None
        if self.is_moe:
            out, prev_router = self.mlp(normed, prev_router)
        else:
            out, new_conv, new_prev = self.self_attn.forward_step(
                normed, cos, sin, kv_cache, conv_in, prev_hs, offset, seq_len, att_idx
            )
        return out, residual, prev_router, new_conv, new_prev


# --------------------------------------------------------------------------- #
# Decoder stack.
# --------------------------------------------------------------------------- #
class ZayaModel(nn.Module):
    def __init__(self, config: ZayaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [ZayaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.res_scale = ResidualScaling(config, not_first_layer=True)  # final merge
        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_epsilon)
        rd = config.rotary_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_buffers(self, device: str = "cpu") -> None:
        rd = self.config.rotary_dim
        self.inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, rd, 2, device=device).float() / rd)
        )

    def rope_cos_sin(self, position_ids: torch.Tensor):
        freqs = position_ids[..., None].float() * self.inv_freq  # [b,s,rd/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [b,s,rd]
        return emb.cos(), emb.sin()

    def _final(self, hidden_states, residual):
        residual, hidden_states = self.res_scale(residual, hidden_states)
        residual = hidden_states + residual
        return self.final_norm(residual.to(self.final_norm.weight.dtype))

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Prefill (no KV cache; causal SDPA over the query block)."""
        h = self.embed_tokens(input_ids)
        cos, sin = self.rope_cos_sin(position_ids)
        residual = None
        prev_router = None
        for layer in self.layers:
            h, residual, prev_router = layer(h, residual, cos, sin, prev_router)
        return self._final(h, residual)

    def forward_stateful(self, input_ids, position_ids, kv_cache, conv_state, prev_hs_state):
        query_len = input_ids.shape[1]
        seq_len = position_ids.shape[1]
        offset = seq_len - query_len
        q_pos = position_ids.narrow(1, offset, query_len)
        h = self.embed_tokens(input_ids)
        cos, sin = self.rope_cos_sin(q_pos)
        residual = None
        prev_router = None
        att_idx = 0
        new_convs: list[torch.Tensor] = []
        new_prevs: list[torch.Tensor] = []
        for layer in self.layers:
            if layer.is_moe:
                h, residual, prev_router = layer(h, residual, cos, sin, prev_router)
            else:
                conv_in = conv_state.narrow(0, att_idx, 1).squeeze(0)
                prev_h = prev_hs_state.narrow(0, att_idx, 1).squeeze(0)
                h, residual, prev_router, new_conv, new_prev = layer.forward_step(
                    h, residual, cos, sin, prev_router, kv_cache, conv_in, prev_h,
                    offset, seq_len, att_idx,
                )
                new_convs.append(new_conv)
                new_prevs.append(new_prev)
                att_idx += 1
        # ONE fused write per state buffer (the GPU delegate drops chained
        # per-slot state writes — see lfm2.py module header).
        zero = torch.tensor((0,), dtype=torch.int32)
        nc = torch.stack(new_convs, dim=0)  # [att, b, 1280, total_pad]
        mutable_slice_update(
            x=conv_state, update=nc,
            begin=torch.cat([zero, zero, zero, zero]),
            end=torch.tensor(tuple(conv_state.shape), dtype=torch.int32),
        )
        npv = torch.stack(new_prevs, dim=0)  # [att, b, d]
        mutable_slice_update(
            x=prev_hs_state, update=npv,
            begin=torch.cat([zero, zero, zero]),
            end=torch.tensor(tuple(prev_hs_state.shape), dtype=torch.int32),
        )
        return self._final(h, residual)


# State names surfaced via export_to_coreai(state_names=...) and the runtime.
# KV pair first (the pipelined engine's native pair) + TWO extra fixed-shape
# states (within the coreai-pipelined-extra-states.patch ≤2 budget).
DECODE_STATE_NAMES = ("keyCache", "valueCache", "convState", "prevHsState")


def build_decode_state(
    config: ZayaConfig,
    max_seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Allocate the (zeroed) decode state for a fresh sequence. Only the att
    layers carry time-state:
      * k_cache / v_cache: [num_att_layers, 1, n_kv_heads, max_seq_len, head_dim]
      * conv_state:        [num_att_layers, 1, conv_in_out_ch, conv_state_width]
      * prev_hs_state:     [num_att_layers, 1, hidden_size]
    """
    na = config.num_att_layers
    return {
        "k_cache": torch.zeros(na, 1, config.num_key_value_heads, max_seq_len,
                               config.head_dim, dtype=dtype),
        "v_cache": torch.zeros(na, 1, config.num_key_value_heads, max_seq_len,
                               config.head_dim, dtype=dtype),
        "conv_state": torch.zeros(na, 1, config.conv_in_out_ch,
                                  config.conv_state_width, dtype=dtype),
        "prev_hs_state": torch.zeros(na, 1, config.hidden_size, dtype=dtype),
    }


class ZayaForCausalLM(nn.Module):
    """Prefill text decoder + tied LM head (no cache; causal SDPA over S)."""

    def __init__(self, config: ZayaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ZayaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(input_ids, position_ids))


class ZayaStatefulForCausalLM(nn.Module):
    """Stateful text decoder: one graph for prefill and decode."""

    def __init__(self, config: ZayaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = ZayaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.last_token_only = False

    def forward(self, input_ids, position_ids, k_cache, v_cache, conv_state, prev_hs_state):
        kv = KVCache(k_cache, v_cache)
        h = self.model.forward_stateful(input_ids, position_ids, kv, conv_state, prev_hs_state)
        if self.last_token_only:
            h = h[:, -1:, :]
        return self.lm_head(h)


# --------------------------------------------------------------------------- #
# Config + weight loading (safetensors-direct; no transformers `zaya` support).
# --------------------------------------------------------------------------- #
def zaya_config_from_dict(raw: dict) -> ZayaConfig:
    return ZayaConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        vocab_size=raw["vocab_size"],
        ffn_hidden_size=raw["ffn_hidden_size"],
        norm_epsilon=raw.get("norm_epsilon", 1e-5),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", True)),
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        cca_time0=raw.get("cca_time0", 2),
        cca_time1=raw.get("cca_time1", 2),
        attention_bias=bool(raw.get("attention_bias", False)),
        rope_theta=raw.get("rope_theta", 5_000_000.0),
        partial_rotary_factor=raw.get("partial_rotary_factor", 0.5),
        num_experts=raw["num_experts"],
        moe_router_topk=raw.get("moe_router_topk", 1),
        zaya_mlp_expansion=raw.get("zaya_mlp_expansion", 256),
        zaya_use_mod=bool(raw.get("zaya_use_mod", True)),
        zaya_use_eda=bool(raw.get("zaya_use_eda", True)),
        scale_residual_merge=bool(raw.get("scale_residual_merge", True)),
        residual_in_fp32=bool(raw.get("residual_in_fp32", True)),
        max_position_embeddings=raw.get("max_position_embeddings", 131072),
    )


def zaya_from_hf(
    model_dir: str,
    target_dtype: torch.dtype = torch.float16,
    stateful: bool = True,
    fp32_attn: bool = True,
):
    """Load ZAYA1-8B from a local HF checkpoint dir into the authored module.

    Experts are stored one tensor per expert with gate+up PACKED in linear_fc1;
    rows 0:I -> gate_proj, I:2I -> up_proj, linear_fc2 -> down_proj, then stacked
    into the SwitchGLU [1,E,out,in] layout. With ``fp32_attn`` the CCA/o_proj
    weights stay fp32 on an fp16 load (GPU-delegate exactness; the QK-norm + temp
    amplify fp16 projection error — same lesson as lfm2.py)."""
    from safetensors import safe_open

    with open(os.path.join(model_dir, "config.json")) as f:
        config = zaya_config_from_dict(json.load(f))
    cls = ZayaStatefulForCausalLM if stateful else ZayaForCausalLM
    with torch.device("meta"):
        model = cls(config)
    model.to(dtype=target_dtype)

    I = config.moe_intermediate_size
    E = config.num_experts
    # expert_stack[(layer_prefix, proj)] = {expert_idx: tensor}
    expert_stack: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
    sd: dict[str, torch.Tensor] = {}

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors in {model_dir}")

    def want_fp32(local: str) -> bool:
        if not fp32_attn:
            return False
        return (".self_attn." in local)

    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                # experts: ...zaya_block.experts.local_experts.{e}.linear_fc{1,2}.weight
                if ".zaya_block.experts.local_experts." in key:
                    parts = key.split(".zaya_block.experts.local_experts.")
                    prefix = parts[0] + ".mlp"  # authored module name
                    tail = parts[1]  # "{e}.linear_fc{1,2}.weight"
                    e_str, fc = tail.split(".linear_fc")
                    e = int(e_str)
                    t = f.get_tensor(key)
                    if t.dtype != target_dtype:
                        t = t.to(target_dtype)
                    if fc.startswith("1"):  # [2I, d] -> gate (0:I), up (I:2I)
                        expert_stack.setdefault((prefix, "gate_proj"), {})[e] = t[:I]
                        expert_stack.setdefault((prefix, "up_proj"), {})[e] = t[I:]
                    else:  # linear_fc2 [d, I] -> down
                        expert_stack.setdefault((prefix, "down_proj"), {})[e] = t
                    continue
                # router rename: ...zaya_block.router.X -> ...mlp.router.X
                local = key.replace(".zaya_block.router.", ".mlp.router.")
                t = f.get_tensor(key)
                # balancing_biases buffer stays fp32
                if local.endswith(".balancing_biases"):
                    sd[local] = t.to(torch.float32)
                    continue
                if t.is_floating_point():
                    t = t.to(torch.float32 if want_fp32(local) else target_dtype)
                sd[local] = t

    for (prefix, proj), per_expert in list(expert_stack.items()):
        if len(per_expert) != E:
            raise RuntimeError(f"{prefix}.switch_mlp.{proj}: {len(per_expert)}/{E} experts")
        stacked = torch.stack([per_expert[e] for e in range(E)], dim=0)  # [E,out,in]
        sd[f"{prefix}.switch_mlp.{proj}.weight"] = stacked.unsqueeze(0).contiguous()
        expert_stack[(prefix, proj)] = {}

    model.load_state_dict(sd, assign=True, strict=False)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.reset_buffers()

    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"Params not loaded: {meta[:8]} ... ({len(meta)} total)")
    model.eval()
    return model
