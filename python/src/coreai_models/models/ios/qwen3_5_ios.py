# Qwen3.5-0.8B hybrid decode in Apple's OFFICIAL iOS static-shape (ANE) contract.
#
# Community port — NOT an Apple model. This is the FAST PATH (static / ANE) counterpart to the
# shipped DYNAMIC qwen3.5 decode (ondevice/QwenChat, 14.7 tok/s ANE). It mirrors the official iOS
# contract used by `export/ios.py` + `models/ios/qwen3.py` so the graph specializes + runs on the
# ANE the way CoreML-LLM hits ~48 tok/s on this exact model:
#
#   * Channels-first `[B, C, 1, L]` tensors everywhere; ALL projections are `Conv2d` 1x1.
#   * Fixed-capacity iOS KV cache `[n_full, 1, n_kv*head_dim, 1, max_ctx]` (seq LAST, update on
#     dim 4 via `KVCacheHandler`) for the 6 full-attn layers; each decode step writes the new k/v
#     column at `in_step` and the per-head iOS `SDPA` reads the WHOLE cache, masked by `causal_mask`
#     `[1, max_ctx, 1, q]`. Every shape is constant across steps — the static config the ANE wants.
#   * The 18 linear (SSM) layers run the loop-free single-step GatedDeltaNet with channels-first
#     `Conv2d` projections + the fixed-shape conv/recurrent states (`conv_state[.,6144,3]`,
#     `rec_state[.,16,128,128]`) carried as Core AI states.
#
# qwen3.5 specifics preserved verbatim from the HF-verified macOS port (`models/macos/qwen3_5.py`,
# 8/8 vs HF): partial RoPE (rotary_dim = 0.25*head_dim = 64), gated full attention (q_proj 2x wide
# -> [query | gate], `out *= sigmoid(gate)`), per-head q/k RMSNormPlusOne, and the exact loop-free
# `_gated_delta_step` recurrence (imported unchanged). Only the *layout/pipeline* changes vs macOS;
# the *math* is the proven one. The wrong-layout `models/ios/qwen3_5.py` (macOS KV + hand-rolled
# SDPA) SIGTRAP'd CoreAIRuntime — this file is its iOS-contract replacement.
from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.models.macos.qwen3_5 import (
    Qwen3_5Config,
    _gated_delta_step,
    apply_rope as apply_partial_rope,
    qwen3_5_config_from_hf,
)
from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.quantization import (
    dequantize_per_tensor,
    quantize_per_tensor,
)

# HF text weights are nested under this prefix in the multimodal checkpoint.
QWEN3_5_TEXT_PREFIX = "model.language_model."

# Diagnostic ONLY (ANE-compile bisection): when set, the attention skips its per-head SDPA and the
# SSM skips its gated-delta recurrence, but BOTH keep their `mutable_slice_update` state writes + the
# projections + lm_head. If the graph then ANE-compiles, the blocker is the skipped op (SDPA /
# recurrence); if it still fails, the blocker is the static KV/state slice-update mechanism itself.
_ANE_DIAG_TRIVIAL = os.environ.get("ANE_DIAG_TRIVIAL") == "1"


# --------------------------------------------------------------------------- #
# iOS-contract norms (reduction on the LAST dim; channels-last `[b, s, 1, dim]`
# or per-head `[b, h, s, dim]`). The fp32 reduction matches HF/macOS numerics.
# --------------------------------------------------------------------------- #
class RMSNormPlusOne(nn.Module):
    """RMSNorm with the (1 + weight) gain convention (qwen3.5 input/post/q/k/final norms)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        with torch.device("cpu"):
            self.weight = nn.Parameter(torch.zeros(dim))
            self._eps = nn.Buffer(torch.tensor(eps), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        xf = x.float()
        inv = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + self._eps)
        normed = (xf * inv).to(input_dtype)
        return normed * (self.weight.float() + 1.0).to(input_dtype)


class RMSNormGated(nn.Module):
    """Gated RMSNorm for the SSM output: normalize (weight as-is) then SiLU-gate by ``z``."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        with torch.device("cpu"):
            self.weight = nn.Parameter(torch.zeros(dim))
            self._eps = nn.Buffer(torch.tensor(eps), persistent=False)

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        xf = x.float()
        inv = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + self._eps)
        normed = xf * inv * self.weight.float()
        normed = normed * F.silu(gate.float())
        return normed.to(input_dtype)


# --------------------------------------------------------------------------- #
# iOS-contract MLP (channels-first Conv2d, SiLU-gated) — qwen3.5 has no MLP bias.
# Mirrors primitives/ios/mlp.py but kept local so the [b,1,1,dim] decode reshape
# is explicit alongside the rest of the hybrid layer.
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.up_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.down_proj = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, query_len, _, dim = x.shape
        x = x.reshape(batch_size * query_len, dim, 1, 1)
        up = self.up_proj(x)
        gate = F.silu(self.gate_proj(x))
        down = self.down_proj(up * gate)
        return down.reshape(batch_size, query_len, 1, dim)


# --------------------------------------------------------------------------- #
# Gated full attention (6 layers), iOS contract: Conv2d projections, channels-first,
# fixed-capacity KV via KVCacheHandler, per-head SDPA, partial RoPE + gate.
# --------------------------------------------------------------------------- #
class Qwen3_5IOSAttention(nn.Module):
    def __init__(self, config: Qwen3_5Config, full_idx: int) -> None:
        super().__init__()
        self.full_idx = full_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        d = config.hidden_size
        # q_proj is 2x wide: per head [query(head_dim) | gate(head_dim)].
        self.q_proj = nn.Conv2d(d, self.n_heads * self.head_dim * 2, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(d, self.n_kv_heads * self.head_dim, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(d, self.n_kv_heads * self.head_dim, kernel_size=1, bias=False)
        self.o_proj = nn.Conv2d(self.n_heads * self.head_dim, d, kernel_size=1, bias=False)
        self.q_norm = RMSNormPlusOne(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormPlusOne(self.head_dim, eps=config.rms_norm_eps)
        with torch.device("cpu"):
            self._scale = nn.Buffer(torch.tensor(self.head_dim**-0.5), persistent=False)

    def forward(
        self,
        x: torch.Tensor,            # [b, s, 1, hidden]
        rope_cos: torch.Tensor,     # [b, s, rotary_dim]
        rope_sin: torch.Tensor,     # [b, s, rotary_dim]
        in_step: torch.IntTensor,   # int32 scalar
        causal_mask: torch.Tensor,  # [1, max_ctx, 1, s]
        cache: KVCacheHandler,
    ) -> torch.Tensor:
        b, s, _, _ = x.shape
        H, HKV, D = self.n_heads, self.n_kv_heads, self.head_dim

        x = x.transpose(-3, -1)             # [b, hidden, 1, s]
        qg = self.q_proj(x)                 # [b, H*D*2, 1, s]
        key = self.k_proj(x)                # [b, HKV*D, 1, s]
        value = self.v_proj(x)              # [b, HKV*D, 1, s]

        # Split [query | gate] per head (matches macОS `.view(b,s,H,D*2).chunk(2,-1)`).
        qg = qg.transpose(-3, -1).reshape(b, s, H, D * 2)
        query, gate = qg.chunk(2, dim=-1)   # each [b, s, H, D]
        query = self.q_norm(query).transpose(-2, -3)  # [b, H, s, D]

        key = key.transpose(-3, -1).reshape(b, s, HKV, D)
        key = self.k_norm(key).transpose(-2, -3)      # [b, HKV, s, D]

        if not _ANE_DIAG_TRIVIAL:
            query, key = apply_partial_rope(query, key, rope_cos, rope_sin)
        # (trivial: skip RoPE -> rope_cos/sin unused -> the in-graph cos/sin trig is DCE'd)

        # Back to channels-first `[b, n*D, 1, s]` for the cache + per-head SDPA.
        query = query.transpose(-2, -3).reshape(b, s, 1, H * D).transpose(-3, -1)
        key = key.transpose(-2, -3).reshape(b, s, 1, HKV * D).transpose(-3, -1)
        gate = gate.reshape(b, s, H * D).transpose(-1, -2).unsqueeze(-2)  # [b, H*D, 1, s]

        key, value = cache.update_and_fetch(self.full_idx, in_step, key, value, s)

        if _ANE_DIAG_TRIVIAL:
            output = query  # skip per-head SDPA (keep proj + KV slice-update + gate + o_proj)
        else:
            output = _per_head_sdpa(query, key, value, causal_mask, self._scale, H, HKV, D)
        output = output * torch.sigmoid(gate)
        output = self.o_proj(output)
        return output.transpose(-3, -1)     # [b, s, 1, hidden]


def _per_head_sdpa(
    query: torch.Tensor,        # [b, H*D, 1, s]
    key: torch.Tensor,          # [b, HKV*D, 1, L]
    value: torch.Tensor,        # [b, HKV*D, 1, L]
    causal_mask: torch.Tensor,  # [1, L, 1, s]
    scale: torch.Tensor,
    H: int,
    HKV: int,
    D: int,
) -> torch.Tensor:
    """Per-head iOS SDPA (mirrors primitives/ios/sdpa.py): each head computed
    individually so the lowering stays ANE-friendly. GQA via head // group."""
    key = key.transpose(-3, -1) * scale          # [b, L, 1, HKV*D]
    queries = query.split(D, dim=1)              # H x [b, D, 1, s]
    keys = list(key.split(D, dim=-1))            # HKV x [b, L, 1, D]
    for i in range(len(keys)):
        keys[i] = keys[i].permute(0, 2, 3, 1)    # [b, 1, D, L]
    group = H // HKV

    scores = []
    for h in range(H):
        q = queries[h].permute(0, 2, 3, 1)       # [b, 1, s, D]
        attn = q @ keys[h // group]              # [b, 1, s, L]
        scores.append(attn.permute(0, 3, 1, 2))  # [b, L, 1, s]
    full_scores = torch.cat(scores, dim=2)       # [b, L, H, s]
    masked = full_scores + torch.cat([causal_mask] * H, dim=2)
    full_scores = masked.softmax(1)              # softmax over L

    scores = full_scores.split(1, dim=2)
    values = list(value.split(D, dim=1))         # HKV x [b, D, 1, L]
    for i in range(len(values)):
        values[i] = values[i].permute(0, 2, 3, 1).squeeze(1)  # [b, L, D]

    weights = []
    for h in range(H):
        sc = scores[h].permute(0, 2, 3, 1).squeeze(1)         # [b, s, L]
        w = (sc @ values[h // group]).unsqueeze(1)            # [b, 1, s, D]
        weights.append(w.permute(0, 3, 1, 2))                 # [b, D, 1, s]
    return torch.cat(weights, dim=1)             # [b, H*D, 1, s]


# --------------------------------------------------------------------------- #
# Gated-delta linear mixer (18 layers), iOS contract: channels-first Conv2d
# projections + the proven loop-free single-step recurrence (`_gated_delta_step`).
# --------------------------------------------------------------------------- #
class Qwen3_5IOSGatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.num_k = config.linear_num_key_heads
        self.num_v = config.linear_num_value_heads
        self.dk = config.linear_key_head_dim
        self.dv = config.linear_value_head_dim
        self.key_dim = self.dk * self.num_k
        self.value_dim = self.dv * self.num_v
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.kernel = config.linear_conv_kernel_dim
        d = config.hidden_size

        self.in_proj_qkv = nn.Conv2d(d, self.conv_dim, kernel_size=1, bias=False)
        self.in_proj_z = nn.Conv2d(d, self.value_dim, kernel_size=1, bias=False)
        self.in_proj_b = nn.Conv2d(d, self.num_v, kernel_size=1, bias=False)
        self.in_proj_a = nn.Conv2d(d, self.num_v, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(self.value_dim, d, kernel_size=1, bias=False)
        # Short causal depthwise conv (kept Conv1d; weight [conv_dim, 1, kernel]).
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, self.kernel, groups=self.conv_dim, bias=False
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v))
        self.A_log = nn.Parameter(torch.zeros(self.num_v))
        self.norm = RMSNormGated(self.dv, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,         # [b, 1, 1, hidden]   (q == 1 decode)
        conv_in: torch.Tensor,   # [b, conv_dim, kernel-1]
        rec_in: torch.Tensor,    # [b, num_v, dk, dv]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        xc = x.transpose(-3, -1)            # [b, hidden, 1, 1]
        qkv = self.in_proj_qkv(xc).reshape(b, self.conv_dim, 1)   # [b, conv_dim, 1]
        z = self.in_proj_z(xc).reshape(b, self.value_dim)         # [b, value_dim]
        beta = torch.sigmoid(self.in_proj_b(xc)).reshape(b, self.num_v).unsqueeze(-1)  # [b,num_v,1]
        a = self.in_proj_a(xc).reshape(b, self.num_v)             # [b, num_v]

        # Short causal depthwise conv over [conv_state ‖ new_col], done as an explicit windowed
        # dot-product (element-wise mul + sum) rather than F.conv1d: the grouped/depthwise conv1d
        # fails the Core AI MPSGraph->ANEC "nonbonded phase" conversion on the iOS-27 ANE, whereas
        # the mul+sum form lowers (this is exactly how the CoreML-LLM ANE build does the GatedDeltaNet
        # conv). Numerically identical. new state = last (kernel-1) cols.
        w = torch.cat([conv_in, qkv], dim=-1)                     # [b, conv_dim, kernel]
        kw = self.conv1d.weight.squeeze(1).unsqueeze(0)           # [1, conv_dim, kernel]
        conv = F.silu((w * kw).sum(dim=-1)).reshape(b, self.conv_dim)  # [b, conv_dim]
        new_conv = w[..., -(self.kernel - 1):]                    # [b, conv_dim, kernel-1]

        q, k, v = torch.split(conv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(b, self.num_k, self.dk).unsqueeze(2)        # [b, num_k, 1, dk]
        k = k.reshape(b, self.num_k, self.dk).unsqueeze(2)
        v = v.reshape(b, self.num_v, self.dv).unsqueeze(2)        # [b, num_v, 1, dv]

        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())  # [b,num_v]
        g = g.unsqueeze(-1)                                       # [b, num_v, 1]

        if _ANE_DIAG_TRIVIAL:
            out = v.transpose(1, 2)          # [b,1,num_v,dv]; skip the gated-delta recurrence
            new_rec = rec_in                 # (keep conv slice-update above; rec passthrough)
        else:
            out, new_rec = _gated_delta_step(q, k, v, g, beta, rec_in)  # out [b,1,num_v,dv]
        # Gated RMSNorm is per value-head (over dv); reshape to [b*num_v, dv] like macОS.
        out = self.norm(out.reshape(b * self.num_v, self.dv), z.reshape(b * self.num_v, self.dv))
        out = self.out_proj(out.reshape(b, self.value_dim, 1, 1))  # [b, hidden, 1, 1]
        out = out.reshape(b, 1, 1, -1)                           # [b, 1, 1, hidden]
        return out, new_conv, new_rec


# --------------------------------------------------------------------------- #
# Unified hybrid decoder layer (q == 1): full-attn (static KV) OR loop-free SSM.
# --------------------------------------------------------------------------- #
class Qwen3_5IOSDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int, full_idx: int) -> None:
        super().__init__()
        d = config.hidden_size
        self.is_full = config.is_full(layer_idx)
        self.input_layernorm = RMSNormPlusOne(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormPlusOne(d, eps=config.rms_norm_eps)
        if self.is_full:
            self.self_attn = Qwen3_5IOSAttention(config, full_idx)
        else:
            self.linear_attn = Qwen3_5IOSGatedDeltaNet(config)
        self.mlp = MLP(d, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler,
        conv_in: torch.Tensor | None,
        rec_in: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        normed = self.input_layernorm(x)
        new_conv = new_rec = None
        if self.is_full:
            r = self.self_attn(normed, rope_cos, rope_sin, in_step, causal_mask, cache)
        else:
            r, new_conv, new_rec = self.linear_attn(normed, conv_in, rec_in)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, new_conv, new_rec


def _slice_write_layer(cache: torch.Tensor, layer_idx: int, new_state: torch.Tensor) -> None:
    """Write ``new_state`` ([1, *dims]) into ``cache`` ([n_layers, 1, *dims]) at ``layer_idx``
    in place -> surfaced as a Core AI STATE update (conv_rec_as_io=False path)."""
    device = cache.device
    li = torch.tensor((layer_idx,), dtype=torch.int32, device=device)
    li_end = torch.tensor((layer_idx + 1,), dtype=torch.int32, device=device)
    zeros = [torch.tensor((0,), dtype=torch.int32, device=device) for _ in range(cache.dim() - 1)]
    ends = [torch.tensor((cache.size(i),), dtype=torch.int32, device=device)
            for i in range(1, cache.dim())]
    mutable_slice_update(
        x=cache, update=new_state.unsqueeze(0),
        begin=torch.cat([li, *zeros]), end=torch.cat([li_end, *ends]),
    )


class Qwen3_5IOSModel(nn.Module):
    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        layers = []
        full_idx = 0
        for i in range(config.num_hidden_layers):
            layers.append(Qwen3_5IOSDecoderLayer(config, i, full_idx if config.is_full(i) else -1))
            if config.is_full(i):
                full_idx += 1
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNormPlusOne(config.hidden_size, eps=config.rms_norm_eps)
        # conv/rec carried as plain I/O (stacked + returned, driver round-trips) when True, or as
        # in-place-mutated Core AI STATES when False. I/O is the safe default (loads+runs on the
        # macOS GPU delegate); state mode avoids the per-step SSM-state round-trip on device but its
        # ANE acceptance is the open risk (CoreML-LLM hit Error 11 with conv/rec-as-state).
        self.conv_rec_as_io = True

    def forward(
        self,
        token_embeddings: torch.Tensor,  # [b, 1, 1, hidden]
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        cache: KVCacheHandler,
        conv_state: torch.Tensor,        # [n_lin, 1, conv_dim, kernel-1]
        rec_state: torch.Tensor,         # [n_lin, 1, num_v, dk, dv]
    ):
        h = token_embeddings
        lin_idx = 0
        new_convs: list[torch.Tensor] = []
        new_recs: list[torch.Tensor] = []
        for layer in self.layers:
            if layer.is_full:
                h, _, _ = layer(h, rope_cos, rope_sin, in_step, causal_mask, cache, None, None)
            else:
                conv_in = conv_state.narrow(0, lin_idx, 1).squeeze(0)
                rec_in = rec_state.narrow(0, lin_idx, 1).squeeze(0)
                h, new_conv, new_rec = layer(
                    h, rope_cos, rope_sin, in_step, causal_mask, cache, conv_in, rec_in
                )
                if self.conv_rec_as_io:
                    new_convs.append(new_conv)
                    new_recs.append(new_rec)
                else:
                    _slice_write_layer(conv_state, lin_idx, new_conv)
                    _slice_write_layer(rec_state, lin_idx, new_rec)
                lin_idx += 1
        if self.conv_rec_as_io:
            # stacked in linear-layer order (matches the conv_state[lin_idx] read order)
            return self.norm(h), torch.stack(new_convs, dim=0), torch.stack(new_recs, dim=0)
        return self.norm(h)  # conv/rec surfaced as states via the in-place writes


# --------------------------------------------------------------------------- #
# Extend module (decode forward) — mirrors models/ios/qwen3.py Qwen3Extend with
# 2 extra SSM states (conv/rec) and the qwen3.5 partial-RoPE table.
# --------------------------------------------------------------------------- #
class Qwen3_5IOSExtend(nn.Module):
    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.model = Qwen3_5IOSModel(config)
        self.emb_zero_point = nn.Parameter(torch.zeros([], dtype=torch.int8), requires_grad=False)
        self.emb_scale = nn.Parameter(torch.ones([], dtype=torch.float16), requires_grad=False)
        self.prefill_mode = False
        # tied lm_head (qwen3.5 ties embed/lm_head) -> logits via the embedding table.

        kv_embed_size = config.num_key_value_heads * config.head_dim
        self.kv_cache = KVCacheHandler(config.num_full_layers, kv_embed_size)

        # Partial-RoPE angles computed ARITHMETICALLY from position_ids (no cached-table
        # gather): the iOS `rope_gather_cached_cos_sin` custom op fails to specialize in this
        # hybrid graph on macOS-27, whereas the pure-arithmetic form (the shipped macOS decode
        # path) lowers cleanly. inv_freq is materialised in-graph in fp32 so the small
        # high-index angles don't underflow fp16.
        self.rotary_dim = config.rotary_dim
        self.rope_theta = float(config.rope_theta)

    def forward(
        self,
        transformer_input: torch.Tensor,  # [b, 1, 1, hidden]
        position_ids: torch.IntTensor,    # [b, 1]
        in_step: torch.IntTensor,         # int32 scalar
        causal_mask: torch.Tensor,        # [1, max_ctx, 1, 1]
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
        embedding_table: torch.Tensor,
    ):
        self.kv_cache.register_kv_cache(key_cache, value_cache)

        rd = self.rotary_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
        freqs = position_ids[..., None].float() * inv_freq            # [b, s, rd/2]
        emb = torch.cat([freqs, freqs], dim=-1)                       # [b, s, rd]
        rope_cos = emb.cos().to(transformer_input.dtype)
        rope_sin = emb.sin().to(transformer_input.dtype)

        b, s, _, hidden_dim = transformer_input.shape
        model_out = self.model(
            transformer_input, rope_cos, rope_sin, in_step, causal_mask,
            self.kv_cache, conv_state, rec_state,
        )
        out = model_out[0] if self.model.conv_rec_as_io else model_out

        # Tied lm_head: dequantize the embedding table and matmul (full fp16 vocab logits).
        if embedding_table.dtype == torch.int8:
            embedding_table = dequantize_per_tensor(
                embedding_table, self.emb_scale, self.emb_zero_point, out.dtype
            )
        embedding_table = embedding_table.reshape(
            embedding_table.shape[1], embedding_table.shape[0], embedding_table.shape[2]
        )
        out = out.transpose(-3, -1).reshape(b, 1, hidden_dim, s)
        logits = (embedding_table @ out).transpose(-2, -1)
        if self.model.conv_rec_as_io:
            return logits, model_out[1], model_out[2]
        return logits  # conv/rec surfaced as states


# --------------------------------------------------------------------------- #
# Top-level iOS model (plugs into the embed-split iOS export contract).
# --------------------------------------------------------------------------- #
class Qwen3_5ForCausalLMForiOS(BaseForCausalLMForiOS):
    _HF_MODEL_CLASS = None  # custom qwen3.5 loaders below (weights load from safetensors).

    def _init_model(self, config: Qwen3_5Config) -> None:
        self.extend = Qwen3_5IOSExtend(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        in_step: torch.IntTensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ):
        # Returns (logits, new_conv_state, new_rec_state) when extend.model.conv_rec_as_io else logits.
        table = self.load_embeddings.embedding_table
        if table.dtype == torch.int8:
            # Manual dequant-gather: the shared GatherEmbeddings' `fused_dequant_gather_reshape`
            # composite does not resolve inline in this single-`main` graph (the official iOS path
            # isolates it as its own entrypoint). Plain index_select + dequant lowers cleanly.
            gathered = table[input_ids]                              # [1, 1, 1, hidden] int8
            scale = self.gather_embeddings.scale
            token_embeddings = gathered.to(scale.dtype) * scale
        else:
            token_embeddings = self.gather_embeddings(input_ids, table)
        return self.extend(
            token_embeddings, position_ids, in_step, causal_mask,
            key_cache, value_cache, conv_state, rec_state,
            self.load_embeddings.embedding_table,
        )

    # ----- config -----
    @classmethod
    def _get_reauthored_config(cls, hf_config, max_context_length=None, num_layers=None):
        cfg = qwen3_5_config_from_hf(hf_config, max_context_length, num_layers)
        cfg.max_position_embeddings = max_context_length or 2048
        return cfg

    # ----- HF state-dict sanitization (hybrid: full vs linear layers) -----
    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        max_layer = -1
        for k in state_dict:
            if k.startswith("model.layers."):
                max_layer = max(max_layer, int(k.split(".")[2]))
        if max_layer < 0:
            raise ValueError("invalid state_dict (no model.layers.*)")

        # Conv2d 1x1 weight reshape: [out, in] -> [out, in, 1, 1].
        conv2d_attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
        conv2d_linear = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
        conv2d_mlp = ["up_proj", "gate_proj", "down_proj"]
        for i in range(max_layer + 1):
            for proj in conv2d_attn:
                key = f"model.layers.{i}.self_attn.{proj}.weight"
                if key in state_dict:
                    state_dict[key] = state_dict[key].unsqueeze(-1).unsqueeze(-1)
            for proj in conv2d_linear:
                key = f"model.layers.{i}.linear_attn.{proj}.weight"
                if key in state_dict:
                    state_dict[key] = state_dict[key].unsqueeze(-1).unsqueeze(-1)
            for proj in conv2d_mlp:
                key = f"model.layers.{i}.mlp.{proj}.weight"
                if key in state_dict:
                    state_dict[key] = state_dict[key].unsqueeze(-1).unsqueeze(-1)
            # linear_attn.conv1d.weight ([conv_dim,1,kernel]) stays a Conv1d weight.

        # Embeddings: [vocab, hidden] -> [vocab, 1, hidden], optionally int8 per-tensor.
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

        # Qwen3_5IOSModel lives inside Qwen3_5IOSExtend -> add the "extend." prefix.
        renamed = {}
        for k in list(state_dict.keys()):
            if k.startswith("model.") and "gather_embeddings" not in k:
                renamed[f"extend.{k}"] = state_dict.pop(k)
        state_dict.update(renamed)

    # ----- qwen3.5 loader (text weights nested under model.language_model.*) -----
    @classmethod
    def from_hf_qwen3_5(
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        num_layers: int | None = None,
        disable_embedding_quantization: bool = False,
    ) -> "Qwen3_5ForCausalLMForiOS":
        """Build + load the iOS qwen3.5 model directly from the HF safetensors.

        Mirrors the macОS Qwen3_5StatefulForCausalLM.from_hf_memory_efficient: the
        checkpoint nests text weights as ``model.language_model.*`` (stripped to
        ``model.*``) and carries vision/MTP weights we skip. Built on CPU (not meta)
        so the iOS non-persistent buffers — RoPE cos/sin, KV index buffers, norm eps —
        are materialised; ``load_state_dict(assign=True)`` then swaps in the weights.
        """
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig

        from coreai_models.models.base import _is_layer_key_beyond
        from coreai_models.models.macos.qwen3_5_config import (  # noqa: F401  (registers configs)
            register_qwen3_5_configs,
        )

        model_dir = snapshot_download(
            huggingface_model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
        raw_config = AutoConfig.from_pretrained(model_dir)
        hf_text = getattr(raw_config, "text_config", raw_config)
        config = cls._get_reauthored_config(hf_text, max_context_length, num_layers=num_layers)

        model = cls(config, model_device="cpu",
                    disable_embedding_quantization=disable_embedding_quantization)
        model.to(dtype=target_dtype)  # casts fp params; leaves the int8 embed placeholder int8

        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors files in {model_dir}")
        prefix = QWEN3_5_TEXT_PREFIX
        sd: dict[str, torch.Tensor] = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if not key.startswith(prefix):
                        continue  # skips model.visual.* and mtp.*
                    local = "model." + key[len(prefix):]
                    if num_layers is not None and _is_layer_key_beyond(local, num_layers):
                        continue
                    tensor = f.get_tensor(key)
                    if tensor.dtype != target_dtype:
                        tensor = tensor.to(target_dtype)
                    sd[local] = tensor

        model._mutate_state_dict(sd)
        model.load_state_dict(sd, assign=True, strict=False)

        meta = [n for n, p in model.named_parameters() if p.is_meta]
        if meta:
            raise RuntimeError(f"Parameters not loaded: {meta}")
        return model.eval()


# State names surfaced as Core AI states + expected by the Swift runner.
STATIC_DECODE_STATE_NAMES = ("key_cache", "value_cache", "conv_state", "rec_state")


def build_static_decode_state(
    config: Qwen3_5Config,
    max_ctx: int,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Allocate the FIXED-capacity iOS-layout decode states (all zero).

    Full layers: KV cache ``[n_full, 1, n_kv*head_dim, 1, max_ctx]`` (iOS layout, seq LAST).
    Linear layers: conv ``[n_lin, 1, conv_dim, kernel-1]``, rec ``[n_lin, 1, num_v, dk, dv]``.
    """
    nf, nl = config.num_full_layers, config.num_linear_layers
    kv_embed = config.num_key_value_heads * config.head_dim
    return {
        "key_cache": torch.zeros(nf, 1, kv_embed, 1, max_ctx, dtype=dtype),
        "value_cache": torch.zeros(nf, 1, kv_embed, 1, max_ctx, dtype=dtype),
        "conv_state": torch.zeros(nl, 1, config.conv_dim, config.conv_state_width, dtype=dtype),
        "rec_state": torch.zeros(nl, 1, config.linear_num_value_heads,
                                 config.linear_key_head_dim, config.linear_value_head_dim,
                                 dtype=dtype),
    }


def build_causal_mask(in_step: int, max_ctx: int, q: int = 1,
                      dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Additive iOS-SDPA mask ``[1, max_ctx, 1, q]``: 0 for cache columns <= in_step, else
    a large finite negative (fp16-safe). For q==1 decode the single query at position
    ``in_step`` may attend to all written columns ``[0 .. in_step]``."""
    mask = torch.full((1, max_ctx, 1, q), -1.0e4, dtype=dtype)
    mask[:, : in_step + 1] = 0.0
    return mask
