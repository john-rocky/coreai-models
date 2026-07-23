# Qwen3.5 text decoder (hybrid linear/full attention) for the Core AI authoring path.
#
# Community port — NOT an Apple model.  Authored to be exportable via
# coreai_models.export.macos.export_to_coreai.  Decoupled from transformers:
# weights load directly from the HF safetensors, config is a local dataclass,
# so the export env (transformers 4.57, no qwen3_5) needs no upstream support.
#
# Layout per layer follows HF Qwen3.5 exactly:
#   - layer_types repeat [linear, linear, linear, full] (full when idx % 4 == 3)
#   - all RMSNorms use the (1 + weight) gain convention -> RMSNormPlusOne
#   - the linear mixer's gated output norm uses weight as-is -> RMSNormGated
# RoPE is applied manually from precomputed cos/sin (passed in as graph inputs):
# the text path collapses mRoPE to standard partial RoPE, and this de-risks the
# mRoPE-interleave convention vs the composite RoPE.
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_torch.composite_ops import GatedDeltaUpdate

from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache, SSMState
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNormGated, RMSNormPlusOne
from coreai_models.primitives.macos.sdpa import SDPA


@dataclass
class Qwen3_5Config:
    """Subset of Qwen3.5 text_config needed for authoring (values from config.json)."""

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    vocab_size: int = 248320
    intermediate_size: int = 3584
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    # full attention
    head_dim: int = 256
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    attn_output_gate: bool = True
    partial_rotary_factor: float = 0.25
    rope_theta: float = 1e7
    # linear mixer (GatedDeltaNet)
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    full_attention_interval: int = 4
    layer_types: list[str] = field(default_factory=list)

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def is_full(self, layer_idx: int) -> bool:
        return layer_idx % self.full_attention_interval == self.full_attention_interval - 1

    @property
    def num_full_layers(self) -> int:
        return sum(self.is_full(i) for i in range(self.num_hidden_layers))

    @property
    def num_linear_layers(self) -> int:
        return self.num_hidden_layers - self.num_full_layers

    @property
    def conv_dim(self) -> int:
        # [q | k | v] depthwise-conv channels: 2*key_dim + value_dim
        return 2 * self.linear_key_head_dim * self.linear_num_key_heads + (
            self.linear_value_head_dim * self.linear_num_value_heads
        )

    @property
    def conv_state_width(self) -> int:
        # Minimal decode conv state = kernel-1 columns of projected qkv.
        return self.linear_conv_kernel_dim - 1


# --------------------------------------------------------------------------- #
# RoPE (manual, from precomputed cos/sin) — matches HF apply_rotary_pos_emb
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q,k: [b, n_heads, s, head_dim]; cos,sin: [b, s, rotary_dim]. Partial RoPE."""
    # cos/sin are computed in fp32; cast to the query dtype so the partial-RoPE
    # concat keeps a single dtype (HF casts the rotary tables to the activation
    # dtype). No-op in fp32; required in fp16 (else cat mixes fp32 rotated dims
    # with the fp16 pass-through dims and raises).
    cos = cos.unsqueeze(1).to(q.dtype)  # [b,1,s,rotary_dim]
    sin = sin.unsqueeze(1).to(q.dtype)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([q_rot * cos + _rotate_half(q_rot) * sin, q_pass], dim=-1)
    k_embed = torch.cat([k_rot * cos + _rotate_half(k_rot) * sin, k_pass], dim=-1)
    return q_embed, k_embed


# --------------------------------------------------------------------------- #
# Full attention (gated GQA, partial RoPE, per-head q/k norm)
# --------------------------------------------------------------------------- #
class Qwen3_5FullAttention(nn.Module):
    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        d = config.hidden_size
        # q_proj is 2x wide: [query | gate] interleaved per head (head_dim*2)
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNormPlusOne(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormPlusOne(self.head_dim, eps=config.rms_norm_eps)
        self.sdpa = SDPA(scale=self.head_dim**-0.5, is_causal=True)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        offset: int = 0,
        seq_len: int | None = None,
        full_idx: int = 0,
    ) -> torch.Tensor:
        """Prefill when ``kv_cache`` is None (causal SDPA over the query block).

        When a cache is supplied, write the query's k/v at ``offset`` and fetch
        the full [0:seq_len] history; the lower-right causal SDPA then lets the
        query block attend to all cached positions (handles prefill AND decode).
        """
        b, s, _ = x.shape
        H, HKV, D = self.n_heads, self.n_kv_heads, self.head_dim

        qg = self.q_proj(x).view(b, s, H, D * 2)
        q, gate = qg.chunk(2, dim=-1)  # each [b,s,H,D]
        gate = gate.reshape(b, s, H * D)

        q = self.q_norm(q).transpose(1, 2)  # [b,H,s,D]
        k = self.k_norm(self.k_proj(x).view(b, s, HKV, D)).transpose(1, 2)  # [b,HKV,s,D]
        v = self.v_proj(x).view(b, s, HKV, D).transpose(1, 2)  # [b,HKV,s,D]

        q, k = apply_rope(q, k, cos, sin)
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(
                full_idx, offset, k, v, seq_len=seq_len, query_len=s
            )
        out = self.sdpa(q, k, v)  # GQA handled internally; [b,H,s,D]
        out = out.transpose(1, 2).reshape(b, s, H * D)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Linear attention mixer (GatedDeltaNet): short causal conv + delta rule
# --------------------------------------------------------------------------- #
def _gated_delta_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2_norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loop-free single-token (``s == 1``) gated-delta update.

    Numerically identical to ``coreai_torch.composite_ops.GatedDeltaUpdate`` at
    ``s == 1`` (same l2-norm, fp32 recurrence, scaling), but expressed WITHOUT the
    ``torch.ops.higher_order.while_loop``. The while_loop op fails to lower on the
    Core AI device delegates (BNNS `compilationFailed`; MPSGraph `scf.while`
    dynamic-shape region mismatch), which is what blocks qwen3.5 on-device. For the
    decode graph (query block = 1 token) the scan is a single step, so we unroll it.

    Shapes: query/key ``[b,h,1,dk]``, value ``[b,h,1,dv]``, g/beta ``[b,h,1]``,
    initial_state ``[b,h,dk,dv]``. Returns ``(out [b,1,h,dv], new_state [b,h,dk,dv])``.
    """
    input_dtype = query.dtype

    def l2norm(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)

    if use_qk_l2_norm:
        query = l2norm(query)
        key = l2norm(key)
    query, key, value, beta, g = (t.to(torch.float32) for t in (query, key, value, beta, g))
    query = query * (query.shape[-1] ** -0.5)
    state = initial_state.to(torch.float32)
    g_exp = g.exp()

    # Single recurrence step (t = 0), mirroring GatedDeltaUpdate.body_fn exactly.
    q_t = query[:, :, 0, :].unsqueeze(-1)            # [b,h,dk,1]
    k_t = key[:, :, 0, :].unsqueeze(-1)              # [b,h,dk,1]
    v_t = value[:, :, 0, :]                          # [b,h,dv]
    g_t = g_exp[:, :, 0].unsqueeze(-1).unsqueeze(-1)  # [b,h,1,1]
    beta_t = beta[:, :, 0].unsqueeze(-1)             # [b,h,1]

    state = state * g_t
    kv_mem = (state * k_t).sum(dim=-2)               # [b,h,dv]
    delta = ((v_t - kv_mem) * beta_t).unsqueeze(-2)  # [b,h,1,dv]
    state = state + k_t * delta                      # [b,h,dk,dv]
    out_val = (state * q_t).sum(dim=-2)              # [b,h,dv]
    out = out_val.unsqueeze(1)                       # [b,1,h,dv] (== output.transpose(1,2))
    return out.to(input_dtype), state.to(input_dtype)


def _gated_delta_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2_norm: bool = True,
    doublings: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loop-free CHUNKED gated-delta update — processes a whole query block of S tokens in
    PARALLEL (the gated-DeltaNet chunk algorithm), so prefill amortizes the weight-heavy
    projections across S instead of paying them per token (the TTFT win). Numerically
    identical to ``_gated_delta_step`` looped S times / ``GatedDeltaUpdate`` (parity gated in
    ``_smoke/test_gdn_chunked_parity.py``), but expressed with ONLY delegate-lowerable ops:
    matmuls, masked exp, and a FIXED ``doublings``-step product for the triangular inverse —
    no ``while_loop`` (the qwen3.5 on-device blocker) and no python token loop. Reduces to the
    single-step result at S==1, so one graph serves both prefill (S>1) and decode (S==1).

    Shapes: query/key ``[b,h,S,dk]``, value ``[b,h,S,dv]``, g/beta ``[b,h,S]``,
    initial_state ``[b,h,dk,dv]``. Returns ``(out [b,S,h,dv], new_state [b,h,dk,dv])`` — the
    same layout as ``GatedDeltaUpdate``.

    ``doublings`` must satisfy ``2**doublings >= S`` for an exact inverse (M strictly-lower is
    nilpotent so the extra factors are identity); 12 covers the 4096 max context.
    """
    input_dtype = query.dtype

    def l2norm(x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)

    if use_qk_l2_norm:
        query = l2norm(query)
        key = l2norm(key)
    q, k, v, beta, g = (t.to(torch.float32) for t in (query, key, value, beta, g))
    q = q * (q.shape[-1] ** -0.5)
    S0 = initial_state.to(torch.float32)
    b, h, s, dk = k.shape

    dev = k.device
    lones = torch.tril(torch.ones(s, s, device=dev))          # cumsum as a matmul (lowers)
    C = g @ lones.transpose(-1, -2)                           # [b,h,s]; C_t = sum_{l<=t} g_l
    tril_strict = torch.tril(torch.ones(s, s, device=dev), -1)
    tril_incl = torch.tril(torch.ones(s, s, device=dev), 0)
    neg_inf = torch.full((s, s), -1e30, device=dev)

    diffC = C.unsqueeze(-1) - C.unsqueeze(-2)                 # diffC[t,l] = C_t - C_l
    decay_lo = torch.where(tril_incl.bool(), diffC, neg_inf).exp()   # 0 above diagonal (no overflow)
    kk = k @ k.transpose(-1, -2)
    qk = q @ k.transpose(-1, -2)

    M = beta.unsqueeze(-1) * (decay_lo * tril_strict) * kk    # strictly-lower [b,h,s,s]
    kS0 = k @ S0                                              # [b,h,s,dv]
    R = beta.unsqueeze(-1) * (v - C.exp().unsqueeze(-1) * kS0)

    eye = torch.eye(s, device=dev).expand(b, h, s, s)
    X = -M
    inv = eye + X                                             # (I + X^(2^0))
    P = X
    for _ in range(doublings - 1):
        P = P @ P                                             # X^(2^j); -> 0 once 2^j >= s
        inv = inv @ (eye + P)
    U = inv @ R                                               # (I+M)^-1 R

    N = (decay_lo * tril_incl) * qk
    out = C.exp().unsqueeze(-1) * (q @ S0) + N @ U            # [b,h,s,dv]

    w = (C[:, :, -1:] - C).exp()                              # exp(C_last - C_l)
    state = C[:, :, -1:, None].exp() * S0 + (k * w.unsqueeze(-1)).transpose(-1, -2) @ U
    out = out.transpose(1, 2)                                 # [b,s,h,dv] (== GatedDeltaUpdate)
    return out.to(input_dtype), state.to(input_dtype)


class Qwen3_5GatedDeltaNet(nn.Module):
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

        self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(d, self.num_v, bias=False)
        self.in_proj_a = nn.Linear(d, self.num_v, bias=False)
        # short causal depthwise conv: pad left k-1, slice [:s]  (Risk C lowering)
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.kernel,
                                groups=self.conv_dim, padding=self.kernel - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.num_v))
        self.A_log = nn.Parameter(torch.zeros(self.num_v))
        self.norm = RMSNormGated(self.dv, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, d, bias=False)
        self.gdu = GatedDeltaUpdate(use_qk_l2_norm=True)
        # When True, the s==1 path uses the loop-free single-step (device delegates
        # can't lower the GatedDeltaUpdate while_loop). Set on all linear layers
        # before tracing the on-device DECODE graph (query_len=1). False keeps the
        # existing dynamic prefill+decode export (composite while_loop) unchanged.
        self.use_loopfree_step = False
        # When True, ANY query block (S>=1) uses the loop-free CHUNKED scan
        # (`_gated_delta_chunk`) — lowers on device AND processes the prompt in
        # parallel, so one graph serves chunked prefill (TTFT win) and S==1 decode.
        # Takes precedence over use_loopfree_step. Set before tracing the
        # dynamic-query-length prefill+decode graph.
        self.use_loopfree_chunk = False
        # When True, the chunk scan runs in a custom fp32 Metal kernel (`metal_chunk`,
        # set by qwen3_5_gdn_metal.metalize_gdn_chunk) — numerically STABLE at large
        # chunks (the in-graph fp16 doubling-inverse NaNs at chunk>=64). Takes precedence.
        self.use_metal_chunk = False
        self.metal_chunk = None

    def forward(
        self,
        x: torch.Tensor,
        conv_in: torch.Tensor | None = None,
        rec_in: torch.Tensor | None = None,
    ):
        """Prefill when ``conv_in`` is None: left-padded depthwise conv + zero
        recurrent init; returns just the output tensor.

        Stateful path (``conv_in`` given, shape [b, conv_dim, kernel-1]): prepend
        the conv state, run a valid (unpadded) depthwise conv, seed the delta
        rule with ``rec_in``, and additionally return the updated
        (conv_state[b, conv_dim, kernel-1], recurrent_state[b, num_v, dk, dv]).
        This generalises across seq lengths (decode S=1 and chunked prefill).
        """
        b, s, _ = x.shape
        mixed = self.in_proj_qkv(x).transpose(1, 2)  # [b, conv_dim, s]
        z = self.in_proj_z(x)
        beta = self.in_proj_b(x).sigmoid()  # [b,s,num_v]
        a = self.in_proj_a(x)
        # negative decay logits (GatedDeltaUpdate applies exp internally)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())  # [b,s,num_v]

        if conv_in is None:
            conv = F.silu(self.conv1d(mixed)[..., :s])  # [b, conv_dim, s]
            new_conv = None
            initial_state = torch.zeros(b, self.num_v, self.dk, self.dv, dtype=x.dtype)
        else:
            w = torch.cat([conv_in, mixed], dim=-1)  # [b, conv_dim, (kernel-1)+s]
            conv = F.silu(
                F.conv1d(w, self.conv1d.weight, bias=None, padding=0, groups=self.conv_dim)
            )  # [b, conv_dim, s]
            new_conv = w[..., -(self.kernel - 1):]  # [b, conv_dim, kernel-1]
            initial_state = rec_in

        mixed = conv.transpose(1, 2)  # [b,s,conv_dim]
        q, k, v = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(b, s, self.num_k, self.dk).transpose(1, 2)  # [b,num_k,s,dk]
        k = k.reshape(b, s, self.num_k, self.dk).transpose(1, 2)
        v = v.reshape(b, s, self.num_v, self.dv).transpose(1, 2)  # [b,num_v,s,dv]
        if self.num_v != self.num_k:
            # GVA (e.g. Qwen3.6-35B-A3B: 32 value / 16 key heads): repeat each q/k
            # head per value group, matching HF's repeat_interleave(num_v//num_k,
            # dim=heads) pairing (v heads 2i, 2i+1 share k head i). Expressed as
            # expand+reshape so it lowers without an aten.repeat_interleave.
            r = self.num_v // self.num_k
            q = q.unsqueeze(2).expand(b, self.num_k, r, s, self.dk).reshape(
                b, self.num_v, s, self.dk)
            k = k.unsqueeze(2).expand(b, self.num_k, r, s, self.dk).reshape(
                b, self.num_v, s, self.dk)
        g = g.transpose(1, 2)  # [b,num_v,s]
        beta = beta.transpose(1, 2)

        if self.use_metal_chunk:
            out, new_rec = self.metal_chunk(q, k, v, g, beta, initial_state)  # fp32 kernel scan
        elif self.use_loopfree_chunk:
            out, new_rec = _gated_delta_chunk(
                q, k, v, g, beta, initial_state, self.gdu.use_qk_l2_norm
            )  # loop-free chunked scan (chunked prefill + S==1 decode, device-lowerable)
        elif self.use_loopfree_step:
            out, new_rec = _gated_delta_step(
                q, k, v, g, beta, initial_state, self.gdu.use_qk_l2_norm
            )  # loop-free s==1 (device decode graph)
        else:
            out, new_rec = self.gdu(q, k, v, g, beta, initial_state)  # out [b,s,num_v,dv]

        out = out.reshape(-1, self.dv)
        out = self.norm(out, z.reshape(-1, self.dv))
        out = out.reshape(b, s, self.value_dim)
        out = self.out_proj(out)
        if conv_in is None:
            return out
        return out, new_conv, new_rec


# --------------------------------------------------------------------------- #
# Unified decoder layer — dispatches linear vs full by layer index
# --------------------------------------------------------------------------- #
class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int) -> None:
        super().__init__()
        d = config.hidden_size
        self.is_full = config.is_full(layer_idx)
        self.input_layernorm = RMSNormPlusOne(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormPlusOne(d, eps=config.rms_norm_eps)
        if self.is_full:
            self.self_attn = Qwen3_5FullAttention(config)
        else:
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        self.mlp = MLP(d, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        conv_cache: SSMState | None = None,
        rec_cache: SSMState | None = None,
        offset: int = 0,
        seq_len: int | None = None,
        full_idx: int = 0,
        lin_idx: int = 0,
    ) -> torch.Tensor:
        normed = self.input_layernorm(x)
        if self.is_full:
            if kv_cache is None:
                r = self.self_attn(normed, cos, sin)
            else:
                r = self.self_attn(normed, cos, sin, kv_cache, offset, seq_len, full_idx)
            h = x + r
        else:
            if conv_cache is None:
                r = self.linear_attn(normed)
                h = x + r
            else:
                conv_in = conv_cache.states.narrow(0, lin_idx, 1).squeeze(0)
                rec_in = rec_cache.states.narrow(0, lin_idx, 1).squeeze(0)
                r, new_conv, new_rec = self.linear_attn(normed, conv_in, rec_in)
                h = x + r
                conv_cache.update_states(lin_idx, new_conv)
                rec_cache.update_states(lin_idx, new_rec)
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# --------------------------------------------------------------------------- #
# Full decoder stack (prefill, no KV cache; in-graph RoPE from position_ids)
# --------------------------------------------------------------------------- #
class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [self._make_layer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNormPlusOne(config.hidden_size, eps=config.rms_norm_eps)
        rd = config.rotary_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _make_layer(self, config: Qwen3_5Config, layer_idx: int) -> nn.Module:
        """Layer factory — the MoE variant (qwen3_5_moe.py) overrides this to swap
        the dense MLP for the sparse-MoE block; everything else is shared."""
        return Qwen3_5DecoderLayer(config, layer_idx)

    def reset_buffers(self, device: str = "cpu") -> None:
        """Recreate non-persistent buffers (inv_freq). Needed after meta-device
        construction + ``load_state_dict`` (non-persistent buffers aren't in the
        state dict, so they stay on the meta device)."""
        rd = self.config.rotary_dim
        self.inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, rd, 2, device=device).float() / rd)
        )

    def rope_cos_sin(self, position_ids: torch.Tensor):
        # position_ids [b,s] -> cos/sin [b,s,rotary_dim].  Text mRoPE == plain RoPE.
        freqs = position_ids[..., None].float() * self.inv_freq  # [b,s,rd/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [b,s,rd]
        return emb.cos(), emb.sin()

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        cos, sin = self.rope_cos_sin(position_ids)
        for layer in self.layers:
            h = layer(h, cos, sin)
        return self.norm(h)

    def forward_stateful(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        conv_cache: SSMState,
        rec_cache: SSMState,
    ) -> torch.Tensor:
        """Stateful prefill/decode. ``input_ids`` carries the query tokens
        ([b, query_len]); ``position_ids`` carries the full positions
        ([b, seq_len]) so ``offset = seq_len - query_len`` (0 for fresh prefill,
        past length for decode). Full layers index ``kv_cache`` by full-layer
        counter; linear layers index the SSM caches by linear-layer counter.
        """
        return self.forward_stateful_core(
            self.embed_tokens(input_ids), position_ids, kv_cache, conv_cache, rec_cache
        )

    def forward_stateful_core(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        conv_cache: SSMState,
        rec_cache: SSMState,
    ) -> torch.Tensor:
        """Stateful hybrid core on pre-computed embeddings (the head-split path):
        the giant tied embed/lm_head table (≈⅓ of this large-vocab model) stays on
        the CPU front-end (embed gather) and the separate head bundle (lm_head), so
        the converted decode core holds only the transformer. Returns the final-norm
        hidden state ([b, query_len, hidden]). ``forward_stateful`` is this with an
        in-graph embed lookup in front.
        """
        query_len = inputs_embeds.shape[1]
        seq_len = position_ids.shape[1]
        offset = seq_len - query_len
        q_pos = position_ids.narrow(1, offset, query_len)  # positions of the query tokens
        h = inputs_embeds
        cos, sin = self.rope_cos_sin(q_pos)
        full_idx = 0
        lin_idx = 0
        for layer in self.layers:
            if layer.is_full:
                h = layer(h, cos, sin, kv_cache=kv_cache, offset=offset,
                          seq_len=seq_len, full_idx=full_idx)
                full_idx += 1
            else:
                h = layer(h, cos, sin, conv_cache=conv_cache, rec_cache=rec_cache,
                          lin_idx=lin_idx)
                lin_idx += 1
        return self.norm(h)


class Qwen3_5ForCausalLM(nn.Module):
    """Prefill text decoder + tied LM head. v1: no KV cache (causal SDPA over S)."""

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(input_ids, position_ids))


def build_decode_state(
    config: Qwen3_5Config,
    max_seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Allocate the hybrid decode state tensors (all zero) for a fresh sequence.

    Hybrid layout — only the layers that need a given state get a slot:
      * k_cache / v_cache: [num_full_layers, 1, n_kv_heads, max_seq_len, head_dim]
      * conv_state:        [num_linear_layers, 1, conv_dim, kernel-1]
      * rec_state:         [num_linear_layers, 1, num_v_heads, dk, dv]
    """
    nf, nl = config.num_full_layers, config.num_linear_layers
    return {
        "k_cache": torch.zeros(nf, 1, config.num_key_value_heads, max_seq_len,
                               config.head_dim, dtype=dtype),
        "v_cache": torch.zeros(nf, 1, config.num_key_value_heads, max_seq_len,
                               config.head_dim, dtype=dtype),
        "conv_state": torch.zeros(nl, 1, config.conv_dim, config.conv_state_width, dtype=dtype),
        "rec_state": torch.zeros(nl, 1, config.linear_num_value_heads,
                                 config.linear_key_head_dim, config.linear_value_head_dim,
                                 dtype=dtype),
    }


# State names surfaced via export_to_coreai(state_names=...) and the runtime.
DECODE_STATE_NAMES = ("keyCache", "valueCache", "convState", "recState")


class Qwen3_5ForCausalLMStateful(nn.Module):
    """Stateful text decoder: one graph for prefill and decode.

    forward inputs: input_ids [b, query_len], position_ids [b, seq_len], plus the
    four state tensors from ``build_decode_state``. The state tensors are mutated
    in place (surfaced as Core AI states via ``DECODE_STATE_NAMES``); logits for
    the query tokens are returned.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        # When True the graph returns only the last token's logits -> a static
        # [b, 1, vocab] output even under a dynamic query length. This is what a
        # generation loop needs and is what makes the dynamic (one-graph) export
        # runnable on the Core AI runtime (which can't infer dynamic output shapes).
        self.last_token_only = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ) -> torch.Tensor:
        kv = KVCache(k_cache, v_cache)
        conv = SSMState(conv_state)
        rec = SSMState(rec_state)
        h = self.model.forward_stateful(input_ids, position_ids, kv, conv, rec)
        if self.last_token_only:
            h = h[:, -1:, :]
        return self.lm_head(h)

    def build_macos_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        query_len: int,
        offset: int,
        trace_kv_len: int,
    ) -> dict:
        """Reference inputs + dynamic shapes for the hybrid export (one graph for
        prefill and decode). ``input_ids`` is the dynamic query block; ``position_ids``
        is the dynamic full-length positions (so offset = seq_len - query_len). KV
        caches grow on the sequence dim; the SSM conv/recurrent states are static.
        Consumed by ``export_macos_model`` via the hybrid hook.
        """
        cfg = self.config
        b = 1
        input_ids = torch.randint(1, cfg.vocab_size, (b, query_len), dtype=torch.int32)
        position_ids = (
            torch.arange(query_len + offset, dtype=torch.int32).unsqueeze(0).expand(b, -1)
        )
        state = build_decode_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)
        reference_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
            "conv_state": state["conv_state"],
            "rec_state": state["rec_state"],
        }
        seq_ids = torch.export.Dim("seq_ids", max=max_context_length - 2)
        seq_pos = torch.export.Dim("seq_pos", min=query_len, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        dynamic_shapes = {
            "input_ids": {1: seq_ids},
            "position_ids": {1: seq_pos},
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
            "conv_state": None,  # static SSM state
            "rec_state": None,
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids"),
            "output_names": ("logits",),
            "state_names": DECODE_STATE_NAMES,
        }


# --------------------------------------------------------------------------- #
# Head-split design: decode core (inputs_embeds -> hidden) + head (hidden -> logits)
# --------------------------------------------------------------------------- #
# The tied embed/lm_head table is vocab*hidden ≈ ⅓ of this large-vocab 0.8B model.
# On device it is held ONCE on the CPU front-end: it gathers ``inputs_embeds`` for the
# core and is the head's ``lm_head`` weight. The converted decode core then holds only
# the transformer (a clean int8 target), matching the Apple / Gemma4 "giant gather
# tables stay on the front-end" design and the CoreML-LLM reference split.
class Qwen3_5DecodeCore(nn.Module):
    """Stateful hybrid decode core on pre-computed embeddings: ``inputs_embeds`` +
    the 4 hybrid state tensors -> final-norm hidden. No embed/lm_head in the graph.
    Same dynamic prefill+decode graph as :class:`Qwen3_5ForCausalLMStateful`, minus
    the giant tables. Set ``last_token_only`` for a static [b,1,hidden] output so the
    dynamic graph is runnable on the Core AI runtime (generation needs only the last).
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        self.last_token_only = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ) -> torch.Tensor:
        kv = KVCache(k_cache, v_cache)
        conv = SSMState(conv_state)
        rec = SSMState(rec_state)
        h = self.model.forward_stateful_core(inputs_embeds, position_ids, kv, conv, rec)
        if self.last_token_only:
            h = h[:, -1:, :]
        return h

    def build_macos_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        query_len: int,
        offset: int,
        trace_kv_len: int,
    ) -> dict:
        """Reference inputs + dynamic shapes for the head-split core export. Mirrors
        :meth:`Qwen3_5ForCausalLMStateful.build_macos_export_spec` but the dynamic
        query input is ``inputs_embeds`` ([b, query_len, hidden], float) and the output
        is ``hidden`` rather than ``logits``."""
        cfg = self.config
        b = 1
        inputs_embeds = torch.zeros(b, query_len, cfg.hidden_size, dtype=target_dtype)
        position_ids = (
            torch.arange(query_len + offset, dtype=torch.int32).unsqueeze(0).expand(b, -1)
        )
        state = build_decode_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)
        reference_inputs = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
            "conv_state": state["conv_state"],
            "rec_state": state["rec_state"],
        }
        seq_ids = torch.export.Dim("seq_ids", max=max_context_length - 2)
        seq_pos = torch.export.Dim("seq_pos", min=query_len, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        dynamic_shapes = {
            "inputs_embeds": {1: seq_ids},
            "position_ids": {1: seq_pos},
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
            "conv_state": None,
            "rec_state": None,
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("inputs_embeds", "position_ids"),
            "output_names": ("hidden",),
            "state_names": DECODE_STATE_NAMES,
        }


class Qwen3_5Head(nn.Module):
    """The ``lm_head`` bundle: final-norm hidden -> logits. The weight is the tied
    embed table (shared with the front-end embed gather on device). No softcap for
    Qwen3.5 (unlike Gemma 4), so this is a single matmul."""

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)


# --------------------------------------------------------------------------- #
# Registry integration (config shim + BaseForCausalLM wrapper)
# --------------------------------------------------------------------------- #
# HF text weights live under this prefix in the multimodal checkpoint.
QWEN3_5_TEXT_PREFIX = "model.language_model."


def qwen3_5_config_from_hf(hf_text_config, max_context_length=None, num_layers=None) -> Qwen3_5Config:
    """Build the authoring :class:`Qwen3_5Config` from an HF text config object
    (e.g. the registered ``Qwen3_5TextConfig`` shim, or any object exposing the
    same attributes)."""
    def g(key, default=None):
        return getattr(hf_text_config, key, default)

    n_layers = num_layers if num_layers is not None else g("num_hidden_layers")
    layer_types = list(g("layer_types") or [])
    if num_layers is not None:
        layer_types = layer_types[:num_layers]
    rope = g("rope_parameters") or {}
    return Qwen3_5Config(
        hidden_size=g("hidden_size"),
        num_hidden_layers=n_layers,
        vocab_size=g("vocab_size"),
        intermediate_size=g("intermediate_size"),
        rms_norm_eps=g("rms_norm_eps", 1e-6),
        tie_word_embeddings=bool(g("tie_word_embeddings", True)),
        head_dim=g("head_dim"),
        num_attention_heads=g("num_attention_heads"),
        num_key_value_heads=g("num_key_value_heads"),
        partial_rotary_factor=rope.get("partial_rotary_factor", 0.25),
        rope_theta=rope.get("rope_theta", 1e7),
        linear_num_key_heads=g("linear_num_key_heads"),
        linear_num_value_heads=g("linear_num_value_heads"),
        linear_key_head_dim=g("linear_key_head_dim"),
        linear_value_head_dim=g("linear_value_head_dim"),
        linear_conv_kernel_dim=g("linear_conv_kernel_dim"),
        full_attention_interval=g("full_attention_interval"),
        layer_types=layer_types,
    )


class Qwen3_5StatefulForCausalLM(BaseForCausalLM):
    """Registry-facing wrapper: same stateful hybrid graph as
    :class:`Qwen3_5ForCausalLMStateful`, adapted to the ``BaseForCausalLM`` /
    export-pipeline contract (config shim + Qwen3.5 weight loading).
    """

    _HF_MODEL_CLASS = None  # custom loaders below; no transformers HF class needed

    # Reuse the proven stateful graph + export spec verbatim.
    forward = Qwen3_5ForCausalLMStateful.forward
    build_macos_export_spec = Qwen3_5ForCausalLMStateful.build_macos_export_spec

    def _init_model(self, config: Qwen3_5Config) -> None:
        self.model = Qwen3_5Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.last_token_only = False

    def _mutate_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        # Param names already match HF (no q/k/v fusion); nothing to rewrite.
        return

    @classmethod
    def _get_reauthored_config(cls, hf_config, max_context_length=None, num_layers=None):
        return qwen3_5_config_from_hf(hf_config, max_context_length, num_layers)

    @classmethod
    def from_hf_memory_efficient(
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        hf_config_attr: str | None = None,
        hf_state_dict_prefix: str = QWEN3_5_TEXT_PREFIX,
    ):
        """Load the Qwen3.5 text decoder from a (possibly multimodal) checkpoint.

        Overrides the base loader because Qwen3.5 nests text weights as
        ``model.language_model.*`` (not the base's ``<prefix>model.*`` shape) and
        carries vision/MTP weights we skip. Weights load straight from
        safetensors; ``AutoConfig`` works via the registered config shim.
        """
        import glob
        import os

        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig

        from coreai_models.models.base import _is_layer_key_beyond

        model_dir = snapshot_download(
            huggingface_model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
        raw_config = AutoConfig.from_pretrained(model_dir)
        hf_text = getattr(raw_config, hf_config_attr) if hf_config_attr else raw_config
        config = cls._get_reauthored_config(hf_text, max_context_length, num_layers=num_layers)

        model = cls(config, model_device="meta")
        model.to(dtype=target_dtype)

        # This checkpoint stores a single oddly-named shard
        # ("model.safetensors-00001-of-00001.safetensors"), so glob *.safetensors
        # rather than relying on the base helper's fixed filenames.
        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors files in {model_dir}")
        prefix = hf_state_dict_prefix
        sd: dict[str, torch.Tensor] = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key == "lm_head.weight" and not config.tie_word_embeddings:
                        # Untied head (e.g. Qwen3.6-27B) lives at the checkpoint
                        # root, not under model.language_model.*
                        sd[key] = f.get_tensor(key).to(target_dtype)
                        continue
                    if not key.startswith(prefix):
                        continue  # skips model.visual.* and mtp.*
                    local = "model." + key[len(prefix):]
                    if num_layers is not None and _is_layer_key_beyond(local, num_layers):
                        continue
                    tensor = f.get_tensor(key)
                    if tensor.dtype != target_dtype:
                        tensor = tensor.to(target_dtype)
                    sd[local] = tensor

        model.load_state_dict(sd, assign=True, strict=False)
        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        # Non-persistent buffers (inv_freq) aren't in the state dict; rebuild on CPU.
        model.model.reset_buffers()

        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(f"Parameters not loaded: {meta_params}")
        return model
