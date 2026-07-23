# Qwen3.5-0.8B hybrid decode in Apple's OFFICIAL iOS STATIC-shape contract.
#
# Community port — NOT an Apple model. This is the optional FAST PATH (static / ANE)
# counterpart to the shipped DYNAMIC qwen3.5 decode (ondevice/QwenChat, 14.7 tok/s ANE).
# It conforms to the iOS static-shape trick used by `export/ios.py` + `models/ios/qwen3.py`:
#
#   * FIXED-capacity KV cache `[n_full, 1, n_kv, MAX_CTX, head_dim]` for the 6 full-attn layers.
#     Each decode step WRITES the single new k/v column at `in_step` and the attention reads
#     the WHOLE fixed-length cache; a `causal_mask` (0 for positions <= in_step, -inf else) masks
#     the unwritten / future columns. Every tensor shape is therefore CONSTANT across steps — no
#     dynamic `position_ids[1,L]`, no dynamic-length narrow — which is what unlocks the ANE static
#     fast path (CoreML-LLM hits ~48 tok/s on this exact model this way).
#   * `in_step` (int32 scalar) selects the write column and indexes the RoPE table.
#   * `position_ids[1, 1]` carries ONLY the current decode token's absolute position (for RoPE).
#
# qwen3.5 specifics preserved verbatim from the HF-verified macOS port (`models/macos/qwen3_5.py`,
# 8/8 vs HF): partial RoPE (rotary_dim = 0.25*head_dim), gated full attention (q_proj 2× wide →
# [query | gate]), per-head RMSNormPlusOne q/k norm, and the loop-free single-step GatedDeltaNet
# for the 18 linear (SSM) layers (conv/recurrent states are already static-shape).
#
# Scope: VALIDATION SLICE = STATIC q=1 decode only (proves the static-shape speedup before the full
# bucketed [8,16,64] prefill port). Reuses the proven macOS math and changes ONLY the two things that
# made the dynamic graph dynamic (the KV fetch + the position handling).
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.models.macos.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5GatedDeltaNet,
    apply_rope,
)
from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNormPlusOne

# Large finite negative for the additive attention mask (fp16-safe; avoids NaN that a
# true -inf can create in softmax and avoids -inf constants the device lowering rejects).
MASK_NEG = -1.0e4


# --------------------------------------------------------------------------- #
# Fixed-capacity KV cache (iOS static trick), macOS [.,.,seq,head_dim] layout.
# --------------------------------------------------------------------------- #
# We keep the macOS KV layout `[n_full, 1, n_kv_heads, MAX_CTX, head_dim]` (so the
# proven qwen3.5 attention math is unchanged) but, unlike the macOS `KVCache`,
# we DO NOT narrow to `seq_len`: the whole fixed-length cache is returned every
# step and the `causal_mask` handles masking. Constant shapes everywhere.
class StaticKVCache:
    """Fixed-capacity KV cache: write the new column at ``in_step``, return the
    WHOLE cache (shape constant). Layout ``[n_full, 1, n_kv_heads, MAX_CTX, head_dim]``.
    """

    def __init__(self, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        self._k = k_cache
        self._v = v_cache

    def update_and_fetch(
        self, full_idx: int, in_step: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """k, v: ``[1, n_kv_heads, 1, head_dim]`` (single decode column). Writes at
        sequence position ``in_step`` for layer ``full_idx`` and returns the full
        ``[1, n_kv_heads, MAX_CTX, head_dim]`` cache for that layer."""
        device = self._k.device
        li = torch.tensor((full_idx,), dtype=torch.int32, device=device)
        li_end = torch.tensor((full_idx + 1,), dtype=torch.int32, device=device)
        z = torch.tensor((0,), dtype=torch.int32, device=device)
        in_step = in_step.to(torch.int32).reshape(1)
        # begin = [full_idx, 0, 0, in_step, 0]; end = [full_idx+1, 1, n_kv, in_step+1, head_dim]
        begin = torch.cat([li, z, z, in_step, z])
        end = torch.cat(
            [
                li_end,
                torch.tensor((self._k.size(1),), dtype=torch.int32, device=device),
                torch.tensor((self._k.size(2),), dtype=torch.int32, device=device),
                in_step + 1,
                torch.tensor((self._k.size(4),), dtype=torch.int32, device=device),
            ]
        )
        mutable_slice_update(x=self._k, update=k.unsqueeze(0), begin=begin, end=end)
        mutable_slice_update(x=self._v, update=v.unsqueeze(0), begin=begin, end=end)
        # Return the WHOLE fixed cache for this layer (no seq narrow → static shape).
        # Use narrow(0)+squeeze (the proven macOS KVCache read pattern) rather than int
        # indexing the just-mutated buffer.
        k = self._k.narrow(0, full_idx, 1).squeeze(0)
        v = self._v.narrow(0, full_idx, 1).squeeze(0)
        return k, v

    @property
    def k(self) -> torch.Tensor:
        return self._k

    @property
    def v(self) -> torch.Tensor:
        return self._v


# --------------------------------------------------------------------------- #
# Static gated full attention (q=1 decode): partial RoPE + gate, masked SDPA over
# the WHOLE fixed-capacity cache.
# --------------------------------------------------------------------------- #
class Qwen3_5StaticFullAttention(nn.Module):
    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        d = config.hidden_size
        # q_proj is 2× wide: [query | gate] interleaved per head (head_dim*2).
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNormPlusOne(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormPlusOne(self.head_dim, eps=config.rms_norm_eps)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,                 # [1, 1, hidden]
        cos: torch.Tensor,               # [1, 1, rotary_dim]
        sin: torch.Tensor,               # [1, 1, rotary_dim]
        in_step: torch.Tensor,           # int32 scalar
        causal_mask: torch.Tensor,       # [1, 1, 1, MAX_CTX] additive (0 / large-neg)
        kv_cache: StaticKVCache,
        full_idx: int,
    ) -> torch.Tensor:
        b, s, _ = x.shape  # s == 1
        H, HKV, D = self.n_heads, self.n_kv_heads, self.head_dim

        qg = self.q_proj(x).view(b, s, H, D * 2)
        q, gate = qg.chunk(2, dim=-1)            # each [b,s,H,D]
        gate = gate.reshape(b, s, H * D)

        q = self.q_norm(q).transpose(1, 2)       # [b,H,1,D]
        k = self.k_norm(self.k_proj(x).view(b, s, HKV, D)).transpose(1, 2)  # [b,HKV,1,D]
        v = self.v_proj(x).view(b, s, HKV, D).transpose(1, 2)              # [b,HKV,1,D]

        q, k = apply_rope(q, k, cos, sin)        # partial RoPE

        # Write the new column at in_step, fetch the WHOLE fixed cache (static shape).
        k_full, v_full = kv_cache.update_and_fetch(full_idx, in_step, k, v)  # [b,HKV,MAX_CTX,D]

        # GQA: expand kv heads to query heads (reshape/expand — avoids repeat_interleave,
        # which fails to lower on the device delegates). [b,HKV,L,D] -> [b,H,L,D].
        rep = H // HKV
        L = k_full.shape[-2]
        k_full = (
            k_full.unsqueeze(2).expand(b, HKV, rep, L, D).reshape(b, H, L, D)
        )
        v_full = (
            v_full.unsqueeze(2).expand(b, HKV, rep, L, D).reshape(b, H, L, D)
        )

        # Masked SDPA over the whole fixed cache. scores [b,H,1,L] + causal_mask, softmax, @ V.
        scores = (q @ k_full.transpose(-1, -2)) * self.scale  # [b,H,1,L]
        scores = scores + causal_mask                          # broadcast [1,1,1,L]
        attn = scores.softmax(dim=-1)
        out = attn @ v_full                                    # [b,H,1,D]

        out = out.transpose(1, 2).reshape(b, s, H * D)   # [b,1,H*D]
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Unified static decoder layer (q=1): full-attn (static KV) OR loop-free SSM.
# --------------------------------------------------------------------------- #
class Qwen3_5StaticDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5Config, layer_idx: int) -> None:
        super().__init__()
        d = config.hidden_size
        self.is_full = config.is_full(layer_idx)
        self.input_layernorm = RMSNormPlusOne(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormPlusOne(d, eps=config.rms_norm_eps)
        if self.is_full:
            self.self_attn = Qwen3_5StaticFullAttention(config)
        else:
            # Reuse the proven macOS GatedDeltaNet; force the loop-free single step.
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
            self.linear_attn.use_loopfree_step = True
        self.mlp = MLP(d, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        in_step: torch.Tensor,
        causal_mask: torch.Tensor,
        kv_cache: StaticKVCache,
        conv_in: torch.Tensor | None,
        rec_in: torch.Tensor | None,
        full_idx: int,
    ):
        normed = self.input_layernorm(x)
        new_conv = new_rec = None
        if self.is_full:
            r = self.self_attn(normed, cos, sin, in_step, causal_mask, kv_cache, full_idx)
        else:
            r, new_conv, new_rec = self.linear_attn(normed, conv_in, rec_in)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r, new_conv, new_rec


# --------------------------------------------------------------------------- #
# Static hybrid decode model (q=1) — RoPE from in_step, in-graph state plumbing.
# --------------------------------------------------------------------------- #
class Qwen3_5StaticModel(nn.Module):
    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5StaticDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNormPlusOne(config.hidden_size, eps=config.rms_norm_eps)
        rd = config.rotary_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_buffers(self, device: str = "cpu") -> None:
        rd = self.config.rotary_dim
        self.inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, rd, 2, device=device).float() / rd)
        )

    def rope_cos_sin(self, position_ids: torch.Tensor):
        # position_ids [b,1] -> cos/sin [b,1,rotary_dim]. Text mRoPE == plain RoPE.
        freqs = position_ids[..., None].float() * self.inv_freq  # [b,1,rd/2]
        emb = torch.cat([freqs, freqs], dim=-1)                  # [b,1,rd]
        return emb.cos(), emb.sin()

    def forward(
        self,
        inputs_embeds: torch.Tensor,     # [1,1,hidden]
        position_ids: torch.Tensor,      # [1,1] absolute position of the decode token
        in_step: torch.Tensor,           # int32 scalar (== position for a contiguous decode)
        causal_mask: torch.Tensor,       # [1,1,1,MAX_CTX]
        kv_cache: StaticKVCache,
        conv_state: torch.Tensor,        # [n_lin,1,conv_dim,kernel-1]
        rec_state: torch.Tensor,         # [n_lin,1,num_v,dk,dv]
    ) -> torch.Tensor:
        h = inputs_embeds
        cos, sin = self.rope_cos_sin(position_ids)
        full_idx = 0
        lin_idx = 0
        conv_updates: list[tuple[int, torch.Tensor]] = []
        rec_updates: list[tuple[int, torch.Tensor]] = []
        for layer in self.layers:
            if layer.is_full:
                h, _, _ = layer(h, cos, sin, in_step, causal_mask, kv_cache, None, None, full_idx)
                full_idx += 1
            else:
                conv_in = conv_state.narrow(0, lin_idx, 1).squeeze(0)
                rec_in = rec_state.narrow(0, lin_idx, 1).squeeze(0)
                h, new_conv, new_rec = layer(
                    h, cos, sin, in_step, causal_mask, kv_cache, conv_in, rec_in, 0
                )
                conv_updates.append((lin_idx, new_conv))
                rec_updates.append((lin_idx, new_rec))
                lin_idx += 1
        # Write the SSM state updates back (mutated in place → surfaced as states).
        for li, nc in conv_updates:
            _slice_write_layer(conv_state, li, nc)
        for li, nr in rec_updates:
            _slice_write_layer(rec_state, li, nr)
        return self.norm(h)


def _slice_write_layer(cache: torch.Tensor, layer_idx: int, new_state: torch.Tensor) -> None:
    """Write ``new_state`` ([1, *dims]) into ``cache`` ([n_layers, 1, *dims]) at ``layer_idx``."""
    device = cache.device
    li = torch.tensor((layer_idx,), dtype=torch.int32, device=device)
    li_end = torch.tensor((layer_idx + 1,), dtype=torch.int32, device=device)
    begin = torch.cat([li, *[torch.tensor((0,), dtype=torch.int32, device=device)
                             for _ in range(cache.dim() - 1)]])
    end = torch.cat([li_end, *[torch.tensor((cache.size(i),), dtype=torch.int32, device=device)
                               for i in range(1, cache.dim())]])
    mutable_slice_update(x=cache, update=new_state.unsqueeze(0), begin=begin, end=end)


# --------------------------------------------------------------------------- #
# Decode core (inputs_embeds -> hidden) for the head-split export, mirroring the
# macOS Qwen3_5DecodeCore: the giant tied embed/lm_head table stays on the CPU
# front-end; the converted core holds only the transformer.
# --------------------------------------------------------------------------- #
class Qwen3_5StaticDecodeCore(nn.Module):
    """Static q=1 hybrid decode core. forward: ``inputs_embeds`` [1,1,hidden],
    ``position_ids`` [1,1], ``in_step`` scalar, ``causal_mask`` [1,1,1,MAX_CTX] +
    4 states (k/v/conv/rec) -> final-norm hidden [1,1,hidden]. All shapes static.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5StaticModel(config)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        in_step: torch.Tensor,
        causal_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ) -> torch.Tensor:
        kv = StaticKVCache(k_cache, v_cache)
        return self.model(
            inputs_embeds, position_ids, in_step, causal_mask, kv, conv_state, rec_state
        )


class Qwen3_5StaticForCausalLM(nn.Module):
    """Static q=1 hybrid decode with in-graph embed gather + tied lm_head. forward:
    ``input_ids`` [1,1], ``position_ids`` [1,1], ``in_step`` scalar, ``causal_mask``
    [1,1,1,MAX_CTX] + 4 states -> logits [1,1,vocab]. All shapes static.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5StaticModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        in_step: torch.Tensor,
        causal_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
        rec_state: torch.Tensor,
    ) -> torch.Tensor:
        kv = StaticKVCache(k_cache, v_cache)
        h = self.model(
            self.model.embed_tokens(input_ids), position_ids, in_step, causal_mask, kv,
            conv_state, rec_state,
        )
        return self.lm_head(h)


# State names surfaced via export_to_coreai(state_names=...) and the Swift runtime.
STATIC_DECODE_STATE_NAMES = ("keyCache", "valueCache", "convState", "recState")


def build_static_decode_state(
    config: Qwen3_5Config,
    max_ctx: int,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Allocate the FIXED-capacity static decode state tensors (all zero).

    Full layers: KV cache `[n_full, 1, n_kv_heads, MAX_CTX, head_dim]` (fixed seq).
    Linear layers: conv `[n_lin, 1, conv_dim, kernel-1]`, rec `[n_lin, 1, num_v, dk, dv]` (static).
    """
    nf, nl = config.num_full_layers, config.num_linear_layers
    return {
        "k_cache": torch.zeros(nf, 1, config.num_key_value_heads, max_ctx,
                               config.head_dim, dtype=dtype),
        "v_cache": torch.zeros(nf, 1, config.num_key_value_heads, max_ctx,
                               config.head_dim, dtype=dtype),
        "conv_state": torch.zeros(nl, 1, config.conv_dim, config.conv_state_width, dtype=dtype),
        "rec_state": torch.zeros(nl, 1, config.linear_num_value_heads,
                                 config.linear_key_head_dim, config.linear_value_head_dim,
                                 dtype=dtype),
    }


def build_causal_mask(in_step: int, max_ctx: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Additive decode mask ``[1,1,1,MAX_CTX]``: 0 for positions <= in_step, MASK_NEG else.

    A single decode query at absolute position ``in_step`` may attend to all cache
    columns it (and earlier steps) have written, i.e. ``[0 .. in_step]``; unwritten /
    future columns are masked to a large finite negative so softmax ignores them."""
    mask = torch.full((1, 1, 1, max_ctx), MASK_NEG, dtype=dtype)
    mask[..., : in_step + 1] = 0.0
    return mask
