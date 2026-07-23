# LFM2 / LFM2.5 text decoder (conv + full-attention hybrid) for the Core AI
# authoring path.
#
# Community port — NOT an Apple model.  Authored to be exportable via
# coreai_models.export.macos.export_to_coreai.  Decoupled from transformers:
# weights load directly from the HF safetensors, config is parsed from the raw
# config.json, so the export env needs no upstream LFM2 support.
#
# Layout per layer follows HF Lfm2 (modeling_lfm2.py) exactly:
#   - layer_types mixes "conv" and "full_attention" (1.2B: 10 conv + 6 attn)
#   - all RMSNorms use the plain weight gain (NO +1) -> RMSNorm
#   - conv block: in_proj -> [B|C|x] -> Bx=B*x -> depthwise causal conv1d
#     (kernel=conv_L_cache, NO activation) -> C*conv_out -> out_proj
#   - attn block: per-head q/k RMSNorm -> full-dim RoPE -> GQA SDPA -> out_proj
# The decode conv state holds the kernel-1 previous Bx columns; the stateful
# step cats [state | Bx] and runs a valid (unpadded) depthwise conv — the same
# math as HF's roll+write rolling buffer, in the qwen3_5 conv-state pattern.
# There is NO recurrent scan anywhere, so the decode-only graph is loop-free by
# construction (no while_loop to escape, unlike qwen3.5's GatedDeltaUpdate).
#
# IMPORTANT (macOS-27-beta GPU delegate): per-layer SSMState.update_states
# calls compile to a read_handle/slice_update/write_handle ROUND TRIP per conv
# layer on the SAME state handle, and the GPU delegate silently DROPS all of
# them when there is more than one (state buffer stays zero; verified with a
# 3-slot repro — a single-slot graph works). The layers therefore RETURN their
# new conv columns and forward_stateful issues ONE fused full-state write at
# the end of the step (disjoint slots + each layer reads only its own slot
# before any write, so the semantics are identical).
#
# IMPORTANT #2 (same delegate): the attention q/k/v/out projections compute
# ~1.3% relative error in fp16 under a dynamic-shape graph (the fused attn
# prologue accumulates in fp16; the same matmul in a static graph is 0.07%).
# LFM2.5's large q/k-norm gains amplify that noise and it compounds across the
# 16-layer stack into garbage logits (full-stack cos 0.71 vs eager). Keeping
# the four attention projections in fp32 (weights fp32, cast in/out around the
# matmul) restores exactness — layer-level cos 1.000000 — at +~126 MB for the
# 1.2B model. Conv-mixer and MLP matmuls measure clean in fp16 and stay fp16.
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA


@dataclass
class Lfm2Config:
    """Subset of the LFM2/LFM2.5 config needed for authoring (from config.json)."""

    hidden_size: int = 2048
    num_hidden_layers: int = 16
    vocab_size: int = 65536
    intermediate_size: int = 12288  # raw value; see ff_dim for the effective size
    block_auto_adjust_ff_dim: bool = True
    block_ffn_dim_multiplier: float = 1.0
    block_multiple_of: int = 256
    norm_eps: float = 1e-5
    tie_embedding: bool = True
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    conv_L_cache: int = 3
    conv_bias: bool = False
    rope_theta: float = 1e6
    max_position_embeddings: int = 128000
    layer_types: list[str] = field(default_factory=list)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def ff_dim(self) -> int:
        # Mirrors Lfm2MLP.__init__: 2/3 * intermediate, multiplier, round up to
        # block_multiple_of (1.2B: 12288 -> 8192).
        inter = self.intermediate_size
        if self.block_auto_adjust_ff_dim:
            inter = int(2 * inter / 3)
            if self.block_ffn_dim_multiplier is not None:
                inter = int(self.block_ffn_dim_multiplier * inter)
                inter = self.block_multiple_of * (
                    (inter + self.block_multiple_of - 1) // self.block_multiple_of
                )
        return inter

    def is_full(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "full_attention"

    @property
    def num_full_layers(self) -> int:
        return sum(self.is_full(i) for i in range(self.num_hidden_layers))

    @property
    def num_conv_layers(self) -> int:
        return self.num_hidden_layers - self.num_full_layers

    @property
    def conv_state_width(self) -> int:
        # Minimal decode conv state = kernel-1 columns of gated input (Bx).
        return self.conv_L_cache - 1


# --------------------------------------------------------------------------- #
# RoPE (manual, from precomputed cos/sin) — full-dim, matches HF apply_rotary_pos_emb
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q,k: [b, n_heads, s, head_dim]; cos,sin: [b, s, head_dim]. Full-dim RoPE.

    cos/sin are computed fp32; cast to the query dtype (HF casts the rotary
    tables to the activation dtype — required in fp16)."""
    cos = cos.unsqueeze(1).to(q.dtype)  # [b,1,s,head_dim]
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


# --------------------------------------------------------------------------- #
# Full attention (GQA, per-head q/k RMSNorm, full-dim RoPE)
# --------------------------------------------------------------------------- #
class Lfm2Attention(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        d = config.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
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

        # Projections run in the weights' dtype (fp32 on the fp16 export path —
        # see IMPORTANT #2 in the module header); no-op when the whole model is
        # one dtype (the fp32 parity tier).
        xp = x.to(self.q_proj.weight.dtype)
        q = self.q_layernorm(self.q_proj(xp).to(x.dtype).view(b, s, H, D)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(xp).to(x.dtype).view(b, s, HKV, D)).transpose(1, 2)
        v = self.v_proj(xp).to(x.dtype).view(b, s, HKV, D).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        if kv_cache is not None:
            k, v = kv_cache.update_and_fetch(
                full_idx, offset, k, v, seq_len=seq_len, query_len=s
            )
        out = self.sdpa(q, k, v)  # GQA handled internally; [b,H,s,D]
        out = out.transpose(1, 2).reshape(b, s, H * D)
        return self.out_proj(out.to(self.out_proj.weight.dtype)).to(x.dtype)


# --------------------------------------------------------------------------- #
# Short-conv mixer: in_proj -> gated depthwise causal conv (no activation)
# --------------------------------------------------------------------------- #
class Lfm2ShortConv(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        d = config.hidden_size
        self.d = d
        self.kernel = config.conv_L_cache
        bias = config.conv_bias
        self.in_proj = nn.Linear(d, 3 * d, bias=bias)
        self.out_proj = nn.Linear(d, d, bias=bias)
        # named `conv` so state-dict keys match HF (layers.N.conv.conv.weight)
        self.conv = nn.Conv1d(d, d, self.kernel, groups=d, bias=bias,
                              padding=self.kernel - 1)

    def forward(self, x: torch.Tensor, conv_in: torch.Tensor | None = None):
        """Prefill when ``conv_in`` is None: left-padded depthwise conv.

        Stateful path (``conv_in`` given, shape [b, hidden, kernel-1]): prepend
        the conv state, run a valid (unpadded) depthwise conv, and additionally
        return the updated conv state (the last kernel-1 Bx columns). Same math
        as HF's rolling [b, hidden, kernel] buffer; generalises across seq lens
        (decode S=1 and chunked prefill).
        """
        b, s, _ = x.shape
        BCx = self.in_proj(x).transpose(1, 2)  # [b, 3d, s]
        gate_b, gate_c, xv = BCx.chunk(3, dim=1)  # each [b, d, s]
        bx = gate_b * xv

        if conv_in is None:
            conv_out = self.conv(bx)[..., :s]  # causal: left pad kernel-1, slice
            new_conv = None
        else:
            w = torch.cat([conv_in, bx], dim=-1)  # [b, d, (kernel-1)+s]
            conv_out = F.conv1d(w, self.conv.weight, bias=self.conv.bias,
                                padding=0, groups=self.d)  # [b, d, s]
            new_conv = w[..., -(self.kernel - 1):]

        y = (gate_c * conv_out).transpose(1, 2)  # [b, s, d]
        out = self.out_proj(y)
        if conv_in is None:
            return out
        return out, new_conv


# --------------------------------------------------------------------------- #
# Unified decoder layer — dispatches conv vs full attention by layer type
# --------------------------------------------------------------------------- #
class Lfm2DecoderLayer(nn.Module):
    def __init__(self, config: Lfm2Config, layer_idx: int) -> None:
        super().__init__()
        d = config.hidden_size
        self.is_full = config.is_full(layer_idx)
        self.operator_norm = RMSNorm(d, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(d, eps=config.norm_eps)
        if self.is_full:
            self.self_attn = Lfm2Attention(config)
        else:
            self.conv = Lfm2ShortConv(config)
        self.feed_forward = MLP(d, config.ff_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        conv_in: torch.Tensor | None = None,
        offset: int = 0,
        seq_len: int | None = None,
        full_idx: int = 0,
    ):
        """Conv layers in the stateful path (``conv_in`` given) return
        ``(hidden, new_conv)`` — the caller issues the fused state write."""
        normed = self.operator_norm(x)
        new_conv = None
        if self.is_full:
            if kv_cache is None:
                r = self.self_attn(normed, cos, sin)
            else:
                r = self.self_attn(normed, cos, sin, kv_cache, offset, seq_len, full_idx)
            h = x + r
        else:
            if conv_in is None:
                h = x + self.conv(normed)
            else:
                r, new_conv = self.conv(normed, conv_in)
                h = x + r
        h = h + self.feed_forward(self.ffn_norm(h))
        if conv_in is not None:
            return h, new_conv
        return h


# --------------------------------------------------------------------------- #
# Full decoder stack (in-graph RoPE from position_ids)
# --------------------------------------------------------------------------- #
class Lfm2Model(nn.Module):
    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Lfm2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        # final norm before the head; HF names it embedding_norm
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        d = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, d, 2).float() / d))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_buffers(self, device: str = "cpu") -> None:
        """Recreate non-persistent buffers (inv_freq) after meta-device load."""
        d = self.config.head_dim
        self.inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, d, 2, device=device).float() / d)
        )

    def rope_cos_sin(self, position_ids: torch.Tensor):
        # position_ids [b,s] -> cos/sin [b,s,head_dim] (full-dim default RoPE)
        freqs = position_ids[..., None].float() * self.inv_freq  # [b,s,d/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [b,s,d]
        return emb.cos(), emb.sin()

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        cos, sin = self.rope_cos_sin(position_ids)
        for layer in self.layers:
            h = layer(h, cos, sin)
        return self.embedding_norm(h)

    def forward_stateful(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        """Stateful prefill/decode. ``input_ids`` carries the query tokens
        ([b, query_len]); ``position_ids`` carries the full positions
        ([b, seq_len]) so ``offset = seq_len - query_len``. Full layers index
        ``kv_cache`` by full-layer counter; conv layers read slot conv_idx of
        ``conv_state`` and the updated columns are written back in ONE fused
        slice_update at the end (see the module header: the GPU delegate drops
        chained per-slot state writes).
        """
        query_len = input_ids.shape[1]
        seq_len = position_ids.shape[1]
        offset = seq_len - query_len
        q_pos = position_ids.narrow(1, offset, query_len)
        h = self.embed_tokens(input_ids)
        cos, sin = self.rope_cos_sin(q_pos)
        full_idx = 0
        conv_idx = 0
        new_convs: list[torch.Tensor] = []
        for layer in self.layers:
            if layer.is_full:
                h = layer(h, cos, sin, kv_cache=kv_cache, offset=offset,
                          seq_len=seq_len, full_idx=full_idx)
                full_idx += 1
            else:
                conv_in = conv_state.narrow(0, conv_idx, 1).squeeze(0)
                h, new_conv = layer(h, cos, sin, conv_in=conv_in)
                new_convs.append(new_conv)
                conv_idx += 1
        new_full = torch.cat([n.unsqueeze(0) for n in new_convs], dim=0)
        zero = torch.tensor((0,), dtype=torch.int32)
        mutable_slice_update(
            x=conv_state,
            update=new_full,
            begin=torch.cat([zero, zero, zero, zero]),
            end=torch.tensor(tuple(conv_state.shape), dtype=torch.int32),
        )
        return self.embedding_norm(h)

    def forward_stateful_embeds(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        """Same as ``forward_stateful`` but the query is already embedded
        (``inputs_embeds`` [b, query_len, hidden]) — used by multimodal callers
        (e.g. LFM2-Audio) that assemble mixed text/audio embeddings on the host.
        embed_tokens is skipped; everything else (RoPE offset, hybrid KV/conv
        state, fused conv write-back) is identical.
        """
        query_len = inputs_embeds.shape[1]
        seq_len = position_ids.shape[1]
        offset = seq_len - query_len
        q_pos = position_ids.narrow(1, offset, query_len)
        h = inputs_embeds
        cos, sin = self.rope_cos_sin(q_pos)
        full_idx = 0
        conv_idx = 0
        new_convs: list[torch.Tensor] = []
        for layer in self.layers:
            if layer.is_full:
                h = layer(h, cos, sin, kv_cache=kv_cache, offset=offset,
                          seq_len=seq_len, full_idx=full_idx)
                full_idx += 1
            else:
                conv_in = conv_state.narrow(0, conv_idx, 1).squeeze(0)
                h, new_conv = layer(h, cos, sin, conv_in=conv_in)
                new_convs.append(new_conv)
                conv_idx += 1
        new_full = torch.cat([n.unsqueeze(0) for n in new_convs], dim=0)
        zero = torch.tensor((0,), dtype=torch.int32)
        mutable_slice_update(
            x=conv_state,
            update=new_full,
            begin=torch.cat([zero, zero, zero, zero]),
            end=torch.tensor(tuple(conv_state.shape), dtype=torch.int32),
        )
        return self.embedding_norm(h)


class Lfm2ForCausalLM(nn.Module):
    """Prefill text decoder + tied LM head (no cache; causal SDPA over S)."""

    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embedding:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.model(input_ids, position_ids))


def build_decode_state(
    config: Lfm2Config,
    max_seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Allocate the hybrid decode state tensors (all zero) for a fresh sequence.

    Hybrid layout — only the layers that need a given state get a slot:
      * k_cache / v_cache: [num_full_layers, 1, n_kv_heads, max_seq_len, head_dim]
      * conv_state:        [num_conv_layers, 1, hidden_size, kernel-1]
    """
    nf, nc = config.num_full_layers, config.num_conv_layers
    return {
        "k_cache": torch.zeros(nf, 1, config.num_key_value_heads, max_seq_len,
                               config.head_dim, dtype=dtype),
        "v_cache": torch.zeros(nf, 1, config.num_key_value_heads, max_seq_len,
                               config.head_dim, dtype=dtype),
        "conv_state": torch.zeros(nc, 1, config.hidden_size,
                                  config.conv_state_width, dtype=dtype),
    }


# State names surfaced via export_to_coreai(state_names=...) and the runtime.
# KV pair first (the pipelined engine's native pair) + ONE extra fixed-shape
# state (within the coreai-pipelined-extra-states.patch ≤2 budget).
DECODE_STATE_NAMES = ("keyCache", "valueCache", "convState")


class Lfm2ForCausalLMStateful(nn.Module):
    """Stateful text decoder: one graph for prefill and decode.

    forward inputs: input_ids [b, query_len], position_ids [b, seq_len], plus
    the three state tensors from ``build_decode_state``. The state tensors are
    mutated in place (surfaced as Core AI states via ``DECODE_STATE_NAMES``);
    logits for the query tokens are returned.
    """

    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embedding:
            self.lm_head.weight = self.model.embed_tokens.weight
        # When True the graph returns only the last token's logits -> a static
        # [b, 1, vocab] output even under a dynamic query length.
        self.last_token_only = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        kv = KVCache(k_cache, v_cache)
        h = self.model.forward_stateful(input_ids, position_ids, kv, conv_state)
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
        """Reference inputs + dynamic shapes for the hybrid export (one graph
        for prefill and decode). Consumed by ``export_macos_model``'s hybrid
        hook; the decode-only export script builds its own spec instead.
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
            "conv_state": None,  # static conv state
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids"),
            "output_names": ("logits",),
            "state_names": DECODE_STATE_NAMES,
        }


class Lfm2EmbedsForCausalLMStateful(nn.Module):
    """Embeds-input stateful decoder: one graph for prefill and decode, but the
    query arrives already embedded (``inputs_embeds`` [b, query_len, hidden]).

    Used by multimodal callers (LFM2-Audio) that assemble mixed text/audio-in/
    audio-out embeddings on the host. The tied text LM head is applied to the
    hidden states; audio-frame logits (depthformer) are produced by a separate
    graph. State layout + fused conv write-back are identical to the input_ids
    stateful graph, so the same runtime state tensors (build_decode_state) apply.
    """

    def __init__(self, config: Lfm2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embedding:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.last_token_only = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        kv = KVCache(k_cache, v_cache)
        h = self.model.forward_stateful_embeds(inputs_embeds, position_ids, kv, conv_state)
        if self.last_token_only:
            h = h[:, -1:, :]
        return self.lm_head(h)


# --------------------------------------------------------------------------- #
# Weight loading (safetensors-direct; no transformers LFM2 support needed)
# --------------------------------------------------------------------------- #
def lfm2_config_from_dict(raw: dict) -> Lfm2Config:
    """Build the authoring config from a raw config.json dict. Handles both the
    LFM2.5 ``layer_types`` list and the LFM2 ``full_attn_idxs`` form."""
    layer_types = raw.get("layer_types")
    if not layer_types:
        full_idxs = set(raw.get("full_attn_idxs") or [])
        layer_types = [
            "full_attention" if i in full_idxs else "conv"
            for i in range(raw["num_hidden_layers"])
        ]
    return Lfm2Config(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        vocab_size=raw["vocab_size"],
        intermediate_size=raw.get("intermediate_size") or raw["block_ff_dim"],
        block_auto_adjust_ff_dim=raw.get("block_auto_adjust_ff_dim", True),
        block_ffn_dim_multiplier=raw.get("block_ffn_dim_multiplier", 1.0),
        block_multiple_of=raw.get("block_multiple_of", 256),
        norm_eps=raw.get("norm_eps", 1e-5),
        tie_embedding=raw.get("tie_embedding", True),
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        conv_L_cache=raw.get("conv_L_cache", 3),
        conv_bias=raw.get("conv_bias", False),
        rope_theta=raw.get("rope_theta", 1e6),
        max_position_embeddings=raw.get("max_position_embeddings", 128000),
        layer_types=list(layer_types),
    )


# HF feed_forward uses w1/w3/w2; the MLP primitive uses gate/up/down.
_MLP_KEY_MAP = {".feed_forward.w1.": ".feed_forward.gate_proj.",
                ".feed_forward.w3.": ".feed_forward.up_proj.",
                ".feed_forward.w2.": ".feed_forward.down_proj."}


def lfm2_from_hf(
    huggingface_model_id: str,
    target_dtype: torch.dtype = torch.float16,
    stateful: bool = True,
    fp32_attn_proj: bool = True,
):
    """Load LFM2/LFM2.5 from the HF checkpoint into the authored module.

    Weights load straight from safetensors (keys match HF except the MLP
    w1/w3/w2 -> gate/up/down rename); config parses from raw config.json.
    With ``fp32_attn_proj`` (default) the attention q/k/v/out projection
    weights stay fp32 on an fp16 load — required for GPU-delegate exactness
    (see IMPORTANT #2 in the module header).
    """
    import glob
    import json
    import os
    import re

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    model_dir = snapshot_download(
        huggingface_model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )
    with open(os.path.join(model_dir, "config.json")) as f:
        config = lfm2_config_from_dict(json.load(f))

    cls = Lfm2ForCausalLMStateful if stateful else Lfm2ForCausalLM
    with torch.device("meta"):
        model = cls(config)
    model.to(dtype=target_dtype)

    attn_proj_re = re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.weight$")
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")
    sd: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                local = key
                for old, new in _MLP_KEY_MAP.items():
                    if old in local:
                        local = local.replace(old, new)
                        break
                tensor = f.get_tensor(key)
                dtype = target_dtype
                if fp32_attn_proj and attn_proj_re.search(local):
                    dtype = torch.float32
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype)
                sd[local] = tensor

    model.load_state_dict(sd, assign=True, strict=False)
    if config.tie_embedding:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.reset_buffers()

    meta_params = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_params:
        raise RuntimeError(f"Parameters not loaded: {meta_params}")
    model.eval()
    return model
