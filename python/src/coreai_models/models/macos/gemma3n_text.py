# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma 3n (E2B / E4B) text-decoder re-authored for the Core AI authoring path.

Text decoder of ``Gemma3nForConditionalGeneration`` (HF ``model_type`` ``gemma3n`` /
text ``gemma3n_text``). Authored with coreai_models primitives so it lowers cleanly
through coreai-torch, and is numerically faithful to HF eager (parity gate: 18/18
argmax, cos~1.0 vs transformers fp32).

Gemma 3n signature features vs a vanilla Gemma decoder (all handled here):
  * AltUp (Alternating Updates): the hidden state is a stack of ``altup_num_inputs``
    (=4) parallel streams. Each layer ``predict``s the streams, runs the transformer
    block on the active stream, then ``correct``s all streams from the block output.
    Streams are initialised by ``altup_projections`` (magnitude-matched) at the model
    input and merged by ``altup_unembed_projections`` + mean at the output.
  * LAuReL (Learned Augmented Residual Layer): a low-rank (rank 64) learned residual
    added to the attention path, gated by 1/sqrt(2).
  * Activation-sparse MLP: the first 10 layers apply a gaussian-topk mask to the gate
    activations (95% sparsity); the cutoff multiplier ``Normal(0,1).icdf(sparsity)``
    is a per-layer constant, precomputed (no torch.distributions in the trace).
  * Per-head Q/K RMSNorm (weighted) + scale-free V RMSNorm, applied before RoPE.
  * Dual RoPE: sliding layers use ``rope_local_base_freq`` (1e4), full layers use
    ``rope_theta`` (1e6); both rotate the full 256-dim head (no proportional/NoPE).
  * KV-sharing: the last ``num_kv_shared_layers`` (=10) layers reuse the K/V produced
    by the last non-shared layer of the SAME attention type (the "producer").
  * Per-Layer Embeddings (PLE): a learned per-layer skip injected after AltUp correct,
    gated by the per-layer embedding for that layer.
  * Final logit softcapping: ``tanh(logits / 30) * 30``.

Gemma 3n RMSNorm multiplies by ``weight`` directly (``_norm(x) * weight``), so we use
the plain ``RMSNorm`` primitive (NOT ``RMSNormPlusOne``); V-norm / router-input use a
scale-free variant.

The transformer core (:meth:`Gemma3nTextModel.decode`) takes pre-computed
``inputs_embeds`` and ``per_layer_inputs`` so the giant embedding / PLE gather tables
stay out of the converted graph (CPU front-end on device), exactly like Gemma 4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from typing_extensions import Self

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA

SLIDING = "sliding_attention"
FULL = "full_attention"


@dataclass
class Gemma3nTextConfig:
    """Lightweight Gemma 3n text-decoder config (built from HF ``text_config``)."""

    vocab_size: int = 262400
    vocab_size_per_layer_input: int = 262144
    hidden_size: int = 2048
    hidden_size_per_layer_input: int = 256
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    intermediate_size: list[int] = field(default_factory=lambda: [8192] * 30)
    num_kv_shared_layers: int = 10
    rms_norm_eps: float = 1e-6
    sliding_window: int = 512
    rope_theta: float = 1000000.0           # global / full attention
    rope_local_base_freq: float = 10000.0   # local / sliding attention
    final_logit_softcapping: float | None = 30.0
    altup_num_inputs: int = 4
    altup_active_idx: int = 0
    altup_coef_clip: float | None = 120.0
    altup_correct_scale: bool = True
    laurel_rank: int = 64
    activation_sparsity_pattern: list[float] = field(default_factory=lambda: [0.95] * 10 + [0.0] * 20)
    hidden_activation: str = "gelu_pytorch_tanh"
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 32768
    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layer_types:
            self.layer_types = [
                FULL if (i + 1) % 5 == 0 else SLIDING for i in range(self.num_hidden_layers)
            ]
        if isinstance(self.intermediate_size, int):
            self.intermediate_size = [self.intermediate_size] * self.num_hidden_layers

    @classmethod
    def from_hf_config(cls, d: dict) -> Gemma3nTextConfig:
        if "text_config" in d:
            d = d["text_config"]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    # ---- derived layout helpers ----
    @property
    def first_kv_shared_layer_idx(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    def is_full(self, i: int) -> bool:
        return self.layer_types[i] == FULL

    def is_kv_shared(self, i: int) -> bool:
        kv_start = self.first_kv_shared_layer_idx
        return kv_start > 0 and i >= kv_start

    def intermediate_of(self, i: int) -> int:
        return self.intermediate_size[i]

    def producer_idx(self, layer_type: str) -> int:
        prev = self.layer_types[: self.first_kv_shared_layer_idx]
        if layer_type not in prev:
            raise ValueError(f"No non-shared {layer_type} layer to produce shared KV.")
        return len(prev) - 1 - prev[::-1].index(layer_type)

    def stateful_routing(self) -> tuple[list[tuple[bool, int, bool]], int, int]:
        """Per-layer KV-cache routing for the stateful decode (mirrors Gemma 4).

        head_dim is uniform (256), but sliding and full layers keep separate caches
        (different masks/windows). Only non-shared layers own a slot; KV-shared layers
        read the producer slot of their type. Returns ``(route, n_sliding, n_full)``.
        """
        sliding_ns = [i for i in range(self.num_hidden_layers)
                      if not self.is_full(i) and not self.is_kv_shared(i)]
        full_ns = [i for i in range(self.num_hidden_layers)
                   if self.is_full(i) and not self.is_kv_shared(i)]
        slot_s = {i: k for k, i in enumerate(sliding_ns)}
        slot_f = {i: k for k, i in enumerate(full_ns)}
        prod_s = len(sliding_ns) - 1
        prod_f = len(full_ns) - 1
        route: list[tuple[bool, int, bool]] = []
        for i in range(self.num_hidden_layers):
            full = self.is_full(i)
            if not self.is_kv_shared(i):
                route.append((full, (slot_f if full else slot_s)[i], True))
            else:
                route.append((full, prod_f if full else prod_s, False))
        return route, len(sliding_ns), len(full_ns)


def _inv_freq(theta: float, head_dim: int) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def _truncated_kv_shared(cfg: Gemma3nTextConfig, num_layers: int) -> int:
    first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers
    return max(0, num_layers - first_shared)


class HighPrecLinear(nn.Linear):
    """Marker subclass for tiny, quant-sensitive linears (AltUp coefficient/router
    matrices, LAuReL low-rank projections). Excluded from INT4 via the export's
    ``module_type_configs`` (kept fp16) — int4-quantizing these 4-/64-wide matrices
    corrupts the AltUp stream mixing (hidden maxdiff ~13 vs ~1e-4 when kept fp16)."""


class ScaleFreeRMSNorm(nn.Module):
    """RMSNorm with no learnable scale (HF ``Gemma3nRMSNorm(with_scale=False)``)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        return x.to(dtype)


class Laurel(nn.Module):
    """Learned Augmented Residual Layer (low-rank learned residual)."""

    def __init__(self, cfg: Gemma3nTextConfig) -> None:
        super().__init__()
        self.linear_left = HighPrecLinear(cfg.hidden_size, cfg.laurel_rank, bias=False)
        self.linear_right = HighPrecLinear(cfg.laurel_rank, cfg.hidden_size, bias=False)
        self.post_laurel_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.post_laurel_norm(self.linear_right(self.linear_left(x)))


class MLP(nn.Module):
    """Gemma 3n MLP with optional gaussian-topk activation sparsity."""

    def __init__(self, cfg: Gemma3nTextConfig, layer_idx: int) -> None:
        super().__init__()
        inter = cfg.intermediate_of(layer_idx)
        self.gate_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, cfg.hidden_size, bias=False)
        sparsity = cfg.activation_sparsity_pattern[layer_idx]
        self.sparsity = sparsity
        # Normal(0,1).icdf(sparsity) is a constant -> precompute (no torch.distributions in trace).
        self.std_mult = (
            float(torch.distributions.normal.Normal(0.0, 1.0).icdf(torch.tensor(float(sparsity))))
            if sparsity > 0.0 else 0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        if self.sparsity > 0.0:
            # gaussian-topk cutoff = mean + std * Normal.icdf(sparsity). Compute the
            # population std manually (torch.std -> aten.var.correction, unsupported).
            mean = gate.mean(-1, keepdim=True)
            std = torch.sqrt((gate - mean).pow(2).mean(-1, keepdim=True))
            gate = nn.functional.relu(gate - (mean + std * self.std_mult))
        gate = nn.functional.gelu(gate, approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


class AltUp(nn.Module):
    """Alternating Updates: predict/correct over ``altup_num_inputs`` parallel streams."""

    def __init__(self, cfg: Gemma3nTextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        P = cfg.altup_num_inputs
        self.correct_output_scale = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.correction_coefs = HighPrecLinear(P, P, bias=False)
        self.prediction_coefs = HighPrecLinear(P, P * P, bias=False)
        self.modality_router = HighPrecLinear(cfg.hidden_size, P, bias=False)
        self.router_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.router_input_scale = cfg.hidden_size ** -1.0

    def _modalities(self, x: torch.Tensor) -> torch.Tensor:
        routed = self.modality_router(self.router_norm(x) * self.router_input_scale)
        return torch.tanh(routed.float()).type_as(x)

    def predict(self, h: torch.Tensor) -> torch.Tensor:  # h: [P, B, T, H]
        P = self.cfg.altup_num_inputs
        mod = self._modalities(h[self.cfg.altup_active_idx])
        all_coefs = self.prediction_coefs(mod).reshape(*mod.shape[:-1], P, P).permute(0, 1, 3, 2)
        pred = torch.matmul(h.permute(1, 2, 3, 0), all_coefs).permute(3, 0, 1, 2)
        return (pred + h).contiguous().type_as(h)

    def correct(self, pred: torch.Tensor, activated: torch.Tensor) -> torch.Tensor:
        P = self.cfg.altup_num_inputs
        mod = self._modalities(activated)
        innov = (activated - pred[self.cfg.altup_active_idx]).repeat(P, 1, 1, 1)
        all_coefs = (self.correction_coefs(mod) + 1.0).permute(2, 0, 1).unsqueeze(-1)
        return (innov * all_coefs + pred).contiguous().type_as(activated)

    def scale_corrected_output(self, x: torch.Tensor) -> torch.Tensor:
        return (x.type_as(self.correct_output_scale) * self.correct_output_scale).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: Gemma3nTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.is_full = cfg.is_full(layer_idx)
        self.is_shared = cfg.is_kv_shared(layer_idx)
        self.is_producer = (not self.is_shared) and (layer_idx == cfg.producer_idx(self.layer_type))

        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = hd = cfg.head_dim

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, cfg.hidden_size, bias=False)

        self.q_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.v_norm = ScaleFreeRMSNorm(eps=cfg.rms_norm_eps)

        theta = cfg.rope_theta if self.is_full else cfg.rope_local_base_freq
        self.inv_freq = _inv_freq(theta, hd).float()
        self.rope = RoPE(scale=1.0)
        self.sdpa = SDPA(
            scale=1.0,                      # Gemma 3n attention scale is 1.0
            is_causal=True,
            window_size=cfg.sliding_window if not self.is_full else 0,
        )
        self.sliding_window = cfg.sliding_window

    def _repeat_kv(self, t: torch.Tensor) -> torch.Tensor:
        return t.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else t

    def _project_kv(self, x: torch.Tensor, position_ids: torch.Tensor):
        b, s, _ = x.shape
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_norm(k)
        k = self.rope(k, position_ids=position_ids, freqs=self.inv_freq)
        v = self.v_norm(v)
        return k, v

    def forward(self, x, position_ids, shared_kv):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        q = self.rope(q, position_ids=position_ids, freqs=self.inv_freq)
        if self.is_shared and shared_kv is not None:
            k, v = shared_kv
        else:
            k, v = self._project_kv(x, position_ids)
        out = self.sdpa(query=q, key=self._repeat_kv(k), value=self._repeat_kv(v))
        out = out.permute(0, 2, 1, 3).reshape(b, s, self.n_heads * self.head_dim)
        out = self.o_proj(out)
        produced_kv = (k, v) if self.is_producer else None
        return out, produced_kv

    def forward_stateful(self, x, position_ids, kv: KVCache, slot: int, write: bool):
        b, s, _ = x.shape
        seq_len = position_ids.shape[-1]
        torch._check_is_size(s)
        torch._check_is_size(seq_len)
        offset = seq_len - s
        torch._check_is_size(offset)
        rope_pos = position_ids.narrow(-1, offset, s)

        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        q = self.rope(q, position_ids=rope_pos, freqs=self.inv_freq)
        if write:
            k, v = self._project_kv(x, rope_pos)
            k, v = kv.update_and_fetch(slot, offset, k, v, seq_len=seq_len, query_len=s)
        else:
            k = kv._k_cache.narrow(0, slot, 1).narrow(-2, 0, seq_len).squeeze(0)
            v = kv._v_cache.narrow(0, slot, 1).narrow(-2, 0, seq_len).squeeze(0)
        out = self.sdpa(query=q, key=self._repeat_kv(k), value=self._repeat_kv(v))
        out = out.permute(0, 2, 1, 3).reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Gemma3nTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.is_shared = cfg.is_kv_shared(layer_idx)
        self.is_producer = (not self.is_shared) and (layer_idx == cfg.producer_idx(self.layer_type))
        h = cfg.hidden_size
        eps = cfg.rms_norm_eps

        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(cfg, layer_idx)
        self.input_layernorm = RMSNorm(h, eps=eps)
        self.post_attention_layernorm = RMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.altup = AltUp(cfg)
        self.laurel = Laurel(cfg)
        ld = cfg.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Linear(h, ld, bias=False)
        self.per_layer_projection = nn.Linear(ld, h, bias=False)
        self.post_per_layer_input_norm = RMSNorm(h, eps=eps)

    def _block(self, hidden, position_ids, per_layer_input, attn_callable):
        """Shared AltUp/LAuReL/PLE flow; ``attn_callable(normed)`` runs the attention."""
        a = self.cfg.altup_active_idx
        pred = self.altup.predict(hidden)
        active = pred[a]
        normed = self.input_layernorm(active)
        laurel_out = self.laurel(normed)
        attn, produced = attn_callable(normed)
        attn = self.post_attention_layernorm(attn)
        attn_gated = active + attn
        attn_laurel = (attn_gated + laurel_out) / math.sqrt(2)
        ffw = self.mlp(self.pre_feedforward_layernorm(attn_laurel))
        ffw = self.post_feedforward_layernorm(ffw)
        attn_ffw_laurel = attn_laurel + ffw
        corrected = self.altup.correct(pred, attn_ffw_laurel)
        first = corrected[a].clone()
        if self.cfg.altup_correct_scale:
            first = self.altup.scale_corrected_output(first)
        first = self.per_layer_input_gate(first)
        first = nn.functional.gelu(first, approximate="tanh")
        first = first * per_layer_input
        first = self.per_layer_projection(first)
        first = self.post_per_layer_input_norm(first)
        corrected = torch.cat([corrected[:1], corrected[1:] + first], dim=0)
        return corrected, produced

    def forward(self, hidden, position_ids, per_layer_input, shared_kv):
        return self._block(
            hidden, position_ids, per_layer_input,
            lambda normed: self.self_attn(normed, position_ids, shared_kv),
        )

    def forward_stateful(self, hidden, position_ids, per_layer_input, kv, slot, write):
        out, _ = self._block(
            hidden, position_ids, per_layer_input,
            lambda normed: (self.self_attn.forward_stateful(normed, position_ids, kv, slot, write), None),
        )
        return out


class ScaledEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float) -> None:
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * torch.tensor(self.embed_scale, dtype=self.weight.dtype)


def _rescale_stream(cur: torch.Tensor, target_magnitude: torch.Tensor) -> torch.Tensor:
    eps = torch.tensor(1e-5, dtype=torch.float32)
    new_mag = torch.sqrt(torch.maximum(cur.float().pow(2).mean(-1, keepdim=True), eps))
    return (cur * target_magnitude / new_mag).type_as(cur)


class Gemma3nTextModel(nn.Module):
    def __init__(self, cfg: Gemma3nTextConfig) -> None:
        super().__init__()
        self.config = cfg
        h = cfg.hidden_size
        ld = cfg.hidden_size_per_layer_input
        P = cfg.altup_num_inputs

        self.embed_tokens = ScaledEmbedding(cfg.vocab_size, h, embed_scale=h ** 0.5)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(h, eps=cfg.rms_norm_eps)

        # PLE front-end (kept out of the convertible decode core).
        self.embed_tokens_per_layer = ScaledEmbedding(
            cfg.vocab_size_per_layer_input, cfg.num_hidden_layers * ld, embed_scale=ld ** 0.5
        )
        self.per_layer_model_projection = nn.Linear(h, cfg.num_hidden_layers * ld, bias=False)
        self.per_layer_model_projection_scale = h ** -0.5
        self.per_layer_input_scale = 2.0 ** -0.5
        self.per_layer_projection_norm = RMSNorm(ld, eps=cfg.rms_norm_eps)

        # Stream init/combine projections set per-stream magnitudes — keep fp16 (HighPrec).
        self.altup_projections = nn.ModuleList([HighPrecLinear(h, h, bias=False) for _ in range(1, P)])
        self.altup_unembed_projections = nn.ModuleList([HighPrecLinear(h, h, bias=False) for _ in range(1, P)])

    def compute_per_layer_inputs(self, input_ids, inputs_embeds):
        cfg = self.config
        ld = cfg.hidden_size_per_layer_input
        b, s = input_ids.shape
        tokens = self.embed_tokens_per_layer(input_ids).reshape(b, s, cfg.num_hidden_layers, ld)
        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        proj = proj.reshape(b, s, cfg.num_hidden_layers, ld)
        proj = self.per_layer_projection_norm(proj)
        return (proj + tokens) * self.per_layer_input_scale

    def _init_streams(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        target = inputs_embeds.float().pow(2).mean(-1, keepdim=True).pow(0.5)
        streams = [inputs_embeds]
        for i in range(1, self.config.altup_num_inputs):
            streams.append(_rescale_stream(self.altup_projections[i - 1](inputs_embeds), target))
        return torch.stack(streams, dim=0)

    def _merge_streams(self, hidden: torch.Tensor) -> torch.Tensor:
        target = hidden[0].float().pow(2).mean(-1, keepdim=True).pow(0.5)
        streams = [hidden[0]]
        for i in range(1, self.config.altup_num_inputs):
            streams.append(_rescale_stream(self.altup_unembed_projections[i - 1](hidden[i]), target))
        return self.norm(torch.stack(streams, dim=0).mean(dim=0))

    def decode(self, inputs_embeds, per_layer_inputs, position_ids):
        hidden = self._init_streams(inputs_embeds)
        producer_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            shared = producer_kv.get(layer.layer_type) if layer.is_shared else None
            hidden, produced = layer(hidden, position_ids, per_layer_inputs[:, :, i, :], shared)
            if layer.is_producer and produced is not None:
                producer_kv[layer.layer_type] = produced
        return self._merge_streams(hidden)

    def decode_stateful(self, inputs_embeds, per_layer_inputs, position_ids, kv_sliding, kv_full):
        route, _, _ = self.config.stateful_routing()
        hidden = self._init_streams(inputs_embeds)
        for i, layer in enumerate(self.layers):
            is_full, slot, write = route[i]
            kv = kv_full if is_full else kv_sliding
            hidden = layer.forward_stateful(
                hidden, position_ids, per_layer_inputs[:, :, i, :], kv, slot, write
            )
        return self._merge_streams(hidden)

    def forward(self, input_ids, position_ids):
        inputs_embeds = self.embed_tokens(input_ids)
        per_layer_inputs = self.compute_per_layer_inputs(input_ids, inputs_embeds)
        return self.decode(inputs_embeds, per_layer_inputs, position_ids)


class Gemma3nForCausalLM(nn.Module):
    """Gemma 3n text decoder with a (tied) LM head and final logit softcapping."""

    def __init__(self, cfg: Gemma3nTextConfig) -> None:
        super().__init__()
        self.config = cfg
        self.model = Gemma3nTextModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        sc = self.config.final_logit_softcapping
        if sc is not None:
            logits = torch.tanh(logits / sc) * sc
        return logits

    def forward(self, input_ids, position_ids):
        return self.logits_from_hidden(self.model(input_ids, position_ids))

    @classmethod
    def from_local(cls: type[Self], model_dir: str, *, num_layers: int | None = None,
                   num_kv_shared: int | None = None, dtype: torch.dtype = torch.float32) -> Self:
        import json
        from pathlib import Path

        d = json.loads((Path(model_dir) / "config.json").read_text())
        cfg = Gemma3nTextConfig.from_hf_config(d)
        if num_layers is not None:
            cfg.num_kv_shared_layers = _truncated_kv_shared(cfg, num_layers)
            cfg.num_hidden_layers = num_layers
            cfg.layer_types = cfg.layer_types[:num_layers]
            cfg.intermediate_size = cfg.intermediate_size[:num_layers]
            cfg.activation_sparsity_pattern = cfg.activation_sparsity_pattern[:num_layers]
        if num_kv_shared is not None:
            cfg.num_kv_shared_layers = num_kv_shared
        model = cls(cfg).to(dtype).eval()
        _load_hf_weights(model, model_dir, dtype=dtype)
        return model


def _load_hf_weights(model: Gemma3nForCausalLM, model_dir: str, *, dtype: torch.dtype = torch.float32) -> None:
    import json
    from pathlib import Path

    from safetensors import safe_open

    prefix = "model.language_model."
    n_layers = model.config.num_hidden_layers
    want = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}

    sd: dict[str, torch.Tensor] = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as h:
            for k in h.keys():  # noqa: SIM118
                if not k.startswith(prefix):
                    continue
                local = "model." + k[len(prefix):]
                if ".layers." in local:
                    li = int(local.split(".layers.")[1].split(".")[0])
                    if li >= n_layers:
                        continue
                if local in want:
                    t = h.get_tensor(k)
                    sd[local] = t.to(dtype) if t.dtype.is_floating_point else t

    # Slice per-layer-embedding tables when the decoder is truncated.
    ple_cols = model.config.num_hidden_layers * model.config.hidden_size_per_layer_input
    if (k := "model.embed_tokens_per_layer.weight") in sd and sd[k].shape[1] != ple_cols:
        sd[k] = sd[k][:, :ple_cols].contiguous()
    if (k := "model.per_layer_model_projection.weight") in sd and sd[k].shape[0] != ple_cols:
        sd[k] = sd[k][:ple_cols, :].contiguous()

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    real_missing = [m for m in missing if "inv_freq" not in m and "lm_head" not in m]
    if real_missing:
        raise RuntimeError(f"missing weights: {real_missing[:10]} (+{len(real_missing) - 10} more)")


# ---------------------------------------------------------------------------
# Stateful decode core (dual KV caches: sliding + full, both head_dim 256).
# ---------------------------------------------------------------------------

DECODE_STATE_NAMES = ("slidingKeyCache", "slidingValueCache", "fullKeyCache", "fullValueCache")


def build_decode_state(cfg: Gemma3nTextConfig, max_seq_len: int, dtype: torch.dtype = torch.float32) -> dict:
    _, n_sliding, n_full = cfg.stateful_routing()
    nkv, hd = cfg.num_key_value_heads, cfg.head_dim
    return {
        "sliding_k": torch.zeros(n_sliding, 1, nkv, max_seq_len, hd, dtype=dtype),
        "sliding_v": torch.zeros(n_sliding, 1, nkv, max_seq_len, hd, dtype=dtype),
        "full_k": torch.zeros(n_full, 1, nkv, max_seq_len, hd, dtype=dtype),
        "full_v": torch.zeros(n_full, 1, nkv, max_seq_len, hd, dtype=dtype),
    }


class Gemma3nDecodeStateful(nn.Module):
    """Stateful decode core: ``(inputs_embeds, per_layer_inputs, position_ids,
    sliding_k, sliding_v, full_k, full_v) -> hidden``."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, text_model: Gemma3nTextModel) -> None:
        super().__init__()
        self.text_model = text_model

    def forward(self, inputs_embeds, per_layer_inputs, position_ids, sliding_k, sliding_v, full_k, full_v):
        return self.text_model.decode_stateful(
            inputs_embeds, per_layer_inputs, position_ids,
            KVCache(sliding_k, sliding_v), KVCache(full_k, full_v),
        )

    def build_macos_export_spec(self, target_dtype, max_context_length, query_len, offset, trace_kv_len) -> dict:
        cfg = self.text_model.config
        b, h, ld = 1, cfg.hidden_size, cfg.hidden_size_per_layer_input
        seq = query_len + offset
        state = build_decode_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)
        reference_inputs = {
            "inputs_embeds": torch.zeros(b, query_len, h, dtype=target_dtype),
            "per_layer_inputs": torch.zeros(b, query_len, cfg.num_hidden_layers, ld, dtype=target_dtype),
            "position_ids": torch.arange(seq, dtype=torch.int32).unsqueeze(0).expand(b, seq),
            **state,
        }
        seq_ids = torch.export.Dim("seq_ids", max=max_context_length - 2)
        seq_pos = torch.export.Dim("seq_pos", min=query_len, max=max_context_length - 1)
        sd = KVCache.seq_len_dim()
        cache_dim = {n: {sd: torch.export.Dim(f"{n}_seq", min=trace_kv_len, max=max_context_length)}
                     for n in ("sliding_k", "sliding_v", "full_k", "full_v")}
        dynamic_shapes = {
            "inputs_embeds": {1: seq_ids},
            "per_layer_inputs": {1: seq_ids},
            "position_ids": {1: seq_pos},
            **cache_dim,
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("inputs_embeds", "per_layer_inputs", "position_ids"),
            "output_names": ("hidden",),
            "state_names": DECODE_STATE_NAMES,
        }


class Gemma3nHead(nn.Module):
    """Tied LM head + final logit softcap, as a standalone export unit."""

    def __init__(self, cfg: Gemma3nTextConfig) -> None:
        super().__init__()
        self.config = cfg
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.softcap = cfg.final_logit_softcapping

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits
