# Community port — NOT an Apple model. BSD-3-Clause (see LICENSE).

"""Gemma 4 (E2B / E4B) text-decoder re-authored for the Core AI authoring path.

This is the text decoder of ``Gemma4ForConditionalGeneration`` (HF ``model_type``
``gemma4`` / text ``gemma4_text``). It is authored with coreai_models primitives so
it lowers cleanly through coreai-torch, and is numerically faithful to HF eager.

Gemma 4 signature features vs Gemma 3 (all handled here):
  * Dual head_dim: sliding_attention head_dim=256, full_attention global_head_dim=512.
  * Attention scale = 1.0 (NOT 1/sqrt(d)); QK-norm keeps magnitudes bounded.
  * Per-head Q/K RMSNorm (weighted) and a scale-free V RMSNorm, applied before RoPE.
  * KV-sharing: the last ``num_kv_shared_layers`` layers reuse the K/V produced by
    the last non-shared layer of the SAME attention type (the "producer"), rather
    than computing their own — this is the cache-path behaviour and is what makes
    a single-pass (prefill) forward match HF's ``use_cache=True`` output.
  * Double-wide MLP (2x intermediate) on the KV-shared layers.
  * Per-Layer Embeddings (PLE): a learned per-layer skip-connection injected after
    the FFN of every layer, plus a learned per-layer scalar.
  * Dual RoPE: sliding uses theta=1e4 over the full 256-dim head; full uses
    theta=1e6 "proportional" RoPE — only the first 64 freq pairs rotate (the
    remaining 192 use zero frequency = identity), via a custom ``freqs`` vector.
  * Final logit softcapping: ``tanh(logits / 30) * 30``.

Gemma 4 RMSNorm multiplies by ``weight`` directly (checkpoint weights are centred
near 1, not 0) — so we use the plain ``RMSNorm`` primitive, NOT ``RMSNormPlusOne``.

The transformer core (:meth:`Gemma4TextModel.decode`) takes pre-computed
``inputs_embeds`` and ``per_layer_inputs`` so the giant embedding / PLE gather
tables stay out of the converted graph (they live on the CPU front-end on device).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from typing_extensions import Self

from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA

SLIDING = "sliding_attention"
FULL = "full_attention"


@dataclass
class Gemma4TextConfig:
    """Lightweight Gemma 4 text-decoder config (built from HF ``text_config``).

    Kept independent of ``transformers.models.gemma4`` so the model imports even
    where the installed transformers predates Gemma 4.
    """

    vocab_size: int = 262144
    hidden_size: int = 1536
    num_hidden_layers: int = 35
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    intermediate_size: int = 6144
    num_kv_shared_layers: int = 20
    use_double_wide_mlp: bool = True
    hidden_size_per_layer_input: int = 256
    vocab_size_per_layer_input: int = 262144
    rms_norm_eps: float = 1e-6
    sliding_window: int = 512
    final_logit_softcapping: float | None = 30.0
    sliding_rope_theta: float = 10000.0
    full_rope_theta: float = 1000000.0
    full_partial_rotary_factor: float = 0.25
    hidden_activation: str = "gelu_pytorch_tanh"
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 131072
    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layer_types:
            # HF default: every 5th layer (1-indexed) is full_attention.
            self.layer_types = [
                FULL if (i + 1) % 5 == 0 else SLIDING for i in range(self.num_hidden_layers)
            ]

    @classmethod
    def from_hf_config(cls, d: dict) -> Gemma4TextConfig:
        """Build from an HF ``text_config`` dict (config.json)."""
        if "text_config" in d:
            d = d["text_config"]
        rope = d.get("rope_parameters", {}) or {}
        sliding_rope = rope.get(SLIDING, {}) or {}
        full_rope = rope.get(FULL, {}) or {}
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs.update(
            sliding_rope_theta=sliding_rope.get("rope_theta", 10000.0),
            full_rope_theta=full_rope.get("rope_theta", 1000000.0),
            full_partial_rotary_factor=full_rope.get("partial_rotary_factor", 0.25),
        )
        return cls(**kwargs)

    # ---- derived layout helpers ----
    @property
    def first_kv_shared_layer_idx(self) -> int:
        return self.num_hidden_layers - self.num_kv_shared_layers

    def is_full(self, i: int) -> bool:
        return self.layer_types[i] == FULL

    def is_kv_shared(self, i: int) -> bool:
        kv_start = self.first_kv_shared_layer_idx
        return kv_start > 0 and i >= kv_start

    def head_dim_of(self, i: int) -> int:
        return self.global_head_dim if self.is_full(i) else self.head_dim

    def intermediate_of(self, i: int) -> int:
        if self.is_kv_shared(i) and self.use_double_wide_mlp:
            return self.intermediate_size * 2
        return self.intermediate_size

    def producer_idx(self, layer_type: str) -> int:
        """Last NON-shared layer of ``layer_type`` (its K/V feeds the shared ones)."""
        prev = self.layer_types[: self.first_kv_shared_layer_idx]
        if layer_type not in prev:
            raise ValueError(
                f"No non-shared {layer_type} layer to produce shared KV "
                f"(first_kv_shared_layer_idx={self.first_kv_shared_layer_idx}). "
                "Config is degenerate for KV-sharing."
            )
        return len(prev) - 1 - prev[::-1].index(layer_type)

    def stateful_routing(self) -> tuple[list[tuple[bool, int, bool]], int, int]:
        """Per-layer KV-cache routing for the stateful (incremental) decode.

        Because of the dual head_dim there are two separate caches: a SLIDING
        cache (head_dim 256) and a FULL cache (head_dim 512). Only NON-shared
        layers own a cache slot — each writes its K/V at its slot. KV-shared
        layers don't write; they read the producer slot (the last non-shared
        slot of their type), exactly mirroring the stateless prefill sharing.

        Returns ``(route, n_sliding_slots, n_full_slots)`` where ``route[i]`` is
        ``(is_full, slot, writes)`` for layer ``i``.
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


def _sliding_inv_freq(cfg: Gemma4TextConfig) -> torch.Tensor:
    hd = cfg.head_dim
    return 1.0 / (cfg.sliding_rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))


def _full_inv_freq(cfg: Gemma4TextConfig) -> torch.Tensor:
    """Proportional RoPE inv_freq for full-attention heads (HF ``_compute_proportional``).

    ``rope_angles`` real frequencies followed by zeros (identity / NoPE) so the
    returned vector has length ``global_head_dim // 2``.
    """
    hd = cfg.global_head_dim
    rope_angles = int(cfg.full_partial_rotary_factor * hd // 2)
    rotated = 1.0 / (
        cfg.full_rope_theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / hd)
    )
    nope = hd // 2 - rope_angles
    if nope > 0:
        return torch.cat([rotated, torch.zeros(nope, dtype=torch.float32)])
    return rotated


def _truncated_kv_shared(cfg: Gemma4TextConfig, num_layers: int) -> int:
    """KV-shared count when truncating the decoder to ``num_layers`` layers.

    The KV-shared layers carry the double-wide MLP and occupy the checkpoint's
    trailing region ``[N - kv_shared, N)``. Truncation must DROP trailing shared
    layers, not relabel early (single-wide) layers as shared — otherwise the
    re-authored MLP widths won't match the checkpoint. So keep only the shared
    layers whose real index is still ``< num_layers``. Call BEFORE mutating
    ``cfg.num_hidden_layers`` (it reads the pre-truncation layout).
    """
    first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers
    return max(0, num_layers - first_shared)


class ScaleFreeRMSNorm(nn.Module):
    """RMSNorm with no learnable scale (HF ``Gemma4RMSNorm(with_scale=False)``)."""

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        return x.to(dtype)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, cfg: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.is_full = cfg.is_full(layer_idx)
        self.is_shared = cfg.is_kv_shared(layer_idx)
        self.is_producer = (not self.is_shared) and (
            layer_idx == cfg.producer_idx(self.layer_type)
        )

        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = hd = cfg.head_dim_of(layer_idx)

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * hd, bias=False)
        self.o_proj = nn.Linear(self.n_heads * hd, cfg.hidden_size, bias=False)

        self.q_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.v_norm = ScaleFreeRMSNorm(eps=cfg.rms_norm_eps)

        # RoPE with a precomputed (type-specific) inv_freq passed as ``freqs``.
        # Kept as a plain fp32 attribute (NOT a buffer) so model.half() can't cast
        # it: the composite RoPE requires fp32 freqs, and the small frequencies
        # (~1e-5) would underflow to zero in fp16, corrupting the rotation.
        self.inv_freq = (_full_inv_freq(cfg) if self.is_full else _sliding_inv_freq(cfg)).float()
        self.rope = RoPE(scale=1.0)

        # Gemma 4 attention scale is 1.0 (not 1/sqrt(head_dim)).
        self.sdpa = SDPA(
            scale=1.0,
            is_causal=True,
            window_size=cfg.sliding_window if not self.is_full else 0,
        )
        # Sliding-window width W, used by the ring-buffer stateful path.
        self.sliding_window = cfg.sliding_window

    def _project_kv(self, x: torch.Tensor, position_ids: torch.Tensor):
        b, s, _ = x.shape
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_norm(k)
        k = self.rope(k, position_ids=position_ids, freqs=self.inv_freq)
        v = self.v_norm(v)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        q = self.rope(q, position_ids=position_ids, freqs=self.inv_freq)

        if self.is_shared and shared_kv is not None:
            k, v = shared_kv
        else:
            k, v = self._project_kv(x, position_ids)

        out = self.sdpa(query=q, key=k, value=v)
        out = out.permute(0, 2, 1, 3).reshape(b, s, self.n_heads * self.head_dim)
        out = self.o_proj(out)

        produced_kv = (k, v) if self.is_producer else None
        return out, produced_kv

    def forward_stateful(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        kv: KVCache,
        slot: int,
        write: bool,
        ring: bool = False,
    ) -> torch.Tensor:
        """Incremental attention against a persistent KV cache.

        ``position_ids`` are the full positions [0..seq_len-1]; the query block is
        the trailing ``query_len`` of them (offset = seq_len - query_len). Non-shared
        layers (``write=True``) append their K/V at ``slot`` and read it back; KV-shared
        layers read the producer ``slot`` (already written this step) without writing.

        When ``ring`` is set and this is a SLIDING layer, the cache is a ring buffer of
        width ``sliding_window`` (W): K/V are written at ``offset % W`` and the whole
        W-ring is read back. RoPE is baked into K at write time, so the slot order does
        not affect the attention scores — an explicit position-derived mask
        (:meth:`_ring_mask`) declares which ring slots hold an already-written, causal,
        in-window position. Full-attention layers always use the linear cache (global
        attention needs every position). The ring assumes the query block does not wrap
        the buffer (``offset % W + query_len <= W``): always true for one-token decode
        and for a from-zero prefill of <= W tokens; see the module note for the
        chunked-prefill-at-offset caveat.
        """
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

        if ring and not self.is_full:
            w = self.sliding_window
            if write:
                k, v = self._project_kv(x, rope_pos)
                # Ring write at offset % W; read the whole W-ring back.
                k, v = kv.update_and_fetch(slot, offset % w, k, v, seq_len=w, query_len=s)
            else:
                k = kv._k_cache.narrow(0, slot, 1).narrow(-2, 0, w).squeeze(0)
                v = kv._v_cache.narrow(0, slot, 1).narrow(-2, 0, w).squeeze(0)
            attn_mask = self._ring_mask(rope_pos, seq_len, s, w, x.device)
            out = self.sdpa(query=q, key=k, value=v, attn_mask=attn_mask)
        else:
            if write:
                k, v = self._project_kv(x, rope_pos)
                k, v = kv.update_and_fetch(slot, offset, k, v, seq_len=seq_len, query_len=s)
            else:
                # Read the producer's full history (it ran earlier this step).
                k = kv._k_cache.narrow(0, slot, 1).narrow(-2, 0, seq_len).squeeze(0)
                v = kv._v_cache.narrow(0, slot, 1).narrow(-2, 0, seq_len).squeeze(0)
            out = self.sdpa(query=q, key=k, value=v)

        out = out.permute(0, 2, 1, 3).reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(out)

    def _ring_mask(
        self, rope_pos: torch.Tensor, seq_len: int, s: int, w: int, device: torch.device
    ) -> torch.Tensor:
        """Boolean attention mask for the sliding ring buffer ``(1, 1, s, W)``.

        Ring slot ``r`` holds the most-recent written position with ``pos % W == r``,
        i.e. ``slot_pos[r] = Pmax - ((Pmax - r) % W)`` with ``Pmax = seq_len - 1``.
        A query at position ``p`` attends to slot ``r`` iff that slot's position is
        (a) actually written (``>= 0``), (b) causal (``<= p``) and (c) inside the
        window (``> p - W``). The mask broadcasts over heads.

        ``(Pmax - r) % W`` is computed without a tensor ``remainder`` (unsupported by
        coreai-torch): the newest slot is ``ring_now = Pmax % W`` (a scalar symint,
        which lowers as a symbolic-int op), and the per-slot age wraps via a single
        ``where`` over the contiguous range ``ring_now - r``.
        """
        pmax = seq_len - 1
        ring_now = pmax % w                                      # scalar (newest slot)
        r = torch.arange(w, device=device, dtype=torch.int32)
        diff = ring_now - r                                      # (W,) in [-(W-1), W-1]
        age = torch.where(diff >= 0, diff, diff + w)             # (W,) = (Pmax - r) % W
        slot_pos = pmax - age                                    # (W,)
        written = (slot_pos >= 0).reshape(1, w)                  # (1, W)
        qcol = rope_pos.reshape(s, 1)                            # (s, 1)
        srow = slot_pos.reshape(1, w)                            # (1, W)
        valid = written & (srow <= qcol) & (srow > qcol - w)    # (s, W)
        return valid.reshape(1, 1, s, w)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Gemma4TextConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.is_shared = cfg.is_kv_shared(layer_idx)
        self.is_producer = (not self.is_shared) and (
            layer_idx == cfg.producer_idx(self.layer_type)
        )
        h = cfg.hidden_size
        eps = cfg.rms_norm_eps

        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(h, cfg.intermediate_of(layer_idx))

        self.input_layernorm = RMSNorm(h, eps=eps)
        self.post_attention_layernorm = RMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps=eps)

        # Per-Layer Embedding (PLE) injection.
        self.ple_dim = cfg.hidden_size_per_layer_input
        self.per_layer_input_gate = nn.Linear(h, self.ple_dim, bias=False)
        self.per_layer_projection = nn.Linear(self.ple_dim, h, bias=False)
        self.post_per_layer_input_norm = RMSNorm(h, eps=eps)
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        per_layer_input: torch.Tensor,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None,
    ):
        residual = x
        x = self.input_layernorm(x)
        x, produced_kv = self.self_attn(x, position_ids, shared_kv)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        # PLE: gated per-layer skip connection.
        residual = x
        x = self.per_layer_input_gate(x)
        x = nn.functional.gelu(x, approximate="tanh")
        x = x * per_layer_input
        x = self.per_layer_projection(x)
        x = self.post_per_layer_input_norm(x)
        x = residual + x

        x = x * self.layer_scalar
        return x, produced_kv

    def forward_stateful(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        per_layer_input: torch.Tensor,
        kv: KVCache,
        slot: int,
        write: bool,
        ring: bool = False,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn.forward_stateful(x, position_ids, kv, slot, write, ring=ring)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        residual = x
        x = self.per_layer_input_gate(x)
        x = nn.functional.gelu(x, approximate="tanh")
        x = x * per_layer_input
        x = self.per_layer_projection(x)
        x = self.post_per_layer_input_norm(x)
        x = residual + x

        return x * self.layer_scalar


class ScaledEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float) -> None:
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * torch.tensor(self.embed_scale, dtype=self.weight.dtype)


class Gemma4TextModel(nn.Module):
    def __init__(self, cfg: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = cfg
        h = cfg.hidden_size
        ld = cfg.hidden_size_per_layer_input

        self.embed_tokens = ScaledEmbedding(cfg.vocab_size, h, embed_scale=h**0.5)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(h, eps=cfg.rms_norm_eps)

        # Per-layer-embedding front-end (kept out of the convertible ``decode`` core).
        self.embed_tokens_per_layer = ScaledEmbedding(
            cfg.vocab_size_per_layer_input,
            cfg.num_hidden_layers * ld,
            embed_scale=ld**0.5,
        )
        self.per_layer_model_projection = nn.Linear(h, cfg.num_hidden_layers * ld, bias=False)
        self.per_layer_model_projection_scale = h**-0.5
        self.per_layer_input_scale = 2.0**-0.5
        self.per_layer_projection_norm = RMSNorm(ld, eps=cfg.rms_norm_eps)

    def compute_per_layer_inputs(
        self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.config
        ld = cfg.hidden_size_per_layer_input
        b, s = input_ids.shape
        tokens = self.embed_tokens_per_layer(input_ids).reshape(
            b, s, cfg.num_hidden_layers, ld
        )
        proj = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        proj = proj.reshape(b, s, cfg.num_hidden_layers, ld)
        proj = self.per_layer_projection_norm(proj)
        return (proj + tokens) * self.per_layer_input_scale

    def decode(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the transformer stack on pre-computed embeddings (convertible core)."""
        x = inputs_embeds
        producer_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(self.layers):
            shared_kv = producer_kv.get(layer.layer_type) if layer.is_shared else None
            x, produced_kv = layer(x, position_ids, per_layer_inputs[:, :, i, :], shared_kv)
            if layer.is_producer and produced_kv is not None:
                producer_kv[layer.layer_type] = produced_kv
        return self.norm(x)

    def decode_stateful(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        position_ids: torch.Tensor,
        kv_sliding: KVCache,
        kv_full: KVCache,
        ring: bool = False,
    ) -> torch.Tensor:
        """Incremental transformer stack against persistent dual KV caches.

        Equivalent to :meth:`decode` for a single full pass, but the producers
        L13/L14 write into ``kv_sliding`` / ``kv_full`` and the KV-shared layers
        read them back — so it also serves chunked prefill and one-token decode.

        With ``ring`` set, the sliding cache is a width-``W`` ring buffer instead of
        a linear ``ctx``-sized cache (a long-context memory optimisation); the full
        cache stays linear. The two modes are numerically identical for one-token
        decode and from-zero prefill.
        """
        route, _, _ = self.config.stateful_routing()
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            is_full, slot, write = route[i]
            kv = kv_full if is_full else kv_sliding
            x = layer.forward_stateful(
                x, position_ids, per_layer_inputs[:, :, i, :], kv, slot, write, ring=ring
            )
        return self.norm(x)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        per_layer_inputs = self.compute_per_layer_inputs(input_ids, inputs_embeds)
        return self.decode(inputs_embeds, per_layer_inputs, position_ids)


class Gemma4ForCausalLM(nn.Module):
    """Gemma 4 text decoder with a (tied) LM head and final logit softcapping."""

    def __init__(self, cfg: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = cfg
        self.model = Gemma4TextModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        sc = self.config.final_logit_softcapping
        if sc is not None:
            logits = torch.tanh(logits / sc) * sc
        return logits

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model(input_ids, position_ids)
        return self.logits_from_hidden(hidden)

    @classmethod
    def from_local(
        cls: type[Self],
        model_dir: str,
        *,
        num_layers: int | None = None,
        num_kv_shared: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        """Build from a local HF Gemma 4 checkpoint dir (config.json + safetensors).

        ``num_layers`` truncates the decoder (smoke tests). ``num_kv_shared``
        overrides how many trailing layers share KV — useful when truncating, so
        the non-shared prefix still contains a full-attention producer.
        """
        import json
        from pathlib import Path

        d = json.loads((Path(model_dir) / "config.json").read_text())
        cfg = Gemma4TextConfig.from_hf_config(d)
        if num_layers is not None:
            cfg.num_kv_shared_layers = _truncated_kv_shared(cfg, num_layers)
            cfg.num_hidden_layers = num_layers
            cfg.layer_types = cfg.layer_types[:num_layers]
        if num_kv_shared is not None:
            cfg.num_kv_shared_layers = num_kv_shared

        model = cls(cfg).to(dtype).eval()
        _load_hf_weights(model, model_dir, dtype=dtype)
        return model


def _load_hf_weights(
    model: Gemma4ForCausalLM, model_dir: str, *, dtype: torch.dtype = torch.float32
) -> None:
    """Copy ``model.language_model.*`` weights from the checkpoint into ``model``."""
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
                # Skip layers beyond a truncated config.
                if ".layers." in local:
                    li = int(local.split(".layers.")[1].split(".")[0])
                    if li >= n_layers:
                        continue
                if local in want:
                    t = h.get_tensor(k)
                    sd[local] = t.to(dtype) if t.dtype.is_floating_point else t

    # When the decoder is truncated, slice the per-layer-embedding tables (whose
    # columns are laid out per real layer) down to the kept layers.
    ple_cols = model.config.num_hidden_layers * model.config.hidden_size_per_layer_input
    if (k := "model.embed_tokens_per_layer.weight") in sd and sd[k].shape[1] != ple_cols:
        sd[k] = sd[k][:, :ple_cols].contiguous()
    if (k := "model.per_layer_model_projection.weight") in sd and sd[k].shape[0] != ple_cols:
        sd[k] = sd[k][:ple_cols, :].contiguous()

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    def _dead_shared_kv(name: str) -> bool:
        """KV-side params on KV-SHARED layers are dead weights (those layers read the
        producer's cache slot and never project K/V). The QAT release checkpoints
        prune them; the base checkpoints still carry them. Either is loadable."""
        if ".layers." not in name:
            return False
        li = int(name.split(".layers.")[1].split(".")[0])
        if not model.config.is_kv_shared(li):
            return False
        return (".self_attn.k_proj." in name or ".self_attn.v_proj." in name
                or ".self_attn.k_norm." in name)

    dead = [m for m in missing if _dead_shared_kv(m)]
    if dead:
        print(f"note: {len(dead)} shared-layer KV params absent from the checkpoint "
              "(pruned dead weights — KV-shared layers never project K/V)")
    real_missing = [
        m for m in missing
        if "inv_freq" not in m and "lm_head" not in m and not _dead_shared_kv(m)
    ]
    if real_missing:
        raise RuntimeError(f"missing weights: {real_missing[:10]} (+{len(real_missing) - 10} more)")


# ---------------------------------------------------------------------------
# Stateful (chunked KV-cache) decode core — efficient generation.
#
# Gemma 4's dual head_dim means a single uniform KV cache won't fit, so the
# stack uses TWO Core AI states: a sliding cache (head_dim 256) and a full
# cache (head_dim 512). Only the non-shared layers own slots; the producers
# (L13 sliding / L14 full) write the slots the KV-shared layers L15-34 read.
# A single full pass (offset 0) is numerically identical to the stateless
# ``decode`` core, so HF parity carries over; chunked prefill and one-token
# decode reuse the same graph.
# ---------------------------------------------------------------------------

# State tensor names surfaced to the Core AI runtime via export_to_coreai(state_names=).
DECODE_STATE_NAMES = ("slidingKeyCache", "slidingValueCache", "fullKeyCache", "fullValueCache")


def build_decode_state(
    cfg: Gemma4TextConfig, max_seq_len: int, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Zero-initialised dual KV state for the stateful decode core.

    Sliding/full caches are stacked over their own non-shared layers only
    (shape ``(num_slots, 1, num_kv_heads, max_seq_len, head_dim)``); see
    :meth:`Gemma4TextConfig.stateful_routing`.
    """
    _, n_sliding, n_full = cfg.stateful_routing()
    nkv = cfg.num_key_value_heads
    return {
        "sliding_k": torch.zeros(n_sliding, 1, nkv, max_seq_len, cfg.head_dim, dtype=dtype),
        "sliding_v": torch.zeros(n_sliding, 1, nkv, max_seq_len, cfg.head_dim, dtype=dtype),
        "full_k": torch.zeros(n_full, 1, nkv, max_seq_len, cfg.global_head_dim, dtype=dtype),
        "full_v": torch.zeros(n_full, 1, nkv, max_seq_len, cfg.global_head_dim, dtype=dtype),
    }


def build_decode_state_ring(
    cfg: Gemma4TextConfig, max_seq_len: int, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Dual KV state for the RING stateful decode core (:class:`Gemma4DecodeStatefulRing`).

    Like :func:`build_decode_state` but the sliding cache is a fixed ``sliding_window``
    (W) wide ring buffer regardless of ``max_seq_len`` — only the last W sliding-window
    keys are ever in scope, so it never needs to grow with the context. The full cache
    is still ``max_seq_len`` (global attention attends to every position).
    """
    _, n_sliding, n_full = cfg.stateful_routing()
    nkv = cfg.num_key_value_heads
    w = cfg.sliding_window
    return {
        "sliding_k": torch.zeros(n_sliding, 1, nkv, w, cfg.head_dim, dtype=dtype),
        "sliding_v": torch.zeros(n_sliding, 1, nkv, w, cfg.head_dim, dtype=dtype),
        "full_k": torch.zeros(n_full, 1, nkv, max_seq_len, cfg.global_head_dim, dtype=dtype),
        "full_v": torch.zeros(n_full, 1, nkv, max_seq_len, cfg.global_head_dim, dtype=dtype),
    }


class Gemma4DecodeStateful(nn.Module):
    """Stateful decode core: dual KV caches mutated in place across calls.

    Forward inputs mirror the stateless core plus the four cache states:
    ``(inputs_embeds, per_layer_inputs, position_ids, sliding_k, sliding_v,
    full_k, full_v) -> hidden``. The giant embedding / PLE / lm_head tables stay
    on the CPU front-end, exactly as in the stateless export.
    """

    # Opt OUT of composite externalization for the macOS export pipeline: this
    # wrapper holds the whole text model, incl. the PLE front-end RMSNorms
    # (per_layer_projection_norm, …) that are NOT in the traced decode graph.
    # Externalizing RMSNorm by class would mark those orphans and then fail with
    # "Custom op … not found in any ancestor program". The decode graph lowers
    # fine without externalize (matches convert_stateful.py's direct path).
    coreai_externalize_specs: tuple = ()

    def __init__(self, text_model: Gemma4TextModel) -> None:
        super().__init__()
        self.text_model = text_model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        position_ids: torch.Tensor,
        sliding_k: torch.Tensor,
        sliding_v: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
    ) -> torch.Tensor:
        kv_sliding = KVCache(sliding_k, sliding_v)
        kv_full = KVCache(full_k, full_v)
        return self.text_model.decode_stateful(
            inputs_embeds, per_layer_inputs, position_ids, kv_sliding, kv_full
        )

    def build_macos_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        query_len: int,
        offset: int,
        trace_kv_len: int,
    ) -> dict:
        """Reference inputs + dynamic shapes for the stateful (one-graph) export.

        ``position_ids`` carries the full positions (so offset = seq_len - query_len);
        the dual KV caches grow on their sequence dim. Consumed by
        ``export_macos_model`` via its hybrid hook.
        """
        cfg = self.text_model.config
        b = 1
        h = cfg.hidden_size
        ld = cfg.hidden_size_per_layer_input
        seq = query_len + offset
        inputs_embeds = torch.zeros(b, query_len, h, dtype=target_dtype)
        per_layer_inputs = torch.zeros(b, query_len, cfg.num_hidden_layers, ld, dtype=target_dtype)
        position_ids = torch.arange(seq, dtype=torch.int32).unsqueeze(0).expand(b, seq)
        state = build_decode_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)

        reference_inputs = {
            "inputs_embeds": inputs_embeds,
            "per_layer_inputs": per_layer_inputs,
            "position_ids": position_ids,
            "sliding_k": state["sliding_k"],
            "sliding_v": state["sliding_v"],
            "full_k": state["full_k"],
            "full_v": state["full_v"],
        }
        seq_ids = torch.export.Dim("seq_ids", max=max_context_length - 2)
        seq_pos = torch.export.Dim("seq_pos", min=query_len, max=max_context_length - 1)
        sd = KVCache.seq_len_dim()
        cache_dim = {
            n: {sd: torch.export.Dim(f"{n}_seq", min=trace_kv_len, max=max_context_length)}
            for n in ("sliding_k", "sliding_v", "full_k", "full_v")
        }
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


class Gemma4DecodeStatefulRing(Gemma4DecodeStateful):
    """Stateful decode core with a RING-BUFFER sliding cache.

    Identical I/O to :class:`Gemma4DecodeStateful` — ``(inputs_embeds,
    per_layer_inputs, position_ids, sliding_k, sliding_v, full_k, full_v) ->
    hidden`` with the four cache states — but the ``sliding_k`` / ``sliding_v``
    states are width-``sliding_window`` (W) ring buffers written at ``pos % W``
    rather than ``ctx``-sized linear caches. Build the matching zero state with
    :func:`build_decode_state_ring`. The full cache is unchanged.

    For one-token decode and from-zero prefill this is numerically identical to
    the linear core (verified in ``verify_stateful_ring.py`` /
    ``convert_stateful_ring.py``); it caps the sliding cache at W regardless of
    context length, which is the long-context memory win. Chunked prefill of a
    block that wraps the ring drops the oldest in-window keys for the block's
    non-final tokens (only the final token's logits are exact) — bound the
    prefill chunk to <= W and feed the prompt from position 0, as the engine does.
    """

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor,
        position_ids: torch.Tensor,
        sliding_k: torch.Tensor,
        sliding_v: torch.Tensor,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
    ) -> torch.Tensor:
        kv_sliding = KVCache(sliding_k, sliding_v)
        kv_full = KVCache(full_k, full_v)
        return self.text_model.decode_stateful(
            inputs_embeds, per_layer_inputs, position_ids, kv_sliding, kv_full, ring=True
        )

    def build_macos_export_spec(self, *args, **kwargs) -> dict:
        # The parent's spec builds a LINEAR state (build_decode_state, all four caches
        # dynamic on the seq dim) — wrong for the ring, whose sliding state is a static
        # W-wide buffer. The verified ring export path is convert_stateful_ring.py (it
        # builds the reference inputs + dynamic shapes inline: sliding static at W, qid
        # bounded by W). CLI routing for the ring core is a deliberate follow-up; fail
        # loudly rather than silently emit a linear-shaped (and oversized) spec.
        raise NotImplementedError(
            "Gemma4DecodeStatefulRing has no CLI export spec yet; convert it with "
            "gemma4/convert_stateful_ring.py (sliding state is a static W-wide ring, "
            "qid <= W)."
        )


# ---------------------------------------------------------------------------
# LM head (tied) + final logit softcap — the on-device "head" stage.
#
# The decode core consumes pre-gathered embeddings and returns ``hidden`` only;
# the 1536->262144 projection onto the (tied) ``embed_tokens`` weight, plus
# Gemma 4's ``tanh(z/30)*30`` softcap, is a separate Core AI export unit. Keeping
# it out of the decode core means the ~0.8 GB (fp16) head weight lives in its own
# function/bundle and the on-device flow becomes:
#     input_ids --(front-end gather)--> embeds --(decode core)--> hidden
#               --(head)--> logits --> argmax/sample
# ---------------------------------------------------------------------------


class Gemma4Head(nn.Module):
    """Tied LM head + Gemma 4 final logit softcap, as a standalone export unit.

    ``hidden (b, s, hidden_size) -> logits (b, s, vocab_size)``: the matmul
    against the (tied) ``embed_tokens`` weight followed by ``tanh(z/c)*c``. The
    weight is the *raw* embedding matrix (the embedding ``embed_scale`` is applied
    only on the gather side, so it must NOT be folded in here).
    """

    def __init__(self, cfg: Gemma4TextConfig) -> None:
        super().__init__()
        self.config = cfg
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.softcap = cfg.final_logit_softcapping

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    @classmethod
    def from_local(
        cls, model_dir: str, *, dtype: torch.dtype = torch.float32
    ) -> "Gemma4Head":
        """Build the head from a checkpoint, loading ONLY the tied embed weight.

        Avoids materialising the 35-layer decoder (or the 9.4 GB per-layer table)
        just to convert the head — the head needs the single
        ``embed_tokens.weight`` (vocab x hidden) and the config softcap/sizes.
        """
        import json
        from pathlib import Path

        from safetensors import safe_open

        cfg = Gemma4TextConfig.from_hf_config(
            json.loads((Path(model_dir) / "config.json").read_text())
        )
        head = cls(cfg).to(dtype).eval()
        key = "model.language_model.embed_tokens.weight"
        for f in sorted(Path(model_dir).glob("*.safetensors")):
            with safe_open(str(f), framework="pt", device="cpu") as h:
                if key in h.keys():  # noqa: SIM118
                    with torch.no_grad():
                        head.lm_head.weight.copy_(h.get_tensor(key).to(dtype))
                    return head
        raise RuntimeError(f"{key} not found in any safetensors under {model_dir}")


# ---------------------------------------------------------------------------
# Embedding-gather FRONT-END — the on-device CPU/ANE pre-stage.
#
# The decode core consumes pre-gathered ``inputs_embeds`` + ``per_layer_inputs``;
# producing them is two giant gathers + a small projection, kept OUT of the core:
#   embed_tokens            262144 x 1536   (-> inputs_embeds, x sqrt(hidden))
#   embed_tokens_per_layer  262144 x 8960   (-> per_layer tokens, x sqrt(256))
#   per_layer_model_projection + per_layer_projection_norm  (small)
# At fp32 these tables are ~1.6 GB + ~9.4 GB, so they must be compressed for device.
# K-means palettization only covers F.linear (not nn.Embedding), and the iOS
# ``fused_dequant_gather_reshape`` custom op does NOT lower on the macOS export path
# (MLIRError) — but a PLAIN int8 per-row dequant-gather (``q_table[ids].to(fp16) *
# scale[ids]``) lowers cleanly and keeps the table int8 in the bundle. That is what
# these modules use. int4 has no native torch dtype / clean gather lowering, so the
# tables are int8 (per the design's int8 fallback).
# ---------------------------------------------------------------------------


def _quant_per_row_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row (per-token) int8 quantization of an embedding table.

    Returns ``(q_int8 (V, D), scale_fp16 (V, 1))`` with ``weight ≈ q * scale``. Per-row
    keeps each token vector's own dynamic range, which is what the gather indexes.
    """
    scale = (weight.abs().amax(dim=-1, keepdim=True) / 127.0).clamp(min=1e-8)
    q = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float16)


class GatherEmbeds(nn.Module):
    """``input_ids -> inputs_embeds`` (embed_tokens gather x sqrt(hidden_size)).

    Holds the (int8 or fp16) ``embed_tokens`` table; the ``embed_scale`` is folded into
    the dequant so the output matches HF's scaled embedding.
    """

    def __init__(self, q_table: torch.Tensor, scale: torch.Tensor, embed_scale: float) -> None:
        super().__init__()
        self.register_buffer("q_table", q_table)  # (V, D) int8, or fp16 (uncompressed)
        self.register_buffer("scale", scale)      # (V, 1) fp16 (ones when q_table is float)
        self.embed_scale = float(embed_scale)

    @classmethod
    def from_weight(cls, weight: torch.Tensor, *, bits: int, embed_scale: float,
                    dtype: torch.dtype) -> "GatherEmbeds":
        if bits == 8:
            q, scale = _quant_per_row_int8(weight)
            scale = scale.to(dtype)  # dequant output dtype = module dtype (fp16 device / fp32 verify)
        elif bits == 16:
            q, scale = weight.to(dtype), torch.ones(weight.shape[0], 1, dtype=dtype)
        else:
            raise NotImplementedError(f"GatherEmbeds bits={bits} unsupported (use 8 or 16)")
        return cls(q, scale, embed_scale)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        g = self.q_table[input_ids]
        if not self.q_table.is_floating_point():
            g = g.to(self.scale.dtype) * self.scale[input_ids]
        return g * self.embed_scale


class GatherPerLayer(nn.Module):
    """``input_ids, inputs_embeds -> per_layer_inputs`` (PLE front-end).

    Mirrors :meth:`Gemma4TextModel.compute_per_layer_inputs`, but the giant
    ``embed_tokens_per_layer`` table is (int8 or fp16) gathered/dequantized. The small
    ``per_layer_model_projection`` (1536x8960) and the RMSNorm stay full precision.
    """

    def __init__(self, q_table: torch.Tensor, scale: torch.Tensor, cfg: Gemma4TextConfig,
                 dtype: torch.dtype) -> None:
        super().__init__()
        self.register_buffer("q_table", q_table)  # (V, L*ld) int8 or fp16
        self.register_buffer("scale", scale)      # (V, 1) fp16
        self.num_layers = cfg.num_hidden_layers
        self.ld = cfg.hidden_size_per_layer_input
        self.embed_scale = float(self.ld ** 0.5)
        self.proj_scale = float(cfg.hidden_size ** -0.5)
        self.input_scale = float(2.0 ** -0.5)
        self.per_layer_model_projection = nn.Linear(
            cfg.hidden_size, self.num_layers * self.ld, bias=False
        ).to(dtype)
        self.per_layer_projection_norm = RMSNorm(self.ld, eps=cfg.rms_norm_eps).to(dtype)

    @classmethod
    def from_weights(cls, table: torch.Tensor, proj_w: torch.Tensor, norm_w: torch.Tensor,
                     cfg: Gemma4TextConfig, *, bits: int, dtype: torch.dtype) -> "GatherPerLayer":
        if bits == 8:
            q, scale = _quant_per_row_int8(table)
            scale = scale.to(dtype)  # dequant output dtype = module dtype
        elif bits == 16:
            q, scale = table.to(dtype), torch.ones(table.shape[0], 1, dtype=dtype)
        else:
            raise NotImplementedError(f"GatherPerLayer bits={bits} unsupported (use 8 or 16)")
        m = cls(q, scale, cfg, dtype)
        with torch.no_grad():
            m.per_layer_model_projection.weight.copy_(proj_w.to(dtype))
            m.per_layer_projection_norm.weight.copy_(norm_w.to(dtype))
        return m

    def forward(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape[0], input_ids.shape[1]
        g = self.q_table[input_ids]
        if not self.q_table.is_floating_point():
            g = g.to(inputs_embeds.dtype) * self.scale[input_ids]
        tokens = (g * self.embed_scale).reshape(b, s, self.num_layers, self.ld)
        proj = self.per_layer_model_projection(inputs_embeds) * self.proj_scale
        proj = self.per_layer_projection_norm(proj.reshape(b, s, self.num_layers, self.ld))
        return (proj + tokens) * self.input_scale


def load_gemma4_front_end(
    model_dir: str, *, embed_bits: int = 8, per_layer_bits: int = 8,
    dtype: torch.dtype = torch.float16,
) -> tuple[GatherEmbeds, GatherPerLayer]:
    """Load ONLY the front-end tables (not the 35 decoder layers) + build the gathers.

    Reads ``embed_tokens`` / ``embed_tokens_per_layer`` / ``per_layer_model_projection`` /
    ``per_layer_projection_norm`` from the checkpoint, quantizes the two big tables to
    ``embed_bits`` / ``per_layer_bits`` (8 = int8 per-row, 16 = fp16), and returns
    ``(GatherEmbeds, GatherPerLayer)`` — the two Core AI front-end export units.
    """
    import json
    from pathlib import Path

    from safetensors import safe_open

    cfg = Gemma4TextConfig.from_hf_config(
        json.loads((Path(model_dir) / "config.json").read_text())
    )
    keys = {
        "embed": "model.language_model.embed_tokens.weight",
        "per_layer": "model.language_model.embed_tokens_per_layer.weight",
        "proj": "model.language_model.per_layer_model_projection.weight",
        "norm": "model.language_model.per_layer_projection_norm.weight",
    }
    found: dict[str, torch.Tensor] = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as h:
            avail = set(h.keys())  # noqa: SIM118
            for name, key in keys.items():
                if name not in found and key in avail:
                    found[name] = h.get_tensor(key)
    missing = [keys[k] for k in keys if k not in found]
    if missing:
        raise RuntimeError(f"front-end tensors not found in {model_dir}: {missing}")

    embeds = GatherEmbeds.from_weight(
        found["embed"].float(), bits=embed_bits, embed_scale=cfg.hidden_size ** 0.5, dtype=dtype
    )
    per_layer = GatherPerLayer.from_weights(
        found["per_layer"].float(), found["proj"].float(), found["norm"].float(),
        cfg, bits=per_layer_bits, dtype=dtype,
    )
    return embeds.eval(), per_layer.eval()


class Gemma4TextForCausalLM(BaseForCausalLM):
    """Registry / CLI-facing Gemma 4 text decoder (``BaseForCausalLM``-compatible).

    Wraps the re-authored decoder so it resolves through the model registry and
    loads via ``from_hf``. Two Gemma-4 specifics shape this class:

      * The giant ``embed_tokens`` / ``embed_tokens_per_layer`` / ``lm_head``
        tables stay on the CPU front-end, so the *exported* unit is the stateful
        decode core (:class:`Gemma4DecodeStateful` via :meth:`export_core`), not
        this whole module. This class is the load + config-plumbing adapter; its
        ``forward`` is the eager full path (input_ids -> logits) for parity.
      * Weights live under ``model.language_model.`` and the pinned transformers
        can't parse ``model_type: gemma4`` — handled by the config shim and the
        custom :meth:`from_hf` (the base memory-efficient loader's prefix
        convention doesn't fit gemma4's infixed keys).
    """

    _HF_MODEL_CLASS = None  # no HF modeling class; load from safetensors directly

    def _init_model(self, config: Gemma4TextConfig) -> None:
        self.model = Gemma4TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _mutate_state_dict(self, state_dict: dict) -> None:
        # Loading is handled by the custom from_hf below; nothing to sanitise.
        pass

    @classmethod
    def _get_reauthored_config(
        cls, hf_config, max_context_length: int | None = None, num_layers: int | None = None
    ) -> Gemma4TextConfig:
        d = hf_config.to_dict() if hasattr(hf_config, "to_dict") else dict(hf_config)
        cfg = Gemma4TextConfig.from_hf_config(d)
        if num_layers is not None:
            cfg.num_kv_shared_layers = _truncated_kv_shared(cfg, num_layers)
            cfg.num_hidden_layers = num_layers
            cfg.layer_types = cfg.layer_types[:num_layers]
        if max_context_length is not None:
            cfg.max_position_embeddings = max_context_length
        return cfg

    def logits_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        sc = self.config.final_logit_softcapping
        if sc is not None:
            logits = torch.tanh(logits / sc) * sc
        return logits

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.logits_from_hidden(self.model(input_ids, position_ids))

    def export_core(self) -> Gemma4DecodeStateful:
        """The actual Core AI export unit: the stateful dual-KV decode core."""
        return Gemma4DecodeStateful(self.model)

    def export_head(self) -> Gemma4Head:
        """The second Core AI export unit: tied LM head + final logit softcap.

        Shares the (tied) ``embed_tokens`` weight — no copy. Pairs with
        :meth:`export_core` to form the on-device ``embeds -> hidden -> logits``
        pipeline (the embedding gather front-end stays on the CPU).
        """
        head = Gemma4Head(self.config)
        head.lm_head.weight = self.lm_head.weight
        return head.eval()

    @classmethod
    def from_hf(
        cls,
        huggingface_model_id: str,
        *,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float32,
        num_layers: int | None = None,
        num_kv_shared: int | None = None,
        **_,
    ) -> "Gemma4TextForCausalLM":
        """Load a Gemma 4 checkpoint into the re-authored decoder.

        Custom loader (not the base memory-efficient path): gemma4 keys are under
        ``model.language_model.`` and the config needs the AutoConfig shim.
        """
        import json
        from pathlib import Path

        from huggingface_hub import snapshot_download

        from coreai_models.models.macos.gemma4_config_shim import register_gemma4_configs

        register_gemma4_configs()
        model_dir = snapshot_download(
            huggingface_model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"],
        )
        d = json.loads((Path(model_dir) / "config.json").read_text())
        cfg = cls._get_reauthored_config(d, max_context_length=max_context_length, num_layers=num_layers)
        if num_kv_shared is not None:
            cfg.num_kv_shared_layers = num_kv_shared
        model = cls(cfg).to(target_dtype).eval()
        _load_hf_weights(model, model_dir, dtype=target_dtype)
        return model

    @classmethod
    def from_hf_memory_efficient(
        cls,
        huggingface_model_id: str,
        *,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        hf_config_attr: str | None = None,
        hf_state_dict_prefix: str = "",
        **_,
    ) -> "Gemma4TextForCausalLM":
        """Delegate to :meth:`from_hf`.

        The base streaming loader strips ``hf_state_dict_prefix`` and matches
        ``model.layers.`` — but gemma4's text weights are *infixed*
        (``model.language_model.layers.``), so that path mis-classifies every
        layer. Our loader handles the gemma4 key layout directly; ``mmap_path`` is
        ignored (the decode core fits in RAM).
        """
        return cls.from_hf(
            huggingface_model_id,
            max_context_length=max_context_length,
            target_dtype=target_dtype,
            num_layers=num_layers,
        )
