# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma 4 *dense* text decoder (e.g. ``google/gemma-4-12B-it``) for Core AI.

This is the text decoder of ``Gemma4UnifiedForConditionalGeneration`` (HF
``model_type: gemma4_unified``; text ``gemma4_unified_text``). Unlike the small
on-device ``gemma-4-E2B/E4B`` siblings (handled by :mod:`gemma4_text`), the
12B-class dense checkpoints carry **none** of the per-layer-embedding (PLE),
AltUp, Laurel, KV-sharing, MoE or double-wide-MLP machinery — they are a clean
interleaved-attention Gemma decoder, much closer to ``gemma3_text`` than to E2B.

Signature features handled here (all confirmed from the 12B checkpoint weights):
  * Interleaved attention, 5:1 sliding:full (``layer_types``; every 6th layer is
    ``full_attention``).
  * **Dual head_dim**: sliding head_dim 256; full ``global_head_dim`` 512.
  * **Dual KV-head count via ``attention_k_eq_v``**: sliding layers use the normal
    ``num_key_value_heads`` (8) with a real ``v_proj``; full layers use
    ``num_global_key_value_heads`` (1), have **no ``v_proj``**, and set value = the
    *raw* ``k_proj`` output (pre-norm / pre-RoPE), then a scale-free V RMSNorm.
  * Per-head Q/K RMSNorm (weighted) + scale-free V RMSNorm, applied before RoPE.
  * Attention scale = 1.0 (NOT 1/sqrt(d)); QK-norm bounds the magnitudes.
  * Dual RoPE: sliding theta 1e4 over the full 256-dim head; full theta 1e6
    "proportional" — only the first 64 freq pairs rotate, the remaining 192 are
    identity (NoPE), over the 512-dim head.
  * A learned per-layer scalar (``layer_scalar``) multiplying each block output.
  * Final logit softcapping ``tanh(z / 30) * 30``.

Gemma 4 RMSNorm multiplies by ``weight`` directly (checkpoint weights centred near
1, not 0), so the plain :class:`RMSNorm` primitive is used — NOT ``RMSNormPlusOne``.

The whole stack (giant tied ``embed_tokens`` / ``lm_head`` included) is small
enough that the in-graph-embed pipelined decode core (:mod:`gemma4_dense_pipelined`)
ships it as ONE Core-AI bundle with a single growing KV pair — no PLE front-end,
no host per-token inputs.
"""
from __future__ import annotations

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
class Gemma4DenseConfig:
    """Lightweight Gemma 4 *dense* text-decoder config (built from HF ``text_config``).

    Independent of ``transformers.models.gemma4`` so the overlay imports on the
    export venv (transformers 4.x, which predates gemma4_unified).
    """

    vocab_size: int = 262144
    hidden_size: int = 3840
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    intermediate_size: int = 15360
    attention_k_eq_v: bool = True
    rms_norm_eps: float = 1e-6
    sliding_window: int = 1024
    final_logit_softcapping: float | None = 30.0
    sliding_rope_theta: float = 10000.0
    full_rope_theta: float = 1000000.0
    full_partial_rotary_factor: float = 0.25
    hidden_activation: str = "gelu_pytorch_tanh"
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 262144
    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layer_types:
            # HF default: every 6th layer (idx 5, 11, …) is full_attention.
            self.layer_types = [
                FULL if (i + 1) % 6 == 0 else SLIDING for i in range(self.num_hidden_layers)
            ]

    @classmethod
    def from_hf_config(cls, d: dict) -> Gemma4DenseConfig:
        """Build from an HF config dict (top-level or already a ``text_config``)."""
        if "text_config" in d:
            d = d["text_config"]
        rope = d.get("rope_parameters", {}) or {}
        sliding_rope = rope.get(SLIDING, {}) or {}
        full_rope = rope.get(FULL, {}) or {}
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs.update(
            sliding_rope_theta=sliding_rope.get("rope_theta", 10000.0),
            full_rope_theta=full_rope.get("rope_theta", 1000000.0),
            full_partial_rotary_factor=full_rope.get("partial_rotary_factor", 0.25),
        )
        return cls(**kwargs)

    # ---- derived layout helpers ----
    def is_full(self, i: int) -> bool:
        return self.layer_types[i] == FULL

    def head_dim_of(self, i: int) -> int:
        return self.global_head_dim if self.is_full(i) else self.head_dim

    def n_kv_of(self, i: int) -> int:
        if self.is_full(i) and self.attention_k_eq_v:
            return self.num_global_key_value_heads
        return self.num_key_value_heads


def _sliding_inv_freq(cfg: Gemma4DenseConfig) -> torch.Tensor:
    hd = cfg.head_dim
    return 1.0 / (cfg.sliding_rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))


def _full_inv_freq(cfg: Gemma4DenseConfig) -> torch.Tensor:
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
    """Gemma 4 dense attention — sliding (8 KV heads, hd 256) or full (1 KV head,
    hd 512, value == raw k_proj output via ``attention_k_eq_v``)."""

    def __init__(self, cfg: Gemma4DenseConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_full = cfg.is_full(layer_idx)
        self.k_eq_v = self.is_full and cfg.attention_k_eq_v

        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.n_kv_of(layer_idx)
        self.head_dim = hd = cfg.head_dim_of(layer_idx)

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * hd, bias=False)
        # Full layers share K as V (no v_proj in the checkpoint).
        self.v_proj = (
            None if self.k_eq_v
            else nn.Linear(cfg.hidden_size, self.n_kv_heads * hd, bias=False)
        )
        self.o_proj = nn.Linear(self.n_heads * hd, cfg.hidden_size, bias=False)

        self.q_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.v_norm = ScaleFreeRMSNorm(eps=cfg.rms_norm_eps)

        # Plain fp32 attribute (NOT a buffer) so model.half() can't cast it: the
        # composite RoPE requires fp32 freqs, and the small full-RoPE frequencies
        # would underflow to zero in fp16.
        self.inv_freq = (_full_inv_freq(cfg) if self.is_full else _sliding_inv_freq(cfg)).float()
        self.rope = RoPE(scale=1.0)

        # Gemma 4 attention scale is 1.0 (not 1/sqrt(head_dim)).
        self.sdpa = SDPA(
            scale=1.0,
            is_causal=True,
            window_size=cfg.sliding_window if not self.is_full else 0,
        )
        self.sliding_window = cfg.sliding_window

    def _project_kv(self, x: torch.Tensor, position_ids: torch.Tensor):
        b, s, _ = x.shape
        k = self.k_proj(x).view(b, s, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        # value = raw k_proj output (pre-norm / pre-RoPE) on k_eq_v full layers.
        v = k if self.k_eq_v else (
            self.v_proj(x).view(b, s, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        )
        k = self.k_norm(k)
        k = self.rope(k, position_ids=position_ids, freqs=self.inv_freq)
        v = self.v_norm(v)
        return k, v

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        q = self.rope(q, position_ids=position_ids, freqs=self.inv_freq)
        k, v = self._project_kv(x, position_ids)
        out = self.sdpa(query=q, key=k, value=v)
        out = out.permute(0, 2, 1, 3).reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Gemma4DenseConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        h = cfg.hidden_size
        eps = cfg.rms_norm_eps

        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(h, cfg.intermediate_size)

        self.input_layernorm = RMSNorm(h, eps=eps)
        self.post_attention_layernorm = RMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_ids)
        x = self.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        return x * self.layer_scalar


class ScaledEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, embed_scale: float) -> None:
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * torch.tensor(self.embed_scale, dtype=self.weight.dtype)


class Gemma4DenseModel(nn.Module):
    def __init__(self, cfg: Gemma4DenseConfig) -> None:
        super().__init__()
        self.config = cfg
        h = cfg.hidden_size
        self.embed_tokens = ScaledEmbedding(cfg.vocab_size, h, embed_scale=h**0.5)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(h, eps=cfg.rms_norm_eps)

    def decode(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x, position_ids)
        return self.norm(x)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.decode(self.embed_tokens(input_ids), position_ids)


class Gemma4DenseForCausalLM(nn.Module):
    """Gemma 4 dense text decoder with a (tied) LM head and final logit softcap.

    The ``forward`` is the eager full path (``input_ids -> logits``) used for HF
    parity; the Core-AI export unit is the decode core in
    :mod:`gemma4_dense_pipelined`.
    """

    def __init__(self, cfg: Gemma4DenseConfig) -> None:
        super().__init__()
        self.config = cfg
        self.model = Gemma4DenseModel(cfg)
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
        return self.logits_from_hidden(self.model(input_ids, position_ids))

    @classmethod
    def from_local(
        cls: type[Self],
        model_dir: str,
        *,
        num_layers: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        """Build from a local HF Gemma 4 dense checkpoint (config.json + safetensors).

        ``num_layers`` truncates the decoder for smoke tests (the layout has no
        KV-sharing, so any truncation keeping >= 1 full layer is self-consistent).
        """
        import json
        from pathlib import Path

        d = json.loads((Path(model_dir) / "config.json").read_text())
        cfg = Gemma4DenseConfig.from_hf_config(d)
        if num_layers is not None:
            cfg.num_hidden_layers = num_layers
            cfg.layer_types = cfg.layer_types[:num_layers]
        model = cls(cfg).to(dtype).eval()
        _load_hf_weights(model, model_dir, dtype=dtype)
        return model


def _load_hf_weights(
    model: Gemma4DenseForCausalLM, model_dir: str, *, dtype: torch.dtype = torch.float32
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
                if ".layers." in local:
                    li = int(local.split(".layers.")[1].split(".")[0])
                    if li >= n_layers:
                        continue
                if local in want:
                    t = h.get_tensor(k)
                    sd[local] = t.to(dtype) if t.dtype.is_floating_point else t

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if model.config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight

    real_missing = [m for m in missing if "inv_freq" not in m and "lm_head" not in m]
    if real_missing:
        raise RuntimeError(f"missing weights: {real_missing[:10]} (+{len(real_missing) - 10} more)")
    # ``v_proj`` is absent on full layers (k_eq_v) and ``lm_head`` is tied — both expected.
    junk = [u for u in unexpected if "rotary_emb" not in u]
    if junk:
        raise RuntimeError(f"unexpected weights: {junk[:10]} (+{len(junk) - 10} more)")
