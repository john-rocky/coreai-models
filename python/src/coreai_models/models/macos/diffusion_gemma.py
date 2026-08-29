# Community port — NOT an Apple model. BSD-3-Clause (see LICENSE).

"""DiffusionGemma (``google/diffusiongemma-26B-A4B-it``) text backbone for the
Core AI authoring path.

The zoo's first diffusion-LM (dLLM). DiffusionGemma is a block-autoregressive
hybrid built on the Gemma 4 backbone (HF ``model_type: diffusion_gemma``): a
prompt is encoded auto-regressively (causal attention, writes a KV cache), then
a fixed-length ``canvas`` of tokens is iteratively denoised by a bidirectional
decoder that reads — but never writes — that prompt KV cache.

ONE backbone, two modes, shared weights (the checkpoint stores them once, under
``model.decoder.*``; the encoder copy is tied):
  * encoder mode = causal attention, writes/returns per-layer K/V. PREFILL + COMMIT.
  * decoder mode = bidirectional attention (``is_causal=False``); per layer it
    concatenates ``[encoder_K, canvas_K]`` / ``[encoder_V, canvas_V]`` and attends
    over the union (cross-attention to the cached prompt KV + self-attention over
    the canvas), plus a self-conditioning branch on the input embeddings. DENOISE.

Signature features vs. a plain Gemma 4 text layer (all handled here):
  * Dual head_dim (Gemma 4 style): sliding_attention head_dim 256 / 8 KV heads;
    full_attention global_head_dim 512 / 2 KV heads (full layers at the explicit
    ``layer_types`` indices 5,11,17,23,29). Attention scale = 1.0; per-head Q/K
    RMSNorm + scale-free V RMSNorm; dual RoPE (sliding theta 1e4 full-head; full
    proportional partial-rotary 0.25 theta 1e6 — reused verbatim from gemma4_text).
  * Full-attention layers have NO v_proj: V reuses the (pre-norm, pre-RoPE) k_proj
    output, then the scale-free V-norm (``attention_k_eq_v`` for full layers only).
  * The FFN is a DUAL PARALLEL branch — a dense MLP ("shared expert") summed with a
    sparse MoE, both fed from the post-attention residual:
        m1 = post_ff_ln_1( mlp( pre_ff_ln(r) ) )                 # dense, inter 2112
        m2 = post_ff_ln_2( moe( router(r), pre_ff_ln_2(r) ) )    # 128 experts top-8, inter 704
        h  = r + post_ff_ln(m1 + m2) ;  h *= layer_scalar
    → 7 RMSNorms + a per-layer ``layer_scalar`` buffer per layer.
  * Router (NOT sigmoid): ``softmax_fp32( proj( norm(r)·scale·hidden^-0.5 ) )`` over
    all 128 experts → top-8 → renormalize to sum 1 → × per-expert learned scale.
    Note the router is fed the RAW residual ``r`` (it norms internally), while the
    experts are fed ``pre_ff_ln_2(r)``.
  * Self-conditioning (decoder only): ``soft = softmax_fp32(prev_logits) @ embed ·
    embed_scale`` → gated MLP → residual-add to the canvas token embeddings, then a
    scale-free post-norm. Step 0 (no prev logits) ⇒ soft = 0, so the decoder input
    is simply ``post_norm(scaled_embed(canvas))``.
  * Final logit softcap ``tanh(z/30)·30``; lm_head tied to the token embedding.

This module is the text path only (phase 1); the ~550M vision encoder is skipped.
Authored with coreai_models primitives (so it lowers through coreai-torch) and
numerically faithful to the HF ``DiffusionGemmaForBlockDiffusion`` eager forward.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from typing_extensions import Self

from coreai_models.models.macos.gemma4_text import MLP, ScaledEmbedding, ScaleFreeRMSNorm
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU

SLIDING = "sliding_attention"
FULL = "full_attention"


@dataclass
class DiffusionGemmaConfig:
    """Lightweight DiffusionGemma text config (built from HF ``text_config``).

    Independent of ``transformers.models.diffusion_gemma`` so the model imports in
    the (older-transformers) conversion venv.
    """

    vocab_size: int = 262144
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    num_key_value_heads: int = 8           # sliding layers
    num_global_key_value_heads: int = 2    # full layers
    head_dim: int = 256                    # sliding layers
    global_head_dim: int = 512             # full layers
    intermediate_size: int = 2112          # dense MLP ("shared expert")
    moe_intermediate_size: int = 704       # per expert
    num_experts: int = 128
    top_k_experts: int = 8
    rms_norm_eps: float = 1e-6
    sliding_window: int = 1024
    final_logit_softcapping: float = 30.0
    sliding_rope_theta: float = 10000.0
    full_rope_theta: float = 1000000.0
    full_partial_rotary_factor: float = 0.25
    canvas_length: int = 256
    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.layer_types:
            # HF default: every 6th layer (1-indexed) is full_attention (5:1).
            self.layer_types = [
                FULL if (i + 1) % 6 == 0 else SLIDING for i in range(self.num_hidden_layers)
            ]

    @classmethod
    def from_hf_config(cls, d: dict) -> DiffusionGemmaConfig:
        """Build from an HF config dict (``config.json`` or its ``text_config``)."""
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

    def is_full(self, i: int) -> bool:
        return self.layer_types[i] == FULL

    def head_dim_of(self, i: int) -> int:
        return self.global_head_dim if self.is_full(i) else self.head_dim

    def n_kv_of(self, i: int) -> int:
        return self.num_global_key_value_heads if self.is_full(i) else self.num_key_value_heads


def _sliding_inv_freq(cfg: DiffusionGemmaConfig) -> torch.Tensor:
    """Default RoPE inv_freq for sliding heads (theta 1e4 over the full 256-dim head)."""
    hd = cfg.head_dim
    return 1.0 / (cfg.sliding_rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))


def _full_inv_freq(cfg: DiffusionGemmaConfig) -> torch.Tensor:
    """Proportional RoPE inv_freq for full heads (HF ``_compute_proportional``).

    The first ``partial_rotary_factor * head_dim / 2`` frequency pairs rotate (theta
    1e6); the rest are zero (identity / NoPE), so the vector has length
    ``global_head_dim // 2``. Identical to gemma4_text's full-RoPE construction.
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


class Attention(nn.Module):
    """Dual head_dim Gemma 4 attention, used in encoder (causal) and decoder
    (bidirectional, cross-attends the cached prompt KV) modes.

    Full-attention layers have no ``v_proj``: value reuses the (pre-norm, pre-RoPE)
    ``k_proj`` output, then the scale-free V-norm.
    """

    def __init__(self, cfg: DiffusionGemmaConfig, layer_idx: int) -> None:
        super().__init__()
        self.is_full = cfg.is_full(layer_idx)
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.n_kv_of(layer_idx)
        self.head_dim = hd = cfg.head_dim_of(layer_idx)
        self.has_v_proj = not self.is_full  # full layers: V = v_norm(k_proj(x))

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv * hd, bias=False)
        self.v_proj = (
            nn.Linear(cfg.hidden_size, self.n_kv * hd, bias=False) if self.has_v_proj else None
        )
        self.o_proj = nn.Linear(self.n_heads * hd, cfg.hidden_size, bias=False)

        self.q_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(hd, eps=cfg.rms_norm_eps)
        self.v_norm = ScaleFreeRMSNorm(eps=cfg.rms_norm_eps)

        # fp32 attribute (NOT a buffer): the proportional RoPE's small frequencies
        # would underflow in fp16; keep them fp32 so model.half() can't cast them.
        self.inv_freq = (_full_inv_freq(cfg) if self.is_full else _sliding_inv_freq(cfg)).float()
        self.rope = RoPE(scale=1.0)

        # Gemma 4 attention scale is 1.0 (QK-norm bounds the magnitudes).
        self.sdpa_causal = SDPA(
            scale=1.0, is_causal=True, window_size=cfg.sliding_window if not self.is_full else 0
        )
        # Decoder: full bidirectional attention over [cached prompt KV; canvas].
        self.sdpa_full = SDPA(scale=1.0, is_causal=False, window_size=0)

    def _qkv(self, x: torch.Tensor, position_ids: torch.Tensor):
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        q = self.rope(q, position_ids=position_ids, freqs=self.inv_freq)

        k_lin = self.k_proj(x).view(b, s, self.n_kv, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_norm(k_lin)
        k = self.rope(k, position_ids=position_ids, freqs=self.inv_freq)

        # Full layers share k_proj for V (k_eq_v): take it BEFORE k_norm/RoPE.
        v = self.v_proj(x).view(b, s, self.n_kv, self.head_dim).permute(0, 2, 1, 3) \
            if self.has_v_proj else k_lin
        v = self.v_norm(v)
        return q, k, v

    def forward_encoder(self, x: torch.Tensor, position_ids: torch.Tensor):
        b, s, _ = x.shape
        q, k, v = self._qkv(x, position_ids)
        out = self.sdpa_causal(query=q, key=k, value=v)
        out = out.permute(0, 2, 1, 3).reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(out), (k, v)

    def forward_decoder(
        self, x: torch.Tensor, position_ids: torch.Tensor,
        enc_k: torch.Tensor, enc_v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q, k, v = self._qkv(x, position_ids)
        k = torch.cat([enc_k, k], dim=2)  # (b, n_kv, S_enc + canvas, head_dim)
        v = torch.cat([enc_v, v], dim=2)
        # attn_mask (additive [1,1,canvas,S_enc+canvas]) masks the right-padded prompt's PAD
        # positions in the cached encoder KV so the canvas never attends to them (free-input).
        # None == no padding (the fixed-SP preset path), identical to the unmasked full attention.
        out = self.sdpa_full(query=q, key=k, value=v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(b, s, self.n_heads * self.head_dim)
        return self.o_proj(out)


class _GeluTanhGate(nn.Module):
    """SwitchGLU gate activation matching Gemma's ``gelu_pytorch_tanh(gate) * up``."""

    def forward(self, up: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return nn.functional.gelu(gate, approximate="tanh") * up


class Router(nn.Module):
    """DiffusionGemma MoE router: ``softmax_fp32(proj(norm(r)·scale·hidden^-0.5))``
    over all experts → top-k → renormalize → × per-expert scale. Fed the RAW
    residual ``r`` (the internal ``norm`` is scale-free, no checkpoint weight)."""

    def __init__(self, cfg: DiffusionGemmaConfig) -> None:
        super().__init__()
        self.top_k = cfg.top_k_experts
        self.scalar_root_size = cfg.hidden_size ** -0.5
        self.norm = ScaleFreeRMSNorm(eps=cfg.rms_norm_eps)
        self.proj = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.scale = nn.Parameter(torch.ones(cfg.hidden_size))
        self.per_expert_scale = nn.Parameter(torch.ones(cfg.num_experts))

    def forward(self, r: torch.Tensor):
        h = self.norm(r) * self.scale * self.scalar_root_size
        probs = torch.softmax(self.proj(h).float(), dim=-1)  # full softmax over E, fp32
        # Concrete (non-negative) topk axis: MPSGraph's TopK lowering reads the axis as a
        # constant int tensor (RuntimeUtils::getAxis -> matchConstantWithIntVector); a -1 axis
        # has been seen to fail that constant-fold (EXC_BAD_ACCESS) in the large 30-layer
        # decoder graph. probs.dim()-1 traces to the same last axis as a plain int.
        scores, idx = torch.topk(probs, self.top_k, dim=probs.dim() - 1)
        scores = scores / scores.sum(dim=-1, keepdim=True)
        scores = scores * self.per_expert_scale[idx]
        return scores, idx


class DiffusionGemmaLayer(nn.Module):
    """One backbone layer: dual head_dim attention + dual parallel dense-MLP‖MoE FFN.

    The attention weights are shared across encoder/decoder modes; only the
    attention call differs (``forward_encoder`` vs ``forward_decoder``)."""

    def __init__(self, cfg: DiffusionGemmaConfig, layer_idx: int) -> None:
        super().__init__()
        h = cfg.hidden_size
        eps = cfg.rms_norm_eps
        self.layer_type = cfg.layer_types[layer_idx]

        self.self_attn = Attention(cfg, layer_idx)
        self.mlp = MLP(h, cfg.intermediate_size)  # dense branch (gelu-tanh GLU)
        self.router = Router(cfg)
        self.switch_mlp = SwitchGLU(
            h, cfg.moe_intermediate_size, cfg.num_experts, bias=False, activation=_GeluTanhGate()
        )

        self.input_layernorm = RMSNorm(h, eps=eps)
        self.post_attention_layernorm = RMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm_1 = RMSNorm(h, eps=eps)   # dense branch
        self.post_feedforward_layernorm_2 = RMSNorm(h, eps=eps)   # MoE branch
        self.pre_feedforward_layernorm_2 = RMSNorm(h, eps=eps)    # MoE expert input
        # The per-layer residual scalar is the ONE text buffer NOT shared between the
        # two modes: the checkpoint stores a separate ``layer_scalar`` for the encoder
        # (``model.encoder.language_model.layers.*``) and the decoder
        # (``model.decoder.layers.*``) — they differ (e.g. layer 1 = 0.204 vs 0.0).
        self.register_buffer("enc_layer_scalar", torch.ones(1), persistent=True)
        self.register_buffer("dec_layer_scalar", torch.ones(1), persistent=True)

    def _ffn(self, r: torch.Tensor) -> torch.Tensor:
        """Dual parallel FFN: ``post_ff_ln( post_ff_ln_1(mlp(pre_ff_ln(r))) +
        post_ff_ln_2(moe(router(r), pre_ff_ln_2(r))) )``."""
        m1 = self.post_feedforward_layernorm_1(self.mlp(self.pre_feedforward_layernorm(r)))

        scores, idx = self.router(r)  # router sees the raw residual
        exp_in = self.pre_feedforward_layernorm_2(r)
        moe = self.switch_mlp(exp_in, idx.to(torch.uint16))  # (b, s, k, h)
        # Router math is fp32; weight + accumulate in fp32 then cast back to the model
        # dtype (HF accumulates per-expert in the model dtype — equal-or-better here).
        moe = (moe * scores.unsqueeze(-1)).sum(dim=-2).to(r.dtype)
        m2 = self.post_feedforward_layernorm_2(moe)

        return self.post_feedforward_layernorm(m1 + m2)

    def forward_encoder(self, x: torch.Tensor, position_ids: torch.Tensor):
        a, kv = self.self_attn.forward_encoder(self.input_layernorm(x), position_ids)
        x = x + self.post_attention_layernorm(a)
        x = x + self._ffn(x)
        return x * self.enc_layer_scalar, kv

    def forward_decoder(
        self, x: torch.Tensor, position_ids: torch.Tensor,
        enc_k: torch.Tensor, enc_v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = self.self_attn.forward_decoder(
            self.input_layernorm(x), position_ids, enc_k, enc_v, attn_mask)
        x = x + self.post_attention_layernorm(a)
        x = x + self._ffn(x)
        return x * self.dec_layer_scalar


class SelfConditioning(nn.Module):
    """Gated MLP that folds the previous step's soft-embeddings into the canvas
    embeddings: ``post_norm( inputs_embeds + down(gelu(gate(pre_norm(soft)))*up(...)) )``.
    ``post_norm`` is scale-free (no checkpoint weight)."""

    def __init__(self, cfg: DiffusionGemmaConfig) -> None:
        super().__init__()
        h = cfg.hidden_size
        inter = cfg.intermediate_size
        self.pre_norm = RMSNorm(h, eps=cfg.rms_norm_eps)
        self.post_norm = ScaleFreeRMSNorm(eps=cfg.rms_norm_eps)
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, inputs_embeds: torch.Tensor, soft_embeds: torch.Tensor) -> torch.Tensor:
        normed = self.pre_norm(soft_embeds)
        sc = self.down_proj(
            nn.functional.gelu(self.gate_proj(normed), approximate="tanh") * self.up_proj(normed)
        )
        return self.post_norm(inputs_embeds + sc)


class DiffusionGemma(nn.Module):
    """The shared Gemma 4 backbone (= the checkpoint's ``model.decoder.*`` weights),
    run auto-regressively in :meth:`encode` and bidirectionally in :meth:`decode`."""

    def __init__(self, cfg: DiffusionGemmaConfig) -> None:
        super().__init__()
        self.config = cfg
        self.embed_tokens = ScaledEmbedding(
            cfg.vocab_size, cfg.hidden_size, embed_scale=cfg.hidden_size ** 0.5
        )
        self.layers = nn.ModuleList(
            [DiffusionGemmaLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_conditioning = SelfConditioning(cfg)

    def encode(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        """Causal prefill over the prompt → per-layer (K, V) cache + encoder hidden."""
        x = self.embed_tokens(input_ids)
        kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            x, kv = layer.forward_encoder(x, position_ids)
            kvs.append(kv)
        return kvs, self.norm(x)

    def decode_with_soft(
        self,
        canvas_ids: torch.Tensor,
        position_ids: torch.Tensor,
        enc_kvs: list[tuple[torch.Tensor, torch.Tensor]],
        soft_embeds: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One bidirectional denoise pass, given the self-conditioning contribution
        (``soft_embeds[b, canvas, hidden]``) directly instead of the previous-step
        logits.

        This is the engine-export entry point. The ``softmax(prev_logits) @ embed``
        that builds ``soft_embeds`` is factored out into a separate ``soft_proj``
        graph because (a) it removes :meth:`decode`'s data-dependent
        ``if self_conditioning_logits is not None`` branch — which ``torch.export``
        would bake to whichever side is traced — so ONE decoder graph serves all 48
        denoise steps (``soft_embeds = 0`` reproduces the ``None`` branch exactly,
        since ``self_conditioning(embeds, 0) = post_norm(embeds)`` as ``RMSNorm(0) = 0``),
        and (b) it shrinks the per-step decoder input from the ``[b, canvas, vocab]``
        logits to the ``[b, canvas, hidden]`` contribution. :meth:`decode` delegates
        here, so the two stay numerically identical (parity-safe)."""
        inputs_embeds = self.embed_tokens(canvas_ids)
        x = self.self_conditioning(inputs_embeds, soft_embeds)
        for i, layer in enumerate(self.layers):
            enc_k, enc_v = enc_kvs[i]
            x = layer.forward_decoder(x, position_ids, enc_k, enc_v, attn_mask)
        return self.norm(x)

    def decode(
        self,
        canvas_ids: torch.Tensor,
        position_ids: torch.Tensor,
        enc_kvs: list[tuple[torch.Tensor, torch.Tensor]],
        self_conditioning_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One bidirectional denoise pass over the canvas, cross-attending ``enc_kvs``.

        Builds the self-conditioning soft-embedding from the previous step's logits
        (zeros at step 0) and delegates to :meth:`decode_with_soft`."""
        if self_conditioning_logits is not None:
            probs = self_conditioning_logits.softmax(dim=-1, dtype=torch.float32)
            soft = torch.matmul(probs.to(self.embed_tokens.weight.dtype), self.embed_tokens.weight)
            soft = soft * self.embed_tokens.embed_scale
        else:
            soft = torch.zeros(
                canvas_ids.shape[0], canvas_ids.shape[1], self.config.hidden_size,
                dtype=self.embed_tokens.weight.dtype, device=canvas_ids.device,
            )
        return self.decode_with_soft(canvas_ids, position_ids, enc_kvs, soft)


class DiffusionGemmaForBlockDiffusion(nn.Module):
    """Encoder → bidirectional decoder → tied LM head + final logit softcap.

    ``forward(input_ids, decoder_input_ids[, self_conditioning_logits])`` mirrors the
    HF deterministic forward used as the P1 parity anchor: encode the prompt, then a
    single denoise pass over the canvas. Returns ``(logits, encoder_last_hidden,
    decoder_last_hidden)`` (the two hidden states are for block-level parity)."""

    def __init__(self, cfg: DiffusionGemmaConfig) -> None:
        super().__init__()
        self.config = cfg
        self.model = DiffusionGemma(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight  # tied

    def _softcap(self, logits: torch.Tensor) -> torch.Tensor:
        sc = self.config.final_logit_softcapping
        logits = logits.float()
        if sc is not None:
            logits = torch.tanh(logits / sc) * sc
        return logits

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        self_conditioning_logits: torch.Tensor | None = None,
    ):
        sp = input_ids.shape[1]
        cl = decoder_input_ids.shape[1]
        device = input_ids.device
        enc_pos = torch.arange(sp, dtype=torch.int32, device=device).unsqueeze(0)
        dec_pos = torch.arange(sp, sp + cl, dtype=torch.int32, device=device).unsqueeze(0)

        enc_kvs, encoder_last_hidden = self.model.encode(input_ids, enc_pos)
        decoder_last_hidden = self.model.decode(
            decoder_input_ids, dec_pos, enc_kvs, self_conditioning_logits
        )
        logits = self._softcap(self.lm_head(decoder_last_hidden))
        return logits, encoder_last_hidden, decoder_last_hidden

    @classmethod
    def from_local(
        cls: type[Self],
        model_dir: str,
        *,
        num_layers: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Self:
        """Build from a local DiffusionGemma checkpoint dir (config.json + safetensors).

        ``num_layers`` truncates the backbone (smoke tests)."""
        d = json.loads((Path(model_dir) / "config.json").read_text())
        cfg = DiffusionGemmaConfig.from_hf_config(d)
        if num_layers is not None:
            cfg.num_hidden_layers = num_layers
            cfg.layer_types = cfg.layer_types[:num_layers]
        model = cls(cfg).to(dtype).eval()
        _load_hf_weights(model, model_dir, dtype=dtype)
        return model


def _load_hf_weights(
    model: DiffusionGemmaForBlockDiffusion, model_dir: str, *, dtype: torch.dtype = torch.float32
) -> None:
    """Load the canonical ``model.decoder.*`` weights into the shared backbone.

    The encoder text copy (``model.encoder.language_model.*``) is tied to the decoder
    for everything EXCEPT the per-layer ``layer_scalar`` buffer, which the checkpoint
    stores separately for each mode (they differ) — so the encoder's layer_scalars are
    loaded into ``enc_layer_scalar`` and the decoder's into ``dec_layer_scalar``. The
    vision tower is skipped (phase 1, text-only). The packed per-expert tensors are
    split into the SwitchGLU layout (mirrors qwen3_5_moe). ``lm_head`` ties to embed."""
    from safetensors import safe_open

    prefix = "model.decoder."
    enc_layer_prefix = "model.encoder.language_model.layers."
    inter = model.config.moe_intermediate_size
    n_layers = model.config.num_hidden_layers
    want = {n for n, _ in model.named_parameters()} | {n for n, _ in model.named_buffers()}

    sd: dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(str(Path(model_dir) / "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():  # noqa: SIM118
                # Encoder-mode per-layer residual scalar (the only non-tied text key).
                if k.startswith(enc_layer_prefix) and k.endswith(".layer_scalar"):
                    li = int(k[len(enc_layer_prefix):].split(".")[0])
                    if li < n_layers:
                        sd[f"model.layers.{li}.enc_layer_scalar"] = h.get_tensor(k).to(dtype)
                    continue
                if not k.startswith(prefix):
                    continue  # skip the rest of the tied encoder copy + vision tower
                local = "model." + k[len(prefix):]
                if local.endswith(".layer_scalar"):  # decoder-mode residual scalar
                    li = int(local.split(".layers.")[1].split(".")[0])
                    if li < n_layers:
                        sd[f"model.layers.{li}.dec_layer_scalar"] = h.get_tensor(k).to(dtype)
                    continue
                if ".layers." in local:
                    li = int(local.split(".layers.")[1].split(".")[0])
                    if li >= n_layers:
                        continue
                t = h.get_tensor(k)
                if t.is_floating_point():
                    t = t.to(dtype)
                # Accept both the HF Parameter layout (``.experts.gate_up_proj``) and the
                # MLX SwitchLinear layout (``.experts.gate_up_proj.weight``) so a dequantized
                # mlx checkpoint (e.g. the QAT-experts working weights) loads unchanged.
                if local.endswith((".experts.gate_up_proj.weight", ".experts.down_proj.weight")):
                    local = local[: -len(".weight")]
                if local.endswith(".experts.gate_up_proj"):  # [E, 2I, d] -> switch_mlp gate/up
                    base = local[: -len("experts.gate_up_proj")] + "switch_mlp"
                    sd[f"{base}.gate_proj.weight"] = t[:, :inter, :].unsqueeze(0).contiguous()
                    sd[f"{base}.up_proj.weight"] = t[:, inter:, :].unsqueeze(0).contiguous()
                    continue
                if local.endswith(".experts.down_proj"):  # [E, d, I] -> switch_mlp down
                    base = local[: -len("experts.down_proj")] + "switch_mlp"
                    sd[f"{base}.down_proj.weight"] = t.unsqueeze(0).contiguous()
                    continue
                if local in want:
                    sd[local] = t

    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.lm_head.weight = model.model.embed_tokens.weight  # re-tie after load

    real_missing = [m for m in missing if "inv_freq" not in m and "lm_head" not in m]
    if real_missing:
        raise RuntimeError(f"missing weights: {real_missing[:10]} (+{len(real_missing) - 10} more)")
    real_unexpected = [u for u in unexpected if "lm_head" not in u]
    if real_unexpected:
        raise RuntimeError(
            f"unexpected weights: {real_unexpected[:10]} (+{len(real_unexpected) - 10} more)"
        )
