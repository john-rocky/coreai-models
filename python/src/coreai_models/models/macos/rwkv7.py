# RWKV-7 "Goose" (pure-recurrent / linear-attention LLM) for the Core AI
# authoring path.
#
# Community port — NOT an Apple model. Authored to be exportable via
# coreai_models.export.macos.export_to_coreai. Decoupled from transformers /
# flash-linear-attention: weights load directly from the HF safetensors, config
# is parsed from the raw config.json, and the per-token WKV7 recurrence is
# transcribed to STANDARD torch ops (no fla, no triton, no custom Metal kernel).
#
# RWKV-7 has NO attention and NO KV cache. Every layer is a time-mixing (WKV7
# delta-rule matrix state) + channel-mixing (sqrelu MLP) block. The decode graph
# is therefore loop-free at S=1 and carries only TWO fixed-shape states (the
# O(1)-per-token win — constant memory, unbounded context):
#   * rec_state  [num_layers, 1, num_heads, head_dim, head_dim]   (the WKV7 matrix S)
#   * shift_state[num_layers, 1, 2, hidden]                       (token-shift prev
#       hidden: slot 0 = time-mix, slot 1 = channel-mix)
#
# Per-token WKV7 recurrence (per head, S = [K, V] = [head_dim, head_dim]), 1:1
# with fla/ops/rwkv7/fused_recurrent.py decode kernel:
#     S = diag(exp w)·S - (kk·a)·(kkᵀ S);  S += k⊗v;  o = rᵀ S
# i.e. a per-head rank-1 delta + diagonal decay + outer-product write + matvec
# read. All small [64,64] ops × 32 heads -> standard ops, exports without a
# custom composite (cf. granite4h's pure-torch Mamba2 step).
#
# IMPORTANT (macOS-27-beta GPU delegate, the lfm2.py lesson): per-layer
# SSMState.update_states calls compile to a read/slice_update/write round trip on
# the SAME state handle, and the GPU delegate silently DROPS all but one when
# there is more than one. The blocks therefore RETURN their new (rec, shift)
# slices and forward_stateful issues ONE fused full-state write per state tensor
# at the end of the step (disjoint slots, each layer reads only its own slot
# before any write -> identical semantics).
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from coreai_models.primitives._ops import mutable_slice_update

# w = -exp(-0.5) * sigmoid(w_lora(xw)) : maps the decay LoRA to a log-decay in
# (-0.6065, 0) so exp(w) in (0.545, 1). The constant is RWKV-7's fixed -exp(-0.5).
W_DECAY = -0.6065306597126334


@dataclass
class RWKV7Config:
    """Subset of the RWKV-7 config needed for authoring (from config.json)."""

    hidden_size: int = 2048
    num_hidden_layers: int = 24
    vocab_size: int = 65536
    head_dim: int = 64
    intermediate_size: int = 8192
    norm_eps: float = 1e-5
    decay_low_rank_dim: int = 96
    a_low_rank_dim: int = 96
    v_low_rank_dim: int = 64
    gate_low_rank_dim: int = 256
    tie_word_embeddings: bool = False

    @property
    def num_heads(self) -> int:
        return self.hidden_size // self.head_dim

    @property
    def gn_eps(self) -> float:
        # fla sets the per-head GroupNorm eps to head_dim * norm_eps.
        return self.head_dim * self.norm_eps


def rwkv7_config_from_dict(raw: dict) -> RWKV7Config:
    return RWKV7Config(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        vocab_size=raw["vocab_size"],
        head_dim=raw["head_dim"],
        intermediate_size=raw["intermediate_size"],
        norm_eps=raw.get("norm_eps", 1e-5),
        decay_low_rank_dim=raw["decay_low_rank_dim"],
        a_low_rank_dim=raw["a_low_rank_dim"],
        v_low_rank_dim=raw["v_low_rank_dim"],
        gate_low_rank_dim=raw["gate_low_rank_dim"],
        tie_word_embeddings=raw.get("tie_word_embeddings", False),
    )


class LoRA(nn.Module):
    """fla LoRA: ``act(x @ A^T) @ B^T + b``. Module hierarchy matches the HF
    checkpoint keys (``.lora.0.weight``, ``.lora.2.weight``, ``.lora.2.bias``)."""

    def __init__(self, in_dim: int, out_dim: int, low: int, act: str | None, bias: bool):
        super().__init__()
        a = {"tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), None: nn.Identity()}[act]
        self.lora = nn.Sequential(
            nn.Linear(in_dim, low, bias=False), a, nn.Linear(low, out_dim, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora(x)


class RWKV7Attention(nn.Module):
    """Time-mixing (WKV7). Names mirror the HF checkpoint exactly."""

    def __init__(self, config: RWKV7Config, layer_idx: int):
        super().__init__()
        H, self.layer_idx = config.hidden_size, layer_idx
        self.NH, self.HD = config.num_heads, config.head_dim
        self.x_r = nn.Parameter(torch.zeros(1, 1, H))
        self.x_w = nn.Parameter(torch.zeros(1, 1, H))
        self.x_k = nn.Parameter(torch.zeros(1, 1, H))
        self.x_v = nn.Parameter(torch.zeros(1, 1, H))
        self.x_a = nn.Parameter(torch.zeros(1, 1, H))
        self.x_g = nn.Parameter(torch.zeros(1, 1, H))
        self.k_k = nn.Parameter(torch.zeros(H))
        self.k_a = nn.Parameter(torch.zeros(H))
        self.r_k = nn.Parameter(torch.zeros(self.NH, self.HD))
        self.r_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, H, bias=False)
        self.v_proj = nn.Linear(H, H, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)
        self.w_lora = LoRA(H, H, config.decay_low_rank_dim, "tanh", bias=True)
        if layer_idx != 0:
            self.v_lora = LoRA(H, H, config.v_low_rank_dim, None, bias=True)
        self.a_lora = LoRA(H, H, config.a_low_rank_dim, None, bias=True)
        self.g_lora = LoRA(H, H, config.gate_low_rank_dim, "sigmoid", bias=False)
        self.g_norm = nn.GroupNorm(self.NH, H, eps=config.gn_eps, affine=True)

    def forward(self, h: torch.Tensor, rec_in: torch.Tensor, shift_attn: torch.Tensor,
                v_first: torch.Tensor | None):
        """h [b,H] = attn_norm output for this token; rec_in [b,NH,HD,HD];
        shift_attn [b,H] = previous token's attn_norm output.
        Returns (o [b,H], new_rec [b,NH,HD,HD], new_shift_attn [b,H], v_first)."""
        b, H, NH, HD = h.shape[0], h.shape[1], self.NH, self.HD
        delta = shift_attn - h                              # token_shift: prev - cur
        new_shift_attn = h
        xr = h + delta * self.x_r.view(-1)
        xw = h + delta * self.x_w.view(-1)
        xk = h + delta * self.x_k.view(-1)
        xv = h + delta * self.x_v.view(-1)
        xa = h + delta * self.x_a.view(-1)
        xg = h + delta * self.x_g.view(-1)

        r = self.r_proj(xr)
        w = W_DECAY * torch.sigmoid(self.w_lora(xw))
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, torch.sigmoid(self.v_lora(xv)))
        a = torch.sigmoid(self.a_lora(xa))
        gate = self.g_lora(xg)

        kk = F.normalize((k * self.k_k).view(b, NH, HD), dim=-1, p=2.0)   # from ORIGINAL k
        k = k * (1.0 + (a - 1.0) * self.k_a)                             # fused_k (overwrite)

        # WKV7 recurrence in fp32 (per head; S = [K, V]).
        rh = r.view(b, NH, HD).float()
        wh = w.view(b, NH, HD).float()
        kh = k.view(b, NH, HD).float()
        ah = a.view(b, NH, HD).float()
        vh = v.view(b, NH, HD).float()
        kkf = kk.float()
        S = rec_in.float()                                  # [b,NH,K,V]
        ew = torch.exp(wh)                                  # decay in (0,1)
        akk = -kkf                                          # a_op
        bb = kkf * ah                                       # b_op
        tmp = (akk.unsqueeze(-1) * S).sum(2)                # [b,NH,V] = akkᵀ S
        S = ew.unsqueeze(-1) * S + bb.unsqueeze(-1) * tmp.unsqueeze(2)
        S = S + kh.unsqueeze(-1) * vh.unsqueeze(2)          # write k ⊗ v
        o = (S * rh.unsqueeze(-1)).sum(2)                   # [b,NH,V] = rᵀ S
        new_rec = S.to(rec_in.dtype)

        o = F.group_norm(o.reshape(b, H), NH, self.g_norm.weight.float(),
                         self.g_norm.bias.float(), self.g_norm.eps)
        # gate output correction: (o + (Σ_d r·k·r_k)·v) · g
        corr = ((rh * kh * self.r_k.float()).sum(-1, keepdim=True) * vh).reshape(b, H)
        o = ((o + corr) * gate.float()).to(h.dtype)
        o = self.o_proj(o)
        return o, new_rec, new_shift_attn, v_first


class RWKV7FeedForward(nn.Module):
    """Channel-mixing: token-shift + sqrelu MLP."""

    def __init__(self, config: RWKV7Config):
        super().__init__()
        H, I = config.hidden_size, config.intermediate_size
        self.x_k = nn.Parameter(torch.zeros(H))
        self.key = nn.Linear(H, I, bias=False)
        self.value = nn.Linear(I, H, bias=False)

    def forward(self, h: torch.Tensor, shift_ffn: torch.Tensor):
        """h [b,H] = ffn_norm output; shift_ffn [b,H] = previous token's ffn_norm output."""
        delta = shift_ffn - h
        xk = h + delta * self.x_k
        inner = torch.relu(self.key(xk)) ** 2               # sqrelu
        return self.value(inner), h


class RWKV7Block(nn.Module):
    def __init__(self, config: RWKV7Config, layer_idx: int):
        super().__init__()
        H, eps = config.hidden_size, config.norm_eps
        self.layer_idx = layer_idx
        if layer_idx == 0:                                  # RWKV ln0 (norm_first, layer 0 only)
            self.pre_norm = nn.LayerNorm(H, eps=eps, bias=True)
        self.attn_norm = nn.LayerNorm(H, eps=eps, bias=True)
        self.attn = RWKV7Attention(config, layer_idx)
        self.ffn_norm = nn.LayerNorm(H, eps=eps, bias=True)
        self.ffn = RWKV7FeedForward(config)

    def forward(self, x: torch.Tensor, rec_in: torch.Tensor, shift_in: torch.Tensor,
                v_first: torch.Tensor | None):
        """x [b,H] running residual; rec_in [b,NH,HD,HD]; shift_in [b,2,H].
        Returns (x [b,H], new_rec [b,NH,HD,HD], new_shift [b,2,H], v_first)."""
        if self.layer_idx == 0:
            x = self.pre_norm(x)
        residual = x
        h = self.attn_norm(x)
        o, new_rec, new_sa, v_first = self.attn(h, rec_in, shift_in[:, 0], v_first)
        x = residual + o
        residual = x
        h2 = self.ffn_norm(x)
        f, new_sf = self.ffn(h2, shift_in[:, 1])
        x = residual + f
        new_shift = torch.stack([new_sa, new_sf], dim=1)    # [b,2,H]
        return x, new_rec, new_shift, v_first


class RWKV7Model(nn.Module):
    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [RWKV7Block(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=True)

    def forward_stateful(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                         rec_state: torch.Tensor, shift_state: torch.Tensor) -> torch.Tensor:
        """Stateful S=1 decode. ``input_ids`` [b, 1]; ``position_ids`` is accepted
        for engine-input parity but UNUSED (RWKV-7 is positionless). Each block
        reads its own rec/shift slot; the updated slices are written back in ONE
        fused slice_update per state at the end (GPU-delegate-safe). Decode-only:
        one token per call, so cross-token state threading happens via the caller
        (each call reads then writes the state) — no inner scan, loop-free graph.
        """
        x = self.embeddings(input_ids)[:, 0]                # [b, H] (query_len == 1)
        v_first = None
        new_recs: list[torch.Tensor] = []
        new_shifts: list[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            rec_in = rec_state.narrow(0, i, 1).squeeze(0)       # [b,NH,HD,HD]
            shift_in = shift_state.narrow(0, i, 1).squeeze(0)   # [b,2,H]
            x, new_rec, new_shift, v_first = layer(x, rec_in, shift_in, v_first)
            new_recs.append(new_rec)
            new_shifts.append(new_shift)

        new_rec_full = torch.stack(new_recs, dim=0)         # [nl, b, NH, HD, HD]
        new_shift_full = torch.stack(new_shifts, dim=0)     # [nl, b, 2, H]
        zero = torch.tensor((0,), dtype=torch.int32)
        mutable_slice_update(
            x=rec_state, update=new_rec_full,
            begin=torch.cat([zero] * rec_state.dim()),
            end=torch.tensor(tuple(rec_state.shape), dtype=torch.int32),
        )
        mutable_slice_update(
            x=shift_state, update=new_shift_full,
            begin=torch.cat([zero] * shift_state.dim()),
            end=torch.tensor(tuple(shift_state.shape), dtype=torch.int32),
        )
        return self.norm(x).unsqueeze(1)                    # [b, 1, H]


def build_decode_state(config: RWKV7Config, max_seq_len: int | None = None,
                       dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    """Allocate the (zero) RWKV-7 decode states. No KV cache — the O(1) win.
    ``max_seq_len`` is accepted for API symmetry but unused (states are fixed)."""
    nl, NH, HD, H = (config.num_hidden_layers, config.num_heads,
                     config.head_dim, config.hidden_size)
    return {
        "rec_state": torch.zeros(nl, 1, NH, HD, HD, dtype=dtype),
        "shift_state": torch.zeros(nl, 1, 2, H, dtype=dtype),
    }


# Surfaced via export_to_coreai(state_names=...). RWKV-7 carries NO keyCache/
# valueCache — two fixed-shape states only (within the ≤2 extra-state budget).
DECODE_STATE_NAMES = ("recState", "shiftState")


class RWKV7ForCausalLMStateful(nn.Module):
    """Stateful decode graph: input_ids [b, query_len], position_ids [b, seq_len]
    (unused), rec_state, shift_state -> logits. States mutate in place."""

    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.config = config
        self.model = RWKV7Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embeddings.weight
        self.last_token_only = False

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                rec_state: torch.Tensor, shift_state: torch.Tensor) -> torch.Tensor:
        h = self.model.forward_stateful(input_ids, position_ids, rec_state, shift_state)
        if self.last_token_only:
            h = h[:, -1:, :]
        return self.lm_head(h)


def _resolve_snapshot(model_id_or_path: str) -> str:
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    root = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{model_id_or_path.replace('/', '--')}/snapshots")
    snaps = sorted(glob.glob(os.path.join(root, "*")))
    if not snaps:
        raise FileNotFoundError(f"no local snapshot for {model_id_or_path}; download it first")
    return snaps[-1]


def rwkv7_from_hf(model_id_or_path: str = "RWKV/RWKV7-Goose-World3-1.5B-HF",
                  target_dtype: torch.dtype = torch.float32) -> RWKV7ForCausalLMStateful:
    """Load RWKV-7 from a local HF snapshot into the authored stateful module.
    Module hierarchy mirrors the checkpoint, so load_state_dict is strict."""
    snap = _resolve_snapshot(model_id_or_path)
    config = rwkv7_config_from_dict(json.load(open(os.path.join(snap, "config.json"))))

    with torch.device("meta"):
        model = RWKV7ForCausalLMStateful(config)
    model = model.to(dtype=target_dtype)

    sd = {}
    for s in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        sd.update(load_file(s))
    # The checkpoint may pack the 6 token-shift mixers as one [6,1,1,H] x_x.
    for i in range(config.num_hidden_layers):
        xx = sd.pop(f"model.layers.{i}.attn.x_x", None)
        if xx is not None:
            for j, n in enumerate(("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")):
                sd[f"model.layers.{i}.attn.{n}"] = xx[j].reshape(1, 1, -1)
    sd = {k: v.to(target_dtype) for k, v in sd.items()}
    model.load_state_dict(sd, assign=True, strict=True)
    model.eval()
    return model
