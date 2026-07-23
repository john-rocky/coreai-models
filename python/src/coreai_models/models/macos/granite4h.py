# Granite 4.0-H text decoder (Mamba2 + attention hybrid) for the Core AI authoring path.
#
# Community port — NOT an Apple model.  Decode-only (S=1) stateful graph aimed at
# the pipelined GPU engine, following the qwen3_5.py loop-free pattern: at S=1 the
# Mamba2 selective scan collapses to a single recurrence step (the HF
# `use_precomputed_states` path), so the graph carries no while_loop and lowers on
# the MPSGraph GPU delegate.  Decoupled from transformers: weights load directly
# from the HF safetensors, config is a local dataclass.
#
# Layout per layer follows HF GraniteMoeHybrid exactly (dense h-350m/h-1b class):
#   - layer_types mixes "mamba" and "attention" (e.g. 350m: 28 mamba + 4 attention)
#   - all RMSNorms use the weight as-is (plain gain, NOT the (1+w) convention)
#   - the Mamba mixer's gated output norm applies the SiLU gate BEFORE the
#     normalization (HF GraniteMoeHybridRMSNormGated) — note this is the OPPOSITE
#     order of primitives.macos.rms_norm.RMSNormGated
#   - attention is NoPE (no positional embedding of any kind) GQA with a fixed
#     config scale (attention_multiplier), no q/k norm, no biases
#   - mup-style scalar multipliers: embedding_multiplier on the embed output,
#     residual_multiplier on every mixer/MLP residual branch, logits_scaling
#     divides the lm_head output
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_torch.composite_ops import RMSNormImpl

from coreai_models.primitives.macos.cache import KVCache, SSMState
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA


@dataclass
class Granite4HConfig:
    """Subset of GraniteMoeHybrid config needed for authoring (values from config.json).

    Dense hybrids only (num_local_experts == 0): the MoE branch is not authored.
    """

    hidden_size: int = 768
    num_hidden_layers: int = 32
    vocab_size: int = 100352
    shared_intermediate_size: int = 2048
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    # mup-style scalar multipliers
    embedding_multiplier: float = 12.0
    residual_multiplier: float = 0.246
    attention_multiplier: float = 0.015625
    logits_scaling: float = 3.0
    # attention (NoPE GQA)
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    # mamba2 mixer
    mamba_n_heads: int = 48
    mamba_d_head: int = 32
    mamba_d_state: int = 128
    mamba_n_groups: int = 1
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    mamba_conv_bias: bool = True
    layer_types: list[str] = field(default_factory=list)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def mamba_intermediate(self) -> int:
        return int(self.mamba_expand * self.hidden_size)

    @property
    def conv_dim(self) -> int:
        # [x | B | C] depthwise-conv channels
        return self.mamba_intermediate + 2 * self.mamba_n_groups * self.mamba_d_state

    @property
    def conv_state_width(self) -> int:
        # Minimal decode conv state = kernel-1 columns of projected xBC.
        return self.mamba_d_conv - 1

    def is_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "attention"

    @property
    def num_attn_layers(self) -> int:
        return sum(self.is_attention(i) for i in range(self.num_hidden_layers))

    @property
    def num_mamba_layers(self) -> int:
        return self.num_hidden_layers - self.num_attn_layers


# --------------------------------------------------------------------------- #
# NoPE GQA attention (fixed config scale, no q/k norm)
# --------------------------------------------------------------------------- #
class Granite4HAttention(nn.Module):
    def __init__(self, config: Granite4HConfig) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        d = config.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.sdpa = SDPA(scale=config.attention_multiplier, is_causal=True)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache,
        offset: int,
        seq_len: int,
        attn_idx: int,
    ) -> torch.Tensor:
        """Stateful attention: write the query's k/v at ``offset`` and fetch the
        full [0:seq_len] history; the lower-right causal SDPA lets the query
        block attend to all cached positions.  No RoPE — Granite-H is NoPE."""
        b, s, _ = x.shape
        H, HKV, D = self.n_heads, self.n_kv_heads, self.head_dim

        q = self.q_proj(x).view(b, s, H, D).transpose(1, 2)  # [b,H,s,D]
        k = self.k_proj(x).view(b, s, HKV, D).transpose(1, 2)
        v = self.v_proj(x).view(b, s, HKV, D).transpose(1, 2)

        k, v = kv_cache.update_and_fetch(attn_idx, offset, k, v, seq_len=seq_len, query_len=s)
        out = self.sdpa(q, k, v)  # GQA handled internally; [b,H,s,D]
        out = out.transpose(1, 2).reshape(b, s, H * D)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Mamba2 mixer — loop-free single-step (S=1) decode path
# --------------------------------------------------------------------------- #
class GraniteGatedRMSNorm(nn.Module):
    """Granite's gated norm body: callers gate (x * silu(z)) BEFORE this norm.
    Parameter name matches HF ``mamba.norm.weight`` (weight as-is)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self._rmsnorm_impl = RMSNormImpl(eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._rmsnorm_impl(x, self.weight)


class Granite4HMamba(nn.Module):
    """Mamba2 block, stateful decode step.

    The S=1 recurrence mirrors HF GraniteMoeHybridMambaLayer.torch_forward's
    ``use_precomputed_states`` path exactly (softplus dt, dA = exp(dt*A),
    state = state*dA + dt*B*x, y = (state*C).sum + D*x), computed in fp32 and
    written back in the state dtype.  The short causal conv consumes a
    [b, conv_dim, kernel-1] state column block and returns the shifted state
    (same convention as qwen3_5's GatedDeltaNet conv state).
    """

    def __init__(self, config: Granite4HConfig) -> None:
        super().__init__()
        self.n_heads = config.mamba_n_heads
        self.d_head = config.mamba_d_head
        self.d_state = config.mamba_d_state
        self.n_groups = config.mamba_n_groups
        self.intermediate = config.mamba_intermediate
        self.conv_dim = config.conv_dim
        self.kernel = config.mamba_d_conv
        self.groups_state = self.n_groups * self.d_state
        d = config.hidden_size

        self.in_proj = nn.Linear(
            d, self.intermediate + self.conv_dim + self.n_heads, bias=False
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, self.kernel,
            groups=self.conv_dim, padding=self.kernel - 1, bias=config.mamba_conv_bias,
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.n_heads))
        self.A_log = nn.Parameter(torch.zeros(self.n_heads))
        self.D = nn.Parameter(torch.zeros(self.n_heads))
        self.norm = GraniteGatedRMSNorm(self.intermediate, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.intermediate, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        conv_in: torch.Tensor,
        rec_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [b,1,hidden]; conv_in [b, conv_dim, kernel-1]; rec_in
        [b, n_heads, d_head, d_state].  Returns (out [b,1,hidden], new_conv,
        new_rec).  Decode-only: the SSM update is a single recurrence step."""
        b, s, _ = x.shape
        proj = self.in_proj(x)
        gate, xBC, dt = torch.split(
            proj, [self.intermediate, self.conv_dim, self.n_heads], dim=-1
        )

        # short causal depthwise conv over [state | current] columns
        w = torch.cat([conv_in, xBC.transpose(1, 2)], dim=-1)  # [b, conv_dim, (k-1)+s]
        conv = F.conv1d(
            w, self.conv1d.weight, bias=self.conv1d.bias, padding=0, groups=self.conv_dim
        )
        xBC_c = F.silu(conv).transpose(1, 2)  # [b, s, conv_dim]
        new_conv = w[..., -(self.kernel - 1):]

        xs, B, C = torch.split(
            xBC_c, [self.intermediate, self.groups_state, self.groups_state], dim=-1
        )

        # fp32 single-step SSM update (t = the one query position)
        dtv = F.softplus(dt[:, 0, :].float() + self.dt_bias.float())  # [b,h]
        A = -torch.exp(self.A_log.float())                            # [h]
        dA = torch.exp(dtv * A)                                       # [b,h]
        xh = xs[:, 0, :].float().reshape(b, self.n_heads, self.d_head)
        Bh = B[:, 0, :].float().reshape(b, self.n_groups, self.d_state)
        Ch = C[:, 0, :].float().reshape(b, self.n_groups, self.d_state)
        rep = self.n_heads // self.n_groups
        Bh = Bh.repeat_interleave(rep, dim=1, output_size=self.n_heads)
        Ch = Ch.repeat_interleave(rep, dim=1, output_size=self.n_heads)

        state = rec_in.to(torch.float32) * dA[..., None, None] + (
            (dtv[..., None] * Bh)[:, :, None, :] * xh[..., None]
        )  # [b,h,p,n]
        y = (state * Ch[:, :, None, :]).sum(dim=-1)                   # [b,h,p]
        y = y + xh * self.D.float()[..., None]
        y = y.reshape(b, 1, self.intermediate)

        # SiLU gate BEFORE the norm (HF order), then the rms_norm composite
        gated = (y * F.silu(gate.float())).to(x.dtype)
        out = self.out_proj(self.norm(gated))
        return out, new_conv, state.to(rec_in.dtype)


# --------------------------------------------------------------------------- #
# Dense MLP (HF shared_mlp: fused [gate|up] input_linear, chunked)
# --------------------------------------------------------------------------- #
class Granite4HMLP(nn.Module):
    def __init__(self, config: Granite4HConfig) -> None:
        super().__init__()
        self.input_linear = nn.Linear(
            config.hidden_size, config.shared_intermediate_size * 2, bias=False
        )
        self.output_linear = nn.Linear(
            config.shared_intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.input_linear(x).chunk(2, dim=-1)
        return self.output_linear(F.silu(gate) * up)


# --------------------------------------------------------------------------- #
# Unified decoder layer — dispatches mamba vs attention by layer type
# --------------------------------------------------------------------------- #
class Granite4HDecoderLayer(nn.Module):
    def __init__(self, config: Granite4HConfig, layer_idx: int) -> None:
        super().__init__()
        d = config.hidden_size
        self.is_attention = config.is_attention(layer_idx)
        self.residual_multiplier = config.residual_multiplier
        self.input_layernorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(d, eps=config.rms_norm_eps)
        if self.is_attention:
            self.self_attn = Granite4HAttention(config)
        else:
            self.mamba = Granite4HMamba(config)
        self.shared_mlp = Granite4HMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache | None = None,
        conv_cache: SSMState | None = None,
        rec_cache: SSMState | None = None,
        offset: int = 0,
        seq_len: int | None = None,
        attn_idx: int = 0,
        mamba_idx: int = 0,
    ) -> torch.Tensor:
        normed = self.input_layernorm(x)
        if self.is_attention:
            r = self.self_attn(normed, kv_cache, offset, seq_len, attn_idx)
            h = x + r * self.residual_multiplier
        else:
            conv_in = conv_cache.states.narrow(0, mamba_idx, 1).squeeze(0)
            rec_in = rec_cache.states.narrow(0, mamba_idx, 1).squeeze(0)
            r, new_conv, new_rec = self.mamba(normed, conv_in, rec_in)
            h = x + r * self.residual_multiplier
            conv_cache.update_states(mamba_idx, new_conv)
            rec_cache.update_states(mamba_idx, new_rec)
        r = self.shared_mlp(self.post_attention_layernorm(h))
        return h + r * self.residual_multiplier


# --------------------------------------------------------------------------- #
# Full decoder stack (stateful decode; hybrid state indexing)
# --------------------------------------------------------------------------- #
class Granite4HModel(nn.Module):
    def __init__(self, config: Granite4HConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Granite4HDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward_stateful(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        conv_cache: SSMState,
        rec_cache: SSMState,
    ) -> torch.Tensor:
        """``input_ids`` carries the query tokens ([b, query_len]);
        ``position_ids`` carries the full positions ([b, seq_len]) so
        ``offset = seq_len - query_len``.  NoPE: position_ids contributes only
        its length (the KV write offset) — no rotary math.  Attention layers
        index ``kv_cache`` by attention-layer counter; mamba layers index the
        conv/rec caches by mamba-layer counter."""
        query_len = input_ids.shape[1]
        seq_len = position_ids.shape[1]
        offset = seq_len - query_len
        h = self.embed_tokens(input_ids) * self.config.embedding_multiplier
        attn_idx = 0
        mamba_idx = 0
        for layer in self.layers:
            if layer.is_attention:
                h = layer(h, kv_cache=kv_cache, offset=offset, seq_len=seq_len,
                          attn_idx=attn_idx)
                attn_idx += 1
            else:
                h = layer(h, conv_cache=conv_cache, rec_cache=rec_cache,
                          mamba_idx=mamba_idx)
                mamba_idx += 1
        return self.norm(h)


def build_decode_state(
    config: Granite4HConfig,
    max_seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Allocate the hybrid decode state tensors (all zero) for a fresh sequence.

    Hybrid layout — only the layers that need a given state get a slot:
      * k_cache / v_cache: [num_attn_layers, 1, n_kv_heads, max_seq_len, head_dim]
      * conv_state:        [num_mamba_layers, 1, conv_dim, kernel-1]
      * rec_state:         [num_mamba_layers, 1, n_heads, d_head, d_state]
    """
    na, nm = config.num_attn_layers, config.num_mamba_layers
    return {
        "k_cache": torch.zeros(na, 1, config.num_key_value_heads, max_seq_len,
                               config.head_dim, dtype=dtype),
        "v_cache": torch.zeros(na, 1, config.num_key_value_heads, max_seq_len,
                               config.head_dim, dtype=dtype),
        "conv_state": torch.zeros(nm, 1, config.conv_dim, config.conv_state_width,
                                  dtype=dtype),
        "rec_state": torch.zeros(nm, 1, config.mamba_n_heads, config.mamba_d_head,
                                 config.mamba_d_state, dtype=dtype),
    }


# Same state-name contract as qwen3_5 (rides the same engine extra-states patch).
DECODE_STATE_NAMES = ("keyCache", "valueCache", "convState", "recState")


class Granite4HForCausalLMStateful(nn.Module):
    """Stateful decode-only text decoder: embed + 32 hybrid layers + tied head.

    forward inputs: input_ids [b, query_len], position_ids [b, seq_len], plus
    the four state tensors from ``build_decode_state`` (mutated in place,
    surfaced as Core AI states via ``DECODE_STATE_NAMES``).  Logits are divided
    by ``logits_scaling`` (HF semantics).
    """

    def __init__(self, config: Granite4HConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Granite4HModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
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
        return self.lm_head(h) / self.config.logits_scaling

    @classmethod
    def from_hf(
        cls,
        huggingface_model_id: str,
        target_dtype: torch.dtype = torch.float16,
        num_layers: int | None = None,
    ) -> "Granite4HForCausalLMStateful":
        """Load a dense GraniteMoeHybrid checkpoint straight from safetensors
        (no transformers modeling dependency; config.json parsed directly)."""
        import glob
        import json
        import os

        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        from coreai_models.models.base import _is_layer_key_beyond

        model_dir = snapshot_download(
            huggingface_model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
        raw = json.loads(open(os.path.join(model_dir, "config.json")).read())
        config = granite4h_config_from_dict(raw, num_layers=num_layers)

        with torch.device("meta"):
            model = cls(config)
        model = model.to(dtype=target_dtype)

        files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"No .safetensors files in {model_dir}")
        sd: dict[str, torch.Tensor] = {}
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if num_layers is not None and _is_layer_key_beyond(key, num_layers):
                        continue
                    tensor = f.get_tensor(key)
                    if tensor.dtype != target_dtype:
                        tensor = tensor.to(target_dtype)
                    sd[key] = tensor

        model.load_state_dict(sd, assign=True, strict=False)
        if config.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight

        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(f"Parameters not loaded: {meta_params}")
        return model


def granite4h_config_from_dict(raw: dict, num_layers: int | None = None) -> Granite4HConfig:
    """Build the authoring config from a parsed HF config.json dict."""
    if raw.get("num_local_experts", 0) > 0:
        raise ValueError("MoE GraniteMoeHybrid variants are not supported (dense only)")
    if raw.get("position_embedding_type", "nope") != "nope":
        raise ValueError("Only NoPE attention is authored (got "
                         f"{raw.get('position_embedding_type')!r})")
    layer_types = list(raw["layer_types"])
    n_layers = num_layers if num_layers is not None else raw["num_hidden_layers"]
    cfg = Granite4HConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=n_layers,
        vocab_size=raw["vocab_size"],
        shared_intermediate_size=raw["shared_intermediate_size"],
        rms_norm_eps=raw["rms_norm_eps"],
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", True)),
        embedding_multiplier=raw["embedding_multiplier"],
        residual_multiplier=raw["residual_multiplier"],
        attention_multiplier=raw["attention_multiplier"],
        logits_scaling=raw["logits_scaling"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        mamba_n_heads=raw["mamba_n_heads"],
        mamba_d_head=raw["mamba_d_head"],
        mamba_d_state=raw["mamba_d_state"],
        mamba_n_groups=raw["mamba_n_groups"],
        mamba_d_conv=raw["mamba_d_conv"],
        mamba_expand=raw["mamba_expand"],
        mamba_conv_bias=raw["mamba_conv_bias"],
        layer_types=layer_types[:n_layers],
    )
    assert cfg.mamba_intermediate == cfg.mamba_n_heads * cfg.mamba_d_head, (
        "mamba_expand*hidden must equal n_heads*d_head"
    )
    return cfg
