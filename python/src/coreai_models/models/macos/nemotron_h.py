# NVIDIA Nemotron-3-Nano (Mamba2 + attention + MLP hybrid) for the Core AI authoring path.
#
# Community port — NOT an Apple model.  Decode-only (S=1) stateful graph aimed at the
# pipelined GPU engine, following granite4h.py: at S=1 the Mamba2 selective scan
# collapses to a single recurrence step (the HF `use_precomputed_states` path), so the
# graph carries no while_loop and lowers on the MPSGraph GPU delegate.  Decoupled from
# transformers: weights load straight from the HF safetensors, config is a local
# dataclass.
#
# Layout per layer follows HF NemotronH exactly (dense "4B" class):
#   - `hybrid_override_pattern` gives one mixer per block: M=mamba, -=mlp, *=attention
#     (4B: 21 mamba + 17 mlp + 4 attention).  A block is norm -> mixer -> residual;
#     there is no second (MLP) branch inside a mamba/attention block.
#   - all RMSNorms use the weight as-is (plain gain, NOT the (1+w) convention)
#   - the mamba gated output norm is Zamba2RMSNormGated: gate with SiLU BEFORE the
#     normalization, and normalize per group of `mamba_intermediate // n_groups`
#   - dt is clamped from below at `time_step_min` after the softplus
#   - MLP is up_proj -> relu^2 -> down_proj (no gate branch)
#   - attention is NoPE GQA with the plain head_dim**-0.5 scale, no q/k norm, no biases
#   - no mup-style multipliers, untied lm_head
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives.macos.cache import KVCache, SSMState
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA

_PATTERN = {"M": "mamba", "-": "mlp", "*": "attention", "E": "moe"}


@dataclass
class NemotronHConfig:
    """Subset of the HF NemotronH config needed for authoring (values from config.json).

    Dense hybrids only: the MoE ("E") mixer is not authored.
    """

    hidden_size: int = 3136
    num_hidden_layers: int = 42
    vocab_size: int = 131072
    intermediate_size: int = 12544
    layer_norm_epsilon: float = 1e-5
    tie_word_embeddings: bool = False
    # attention (NoPE GQA, explicit head_dim)
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    # mamba2 mixer
    mamba_num_heads: int = 96
    mamba_head_dim: int = 80
    ssm_state_size: int = 128
    n_groups: int = 8
    conv_kernel: int = 4
    use_conv_bias: bool = True
    time_step_min: float = 0.001
    layer_types: list[str] = field(default_factory=list)

    @property
    def mamba_intermediate(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def conv_dim(self) -> int:
        # [x | B | C] depthwise-conv channels
        return self.mamba_intermediate + 2 * self.n_groups * self.ssm_state_size

    @property
    def conv_state_width(self) -> int:
        # Minimal decode conv state = kernel-1 columns of projected xBC.
        return self.conv_kernel - 1

    @property
    def group_size(self) -> int:
        return self.mamba_intermediate // self.n_groups

    def block_type(self, layer_idx: int) -> str:
        return self.layer_types[layer_idx]

    def is_attention(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "attention"

    def is_mamba(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "mamba"

    @property
    def num_attn_layers(self) -> int:
        return sum(self.is_attention(i) for i in range(self.num_hidden_layers))

    @property
    def num_mamba_layers(self) -> int:
        return sum(self.is_mamba(i) for i in range(self.num_hidden_layers))


# --------------------------------------------------------------------------- #
# NoPE GQA attention (head_dim**-0.5 scale, no q/k norm)
# --------------------------------------------------------------------------- #
class NemotronHAttention(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        d = config.hidden_size
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.sdpa = SDPA(scale=self.head_dim**-0.5, is_causal=True)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVCache,
        offset: int,
        seq_len: int,
        attn_idx: int,
    ) -> torch.Tensor:
        """Write the query's k/v at ``offset`` and fetch the full [0:seq_len] history;
        the lower-right causal SDPA lets the query block attend to all cached
        positions.  No RoPE — Nemotron-H is NoPE."""
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
class NemotronHGatedRMSNorm(nn.Module):
    """HF Zamba2RMSNormGated: SiLU-gate first, then RMS-normalize within each group of
    ``group_size`` channels.  Written out explicitly rather than via the RMSNorm
    composite, whose gain is applied over the whole last dim."""

    def __init__(self, dim: int, group_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.group_size = group_size
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        b, s, d = x.shape
        h = x.float() * F.silu(gate.float())
        h = h.view(b, s, d // self.group_size, self.group_size)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.view(b, s, d).to(dtype)


class NemotronHMamba(nn.Module):
    """Mamba2 block, stateful decode step.

    The S=1 recurrence mirrors HF NemotronHMamba2Mixer.torch_forward's
    ``use_precomputed_states`` path exactly (softplus dt, clamp at time_step_min,
    dA = exp(dt*A), state = state*dA + dt*B*x, y = (state*C).sum + D*x), computed in
    fp32 and written back in the state dtype.  The short causal conv consumes a
    [b, conv_dim, kernel-1] state column block and returns the shifted state.
    """

    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.n_heads = config.mamba_num_heads
        self.d_head = config.mamba_head_dim
        self.d_state = config.ssm_state_size
        self.n_groups = config.n_groups
        self.intermediate = config.mamba_intermediate
        self.conv_dim = config.conv_dim
        self.kernel = config.conv_kernel
        self.groups_state = self.n_groups * self.d_state
        self.time_step_min = config.time_step_min
        d = config.hidden_size

        self.in_proj = nn.Linear(
            d, self.intermediate + self.conv_dim + self.n_heads, bias=False
        )
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, self.kernel,
            groups=self.conv_dim, padding=self.kernel - 1, bias=config.use_conv_bias,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.n_heads))
        self.A_log = nn.Parameter(torch.zeros(self.n_heads))
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.norm = NemotronHGatedRMSNorm(
            self.intermediate, config.group_size, eps=config.layer_norm_epsilon
        )
        self.out_proj = nn.Linear(self.intermediate, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        conv_in: torch.Tensor,
        rec_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [b,1,hidden]; conv_in [b, conv_dim, kernel-1]; rec_in
        [b, n_heads, d_head, d_state].  Returns (out [b,1,hidden], new_conv, new_rec)."""
        b, _, _ = x.shape
        proj = self.in_proj(x)
        gate, xBC, dt = torch.split(
            proj, [self.intermediate, self.conv_dim, self.n_heads], dim=-1
        )

        # short causal depthwise conv over [state | current] columns
        w = torch.cat([conv_in, xBC.transpose(1, 2)], dim=-1)  # [b, conv_dim, (k-1)+1]
        conv = F.conv1d(
            w, self.conv1d.weight, bias=self.conv1d.bias, padding=0, groups=self.conv_dim
        )
        xBC_c = F.silu(conv).transpose(1, 2)  # [b, 1, conv_dim]
        new_conv = w[..., -(self.kernel - 1):]

        xs, B, C = torch.split(
            xBC_c, [self.intermediate, self.groups_state, self.groups_state], dim=-1
        )

        # fp32 single-step SSM update (t = the one query position)
        dtv = F.softplus(dt[:, 0, :].float() + self.dt_bias.float())  # [b,h]
        dtv = torch.clamp(dtv, min=self.time_step_min)
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

        out = self.out_proj(self.norm(y.to(x.dtype), gate))
        return out, new_conv, state.to(rec_in.dtype)


# --------------------------------------------------------------------------- #
# Dense MLP (up -> relu^2 -> down; no gate branch)
# --------------------------------------------------------------------------- #
class NemotronHMLP(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.square(F.relu(self.up_proj(x))))


# --------------------------------------------------------------------------- #
# One block = norm -> mixer -> residual (exactly one mixer per block)
# --------------------------------------------------------------------------- #
class NemotronHBlock(nn.Module):
    def __init__(self, config: NemotronHConfig, layer_idx: int) -> None:
        super().__init__()
        self.block_type = config.block_type(layer_idx)
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        if self.block_type == "attention":
            self.mixer = NemotronHAttention(config)
        elif self.block_type == "mamba":
            self.mixer = NemotronHMamba(config)
        elif self.block_type == "mlp":
            self.mixer = NemotronHMLP(config)
        else:
            raise ValueError(f"unsupported mixer {self.block_type!r} (MoE is not authored)")

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
        normed = self.norm(x)
        if self.block_type == "attention":
            return x + self.mixer(normed, kv_cache, offset, seq_len, attn_idx)
        if self.block_type == "mamba":
            conv_in = conv_cache.states.narrow(0, mamba_idx, 1).squeeze(0)
            rec_in = rec_cache.states.narrow(0, mamba_idx, 1).squeeze(0)
            r, new_conv, new_rec = self.mixer(normed, conv_in, rec_in)
            conv_cache.update_states(mamba_idx, new_conv)
            rec_cache.update_states(mamba_idx, new_rec)
            return x + r
        return x + self.mixer(normed)


# --------------------------------------------------------------------------- #
# Full decoder stack (stateful decode; hybrid state indexing)
# --------------------------------------------------------------------------- #
class NemotronHBackbone(nn.Module):
    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [NemotronHBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward_stateful(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        conv_cache: SSMState,
        rec_cache: SSMState,
    ) -> torch.Tensor:
        """``input_ids`` carries the query tokens ([b, query_len]); ``position_ids``
        carries the full positions ([b, seq_len]) so ``offset = seq_len - query_len``.
        NoPE: position_ids contributes only its length (the KV write offset)."""
        query_len = input_ids.shape[1]
        seq_len = position_ids.shape[1]
        offset = seq_len - query_len
        h = self.embeddings(input_ids)
        attn_idx = 0
        mamba_idx = 0
        for layer in self.layers:
            if layer.block_type == "attention":
                h = layer(h, kv_cache=kv_cache, offset=offset, seq_len=seq_len,
                          attn_idx=attn_idx)
                attn_idx += 1
            elif layer.block_type == "mamba":
                h = layer(h, conv_cache=conv_cache, rec_cache=rec_cache,
                          mamba_idx=mamba_idx)
                mamba_idx += 1
            else:
                h = layer(h)
        return self.norm_f(h)


def build_decode_state(
    config: NemotronHConfig,
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
        "rec_state": torch.zeros(nm, 1, config.mamba_num_heads, config.mamba_head_dim,
                                 config.ssm_state_size, dtype=dtype),
    }


# Same state-name contract as granite4h / qwen3_5 (rides the engine extra-states patch).
DECODE_STATE_NAMES = ("keyCache", "valueCache", "convState", "recState")


class NemotronHForCausalLMStateful(nn.Module):
    """Stateful decode-only text decoder: embed + hybrid blocks + untied head.

    forward inputs: input_ids [b, query_len], position_ids [b, seq_len], plus the four
    state tensors from ``build_decode_state`` (mutated in place, surfaced as Core AI
    states via ``DECODE_STATE_NAMES``).
    """

    def __init__(self, config: NemotronHConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = NemotronHBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embeddings.weight
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
        h = self.backbone.forward_stateful(input_ids, position_ids, kv, conv, rec)
        if self.last_token_only:
            h = h[:, -1:, :]
        return self.lm_head(h)

    @classmethod
    def from_hf(
        cls,
        huggingface_model_id: str,
        target_dtype: torch.dtype = torch.float16,
        num_layers: int | None = None,
    ) -> "NemotronHForCausalLMStateful":
        """Load a dense NemotronH checkpoint straight from safetensors (no transformers
        modeling dependency; config.json parsed directly)."""
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
        config = nemotron_h_config_from_dict(raw, num_layers=num_layers)

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
            model.lm_head.weight = model.backbone.embeddings.weight

        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(f"Parameters not loaded: {meta_params}")
        return model


def nemotron_h_config_from_dict(raw: dict, num_layers: int | None = None) -> NemotronHConfig:
    """Build the authoring config from a parsed HF config.json dict."""
    pattern = raw["hybrid_override_pattern"]
    if "E" in pattern:
        raise ValueError("MoE NemotronH variants are not supported (dense only)")
    layer_types = [_PATTERN[c] for c in pattern]
    n_layers = num_layers if num_layers is not None else raw["num_hidden_layers"]
    eps = raw.get("layer_norm_epsilon", raw.get("norm_eps", 1e-5))
    cfg = NemotronHConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=n_layers,
        vocab_size=raw["vocab_size"],
        intermediate_size=raw["intermediate_size"],
        layer_norm_epsilon=eps,
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        head_dim=raw["head_dim"],
        mamba_num_heads=raw["mamba_num_heads"],
        mamba_head_dim=raw["mamba_head_dim"],
        ssm_state_size=raw["ssm_state_size"],
        n_groups=raw["n_groups"],
        conv_kernel=raw["conv_kernel"],
        use_conv_bias=bool(raw.get("use_conv_bias", True)),
        time_step_min=raw.get("time_step_min", 0.001),
        layer_types=layer_types[:n_layers],
    )
    assert cfg.mamba_intermediate == cfg.mamba_num_heads * cfg.mamba_head_dim
    return cfg
