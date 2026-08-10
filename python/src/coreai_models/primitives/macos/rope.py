# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import math
import os

import coreai_torch
import coreai_torch.composite_ops
import torch
from typing_extensions import Self


class RoPE(coreai_torch.composite_ops.RoPE):
    """Apply rotary positional embedding to input tensors."""

    def __init__(
        self: Self,
        scale: float = 1.0,
        base: float = 1e4,
        dims: int | None = None,
        interleaved: bool = False,
    ) -> None:
        _use_hf_impl = os.environ.get("USE_HF_IMPL", "False").lower() == "true"
        super().__init__(
            scale=scale,
            base=base,
            dims=dims,
            interleaved=interleaved,
            _use_hf_impl=_use_hf_impl,
        )


class YarnRoPE(torch.nn.Module):
    def __init__(
        self: Self,
        dims: int,
        interleaved: bool = False,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
        truncate: bool = True,
    ) -> None:
        super().__init__()

        def yarn_find_correction_dim(num_rotations):
            return (
                dims * math.log(original_max_position_embeddings / (num_rotations * 2 * math.pi))
            ) / (2 * math.log(base))

        def yarn_find_correction_range():
            low = yarn_find_correction_dim(beta_fast)
            high = yarn_find_correction_dim(beta_slow)
            if truncate:
                low = math.floor(low)
                high = math.ceil(high)
            return max(low, 0), min(high, dims - 1)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001  # Prevent singularity

            linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
            return torch.clip(linear_func, 0, 1)

        # Initialize constants that aren't a part of state-dict on with cpu
        # device so that they don't get "faked" on meta device when initializing
        # model structure.
        with torch.device("cpu"):
            self.dims = dims
            self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
                scaling_factor, mscale_all_dim
            )
            freq_extra = base ** (torch.arange(0, dims, 2, dtype=torch.float32) / dims)
            freq_inter = scaling_factor * freq_extra
            low, high = yarn_find_correction_range()
            freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
            self._freqs = (freq_inter * freq_mask + freq_extra * (1 - freq_mask)) / (
                freq_inter * freq_extra
            )
            self._rope = RoPE(scale=1.0, interleaved=interleaved)

    def forward(
        self: Self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mscale != 1.0:
            head_dim = x.shape[-1]
            message = "torch.export fails partial Yarn RoPE"
            torch._check(self.dims >= head_dim, message=message)
            # In principle the general formula that supports partial Yarn RoPE is
            #     x[..., : self.dims] = self.mscale * x[..., : self.dims]
            # In practice torch.export does not support partial sliced assignment,
            # so we apply mscale to the full tensor (full Yarn RoPE only).
            x = self.mscale * x
        return self._rope(
            x,
            position_ids=position_ids,
            freqs=self._freqs.to(x.device),
            offset=offset,
        )


class Llama3RoPE(torch.nn.Module):
    """Llama-3 RoPE frequency rescaling (`rope_type: "llama3"`).

    Mirrors HuggingFace `_compute_llama3_parameters`: the default inverse
    frequencies are rescaled per-dimension — low-frequency components are
    divided by `factor`, high-frequency components are untouched, and a smooth
    ramp interpolates in between. `attention_scaling` is 1.0, so no logit
    rescaling is needed. The resulting inv_freq is fed to the base RoPE op as
    `freqs` (angle = position * freqs).
    """

    def __init__(
        self: Self,
        dims: int,
        base: float,
        factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        original_max_position_embeddings: int,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        with torch.device("cpu"):
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dims, 2, dtype=torch.float32) / dims)
            )
            low_freq_wavelen = original_max_position_embeddings / low_freq_factor
            high_freq_wavelen = original_max_position_embeddings / high_freq_factor
            wavelen = 2 * math.pi / inv_freq

            inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
            smooth_factor = (original_max_position_embeddings / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
            inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

            self._freqs = inv_freq_llama
            self._rope = RoPE(scale=1.0, interleaved=interleaved)

    def forward(
        self: Self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._rope(x, position_ids=position_ids, freqs=self._freqs, offset=offset)


class LongRoPE(torch.nn.Module):
    """LongRoPE (`rope_type: "longrope"`, e.g. Phi-4-mini), short-factor regime.

    Two Phi-specific features are handled:
      * **Partial rotary** — only the first ``rotary_dim`` (= head_dim *
        partial_rotary_factor) dims are rotated; the rest pass through. The base
        RoPE op does this natively via ``dims=rotary_dim``.
      * **LongRoPE rescale + attention scaling** — inv_freq is divided by the
        per-dim ``short_factor``; cos/sin are scaled by ``attention_scaling``.
        Since attention_scaling distributes over the rotation, it is applied as
        a per-dim multiply on the rotary output (1.0 on the pass-through dims),
        which is torch.export-friendly (no partial sliced assignment).

    Only the short-factor branch is materialized — correct for context lengths
    up to ``original_max_position_embeddings`` (the benchmark regime). The
    long-factor branch (positions beyond that) is not wired.
    """

    def __init__(
        self: Self,
        head_dim: int,
        rotary_dim: int,
        base: float,
        short_factor: list[float],
        attention_scaling: float = 1.0,
        interleaved: bool = False,
    ) -> None:
        super().__init__()
        with torch.device("cpu"):
            ext = torch.tensor(short_factor, dtype=torch.float32)
            inv_freq = 1.0 / (
                ext * base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
            )
            self._freqs = inv_freq
            self._rope = RoPE(scale=1.0, dims=rotary_dim, interleaved=interleaved)
            scale_vec = torch.ones(head_dim, dtype=torch.float32)
            scale_vec[:rotary_dim] = attention_scaling
            self._scale_vec = scale_vec

    def forward(
        self: Self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self._rope(x, position_ids=position_ids, freqs=self._freqs, offset=offset)
        return out * self._scale_vec.to(out.dtype)


def compute_longrope_attention_scaling(
    max_position_embeddings: int, original_max_position_embeddings: int
) -> float:
    """HuggingFace `_compute_longrope_parameters` attention factor."""
    factor = max_position_embeddings / original_max_position_embeddings
    if factor <= 1.0:
        return 1.0
    return math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))


def initialize_rope(
    dims: int | None = None,
    base: float = 1e4,
    interleaved: bool = False,
    scaling_config: dict | None = None,
    max_position_embeddings: int | None = None,
) -> torch.nn.Module:
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")
    else:
        rope_type = "default"

    rope: torch.nn.Module
    match rope_type:
        case "default" | "linear":
            scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
            rope = RoPE(scale=float(scale), base=float(base), dims=dims, interleaved=interleaved)

        case "llama3":
            if dims is None:
                msg = "dims is required for llama3 rope"
                raise ValueError(msg)
            rope = Llama3RoPE(
                dims,
                base=float(base),
                factor=float(scaling_config["factor"]),
                low_freq_factor=float(scaling_config["low_freq_factor"]),
                high_freq_factor=float(scaling_config["high_freq_factor"]),
                original_max_position_embeddings=int(
                    scaling_config["original_max_position_embeddings"]
                ),
                interleaved=interleaved,
            )

        case "yarn":
            if dims is None:
                msg = "dims is required for yarn rope"
                raise ValueError(msg)
            scaling_factor = scaling_config["factor"]
            rope_kwargs = {
                key: scaling_config[key]
                for key in [
                    "original_max_position_embeddings",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                ]
                if key in scaling_config
            }
            # Default truncate=True preserves prior behavior for gemma3 / qwen3_next / etc.
            # gpt-oss sets truncate=False; match HF `_compute_yarn_parameters` (line 359 of
            # transformers/modeling_rope_utils.py).
            truncate = scaling_config.get("truncate", True)
            rope = YarnRoPE(
                dims,
                interleaved=interleaved,
                max_position_embeddings=max_position_embeddings,
                base=float(base),
                scaling_factor=float(scaling_factor),
                truncate=bool(truncate),
                **rope_kwargs,
            )

        case _:
            msg = f"Unsupported RoPE type {rope_type}"
            raise ValueError(msg)

    return rope
