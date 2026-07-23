# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import math
import os

import torch


def compute_llama3_inv_freq(
    head_dim: int,
    base: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    """Llama-3 RoPE inverse-frequency rescaling (`rope_type: "llama3"`).

    Mirrors HuggingFace `_compute_llama3_parameters` and the macOS
    `Llama3RoPE`. Returns the per-dim inv_freq (length head_dim/2, fp32).
    """
    # Force CPU: this may run inside a `torch.device("meta")` init context
    # (BaseForCausalLM builds the module tree on meta), and the inv_freq must be
    # a real CPU tensor so RoPECache can build its cos/sin cache.
    with torch.device("cpu"):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
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
        return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


def compute_longrope_inv_freq(
    rotary_dim: int,
    base: float,
    short_factor: list[float],
) -> torch.Tensor:
    """LongRoPE inverse frequencies (short-factor regime), length rotary_dim/2.

    Mirrors HuggingFace `_compute_longrope_parameters`: inv_freq is the default
    schedule divided per-dim by `short_factor`. CPU-forced for meta-device init.
    """
    with torch.device("cpu"):
        ext = torch.tensor(short_factor, dtype=torch.float32)
        return 1.0 / (
            ext * base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )


def compute_longrope_attention_scaling(
    max_position_embeddings: int, original_max_position_embeddings: int
) -> float:
    factor = max_position_embeddings / original_max_position_embeddings
    if factor <= 1.0:
        return 1.0
    return math.sqrt(1 + math.log(factor) / math.log(original_max_position_embeddings))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    GPT NeoX style: rotates [repeat] half the hidden dims of the input.
    for sin [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    """

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 : x.shape[-1]]
    return torch.cat((-x2, x1), dim=-1)


@torch.library.custom_op("coreai::rope_gather_cached_cos_sin", mutates_args=[])
def rope_gather_cached_cos_sin(
    position_ids: torch.Tensor, cos_cached: torch.Tensor, sin_cached: torch.Tensor
) -> list[torch.Tensor]:
    position_ids = position_ids.to(torch.int32)
    rope_cos = cos_cached[position_ids]
    rope_sin = sin_cached[position_ids]
    return rope_cos, rope_sin


@rope_gather_cached_cos_sin.register_fake
def _fake(
    position_ids: torch.Tensor, cos_cached: torch.Tensor, sin_cached: torch.Tensor
) -> list[torch.Tensor]:
    position_ids = position_ids.to(torch.int32)
    rope_cos = cos_cached[position_ids]
    rope_sin = sin_cached[position_ids]
    return rope_cos, rope_sin


def apply_rope(x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
    rope_cos = rope_cos.unsqueeze(1)
    rope_sin = rope_sin.unsqueeze(1)

    torch._check(len(rope_cos.shape) == 4)
    torch._check(len(rope_sin.shape) == 4)

    # Apply rotary position embedding
    return (x * rope_cos) + (rotate_half(x) * rope_sin)


def apply_rope_partial(
    x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """Partial rotary (Phi-4-mini): rotate only the first ``rotary_dim`` of each
    head, pass the remaining dims through unchanged. ``rope_cos``/``rope_sin``
    must be sized to ``rotary_dim`` (built from a length-``rotary_dim/2``
    inv_freq)."""
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x_rot = apply_rope(x_rot, rope_cos, rope_sin)
    return torch.cat((x_rot, x_pass), dim=-1)


# On iOS, it is more efficient to compute RoPE using precomputed and cached cos/sin values
class RoPECache(torch.nn.Module):
    """
    RoPE module.

    Paper reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        head_dim: int,
        max_cache_size: int,
        base: float = 500_000,
        inv_freq: torch.Tensor | None = None,
        attention_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self._head_dim = head_dim
        self._max_cache_size = max_cache_size
        self._base = base
        # Optional precomputed per-dim inverse frequencies (length head_dim/2),
        # used for non-default RoPE scaling (e.g. llama3). When None the default
        # `1 / base**(2i/d)` schedule is used.
        self._inv_freq_override = (
            inv_freq.to(torch.float32) if inv_freq is not None else None
        )
        self._attention_scaling = float(attention_scaling)
        self._use_hf_impl = os.environ.get("USE_HF_IMPL", "False").lower() == "true"
        self._compute_sin_and_cos()

    def _apply(self, fn):
        # the `.to()` function implicitly calls into this function,
        # and we should recompute the cos / sin rather then just do
        # a simple cast.
        super()._apply(fn)
        dummy = torch.tensor(0.0)
        transformed = fn(dummy)
        self._compute_sin_and_cos(transformed.dtype)

        target_device = transformed.device
        self.cos_cached = self.cos_cached.to(device=target_device)
        self.sin_cached = self.sin_cached.to(device=target_device)
        return self

    def _compute_sin_and_cos(self, dtype: torch.dtype = torch.float32) -> None:
        head_dim = self._head_dim
        max_cache_size = self._max_cache_size
        base = self._base

        with torch.device("cpu"):
            if self._inv_freq_override is not None:
                theta = self._inv_freq_override.to(torch.float32)
            else:
                theta = 1.0 / (
                    base
                    ** (torch.arange(start=0, end=head_dim, step=2, dtype=torch.float32) / head_dim)
                )

            if self._use_hf_impl:
                theta = theta.to(dtype)
                theta = theta.float()

            # Create position index [0, 1, ..., seq_len -1]
            seq_idx = torch.arange(end=max_cache_size, dtype=torch.int32)

            # Calculate product of position index and theta
            freqs = seq_idx[:, None] * theta

            # Cache cos sin values (attention_scaling is 1.0 for default/llama3).
            emb = torch.concatenate((freqs, freqs), dim=-1)
            cos = torch.cos(emb) * self._attention_scaling
            sin = torch.sin(emb) * self._attention_scaling
            self.cos_cached = torch.nn.Buffer(cos.to(dtype=dtype), persistent=False)
            self.sin_cached = torch.nn.Buffer(sin.to(dtype=dtype), persistent=False)

    def gather_cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather the cached cos/sin values using the position_ids"""
        return rope_gather_cached_cos_sin(position_ids, self.cos_cached, self.sin_cached)
