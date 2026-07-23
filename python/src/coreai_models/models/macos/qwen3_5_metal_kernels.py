# Community port — NOT an Apple model.
"""Fused int8v3 / int4km Metal-kernel wrappers for qwen3.5-0.8B decode (Session Q / SPEEDUP_PLAN Q1).

The qwen3.5 iOS GPU monolith decodes at 27.7-28.3 tok/s, weight-bandwidth-bound: ~1.5 GB of fp16
weights are read per token (ondevice/_qwen_traffic_audit.py: lm_head 508 MB + 24 layers 996 MB; ctx
8x costs only -9%, so the stream IS the weights). The proven lever is Session B's fused
dequant-in-matvec Metal kernels (gemma4: device FFN 2.9x int8, +1.43x int4km, fp32-accumulating,
8/8 EXACT, AOT-surviving). This module wraps the qwen3.5 nn.Linear call sites with those kernels.

ADDITIVE: imports ONLY the generic, device-proven builders/quantizers from ``gemma4_metal_mlp``
(that file is unchanged):
  * int8v3 = k-means 256-entry codebook per 32-output-row group, uint32-packed indices (4/word),
    codebook staged in threadgroup memory, R=4 rows/simd-group (MLX-qmv structure).
  * int4km = same k-means but 16-entry codebook, 8 nibbles/uint32. K-MEANS ONLY — affine int4 is
    banned for GPU kernels (gemma4: 7/8).
  * head   = same matvec at N=vocab + two-level argmax returning per-threadgroup partials
    (greedy-only; host reduces vocab/8 partials to the token id).

Coverage (== the audit's 99% of per-token bytes): MLP gate/up/down (all 24 layers), attention
q_proj/o_proj (6 full layers; k/v stay fp16 — N=512 matvecs never pay, Mac lesson), GatedDeltaNet
in_proj_qkv/in_proj_z/out_proj (18 SSM layers; in_proj_b/a are N=16, kernel-incompatible and
negligible), tied lm_head (the single biggest stream at 508 MB).

Shape contracts (qwen3.5-0.8B all satisfied): N % 32 == 0 (one palette group per threadgroup),
K % 256 == 0 (32 lanes x 8 values per block: K in {1024, 2048, 3584}), vocab % 8 == 0 (head SGY).
"""
from __future__ import annotations

import torch.nn as nn

from coreai_torch import TorchMetalKernel

from coreai_models.models.macos.gemma4_metal_mlp import (
    Gemma4MetalHeadArgmax,
    Gemma4MetalHeadArgmaxInt4,
    build_fused_int4km_kernel,
    build_fused_int8_kernel_v3,
    build_head_argmax_int4km_kernel,
    build_head_argmax_kernel,
    fused_int4km_call,
    fused_int8_v3_call,
    pack_idx_nib_u32,
    pack_idx_u32,
    palettize_grouped,
)

KINDS = ("int8v3", "int4km")


def build_qwen_matvec_kernel(kind: str) -> TorchMetalKernel:
    """The shared body matvec kernel (one MSL, all MLP/attn/SSM call sites) under a qwen name."""
    if kind == "int8v3":
        return build_fused_int8_kernel_v3("qwen3_5_fused_int8_v3")
    if kind == "int4km":
        return build_fused_int4km_kernel("qwen3_5_fused_int4km")
    raise ValueError(f"unknown kind {kind!r} (use one of {KINDS})")


def build_qwen_head_kernel(kind: str) -> TorchMetalKernel:
    """The head matvec + two-level argmax kernel under a qwen name."""
    if kind == "int8v3":
        return build_head_argmax_kernel("qwen3_5_head_int8_argmax")
    if kind == "int4km":
        return build_head_argmax_int4km_kernel("qwen3_5_head_int4km_argmax")
    raise ValueError(f"unknown kind {kind!r} (use one of {KINDS})")


class QwenMetalLinear(nn.Module):
    """Drop-in for an ``nn.Linear`` (q=1 decode row) running the fused int8v3/int4km matvec.

    Mirrors ``nn.Linear.forward`` (``y = x @ W.T``) for a [1, 1, K] activation -> [1, 1, N]. The
    weight is k-means palettized at construction (fp16 codebook == the dtype the fp16 monolith
    reads), packed for wide uint32 index loads, and stored as buffers the kernel dequantizes
    inline with fp32 accumulation.
    """

    def __init__(self, lin: nn.Linear, kernel: TorchMetalKernel, kind: str, iters: int = 10) -> None:
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        N, K = lin.weight.shape
        if N % 32:
            raise ValueError(f"out dim {N} not divisible by 32 (palette group / threadgroup rows)")
        if K % 256:
            raise ValueError(f"in dim {K} not divisible by 256 (32 lanes x 8 values per block)")
        self.kernel = kernel
        self.kind = kind
        self.N = int(N)
        if kind == "int8v3":
            idx, cb = palettize_grouped(lin.weight, iters=iters)
            self.register_buffer("idxp", pack_idx_u32(idx))   # [N, K/4] uint32 (4 indices/word)
            self.register_buffer("cb", cb)                     # [N/32, 256] fp16
        else:
            idx, cb = palettize_grouped(lin.weight, n_bits=4, iters=iters)
            self.register_buffer("qp", pack_idx_nib_u32(idx))  # [N, K/8] uint32 (8 nibbles/word)
            self.register_buffer("cb", cb)                     # [N/32, 16] fp16

    def forward(self, x):
        b, s, k = x.shape  # decode: b == s == 1
        xr = x.reshape(s, k)
        if self.kind == "int8v3":
            y = fused_int8_v3_call(self.kernel, xr, self.idxp, self.cb)
        else:
            y = fused_int4km_call(self.kernel, xr, self.qp, self.cb)
        return y.reshape(b, s, self.N)


# Per-layer Linear attributes the port covers (audit-grounded; see module docstring).
_MLP_PROJS = ("gate_proj", "up_proj", "down_proj")
_ATTN_PROJS = ("q_proj", "o_proj")            # k/v: N=512 matvec never pays (Mac lesson)
_SSM_PROJS = ("in_proj_qkv", "in_proj_z", "out_proj")  # in_proj_b/a: N=16, incompatible + negligible


def metalize_qwen_layers(layers, kind: str, kernel: TorchMetalKernel | None = None,
                         iters: int = 10) -> TorchMetalKernel:
    """Swap the bandwidth-dominant Linears of qwen3.5 decoder ``layers`` for fused-kernel wrappers.

    In-place on the given layer modules (MLP gate/up/down everywhere, attn q/o on full layers,
    GatedDeltaNet in_qkv/in_z/out on SSM layers). Norms, RoPE, SDPA, conv1d, the SSM recurrence and
    all numerics-critical plumbing are untouched. Returns the shared matvec kernel — register it
    with the converter BEFORE ``add_pytorch_module``.
    """
    if kernel is None:
        kernel = build_qwen_matvec_kernel(kind)
    for layer in layers:
        mods = [(layer.mlp, _MLP_PROJS)]
        if getattr(layer, "is_full", False):
            mods.append((layer.self_attn, _ATTN_PROJS))
        else:
            mods.append((layer.linear_attn, _SSM_PROJS))
        for owner, names in mods:
            for name in names:
                lin = getattr(owner, name)
                if isinstance(lin, nn.Linear):
                    setattr(owner, name, QwenMetalLinear(lin, kernel, kind, iters=iters))
    return kernel


def build_qwen_head_argmax(head_weight, kind: str, kernel: TorchMetalKernel | None = None,
                           iters: int = 10) -> nn.Module:
    """Tied 248320-vocab lm_head -> fused matvec + two-level GPU argmax (greedy-only).

    Returns the partials-contract module (``hidden [1,1,K] -> (pv [vocab/8] fp32, pi [vocab/8]
    int32)``; host token = ``pi[argmax(pv)]``). qwen3.5 has no final softcap, so GPU argmax ==
    ``argmax(lm_head(hidden))`` exactly. The palettized copy is independent of the tied embed
    table (the embed gather keeps reading the fp16 table; only the head matvec goes int8/int4).
    """
    if kernel is None:
        kernel = build_qwen_head_kernel(kind)
    cls = Gemma4MetalHeadArgmax if kind == "int8v3" else Gemma4MetalHeadArgmaxInt4
    return cls(head_weight, kernel, iters=iters)
