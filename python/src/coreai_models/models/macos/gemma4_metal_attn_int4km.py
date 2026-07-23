# int4-K-MEANS attention projections for the gemma4 device-GPU monolith (Session G / G1 stretch).
#
# Additive companion to gemma4_metal_mlp.py (Session B's generic kernels — imported, not modified):
# `MetalInt4KMLinear` is the nn.Linear drop-in twin of `MetalIntLinear`, but running the 4-bit
# k-means fused matvec (`build_fused_int4km_kernel`: 8 nibbles/uint32 wide loads, 16-entry
# threadgroup-staged codebook) — HALF the int8 index bytes, for the BANDWIDTH-bound device GPU.
#
# Gated by the PyTorch numerics check `ondevice/_iso_gemma4_attn_int4_numerics.py`
# (2026-06-10: int4 FFN + int4 attn q/o = 8/8 EXACT vs the HF greedy oracle; int4 AFFINE stays
# banned for GPU kernels — this is k-means, the de-risked scheme).
#
# Shape constraints (the int4km kernel's): K % 256 == 0 (32 lanes x 8 nibbles) and N % 32 == 0
# (one palette group per threadgroup). gemma4 E2B q (2048/4096 x 1536), o (1536 x 2048/4096)
# all qualify; k/v are NOT kernelized anywhere (small-N kernels never pay — the Mac lesson).
from __future__ import annotations

from torch import nn

from coreai_models.models.macos.gemma4_metal_mlp import (
    TorchMetalKernel,
    build_fused_int4km_kernel,
    fused_int4km_call,
    pack_idx_nib_u32,
    palettize_grouped,
)


class MetalInt4KMLinear(nn.Module):
    """Drop-in for an ``nn.Linear`` (q=1 decode) whose matvec runs the FUSED int4-k-means kernel.

    Mirrors ``MetalIntLinear`` (the int8 twin) with the int4km storage: packed nibble indices
    ``qp [N, K/8] uint32`` + 16-entry codebook ``cb [G, 16] fp16``. Share the SAME kernel object
    as ``metalize_mlps_int4km`` so one MSL function serves FFN + attention.
    """

    def __init__(self, lin: nn.Linear, kernel: TorchMetalKernel, iters: int = 10) -> None:
        super().__init__()
        self.kernel = kernel
        n, k = lin.weight.shape
        if k % 256:
            raise ValueError(f"K={k} not divisible by 256 (32 lanes x 8 nibbles)")
        if n % 32:
            raise ValueError(f"N={n} not divisible by 32 (one palette group per threadgroup)")
        self.N = int(n)
        idx, cb = palettize_grouped(lin.weight, n_bits=4, iters=iters)
        self.register_buffer("qp", pack_idx_nib_u32(idx))  # [N, K/8] uint32
        self.register_buffer("cb", cb)                      # [G, 16] fp16

    @property
    def weight(self):
        # Some model forwards read ``.weight.dtype`` to cast the input before the matmul
        # (e.g. lfm2 attention: ``x.to(self.q_proj.weight.dtype)``). Expose the codebook so this
        # int4 drop-in answers ``.dtype`` like the nn.Linear it replaces (its compute dtype = cb's).
        return self.cb

    def forward(self, x):
        b, s, k = x.shape  # decode: b=1, s=1
        y = fused_int4km_call(self.kernel, x.reshape(s, k), self.qp, self.cb)
        return y.reshape(b, s, self.N)


def metalize_attn_int4km(core: nn.Module, kernel: TorchMetalKernel | None = None, iters: int = 10,
                         projections: tuple[str, ...] = ("q_proj", "o_proj")) -> TorchMetalKernel:
    """Swap selected attention projections for ``MetalInt4KMLinear``.

    Pass the kernel returned by ``metalize_mlps_int4km`` to share one MSL across FFN + attention.
    Only the named projection matmuls change; q_norm/k_norm/RoPE/SDPA/KV plumbing are untouched.
    Default q/o only (the big projections) — k/v stay fp16-MPSGraph.
    """
    if kernel is None:
        kernel = build_fused_int4km_kernel()
    for layer in core.text_model.layers:
        attn = layer.self_attn
        for name in projections:
            lin = getattr(attn, name)
            if isinstance(lin, nn.Linear):
                setattr(attn, name, MetalInt4KMLinear(lin, kernel, iters=iters))
    return kernel
