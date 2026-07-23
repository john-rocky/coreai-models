# Community port — NOT an Apple model.
"""``gather_qmm`` — a custom Metal kernel that fixes the MoE decode dense-overread.

THE PROBLEM. MoE FFN decode lowers through the ``SwitchGLU``/``GatherMM`` composite
(``primitives/macos/switch.py``): it gathers the routed experts' weights, then runs a
DENSE matmul. As MPSGraph lowers it, the matmul READS ALL ``E`` experts' weight rows
even though only the top-``k`` are routed — so MoE decode is over-read-bound, not
active-param-bound (Qwen3.6-35B-A3B int8 sits at ~25% of BW; LFM2.5-8B-A1B int8 decode
is BW-saturated at 39 tok/s reading its full 8.8 GB/token). The over-read is the whole
reason LFM2.5-8B-A1B is VALIDATED-but-UNSHIPPED.

THE FIX (this file). A custom ``coreai_torch.TorchMetalKernel`` matvec that takes the
routed expert indices as a kernel INPUT and reads ONLY those ``k`` experts' weight slabs
out of the ``[E, N, K]`` tensor (``QP[w, n, e]`` with ``e = IDX[slot]`` — an indexed
global load, so the other ``E-k`` experts' rows are never fetched). Decode then runs at
ACTIVE-param bandwidth (k/E of the experts), the win the byte-count predicts.

The kernel reads weights quantized in-file (it owns its layout, like the gemma4 FFN
kernels in ``gemma4_metal_mlp.py``, whose multi-row + tg-staged-codebook structure this
reuses). Two precisions:
  * ``int8km`` — 256-entry k-means codebook per 32-output-row group (the quality floor;
    matches the shipped int8 MoE numerics).
  * ``int4km`` — 16-entry codebook, 8 nibbles/uint32 (half the bytes; k-means int4 was
    de-risked 8/8 on gemma4 where affine-along-K was 7/8, so it is the quality-safe 4-bit).

DSL indexing (from ``coreai-torch/tests/dsl/conftest.py`` + verified in
``ondevice/_moe_kernel_probe.py``): the Metal view reverses axis order vs torch, so a
torch ``[E, N, M]`` tensor is read in-kernel as ``T[m, n, e]`` = torch ``T[e, n, m]``;
``get_extent(0)`` is the innermost (last torch) dim. Rank-3 indexing + a data-dependent
gather both lower and run on the Mac GPU (the probe PASSES).

Boundary: purely additive. ``MetalSwitchGLU`` is a drop-in for ``SwitchGLU`` (same
``forward(x, indices) -> [b, s, k, hidden]`` contract); it does not change the routing /
combine math in the MoE block, only how the three expert projections are computed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

from coreai_models.models.macos.gemma4_metal_mlp import (
    palettize_grouped,
    pack_idx_nib_u32,
    pack_idx_u32,
)

# One threadgroup owns R*SGY = 32 output rows = exactly ONE palette group, so the staged
# codebook is that group's L entries (16 for int4, 256 for int8). SGY=8 => the int8 256-entry
# codebook is staged by exactly the 8*32=256 threads. K must be divisible by 256 (32 lanes *
# 8 vals/lane per block); N by R*SGY=32. LFM2.5: hidden 2048 / moe_inter 1792 satisfy both.
_R, _SGY = 4, 8
_PALETTE_GROUP = 32

# --- int4 k-means gather matvec: 16-entry codebook, 8 nibbles/uint32 -------------------------------
# QP torch [E, N, K/8] uint32 ; CB torch [E, G, 16] fp16 (G = N/32) ; IDX torch [k] int32.
# In-kernel QP[w,n,e]=torch qp[e,n,w] ; CB[q,group,e]=torch cb[e,group,q]. Only expert e=IDX[slot] read.
_INT4KM_SRC = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);          // A torch [k, K] (per-slot activation)
    const uint lane = tid.x;                  // 0..31
    const uint sg = tid.y;                    // 0..SGY-1
    const uint group = tgid.y;                // R*SGY=32 rows/tg = ONE palette group
    const uint slot = tgid.z;                 // which routed-expert slot (0..k-1)
    const uint e = uint(IDX[slot]);           // GATHER: only this expert's weight is read
    const uint base_row = (tgid.y * SGY + sg) * R;

    threadgroup float cbt[16];                // stage expert e's 16-entry codebook for this group
    uint t = sg * 32 + lane;
    if (t < 16) cbt[t] = float(CB[t, group, e]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {    // block = 32 lanes * 8 nibbles = 256 K
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(A[k0 + j, slot]);  // this slot's activation row
        uint w0 = (kb >> 3) + lane;           // one uint32 word (8 nibbles) per lane
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = uint(QP[w0, n, e]); // QP[w,n,e]=torch qp[e,n,w]; ONLY expert e
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uint q = (packed >> (j * 4)) & 0xf;
                s += xr[j] * cbt[q];
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r, slot] = TYPE(tot);   // C torch [k, N]
    }
"""

# --- int8 k-means gather matvec: 256-entry codebook, 4 bytes/uint32 (8 indices/lane = 2 words) -----
# QP torch [E, N, K/4] uint32 ; CB torch [E, G, 256] fp16. Quality floor (matches shipped int8 MoE).
_INT8KM_SRC = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);           // A torch [k, K] (per-slot activation)
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint group = tgid.y;
    const uint slot = tgid.z;
    const uint e = uint(IDX[slot]);
    const uint base_row = (tgid.y * SGY + sg) * R;

    threadgroup float cbt[256];               // SGY*32 = 256 threads stage 256 entries exactly
    uint t = sg * 32 + lane;
    if (t < 256) cbt[t] = float(CB[t, group, e]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {    // 32 lanes * 8 bytes = 256 K
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(A[k0 + j, slot]);  // this slot's activation row
        uint w0 = (kb >> 2) + lane * 2;       // 2 uint32 words (8 bytes) per lane
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            float s = 0.0f;
            for (uint wi = 0; wi < 2; ++wi) {
                uint packed = uint(QP[w0 + wi, n, e]);
                uint xb = wi * 4;
                s += xr[xb + 0] * cbt[packed & 0xff]
                   + xr[xb + 1] * cbt[(packed >> 8) & 0xff]
                   + xr[xb + 2] * cbt[(packed >> 16) & 0xff]
                   + xr[xb + 3] * cbt[(packed >> 24) & 0xff];
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r, slot] = TYPE(tot);
    }
"""


def _unpack_nib(qp: torch.Tensor) -> torch.Tensor:
    """[E, N, K/8] uint32 -> [E, N, K] int64 palette indices."""
    E, N, K8 = qp.shape
    p = qp.to(torch.int64)
    return torch.stack([(p >> (j * 4)) & 0xF for j in range(8)], dim=-1).reshape(E, N, K8 * 8)


def _unpack_byte(qp: torch.Tensor) -> torch.Tensor:
    """[E, N, K/4] uint32 -> [E, N, K] int64 palette indices."""
    E, N, K4 = qp.shape
    p = qp.to(torch.int64)
    return torch.stack([(p >> (j * 8)) & 0xFF for j in range(4)], dim=-1).reshape(E, N, K4 * 4)


def _dequant_gathered(qg: torch.Tensor, cg: torch.Tensor, nbits: int) -> torch.Tensor:
    """Per-slot dequantized weight [k, N, K] from gathered indices + codebooks.

    qg = gathered packed indices [k, N, K/(8 or 4)] uint32; cg = gathered codebooks
    [k, G, L] fp16. Mirrors the kernel's per-32-output-row group codebook lookup."""
    k, N = qg.shape[0], qg.shape[1]
    idx = _unpack_nib(qg) if nbits == 4 else _unpack_byte(qg)  # [k, N, K]
    G = cg.shape[1]
    grp = (torch.arange(N, device=qg.device) // (N // G))      # [N]
    cg_per_row = cg[:, grp, :]                                 # [k, N, L]
    return torch.gather(cg_per_row, 2, idx)                    # [k, N, K]


def _gather_qmm_defn(x: torch.Tensor, qp: torch.Tensor, cb: torch.Tensor, idx: torch.Tensor,
                     nbits: int) -> torch.Tensor:
    """Reference + shape inference: gather k experts, k-means dequant, PER-SLOT matvec.

    x [k, K] (each routed slot's own activation row — gate/up share the token x replicated
    k-wide, down feeds each expert its own gated activation) ; qp [E, N, K/(8|4)] uint32 ;
    cb [E, G, L] fp16 ; idx [k] int32 -> [k, N]. index_select is shape-static (no ``.item()``)
    so this traces under FakeTensor for the kernel's output-shape baking, and is numerically
    exact for the eager-vs-GPU gate."""
    i = idx.to(torch.int32)
    qg = torch.index_select(qp, 0, i)            # [k, N, K/8|4]
    cg = torch.index_select(cb, 0, i)            # [k, G, L]
    w = _dequant_gathered(qg, cg, nbits)         # [k, N, K]
    return torch.einsum("sc,snc->sn", x.float(), w.float()).to(x.dtype)


def _build_kernel(nbits: int, name: str) -> TorchMetalKernel:
    src = (_INT4KM_SRC if nbits == 4 else _INT8KM_SRC).replace("__R__", str(_R)).replace("__SGY__", str(_SGY))

    def defn(x: torch.Tensor, qp: torch.Tensor, cb: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return _gather_qmm_defn(x, qp, cb, idx, nbits)

    return TorchMetalKernel(
        name, input_names=["A", "QP", "CB", "IDX"], result_names=["C"], src=src, torch_defn=defn,
        metal_params=[MetalParameter("tid", "uint3", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint3", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def build_gather_qmm_int4km(name: str = "gather_qmm_int4km") -> TorchMetalKernel:
    """int4 k-means gather matvec (16-entry codebook, 8 nibbles/uint32) — the size/speed lever."""
    return _build_kernel(4, name)


def build_gather_qmm_int8km(name: str = "gather_qmm_int8km") -> TorchMetalKernel:
    """int8 k-means gather matvec (256-entry codebook) — the quality floor."""
    return _build_kernel(8, name)


def _quantize_experts(weight: torch.Tensor, nbits: int, iters: int = 10
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-expert k-means palettization of a SwitchLinear weight stack.

    weight = [E, out, in] (one expert per slab) -> (qp, cb) where qp is
    [E, out, in/(8|4)] uint32 (packed indices) and cb is [E, out/32, L] fp16. Each expert is
    palettized independently along its OUTPUT axis (the proven k-means layout), then the
    indices are packed for wide 32-bit reads."""
    E = weight.shape[0]
    qps, cbs = [], []
    pack = pack_idx_nib_u32 if nbits == 4 else pack_idx_u32
    for e in range(E):
        idx, cb = palettize_grouped(weight[e], group_size=_PALETTE_GROUP, n_bits=nbits, iters=iters)
        qps.append(pack(idx))    # [out, in/8] or [out, in/4]
        cbs.append(cb)           # [G, L]
    return torch.stack(qps, 0).contiguous(), torch.stack(cbs, 0).contiguous()


# =================================================================================================== #
# SYMMETRIC LINEAR int8 — matches the SHIPPED int8-linear recipe (`symmetric_with_clipping`,
# per-block-32 along the K/input axis), the precision that was greedy-CLEAN on this model. Same
# 1-byte/weight bandwidth as int8 k-means (so ~same decode speed) but a UNIFORM dequant
# (scale·q, no codebook) → the quality the k-means gather kernel was missing. Each (output row,
# 32-wide K-block) gets one fp16 scale; q is signed int8 packed 4-per-uint32 for wide reads.
_INT8_KBLOCK = 32  # per-block size along K (axis=1), matching the shipped int8lin spec
# MSE-optimal symmetric clip ratios (shrink the range below absmax to tame outliers, AWQ-lite).
_SYM_CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)

_INT8SYM_SRC = """
    const uint R = __R__, SGY = __SGY__, KB = __KB__;
    const uint K = A.get_extent(0);           // A torch [k, K] (per-slot activation)
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint slot = tgid.z;
    const uint e = uint(IDX[slot]);           // GATHER: only this expert's weight is read
    const uint base_row = (tgid.y * SGY + sg) * R;

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {    // 32 lanes * 8 int8 = 256 K
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(A[k0 + j, slot]);
        uint w0 = (kb >> 2) + lane * 2;       // 2 uint32 words (8 int8) per lane
        uint kblk = (kb + lane * 8) / KB;     // this lane's 8 vals fall in one K-block of KB
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            float sc = float(SC[kblk, n, e]); // SC[kblk,n,e]=torch sc[e,n,kblk]; one per (lane,n)
            float s = 0.0f;
            for (uint wi = 0; wi < 2; ++wi) {
                uint packed = uint(QP[w0 + wi, n, e]);   // 4 signed int8 / word
                uint xb = wi * 4;
                s += xr[xb + 0] * float(int(char(packed & 0xff)))
                   + xr[xb + 1] * float(int(char((packed >> 8) & 0xff)))
                   + xr[xb + 2] * float(int(char((packed >> 16) & 0xff)))
                   + xr[xb + 3] * float(int(char((packed >> 24) & 0xff)));
            }
            acc[r] += sc * s;                  // all 8 of this lane's vals share one scale
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r, slot] = TYPE(tot);
    }
"""


def pack_int8_u32(q: torch.Tensor) -> torch.Tensor:
    """[N, K] signed int8 -> [N, K//4] uint32 (4 little-endian bytes/word)."""
    N, K = q.shape
    if K % 4:
        raise ValueError(f"K={K} not divisible by 4")
    b = (q.to(torch.int64) & 0xFF).reshape(N, K // 4, 4)
    return (b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24)).to(torch.uint32).contiguous()


def _unpack_int8(qp: torch.Tensor) -> torch.Tensor:
    """[E, N, K//4] uint32 -> [E, N, K] signed int (the inverse of pack_int8_u32)."""
    E, N, K4 = qp.shape
    p = qp.to(torch.int64)
    by = torch.stack([(p >> (8 * j)) & 0xFF for j in range(4)], dim=-1)   # 0..255
    signed = by - 256 * (by >= 128)                                       # -> [-128, 127]
    return signed.reshape(E, N, K4 * 4)


def quantize_sym_int8_blockK(weight: torch.Tensor, block: int = _INT8_KBLOCK
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-(output-row, K-block) int8 with MSE-optimal clipping (the int8lin scheme).

    weight [N, K] -> (qp [N, K//4] uint32 packed signed int8, scale [N, K//block] fp16).
    ``w[n,k] ≈ scale[n, k//block] · q[n,k]`` with q in [-127, 127]; the per-block range is searched
    over :data:`_SYM_CLIPS` to minimize quant MSE (outlier-robust)."""
    W = weight.detach().float()
    N, K = W.shape
    if K % block:
        raise ValueError(f"K={K} not divisible by block {block}")
    Wb = W.reshape(N, K // block, block)
    amax = Wb.abs().amax(-1)                                   # [N, NB]
    best_err = best_scale = best_q = None
    for c in _SYM_CLIPS:
        scale = (amax * c / 127.0).clamp(min=1e-8)            # [N, NB]
        q = torch.round(Wb / scale.unsqueeze(-1)).clamp(-127, 127)
        err = ((scale.unsqueeze(-1) * q) - Wb).square().sum(-1)  # [N, NB]
        if best_err is None:
            best_err, best_scale, best_q = err, scale, q
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_scale = torch.where(take, scale, best_scale)
            best_q = torch.where(take.unsqueeze(-1), q, best_q)
    q_int8 = best_q.reshape(N, K).to(torch.int8)
    return pack_int8_u32(q_int8), best_scale.to(torch.float16).contiguous()


def _gather_qmm_int8sym_defn(x: torch.Tensor, qp: torch.Tensor, sc: torch.Tensor,
                             idx: torch.Tensor, block: int) -> torch.Tensor:
    """Reference + shape inference: gather k experts, symmetric-int8 dequant (scale·q), per-slot
    matvec. x [k, K]; qp [E, N, K//4] uint32; sc [E, N, K//block] fp16; idx [k] -> [k, N]."""
    i = idx.to(torch.int32)
    qg = torch.index_select(qp, 0, i)                 # [k, N, K//4]
    sg = torch.index_select(sc, 0, i)                 # [k, N, K//block]
    q = _unpack_int8(qg).float()                      # [k, N, K]
    scale = sg.float().repeat_interleave(block, dim=2)  # [k, N, K]
    w = scale * q
    return torch.einsum("sc,snc->sn", x.float(), w).to(x.dtype)


def build_gather_qmm_int8sym(name: str = "gather_qmm_int8sym",
                             block: int = _INT8_KBLOCK) -> TorchMetalKernel:
    """Symmetric linear int8 gather matvec (per-K-block-32 scale, no codebook) — the CLEAN int8."""
    src = (_INT8SYM_SRC.replace("__R__", str(_R)).replace("__SGY__", str(_SGY))
           .replace("__KB__", str(block)))

    def defn(x: torch.Tensor, qp: torch.Tensor, sc: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return _gather_qmm_int8sym_defn(x, qp, sc, idx, block)

    return TorchMetalKernel(
        name, input_names=["A", "QP", "SC", "IDX"], result_names=["C"], src=src, torch_defn=defn,
        metal_params=[MetalParameter("tid", "uint3", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint3", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def _quantize_experts_sym(weight: torch.Tensor, block: int = _INT8_KBLOCK
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric-int8 quantize a [E, out, in] expert stack -> (qp [E,out,in/4] uint32,
    sc [E, out, in/block] fp16)."""
    qps, scs = [], []
    for e in range(weight.shape[0]):
        qp, sc = quantize_sym_int8_blockK(weight[e], block=block)
        qps.append(qp)
        scs.append(sc)
    return torch.stack(qps, 0).contiguous(), torch.stack(scs, 0).contiguous()


# =================================================================================================== #
# AFFINE int4 (block-32 along K, scale·q + bias) — the iPhone size lever, but a BETTER 4-bit than
# k-means: asymmetric (per-block zero-point) + MSE-optimal clip handles the expert weight
# distribution far better than a 16-entry palette (the Apple "asymmetric +3-5 dB" lever). Aux
# packs (scale, bias) into ONE fp16 buffer of width 2·(K/block): aux[..., 0::2]=scale, [...,1::2]=bias
# (keeps the uniform 2-buffer MetalSwitchGLU contract). q in [0,15], 8 nibbles/uint32.
_INT4_KBLOCK = 32
_AFF_CLIPS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
_INT4AFF_SRC = """
    const uint R = __R__, SGY = __SGY__, KB = __KB__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint slot = tgid.z;
    const uint e = uint(IDX[slot]);
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint NSB = (K / KB) * 2;            // aux row width = 2 per K-block (scale, bias)

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {    // 32 lanes * 8 nibbles = 256 K
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(A[k0 + j, slot]);
        uint w0 = (kb >> 3) + lane;           // one uint32 (8 nibbles) per lane
        uint sbi = ((kb + lane * 8) / KB) * 2; // (scale,bias) pair index for this lane's block
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            float sc = float(SC[sbi, n, e]);       // SC[idx,n,e]=torch aux[e,n,idx]
            float bi = float(SC[sbi + 1, n, e]);
            uint packed = uint(QP[w0, n, e]);
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uint q = (packed >> (j * 4)) & 0xf;
                s += xr[j] * (sc * float(q) + bi);
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r, slot] = TYPE(tot);
    }
"""


def quantize_affine_int4_blockK(weight: torch.Tensor, block: int = _INT4_KBLOCK
                                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Affine 4-bit per-(output-row, K-block) quant with MSE-optimal clip. weight [N,K] ->
    (qp [N, K//8] uint32 (8 nibbles), aux [N, 2·K//block] fp16 interleaved scale,bias).
    ``w ≈ scale·q + bias``, q in [0,15]."""
    W = weight.detach().float()
    N, K = W.shape
    if K % block:
        raise ValueError(f"K={K} not divisible by block {block}")
    Wb = W.reshape(N, K // block, block)
    amax, amin = Wb.amax(-1), Wb.amin(-1)
    mid, half = (amax + amin) * 0.5, (amax - amin) * 0.5         # [N, NB]
    best_err = best_sc = best_lo = best_q = None
    for c in _AFF_CLIPS:
        lo = mid - half * c
        scale = (2.0 * half * c / 15.0).clamp(min=1e-8)
        q = torch.round((Wb - lo.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(0, 15)
        err = ((scale.unsqueeze(-1) * q + lo.unsqueeze(-1)) - Wb).square().sum(-1)
        if best_err is None:
            best_err, best_sc, best_lo, best_q = err, scale, lo, q
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_sc = torch.where(take, scale, best_sc)
            best_lo = torch.where(take, lo, best_lo)
            best_q = torch.where(take.unsqueeze(-1), q, best_q)
    q = best_q.reshape(N, K // 8, 8).to(torch.int64)
    qp = torch.zeros(N, K // 8, dtype=torch.int64)
    for j in range(8):
        qp |= q[..., j] << (j * 4)
    NB = K // block
    aux = torch.empty(N, NB * 2, dtype=torch.float32)
    aux[:, 0::2] = best_sc
    aux[:, 1::2] = best_lo
    return qp.to(torch.uint32).contiguous(), aux.to(torch.float16).contiguous()


def _gather_qmm_int4aff_defn(x: torch.Tensor, qp: torch.Tensor, sc: torch.Tensor,
                             idx: torch.Tensor, block: int) -> torch.Tensor:
    """Reference: gather k experts, affine-int4 dequant (scale·q+bias), per-slot matvec.
    x [k,K]; qp [E,N,K//8] uint32; sc [E,N,2·K//block] fp16 (interleaved scale,bias); idx -> [k,N]."""
    i = idx.to(torch.int32)
    qg = torch.index_select(qp, 0, i)            # [k, N, K//8]
    sg = torch.index_select(sc, 0, i).float()    # [k, N, 2·NB]
    k_, N = qg.shape[0], qg.shape[1]
    K = qg.shape[2] * 8
    p = qg.to(torch.int64)
    q = torch.stack([(p >> (j * 4)) & 0xF for j in range(8)], dim=-1).reshape(k_, N, K).float()
    scale = sg[..., 0::2].repeat_interleave(block, dim=2)   # [k,N,K]
    bias = sg[..., 1::2].repeat_interleave(block, dim=2)
    w = scale * q + bias
    return torch.einsum("sc,snc->sn", x.float(), w).to(x.dtype)


def build_gather_qmm_int4aff(name: str = "gather_qmm_int4aff",
                             block: int = _INT4_KBLOCK) -> TorchMetalKernel:
    """Affine 4-bit gather matvec (per-K-block-32 scale+bias) — the better iPhone 4-bit."""
    src = (_INT4AFF_SRC.replace("__R__", str(_R)).replace("__SGY__", str(_SGY))
           .replace("__KB__", str(block)))

    def defn(x: torch.Tensor, qp: torch.Tensor, sc: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return _gather_qmm_int4aff_defn(x, qp, sc, idx, block)

    return TorchMetalKernel(
        name, input_names=["A", "QP", "SC", "IDX"], result_names=["C"], src=src, torch_defn=defn,
        metal_params=[MetalParameter("tid", "uint3", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint3", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def _quantize_experts_aff4(weight: torch.Tensor, block: int = _INT4_KBLOCK
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    qps, scs = [], []
    for e in range(weight.shape[0]):
        qp, sc = quantize_affine_int4_blockK(weight[e], block=block)
        qps.append(qp)
        scs.append(sc)
    return torch.stack(qps, 0).contiguous(), torch.stack(scs, 0).contiguous()


def _kpad256(w: torch.Tensor, mult: int = 256) -> torch.Tensor:
    """Pad the LAST dim (in-features K) of an [E, out, K] stack up to a multiple of ``mult`` with
    zeros — the gather kernel processes K in 256-wide blocks; zero columns contribute 0. No-op when
    K is already a multiple (LFM/Qwen/GLM); needed for gpt-oss (K=2880 -> 3072)."""
    K = w.shape[-1]
    pad = (-K) % mult
    return w if pad == 0 else torch.nn.functional.pad(w, (0, pad)).contiguous()


class MetalSwitchGLU(nn.Module):
    """Drop-in for ``primitives.macos.switch.SwitchGLU`` whose three expert projections run
    the ``gather_qmm`` kernel — reading ONLY the routed experts (the dense-overread fix).

    Same contract: ``forward(x [b, s, hidden], indices [b, s, k]) -> [b, s, k, hidden]`` (the
    per-expert outputs; the MoE block does the routing-weight combine). Decode-only (b=s=1 ->
    the activation is a single row [1, hidden]; the kernel assumes the token batch is 1).
    Weights are k-means palettized at construction and stored as buffers.
    """

    def __init__(self, switch_glu: nn.Module, kernel: TorchMetalKernel, scheme: str = "sym8",
                 iters: int = 10) -> None:
        super().__init__()
        self.kernel = kernel
        self.scheme = scheme
        self.has_bias = False
        # SwitchLinear.weight is [num_weight_sets=1, E, out, in]; take set 0 -> [E, out, in].
        # Each scheme stores two buffers (qp + aux); the kernel call is uniform (x, qp, aux, idx).
        # K (in-dim) is padded to a multiple of 256 (the kernel's block) — gpt-oss has K=2880;
        # the pad columns are zero weights (contribute 0). Optional per-expert bias (gpt-oss MoE).
        for name in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(switch_glu, name)
            w = _kpad256(lin.weight.detach()[0])               # [E, out, K_pad]
            qp, aux = _quantize_experts_for(w, scheme, iters=iters)
            self.register_buffer(f"{name}_qp", qp)
            self.register_buffer(f"{name}_aux", aux)
            b = getattr(lin, "bias", None)
            if b is not None:
                self.has_bias = True
                self.register_buffer(f"{name}_bias", b.detach()[0].contiguous())  # [E, out]
        self._activate = switch_glu._activate

    @classmethod
    def from_weight_stacks(cls, gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor,
                           activation: nn.Module, kernel: TorchMetalKernel, scheme: str = "sym8",
                           iters: int = 10) -> "MetalSwitchGLU":
        """Build a MetalSwitchGLU directly from three [E, out, in] fp16 expert stacks (no fp16
        SwitchGLU needed) — the STREAMING path: read one MoE layer's experts from the checkpoint,
        quantize here, free the fp16, never materialize the whole model in fp16. ``gate_w``/``up_w``
        are [E, moe_inter, hidden], ``down_w`` is [E, hidden, moe_inter]."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.kernel = kernel
        self.scheme = scheme
        self.has_bias = False
        for name, w in (("gate_proj", gate_w), ("up_proj", up_w), ("down_proj", down_w)):
            qp, aux = _quantize_experts_for(_kpad256(w.detach()), scheme, iters=iters)
            self.register_buffer(f"{name}_qp", qp)
            self.register_buffer(f"{name}_aux", aux)
        self._activate = activation
        return self

    def _proj(self, x_slots: torch.Tensor, name: str, idx: torch.Tensor) -> torch.Tensor:
        """x_slots [k, K] -> [k, N]. Pads K to the kernel's 256 multiple; adds per-expert bias."""
        qp = getattr(self, f"{name}_qp")
        aux = getattr(self, f"{name}_aux")
        # K encoded in qp: sym8/km8 pack 4 bytes/uint32 (qp[2]=K/4); km4/aff4 pack
        # 8 nibbles/uint32 (qp[2]=K/8). (km8 was *8 here — wrong, K/4*8 = 2K.)
        k_pad = qp.shape[2] * (4 if self.scheme in ("sym8", "km8") else 8)
        if x_slots.shape[1] != k_pad:
            x_slots = torch.nn.functional.pad(x_slots, (0, k_pad - x_slots.shape[1])).contiguous()
        N = qp.shape[1]
        y = self.kernel(x_slots, qp, aux, idx, threads_per_grid=(32, N // _R, idx.shape[0]),
                        threads_per_thread_group=(32, _SGY, 1), result_shapes=[[idx.shape[0], N]])
        if self.has_bias:
            b = getattr(self, f"{name}_bias", None)
            if b is not None:
                y = y + torch.index_select(b, 0, idx)   # [k, N] gathered per-expert bias
        return y

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        b, s, hidden = x.shape
        k = indices.shape[-1]
        idx = indices.reshape(k).to(torch.int32)
        # gate/up share the token x: replicate it to one row per routed slot.
        x_rep = x.reshape(1, hidden).expand(k, hidden).contiguous()
        gate = self._proj(x_rep, "gate_proj", idx)   # [k, moe_inter]
        up = self._proj(x_rep, "up_proj", idx)        # [k, moe_inter]
        gated = self._activate(up, gate)              # [k, moe_inter] (per-slot)
        out = self._proj(gated, "down_proj", idx)     # [k, hidden]
        return out.reshape(b, s, k, hidden)


def _quantize_experts_for(weight: torch.Tensor, scheme: str, iters: int = 10
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch the per-scheme expert quantizer -> (qp, aux)."""
    if scheme == "sym8":
        return _quantize_experts_sym(weight)
    if scheme == "aff4":
        return _quantize_experts_aff4(weight)
    if scheme == "km8":
        return _quantize_experts(weight, 8, iters=iters)
    if scheme == "km4":
        return _quantize_experts(weight, 4, iters=iters)
    raise ValueError(f"unknown scheme {scheme!r} (use 'sym8', 'aff4', 'km8', 'km4')")


def _kernel_for(scheme: str) -> TorchMetalKernel:
    if scheme == "sym8":
        return build_gather_qmm_int8sym()
    if scheme == "aff4":
        return build_gather_qmm_int4aff()
    if scheme == "km8":
        return build_gather_qmm_int8km()
    if scheme == "km4":
        return build_gather_qmm_int4km()
    raise ValueError(f"unknown scheme {scheme!r} (use 'sym8', 'aff4', 'km8', 'km4')")


def metalize_moe(model: nn.Module, scheme: str = "sym8", kernel: TorchMetalKernel | None = None,
                 iters: int = 10) -> TorchMetalKernel:
    """Swap every MoE block's ``switch_mlp`` (a ``SwitchGLU``) for a ``MetalSwitchGLU`` reading
    only the routed experts. ``scheme`` in {'sym8' (clean linear int8, default), 'km8', 'km4'}.
    Returns the shared kernel (register it before exporting).

    Walks ``model.model.layers[*].feed_forward.switch_mlp`` — the LFM2.5-MoE / Qwen3.5-MoE
    layout. Additive: routing + combine in the MoE block are untouched."""
    if kernel is None:
        kernel = _kernel_for(scheme)
    n = 0
    for layer in model.model.layers:
        # The SwitchGLU lives under different (holder, attr) across families:
        # LFM2.5-MoE -> feed_forward.switch_mlp ; Qwen3.x/GLM -> mlp.switch_mlp ; gpt-oss -> mlp.experts.
        for holder_name, attr in (("feed_forward", "switch_mlp"), ("mlp", "switch_mlp"), ("mlp", "experts")):
            holder = getattr(layer, holder_name, None)
            sm = getattr(holder, attr, None) if holder is not None else None
            from coreai_models.primitives.macos.switch import SwitchGLU as _SwitchGLU
            if isinstance(sm, _SwitchGLU):
                setattr(holder, attr, MetalSwitchGLU(sm, kernel, scheme, iters=iters))
                n += 1
    if n == 0:
        raise RuntimeError("metalize_moe found no SwitchGLU under feed_forward.switch_mlp / mlp.switch_mlp / mlp.experts")
    return kernel


class BatchedMetalSwitchGLU(MetalSwitchGLU):
    """q=N (batched / diffusion-canvas) variant of :class:`MetalSwitchGLU`.

    q=1 ``MetalSwitchGLU`` runs the gather kernel over the ``k`` (token, expert) slots of a
    SINGLE token. For a batch of ``T = b*s`` tokens (e.g. a 64-token diffusion canvas) each
    routing to ``k`` experts, there are ``P = T*k`` (token, expert) pairs — and the SAME kernel
    runs over all ``P`` of them (its grid z-dim is just "number of pairs"; nothing in the Metal
    source assumes ``k``). The speed lever — matching MLX's ``gather_qmm(sorted_indices=True)``
    and the standard sort→grouped-GEMM MoE technique — is to SORT the pairs by expert id so
    same-expert pairs are adjacent: each expert's weight slab is then read into L2 once and
    reused across all its tokens (grouped locality) instead of being re-fetched per token. The
    output is un-sorted back to (token, slot) order so the MoE block's routing-weight combine is
    unchanged. b*s==1 falls back to the proven q=1 decode path verbatim.
    """

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        b, s, hidden = x.shape
        k = indices.shape[-1]
        if b * s == 1:
            return super().forward(x, indices)            # q=1 decode: proven path, unchanged
        T = b * s
        P = T * k
        x_flat = x.reshape(T, hidden)
        expert_of_pair = indices.reshape(P).to(torch.int32)                     # [P] in (t,j) order
        tok_of_pair = torch.arange(T, device=x.device).repeat_interleave(k)     # [P] token per pair
        # Sort pairs by expert -> same-expert pairs adjacent -> each expert's weight slab is read
        # once into L2 and reused across its tokens (the grouped-GEMM win; MLX sorted_indices).
        # coreai-torch can't lower aten.sort/argsort, but CAN lower topk (the router uses it):
        # topk(-x, P) returns indices in ascending-x order == argsort(x). Explicit dim=0 (the lone
        # axis of this 1-D tensor) dodges the MPSGraph TopK negative-axis constant-fold trap.
        order = torch.topk(-expert_of_pair.to(torch.float32), P, dim=0).indices       # [P] sorted->orig
        inv_order = torch.topk(-order.to(torch.float32), P, dim=0).indices            # [P] orig->sorted
        e_sorted = expert_of_pair[order]                                        # [P]
        a_sorted = x_flat[tok_of_pair[order]]                                   # [P, hidden]
        gate = self._proj(a_sorted, "gate_proj", e_sorted)                      # [P, moe_inter]
        up = self._proj(a_sorted, "up_proj", e_sorted)                          # [P, moe_inter]
        gated = self._activate(up, gate)                                        # [P, moe_inter]
        down = self._proj(gated, "down_proj", e_sorted)                         # [P, hidden]
        out = down[inv_order]                                                   # un-sort -> (t,j)
        return out.reshape(b, s, k, hidden)


def metalize_moe_batched(model: nn.Module, scheme: str = "sym8",
                         kernel: TorchMetalKernel | None = None, iters: int = 10) -> TorchMetalKernel:
    """Like :func:`metalize_moe` but installs the q=N :class:`BatchedMetalSwitchGLU` (for a
    bidirectional diffusion canvas / any q>1 MoE forward). Also matches the DiffusionGemma
    backbone layout, where the ``SwitchGLU`` is ``layer.switch_mlp`` (the layer itself is the
    holder). Returns the shared kernel (register it via ``export_to_coreai_with_kernels``)."""
    if kernel is None:
        kernel = _kernel_for(scheme)
    from coreai_models.primitives.macos.switch import SwitchGLU as _SwitchGLU
    n = 0
    for layer in model.model.layers:
        # (None, attr) => the layer itself holds the SwitchGLU (DiffusionGemma: layer.switch_mlp).
        for holder_name, attr in ((None, "switch_mlp"), ("feed_forward", "switch_mlp"),
                                  ("mlp", "switch_mlp"), ("mlp", "experts")):
            holder = layer if holder_name is None else getattr(layer, holder_name, None)
            sm = getattr(holder, attr, None) if holder is not None else None
            if isinstance(sm, _SwitchGLU):
                setattr(holder, attr, BatchedMetalSwitchGLU(sm, kernel, scheme, iters=iters))
                n += 1
    if n == 0:
        raise RuntimeError("metalize_moe_batched found no SwitchGLU (layer.switch_mlp / "
                           "feed_forward.switch_mlp / mlp.switch_mlp / mlp.experts)")
    return kernel
