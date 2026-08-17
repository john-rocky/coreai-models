# Community port — NOT an Apple model.
"""``qmm_block32`` — fused dequant-gemm Metal kernel vs the static-M quantized-matmul plateau.

THE PROBLEM (SPEC38 Phase H). Statically-compiled per-block-32 quantized matmuls run
~2.1x slow at M=9 (plateaus [1-2]/[4-8]/[9-16]/[17+]; `ondevice/_spec38_gemm_static_probe.py`
is the 5-min repro). The S=9 verify graph's MLP/proj matmuls pay ~30 ms of pure plateau
tax at 27B scale. The bet: read each weight block ONCE and apply it to all M rows, so an
M=9 gemm costs ~an M=1 gemv plus cached activation reads.

VERDICT (2026-08-17, 850M-stack probe, 9 kernel variants): the bet FAILED — best config
(this file) TIES the plateau path at M=9 (3.19 vs stock 3.05 ms) and loses at M=1
(2.02 vs 1.45) and M=17 (12.9 vs 4.8). Both stock-M9 and this kernel land at the same
~5 TFLOP/s effective rate — the M-plateau cost is the practical streaming-gemm ceiling
of this GPU class (matching the A19 matmul2d ~5-6 TFLOP/s finding), not a defective
kernel pick that a custom kernel can dodge. Numerics are clean (single-layer elementwise
exact to fp16-dot rounding, abs err <= 0.004 @ rms 1.4). Kept as the OS-retest harness
(`ondevice/_spec38_qmm_probe.py` / `_spec38_qmm_single.py`) and as the radar companion;
integration below is wired (`--qmm-kernel`) but UNUSED — parity vs a stock decode bundle
was never exercised. Full record: SPEC38_VERIFY_LEVER_STATE.md §PHASE H.

THE KERNEL. Symmetric per-(output-row, K-block-32) int4 — the ship
`symmetric_with_clipping` per_block-32 recipe, shift-free so it mirrors the fast-path
dequant math (scale*q, no zero point). Weights arrive PACKED as baked constants
(uint32, 8 nibbles/word) + fp16 scales — constants are proven-safe kernel edges
(gather_qmm precedent). The activation X [M, K] must be matmul/norm/reshape-derived,
NEVER transpose-derived (kernel-boundary rule, Apple 178056451). fp32 accumulators in
registers; one simd_sum per (output row, M row) at the end.

Structure mirrors `moe_metal.py`'s ``_INT8SYM_SRC`` matvec (R=4 rows x SGY=8 simdgroups
= 32 output rows per threadgroup; K in 256-wide blocks = 32 lanes x 8 nibbles), with the
M rows carried in per-thread registers. M is baked per kernel build (static-S verify
exports; unrolled inner loops).

Constraints: K % 256 == 0, N % 32 == 0, M <= 32.

Integration: ``QuantLinearMetal`` is a drop-in for a QUANTIZED ``nn.Linear`` — either
built from the coreai-opt parametrization (``from_quantized_linear``, bit-identical
weights to the stock bundle => spec-decode losslessness preserved) or self-quantized
from fp16 (``from_fp16``, probe path). ``metalize_qmm`` swaps the tax-carrying Linears
(MLP gate/up/down + GDN in_proj*/out_proj) after ``quantize_pytorch_model``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

_R, _SGY = 4, 8          # R*SGY = 32 output rows per threadgroup (the measured-best tiling)
QMM_KBLOCK = 32          # per-block quant granularity along K (ship recipe)
_SYM_CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)   # MSE-optimal symmetric clip search

# DSL axes are reversed vs torch (DSL dim0 = torch innermost):
#   X  torch [M, K]     -> DSL [K, M]:     X[k, m]
#   QP torch [N, K/8]   -> DSL [K/8, N]:   QP[w, n]   (uint32, 8 nibbles/word, nibble j = k w*8+j)
#   SC torch [N, K/32]  -> DSL [K/32, N]:  SC[b, n]
#   C  torch [M, N]     -> DSL [N, M]:     C[n, m]
#
# Register budget is the M>1 trap. v1 (per-thread xr[M*8] arrays) SPILLED (M=9 7.3 ms).
# v2 (tg-staged X, scalar array wv, tg-load inside the fma chain) reached 4.16. v3
# (m-outer, private xf[8]/wv[R*8] scalar arrays) REGRESSED to 9.6 — the Metal compiler
# does not promote runtime-indexed private arrays; they land in stack memory. v4: NO
# private scalar arrays in the hot path — activations and weights live in half4/float4
# VECTOR registers (two per 8-value group), the 8-term dot is two dot() intrinsics, and
# only acc[R*M] (fully-unrollable literal-indexed loops) remains an array.
_QMM_B32_SRC = """
    const uint K = X.get_extent(0);           // torch innermost of X
    const uint lane = tid.x;                   // 0..31
    const uint sg = tid.y;                     // 0..SGY-1
    const uint t256 = sg * 32 + lane;          // linear id for co-op activation loads
    const uint base_row = (tgid.y * __SGY__ + sg) * __R__;
    const uint xb = lane * 8;                  // this lane's 8-value offset in the staged block

    float acc[__R__ * __M__];
    for (uint i = 0; i < __R__ * __M__; ++i) acc[i] = 0.0f;

    threadgroup half xsh[256 * __M__];         // one 256-K block of X, all M rows
    threadgroup half4* xsh4 = (threadgroup half4*)xsh;

    half4 wlo[__R__];                          // dequanted weights: 2 half4 per output row
    half4 whi[__R__];
    const uint4 shlo = uint4(0, 4, 8, 12);
    const uint4 shhi = uint4(16, 20, 24, 28);

    for (uint kb = 0; kb < K; kb += 256) {     // 32 lanes * 8 nibbles = 256 K per block
        for (uint m = 0; m < __M__; ++m)       // co-op: 256 threads x M, coalesced in k
            xsh[m * 256 + t256] = X[kb + t256, m];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint w0 = (kb >> 3) + lane;            // this lane's uint32 word (8 nibbles)
        uint sb = (kb + xb) >> 5;              // K-block-32 scale index (8 | 32, never straddles)
        for (uint r = 0; r < __R__; ++r) {     // dequant ONCE per K-block (vector nibble ops)
            half hsc = SC[sb, base_row + r];
            uint packed = uint(QP[w0, base_row + r]);
            int4 qlo = int4((uint4(packed) >> shlo) & 0xfu);
            int4 qhi = int4((uint4(packed) >> shhi) & 0xfu);
            wlo[r] = half4((qlo ^ 8) - 8) * hsc;   // two's-complement nibbles * scale (fp16)
            whi[r] = half4((qhi ^ 8) - 8) * hsc;
        }
        for (uint m = 0; m < __M__; ++m) {
            half4 a = xsh4[m * 64 + lane * 2];         // this lane's 8 activations, row m
            half4 b = xsh4[m * 64 + lane * 2 + 1];
            for (uint r = 0; r < __R__; ++r)           // 8-term dot in fp16, accumulate fp32
                acc[r * __M__ + m] += float(dot(a, wlo[r]) + dot(b, whi[r]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // before next block overwrites xsh
    }
    for (uint r = 0; r < __R__; ++r)
        for (uint m = 0; m < __M__; ++m) {
            float tot = simd_sum(acc[r * __M__ + m]);
            if (lane == 0) C[base_row + r, m] = TYPE(tot);
        }
"""


def qmm_rows_per_thread(m: int) -> int:
    """Output rows per thread (R). Loads/cvt overhead scales with M * total threads =
    M * 32N/R, so KEEP R HIGH as M grows (the v3/v4 lesson — halving R doubled the m-loop
    overhead and M=9 regressed 4.2 -> 9.5). R=4 everywhere; acc[4*M] spills past M~9
    (M=17 measured 2.7x stock) but M<=9 is the only shipping shape."""
    return 4


def pack_int4_nib_u32(q: torch.Tensor) -> torch.Tensor:
    """[N, K] signed int4 (int8 storage, [-8, 7]) -> [N, K//8] uint32, nibble j = k w*8+j."""
    N, K = q.shape
    if K % 8:
        raise ValueError(f"K={K} not divisible by 8")
    nib = (q.to(torch.int64) & 0xF).reshape(N, K // 8, 8)
    w = torch.zeros(N, K // 8, dtype=torch.int64)
    for j in range(8):
        w |= nib[..., j] << (4 * j)
    return w.to(torch.uint32).contiguous()


def _unpack_int4_signed(qp: torch.Tensor) -> torch.Tensor:
    """[N, K//8] uint32 -> [N, K] signed int64 (inverse of pack_int4_nib_u32)."""
    N, K8 = qp.shape
    p = qp.to(torch.int64)
    nib = torch.stack([(p >> (4 * j)) & 0xF for j in range(8)], dim=-1)   # [N, K/8, 8] in 0..15
    return (nib - 16 * (nib >= 8)).reshape(N, K8 * 8)


def quantize_sym_int4_blockK(weight: torch.Tensor, block: int = QMM_KBLOCK
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-(output-row, K-block) int4 with MSE-optimal clipping (probe path).

    weight [N, K] -> (qp [N, K//8] uint32 packed signed int4, scale [N, K//block] fp16).
    ``w[n,k] ≈ scale[n, k//block] · q[n,k]`` with q in [-7, 7]. NOTE: for model
    integration prefer :meth:`QuantLinearMetal.from_quantized_linear` (bit-identical to
    the coreai-opt ship quantizer); this local search only mirrors its semantics."""
    W = weight.detach().float()
    N, K = W.shape
    if K % block:
        raise ValueError(f"K={K} not divisible by block {block}")
    Wb = W.reshape(N, K // block, block)
    amax = Wb.abs().amax(-1)                                   # [N, NB]
    best_err = best_scale = best_q = None
    for c in _SYM_CLIPS:
        scale = (amax * c / 7.0).clamp(min=1e-8)
        q = torch.round(Wb / scale.unsqueeze(-1)).clamp(-7, 7)
        err = ((scale.unsqueeze(-1) * q) - Wb).square().sum(-1)
        if best_err is None:
            best_err, best_scale, best_q = err, scale, q
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_scale = torch.where(take, scale, best_scale)
            best_q = torch.where(take.unsqueeze(-1), q, best_q)
    q_int4 = best_q.reshape(N, K).to(torch.int8)
    return pack_int4_nib_u32(q_int4), best_scale.to(torch.float16).contiguous()


def _qmm_defn(x: torch.Tensor, qp: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    """Reference + shape inference: x [M, K] fp16, qp [N, K//8] uint32, sc [N, K//32]
    fp16 -> [M, N]. Numerically exact dequant matmul (the eager-vs-GPU gate)."""
    q = _unpack_int4_signed(qp).to(torch.float32)                       # [N, K]
    w = q * sc.to(torch.float32).repeat_interleave(QMM_KBLOCK, dim=1)   # [N, K]
    return (x.to(torch.float32) @ w.t()).to(x.dtype)


def build_qmm_block32_kernel(m: int, name: str | None = None) -> TorchMetalKernel:
    """Build the fused dequant-gemm kernel for a fixed row count ``m`` (static-S graphs)."""
    if not 1 <= m <= 32:
        raise ValueError(f"M={m} out of range (M<=32: rows are carried in registers)")
    r = qmm_rows_per_thread(m)
    src = (_QMM_B32_SRC.replace("__R__", str(r)).replace("__SGY__", str(_SGY))
           .replace("__NT__", str(32 * _SGY)).replace("__M__", str(m)))
    kernel = TorchMetalKernel(
        name or f"qmm_block32_m{m}",
        input_names=["X", "QP", "SC"], result_names=["C"], src=src, torch_defn=_qmm_defn,
        metal_params=[MetalParameter("tid", "uint3", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint3", "threadgroup_position_in_grid")],
        template_dtypes={"X": "TYPE"},
    )
    kernel._qmm_r = r          # rows/thread — QuantLinearMetal derives its dispatch from this
    return kernel


class QuantLinearMetal(nn.Module):
    """Drop-in for a quantized ``nn.Linear`` running the ``qmm_block32`` kernel.

    Holds the packed weights/scales as buffers (baked constants in the graph).
    ``forward(x [b, s, K]) -> [b, s, N]`` with b == 1 and s == the kernel's baked M.
    The input must be matmul/norm/reshape-derived (kernel-boundary rule)."""

    def __init__(self, kernel: TorchMetalKernel, qp: torch.Tensor, sc: torch.Tensor,
                 bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.kernel = kernel
        self.r = getattr(kernel, "_qmm_r", _R)
        n, k8 = qp.shape
        if n % (self.r * _SGY):
            raise ValueError(f"N={n} not divisible by {self.r * _SGY}")
        if (k8 * 8) % 256:
            raise ValueError(f"K={k8 * 8} not divisible by 256")
        self.n_out, self.k_in = n, k8 * 8
        self.register_buffer("qp", qp)
        self.register_buffer("sc", sc)
        self.bias = None
        if bias is not None:
            self.register_buffer("b", bias.detach().clone())
            self.bias = True

    @classmethod
    def from_fp16(cls, kernel: TorchMetalKernel, weight: torch.Tensor,
                  bias: torch.Tensor | None = None) -> "QuantLinearMetal":
        """Self-quantize from an fp16 weight (probe path; NOT bit-identical to coreai-opt)."""
        qp, sc = quantize_sym_int4_blockK(weight)
        return cls(kernel, qp, sc, bias)

    @classmethod
    def from_quantized_linear(cls, kernel: TorchMetalKernel, lin: nn.Linear,
                              ) -> "QuantLinearMetal":
        """Extract the EXACT (q, scale) a coreai-opt ``quantize_pytorch_model`` pass left in
        ``lin.parametrizations['weight']`` — weights bit-identical to the stock bundle.
        Asserts the packed dequant reproduces ``lin.weight`` exactly."""
        par = lin.parametrizations["weight"][0]
        qd = par.quantized_data.detach()            # int4 stored as int8, [N, K]
        sc = par.scale.detach()                     # per-block scale, [N, K/32(, 1)]
        zp = getattr(par, "zero_point", None)
        if zp is not None and torch.count_nonzero(zp):
            raise ValueError("qmm_block32 is symmetric-only (nonzero zero_point found)")
        N, K = qd.shape[0], qd.shape[-1] if qd.ndim == 2 else None
        if qd.ndim != 2:
            raise ValueError(f"unexpected quantized_data shape {tuple(qd.shape)}")
        sc2 = sc.reshape(N, -1).to(torch.float16)
        if sc2.shape[1] != K // QMM_KBLOCK:
            raise ValueError(f"scale shape {tuple(sc.shape)} != [N, K/{QMM_KBLOCK}]")
        qp = pack_int4_nib_u32(qd.to(torch.int8))
        self = cls(kernel, qp, sc2,
                   lin.bias.detach() if lin.bias is not None else None)
        # exactness gate: packed dequant == the parametrization's dequant output
        want = lin.weight.detach()
        got = (_unpack_int4_signed(qp).to(torch.float32)
               * sc2.to(torch.float32).repeat_interleave(QMM_KBLOCK, dim=1)).to(want.dtype)
        if not torch.equal(got, want):
            diff = (got.float() - want.float()).abs().max().item()
            raise ValueError(f"packed dequant != parametrized weight (max diff {diff})")
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, k = x.shape
        x2 = x.reshape(s, k)                       # reshape-derived dense edge (b == 1)
        y = self.kernel(
            x2, self.qp, self.sc,
            threads_per_grid=(32, self.n_out // self.r, 1),
            threads_per_thread_group=(32, _SGY, 1),
            result_shapes=[[s, self.n_out]])
        y = y.reshape(b, s, self.n_out)
        if self.bias:
            y = y + self.b
        return y


def metalize_qmm(model: nn.Module, s: int, kernel: TorchMetalKernel | None = None,
                 ) -> TorchMetalKernel:
    """Swap the tax-carrying quantized Linears for QuantLinearMetal (call AFTER
    ``quantize_pytorch_model``). Targets: MLP gate/up/down (all layers) + GDN
    in_proj_qkv/in_proj_z/in_proj_b/in_proj_a/out_proj (linear layers). Attention
    q/k/v/o stay stock (o_proj input is transpose-derived — boundary rule).
    Skips any Linear whose (N, K) violates the kernel divisibility constraints.
    Returns the shared kernel (register it with the converter)."""
    if kernel is None:
        kernel = build_qmm_block32_kernel(s)
    n_swap, n_skip = 0, 0

    def _try_swap(holder: nn.Module, attr: str) -> None:
        nonlocal n_swap, n_skip
        lin = getattr(holder, attr, None)
        if lin is None or not isinstance(lin, nn.Linear):
            return
        if not hasattr(lin, "parametrizations"):
            return                                  # not quantized (e.g. excluded module)
        n, k = lin.weight.shape
        if n % (getattr(kernel, "_qmm_r", _R) * _SGY) or k % 256:
            n_skip += 1
            return
        setattr(holder, attr, QuantLinearMetal.from_quantized_linear(kernel, lin))
        n_swap += 1

    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for attr in ("gate_proj", "up_proj", "down_proj"):
                _try_swap(mlp, attr)
        la = getattr(layer, "linear_attn", None)
        if la is not None:
            for attr in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                _try_swap(la, attr)
    if n_swap == 0:
        raise RuntimeError("metalize_qmm swapped nothing (quantize first?)")
    print(f"[qmm] swapped {n_swap} quantized Linears to qmm_block32 (skipped {n_skip} "
          f"on divisibility)")
    return kernel
