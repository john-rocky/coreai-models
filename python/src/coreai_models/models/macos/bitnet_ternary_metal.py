# Community port — NOT an Apple model.
"""Fused 1.58-bit TERNARY matvec Metal kernel for BitNet b1.58 (W1.58-A8) decode (q=1).

Generalization of ``bitcpm_ternary_metal`` for the BitVLA / BitNet-b1.58-2B-4T weight scheme,
which differs from BitCPM's TQ2_0 gguf in two ways that the BitCPM kernel can't express:

1. **Scale is PER-TENSOR absmean, not per-256-block.** transformers ``WeightQuant``:
   ``s = 1/mean(|W|); W_q = round(W*s).clamp(-1,1)/s`` -> values in {-mean_w, 0, +mean_w}.
   So one scalar ``mean_w`` per weight (carried here as a per-row buffer, all rows equal).
2. **Activations are quantized too (A8).** ``ActQuant``: per-token int8 absmax,
   ``s = 127/max|x|; x_q = round(x*s).clamp(-128,127)/s`` -- applied in the wrapper BEFORE the
   kernel so ``BitLinearMetal(x) == F.linear(ActQuant(x), WeightQuant(W))`` bit-for-bit.

It also drops the BitCPM kernel's dimension constraints, which BitVLA's shapes violate
(down_proj K=6912 %512=256; every SigLIP vision linear K in {1152,4304}; fc1 N=4304 %32=16):

* **Arbitrary K** (only K % 16 == 0, for 16-codes/uint32 packing). The 32-lane * 16-code super
  block still steps by 512, but a per-lane ``k0 < K`` guard zeroes the tail. Because K % 16 == 0
  and k0 is a multiple of 16, each lane's 16-code chunk is wholly in- or out-of-range -- no
  partial-chunk masking needed, and all 32 lanes still reach ``simd_sum``.
* **Arbitrary N** -- the wrapper pads output rows to a multiple of R*SGY=32 (padded rows are
  computed then sliced off).

Packing/indexing identical to ``bitcpm_ternary_metal`` (QP torch [N, K/16] uint32, QP[w,n]=qp[n,w];
16 codes/word, code q in {0,1,2} = value (q-1)). Scale applied once per row after ``simd_sum``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

_R, _SGY = 4, 8          # rows/simd-group, simd-groups/threadgroup -> 32 rows/threadgroup
_NPAD = _R * _SGY        # 32: N padded to a multiple of this

_TERN_SRC = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);        // A torch [1, K]
    const uint lane = tid.x;               // 0..31
    const uint sg = tid.y;                 // 0..SGY-1
    const uint base_row = (tgid.y * SGY + sg) * R;

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 512) {      // 32 lanes * 16 codes = 512
        uint k0 = kb + lane * 16;
        if (k0 < K) {                            // tail guard (K%16==0 -> chunk wholly in/out)
            float xr[16];
            for (uint j = 0; j < 16; ++j) xr[j] = float(A[k0 + j, 0]);
            uint w0 = (kb >> 4) + lane;          // one uint32 (16 codes) per lane = k0/16
            for (uint r = 0; r < R; ++r) {
                uint n = base_row + r;
                uint packed = uint(QP[w0, n]);   // QP[w,n]=torch qp[n,w]
                float p = 0.0f;
                for (uint j = 0; j < 16; ++j) {
                    int q = int((packed >> (j * 2)) & 0x3);   // 0,1,2
                    p += xr[j] * float(q - 1);                // {-1,0,+1}
                }
                acc[r] += p;
            }
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r, 0] = TYPE(tot * float(D[0, base_row + r]));
    }
"""


def bitnet_decompose(weight: torch.Tensor) -> tuple[torch.Tensor, float]:
    """transformers BitNet ``WeightQuant``: per-tensor absmean ternary.
    Returns (codes {0,1,2} [N,K] uint8, mean_w scalar). Value W_q[n,k] = mean_w * (code-1)."""
    W = weight.detach().float()
    mean_w = float(W.abs().mean().clamp(min=1e-5))
    q = (W / mean_w).round().clamp(-1, 1)        # {-1,0,1}
    codes = (q + 1).to(torch.uint8)              # {0,1,2}
    return codes, mean_w


def act_quant(x: torch.Tensor) -> torch.Tensor:
    """transformers BitNet ``ActQuant``: per-token int8 absmax, dequantized back to fp."""
    xf = x.float()
    s = 127.0 / xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
    return ((xf * s).round().clamp(-128, 127) / s).to(x.dtype)


def pack_tern_u32(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] uint8 codes in {0,1,2} -> [N, K//16] uint32 (16 codes/word, low = lowest k)."""
    N, K = codes.shape
    if K % 16:
        raise ValueError(f"K={K} not divisible by 16")
    if int(codes.max()) > 2:
        raise ValueError("ternary codes exceed {0,1,2}")
    c = codes.to(torch.int64).reshape(N, K // 16, 16)
    qp = torch.zeros(N, K // 16, dtype=torch.int64)
    for j in range(16):
        qp |= c[..., j] << (j * 2)
    return qp.to(torch.uint32).contiguous()


def _unpack_tern(qp: torch.Tensor) -> torch.Tensor:
    """[N, K//16] uint32 -> [N, K] int64 codes {0,1,2}."""
    N, K16 = qp.shape
    p = qp.to(torch.int64)
    return torch.stack([(p >> (j * 2)) & 0x3 for j in range(16)], dim=-1).reshape(N, K16 * 16)


def _tern_torch_defn(x: torch.Tensor, qp: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Reference: C[1,N] = x @ (mean_w * (codes-1)).T. d is [N, 1] per-row scale."""
    codes = _unpack_tern(qp)                       # [N, K] {0,1,2}
    val = (codes - 1).float()                      # {-1,0,1}
    w = val * d.reshape(-1, 1).float()             # [N, K] * [N, 1]
    return torch.nn.functional.linear(x, w.to(x.dtype))


def build_tern_kernel(name: str = "bitnet_fused_ternary") -> TorchMetalKernel:
    """Fused 2-bit ternary matvec (per-row scale, arbitrary K%16, multi-row + simd_sum)."""
    return TorchMetalKernel(
        name, input_names=["A", "QP", "D"], result_names=["C"],
        src=_TERN_SRC.replace("__R__", str(_R)).replace("__SGY__", str(_SGY)),
        torch_defn=_tern_torch_defn,
        metal_params=[MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def tern_call(kernel: TorchMetalKernel, x_row: torch.Tensor, qp: torch.Tensor,
              d: torch.Tensor) -> torch.Tensor:
    """Dispatch ternary (qp [N,K/16] uint32, d [N,1] fp16) -> [1, N]. N must be %32."""
    N = qp.shape[0]
    return kernel(x_row, qp, d, threads_per_grid=(32, N // _R, 1),
                  threads_per_thread_group=(32, _SGY, 1), result_shapes=[[1, N]])


class BitLinearMetal(nn.Module):
    """nn.Linear drop-in for BitNet b1.58 (W1.58-A8, y = ActQuant(x) @ WeightQuant(W).T, q=1 decode).

    Construct from the bf16 master weight [N, K]; per-tensor absmean ternary codes + scalar scale
    are derived exactly. Activations are int8-quantized here before the kernel (== HF reference).
    K % 16 == 0 (always true); N is padded internally to a multiple of 32."""

    def __init__(self, weight: torch.Tensor, kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.kernel = kernel
        self.N = int(weight.shape[0])
        K = int(weight.shape[1])
        if K % 16:
            raise ValueError(f"K={K} not divisible by 16")
        codes, mean_w = bitnet_decompose(weight)
        n_pad = (self.N + _NPAD - 1) // _NPAD * _NPAD
        if n_pad != self.N:                                # pad output rows to %32
            codes = torch.cat([codes, torch.ones(n_pad - self.N, K, dtype=codes.dtype)], 0)
        self.register_buffer("qp", pack_tern_u32(codes))   # [N_pad, K/16] uint32
        # d torch [N_pad, 1] so the DSL (axes reversed) reads D[0, n] == torch d[n, 0] = mean_w
        self.register_buffer("d", torch.full((n_pad, 1), mean_w, dtype=torch.float16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, k = x.shape                                   # decode: 1, 1, K
        xq = act_quant(x)
        y = tern_call(self.kernel, xq.reshape(s, k), self.qp, self.d)   # [1, N_pad]
        return y[:, :self.N].reshape(b, s, self.N)
