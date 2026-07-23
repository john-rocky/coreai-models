# Community port — NOT an Apple model.
"""Fused 1.58-bit TERNARY matvec Metal kernel for BitCPM (MiniCPM4-8B, TQ2_0) decode (q=1).

zoo's first sub-int8 PACKED-GEMM kernel — the moat. A simpler sibling of the int4 k-means matvec
in ``gemma4_metal_mlp.py``: no codebook gather (the "palette" is just {-1,0,+1}), 2 bits/weight.

TQ2_0 semantics (llama.cpp): per (output row n, block of 256 reduction-axis K) a weight is a 2-bit
code q in {0,1,2} with value ``(q-1)`` in {-1,0,+1}, scaled by one fp16 ``d`` per block:
    ``W[n,k] = d[n, k//256] * (q[n,k] - 1)``.

Packing: 16 codes / uint32 (true 2-bit). The decode block is 512 K = 32 lanes x 16 codes/lane, so
each lane's 16 codes lie inside ONE 256-scale-block (16 | 256) and the per-block scale is applied
PER LANE before ``simd_sum`` (dequant is linear: ``sum_k x_k d_b (q_k-1) = sum_b d_b sum_{k in b}
x_k (q_k-1)``). Multi-row (R rows/simd-group) + tg structure copied from the int4 affine kernel.

INDEXING (DSL reverses axes, see gemma4_metal_mlp): QP torch [N, K/16] uint32, QP[w,n]=torch qp[n,w]
(coalesced over lanes at fixed kb); D torch [N, K/256] fp16, D[g,n]=torch d[n,g].

Constraints: K % 512 == 0 (4096/16384 ok), N % (R*SGY)=32 == 0 (256/4096/16384 ok). Covers the 7
per-layer linears (q/k/v/o, gate/up/down); embed (Q4_K) + head (Q6_K) stay higher-precision.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

_R, _SGY = 4, 8          # rows/simd-group, simd-groups/threadgroup -> 32 rows (= one scale never needed across rows)
_KB = 512                # K per decode block (32 lanes * 16 codes)
_SCALE_BLK = 256         # TQ2_0 scale granularity along K

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
        float xr[16];
        for (uint j = 0; j < 16; ++j) xr[j] = float(A[k0 + j, 0]);
        uint w0 = (kb >> 4) + lane;             // one uint32 (16 codes) per lane
        uint g  = k0 >> 8;                       // 256-scale block (this lane's 16 codes are inside it)
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = uint(QP[w0, n]);       // QP[w,n]=torch qp[n,w]
            float p = 0.0f;
            for (uint j = 0; j < 16; ++j) {
                int q = int((packed >> (j * 2)) & 0x3);   // 0,1,2
                p += xr[j] * float(q - 1);                // {-1,0,+1}
            }
            acc[r] += p * float(D[g, n]);        // per-lane-block scale, then accumulate
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r, 0] = TYPE(tot);
    }
"""


def ternary_from_dequant(weight: torch.Tensor, scale_blk: int = _SCALE_BLK
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover (codes {0,1,2} [N,K] uint8, d [N, K/scale_blk] fp16) from a TQ2_0-dequantized weight
    [N, K] (== ``d*(q-1)``). d_block = the block's nonzero magnitude (exact); all-zero block -> d=0."""
    W = weight.detach().float()
    N, K = W.shape
    if K % scale_blk:
        raise ValueError(f"K={K} not divisible by scale block {scale_blk}")
    Wb = W.reshape(N, K // scale_blk, scale_blk)
    d = Wb.abs().amax(-1)                                   # [N, K/blk]; nonzero |w| == d
    q = torch.where(d.unsqueeze(-1) > 0, (Wb / d.unsqueeze(-1).clamp(min=1e-12)).round(),
                    torch.zeros_like(Wb))                    # {-1,0,1}
    codes = (q.clamp(-1, 1) + 1).to(torch.uint8).reshape(N, K)   # {0,1,2}
    return codes, d.to(torch.float16)


def pack_tern_u32(codes: torch.Tensor) -> torch.Tensor:
    """[N, K] uint8 codes in {0,1,2} -> [N, K//16] uint32 (16 codes/word, 2 bits each, low = lowest k)."""
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
    """Reference: unpack codes, dequant (q-1)*d per 256-block, linear. C[1,N] = x @ W.T."""
    N, K16 = qp.shape
    K = K16 * 16
    blk = K // d.shape[1]
    codes = _unpack_tern(qp)                                # [N, K] {0,1,2}
    val = (codes - 1).float()                              # {-1,0,1}
    scale = d.float().repeat_interleave(blk, dim=1)        # [N, K]
    w = val * scale
    return torch.nn.functional.linear(x, w.to(x.dtype))


def build_tern_kernel(name: str = "bitcpm_ffn_fused_ternary") -> TorchMetalKernel:
    """Fused 2-bit ternary matvec (16 codes/uint32, (q-1)*d dequant, multi-row + simd_sum)."""
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
    """Dispatch ternary (qp [N,K/16] uint32, d [N,K/256] fp16) -> [1, N]."""
    N = qp.shape[0]
    return kernel(x_row, qp, d, threads_per_grid=(32, N // _R, 1),
                  threads_per_thread_group=(32, _SGY, 1), result_shapes=[[1, N]])


class MetalTernaryLinear(nn.Module):
    """nn.Linear drop-in (y = x @ W.T, q=1 decode) on the fused 2-bit ternary matvec.

    Construct from the TQ2_0-dequantized weight [N, K]; the codes+scale are re-derived exactly.
    N % 32 == 0 and K % 512 == 0 (every BitCPM-8B layer linear qualifies)."""

    def __init__(self, weight: torch.Tensor, kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.kernel = kernel
        self.N = int(weight.shape[0])
        if self.N % (_R * _SGY):
            raise ValueError(f"N={self.N} not divisible by {_R * _SGY}")
        if weight.shape[1] % _KB:
            raise ValueError(f"K={weight.shape[1]} not divisible by {_KB}")
        codes, d = ternary_from_dequant(weight)
        self.register_buffer("qp", pack_tern_u32(codes))   # [N, K/16] uint32
        self.register_buffer("d", d)                        # [N, K/256] fp16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, k = x.shape                                   # decode: 1, 1, K
        y = tern_call(self.kernel, x.reshape(s, k), self.qp, self.d)
        return y.reshape(b, s, self.N)
