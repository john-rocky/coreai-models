# Community port — NOT an Apple model.
"""FP4 (E2M1) fused decode matvec — the QUALITY-SAFE twin of the int4km dense-path lever.

Companion to ``gemma4_metal_mlp.py`` (int4km / int8 kernels — imported, not modified) and
``gemma4_metal_attn_int4km.py``. The dense-path-coverage lever (lm_head + attn q/o on a fused
4-bit matvec) gives the flagship decode ~2x (¼ the weight bytes), but **int4 degrades flagship
quality** (the Nanbeige/RTN cliff). FP4 (E2M1) reads the SAME ¼ bytes yet holds int8-level
quality (de-risked: fp4 ≈ int8 perplexity, ``project_quant_d_port``). This file is the drop-in
fp4 replacement for that matvec.

WHAT DIFFERS FROM int4km (everything else — packing, tiling, dispatch — is identical):
  * dequant. int4km gathers a per-output-group 16-entry k-means codebook (``cbt[q]``). fp4 maps
    the 4-bit code through the FIXED universal E2M1 grid (16 constants) and multiplies by a
    per-K-block **e8m0 power-of-2 scale** — so the kernel is structured like the AFFINE int4 kernel
    (scale along the reduction axis K), minus the bias. No codebook, no tg-staging.
  * quantization is coreai-opt's fp4: weight-only, PerBlockGranularity block_size=32 along K,
    scale = ``2^(floor(log2(max_abs)) - 2)`` (e8m0), codes via torchao ``f32_to_f4_unpacked`` — so
    the kernel reproduces the de-risked fp4 numerics bit-for-bit.

This is a DECODE-bandwidth lever on a hand-rolled matvec — it does NOT use TensorOps ``matmul2d``
(the A19-refuted *prefill* lever) and needs no OS27 (runs on the Mac GPU today). OS27 only matters
if/when this moves to the neural accelerator.

Shape constraints (same as int4km): K % 256 == 0 (32 lanes × 8 codes) and N % 32 == 0. Also K % 32
== 0 (the scale block). Flagship dense shapes qualify: q 2048×4096, o 4096×2048, lm_head 2048×248320.
"""
from __future__ import annotations

import torch
from torch import nn

from coreai_models.models.macos.gemma4_metal_mlp import (
    MetalParameter,
    TorchMetalKernel,
    _unpack_nib,
    _V2_R,
    _V2_SGY,
    pack_idx_nib_u32,
)

# E2M1 grid in torchao code order (code = f4_unpacked; verified vs f4_unpacked_to_f32(arange(16))).
# bit3 = sign, bits2-0 = magnitude index into [0, 0.5, 1, 1.5, 2, 3, 4, 6].
FP4_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
FP4_LUT = torch.tensor(FP4_VALUES, dtype=torch.float32)
FP4_MAX = 6.0

_FP4_G = 32  # e8m0 scale block along K (coreai-opt fp4 block_size; 8 | G, 32 | 256)

# MSL: same multi-row (R) + simd_sum tiling as the affine/int4km matvec; dequant = FP4[code] * scale.
# QP torch [N, K/8] uint32 (8 codes/word); SC torch [N, K/G] fp16; QP[w,n]=torch qp[n,w], SC[g,n]=torch sc[n,g].
_FP4_SRC = """
    const uint R = __R__, SGY = __SGY__, G = __G__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    // Stage the UNIVERSAL 16-entry E2M1 grid into threadgroup memory ONCE (like int4km's codebook):
    // the hot-loop lookup ``lut[c]`` is then a fast tg-mem gather, not a dynamically-indexed local
    // array (which spills to stack on Apple GPUs and made this kernel ~1.4x slower than int4km).
    threadgroup float lut[16];
    if (sg == 0 && lane < 16) {
        const float FP4[16] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
        lut[lane] = FP4[lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {          // block = 32 lanes * 8 codes = 256 K
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(A[k0 + j, 0]);
        uint w0 = (kb >> 3) + lane;                 // one uint32 word (8 codes) per lane
        uint grp = k0 / G;                          // scale block (const for this lane's 8 vals; 8 | G)
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = uint(QP[w0, n]);
            float sc = float(SC[grp, n]);
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uint c = (packed >> (j * 4)) & 0xf;
                s += xr[j] * (lut[c] * sc);         // E2M1 dequant: grid[code] * e8m0 scale (tg-mem gather)
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r, 0] = TYPE(tot);
    }
"""


def quantize_fp4_e2m1(weight: torch.Tensor, block_size: int = _FP4_G
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """FP4 (E2M1) weight-only quant of an nn.Linear weight [N, K], per-block along K (block_size=32).

    Reproduces coreai-opt's fp4 path (bit-identical): e8m0 power-of-2 block scale
    ``2^(floor(log2(max_abs)) - 2)`` and torchao ``f32_to_f4_unpacked`` code assignment. Returns
    ``(qp [N, K//8] uint32 (8 codes/word), scale [N, K//block] fp16)`` — the format the kernel reads.
    """
    from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked  # noqa: PLC0415

    W = weight.detach().float()
    N, K = W.shape
    if K % block_size:
        raise ValueError(f"K={K} not divisible by block_size {block_size}")
    if K % 8:
        raise ValueError(f"K={K} not divisible by 8 (uint32 code pack)")
    Wb = W.reshape(N, K // block_size, block_size)
    amax = Wb.abs().amax(-1, keepdim=True).clamp_min(1e-20)
    scale = torch.exp2(torch.floor(torch.log2(amax)) - 2.0)     # e8m0, target_max_pow2 = 2 (E2M1)
    scaled = (Wb / scale).clamp(-FP4_MAX, FP4_MAX)
    codes = f32_to_f4_unpacked(scaled).reshape(N, K).to(torch.uint8)   # 0..15
    qp = pack_idx_nib_u32(codes)                                # [N, K/8] uint32
    sc = scale.reshape(N, K // block_size).to(torch.float16).contiguous()
    return qp, sc


def _fused_fp4_torch_defn(x: torch.Tensor, qp: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    """Reference: unpack 8 codes/word, E2M1-dequant (grid[code] * block scale), linear. C[1,N] = x @ W.T."""
    N, K8 = qp.shape
    K = K8 * 8
    G = K // sc.shape[1]
    codes = _unpack_nib(qp)                                     # [N, K] int64, 0..15
    vals = FP4_LUT.to(qp.device)[codes]                         # [N, K] E2M1 grid values
    scale = sc.float().repeat_interleave(G, dim=1)             # [N, K]
    w = vals * scale
    return torch.nn.functional.linear(x, w.to(x.dtype))


def build_fused_fp4_kernel(name: str = "gemma4_ffn_fused_fp4",
                           block_size: int = _FP4_G) -> TorchMetalKernel:
    """FP4 (E2M1) fused matvec (8 codes/uint32, grid[code]*e8m0-scale dequant, multi-row + simd_sum).

    ``block_size`` MUST match the quantization block_size (it indexes the scale along K); 8 | block_size."""
    return TorchMetalKernel(
        name, input_names=["A", "QP", "SC"], result_names=["C"],
        src=(_FP4_SRC.replace("__R__", str(_V2_R)).replace("__SGY__", str(_V2_SGY))
             .replace("__G__", str(block_size))),
        torch_defn=_fused_fp4_torch_defn,
        metal_params=[MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def fused_fp4_call(kernel: TorchMetalKernel, x_row: torch.Tensor, qp: torch.Tensor,
                   sc: torch.Tensor) -> torch.Tensor:
    """Dispatch fp4 (qp [N,K/8] uint32, sc [N,K/block] fp16) -> [1, N]."""
    N = qp.shape[0]
    return kernel(x_row, qp, sc, threads_per_grid=(32, N // _V2_R, 1),
                  threads_per_thread_group=(32, _V2_SGY, 1), result_shapes=[[1, N]])


class MetalFP4Linear(nn.Module):
    """Drop-in for an ``nn.Linear`` (q=1 decode) whose matvec runs the FUSED fp4-E2M1 kernel.

    The quality-safe twin of ``MetalInt4KMLinear``: SAME ¼-byte weight reads, E2M1 numerics instead of
    int4 k-means. Storage = packed fp4 codes ``qp [N, K/8] uint32`` + e8m0 block scale ``sc [N, K/block]
    fp16``. Share ONE kernel object across FFN + attention + head.
    """

    def __init__(self, lin: nn.Linear, kernel: TorchMetalKernel, block_size: int = _FP4_G) -> None:
        super().__init__()
        self.kernel = kernel
        n, k = lin.weight.shape
        if k % 256:
            raise ValueError(f"K={k} not divisible by 256 (32 lanes x 8 codes)")
        if n % 32:
            raise ValueError(f"N={n} not divisible by 32 (multi-row tiling R*SGY)")
        self.N = int(n)
        qp, sc = quantize_fp4_e2m1(lin.weight, block_size=block_size)
        self.register_buffer("qp", qp)  # [N, K/8] uint32
        self.register_buffer("sc", sc)  # [N, K/block] fp16

    @property
    def weight(self):
        # Mirror MetalInt4KMLinear: some forwards read ``.weight.dtype`` to cast the input
        # (e.g. lfm2/qwen attention). Expose the fp16 scale so ``.dtype`` answers like the nn.Linear.
        return self.sc

    def forward(self, x):
        b, s, k = x.shape  # decode: b=1, s=1
        y = fused_fp4_call(self.kernel, x.reshape(s, k), self.qp, self.sc)
        return y.reshape(b, s, self.N)


def metalize_dense_fp4(model, kernel) -> int:
    """FP4 twin of ``metalize_dense_int4km``: swap the dense matvecs that dominate the per-token read
    (lm_head + full-attention q_proj / o_proj) for the fused fp4 kernel. k/v_proj stay fp16 (small-N).
    Shares ONE kernel. Returns the number of matvecs replaced."""
    n = 0
    model.lm_head = MetalFP4Linear(model.lm_head, kernel)
    n += 1
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "q_proj") and hasattr(attn, "o_proj"):
            attn.q_proj = MetalFP4Linear(attn.q_proj, kernel)
            attn.o_proj = MetalFP4Linear(attn.o_proj, kernel)
            n += 2
    return n
