# Community port — NOT an Apple model.
"""INT2 pure-symmetric fused decode matvec + mixed-bit TRANSPLANT modules (Gemma-4 E2B mobile QAT).

Companion to ``gemma4_metal_mlp.py`` (int8/int4km/affine kernels — imported, not modified) and
``gemma4_metal_mlp_fp4.py``. Google's mobile QAT recipe for Gemma-4 E2B (extracted bit-exact from
the ``.litertlm`` release, see coreai-models-community/knowledge/gemma4-mixedbit-qat-transplant.md)
is per-channel PURE SYMMETRIC linear everywhere — all zero_points are 0, so

    dequant = code * scale[row]        (scale constant along the reduction axis K)

This file adds the ONE genuinely new kernel that recipe needs — an INT2 symmetric matvec — plus
the transplant module classes that carry PRE-QUANTIZED codes (no quantization at construction;
the packed extract tensors go straight into the graph):

  * INT2 (FFN layers 15-34 at 2x width, and the tied 262144-row lm_head): ``_INT2SYM_SRC``.
    16 codes per uint32; sign decode is branchless arithmetic ``(q ^ 2) - 2`` (no LUT, no
    tg-memory staging needed — the fp4 stack-spill lesson does not apply because nothing is
    dynamically indexed); because the scale is per-OUTPUT-ROW the inner loop accumulates the
    raw integer-weighted sum and multiplies by ``scale[n]`` ONCE after the simd reduction.
  * INT4 (FFN layers 0-14, attn q/o all layers): the EXISTING affine kernel
    (``build_fused_int4_kernel``, w = sc*q + bi with q in [0,15]) hosts the symmetric codes
    exactly via  q_u = q_s + 8,  sc[n,g] = scale[n],  bi[n,g] = -8*scale[n].
    ``-8*scale`` is exact in fp16 (power-of-two multiple), so the only deviation from the
    extracted fp32 scales is the fp16 rounding of the scale itself (~2^-11 relative), below
    the bf16 oracle that already gated the transplant.

Packing convention (verified vs the MLX dequant oracle in the extraction repo): INT2 = 4 codes
per byte, first code in bits[1:0], signed two's complement (-2..1); INT4 = 2 codes per byte,
low nibble first, signed (-8..7). A flat little-endian byte stream viewed as uint32 therefore
has code j of a 16-code word at bits [2j+1:2j] — the raw extract bytes ARE the kernel layout
(``packed_u8.view(torch.uint32)``, zero repacking for INT2).

Shape constraints: INT2 K % 512 == 0 (32 lanes x 16 codes) — gemma4 K = 1536 / 12288 qualify;
INT4 K % 256 == 0 (existing kernel); both N % 32 == 0 (multi-row tiling R*SGY).
"""
from __future__ import annotations

import torch
from torch import nn

from coreai_models.models.macos.gemma4_metal_mlp import (
    MetalParameter,
    TorchMetalKernel,
    _INT4_G,
    _V2_R,
    _V2_SGY,
    build_fused_int4_kernel,
    fused_int4_call,
    pack_idx_nib_u32,
)

# MSL: multi-row (R) + simd_sum tiling as the int4km/fp4 matvecs, but 16 codes/uint32 and NO
# dequant in the hot loop — the per-row scale multiplies the reduced sum once at the end.
# QP torch [N, K/16] uint32 (16 codes/word); SC torch [N] fp32; QP[w,n]=torch qp[n,w].
_INT2SYM_SRC = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 512) {          // block = 32 lanes * 16 codes = 512 K
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) xr[j] = float(A[k0 + j, 0]);
        uint w0 = (kb >> 4) + lane;                 // one uint32 word (16 codes) per lane
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = uint(QP[w0, n]);
            float s = 0.0f;
            for (uint j = 0; j < 16; ++j) {
                uint q = (packed >> (2 * j)) & 0x3;
                s += xr[j] * float(int(q ^ 2u) - 2);   // branchless 2-bit two's complement
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) {
            uint n = base_row + r;
            C[n, 0] = TYPE(tot * float(SC[n]));     // per-row symmetric scale, applied ONCE
        }
    }
"""


# ---- packed-code helpers (extract layout -> kernel layout) ---------------------------------------
def unpack_int2_codes(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Flat/2-D uint8 packed INT2 -> signed codes [rows, cols] int8 (bits[1:0] first)."""
    p = packed_u8.reshape(rows, cols // 4).to(torch.int16)
    c = torch.stack([(p >> s) & 3 for s in (0, 2, 4, 6)], dim=-1).reshape(rows, cols)
    return torch.where(c >= 2, c - 4, c).to(torch.int8)


def unpack_int4_codes(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Flat/2-D uint8 packed INT4 -> signed codes [rows, cols] int8 (low nibble first)."""
    p = packed_u8.reshape(rows, cols // 2).to(torch.int16)
    c = torch.stack([p & 0xF, p >> 4], dim=-1).reshape(rows, cols)
    return torch.where(c >= 8, c - 16, c).to(torch.int8)


def qp2_from_packed(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Raw extract INT2 bytes -> the kernel's uint32 word layout [rows, cols/16] (pure view)."""
    if cols % 512:
        raise ValueError(f"K={cols} not divisible by 512 (32 lanes x 16 codes)")
    return packed_u8.reshape(rows, cols // 4).contiguous().view(torch.uint32)


def _unpack_int2_u32(qp: torch.Tensor) -> torch.Tensor:
    """[N, K/16] uint32 -> [N, K] float signed codes (reference-side inverse of the kernel read)."""
    N, K16 = qp.shape
    p = qp.to(torch.int64)
    c = torch.stack([(p >> (2 * j)) & 0x3 for j in range(16)], dim=-1).reshape(N, K16 * 16)
    return torch.where(c >= 2, c - 4, c).float()


def _fused_int2sym_torch_defn(x: torch.Tensor, qp: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    """Reference: unpack 16 codes/word, per-row symmetric dequant, linear. C[1,N] = x @ W.T."""
    w = _unpack_int2_u32(qp) * sc.float().unsqueeze(1)
    return torch.nn.functional.linear(x, w.to(x.dtype))


def build_fused_int2sym_kernel(name: str = "gemma4_ffn_fused_int2sym") -> TorchMetalKernel:
    """INT2 pure-symmetric fused matvec (16 codes/uint32, per-row fp32 scale after simd_sum)."""
    return TorchMetalKernel(
        name, input_names=["A", "QP", "SC"], result_names=["C"],
        src=_INT2SYM_SRC.replace("__R__", str(_V2_R)).replace("__SGY__", str(_V2_SGY)),
        torch_defn=_fused_int2sym_torch_defn,
        metal_params=[MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def fused_int2sym_call(kernel: TorchMetalKernel, x_row: torch.Tensor, qp: torch.Tensor,
                       sc: torch.Tensor) -> torch.Tensor:
    """Dispatch int2sym (qp [N,K/16] uint32, sc [N] fp32) -> [1, N]."""
    N = qp.shape[0]
    return kernel(x_row, qp, sc, threads_per_grid=(32, N // _V2_R, 1),
                  threads_per_thread_group=(32, _V2_SGY, 1), result_shapes=[[1, N]])


class MetalInt2SymLinear(nn.Module):
    """``nn.Linear`` stand-in (q=1 decode) carrying PRE-PACKED INT2 symmetric codes.

    Constructed from the transplant tensors directly: ``packed_u8`` in the extract's byte
    layout (any shape, ``rows*cols/4`` bytes) + ``scale [N] fp32``. Used for the tied lm_head
    (N=262144) and reusable for any dense INT2 matvec. Share ONE kernel across all call sites.
    """

    def __init__(self, packed_u8: torch.Tensor, scale: torch.Tensor, rows: int, cols: int,
                 kernel: TorchMetalKernel) -> None:
        super().__init__()
        if rows % 32:
            raise ValueError(f"N={rows} not divisible by 32 (multi-row tiling R*SGY)")
        self.kernel = kernel
        self.N = rows
        self.register_buffer("qp", qp2_from_packed(packed_u8, rows, cols))  # [N, K/16] uint32
        self.register_buffer("sc", scale.detach().float().contiguous())     # [N] fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, k = x.shape  # decode: b=1, s=1
        y = fused_int2sym_call(self.kernel, x.reshape(s, k), self.qp, self.sc)
        return y.reshape(b, s, self.N)


class MetalInt2SymMLP(nn.Module):
    """Gemma-4 ``MLP`` stand-in on the INT2 symmetric kernel (the L15-34 double-wide FFN).

    ``packed`` is a dict with keys gate/up/down, each ``(packed_u8, scale, rows, cols)``.
    Same math as ``MLP.forward``: down(gelu_tanh(gate(x)) * up(x)).
    """

    def __init__(self, packed: dict, kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.kernel = kernel
        for name in ("gate", "up", "down"):
            packed_u8, scale, rows, cols = packed[name]
            if rows % 32:
                raise ValueError(f"{name}: N={rows} not divisible by 32")
            self.register_buffer(f"{name}_qp", qp2_from_packed(packed_u8, rows, cols))
            self.register_buffer(f"{name}_sc", scale.detach().float().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        xr = x.reshape(s, d)
        gate = nn.functional.gelu(fused_int2sym_call(self.kernel, xr, self.gate_qp, self.gate_sc),
                                  approximate="tanh")
        up = fused_int2sym_call(self.kernel, xr, self.up_qp, self.up_sc)
        out = fused_int2sym_call(self.kernel, gate * up, self.down_qp, self.down_sc)
        return out.reshape(b, s, d)


# ---- INT4 symmetric transplant riding the EXISTING affine kernel ---------------------------------
def int4sym_to_affine(packed_u8: torch.Tensor, scale: torch.Tensor, rows: int, cols: int,
                      group_size: int = _INT4_G
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-row INT4 codes -> the affine kernel's (qp, sc, bi) — EXACT.

    w = scale[n] * q_s  ==  sc*(q_s+8) + (-8*sc)  with q_u = q_s + 8 in [0,15].
    sc/bi are the per-row scale broadcast along the K groups ([N, K/G] fp16; -8*sc exact in fp16).
    """
    if cols % 256:
        raise ValueError(f"K={cols} not divisible by 256 (affine int4 kernel block)")
    q_u = (unpack_int4_codes(packed_u8, rows, cols).to(torch.int16) + 8).to(torch.uint8)
    qp = pack_idx_nib_u32(q_u)                                    # [N, K/8] uint32
    sc = scale.detach().float().to(torch.float16).reshape(rows, 1)
    sc = sc.expand(rows, cols // group_size).contiguous()         # [N, K/G] fp16
    bi = (sc * -8.0).contiguous()                                 # exact in fp16
    return qp, sc, bi


class MetalInt4AffLinear(nn.Module):
    """``nn.Linear`` stand-in carrying PRE-PACKED symmetric INT4 codes on the affine kernel."""

    def __init__(self, packed_u8: torch.Tensor, scale: torch.Tensor, rows: int, cols: int,
                 kernel: TorchMetalKernel) -> None:
        super().__init__()
        if rows % 32:
            raise ValueError(f"N={rows} not divisible by 32 (multi-row tiling R*SGY)")
        self.kernel = kernel
        self.N = rows
        qp, sc, bi = int4sym_to_affine(packed_u8, scale, rows, cols)
        self.register_buffer("qp", qp)  # [N, K/8] uint32
        self.register_buffer("sc", sc)  # [N, K/G] fp16
        self.register_buffer("bi", bi)  # [N, K/G] fp16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, k = x.shape  # decode: b=1, s=1
        y = fused_int4_call(self.kernel, x.reshape(s, k), self.qp, self.sc, self.bi)
        return y.reshape(b, s, self.N)


class MetalInt4AffMLP(nn.Module):
    """Gemma-4 ``MLP`` stand-in on the affine int4 kernel with transplanted symmetric codes."""

    def __init__(self, packed: dict, kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.kernel = kernel
        for name in ("gate", "up", "down"):
            packed_u8, scale, rows, cols = packed[name]
            qp, sc, bi = int4sym_to_affine(packed_u8, scale, rows, cols)
            self.register_buffer(f"{name}_qp", qp)
            self.register_buffer(f"{name}_sc", sc)
            self.register_buffer(f"{name}_bi", bi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        xr = x.reshape(s, d)
        gate = nn.functional.gelu(
            fused_int4_call(self.kernel, xr, self.gate_qp, self.gate_sc, self.gate_bi),
            approximate="tanh")
        up = fused_int4_call(self.kernel, xr, self.up_qp, self.up_sc, self.up_bi)
        out = fused_int4_call(self.kernel, gate * up, self.down_qp, self.down_sc, self.down_bi)
        return out.reshape(b, s, d)


# ---- FUSED gate+up+gelu+mul: the FFN first half in ONE dispatch ----------------------------------
# The un-closed dispatch lever (glue fusion measured a wash 2026-06-10 because MPSGraph already
# fuses elementwise chains; SDPA fold regressed on occupancy — but merging the two REAL gate/up
# matvecs keeps full occupancy while removing a dispatch + the gelu/mul elementwise group + two
# [1,N] intermediate round-trips, and reads the activation once instead of twice). Same tiling as
# the plain matvecs; each simd-group accumulates gate AND up dots for its R rows, then
# C[n] = gelu_tanh(dot_g*scg[n]) * (dot_u*scu[n]). fp32 gelu inlined in the epilogue (the DSL
# src is a kernel BODY — no helper functions), precise::tanh to match torch approximate="tanh".
_INT2SYM_GATEUP_SRC = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    float accg[__R__], accu[__R__];
    for (uint r = 0; r < R; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 512) {          // block = 32 lanes * 16 codes = 512 K
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) xr[j] = float(A[k0 + j, 0]);
        uint w0 = (kb >> 4) + lane;                 // one uint32 word (16 codes) per lane
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = uint(QPG[w0, n]);
            uint pu = uint(QPU[w0, n]);
            float sgm = 0.0f, sum = 0.0f;
            for (uint j = 0; j < 16; ++j) {
                float x = xr[j];
                sgm += x * float(int(((pg >> (2 * j)) & 0x3) ^ 2u) - 2);
                sum += x * float(int(((pu >> (2 * j)) & 0x3) ^ 2u) - 2);
            }
            accg[r] += sgm;
            accu[r] += sum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tg = simd_sum(accg[r]);
        float tu = simd_sum(accu[r]);
        if (lane == 0) {
            uint n = base_row + r;
            float xg = tg * float(SCG[n]);
            float gel = 0.5f * xg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (xg + 0.044715f * xg * xg * xg)));
            C[n, 0] = TYPE(gel * (tu * float(SCU[n])));
        }
    }
"""

_INT4AFF_GATEUP_SRC = """
    const uint R = __R__, SGY = __SGY__, G = __G__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    float accg[__R__], accu[__R__];
    for (uint r = 0; r < R; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 256) {          // block = 32 lanes * 8 codes = 256 K
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(A[k0 + j, 0]);
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / G;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = uint(QPG[w0, n]);
            uint pu = uint(QPU[w0, n]);
            float scg = float(SCG[grp, n]), big = float(BIG[grp, n]);
            float scu = float(SCU[grp, n]), biu = float(BIU[grp, n]);
            float sgm = 0.0f, sum = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                float x = xr[j];
                sgm += x * (scg * float((pg >> (j * 4)) & 0xf) + big);
                sum += x * (scu * float((pu >> (j * 4)) & 0xf) + biu);
            }
            accg[r] += sgm;
            accu[r] += sum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tg = simd_sum(accg[r]);
        float tu = simd_sum(accu[r]);
        if (lane == 0) {
            float gel = 0.5f * tg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (tg + 0.044715f * tg * tg * tg)));
            C[base_row + r, 0] = TYPE(gel * tu);
        }
    }
"""


def _gateup_int2sym_torch_defn(x: torch.Tensor, qpg: torch.Tensor, scg: torch.Tensor,
                               qpu: torch.Tensor, scu: torch.Tensor) -> torch.Tensor:
    """Reference: gelu_tanh(x @ Wg.T) * (x @ Wu.T) with per-row symmetric dequant."""
    g = torch.nn.functional.linear(x, (_unpack_int2_u32(qpg) * scg.float().unsqueeze(1)).to(x.dtype))
    u = torch.nn.functional.linear(x, (_unpack_int2_u32(qpu) * scu.float().unsqueeze(1)).to(x.dtype))
    return torch.nn.functional.gelu(g, approximate="tanh") * u


def _gateup_int4aff_torch_defn(x: torch.Tensor, qpg: torch.Tensor, scg: torch.Tensor,
                               big: torch.Tensor, qpu: torch.Tensor, scu: torch.Tensor,
                               biu: torch.Tensor) -> torch.Tensor:
    """Reference: gelu_tanh(affine-dequant gate matvec) * (affine-dequant up matvec)."""
    from coreai_models.models.macos.gemma4_metal_mlp import _fused_int4_torch_defn

    g = _fused_int4_torch_defn(x, qpg, scg, big)
    u = _fused_int4_torch_defn(x, qpu, scu, biu)
    return torch.nn.functional.gelu(g, approximate="tanh") * u


def build_gateup_int2sym_kernel(name: str = "gemma4_ffn_gateup_int2sym") -> TorchMetalKernel:
    """Fused gate+up+gelu+mul, INT2 symmetric (one dispatch for the FFN first half)."""
    return TorchMetalKernel(
        name, input_names=["A", "QPG", "SCG", "QPU", "SCU"], result_names=["C"],
        src=_INT2SYM_GATEUP_SRC.replace("__R__", str(_V2_R)).replace("__SGY__", str(_V2_SGY)),
        torch_defn=_gateup_int2sym_torch_defn,
        metal_params=[MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def build_gateup_int4aff_kernel(name: str = "gemma4_ffn_gateup_int4aff",
                                group_size: int = _INT4_G) -> TorchMetalKernel:
    """Fused gate+up+gelu+mul, INT4 affine (transplanted symmetric codes ride the same mapping)."""
    return TorchMetalKernel(
        name, input_names=["A", "QPG", "SCG", "BIG", "QPU", "SCU", "BIU"], result_names=["C"],
        src=(_INT4AFF_GATEUP_SRC.replace("__R__", str(_V2_R)).replace("__SGY__", str(_V2_SGY))
             .replace("__G__", str(group_size))),
        torch_defn=_gateup_int4aff_torch_defn,
        metal_params=[MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


class MetalInt2SymMLPFused(nn.Module):
    """INT2 FFN in TWO dispatches: fused gateup(+gelu+mul) kernel, then the down matvec."""

    def __init__(self, packed: dict, gateup_kernel: TorchMetalKernel,
                 matvec_kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.gateup_kernel = gateup_kernel
        self.matvec_kernel = matvec_kernel
        for name in ("gate", "up", "down"):
            packed_u8, scale, rows, cols = packed[name]
            if rows % 32:
                raise ValueError(f"{name}: N={rows} not divisible by 32")
            self.register_buffer(f"{name}_qp", qp2_from_packed(packed_u8, rows, cols))
            self.register_buffer(f"{name}_sc", scale.detach().float().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        xr = x.reshape(s, d)
        N = self.gate_qp.shape[0]
        h = self.gateup_kernel(
            xr, self.gate_qp, self.gate_sc, self.up_qp, self.up_sc,
            threads_per_grid=(32, N // _V2_R, 1),
            threads_per_thread_group=(32, _V2_SGY, 1), result_shapes=[[1, N]])
        out = fused_int2sym_call(self.matvec_kernel, h, self.down_qp, self.down_sc)
        return out.reshape(b, s, d)


class MetalInt4AffMLPFused(nn.Module):
    """INT4 FFN in TWO dispatches: fused gateup(+gelu+mul) kernel, then the down matvec."""

    def __init__(self, packed: dict, gateup_kernel: TorchMetalKernel,
                 matvec_kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.gateup_kernel = gateup_kernel
        self.matvec_kernel = matvec_kernel
        for name in ("gate", "up", "down"):
            packed_u8, scale, rows, cols = packed[name]
            qp, sc, bi = int4sym_to_affine(packed_u8, scale, rows, cols)
            self.register_buffer(f"{name}_qp", qp)
            self.register_buffer(f"{name}_sc", sc)
            self.register_buffer(f"{name}_bi", bi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        xr = x.reshape(s, d)
        N = self.gate_qp.shape[0]
        h = self.gateup_kernel(
            xr, self.gate_qp, self.gate_sc, self.gate_bi, self.up_qp, self.up_sc, self.up_bi,
            threads_per_grid=(32, N // _V2_R, 1),
            threads_per_thread_group=(32, _V2_SGY, 1), result_shapes=[[1, N]])
        out = fused_int4_call(self.matvec_kernel, h, self.down_qp, self.down_sc, self.down_bi)
        return out.reshape(b, s, d)


__all__ = [
    "MetalInt2SymLinear",
    "MetalInt2SymMLP",
    "MetalInt2SymMLPFused",
    "MetalInt4AffLinear",
    "MetalInt4AffMLP",
    "MetalInt4AffMLPFused",
    "build_fused_int2sym_kernel",
    "build_fused_int4_kernel",
    "build_gateup_int2sym_kernel",
    "build_gateup_int4aff_kernel",
    "fused_int2sym_call",
    "int4sym_to_affine",
    "qp2_from_packed",
    "unpack_int2_codes",
    "unpack_int4_codes",
]
