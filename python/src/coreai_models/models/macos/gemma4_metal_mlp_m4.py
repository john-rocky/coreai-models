# Community port — NOT an Apple model.
"""M=4 variants of the gemma4 transplant matvec kernels — the spec-decode VERIFY path.

Same weight layouts and torch_defns as the M=1 kernels (gemma4_metal_mlp{,_int2}.py);
the only change is 4 query rows per dispatch: x is loaded as float4 across the M
dimension and each row's accumulator is a float4, so the WEIGHTS ARE READ ONCE FOR
4 TOKENS — the whole point of verify (S=4 forward ~= one decode step in bytes).
R=2 (vs 4) keeps registers in budget with the 16-word x4 cache.

A [4, K] -> C [4, N] (MSL sees the torch shapes reversed: A[k, m], C[n, m]).
"""
from __future__ import annotations

import torch
from torch import nn

from coreai_torch import MetalParameter, TorchMetalKernel

from coreai_models.models.macos.gemma4_metal_mlp import (
    _INT4_G,
    _fused_int4_torch_defn,
)
from coreai_models.models.macos.gemma4_metal_mlp_int2 import (
    _fused_int2sym_torch_defn,
    _gateup_int2sym_torch_defn,
    _gateup_int4aff_torch_defn,
    int4sym_to_affine,
    qp2_from_packed,
)

_M4_R = 2
_M4_SGY = 8

_INT2SYM_M4_SRC = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    float4 acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = float4(0.0f);

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float4 x4[16];
        for (uint j = 0; j < 16; ++j)
            x4[j] = float4(float(A[k0 + j, 0]), float(A[k0 + j, 1]),
                           float(A[k0 + j, 2]), float(A[k0 + j, 3]));
        uint w0 = (kb >> 4) + lane;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = uint(QP[w0, n]);
            float4 s4 = float4(0.0f);
            for (uint j = 0; j < 16; ++j) {
                uint q = (packed >> (2 * j)) & 0x3;
                s4 += x4[j] * float(int(q ^ 2u) - 2);
            }
            acc[r] += s4;
        }
    }
    for (uint r = 0; r < R; ++r) {
        uint n = base_row + r;
        float sc = float(SC[n]);
        float t0 = simd_sum(acc[r].x);
        float t1 = simd_sum(acc[r].y);
        float t2 = simd_sum(acc[r].z);
        float t3 = simd_sum(acc[r].w);
        if (lane == 0) {
            C[n, 0] = TYPE(t0 * sc);
            C[n, 1] = TYPE(t1 * sc);
            C[n, 2] = TYPE(t2 * sc);
            C[n, 3] = TYPE(t3 * sc);
        }
    }
"""

_INT2SYM_GATEUP_M4_SRC = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    float4 accg[__R__], accu[__R__];
    for (uint r = 0; r < R; ++r) { accg[r] = float4(0.0f); accu[r] = float4(0.0f); }

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float4 x4[16];
        for (uint j = 0; j < 16; ++j)
            x4[j] = float4(float(A[k0 + j, 0]), float(A[k0 + j, 1]),
                           float(A[k0 + j, 2]), float(A[k0 + j, 3]));
        uint w0 = (kb >> 4) + lane;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = uint(QPG[w0, n]);
            uint pu = uint(QPU[w0, n]);
            float4 sg4 = float4(0.0f), su4 = float4(0.0f);
            for (uint j = 0; j < 16; ++j) {
                sg4 += x4[j] * float(int(((pg >> (2 * j)) & 0x3) ^ 2u) - 2);
                su4 += x4[j] * float(int(((pu >> (2 * j)) & 0x3) ^ 2u) - 2);
            }
            accg[r] += sg4;
            accu[r] += su4;
        }
    }
    for (uint r = 0; r < R; ++r) {
        uint n = base_row + r;
        float scg = float(SCG[n]);
        float scu = float(SCU[n]);
        for (uint m = 0; m < 4; ++m) {
            float tg = simd_sum(m == 0 ? accg[r].x : (m == 1 ? accg[r].y : (m == 2 ? accg[r].z : accg[r].w)));
            float tu = simd_sum(m == 0 ? accu[r].x : (m == 1 ? accu[r].y : (m == 2 ? accu[r].z : accu[r].w)));
            if (lane == 0) {
                float xg = tg * scg;
                float gel = 0.5f * xg * (1.0f + metal::precise::tanh(
                    0.7978845608028654f * (xg + 0.044715f * xg * xg * xg)));
                C[n, m] = TYPE(gel * (tu * scu));
            }
        }
    }
"""

_INT4AFF_M4_SRC = """
    const uint R = __R__, SGY = __SGY__, G = __G__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    float4 acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = float4(0.0f);

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float4 x4[8];
        for (uint j = 0; j < 8; ++j)
            x4[j] = float4(float(A[k0 + j, 0]), float(A[k0 + j, 1]),
                           float(A[k0 + j, 2]), float(A[k0 + j, 3]));
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / G;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = uint(QP[w0, n]);
            float sc = float(SC[grp, n]);
            float bi = float(BI[grp, n]);
            float4 s4 = float4(0.0f);
            for (uint j = 0; j < 8; ++j) {
                uint q = (packed >> (j * 4)) & 0xf;
                s4 += x4[j] * (sc * float(q) + bi);
            }
            acc[r] += s4;
        }
    }
    for (uint r = 0; r < R; ++r) {
        uint n = base_row + r;
        float t0 = simd_sum(acc[r].x);
        float t1 = simd_sum(acc[r].y);
        float t2 = simd_sum(acc[r].z);
        float t3 = simd_sum(acc[r].w);
        if (lane == 0) {
            C[n, 0] = TYPE(t0);
            C[n, 1] = TYPE(t1);
            C[n, 2] = TYPE(t2);
            C[n, 3] = TYPE(t3);
        }
    }
"""

_INT4AFF_GATEUP_M4_SRC = """
    const uint R = __R__, SGY = __SGY__, G = __G__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    float4 accg[__R__], accu[__R__];
    for (uint r = 0; r < R; ++r) { accg[r] = float4(0.0f); accu[r] = float4(0.0f); }

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float4 x4[8];
        for (uint j = 0; j < 8; ++j)
            x4[j] = float4(float(A[k0 + j, 0]), float(A[k0 + j, 1]),
                           float(A[k0 + j, 2]), float(A[k0 + j, 3]));
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / G;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = uint(QPG[w0, n]);
            uint pu = uint(QPU[w0, n]);
            float scg = float(SCG[grp, n]), big = float(BIG[grp, n]);
            float scu = float(SCU[grp, n]), biu = float(BIU[grp, n]);
            float4 sg4 = float4(0.0f), su4 = float4(0.0f);
            for (uint j = 0; j < 8; ++j) {
                float4 x = x4[j];
                sg4 += x * (scg * float((pg >> (j * 4)) & 0xf) + big);
                su4 += x * (scu * float((pu >> (j * 4)) & 0xf) + biu);
            }
            accg[r] += sg4;
            accu[r] += su4;
        }
    }
    for (uint r = 0; r < R; ++r) {
        uint n = base_row + r;
        for (uint m = 0; m < 4; ++m) {
            float tg = simd_sum(m == 0 ? accg[r].x : (m == 1 ? accg[r].y : (m == 2 ? accg[r].z : accg[r].w)));
            float tu = simd_sum(m == 0 ? accu[r].x : (m == 1 ? accu[r].y : (m == 2 ? accu[r].z : accu[r].w)));
            if (lane == 0) {
                float gel = 0.5f * tg * (1.0f + metal::precise::tanh(
                    0.7978845608028654f * (tg + 0.044715f * tg * tg * tg)));
                C[n, m] = TYPE(gel * tu);
            }
        }
    }
"""

_PARAMS = [MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
           MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")]


def _tmpl(src: str, group_size: int | None = None) -> str:
    src = src.replace("__R__", str(_M4_R)).replace("__SGY__", str(_M4_SGY))
    if group_size is not None:
        src = src.replace("__G__", str(group_size))
    return src


def build_int2sym_m4_kernel(name: str = "gemma4_m4_int2sym") -> TorchMetalKernel:
    return TorchMetalKernel(
        name, input_names=["A", "QP", "SC"], result_names=["C"],
        src=_tmpl(_INT2SYM_M4_SRC), torch_defn=_fused_int2sym_torch_defn,
        metal_params=_PARAMS, template_dtypes={"A": "TYPE"})


def build_gateup_int2sym_m4_kernel(name: str = "gemma4_m4_gateup_int2sym") -> TorchMetalKernel:
    return TorchMetalKernel(
        name, input_names=["A", "QPG", "SCG", "QPU", "SCU"], result_names=["C"],
        src=_tmpl(_INT2SYM_GATEUP_M4_SRC), torch_defn=_gateup_int2sym_torch_defn,
        metal_params=_PARAMS, template_dtypes={"A": "TYPE"})


def build_int4aff_m4_kernel(name: str = "gemma4_m4_int4aff",
                            group_size: int = _INT4_G) -> TorchMetalKernel:
    return TorchMetalKernel(
        name, input_names=["A", "QP", "SC", "BI"], result_names=["C"],
        src=_tmpl(_INT4AFF_M4_SRC, group_size), torch_defn=_fused_int4_torch_defn,
        metal_params=_PARAMS, template_dtypes={"A": "TYPE"})


def build_gateup_int4aff_m4_kernel(name: str = "gemma4_m4_gateup_int4aff",
                                   group_size: int = _INT4_G) -> TorchMetalKernel:
    return TorchMetalKernel(
        name, input_names=["A", "QPG", "SCG", "BIG", "QPU", "SCU", "BIU"], result_names=["C"],
        src=_tmpl(_INT4AFF_GATEUP_M4_SRC, group_size), torch_defn=_gateup_int4aff_torch_defn,
        metal_params=_PARAMS, template_dtypes={"A": "TYPE"})


def _grid(n: int) -> dict:
    return dict(threads_per_grid=(32, n // _M4_R, 1),
                threads_per_thread_group=(32, _M4_SGY, 1))


class M4Int2SymLinear(nn.Module):
    """[4, K] x packed-INT2 [N, K] -> [4, N] (per-row symmetric scale)."""

    def __init__(self, packed_u8: torch.Tensor, scale: torch.Tensor, rows: int, cols: int,
                 kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.kernel = kernel
        self.N = rows
        self.register_buffer("qp", qp2_from_packed(packed_u8, rows, cols))
        self.register_buffer("sc", scale.detach().float().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, k = x.shape
        y = self.kernel(x.reshape(s, k), self.qp, self.sc,
                        result_shapes=[[s, self.N]], **_grid(self.N))
        return y.reshape(b, s, self.N)


class M4Int4AffLinear(nn.Module):
    def __init__(self, packed_u8: torch.Tensor, scale: torch.Tensor, rows: int, cols: int,
                 kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.kernel = kernel
        self.N = rows
        qp, sc, bi = int4sym_to_affine(packed_u8, scale, rows, cols)
        self.register_buffer("qp", qp)
        self.register_buffer("sc", sc)
        self.register_buffer("bi", bi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, k = x.shape
        y = self.kernel(x.reshape(s, k), self.qp, self.sc, self.bi,
                        result_shapes=[[s, self.N]], **_grid(self.N))
        return y.reshape(b, s, self.N)


class M4Int2SymMLPFused(nn.Module):
    def __init__(self, packed: dict, gateup_kernel: TorchMetalKernel,
                 matvec_kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.gateup_kernel = gateup_kernel
        self.matvec_kernel = matvec_kernel
        for name in ("gate", "up", "down"):
            packed_u8, scale, rows, cols = packed[name]
            self.register_buffer(f"{name}_qp", qp2_from_packed(packed_u8, rows, cols))
            self.register_buffer(f"{name}_sc", scale.detach().float().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        xr = x.reshape(s, d)
        n = self.gate_qp.shape[0]
        h = self.gateup_kernel(xr, self.gate_qp, self.gate_sc, self.up_qp, self.up_sc,
                               result_shapes=[[s, n]], **_grid(n))
        nd = self.down_qp.shape[0]
        y = self.matvec_kernel(h, self.down_qp, self.down_sc,
                               result_shapes=[[s, nd]], **_grid(nd))
        return y.reshape(b, s, d)


class M4Int4AffMLPFused(nn.Module):
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
        n = self.gate_qp.shape[0]
        h = self.gateup_kernel(xr, self.gate_qp, self.gate_sc, self.gate_bi,
                               self.up_qp, self.up_sc, self.up_bi,
                               result_shapes=[[s, n]], **_grid(n))
        nd = self.down_qp.shape[0]
        y = self.matvec_kernel(h, self.down_qp, self.down_sc, self.down_bi,
                               result_shapes=[[s, nd]], **_grid(nd))
        return y.reshape(b, s, d)
