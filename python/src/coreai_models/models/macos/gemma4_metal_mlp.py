# Community port — NOT an Apple model.
"""Custom Metal kernel FFN for gemma4 E2B decode (Session B / macOS speed) — ADDITIVE.

The gemma4 decode core is FFN-bound at the MPSGraph floor (~28 ms/step core, 35 layers of int8
matmuls; see ondevice/_macos_speed_RESULTS.md). The lever to go faster WITHIN Core AI is a custom
Metal kernel for the hot path — Core AI's first-class `coreai_torch.TorchMetalKernel`, which traces
through torch.export as a custom op and embeds its MSL right in the .aimodel (no raw-Metal bypass;
WWDC26 session 325). This file is the FFN matvec kernel + a drop-in `MetalMLP`.

For q=1 decode the three FFN projections (gate/up: dim->hidden, down: hidden->dim) are matrix-VECTOR
products — memory-bandwidth-bound (read each weight once, little compute), exactly where a tuned kernel
+ narrow weights help most. Stage 1 here is an fp16 matvec to de-risk the integration and beat (or
match) MPSGraph's generic matmul; Stage 2 (separate) fuses the int8 k-means dequant into the matvec.

INDEXING CONVENTION (from coreai-torch/tests/dsl/conftest.py, verified numerically): the Metal DSL
exposes a tensor with REVERSED axis order vs torch — `get_extent(0)` is the innermost (last torch)
dim and a 2-D access `T[i, j]` reads torch element `T[j, i]`. So to get COALESCED weight reads (the
whole game for a bandwidth-bound matvec) we store each Linear weight TRANSPOSED as `Wt` (torch shape
[K, N], row-major): in-kernel `B[n, k]` then reads torch `Wt[k, n]` = address `k*N + n`, so adjacent
threads (n, n+1) touch adjacent addresses. That layout makes the op exactly `torch.matmul(x, Wt)`.

Boundary: purely additive. Reuses an existing `MLP`'s weights; does not modify gemma4_text.py /
gemma4_bucketed.py numerics. `metalize_mlps()` swaps the `.mlp` of each layer in a host-cache core.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

from coreai_models.models.macos.gemma4_text import MLP

# --- Kernel variant "scalar": one thread per output column n, serial reduce over K. -----------------
# Simple + coalesced ACROSS the simdgroup, but each output is a single thread with a long serial K-loop
# -> too few threads in flight to hide ~400-cycle memory latency (measured ~24 GB/s, ~10% of peak).
# Weight stored TRANSPOSED (Wt = W.T, torch [K, N]) so B[n,k]=torch Wt[k,n]=W[n,k] is coalesced over n.
_MATVEC_TG = 64
_SCALAR_SRC = """
    const uint K = A.get_extent(0);   // A is torch [1, K] -> innermost extent is K
    const uint N = B.get_extent(0);   // B is torch [K, N] -> innermost extent is N
    const uint n = gid.x;
    if (n >= N) return;               // bounds guard (grid padded to threadgroup multiple)

    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        sum += float(A[k, 0]) * float(B[n, k]);  // A[k,0]=x[k];  B[n,k]=Wt_torch[k,n]=W[n,k]
    }
    C[n, 0] = TYPE(sum);              // C is torch [1, N];  C[n,0]=torch[0,n]
"""

# --- Kernel variant "simd": one 32-lane SIMD-group per output, split-K + simd_sum. -----------------
# 32x more threads in flight -> hides memory latency; each y-row of the (32, SGY) threadgroup is exactly
# one SIMD-group, so simd_sum reduces the 32 lane partials. Weight stored UNtransposed (torch [N, K]):
# W[k,n]=torch W[n,k], and adjacent lanes (k, k+1) read adjacent torch addresses n*K+k -> coalesced.
# grid = (32, N, 1); threads_per_threadgroup = (32, SGY, 1). gid.x=lane(0..31), gid.y=output n.
_SIMD_SGY = 8
_SIMD_SRC = """
    const uint K = A.get_extent(0);   // A torch [1, K]
    const uint N = C.get_extent(0);   // C torch [1, N]
    const uint n = gid.y;             // one SIMD-group per output column
    if (n >= N) return;               // simd-uniform (all 32 lanes share gid.y)
    const uint lane = gid.x;          // 0..31 within the SIMD-group

    float partial = 0.0f;
    for (uint k = lane; k < K; k += 32) {
        partial += float(A[k, 0]) * float(W[k, n]);  // W[k,n]=torch W[n,k]; coalesced over lanes
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        C[n, 0] = TYPE(total);        // C torch [1, N]; C[n,0]=torch[0,n]
    }
"""


def _matmul_torch_defn(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """scalar: C[1,N] = x[1,K] @ Wt[K,N]."""
    return torch.matmul(x, b)


def _linear_torch_defn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """simd: C[1,N] = x[1,K] @ W[N,K].T  (== F.linear(x, W))."""
    return torch.nn.functional.linear(x, w)


def build_matvec_kernel(kind: str = "simd", name: str | None = None) -> TorchMetalKernel:
    """Shared FFN matvec kernel (one MSL, many call sites). ``kind`` in {"simd", "scalar"}."""
    if name is None:
        name = f"gemma4_ffn_matvec_{kind}"
    if kind == "scalar":
        src, defn, in_names = _SCALAR_SRC, _matmul_torch_defn, ["A", "B"]
    elif kind == "simd":
        src, defn, in_names = _SIMD_SRC, _linear_torch_defn, ["A", "W"]
    else:
        raise ValueError(f"unknown matvec kind {kind!r} (use 'simd' or 'scalar')")
    return TorchMetalKernel(
        name, input_names=in_names, result_names=["C"], src=src, torch_defn=defn,
        metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
        template_dtypes={in_names[0]: "TYPE"},
    )


class Matvec:
    """Bundles a matvec kernel with its weight layout + dispatch config (one object, all FFN calls)."""

    def __init__(self, kind: str = "simd", name: str | None = None) -> None:
        self.kind = kind
        self.kernel = build_matvec_kernel(kind, name)

    def store_weight(self, w: torch.Tensor) -> torch.Tensor:
        """w = nn.Linear weight [N, K] -> the layout this kernel reads."""
        return w.detach().t().contiguous() if self.kind == "scalar" else w.detach().contiguous()

    def __call__(self, x_row: torch.Tensor, stored_w: torch.Tensor) -> torch.Tensor:
        """x_row [1, K], stored_w (as from store_weight) -> [1, N]."""
        if self.kind == "scalar":
            N = stored_w.shape[1]  # [K, N]
            return self.kernel(x_row, stored_w, threads_per_grid=(N, 1, 1),
                               threads_per_thread_group=(_MATVEC_TG, 1, 1), result_shapes=[[1, N]])
        N = stored_w.shape[0]  # [N, K]
        return self.kernel(x_row, stored_w, threads_per_grid=(32, N, 1),
                           threads_per_thread_group=(32, _SIMD_SGY, 1), result_shapes=[[1, N]])


class MetalMLP(nn.Module):
    """Drop-in replacement for gemma4 ``MLP`` whose three projections call a Metal matvec kernel.

    Same math as ``MLP.forward`` (``down(gelu_tanh(gate(x)) * up(x))``). Decode-only (q=1 -> the
    activation is a single row [1, dim]; the kernel assumes M=1). Weight layout is chosen by ``mv``.
    """

    def __init__(self, mlp: MLP, mv: Matvec) -> None:
        super().__init__()
        self.mv = mv
        self.register_buffer("gate_w", mv.store_weight(mlp.gate_proj.weight))
        self.register_buffer("up_w", mv.store_weight(mlp.up_proj.weight))
        self.register_buffer("down_w", mv.store_weight(mlp.down_proj.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape  # decode: 1, 1, dim
        x_row = x.reshape(s, d)  # [1, dim]
        gate = nn.functional.gelu(self.mv(x_row, self.gate_w), approximate="tanh")
        up = self.mv(x_row, self.up_w)
        h = gate * up  # [1, hidden]
        out = self.mv(h, self.down_w)  # [1, dim]
        return out.reshape(b, s, d)


def metalize_mlps(core: nn.Module, kind: str = "simd", mv: Matvec | None = None) -> TorchMetalKernel:
    """Replace every decoder layer's ``.mlp`` (an ``MLP``) in a host-cache core with a ``MetalMLP``.

    Returns the shared kernel (register it with the converter before exporting). Additive: only the
    FFN projection matmuls change; attention / norms / PLE / KV plumbing are untouched.
    """
    if mv is None:
        mv = Matvec(kind)
    for layer in core.text_model.layers:
        if isinstance(layer.mlp, MLP):
            layer.mlp = MetalMLP(layer.mlp, mv)
    return mv.kernel


# =================================================================================================== #
# FUSED int8 k-means (palettization) + matvec — the memory-preserving decode win.
#
# int8 here is k-means PALETTIZATION (n_bits=8, per-grouped-channel axis=0 group_size=32): each group of
# 32 output rows shares a 256-entry fp16 codebook; the weight is stored as uint8 indices into it. MPSGraph
# dequantizes (LUT gather) then matmuls — measured SLOWER than fp16 on this Mac GPU (27 vs 18 ms): the
# dequant traffic exceeds the byte savings. A FUSED kernel folds the LUT gather into the matvec so it reads
# only the uint8 indices (HALF the fp16 bytes) + the tiny codebook — the real bandwidth win. Same coalesced
# simd-per-output structure as the fp16 "simd" kernel.
#
# CB layout note (DSL reverses axes): codebook stored torch [G, 256]; in-kernel CB[q, group] = torch
# cb[group, q]. IDX stored torch [N, K]; IDX[k, n] = torch idx[n, k] (coalesced over lanes at fixed k).
_FUSED_INT8_SRC = """
    const uint K = A.get_extent(0);    // A torch [1, K]
    const uint N = C.get_extent(0);    // C torch [1, N]
    const uint n = gid.y;              // one SIMD-group per output column
    if (n >= N) return;
    const uint lane = gid.x;           // 0..31
    const uint group = n >> 5;         // n / 32  (group_size = 32)

    float partial = 0.0f;
    for (uint k = lane; k < K; k += 32) {
        uint q = uint(IDX[k, n]);      // uint8 palette index;  IDX[k,n]=torch idx[n,k]
        float w = float(CB[q, group]); // codebook lookup;      CB[q,group]=torch cb[group,q]
        partial += float(A[k, 0]) * w;
    }
    float total = simd_sum(partial);
    if (lane == 0) {
        C[n, 0] = TYPE(total);
    }
"""

_PALETTE_GROUP = 32  # per-grouped-channel group size on the output axis
_PALETTE_BITS = 8


def palettize_grouped(weight: torch.Tensor, group_size: int = _PALETTE_GROUP, n_bits: int = _PALETTE_BITS,
                      iters: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D k-means palettization of an nn.Linear weight [N, K], grouped over the output axis.

    Each group of ``group_size`` rows shares one ``2**n_bits``-entry codebook (Lloyd's algorithm,
    vectorized across groups; centroids init at value quantiles, kept sorted so assignment is a batched
    ``searchsorted``). Returns ``(idx [N, K] uint8, codebook [G, 2**n_bits] fp16)`` with
    ``weight[n, k] ≈ codebook[n // group_size, idx[n, k]]`` — the exact format the fused kernel reads.
    """
    W = weight.detach().float()
    N, K = W.shape
    if N % group_size:
        raise ValueError(f"out dim {N} not divisible by group_size {group_size}")
    G, L = N // group_size, 1 << n_bits
    Wf = W.reshape(G, group_size * K)  # [G, M]; row g = the 32 rows of group g, flattened
    qs = torch.linspace(0.0, 1.0, L)
    c = torch.quantile(Wf, qs, dim=1).T.contiguous()  # [G, L] sorted centroids
    for _ in range(iters):
        mids = (c[:, 1:] + c[:, :-1]) * 0.5  # [G, L-1]
        a = torch.searchsorted(mids, Wf)  # [G, M] -> cluster in [0, L-1]
        sums = torch.zeros(G, L).scatter_add_(1, a, Wf)
        cnts = torch.zeros(G, L).scatter_add_(1, a, torch.ones_like(Wf))
        c = torch.where(cnts > 0, sums / cnts.clamp(min=1.0), c)
        c, _ = torch.sort(c, dim=1)
    mids = (c[:, 1:] + c[:, :-1]) * 0.5
    idx = torch.searchsorted(mids, Wf).to(torch.uint8).reshape(N, K).contiguous()
    return idx, c.to(torch.float16).contiguous()


def _fused_int8_torch_defn(x: torch.Tensor, idx: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """Reference (eager + fake/shape): dequant via the codebook, then linear. C[1,N] = x @ W.T."""
    n = idx.shape[0]
    grp = torch.arange(n, device=idx.device) // _PALETTE_GROUP
    w = cb[grp.unsqueeze(1), idx.long()]  # [N, K] dequantized weight
    return torch.nn.functional.linear(x, w.to(x.dtype))


def build_fused_int8_kernel(name: str = "gemma4_ffn_fused_int8") -> TorchMetalKernel:
    """Fused dequant-LUT + matvec kernel (uint8 indices + fp16 codebook -> fp16 out)."""
    return TorchMetalKernel(
        name, input_names=["A", "IDX", "CB"], result_names=["C"], src=_FUSED_INT8_SRC,
        torch_defn=_fused_int8_torch_defn,
        metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


# --- V2: MLX-qmv-style tuned int8 matvec (the FFN-tuning lever, ATTENTION_KERNEL_HANDOFF §A) -----------
# The v1 fused kernel is at 52% of the M4 Max's bandwidth — latency-bound (1 output/simd-group re-reads x
# from global every step; per-weight codebook gather hits global). MLX's `qmv_fast` recipe (mlx
# backend/metal/kernels/quantized.h) closes this: (1) each simd-group computes R OUTPUT ROWS, loading x ONCE
# into registers and reusing it across the R rows (amortizes the x traffic R×); (2) each lane processes V
# weight values per block (vectorized, more loads in flight to hide latency). Adapted to our k-means LUT:
# (3) a whole threadgroup owns exactly 32 rows = ONE palette group, so we stage that group's 256-entry
# codebook into threadgroup memory once, turning the per-weight global gather into a fast tg-mem read.
# Grid (32, N/R, 1), tg (32, SGY, 1) -> SGY*R outputs/tg; N divisible by R*SGY and by 32 (gemma4 6144/1536 ok).
_V2_R, _V2_SGY, _V2_V = 4, 8, 8  # rows/simd-group, simd-groups/threadgroup, values/lane/block
_FUSED_INT8_SRC_V2 = """
    const uint R = __R__, SGY = __SGY__, V = __V__;
    const uint K = A.get_extent(0);        // A torch [1, K]
    const uint lane = tid.x;               // 0..31
    const uint sg = tid.y;                 // 0..SGY-1
    const uint group = tgid.y;             // all 32 rows of this threadgroup share ONE palette group
    const uint base_row = (tgid.y * SGY + sg) * R;   // first of this simd-group's R output rows

    threadgroup float cbt[256];            // stage this group's codebook (256 entries) once
    uint t = sg * 32 + lane;               // 0..255
    cbt[t] = float(CB[t, group]);          // CB[q,group]=torch cb[group,q]
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 32 * V) {
        uint k0 = kb + lane * V;
        float xr[__V__];                   // load V x-values into registers, reuse across the R rows
        for (uint j = 0; j < V; ++j) xr[j] = float(A[k0 + j, 0]);
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            float s = 0.0f;
            for (uint j = 0; j < V; ++j) {
                uint q = uint(IDX[k0 + j, n]);   // IDX[k,n]=torch idx[n,k]; V consecutive, coalesced
                s += xr[j] * cbt[q];             // codebook from fast threadgroup memory
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r, 0] = TYPE(tot);
    }
"""


def build_fused_int8_kernel_v2(name: str = "gemma4_ffn_fused_int8_v2") -> TorchMetalKernel:
    """MLX-qmv-style tuned fused int8 matvec (R rows/simd-group, V/lane vectorized, codebook in tg mem)."""
    return TorchMetalKernel(
        name, input_names=["A", "IDX", "CB"], result_names=["C"],
        src=(_FUSED_INT8_SRC_V2.replace("__R__", str(_V2_R)).replace("__SGY__", str(_V2_SGY))
             .replace("__V__", str(_V2_V))),
        torch_defn=_fused_int8_torch_defn,
        metal_params=[MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def fused_int8_v2_call(kernel: TorchMetalKernel, x_row: torch.Tensor, idx: torch.Tensor,
                       cb: torch.Tensor) -> torch.Tensor:
    """Dispatch the v2 kernel for a [1,K] activation and palettized weight (idx [N,K], cb [G,256]) -> [1,N]."""
    N = idx.shape[0]
    return kernel(x_row, idx, cb, threads_per_grid=(32, N // _V2_R, 1),
                  threads_per_thread_group=(32, _V2_SGY, 1), result_shapes=[[1, N]])


# --- V3: v2 + uint32-PACKED index reads (wide loads). uint8 byte-loads cap the v1/v2 matvec at ~51% of peak;
# MLX reads weights as uint32. Pack 4 consecutive k-indices per uint32 so the index stream is read in 32-bit
# transactions (4 indices/load), then unpack with shifts + 4 codebook gathers. Same multi-row + tg-codebook
# structure as v2. IDXP torch [N, K/4] uint32; IDXP[w,n]=torch idxp[n,w].
def pack_idx_u32(idx: torch.Tensor) -> torch.Tensor:
    """[N, K] uint8 palette indices -> [N, K//4] uint32 (little-endian 4-byte pack)."""
    N, K = idx.shape
    if K % 4:
        raise ValueError(f"K={K} not divisible by 4")
    i = idx.to(torch.int64).reshape(N, K // 4, 4)
    packed = i[..., 0] | (i[..., 1] << 8) | (i[..., 2] << 16) | (i[..., 3] << 24)
    return packed.to(torch.uint32).contiguous()


_FUSED_INT8_SRC_V3 = """
    const uint R = __R__, SGY = __SGY__, V = __V__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint group = tgid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    threadgroup float cbt[256];
    uint t = sg * 32 + lane;
    cbt[t] = float(CB[t, group]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 32 * V) {
        uint k0 = kb + lane * V;
        float xr[__V__];
        for (uint j = 0; j < V; ++j) xr[j] = float(A[k0 + j, 0]);
        uint w0 = (kb >> 2) + lane * (V / 4);    // first uint32 word for this lane (V/4 words/lane)
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            float s = 0.0f;
            for (uint wi = 0; wi < V / 4; ++wi) {
                uint packed = uint(IDXP[w0 + wi, n]);    // IDXP[w,n]=torch idxp[n,w]; 4 indices/word
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
        if (lane == 0) C[base_row + r, 0] = TYPE(tot);
    }
"""


def _fused_int8_v3_torch_defn(x: torch.Tensor, idxp: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """Reference for the uint32-packed kernel: unpack 4 indices/word, codebook-dequant, linear."""
    N, K4 = idxp.shape
    p = idxp.to(torch.int64)
    idx = torch.stack([p & 0xFF, (p >> 8) & 0xFF, (p >> 16) & 0xFF, (p >> 24) & 0xFF], dim=-1).reshape(N, K4 * 4)
    grp = torch.arange(N, device=idxp.device) // _PALETTE_GROUP
    w = cb[grp.unsqueeze(1), idx]
    return torch.nn.functional.linear(x, w.to(x.dtype))


def build_fused_int8_kernel_v3(name: str = "gemma4_ffn_fused_int8_v3") -> TorchMetalKernel:
    """v2 + uint32-packed index reads (wide 32-bit loads of the palette-index stream)."""
    return TorchMetalKernel(
        name, input_names=["A", "IDXP", "CB"], result_names=["C"],
        src=(_FUSED_INT8_SRC_V3.replace("__R__", str(_V2_R)).replace("__SGY__", str(_V2_SGY))
             .replace("__V__", str(_V2_V))),
        torch_defn=_fused_int8_v3_torch_defn,
        metal_params=[MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def fused_int8_v3_call(kernel: TorchMetalKernel, x_row: torch.Tensor, idxp: torch.Tensor,
                       cb: torch.Tensor) -> torch.Tensor:
    """Dispatch v3 (idxp = uint32-packed indices [N, K/4]) -> [1, N]."""
    N = idxp.shape[0]
    return kernel(x_row, idxp, cb, threads_per_grid=(32, N // _V2_R, 1),
                  threads_per_thread_group=(32, _V2_SGY, 1), result_shapes=[[1, N]])


class MetalIntMLPv3(nn.Module):
    """``MLP`` replacement running the v3 (uint32-packed, MLX-qmv-style) fused int8 matvec — ~1.3-1.4× v1."""

    def __init__(self, mlp: MLP, kernel: TorchMetalKernel, iters: int = 10) -> None:
        super().__init__()
        self.kernel = kernel
        for name, lin in (("gate", mlp.gate_proj), ("up", mlp.up_proj), ("down", mlp.down_proj)):
            idx, cb = palettize_grouped(lin.weight, iters=iters)
            self.register_buffer(f"{name}_idxp", pack_idx_u32(idx))  # [N, K/4] uint32
            self.register_buffer(f"{name}_cb", cb)                    # [G, 256] fp16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        xr = x.reshape(s, d)
        gate = nn.functional.gelu(fused_int8_v3_call(self.kernel, xr, self.gate_idxp, self.gate_cb),
                                  approximate="tanh")
        up = fused_int8_v3_call(self.kernel, xr, self.up_idxp, self.up_cb)
        out = fused_int8_v3_call(self.kernel, gate * up, self.down_idxp, self.down_cb)
        return out.reshape(b, s, d)


def metalize_mlps_int8_v3(core: nn.Module, kernel: TorchMetalKernel | None = None,
                          iters: int = 10) -> TorchMetalKernel:
    """Swap every layer's ``.mlp`` for the v3 (uint32-packed) fused-int8 MLP. Returns the shared kernel."""
    if kernel is None:
        kernel = build_fused_int8_kernel_v3()
    for layer in core.text_model.layers:
        if isinstance(layer.mlp, MLP):
            layer.mlp = MetalIntMLPv3(layer.mlp, kernel, iters=iters)
    return kernel


# =================================================================================================== #
# INT4 AFFINE fused matvec — the LEVER the MLX-gap analysis pointed to (4-bit, HALF the int8 bytes + NO LUT).
#
# int8-v3 reads 1 byte/weight (uint8 palette index) + a 256-entry codebook GATHER. MLX is ~4-bit AFFINE: it
# stores w ≈ scale·q + bias with q in [0,15], grouped along the REDUCTION axis K (group_size G), so dequant is
# a single FMA (scale·q + bias) — HALF the index bytes AND no codebook gather. Affine-along-K (vs the int8
# kernel's k-means-along-output) is also what makes 4 bits ACCURATE: each G-wide K-segment gets its own
# scale/bias. Same multi-row (R) + tg structure as v3; V=8 values/lane == exactly ONE uint32 (8 nibbles), and
# G is a multiple of 8 so a lane's 8 weights share one (scale,bias). QP torch [N,K/8] uint32; SC/BI torch
# [N,K/G] fp16; QP[w,n]=torch qp[n,w], SC[g,n]=torch sc[n,g].
_INT4_G = 64  # affine group size along K (K=1536/6144 both divisible; 8 | G so a lane's 8 vals share a group)
_INT4_SRC = """
    const uint R = __R__, SGY = __SGY__, G = __G__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {          // block = 32 lanes * 8 vals = 256 K
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(A[k0 + j, 0]);
        uint w0 = (kb >> 3) + lane;                 // one uint32 word (8 nibbles) per lane
        uint grp = k0 / G;                          // affine group (constant for this lane's 8 vals)
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = uint(QP[w0, n]);
            float sc = float(SC[grp, n]);
            float bi = float(BI[grp, n]);
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uint q = (packed >> (j * 4)) & 0xf;
                s += xr[j] * (sc * float(q) + bi);  // affine dequant: scale*q + bias (no LUT gather)
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r, 0] = TYPE(tot);
    }
"""


# Clip ratios searched per group for the MSE-optimal 4-bit range. Plain min/max (c=1.0) is outlier-dominated
# (one big weight blows up the scale, coarsening the other 31/63) — at 16 levels that costs ~1 token (7/8);
# shrinking the range per group to minimize quant MSE recovers 8/8. (AWQ/GPTQ-lite clip search.)
_INT4_CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5)


def quantize_affine_int4(weight: torch.Tensor, group_size: int = _INT4_G
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Affine 4-bit group quantization of an nn.Linear weight [N, K], grouped along K (MSE-optimal clip).

    Per (row n, group g): ``w ≈ scale·q + bias`` with ``q in [0,15]``; the range [bias, bias+15·scale] is
    chosen per group to MINIMIZE the dequant MSE (search over :data:`_INT4_CLIPS` shrink ratios about the
    group mean) — plain min/max is outlier-dominated and costs ~1 token at 4-bit. Returns ``(qp [N, K//8]
    uint32 (8 nibbles/word), scale [N, K//G] fp16, bias [N, K//G] fp16)`` — the exact format the kernel reads."""
    W = weight.detach().float()
    N, K = W.shape
    if K % group_size:
        raise ValueError(f"K={K} not divisible by group_size {group_size}")
    if K % 8:
        raise ValueError(f"K={K} not divisible by 8 (uint32 nibble pack)")
    Wg = W.reshape(N, K // group_size, group_size)
    mid = (Wg.amax(-1) + Wg.amin(-1)) * 0.5                       # [N, NG]
    half = (Wg.amax(-1) - Wg.amin(-1)) * 0.5
    best_err = best_scale = best_lo = None
    for c in _INT4_CLIPS:
        lo = mid - half * c
        scale = (2.0 * half * c / 15.0).clamp(min=1e-8)
        q = torch.round((Wg - lo.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(0, 15)
        err = ((scale.unsqueeze(-1) * q + lo.unsqueeze(-1)) - Wg).square().sum(-1)  # [N, NG]
        if best_err is None:
            best_err, best_scale, best_lo = err, scale, lo
        else:
            take = err < best_err
            best_err = torch.where(take, err, best_err)
            best_scale = torch.where(take, scale, best_scale)
            best_lo = torch.where(take, lo, best_lo)
    scale, lo = best_scale, best_lo
    q = torch.round((Wg - lo.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp(0, 15).to(torch.int64)
    q = q.reshape(N, K // 8, 8)
    qp = torch.zeros(N, K // 8, dtype=torch.int64)
    for j in range(8):
        qp |= q[..., j] << (j * 4)
    return (qp.to(torch.uint32).contiguous(), scale.to(torch.float16).contiguous(),
            lo.to(torch.float16).contiguous())


def _fused_int4_torch_defn(x: torch.Tensor, qp: torch.Tensor, sc: torch.Tensor,
                           bi: torch.Tensor) -> torch.Tensor:
    """Reference: unpack 8 nibbles/word, affine-dequant (scale·q+bias), linear. C[1,N] = x @ W.T."""
    N, K8 = qp.shape
    K = K8 * 8
    G = K // sc.shape[1]
    p = qp.to(torch.int64)
    q = torch.stack([(p >> (j * 4)) & 0xF for j in range(8)], dim=-1).reshape(N, K)
    scale = sc.float().repeat_interleave(G, dim=1)
    bias = bi.float().repeat_interleave(G, dim=1)
    w = scale * q.float() + bias
    return torch.nn.functional.linear(x, w.to(x.dtype))


def build_fused_int4_kernel(name: str = "gemma4_ffn_fused_int4",
                            group_size: int = _INT4_G) -> TorchMetalKernel:
    """Affine 4-bit fused matvec (8 nibbles/uint32, scale·q+bias dequant, multi-row + simd_sum).

    ``group_size`` MUST match the quantization group_size (it indexes scale/bias along K); 8 | group_size."""
    return TorchMetalKernel(
        name, input_names=["A", "QP", "SC", "BI"], result_names=["C"],
        src=(_INT4_SRC.replace("__R__", str(_V2_R)).replace("__SGY__", str(_V2_SGY))
             .replace("__G__", str(group_size))),
        torch_defn=_fused_int4_torch_defn,
        metal_params=[MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def fused_int4_call(kernel: TorchMetalKernel, x_row: torch.Tensor, qp: torch.Tensor,
                    sc: torch.Tensor, bi: torch.Tensor) -> torch.Tensor:
    """Dispatch int4 (qp [N,K/8] uint32, sc/bi [N,K/G] fp16) -> [1, N]."""
    N = qp.shape[0]
    return kernel(x_row, qp, sc, bi, threads_per_grid=(32, N // _V2_R, 1),
                  threads_per_thread_group=(32, _V2_SGY, 1), result_shapes=[[1, N]])


class MetalInt4MLP(nn.Module):
    """``MLP`` replacement running the affine-4-bit fused matvec (half the int8 bytes, no LUT gather)."""

    def __init__(self, mlp: MLP, kernel: TorchMetalKernel, group_size: int = _INT4_G) -> None:
        super().__init__()
        self.kernel = kernel
        for name, lin in (("gate", mlp.gate_proj), ("up", mlp.up_proj), ("down", mlp.down_proj)):
            qp, sc, bi = quantize_affine_int4(lin.weight, group_size=group_size)
            self.register_buffer(f"{name}_qp", qp)   # [N, K/8] uint32
            self.register_buffer(f"{name}_sc", sc)   # [N, K/G] fp16
            self.register_buffer(f"{name}_bi", bi)   # [N, K/G] fp16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        xr = x.reshape(s, d)
        gate = nn.functional.gelu(fused_int4_call(self.kernel, xr, self.gate_qp, self.gate_sc, self.gate_bi),
                                  approximate="tanh")
        up = fused_int4_call(self.kernel, xr, self.up_qp, self.up_sc, self.up_bi)
        out = fused_int4_call(self.kernel, gate * up, self.down_qp, self.down_sc, self.down_bi)
        return out.reshape(b, s, d)


def metalize_mlps_int4(core: nn.Module, kernel: TorchMetalKernel | None = None,
                       group_size: int = _INT4_G) -> TorchMetalKernel:
    """Swap every layer's ``.mlp`` for the affine-4-bit ``MetalInt4MLP``. Returns the shared kernel.

    The kernel is built with the SAME ``group_size`` as the quantization (it must, to index scale/bias)."""
    if kernel is None:
        kernel = build_fused_int4_kernel(group_size=group_size)
    for layer in core.text_model.layers:
        if isinstance(layer.mlp, MLP):
            layer.mlp = MetalInt4MLP(layer.mlp, kernel, group_size=group_size)
    return kernel


class MetalIntMLP(nn.Module):
    """``MLP`` replacement whose projections run a FUSED int8-palettized matvec kernel (decode q=1).

    Each weight is k-means palettized at construction (uint8 indices + fp16 codebook stored as buffers);
    the kernel dequantizes inline. Half the weight bytes of fp16, no separate dequant pass.
    """

    def __init__(self, mlp: MLP, kernel: TorchMetalKernel, iters: int = 10) -> None:
        super().__init__()
        self.kernel = kernel
        for name, lin in (("gate", mlp.gate_proj), ("up", mlp.up_proj), ("down", mlp.down_proj)):
            idx, cb = palettize_grouped(lin.weight, iters=iters)
            self.register_buffer(f"{name}_idx", idx)  # [N, K] uint8
            self.register_buffer(f"{name}_cb", cb)     # [G, 256] fp16

    def _fused(self, x_row: torch.Tensor, idx: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        N = idx.shape[0]
        return self.kernel(x_row, idx, cb, threads_per_grid=(32, N, 1),
                           threads_per_thread_group=(32, _SIMD_SGY, 1), result_shapes=[[1, N]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        x_row = x.reshape(s, d)
        gate = nn.functional.gelu(self._fused(x_row, self.gate_idx, self.gate_cb), approximate="tanh")
        up = self._fused(x_row, self.up_idx, self.up_cb)
        h = gate * up
        out = self._fused(h, self.down_idx, self.down_cb)
        return out.reshape(b, s, d)


def metalize_mlps_int8(core: nn.Module, kernel: TorchMetalKernel | None = None,
                       iters: int = 10) -> TorchMetalKernel:
    """Swap every layer's ``.mlp`` for a fused-int8 ``MetalIntMLP``. Returns the shared kernel."""
    if kernel is None:
        kernel = build_fused_int8_kernel()
    for layer in core.text_model.layers:
        if isinstance(layer.mlp, MLP):
            layer.mlp = MetalIntMLP(layer.mlp, kernel, iters=iters)
    return kernel


class MetalIntLinear(nn.Module):
    """Drop-in for an ``nn.Linear`` (q=1 decode) whose matvec runs the FUSED int8 kernel.

    Mirrors ``nn.Linear.forward`` (``y = x @ W.T``) for a single-row activation [b, 1, K] -> [b, 1, N],
    reusing the SAME ``build_fused_int8_kernel`` matvec as the FFN (it is a generic matvec). Used to
    int8-kernelize attention's q/k/v/o projections — the only attention weights that are plain matvecs;
    q_norm/k_norm/RoPE/SDPA/KV plumbing are untouched. ``N`` (out features) must be divisible by the
    palette ``group_size`` (32); gemma4's q=2048/4096, k/v=256/512, o=1536 all qualify.
    """

    def __init__(self, lin: nn.Linear, kernel: TorchMetalKernel, iters: int = 10) -> None:
        super().__init__()
        self.kernel = kernel
        self.N = int(lin.weight.shape[0])
        idx, cb = palettize_grouped(lin.weight, iters=iters)
        self.register_buffer("idx", idx)  # [N, K] uint8
        self.register_buffer("cb", cb)     # [G, 256] fp16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, k = x.shape  # decode: b=1, s=1
        y = self.kernel(x.reshape(s, k), self.idx, self.cb, threads_per_grid=(32, self.N, 1),
                        threads_per_thread_group=(32, _SIMD_SGY, 1), result_shapes=[[1, self.N]])
        return y.reshape(b, s, self.N)


def metalize_attn_int8(core: nn.Module, kernel: TorchMetalKernel | None = None, iters: int = 10,
                       projections: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
                       ) -> TorchMetalKernel:
    """Swap selected attention projections for fused-int8 ``MetalIntLinear``.

    Pass the SAME kernel returned by ``metalize_mlps_int8`` to share one MSL across FFN + attention.
    Additive: only the named projection matmuls change. Shrinks those fp16 attention weights to int8
    (toward the iOS memory budget). NOTE the small k/v projections (N=256/512) are LESS GPU-efficient as
    a custom matvec than fp16-MPSGraph, so ``("q_proj", "o_proj")`` (the big projections) is the
    speed-vs-memory sweet spot; all four maximizes the memory saving at a small decode-speed cost.
    """
    if kernel is None:
        kernel = build_fused_int8_kernel()
    for layer in core.text_model.layers:
        attn = layer.self_attn
        for name in projections:
            lin = getattr(attn, name)
            if isinstance(lin, nn.Linear):
                setattr(attn, name, MetalIntLinear(lin, kernel, iters=iters))
    return kernel


# =================================================================================================== #
# FUSED int8 head + two-level argmax — Lever #3 (kill the CPU head; ondevice/HEAD_KERNEL_HANDOFF.md).
#
# The tied LM head is the SAME fused-int8 matvec as the FFN, just N = vocab_size (262144): one SIMD-group
# per logit (int8 dequant-LUT, fp32 accumulate). Two prior dead-ends shelved a GPU head — returning the
# full 262144 logits was SLOWER than the CPU head (1 MB readback/step), and an in-graph MPSGraph argmax
# failed to lower (CoreAIError 3). A custom kernel sidesteps BOTH: it does the matvec AND a partial argmax
# on the GPU and returns only num_tg = N/SGY partials (NOT 262144 logits, NOT an MPSGraph argmax). The host
# reduces the partials to the token id (reading ~32k floats is <0.05 ms vs 1 MB of logits).
#
# Two-level argmax: each threadgroup owns SGY consecutive logits (one SIMD-group each). After the matvec,
# the SGY SIMD-group leaders write (logit, n) to threadgroup memory; one thread reduces them to this
# threadgroup's (max value, GLOBAL token index) and writes ONE partial. Grid (32, N, 1), tg (32, SGY, 1);
# N is a power of two (262144) so N % SGY == 0 and no thread takes an early return before the barrier.
# Partial VALUES are fp32 (not fp16) so the host's cross-threadgroup argmax is exact to the matvec — a
# truncated max could flip a near-tie, i.e. emit the wrong token. GREEDY only (argmax); sampling keeps the
# plain Gemma4Head. Strict '>' in both reduction levels keeps the LOWEST index on ties == torch.argmax.
_HEAD_SGY = 8  # SIMD-groups per threadgroup == logits reduced per partial (tg = 32*SGY threads)
_HEAD_ARGMAX_SRC = """
    const uint SGY = __SGY__;          // SIMD-groups per threadgroup (compile constant)
    const uint K = A.get_extent(0);    // A torch [1, K]
    const uint lane = tid.x;           // 0..31 within the SIMD-group
    const uint sg = tid.y;             // 0..SGY-1 : which SIMD-group in this threadgroup
    const uint n = gid.y;              // global output column == candidate token id
    const uint group = n >> 5;         // palette group (group_size = 32)

    float partial = 0.0f;
    for (uint k = lane; k < K; k += 32) {
        uint q = uint(IDX[k, n]);      // uint8 palette index;  IDX[k,n]=torch idx[n,k]
        float w = float(CB[q, group]); // codebook lookup;      CB[q,group]=torch cb[group,q]
        partial += float(A[k, 0]) * w;
    }
    float total = simd_sum(partial);   // logit[n]  (simd_sum broadcasts to all lanes)

    threadgroup float sval[__SGY__];
    threadgroup uint  sidx[__SGY__];
    if (lane == 0) { sval[sg] = total; sidx[sg] = n; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0 && sg == 0) {        // one thread reduces this threadgroup's SGY logits
        float best = sval[0];
        uint  bidx = sidx[0];
        for (uint j = 1; j < SGY; ++j) {
            if (sval[j] > best) { best = sval[j]; bidx = sidx[j]; }
        }
        PV[tgid.y] = best;             // fp32 partial max value (one per threadgroup)
        PI[tgid.y] = int(bidx);        // its GLOBAL token index
    }
"""


def _head_argmax_torch_defn(x: torch.Tensor, idx: torch.Tensor,
                            cb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference + fake: per-threadgroup (max logit, global index) over the dequantized head.

    Computed in fp32 (matching the kernel's fp32 accumulate) so the kernel-vs-reference check isolates
    the reduction. Returns ``(values [N/SGY] fp32, indices [N/SGY] int32)``; the host does the final argmax
    (``token = indices[argmax(values)]``). The reshape ``[N] -> [N/SGY, SGY]`` mirrors the kernel's
    threadgroup ownership: row t holds logits ``n in [t*SGY, t*SGY+SGY)`` == threadgroup t's SIMD-groups.
    """
    N = idx.shape[0]
    grp = torch.arange(N, device=idx.device) // _PALETTE_GROUP
    w = cb[grp.unsqueeze(1), idx.long()].float()              # [N, K] dequantized, fp32
    logits = torch.nn.functional.linear(x.float(), w)         # [1, N] fp32
    tg = logits.reshape(N // _HEAD_SGY, _HEAD_SGY)            # [num_tg, SGY]
    vals, local = tg.max(dim=1)                               # torch .max ties -> first (lowest local idx)
    base = torch.arange(N // _HEAD_SGY, device=idx.device) * _HEAD_SGY
    return vals.float(), (base + local).to(torch.int32)


def build_head_argmax_kernel(name: str = "gemma4_head_fused_int8_argmax") -> TorchMetalKernel:
    """Fused int8 head matvec + two-level argmax (returns per-threadgroup value/index partials)."""
    return TorchMetalKernel(
        name, input_names=["A", "IDX", "CB"], result_names=["PV", "PI"],
        src=_HEAD_ARGMAX_SRC.replace("__SGY__", str(_HEAD_SGY)),
        torch_defn=_head_argmax_torch_defn,
        metal_params=[
            MetalParameter("gid", "uint2", "thread_position_in_grid"),
            MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
            MetalParameter("tgid", "uint2", "threadgroup_position_in_grid"),
        ],
        template_dtypes={"A": "TYPE"},
    )


class Gemma4MetalHeadArgmax(nn.Module):
    """Greedy LM head as a fused int8 matvec + GPU argmax (returns partials; host picks the token).

    ``hidden [1,1,K] -> (partial_values [N/SGY] fp32, partial_indices [N/SGY] int32)``. The tied head
    weight (``embed_tokens.weight`` [vocab, hidden]) is k-means palettized once (the same format the fused
    FFN kernel reads). The caller reduces ``token = int(partial_indices[argmax(partial_values)])`` — exactly
    ``argmax(lm_head(hidden))`` (the Gemma-4 final softcap is monotonic, so greedy needs no softcap). Greedy
    only; sampling needs the real logits -> use ``Gemma4Head``.
    """

    coreai_externalize_specs: tuple = ()

    def __init__(self, head_weight: torch.Tensor, kernel: TorchMetalKernel, iters: int = 10) -> None:
        super().__init__()
        self.kernel = kernel
        self.vocab = int(head_weight.shape[0])
        if self.vocab % _HEAD_SGY:
            raise ValueError(f"vocab {self.vocab} not divisible by SGY {_HEAD_SGY}")
        self.num_tg = self.vocab // _HEAD_SGY
        idx, cb = palettize_grouped(head_weight, iters=iters)
        self.register_buffer("idx", idx)  # [vocab, hidden] uint8
        self.register_buffer("cb", cb)    # [G, 256] fp16

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_row = hidden.reshape(hidden.shape[-2], hidden.shape[-1])  # [1, hidden]
        return self.kernel(
            x_row, self.idx, self.cb,
            threads_per_grid=(32, self.vocab, 1),
            threads_per_thread_group=(32, _HEAD_SGY, 1),
            result_shapes=[[self.num_tg], [self.num_tg]],
        )


# =================================================================================================== #
# INT4 K-MEANS fused matvec + head — the DEVICE bandwidth lever (ondevice/INT4_KERNEL_HANDOFF.md).
#
# The iPhone GPU decode is BANDWIDTH-bound (FFN ~33 GB/s, head ~26 GB/s reading uint8 indices), unlike the
# Mac where int8-v3 is ALU/overhead-bound — so halving the index bytes is the ~2× device lever even though
# it gains nothing on Mac (the affine-int4 measurement). Two differences vs the affine kernel above:
#   * quantization = the SAME k-means palettization as the proven int8 path but n_bits=4 (16-entry codebook
#     per 32-output-row group) — de-risked 8/8 vs HF greedy in ondevice/_iso_gemma4_int4_numerics.py, where
#     affine-along-K was 7/8. The codebook gather comes back, but tg-staged (16 floats) it is ~free.
#   * packing = 8 nibbles/uint32 (the v3 lesson: wide 32-bit index loads, not byte loads).
# Same multi-row R + SGY structure as v3; a threadgroup's R*SGY = 32 rows = exactly ONE palette group, so
# the staged codebook is 16 entries. K must be divisible by 256 (32 lanes × 8 nibbles/lane per block) —
# gemma4 K = 1536/6144 both qualify. QP torch [N, K/8] uint32; QP[w,n]=torch qp[n,w] (coalesced over w).
_INT4KM_SRC = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint group = tgid.y;             // R*SGY = 32 rows/threadgroup = ONE palette group
    const uint base_row = (tgid.y * SGY + sg) * R;

    threadgroup float cbt[16];             // stage this group's 16-entry codebook once
    uint t = sg * 32 + lane;
    if (t < 16) cbt[t] = float(CB[t, group]);   // CB[q,group]=torch cb[group,q]
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {  // block = 32 lanes * 8 nibbles = 256 K
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(A[k0 + j, 0]);
        uint w0 = (kb >> 3) + lane;         // one uint32 word (8 nibbles) per lane
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = uint(QP[w0, n]);  // QP[w,n]=torch qp[n,w]; 8 indices/word
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uint q = (packed >> (j * 4)) & 0xf;
                s += xr[j] * cbt[q];        // 16-entry tg-codebook gather (k-means, not affine)
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r, 0] = TYPE(tot);
    }
"""


def pack_idx_nib_u32(idx: torch.Tensor) -> torch.Tensor:
    """[N, K] uint8 palette indices in 0..15 -> [N, K//8] uint32 (8 nibbles/word, low nibble = lowest k)."""
    N, K = idx.shape
    if K % 8:
        raise ValueError(f"K={K} not divisible by 8 (uint32 nibble pack)")
    if int(idx.max()) > 15:
        raise ValueError("palette indices exceed 4 bits (use palettize_grouped(n_bits=4))")
    i = idx.to(torch.int64).reshape(N, K // 8, 8)
    qp = torch.zeros(N, K // 8, dtype=torch.int64)
    for j in range(8):
        qp |= i[..., j] << (j * 4)
    return qp.to(torch.uint32).contiguous()


def _unpack_nib(qp: torch.Tensor) -> torch.Tensor:
    """[N, K//8] uint32 -> [N, K] int64 palette indices (inverse of pack_idx_nib_u32)."""
    N, K8 = qp.shape
    p = qp.to(torch.int64)
    return torch.stack([(p >> (j * 4)) & 0xF for j in range(8)], dim=-1).reshape(N, K8 * 8)


def _fused_int4km_torch_defn(x: torch.Tensor, qp: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    """Reference: unpack 8 nibbles/word, 16-entry codebook dequant (per 32-row group), linear."""
    idx = _unpack_nib(qp)
    grp = torch.arange(qp.shape[0], device=qp.device) // _PALETTE_GROUP
    w = cb[grp.unsqueeze(1), idx]
    return torch.nn.functional.linear(x, w.to(x.dtype))


def build_fused_int4km_kernel(name: str = "gemma4_ffn_fused_int4km") -> TorchMetalKernel:
    """4-bit k-means fused matvec: v3 multi-row structure, 8 nibbles/uint32, 16-entry tg-staged codebook."""
    return TorchMetalKernel(
        name, input_names=["A", "QP", "CB"], result_names=["C"],
        src=_INT4KM_SRC.replace("__R__", str(_V2_R)).replace("__SGY__", str(_V2_SGY)),
        torch_defn=_fused_int4km_torch_defn,
        metal_params=[MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def fused_int4km_call(kernel: TorchMetalKernel, x_row: torch.Tensor, qp: torch.Tensor,
                      cb: torch.Tensor) -> torch.Tensor:
    """Dispatch int4km (qp [N,K/8] uint32 nibbles, cb [G,16] fp16) -> [1, N]."""
    N = qp.shape[0]
    return kernel(x_row, qp, cb, threads_per_grid=(32, N // _V2_R, 1),
                  threads_per_thread_group=(32, _V2_SGY, 1), result_shapes=[[1, N]])


class MetalInt4KMMLP(nn.Module):
    """``MLP`` replacement running the 4-bit K-MEANS fused matvec — HALF the int8 index bytes.

    Same k-means palettization as the proven int8 path but n_bits=4 (de-risked 8/8 in
    ondevice/_iso_gemma4_int4_numerics.py). For the BANDWIDTH-bound device GPU; Mac is ALU-bound (no win).
    """

    def __init__(self, mlp: MLP, kernel: TorchMetalKernel, iters: int = 10) -> None:
        super().__init__()
        self.kernel = kernel
        for name, lin in (("gate", mlp.gate_proj), ("up", mlp.up_proj), ("down", mlp.down_proj)):
            if lin.weight.shape[1] % 256:
                raise ValueError(f"K={lin.weight.shape[1]} not divisible by 256 (32 lanes x 8 nibbles)")
            idx, cb = palettize_grouped(lin.weight, n_bits=4, iters=iters)
            self.register_buffer(f"{name}_qp", pack_idx_nib_u32(idx))  # [N, K/8] uint32
            self.register_buffer(f"{name}_cb", cb)                      # [G, 16] fp16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        xr = x.reshape(s, d)
        gate = nn.functional.gelu(fused_int4km_call(self.kernel, xr, self.gate_qp, self.gate_cb),
                                  approximate="tanh")
        up = fused_int4km_call(self.kernel, xr, self.up_qp, self.up_cb)
        out = fused_int4km_call(self.kernel, gate * up, self.down_qp, self.down_cb)
        return out.reshape(b, s, d)


def metalize_mlps_int4km(core: nn.Module, kernel: TorchMetalKernel | None = None,
                         iters: int = 10) -> TorchMetalKernel:
    """Swap every layer's ``.mlp`` for the 4-bit k-means ``MetalInt4KMMLP``. Returns the shared kernel."""
    if kernel is None:
        kernel = build_fused_int4km_kernel()
    for layer in core.text_model.layers:
        if isinstance(layer.mlp, MLP):
            layer.mlp = MetalInt4KMMLP(layer.mlp, kernel, iters=iters)
    return kernel


# 4-bit head: same two-level argmax as _HEAD_ARGMAX_SRC, but the matvec reads 8-nibble uint32 words and a
# tg-staged 16-entry codebook. A threadgroup's SGY=8 consecutive logits always sit inside ONE 32-row palette
# group (8 | 32), so the staged codebook is uniform per tg. K (hidden 1536) divisible by 256 as above.
_HEAD_ARGMAX_INT4KM_SRC = """
    const uint SGY = __SGY__;          // SIMD-groups per threadgroup == logits reduced per partial
    const uint K = A.get_extent(0);    // A torch [1, K]
    const uint lane = tid.x;           // 0..31 within the SIMD-group
    const uint sg = tid.y;             // 0..SGY-1 : which SIMD-group in this threadgroup
    const uint n = gid.y;              // global output column == candidate token id
    const uint group = (tgid.y * SGY) >> 5;   // palette group; uniform per tg (SGY | 32)

    threadgroup float cbt[16];         // stage this tg's 16-entry codebook once
    uint t = sg * 32 + lane;
    if (t < 16) cbt[t] = float(CB[t, group]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float partial = 0.0f;
    for (uint kb = 0; kb < K; kb += 256) {     // block = 32 lanes * 8 nibbles
        uint k0 = kb + lane * 8;
        uint packed = uint(QP[(k0 >> 3), n]);  // QP[w,n]=torch qp[n,w]; 8 indices/word
        float s = 0.0f;
        for (uint j = 0; j < 8; ++j) {
            uint q = (packed >> (j * 4)) & 0xf;
            s += float(A[k0 + j, 0]) * cbt[q];
        }
        partial += s;
    }
    float total = simd_sum(partial);   // logit[n]  (simd_sum broadcasts to all lanes)

    threadgroup float sval[__SGY__];
    threadgroup uint  sidx[__SGY__];
    if (lane == 0) { sval[sg] = total; sidx[sg] = n; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lane == 0 && sg == 0) {        // one thread reduces this threadgroup's SGY logits
        float best = sval[0];
        uint  bidx = sidx[0];
        for (uint j = 1; j < SGY; ++j) {
            if (sval[j] > best) { best = sval[j]; bidx = sidx[j]; }
        }
        PV[tgid.y] = best;             // fp32 partial max value (one per threadgroup)
        PI[tgid.y] = int(bidx);        // its GLOBAL token index
    }
"""


def _head_argmax_int4km_torch_defn(x: torch.Tensor, qp: torch.Tensor,
                                   cb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference + fake: per-threadgroup (max logit, global index) over the 4-bit dequantized head."""
    N = qp.shape[0]
    idx = _unpack_nib(qp)
    grp = torch.arange(N, device=qp.device) // _PALETTE_GROUP
    w = cb[grp.unsqueeze(1), idx].float()                     # [N, K] dequantized, fp32
    logits = torch.nn.functional.linear(x.float(), w)         # [1, N] fp32
    tg = logits.reshape(N // _HEAD_SGY, _HEAD_SGY)            # [num_tg, SGY]
    vals, local = tg.max(dim=1)                               # torch .max ties -> first (lowest local idx)
    base = torch.arange(N // _HEAD_SGY, device=qp.device) * _HEAD_SGY
    return vals.float(), (base + local).to(torch.int32)


def build_head_argmax_int4km_kernel(name: str = "gemma4_head_fused_int4km_argmax") -> TorchMetalKernel:
    """4-bit k-means head matvec + two-level argmax (same partials contract as the int8 head kernel)."""
    return TorchMetalKernel(
        name, input_names=["A", "QP", "CB"], result_names=["PV", "PI"],
        src=_HEAD_ARGMAX_INT4KM_SRC.replace("__SGY__", str(_HEAD_SGY)),
        torch_defn=_head_argmax_int4km_torch_defn,
        metal_params=[
            MetalParameter("gid", "uint2", "thread_position_in_grid"),
            MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
            MetalParameter("tgid", "uint2", "threadgroup_position_in_grid"),
        ],
        template_dtypes={"A": "TYPE"},
    )


class Gemma4MetalHeadArgmaxInt4(nn.Module):
    """4-bit k-means variant of ``Gemma4MetalHeadArgmax`` — same partials contract, HALF the index bytes."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, head_weight: torch.Tensor, kernel: TorchMetalKernel, iters: int = 10) -> None:
        super().__init__()
        self.kernel = kernel
        self.vocab = int(head_weight.shape[0])
        if self.vocab % _HEAD_SGY:
            raise ValueError(f"vocab {self.vocab} not divisible by SGY {_HEAD_SGY}")
        if head_weight.shape[1] % 256:
            raise ValueError(f"hidden {head_weight.shape[1]} not divisible by 256 (32 lanes x 8 nibbles)")
        self.num_tg = self.vocab // _HEAD_SGY
        idx, cb = palettize_grouped(head_weight, n_bits=4, iters=iters)
        self.register_buffer("qp", pack_idx_nib_u32(idx))  # [vocab, hidden/8] uint32
        self.register_buffer("cb", cb)                      # [G, 16] fp16

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_row = hidden.reshape(hidden.shape[-2], hidden.shape[-1])  # [1, hidden]
        return self.kernel(
            x_row, self.qp, self.cb,
            threads_per_grid=(32, self.vocab, 1),
            threads_per_thread_group=(32, _HEAD_SGY, 1),
            result_shapes=[[self.num_tg], [self.num_tg]],
        )


# =================================================================================================== #
# FUSED FLASH-DECODE SDPA — Lever #4 (WHOLE-LAYER fusion, ATTENTION_KERNEL_HANDOFF §B / §C).
#
# The host-cache attention does `cat(past,cur)` then a COMPOSITE SDPA (scale·q, q·Kᵀ, +mask, softmax,
# ·V) — ~12 MPSGraph dispatches/layer that S0b pinned as part of the 51% op-DISPATCH-bound "rest". This
# kernel collapses the whole q=1 decode attention (the two `cat`s + the SDPA subgraph) into ONE dispatch
# with fp32 online (flash) softmax, the MLX `sdpa_vector` structure:
#   * ONE SIMD-group (32 lanes) per query head; grid (32, n_heads, 1), tg (32, 1, 1) -> no tg-mem, no barrier.
#   * each lane owns ept = head_dim/32 CONTIGUOUS head dims (256->8, 512->16; one kernel serves both).
#   * per key j: each lane computes a partial dot over its dims; `simd_sum` -> the full score (broadcast to
#     all 32 lanes); +additive mask[j] (scale=1.0). Online softmax (running max m, sum l, output acc[ept]) is
#     identical on every lane (simd_sum already synchronised the simd-group), so NO threadgroup barrier.
#   * GQA is free: nkv=1, so all n_heads simd-groups read the SAME K/V. The `cat` is folded — key j<B reads
#     past, j==B reads the current token (returned separately for the host write-back).
# fp32 accumulate (same reason as the FFN/head kernels: gemma4's large activations make fp16 matmul/softmax
# accumulation lossy). additive mask is 0 / -1e4 == the composite's hardcoded -1e4 for not-attended, so it is
# added DIRECTLY (no bool round-trip). Output context [n_heads, head_dim] is head-major == the composite's
# `out.permute(0,2,1,3).reshape(1,1,H*hd)` fed to o_proj, so the caller skips the permute.
_SDPA_MAX_EPT = 16  # head_dim/32 upper bound (global_head_dim 512 -> 16); register arrays sized to this
_SDPA_DECODE_SRC = """
    const uint hd  = A.get_extent(0);      // A = q, torch [n_heads, head_dim] -> innermost = head_dim
    const uint Bp1 = MASK.get_extent(0);   // MASK torch [B+1]  (past B columns ++ current)
    const uint B   = Bp1 - 1;
    const uint lane = gid.x;               // 0..31 within the simd-group
    const uint h    = gid.y;               // query head (one simd-group each)
    const uint ept  = hd / 32;             // contiguous head dims owned by this lane (<= 16)

    float qx[__MAXEPT__];                  // this lane's slice of q[h] (scale = 1.0)
    for (uint i = 0; i < ept; ++i) qx[i] = float(A[lane * ept + i, h]);

    float o[__MAXEPT__];                   // unnormalised output accumulator (this lane's dims)
    for (uint i = 0; i < ept; ++i) o[i] = 0.0f;
    float m = -1e30f;                      // running max score
    float l = 0.0f;                        // running sum of exp

    for (uint j = 0; j < Bp1; ++j) {
        bool cur = (j == B);               // current token lives in CK/CV at column 0; past in PK/PV[.,j]
        float p = 0.0f;
        for (uint i = 0; i < ept; ++i) {
            uint d = lane * ept + i;
            p += qx[i] * (cur ? float(CK[d, 0]) : float(PK[d, j]));   // PK[d,j]=torch past_k[j,d]
        }
        float s = simd_sum(p) + float(MASK[j]);   // full q·k_j (broadcast) + additive mask
        float mnew = max(m, s);
        float corr = exp(m - mnew);        // rescale prior accumulators
        float e    = exp(s - mnew);        // this key's (unnormalised) weight
        l = l * corr + e;
        for (uint i = 0; i < ept; ++i) {
            uint d = lane * ept + i;
            o[i] = o[i] * corr + e * (cur ? float(CV[d, 0]) : float(PV[d, j]));
        }
        m = mnew;
    }
    float inv = 1.0f / l;
    for (uint i = 0; i < ept; ++i) CTX[lane * ept + i, h] = TYPE(o[i] * inv);  // CTX[d,h]=torch ctx[h,d]
"""


def _sdpa_decode_torch_defn(q: torch.Tensor, past_k: torch.Tensor, cur_k: torch.Tensor,
                            past_v: torch.Tensor, cur_v: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
    """Reference (fp32) for the flash-decode kernel: cat past+current, scale-1.0 masked softmax, ·V.

    Shapes: q [H, hd]; past_k/past_v [B, hd]; cur_k/cur_v [1, hd]; mask [B+1] additive (0 / -1e4).
    Plain (non-online) fp32 softmax == the kernel's online softmax; returns context [H, hd] in q's dtype.
    """
    hd = q.shape[-1]
    k = torch.cat([past_k, cur_k.reshape(1, hd)], dim=0).float()   # [B+1, hd]
    v = torch.cat([past_v, cur_v.reshape(1, hd)], dim=0).float()
    scores = q.float() @ k.t() + mask.float().reshape(1, -1)       # [H, B+1], scale = 1.0
    w = torch.softmax(scores, dim=-1)
    return (w @ v).to(q.dtype)                                     # [H, hd]


def build_sdpa_decode_kernel(name: str = "gemma4_sdpa_decode") -> TorchMetalKernel:
    """Fused q=1 flash-decode SDPA (GQA, dual head_dim 256/512, additive mask, scale=1.0, fp32 softmax)."""
    return TorchMetalKernel(
        name, input_names=["A", "PK", "CK", "PV", "CV", "MASK"], result_names=["CTX"],
        src=_SDPA_DECODE_SRC.replace("__MAXEPT__", str(_SDPA_MAX_EPT)),
        torch_defn=_sdpa_decode_torch_defn,
        metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


def sdpa_decode_call(kernel: TorchMetalKernel, q: torch.Tensor, past_k: torch.Tensor,
                     cur_k: torch.Tensor, past_v: torch.Tensor, cur_v: torch.Tensor,
                     mask: torch.Tensor) -> torch.Tensor:
    """Dispatch the flash-decode kernel: q [H,hd], past [B,hd], cur [1,hd], mask [B+1] -> context [H,hd]."""
    H, hd = q.shape
    return kernel(q, past_k, cur_k, past_v, cur_v, mask,
                  threads_per_grid=(32, H, 1), threads_per_thread_group=(32, 1, 1),
                  result_shapes=[[H, hd]])


class MetalFlashSDPA(nn.Module):
    """Weightless drop-in for the host-cache composite SDPA + the two KV `cat`s (q=1 decode).

    ``forward(q [1,H,1,hd], past_k [1,nkv,B,hd], cur_k [1,nkv,1,hd], past_v, cur_v, mask [1,1,1,B+1])``
    -> context [1, 1, H*hd] (head-major, ready for o_proj). nkv must be 1 (gemma4 GQA)."""

    def __init__(self, kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.kernel = kernel

    def forward(self, q: torch.Tensor, past_k: torch.Tensor, cur_k: torch.Tensor,
                past_v: torch.Tensor, cur_v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        H, hd = q.shape[1], q.shape[3]
        ctx = sdpa_decode_call(
            self.kernel, q.reshape(H, hd), past_k.reshape(-1, hd), cur_k.reshape(1, hd),
            past_v.reshape(-1, hd), cur_v.reshape(1, hd), mask.reshape(-1),
        )
        return ctx.reshape(1, 1, H * hd)


# =================================================================================================== #
# FUSED RMSNorm BRIDGE — Lever #4 glue fusion (the dispatch-bound "rest", ATTENTION_KERNEL_HANDOFF §B).
#
# S0b pinned the 51% "rest" as ~28 tiny op-dispatches/layer (launch overhead). Unlike the SDPA fold (which
# REGRESSED — a low-occupancy flash kernel can't match MPSGraph's high-occupancy attention GEMM), the per-layer
# norm/residual GLUE is low-parallelism for q=1 EITHER way (one [1,1,1536] row), so a custom kernel does NOT
# lose occupancy — it only removes launches. This kernel fuses the post-attention bridge
#   x2 = residual + RMSNorm(attn_out, w_post_attn);  ffn_in = RMSNorm(x2, w_pre_ffn)
# (3 composite ops -> 1 dispatch, 2 outputs). One threadgroup reduces the 1536-dim row twice (simd_sum +
# threadgroup-mem across NSIMD simd-groups), staging x2 in tg memory between the two norms. EXACT Gemma-4
# RMSNorm numerics (fp32 reduce; `(x*inv_rms).to(fp16)*weight`, weight applied directly — no +1). It is the
# probe for whether manual glue-fusion recovers dispatch overhead (vs MPSGraph already fusing the elementwise
# chain); measure the full-core delta before extending to the other bridges.
_NB_T = 256       # threads/threadgroup (one tg owns the whole row)
_NB_NSIMD = 8     # = _NB_T / 32
_NB_D = 1536      # gemma4 E2B hidden_size (the bridge operates on the hidden dim)
_NORM_BRIDGE_SRC = """
    const uint D = X.get_extent(0);    // X torch [D]
    const uint T = __T__;
    const uint tid = gtid.x;           // 0..T-1 (one threadgroup)
    const uint sid = tid >> 5;         // simd-group index
    const uint lane = tid & 31;
    threadgroup float sgp[__NSIMD__];
    threadgroup TYPE  x2s[__D__];       // staged residual-stream x2 between the two norms

    // --- pass 1: x2 = RESID + RMSNorm(X, W1) ---
    float ss = 0.0f;
    for (uint i = tid; i < D; i += T) { float a = float(X[i]); ss += a * a; }
    float sg = simd_sum(ss);
    if (lane == 0) sgp[sid] = sg;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint s = 0; s < __NSIMD__; ++s) tot += sgp[s];
    float inv1 = rsqrt(tot / float(D) + __EPS__);
    for (uint i = tid; i < D; i += T) {
        TYPE t  = TYPE(float(X[i]) * inv1) * W1[i];   // (x*inv_rms cast to T) * weight
        TYPE x2 = RESID[i] + t;
        x2s[i] = x2;
        XR[i]  = x2;                                   // output: new residual stream
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- pass 2: ffn_in = RMSNorm(x2, W2) ---
    float ss2 = 0.0f;
    for (uint i = tid; i < D; i += T) { float v = float(x2s[i]); ss2 += v * v; }
    float sg2 = simd_sum(ss2);
    if (lane == 0) sgp[sid] = sg2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot2 = 0.0f;
    for (uint s = 0; s < __NSIMD__; ++s) tot2 += sgp[s];
    float inv2 = rsqrt(tot2 / float(D) + __EPS__);
    for (uint i = tid; i < D; i += T) {
        XN[i] = TYPE(float(x2s[i]) * inv2) * W2[i];    // output: FFN input
    }
"""


def _rms(t: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Gemma-4 RMSNorm matching RMSNormImpl: (orig·inv_rms_f32).to(dtype)·weight (no +1)."""
    f = t.float()
    inv = torch.rsqrt((f * f).mean(-1, keepdim=True) + eps)
    return (t * inv).to(t.dtype) * w


def _norm_bridge_torch_defn(x: torch.Tensor, resid: torch.Tensor, w1: torch.Tensor,
                            w2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: x2 = resid + RMSNorm(x, w1); ffn_in = RMSNorm(x2, w2). All [D]."""
    x2 = resid + _rms(x, w1)
    return x2, _rms(x2, w2)


def build_norm_bridge_kernel(name: str = "gemma4_norm_bridge") -> TorchMetalKernel:
    """Fused post-attention bridge (RMSNorm + residual + RMSNorm), 2 outputs (x2, ffn_in)."""
    src = (_NORM_BRIDGE_SRC.replace("__T__", str(_NB_T)).replace("__NSIMD__", str(_NB_NSIMD))
           .replace("__D__", str(_NB_D)).replace("__EPS__", "0.000001f"))
    return TorchMetalKernel(
        name, input_names=["X", "RESID", "W1", "W2"], result_names=["XR", "XN"], src=src,
        torch_defn=_norm_bridge_torch_defn,
        metal_params=[MetalParameter("gtid", "uint2", "thread_position_in_threadgroup")],
        template_dtypes={"X": "TYPE"},
    )


class MetalNormBridge(nn.Module):
    """Drop-in for ``post_attention_layernorm -> +residual -> pre_feedforward_layernorm`` (one kernel).

    ``forward(attn_out [1,1,D], residual [1,1,D]) -> (x2 [1,1,D], ffn_in [1,1,D])`` using the layer's two
    RMSNorm weights (captured at construction)."""

    def __init__(self, post_attn_norm: nn.Module, pre_ffn_norm: nn.Module,
                 kernel: TorchMetalKernel) -> None:
        super().__init__()
        self.kernel = kernel
        self.register_buffer("w_post", post_attn_norm.weight.detach().contiguous())
        self.register_buffer("w_pre", pre_ffn_norm.weight.detach().contiguous())

    def forward(self, attn_out: torch.Tensor, residual: torch.Tensor):
        D = attn_out.shape[-1]
        xr, xn = self.kernel(
            attn_out.reshape(D), residual.reshape(D), self.w_post, self.w_pre,
            threads_per_grid=(_NB_T, 1, 1), threads_per_thread_group=(_NB_T, 1, 1),
            result_shapes=[[D], [D]],
        )
        return xr.reshape(1, 1, D), xn.reshape(1, 1, D)


def export_to_coreai_with_kernels(
    model: nn.Module,
    reference_inputs: dict,
    *,
    custom_kernels: list,
    dynamic_shapes: dict | None = None,
    input_names: tuple | None = None,
    output_names: tuple | None = None,
    state_names: tuple | None = None,
    externalize_modules: list | tuple | None = (),
):
    """``coreai_models.export.macos.export_to_coreai`` + the missing custom-kernel hook.

    Mirrors that function exactly (torch.export -> coreai decomp -> remove_functionalization ->
    TorchConverter.add_pytorch_module -> register_custom_torch_lowering -> to_coreai) but calls
    ``converter.register_custom_kernels(custom_kernels)`` BEFORE ``add_pytorch_module`` — required
    because add_pytorch_module validates the exported program against the converter's known lowerings,
    which must already include the ``coreai_metal_kernels::*`` custom ops (WWDC26 session 325:
    "register my custom kernels with the converter, then add the exported program as before").

    ``externalize_modules`` defaults to ``()`` (disabled) — the gemma4 host-cache core opts out of
    composite externalization (it keeps PLE front-end norms as attributes). Pass the default specs
    explicitly only for a core that wants them.
    """
    import coreai_torch
    from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization

    if input_names is None:
        state_set = set(state_names or ())
        input_names = tuple(k for k in reference_inputs if k not in state_set)

    def export_fn(module: nn.Module):
        with torch.no_grad():
            ep = torch.export.export(module, args=(), kwargs=reference_inputs, dynamic_shapes=dynamic_shapes)
        ep = ep.run_decompositions(coreai_torch.get_decomp_table())
        remove_functionalization(ep)
        return ep

    model.eval()
    converter = coreai_torch.TorchConverter()
    converter.register_custom_kernels(custom_kernels)
    converter.add_pytorch_module(
        model, export_fn=export_fn,
        externalize_modules=list(externalize_modules) or None,
        input_names=input_names, output_names=output_names, state_names=state_names,
    )
    register_custom_torch_lowering(converter)
    return converter.to_coreai()
