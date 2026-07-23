# Community port — NOT an Apple model.
"""fp32 gated-delta-net CHUNKED-SCAN Metal kernel for qwen3.5-family GDN layers (MiniCPM-V-4.6 etc.).

THE prefill TTFT lever. The in-graph chunked scan (`_gated_delta_chunk`, the matmul/doubling-inverse
form) is bit-exact in fp32 but the engine runs it in fp16, where the (I+M)^-1 Neumann/doubling
expansion is numerically UNSTABLE for big chunks — NaN at chunk>=64, precision loss at 32 (measured
`_smoke/test_chunk_fp16_numerics.py`). That caps the stock-engine safe chunk at 8 (~8x prefill);
but the per-chunk call time is FLAT to S=32 (`_smoke/probe_chunk_curve.py`: S=1..32 all ~20 ms),
so chunk=32/64 would be 32-45x — IF the scan were numerically stable at that size.

This kernel makes it stable: it runs the *sequential* gated-delta recurrence (the decode-exact math,
fp32, no matrix inverse) for a whole chunk of any S in ONE GPU dispatch, replacing the fragile
in-graph scan. The weight-heavy projections (in_proj/out_proj/MLP/attn) still batch across the chunk
in-graph (the amortization that makes prefill fast); only the cheap recurrence moves to the kernel.

Layout: ONE thread per (head, value-column). Thread (c = value col, hh = head) owns the recurrent
state COLUMN ``state[0:dk, c]`` (dk floats in registers) and runs the full S-step recurrence for that
column. All reductions are over dk (the rows this thread owns) -> purely intra-thread, NO cross-lane
sums. k_t / q_t (the dk-vectors every column needs each step) are staged in threadgroup memory by the
column threads (requires dv == dk, true for this family: 128 == 128). 16 heads = 16 threadgroups.

l2-norm of q,k and the q-scale are done IN-GRAPH before the kernel (cheap, lowers fine); the kernel is
the pure recurrence. Register with the converter via ``export_to_coreai_with_kernels``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

# DSL axes are reversed vs torch: torch [h, S, dk] -> DSL[dk, S, h], so KN[d,t,hh] = torch kn[hh,t,d].
# torch S0/Snew [h, dk, dv] -> DSL[dv, dk, h]: S0[c,d,hh] = torch S0[hh,d,c]. G/BETA [h,S] -> DSL[S,h].
_GDN_CHUNK_SRC = """
    const uint dk = KN.get_extent(0);          // key/state-row dim (== dv here)
    const uint S  = KN.get_extent(1);          // chunk length (dynamic)
    const uint c  = gid.x;                      // value column (0..dv-1) — this thread's state column
    const uint hh = gid.y;                      // head (one threadgroup per head)

    float st[__MAXDK__];                         // state column [dk] for value-col c (fp32, persists over S)
    for (uint d = 0; d < dk; ++d) st[d] = float(S0[c, d, hh]);

    threadgroup float ksh[__MAXDK__];            // k_t / q_t staged for the whole head each step
    threadgroup float qsh[__MAXDK__];

    for (uint t = 0; t < S; ++t) {
        ksh[c] = float(KN[c, t, hh]);            // 128 column-threads load the 128-dim k_t / q_t
        qsh[c] = float(QN[c, t, hh]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float gt = float(G[t, hh]);              // per-head scalar decay (already exp'd? no: exp here)
        float bt = float(BETA[t, hh]);
        float vc = float(V[c, t, hh]);
        float ge = exp(gt);                       // g is the NEGATIVE log-decay -> multiplier exp(g)

        float kv = 0.0f;
        for (uint d = 0; d < dk; ++d) { st[d] *= ge; kv += st[d] * ksh[d]; }   // decay, then k^T state
        float delta = (vc - kv) * bt;
        float oc = 0.0f;
        for (uint d = 0; d < dk; ++d) { st[d] += ksh[d] * delta; oc += st[d] * qsh[d]; }  // write, then q^T state
        OUT[c, t, hh] = TYPE(oc);
        threadgroup_barrier(mem_flags::mem_threadgroup);   // before next step overwrites ksh/qsh
    }
    for (uint d = 0; d < dk; ++d) SNEW[c, d, hh] = TYPE(st[d]);
"""


def build_gdn_chunk_kernel(name: str = "qwen3_5_gdn_chunk", max_dk: int = 128,
                           chunk_max: int = 64) -> TorchMetalKernel:
    """``chunk_max`` = the static seq extent of the OUT buffer. Custom-kernel result_shapes can't carry
    a dynamic dim, so OUT is fixed [h, chunk_max, dv]; the MSL writes only the actual [0:S] rows and the
    module slices [:, :S, :] in-graph. Requires the engine's prefill chunk size <= chunk_max."""

    def _torch_defn(
        QN: torch.Tensor, KN: torch.Tensor, V: torch.Tensor,
        G: torch.Tensor, BETA: torch.Tensor, S0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shape-inference reference for torch.export (the real numerics are the MSL on the engine;
        validated against `_gated_delta_chunk` in `_smoke/gate_gdn_kernel.py`). MUST NOT iterate the
        dynamic seq dim S (a python `range(S)` loop specializes S to a constant and breaks the dynamic
        export). Returns the correct shapes: OUT [h, chunk_max, dv] (kernel writes only [0:S]),
        SNEW [h, dk, dv]. The value content is irrelevant — these bundles only run on the engine."""
        h, dk = KN.shape[0], KN.shape[-1]
        dv = V.shape[-1]
        out = QN.new_zeros(h, chunk_max, dv)
        snew = S0.clone()
        return out, snew

    return TorchMetalKernel(
        name,
        input_names=["QN", "KN", "V", "G", "BETA", "S0"],
        result_names=["OUT", "SNEW"],
        src=_GDN_CHUNK_SRC.replace("__MAXDK__", str(max_dk)),
        torch_defn=_torch_defn,
        metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
        template_dtypes={"QN": "TYPE"},
    )


class MetalGDNChunk(nn.Module):
    """Drop-in for the GDN scan (same signature as `_gated_delta_chunk`): l2-norm + scale in-graph,
    then the fp32 recurrence kernel. ``forward(q,k,v,g,beta,S0)`` with q,k [b,h,S,dk], v [b,h,S,dv],
    g,beta [b,h,S], S0 [b,h,dk,dv] -> (out [b,S,h,dv], Snew [b,h,dk,dv]) — matching the in-graph path.
    """

    coreai_externalize_specs: tuple = ()

    def __init__(self, kernel: TorchMetalKernel, use_qk_l2_norm: bool = True,
                 chunk_max: int = 64) -> None:
        super().__init__()
        self.kernel = kernel
        self.use_qk_l2_norm = use_qk_l2_norm
        self.chunk_max = chunk_max

    def forward(self, q, k, v, g, beta, S0):
        def l2norm(x):
            return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)
        b, h, S, dk = k.shape
        dv = v.shape[-1]
        if self.use_qk_l2_norm:
            q = l2norm(q)
            k = l2norm(k)
        q = q * (dk ** -0.5)
        # drop batch (b == 1) for the kernel; inputs contiguous in [h,S,*] / [h,dk,dv]
        qn = q[0].contiguous(); kn = k[0].contiguous(); vv = v[0].contiguous()
        gg = g[0].contiguous(); bb = beta[0].contiguous(); s0 = S0[0].contiguous()
        out, snew = self.kernel(            # OUT fixed [h,chunk_max,dv] (result_shapes can't be dynamic)
            qn, kn, vv, gg, bb, s0,
            threads_per_grid=(dv, h, 1), threads_per_thread_group=(dv, 1, 1),
            result_shapes=[[h, self.chunk_max, dv], [h, dk, dv]])
        out = out[:, :S, :]                       # slice the valid rows (S dynamic)
        out = out.unsqueeze(0).transpose(1, 2)    # [1,h,S,dv] -> [1,S,h,dv]
        return out, snew.unsqueeze(0)             # [1,h,dk,dv]


def metalize_gdn_chunk(model: nn.Module, kernel: TorchMetalKernel | None = None) -> TorchMetalKernel:
    """Attach the kernel to every linear (GDN) layer's `linear_attn` so its forward uses the kernel
    scan (set `use_metal_chunk=True`). Returns the shared kernel — register with the converter."""
    if kernel is None:
        kernel = build_gdn_chunk_kernel()
    n = 0
    for layer in model.model.layers:
        if not layer.is_full:
            la = layer.linear_attn
            la.metal_chunk = MetalGDNChunk(kernel, use_qk_l2_norm=la.gdu.use_qk_l2_norm)
            la.use_metal_chunk = True
            n += 1
    if n == 0:
        raise RuntimeError("metalize_gdn_chunk: no linear/GDN layers found")
    return kernel
