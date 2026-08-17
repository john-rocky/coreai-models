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

KERNEL-BOUNDARY RULE (Apple 178056451): the custom-op binder requires every input edge to be an
innermost-stride-1 dense tensor, and MPSGraph values carry no strides — an authoring-side
`.contiguous()` cannot survive conversion, the layout planner picks the edge layout and it picks
transposed views (measured: the [b,s,16]->transpose g/beta view binds with stride 16 -> hard assert
in GPUCustomMetalKernelOps.mm). So NOTHING transpose-derived may feed the kernel. This module
therefore takes the projection/conv-NATIVE tensors — the post-silu conv activation [b, conv_dim, S]
(channel-major, straight off the causal conv), un-transposed g/beta [b, S, h], and the state
[b, h, dk, dv] — and absorbs the q/k/v split, the (head, dim) split, the GVA head mapping, and the
qk l2-norm + q-scale INTO the kernel indexing. The strided edge never exists in the graph.

Layout: ONE thread per (head, value-column). Thread (c = value col, hh = head) owns the recurrent
state COLUMN ``state[0:dk, c]`` (dk floats in registers) and runs the full S-step recurrence for that
column. All reductions are over dk (the rows this thread owns) -> purely intra-thread, NO cross-lane
sums. k_t / q_t (the dk-vectors every column needs each step) are staged in threadgroup memory by the
column threads (requires dk <= dv, true for this family: 128 == 128). One threadgroup per value head.
Register with the converter via ``export_to_coreai_with_kernels``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

# DSL axes are reversed vs torch (DSL dim0 = torch innermost):
#   MIXED torch [conv_dim, S] -> DSL [S, conv_dim]:       MIXED[t, channel]
#   G/BETA torch [S, h]       -> DSL [h, S]:              G[hh, t]
#   S0/SNEW torch [h, dk, dv] -> DSL [dv, dk, h]:         S0[c, d, hh]
#   OUT torch [chunk_max, h*dv] -> DSL [h*dv, chunk_max]: OUT[hh*dv + c, t]
_GDN_CHUNK_SRC = """
    const uint S  = MIXED.get_extent(0);       // chunk length (dynamic; torch-innermost of MIXED)
    const uint c  = gid.x;                      // value column (0..dv-1) — this thread's state column
    const uint hh = gid.y;                      // value head (one threadgroup per head)
    const uint kb  = (hh / __R__) * __DK__;     // k/q channel base (GVA: __R__ value heads per key head)
    const uint qch = kb + c;                    // this thread's staged q channel
    const uint kch = __KEYDIM__ + kb + c;       // this thread's staged k channel
    const uint vch = __VOFF__ + hh * __DV__ + c;   // this thread's v channel

    float st[__DK__];                            // state column [dk] for value-col c (fp32, persists over S)
    for (uint d = 0; d < __DK__; ++d) st[d] = float(S0[c, d, hh]);

    threadgroup float ksh[__DK__];               // raw k_t / q_t staged for the whole head each step
    threadgroup float qsh[__DK__];

    for (uint t = 0; t < S; ++t) {
        qsh[c] = float(MIXED[t, qch]);           // dv column-threads stage the dk-dim raw q_t / k_t
        ksh[c] = float(MIXED[t, kch]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float qsc = __QSCALE__;                  // dk^-0.5 (baked)
        float ksc = 1.0f;
        if (__L2__) {                            // qk l2-norm, in-kernel fp32 (every thread computes
            float qs = 0.0f, ks = 0.0f;          //  the same scalars from threadgroup memory)
            for (uint d = 0; d < __DK__; ++d) { qs += qsh[d] * qsh[d]; ks += ksh[d] * ksh[d]; }
            qsc *= rsqrt(qs + 1e-6f);
            ksc  = rsqrt(ks + 1e-6f);
        }

        float gt = float(G[hh, t]);              // per-head scalar decay logit
        float bt = float(BETA[hh, t]);
        float vc = float(MIXED[t, vch]);
        float ge = exp(gt);                      // g is the NEGATIVE log-decay -> multiplier exp(g)

        float kv = 0.0f;
        for (uint d = 0; d < __DK__; ++d) { st[d] *= ge; kv += st[d] * ksh[d]; }   // decay, then k^T state
        float kd = ksc * (vc - ksc * kv) * bt;   // ksc*delta: k_eff = ksc*ksh both in the dot and the write
        float oc = 0.0f;
        for (uint d = 0; d < __DK__; ++d) { st[d] += ksh[d] * kd; oc += st[d] * qsh[d]; }  // write, then q^T state
        OUT[hh * __DV__ + c, t] = TYPE(oc * qsc);
        threadgroup_barrier(mem_flags::mem_threadgroup);   // before next step overwrites ksh/qsh
    }
    for (uint d = 0; d < __DK__; ++d) SNEW[c, d, hh] = TYPE(st[d]);
"""


def build_gdn_chunk_kernel(name: str = "qwen3_5_gdn_chunk", num_k: int = 16, num_v: int = 16,
                           dk: int = 128, dv: int = 128, use_qk_l2_norm: bool = True,
                           chunk_max: int = 64) -> TorchMetalKernel:
    """``chunk_max`` = the static seq extent of the OUT buffer. Custom-kernel result_shapes can't carry
    a dynamic dim, so OUT is fixed [chunk_max, h*dv]; the MSL writes only the actual [0:S] rows and the
    module slices [:S] in-graph. Requires the engine's prefill chunk size <= chunk_max. All head/dim
    geometry is baked into the MSL — the graph hands over only conv-native dense tensors."""
    if dk > dv:
        raise ValueError(f"staging needs dk <= dv (dv column-threads stage dk dims): dk={dk} dv={dv}")
    if num_v % num_k:
        raise ValueError(f"GVA needs num_v % num_k == 0: num_v={num_v} num_k={num_k}")

    def _torch_defn(
        MIXED: torch.Tensor, G: torch.Tensor, BETA: torch.Tensor, S0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shape-inference reference for torch.export (the real numerics are the MSL on the engine).
        MUST NOT iterate the dynamic seq dim S (a python `range(S)` loop specializes S to a constant
        and breaks the dynamic export). Returns the correct shapes: OUT [chunk_max, h*dv] (kernel
        writes only [0:S]), SNEW [h, dk, dv]. The value content is irrelevant — these bundles only
        run on the engine."""
        h, dv_ = S0.shape[0], S0.shape[2]
        out = MIXED.new_zeros(chunk_max, h * dv_)
        snew = S0.clone()
        return out, snew

    src = (_GDN_CHUNK_SRC
           .replace("__KEYDIM__", str(num_k * dk))
           .replace("__VOFF__", str(2 * num_k * dk))
           .replace("__R__", str(num_v // num_k))
           .replace("__DK__", str(dk))
           .replace("__DV__", str(dv))
           .replace("__QSCALE__", f"{dk ** -0.5!r}f")
           .replace("__L2__", "1" if use_qk_l2_norm else "0"))
    return TorchMetalKernel(
        name,
        input_names=["MIXED", "G", "BETA", "S0"],
        result_names=["OUT", "SNEW"],
        src=src,
        torch_defn=_torch_defn,
        metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
        template_dtypes={"MIXED": "TYPE"},
    )


class MetalGDNChunk(nn.Module):
    """Drop-in for the GDN scan, called from `Qwen3_5GatedDeltaNet.forward` BEFORE the q/k/v
    reshape/transpose block (kernel-boundary rule above). ``forward(conv, g, beta, S0)`` with
    conv [b, conv_dim, S] (post-silu, channel-major), g/beta [b, S, h], S0 [b, h, dk, dv]
    -> (out [b, S, h, dv], Snew [b, h, dk, dv]) — matching the in-graph scan paths.
    l2-norm + q-scale run INSIDE the kernel (fp32)."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, kernel: TorchMetalKernel, chunk_max: int = 64) -> None:
        super().__init__()
        self.kernel = kernel
        self.chunk_max = chunk_max

    def forward(self, conv, g, beta, S0):
        b, _, S = conv.shape
        h, dk, dv = S0.shape[1], S0.shape[2], S0.shape[3]
        # drop batch (b == 1) by slicing — reshape/slice-derived edges only, never a transpose
        out, snew = self.kernel(
            conv[0], g[0], beta[0], S0[0],
            threads_per_grid=(dv, h, 1), threads_per_thread_group=(dv, 1, 1),
            result_shapes=[[self.chunk_max, h * dv], [h, dk, dv]])
        out = out[:S].reshape(1, S, h, dv)   # rows [0:S] valid (S dynamic); reshape, not transpose
        return out, snew.unsqueeze(0)        # [1,h,dk,dv]


def metalize_gdn_chunk(model: nn.Module, kernel: TorchMetalKernel | None = None) -> TorchMetalKernel:
    """Attach the kernel to every linear (GDN) layer's `linear_attn` so its forward uses the kernel
    scan (set `use_metal_chunk=True`). Returns the shared kernel — register with the converter."""
    lins = [layer.linear_attn for layer in model.model.layers if not layer.is_full]
    if not lins:
        raise RuntimeError("metalize_gdn_chunk: no linear/GDN layers found")
    la0 = lins[0]
    if kernel is None:
        kernel = build_gdn_chunk_kernel(
            num_k=la0.num_k, num_v=la0.num_v, dk=la0.dk, dv=la0.dv,
            use_qk_l2_norm=la0.gdu.use_qk_l2_norm)
    for la in lins:
        la.metal_chunk = MetalGDNChunk(kernel)
        la.use_metal_chunk = True
    return kernel
