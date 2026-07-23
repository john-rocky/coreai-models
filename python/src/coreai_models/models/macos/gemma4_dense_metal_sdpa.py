# Community port — NOT an Apple model.
"""Custom flash-decode SDPA Metal kernel for the Gemma 4 *dense* (12B) FULL layers.

THE BYPASS for the engine blocker (see ``GEMMA4_12B_STATE.md`` / apple/coreai-models#27).
Every full-attention layer of the dense 12B crashes the pipelined GPU engine at the first
decode token:

    allocateMTLBufferFromMTLHeap: offset 198400 + size 16384 exceeds heap total 212992
    GPUMemrefOps.mm:687: failed assertion 'Failed to acquire the source buffer for the ViewOp'

The 16384 B buffer is the full layer's Q ``[1, 16, 1, 512]`` fp16 (16 heads x global_head_dim
512); MPSGraph's **SDPA lowering** materialises a ViewOp on it that overflows the ~208 KB decode
scratch heap. Sliding layers (Q ``[1,16,1,256]`` = 8 KB) and the E2B/E4B full layers (8 heads x
512 = 8 KB) fit; only the 12B's 16x512 Q tips it. Every model-side SDPA tweak (KV pad<->replicate,
narrow, ``.contiguous()``, ``USE_HF_IMPL``) left the heap/offset/size identical, because they all
kept MPSGraph's SDPA op. **Replacing the SDPA OP with a custom Metal kernel removes that
ViewOp/scratch alloc entirely**, which is the one fix that is ours to make (the alternative —
resizing the engine's scratch heap — is Apple's).

What this kernel computes (FULL/global layers only, q=1 decode)
--------------------------------------------------------------
A full layer is **global causal** with a single global KV head (``num_global_key_value_heads`` 1,
``attention_k_eq_v`` so V == raw k_proj). In the unified-cache decode core
(:mod:`gemma4_dense_pipelined`) that one KV head is REPLICATED across the cache's ``n_kv_max`` (8)
slots, so we read slot 0 (the real head) and all 16 query heads attend to it (GQA 16:1). A q=1
decode step is at the newest position, so causal attention covers **every** grown key — there is
no mask. Attention scale is 1.0 (QK-norm bounds the magnitudes; gemma4 has no attn-logit softcap).

The MSL is the proven flash math of ``gemma4_metal_mlp.build_sdpa_decode_kernel`` adapted from the
host-cache ``cat(past,cur)`` layout to the single grown cache: one SIMD-group (32 lanes) per query
head, each lane owns ``ept = head_dim/32 = 16`` contiguous dims, ``simd_sum`` for the full q.k dot,
fp32 online (running max/sum) softmax — no threadgroup memory, no barrier. Output context is
head-major ``[n_heads, head_dim]`` so the caller's ``permute/reshape -> o_proj`` is unchanged.

Boundary: purely additive. ``metalize_full_sdpa`` swaps ONLY the full layers' ``self_attn.sdpa``
for :class:`MetalDenseFullSDPA` (same ``query=/key=/value=`` call signature as the composite SDPA,
same ``[1,H,1,hd]`` output) — sliding layers keep MPSGraph SDPA (they carry the window mask and do
not crash). Register the returned kernel with the converter (``export_to_coreai_with_kernels``).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

# head_dim/32 upper bound (global_head_dim 512 -> 16); per-lane register arrays sized to this.
_MAX_EPT = 16

# One SIMD-group (32 lanes) per query head h (grid.y); lane (grid.x) owns ept contiguous head dims.
# DSL axes are reversed vs torch: A torch [n_heads, hd] -> A[d, h] = torch q[h, d]; K/V torch
# [nkv, S, hd] -> K[d, j, kv] = torch k[kv, j, d]; CTX torch [n_heads, hd] -> CTX[d, h] = torch ctx[h, d].
# GQA over the grown unified cache: query head h reads KV slot ``h / (H / nkv)`` (block mapping, the
# stock SDPA convention) — the cache's nkv slots are the layer's real global KV heads replicated by
# repeat_interleave (single-head 12B -> all slots identical -> GQA H:1; 4-head 31B -> GQA H:4). Global
# causal q=1 decode attends ALL keys (no mask); scale = 1.0; fp32 online softmax (gemma4's large
# activations make fp16 softmax accumulation lossy).
_DENSE_FULL_SDPA_SRC = """
    const uint hd   = A.get_extent(0);     // A = q, torch [n_heads, head_dim] -> innermost = head_dim
    const uint H    = A.get_extent(1);     // query heads
    const uint S    = K.get_extent(1);     // K torch [nkv, S, head_dim] -> extent(1) = grown seq length
    const uint nkv  = K.get_extent(2);     // KV slots in the (replicated) grown cache
    const uint lane = gid.x;               // 0..31 within the simd-group
    const uint h    = gid.y;               // query head (one simd-group each)
    const uint ept  = hd / 32;             // contiguous head dims owned by this lane (<= __MAXEPT__)
    const uint kv   = h / (H / nkv);       // GQA: query head -> KV slot (block mapping)

    float qx[__MAXEPT__];                   // this lane's slice of q[h] (scale = 1.0)
    for (uint i = 0; i < ept; ++i) qx[i] = float(A[lane * ept + i, h]);

    float o[__MAXEPT__];                    // unnormalised output accumulator (this lane's dims)
    for (uint i = 0; i < ept; ++i) o[i] = 0.0f;
    float m = -1e30f;                       // running max score
    float l = 0.0f;                         // running sum of exp

    for (uint j = 0; j < S; ++j) {
        float p = 0.0f;
        for (uint i = 0; i < ept; ++i) {
            uint d = lane * ept + i;
            p += qx[i] * float(K[d, j, kv]);    // K[d,j,kv] = torch k[kv,j,d]
        }
        float s = simd_sum(p);              // full q.k_j (broadcast to all lanes), scale 1.0, no mask
        float mnew = max(m, s);
        float corr = exp(m - mnew);         // rescale prior accumulators
        float e    = exp(s - mnew);         // this key's (unnormalised) weight
        l = l * corr + e;
        for (uint i = 0; i < ept; ++i) {
            uint d = lane * ept + i;
            o[i] = o[i] * corr + e * float(V[d, j, kv]);   // V[d,j,kv] = torch v[kv,j,d]
        }
        m = mnew;
    }
    float inv = 1.0f / l;
    for (uint i = 0; i < ept; ++i) CTX[lane * ept + i, h] = TYPE(o[i] * inv);  // CTX[d,h]=torch ctx[h,d]
"""


def _dense_full_sdpa_torch_defn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reference (fp32) for the flash-decode kernel: scale-1.0 unmasked GQA softmax over the cache.

    Shapes: ``q [H, hd]``; ``k, v [nkv, S, hd]`` (the grown cache's KV slots). Query head ``h`` attends
    KV slot ``h // (H // nkv)`` (block GQA). Plain (non-online) fp32 softmax == the kernel's online
    softmax; returns context ``[H, hd]`` in q's dtype.
    """
    H, hd = q.shape[0], q.shape[1]
    nkv = k.shape[0]
    rep = H // nkv
    kk = k.repeat_interleave(rep, dim=0)          # [H, S, hd] — query h -> kv slot h // rep
    vv = v.repeat_interleave(rep, dim=0)
    scores = torch.einsum("hd,hsd->hs", q.float(), kk.float())   # [H, S], scale = 1.0, no mask
    w = torch.softmax(scores, dim=-1)
    return torch.einsum("hs,hsd->hd", w, vv.float()).to(q.dtype)  # [H, hd]


def build_dense_full_sdpa_kernel(name: str = "gemma4_dense_full_sdpa") -> TorchMetalKernel:
    """Fused q=1 flash-decode SDPA for the dense Gemma4 FULL layers (GQA over the cache, no mask, scale 1.0)."""
    return TorchMetalKernel(
        name, input_names=["A", "K", "V"], result_names=["CTX"],
        src=_DENSE_FULL_SDPA_SRC.replace("__MAXEPT__", str(_MAX_EPT)),
        torch_defn=_dense_full_sdpa_torch_defn,
        metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


# --- Higher-occupancy variant (FlashDecoding-style G-way sequence split) -----------------------------
# The simple kernel uses ONE SIMD-group per query head, looping serially over all S grown keys. The full
# (global) layers read EVERY grown key (no window), so as the context grows their serial KV loop dominates
# decode (measured: 12B int8 decode 28.2 tok/s at g=64 -> 17.5 at g=1024 = ~38% drop, the global layers'
# unbounded KV scan). This variant raises occupancy ~G× by splitting the sequence across G SIMD-groups per
# head WITHIN one threadgroup: sub-group g scans the strided keys j = g, g+G, g+2G, ... (data-independent —
# no chunk arithmetic on the dynamic S, so the torch_defn is still the plain full-attention reference and
# the export traces), each keeps its own online-softmax partial, and sub-group 0 merges the G partials
# through threadgroup memory. Same math, same result as the simple kernel — just more SIMD-groups in flight
# to hide the KV-read latency. Threadgroup layout: grid (32, G*H, 1), tg (32, G, 1) -> H threadgroups (one
# per query head), each G SIMD-groups; tid.x = lane (0..31), tid.y = g (0..G-1), tgid.y = h.
_HD_MAX = 512  # global_head_dim — sizes the per-sub-group output staging in threadgroup memory
_DENSE_FULL_SDPA_OCC_SRC = """
    const uint hd  = A.get_extent(0);      // A = q, torch [n_heads, head_dim]
    const uint H   = A.get_extent(1);      // query heads
    const uint S   = K.get_extent(1);      // K torch [nkv, S, head_dim] -> extent(1) = grown seq length
    const uint nkv = K.get_extent(2);
    const uint G   = __G__;                // SIMD-groups per head (sequence-split factor)
    const uint lane = tid.x;               // 0..31
    const uint g    = tid.y;               // 0..G-1 : this head's sub-group
    const uint h    = tgid.y;              // query head (one threadgroup per head)
    const uint ept  = hd / 32;
    const uint kv   = h / (H / nkv);       // block GQA

    float qx[__MAXEPT__];
    for (uint i = 0; i < ept; ++i) qx[i] = float(A[lane * ept + i, h]);

    float o[__MAXEPT__];                    // this sub-group's unnormalised output (this lane's dims)
    for (uint i = 0; i < ept; ++i) o[i] = 0.0f;
    float m = -1e30f, l = 0.0f;
    for (uint j = g; j < S; j += G) {       // G-way strided keys (data-independent loop)
        float p = 0.0f;
        for (uint i = 0; i < ept; ++i) { uint d = lane * ept + i; p += qx[i] * float(K[d, j, kv]); }
        float s = simd_sum(p);
        float mnew = max(m, s);
        float corr = exp(m - mnew);
        float e    = exp(s - mnew);
        l = l * corr + e;
        for (uint i = 0; i < ept; ++i) o[i] = o[i] * corr + e * float(V[lane * ept + i, j, kv]);
        m = mnew;
    }

    threadgroup float sm[__G__];            // per-sub-group running max / sum / output
    threadgroup float sl[__G__];
    threadgroup float so[__G__ * __HDMAX__];
    if (lane == 0) { sm[g] = m; sl[g] = l; }
    for (uint i = 0; i < ept; ++i) so[g * hd + lane * ept + i] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (g == 0) {                           // sub-group 0 merges the G partials (online-softmax)
        float gm = -1e30f;
        for (uint gg = 0; gg < G; ++gg) gm = max(gm, sm[gg]);
        float gl = 0.0f;
        for (uint gg = 0; gg < G; ++gg) gl += sl[gg] * exp(sm[gg] - gm);
        float inv = 1.0f / gl;
        for (uint i = 0; i < ept; ++i) {
            uint d = lane * ept + i;
            float acc = 0.0f;
            for (uint gg = 0; gg < G; ++gg) acc += so[gg * hd + d] * exp(sm[gg] - gm);
            CTX[d, h] = TYPE(acc * inv);
        }
    }
"""


def build_dense_full_sdpa_occ_kernel(split_g: int = 8, name: str | None = None) -> TorchMetalKernel:
    """Higher-occupancy flash-decode SDPA: ``split_g`` SIMD-groups per head split the sequence scan.

    Same numerics as :func:`build_dense_full_sdpa_kernel` (the reference torch_defn is identical — the
    G-way split is internal), just ~G× more SIMD-groups in flight to hide the global layers' KV-read
    latency at long context."""
    if name is None:
        name = f"gemma4_dense_full_sdpa_occ{split_g}"
    src = (_DENSE_FULL_SDPA_OCC_SRC.replace("__MAXEPT__", str(_MAX_EPT))
           .replace("__HDMAX__", str(_HD_MAX)).replace("__G__", str(split_g)))
    return TorchMetalKernel(
        name, input_names=["A", "K", "V"], result_names=["CTX"], src=src,
        torch_defn=_dense_full_sdpa_torch_defn,   # G-way split is internal -> same reference
        metal_params=[
            MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
            MetalParameter("tgid", "uint2", "threadgroup_position_in_grid"),
        ],
        template_dtypes={"A": "TYPE"},
    )


class MetalDenseFullSDPA(nn.Module):
    """Drop-in for a FULL layer's composite ``SDPA`` (same ``query=/key=/value=`` call, q=1 decode).

    ``forward(query [1,H,1,hd], key [1,nkv,S,hd], value [1,nkv,S,hd]) -> context [1,H,1,hd]``. The
    grown unified cache holds the layer's real global KV heads replicated (``repeat_interleave``) across
    its ``nkv`` slots; the kernel does block GQA (query head ``h`` -> slot ``h // (H // nkv)``), so this
    is correct for one global KV head (12B, GQA H:1) and for several (31B, GQA H:4). Returns the SAME
    ``[1,H,1,hd]`` shape MPSGraph's SDPA returns, so the caller's ``permute/reshape -> o_proj`` is untouched.
    """

    # Hold no quantizable params; opt out of composite externalization like the pipelined core.
    coreai_externalize_specs: tuple = ()

    def __init__(self, kernel: TorchMetalKernel, split_g: int | None = None) -> None:
        super().__init__()
        self.kernel = kernel
        self.split_g = split_g   # None -> simple 1-SIMD-group/head; G -> G-way sequence-split kernel

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        b, h, s, hd = query.shape          # b=1, s=1 (decode)
        nkv = key.shape[1]                 # KV slots in the grown cache
        seq = key.shape[2]                 # grown seq length S
        q2 = query.reshape(h, hd).contiguous()        # [H, hd]
        k2 = key.reshape(nkv, seq, hd).contiguous()   # [nkv, S, hd]
        v2 = value.reshape(nkv, seq, hd).contiguous()
        if self.split_g is None:
            grid = (32, h, 1)
            tg = (32, 1, 1)
        else:                               # G-way: H threadgroups (one/head) of (32, G) threads
            grid = (32, self.split_g * h, 1)
            tg = (32, self.split_g, 1)
        ctx = self.kernel(q2, k2, v2, threads_per_grid=grid,
                          threads_per_thread_group=tg, result_shapes=[[h, hd]])
        return ctx.reshape(b, h, s, hd)


def metalize_full_sdpa(core: nn.Module, kernel: TorchMetalKernel | None = None,
                       split_g: int | None = None) -> TorchMetalKernel:
    """Swap every FULL (global) layer's ``self_attn.sdpa`` for the custom flash-decode kernel.

    ``core`` is a :class:`gemma4_dense_pipelined.Gemma4DensePipelinedForCausalLM` (``core.model.layers``
    holds the decoder). Sliding layers keep MPSGraph SDPA (window mask, no crash). ``split_g`` selects the
    higher-occupancy G-way sequence-split kernel (``split_g`` SIMD-groups per head) instead of the simple
    1-SIMD-group kernel — same numerics, faster at long context. Returns the shared kernel — register it
    with the converter before exporting (``export_to_coreai_with_kernels``).
    """
    if kernel is None:
        kernel = (build_dense_full_sdpa_occ_kernel(split_g) if split_g
                  else build_dense_full_sdpa_kernel())
    n = 0
    for layer in core.model.layers:
        attn = layer.self_attn
        if getattr(attn, "is_full", False):
            attn.sdpa = MetalDenseFullSDPA(kernel, split_g=split_g)
            n += 1
    if n == 0:
        raise RuntimeError("metalize_full_sdpa: no full-attention layers found (nothing swapped)")
    return kernel
