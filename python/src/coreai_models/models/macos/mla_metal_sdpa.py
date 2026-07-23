# Community port — NOT an Apple model.
"""Absorbed-MLA flash-decode Metal kernel — the speed lever for the DeepSeek/MLA family.

Absorbed MLA (``glm4_moe_lite_absorbed.py``) attends, per head, a query
``[ql (kv_lora) ++ q_rope (qk_rope)]`` against ONE shared key ``c_kv ++ k_rope`` and
ONE shared value ``c_kv`` (MQA over the compressed latent). With the combined cache it
is ONE state ``kv = [c_kv (kv_lora) ++ k_rope (qk_rope)]`` (576 for GLM/DeepSeek-V2-Lite):
the latent slice ``[:kv_lora]`` is both the nope-key AND the value, the rope slice
``[kv_lora:]`` is a score-only key. Two reasons MPSGraph's fused SDPA can't do it:
  * K (576) != V (512): a stock SDPA wants one head_dim; padding V to 576 wastes the
    decode scratch heap and head_dim 576 > 512 trips the #27 ViewOp overflow.
  * the eager fallback (materialise ``[H, S]`` scores, softmax, re-read the latent for
    the weighted sum) makes multiple passes over the grown cache.

THE MQA WIN — cross-head threadgroup staging
--------------------------------------------
The first kernel here (a per-head flash-decode ported from ``gemma4_dense_metal_sdpa``)
was correct but SLOWER than naive: each of the H heads independently re-read the shared
latent from global memory, so the decode read was ``H*S*576`` — MORE than naive's
materialized ``H*S*512``, not the 17.8x-smaller cache storage. The latent is shared
across heads (MQA), so the read only shrinks if the kernel **stages each KV tile in
threadgroup memory and reuses it across all H heads**.

This is a 2-pass (split-K / FlashDecoding) kernel:

* **Main** (:func:`build_mla_staged_main_kernel`): ``grid=(32, H, G)``, ``tg=(32, H, 1)``
  -> G threadgroups, each holding ALL H heads (one simd-group per head, ``tid.y``).
  Threadgroup ``g`` (``tgid.z``) owns the strided key set ``j = g, g+G, g+2G, ...``. It
  loops that set in tiles of ``T`` rows: all ``32*H`` threads **cooperatively load** the
  ``[T, 576]`` tile into threadgroup memory ONCE, barrier, then each head's simd-group
  reads the tile from threadgroup memory and online-softmax-accumulates its partial
  ``(m, l, o[kv_lora])``. => total global latent read = ``S*576`` ONCE (shared across
  heads), not ``H*S*576``. Each threadgroup writes its per-head partial to
  ``P[H, G, kv_lora+2]`` (``[:kv_lora]``=o, ``kv_lora``=m, ``kv_lora+1``=l), fp32.
* **Merge** (:func:`build_mla_staged_merge_kernel`): ``grid=(32, H)`` — each head combines
  its G partials by online-softmax (``M=max m_g``, ``L=sum l_g e^{m_g-M}``,
  ``o=sum o_g e^{m_g-M}/L``) -> ``ctx[H, kv_lora]`` fp16.

The seq-split is mandatory: a single-threadgroup-all-heads design (G=1) would stage the
latent perfectly but run on ONE GPU core (terrible occupancy). G threadgroups (G≈24-40)
fill the ~40-core GPU; the cross-threadgroup merge is the price (a tiny second pass).

Config-driven: ``kv_lora``/``qk_rope`` and the scale are baked per model (GLM 256^-0.5,
DeepSeek-V2-Lite 192^-0.5), so the same kernel pair serves both without a code change.

Boundary: purely additive. ``metalize_mla`` swaps each absorbed attention's ``mla_core``
(an ``EagerMLACore``) for :class:`MetalMLACore` (same ``forward(q_full, kv, q_offset)``
-> ctx contract, q=1 decode). Register the returned kernel(s) with the converter
(``export_to_coreai_with_kernels``).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

# Per-lane register bounds: kv_lora/32 and qk_rope/32. GLM & DeepSeek-V2-Lite both have
# kv_lora 512 (->16) and qk_rope 64 (->2); bump if a config exceeds these.
_MAX_EPTL = 16   # kv_lora / 32
_MAX_EPTR = 2    # qk_rope / 32
_MAX_LAT = 512   # kv_lora upper bound (sizes the occ value staging)
_MAX_G = 64      # split-G upper bound (sizes the merge's per-chunk weight register array)
_STAGE_TILE = 16  # rows per cooperatively-staged KV tile (T*576*2B = 18 KB tg-mem < 32 KB)


def _fmt_scale(scale: float) -> str:
    return f"{scale:.10g}f"


# =================================================================================================== #
# Per-head flash-decode (the ORIGINAL kernel — kept for A/B; it LOSES because it does not stage). One
# SIMD-group (32 lanes) per query head h (grid.y); lane (grid.x) owns eptL=kv_lora/32 contiguous latent
# dims AND eptR=qk_rope/32 contiguous rope dims. Q torch [H, kv_lora+qk_rope] laid out [ql | q_rope];
# KV torch [S, kv_lora+qk_rope] laid out [c_kv | k_rope] (shared latent key/value ++ shared rope key).
# DSL reverses axes: A[d,h]=torch q[h,d], KV[d,j]=torch kv[j,d]. MQA: every head reads the same KV (no
# kv index). q=1 decode attends ALL grown keys (no mask). fp32 online softmax; scale baked.
# =================================================================================================== #
_MLA_PERHEAD_SRC = """
    const uint LAT  = __LAT__;
    const uint S    = KV.get_extent(1);    // grown seq length
    const uint H    = A.get_extent(1);     // query heads (A = q, torch [H, lat+rope])
    const uint lane = gid.x;               // 0..31
    const uint h    = gid.y;               // query head (one simd-group each)
    const uint eptL = __EPTL__;            // latent dims/lane
    const uint eptR = __EPTR__;            // rope dims/lane
    const float scale = __SCALE__;

    float ql[__MAXEPTL__];                  // this lane's latent-query slice
    for (uint i = 0; i < eptL; ++i) ql[i] = float(A[lane * eptL + i, h]);
    float qr[__MAXEPTR__];                  // this lane's rope-query slice (after the latent in A)
    for (uint i = 0; i < eptR; ++i) qr[i] = float(A[LAT + lane * eptR + i, h]);

    float o[__MAXEPTL__];                   // unnormalised latent context (this lane's dims)
    for (uint i = 0; i < eptL; ++i) o[i] = 0.0f;
    float m = -1e30f, l = 0.0f;

    for (uint j = 0; j < S; ++j) {
        float p = 0.0f;
        for (uint i = 0; i < eptL; ++i) p += ql[i] * float(KV[lane * eptL + i, j]);        // nope
        for (uint i = 0; i < eptR; ++i) p += qr[i] * float(KV[LAT + lane * eptR + i, j]);   // rope
        float s = simd_sum(p) * scale;      // full q.k_j, broadcast to all lanes, no mask
        float mnew = max(m, s);
        float corr = exp(m - mnew);
        float e    = exp(s - mnew);
        l = l * corr + e;
        for (uint i = 0; i < eptL; ++i) o[i] = o[i] * corr + e * float(KV[lane * eptL + i, j]);  // value=c_kv
        m = mnew;
    }
    float inv = 1.0f / l;
    for (uint i = 0; i < eptL; ++i) CTX[lane * eptL + i, h] = TYPE(o[i] * inv);  // CTX[d,h]=torch ctx[h,d]
"""


def _mla_perhead_torch_defn(scale: float, lat: int):
    """Reference (fp32): scale*(ql.c_kv + q_rope.k_rope), softmax, latent-value readout.
    q [H, lat+rope]; kv [S, lat+rope] -> ctx [H, lat]."""

    def defn(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        ql = q[:, :lat].float()
        qr = q[:, lat:].float()
        ck = kv[:, :lat].float()
        kr = kv[:, lat:].float()
        scores = (ql @ ck.t() + qr @ kr.t()) * scale   # [H, S]
        w = torch.softmax(scores, dim=-1)
        return (w @ ck).to(q.dtype)                     # [H, lat]

    return defn


def build_mla_perhead_kernel(scale: float, lat: int, rope: int,
                             name: str = "mla_absorbed_perhead") -> TorchMetalKernel:
    """Fused q=1 absorbed-MLA flash-decode, ONE SIMD-group per head (no staging — A/B baseline)."""
    src = (_MLA_PERHEAD_SRC.replace("__LAT__", str(lat)).replace("__EPTL__", str(lat // 32))
           .replace("__EPTR__", str(rope // 32)).replace("__MAXEPTL__", str(_MAX_EPTL))
           .replace("__MAXEPTR__", str(_MAX_EPTR)).replace("__SCALE__", _fmt_scale(scale)))
    return TorchMetalKernel(
        name, input_names=["A", "KV"], result_names=["CTX"], src=src,
        torch_defn=_mla_perhead_torch_defn(scale, lat),
        metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )


# =================================================================================================== #
# THE LEVER — cross-head threadgroup-staging flash-decode (2-pass split-K).
#
# MAIN kernel: grid (32, H, G), tg (32, H, 1) -> G threadgroups, each holds ALL H heads (head = tid.y =
# one simd-group, lane = tid.x). Threadgroup g (= tgid.z) owns the strided keys j = g, g+G, g+2G, ...
# It loops that key set in tiles of T rows: all 32*H threads cooperatively load the [T, 576] tile into
# threadgroup memory ONCE (coalesced — each 576-row is contiguous), barrier, then each head's simd-group
# reads the tile FROM THREADGROUP MEMORY (no per-head global re-read) and online-softmax-accumulates its
# partial (m, l, o[LAT]). => total global latent read = S*576 ONCE, shared across heads (the MQA win),
# not H*S*576. The cache is two EQUAL halves KA/KB ([.,288] each, A=combined[:288], B=combined[288:]) —
# the load stitches them back into the [576] tile row (d<HALF -> KA[d], else KB[d-HALF]) so reading the
# two halves directly (no per-token cat) keeps the read at S*576. The per-(head, chunk) partial is
# written to P[H, G, LAT+2] fp32: [:LAT]=o, LAT=m, LAT+1=l. (Empty chunk, when S<G: rows_g=0 -> the loop
# never runs -> m stays -1e30, l=o=0; the merge's exp(m - global_max) ~ exp(-1e30) zeros it.)
# =================================================================================================== #
_MLA_STAGED_MAIN_SRC = """
    const uint LAT  = __LAT__;
    const uint ROPE = __ROPE__;
    const uint TOT  = __TOT__;              // LAT + ROPE (the reassembled feature dim)
    const uint HALF = __HALF__;             // TOT / 2 (each stored cache half)
    const uint S    = KA.get_extent(1);     // grown seq length
    const uint H    = A.get_extent(1);      // query heads
    const uint G    = __G__;                // seq-split factor (# threadgroups)
    const uint T    = __T__;                // rows per staged tile
    const uint eptL = __EPTL__;
    const uint eptR = __EPTR__;
    const float scale = __SCALE__;
    const uint lane = tid.x;                // 0..31
    const uint h    = tid.y;                // 0..H-1 (this thread's head == its simd-group)
    const uint g    = tgid.z;               // 0..G-1 (this threadgroup's strided key chunk)
    const uint flat = h * 32u + lane;       // 0..32*H-1 (linear thread id, for cooperative loads)
    const uint NT   = 32u * H;              // threads per threadgroup

    float ql[__MAXEPTL__];
    for (uint i = 0; i < eptL; ++i) ql[i] = float(A[lane * eptL + i, h]);
    float qr[__MAXEPTR__];
    for (uint i = 0; i < eptR; ++i) qr[i] = float(A[LAT + lane * eptR + i, h]);

    float o[__MAXEPTL__];
    for (uint i = 0; i < eptL; ++i) o[i] = 0.0f;
    float m = -1e30f, l = 0.0f;

    threadgroup KVT tile[__T__ * __TOT__];  // staged KV tile [T, 576], reused by all H heads

    // rows this threadgroup owns: j = g, g+G, ... < S  -> rows_g of them, in ntiles tiles of T.
    uint rows_g = (g < S) ? ((S - 1u - g) / G + 1u) : 0u;
    uint ntiles = (rows_g + T - 1u) / T;
    for (uint t = 0; t < ntiles; ++t) {
        uint base = t * T;                  // first local row index of this tile
        threadgroup_barrier(mem_flags::mem_threadgroup);   // prior tile's readers done
        for (uint idx = flat; idx < T * TOT; idx += NT) {  // cooperative load (stitch KA||KB per row)
            uint r = idx / TOT;
            uint d = idx % TOT;
            uint local = base + r;
            uint j = g + local * G;
            tile[idx] = (local < rows_g) ? ((d < HALF) ? KA[d, j] : KB[d - HALF, j]) : KVT(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // tile loaded before any head reads it
        for (uint r = 0; r < T; ++r) {
            uint local = base + r;
            if (local >= rows_g) break;     // uniform across the threadgroup (rows_g, base uniform)
            uint row = r * TOT;
            float p = 0.0f;
            for (uint i = 0; i < eptL; ++i) p += ql[i] * float(tile[row + lane * eptL + i]);
            for (uint i = 0; i < eptR; ++i) p += qr[i] * float(tile[row + LAT + lane * eptR + i]);
            float s = simd_sum(p) * scale;
            float mnew = max(m, s);
            float corr = exp(m - mnew);
            float e    = exp(s - mnew);
            l = l * corr + e;
            for (uint i = 0; i < eptL; ++i) o[i] = o[i] * corr + e * float(tile[row + lane * eptL + i]);
            m = mnew;
        }
    }
    // write this (head, chunk) partial to P[H, G, LAT+2]  (DSL P[c,g,h] = torch P[h,g,c])
    for (uint i = 0; i < eptL; ++i) P[lane * eptL + i, g, h] = o[i];
    if (lane == 0) { P[LAT, g, h] = m; P[LAT + 1u, g, h] = l; }
"""


def _mla_staged_main_torch_defn(scale: float, lat: int, split_g: int):
    """Reference (fp32) for the MAIN pass: per-(head, strided-chunk) online-softmax partials.
    q [H, lat+rope]; ka,kb [S, (lat+rope)//2] (the two stored halves) -> P [H, G, lat+2]
    ([:lat]=unnormalised o, lat=max, lat+1=sumexp). Composes with the merge reference to the full
    absorbed attention (and that composition, not the raw partials, is what is gated — empty chunks
    get weight exp(m-global_max)~0 in the merge either way)."""
    G = split_g

    def defn(q: torch.Tensor, ka: torch.Tensor, kb: torch.Tensor) -> torch.Tensor:
        H, S = q.shape[0], ka.shape[0]
        kv = torch.cat([ka, kb], dim=-1)                      # [S, lat+rope]
        ql = q[:, :lat].float()
        qr = q[:, lat:].float()
        ck = kv[:, :lat].float()
        kr = kv[:, lat:].float()
        scores = (ql @ ck.t() + qr @ kr.t()) * scale          # [H, S]
        cid = torch.arange(S, device=q.device) % G            # [S] strided chunk id
        gids = torch.arange(G, device=q.device)               # [G]
        onehot = (cid.unsqueeze(0) == gids.unsqueeze(1)).to(scores.dtype)  # [G, S]
        addmask = (onehot - 1.0) * 1e30                        # 0 in-chunk, -1e30 else  [G, S]
        sm = scores.unsqueeze(1) + addmask.unsqueeze(0)        # [H, G, S]
        m = sm.amax(dim=2)                                     # [H, G]
        e = torch.exp(sm - m.unsqueeze(-1))                    # [H, G, S]
        ll = e.sum(dim=2)                                      # [H, G]
        o = torch.einsum("hgs,sl->hgl", e, ck)                 # [H, G, lat]
        return torch.cat([o, m.unsqueeze(-1), ll.unsqueeze(-1)], dim=-1).float()  # [H, G, lat+2]

    return defn


def build_mla_staged_main_kernel(scale: float, lat: int, rope: int, split_g: int,
                                 tile: int = _STAGE_TILE,
                                 name: str | None = None) -> TorchMetalKernel:
    """MAIN pass of the staged absorbed-MLA flash-decode (cross-head tg-staged, G-way seq-split).
    Reads the two stored cache halves KA/KB ([.,(lat+rope)//2] each) and stitches them per tile row."""
    if name is None:
        name = f"mla_staged_main_g{split_g}"
    tot = lat + rope
    src = (_MLA_STAGED_MAIN_SRC.replace("__LAT__", str(lat)).replace("__ROPE__", str(rope))
           .replace("__TOT__", str(tot)).replace("__HALF__", str(tot // 2))
           .replace("__G__", str(split_g)).replace("__T__", str(tile))
           .replace("__EPTL__", str(lat // 32)).replace("__EPTR__", str(rope // 32))
           .replace("__MAXEPTL__", str(_MAX_EPTL)).replace("__MAXEPTR__", str(_MAX_EPTR))
           .replace("__SCALE__", _fmt_scale(scale)))
    return TorchMetalKernel(
        name, input_names=["A", "KA", "KB"], result_names=["P"], src=src,
        torch_defn=_mla_staged_main_torch_defn(scale, lat, split_g),
        metal_params=[
            MetalParameter("tid", "uint3", "thread_position_in_threadgroup"),
            MetalParameter("tgid", "uint3", "threadgroup_position_in_grid"),
        ],
        template_dtypes={"KA": "KVT"},
    )


# MERGE kernel: grid (32, H), tg (32, 1) — one simd-group per head merges its G partials by online
# softmax. lane owns eptL = LAT/32 latent dims. Reads P[H, G, LAT+2] fp32, writes CTX[H, LAT] fp16.
_MLA_STAGED_MERGE_SRC = """
    const uint LAT  = __LAT__;
    const uint G    = __G__;
    const uint eptL = __EPTL__;
    const uint lane = gid.x;                // 0..31
    const uint h    = gid.y;                // query head

    float gm = -1e30f;
    for (uint g = 0; g < G; ++g) gm = max(gm, float(P[LAT, g, h]));   // global max over chunks
    float w[__MAXG__];                       // per-chunk softmax rescale e^{m_g - gm}
    float gl = 0.0f;
    for (uint g = 0; g < G; ++g) { w[g] = exp(float(P[LAT, g, h]) - gm); gl += float(P[LAT + 1u, g, h]) * w[g]; }
    float inv = 1.0f / gl;
    for (uint i = 0; i < eptL; ++i) {
        uint d = lane * eptL + i;
        float acc = 0.0f;
        for (uint g = 0; g < G; ++g) acc += float(P[d, g, h]) * w[g];
        CTX[d, h] = half(acc * inv);         // CTX[d,h] = torch ctx[h,d]
    }
"""


def _mla_staged_merge_torch_defn(lat: int):
    """Reference (fp16) for the MERGE pass: online-softmax combine of the G partials. P [H, G, lat+2]
    -> ctx [H, lat]. (P[:,:,:lat]=o, P[:,:,lat]=m, P[:,:,lat+1]=l.)"""

    def defn(p: torch.Tensor) -> torch.Tensor:
        o = p[:, :, :lat].float()
        m = p[:, :, lat].float()
        ll = p[:, :, lat + 1].float()
        gm = m.amax(dim=1, keepdim=True)            # [H, 1]
        w = torch.exp(m - gm)                        # [H, G]
        gl = (ll * w).sum(dim=1)                      # [H]
        acc = (o * w.unsqueeze(-1)).sum(dim=1)        # [H, lat]
        return (acc / gl.unsqueeze(-1)).to(torch.float16)   # [H, lat]

    return defn


def build_mla_staged_merge_kernel(lat: int, split_g: int,
                                  name: str | None = None) -> TorchMetalKernel:
    """MERGE pass of the staged absorbed-MLA flash-decode (online-softmax combine of G partials)."""
    if name is None:
        name = f"mla_staged_merge_g{split_g}"
    src = (_MLA_STAGED_MERGE_SRC.replace("__LAT__", str(lat)).replace("__G__", str(split_g))
           .replace("__EPTL__", str(lat // 32)).replace("__MAXG__", str(_MAX_G)))
    return TorchMetalKernel(
        name, input_names=["P"], result_names=["CTX"], src=src,
        torch_defn=_mla_staged_merge_torch_defn(lat),
        metal_params=[MetalParameter("gid", "uint2", "thread_position_in_grid")],
    )


class MetalMLACore(nn.Module):
    """Drop-in for ``EagerMLACore`` (absorbed MLA), q=1 decode. ``forward(q_full
    [b,H,1,lat+rope], kv [b,1,Sg,lat+rope], q_offset) -> ctx [b,H,1,lat]``. MQA: the single
    shared latent key/value is broadcast to all heads by the kernel.

    ``staged`` (default) runs the cross-head threadgroup-staged 2-pass kernel (the MQA win);
    otherwise the per-head 1-SIMD-group baseline (the A/B reference that does not stage)."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, kv_lora: int, qk_rope: int, *, staged: bool = True,
                 split_g: int = 32, main_kernel: TorchMetalKernel | None = None,
                 merge_kernel: TorchMetalKernel | None = None,
                 perhead_kernel: TorchMetalKernel | None = None) -> None:
        super().__init__()
        self.kv_lora = kv_lora
        self.qk_rope = qk_rope
        self.staged = staged
        self.split_g = split_g
        self.main_kernel = main_kernel
        self.merge_kernel = merge_kernel
        self.perhead_kernel = perhead_kernel

    def forward(self, q_full: torch.Tensor, ka: torch.Tensor, kb: torch.Tensor,
                q_offset: int = 0) -> torch.Tensor:
        b, H, s, qk = q_full.shape          # b=1, s=1 (decode)
        Sg = ka.shape[2]
        half = ka.shape[-1]
        q2 = q_full.reshape(H, qk).contiguous()      # [H, lat+rope]
        ka2 = ka.reshape(Sg, half).contiguous()      # [Sg, (lat+rope)//2]
        kb2 = kb.reshape(Sg, half).contiguous()
        if self.staged:
            P = self.main_kernel(
                q2, ka2, kb2, threads_per_grid=(32, H, self.split_g),
                threads_per_thread_group=(32, H, 1),
                result_shapes=[[H, self.split_g, self.kv_lora + 2]])
            ctx = self.merge_kernel(
                P, threads_per_grid=(32, H, 1), threads_per_thread_group=(32, 1, 1),
                result_shapes=[[H, self.kv_lora]])
        else:                                         # per-head baseline reads a combined [Sg,576]
            kv2 = torch.cat([ka2, kb2], dim=-1)
            ctx = self.perhead_kernel(
                q2, kv2, threads_per_grid=(32, H, 1),
                threads_per_thread_group=(32, 1, 1), result_shapes=[[H, self.kv_lora]])
        return ctx.reshape(b, H, s, self.kv_lora)


def metalize_mla(causal_lm: nn.Module, scale: float, *, staged: bool = True,
                 split_g: int = 32) -> list[TorchMetalKernel]:
    """Swap every absorbed attention's ``mla_core`` (``EagerMLACore``) for the custom flash-decode
    kernel(s). ``causal_lm`` exposes ``.model.layers`` (a ``Glm4MoeLite[Absorbed]StatefulForCausalLM``).
    ``staged`` selects the cross-head threadgroup-staged 2-pass kernel (the MQA win); else the per-head
    baseline. Returns the shared kernel(s) — register them before exporting.

    Reads ``kv_lora``/``qk_rope`` from the first absorbed attention (uniform across layers)."""
    layers = [layer for layer in causal_lm.model.layers
              if getattr(layer.self_attn, "mla_core", None) is not None]
    if not layers:
        raise RuntimeError("metalize_mla: no absorbed attention (mla_core) found")
    lat = layers[0].self_attn.kv_lora
    rope = layers[0].self_attn.qk_rope
    if staged:
        main_k = build_mla_staged_main_kernel(scale, lat, rope, split_g)
        merge_k = build_mla_staged_merge_kernel(lat, split_g)
        kernels: list[TorchMetalKernel] = [main_k, merge_k]
        for layer in layers:
            layer.self_attn.mla_core = MetalMLACore(
                layer.self_attn.kv_lora, layer.self_attn.qk_rope,
                staged=True, split_g=split_g, main_kernel=main_k, merge_kernel=merge_k)
    else:
        perhead_k = build_mla_perhead_kernel(scale, lat, rope)
        kernels = [perhead_k]
        for layer in layers:
            layer.self_attn.mla_core = MetalMLACore(
                layer.self_attn.kv_lora, layer.self_attn.qk_rope,
                staged=False, perhead_kernel=perhead_k)
    return kernels
