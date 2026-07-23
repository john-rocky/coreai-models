# Community port — NOT an Apple model.
"""Gemma 4 E2B CHUNKED host-cache decode core — the device-ANE port (Session A).

The de-risk (``ondevice/_gemma4_ane_derisk_RESULTS.md``) proved gemma4's dual head_dim
(sliding 256 / full 512) KV LOWERS + RUNS on the iPhone 17 Pro ANE *in host-cache form*
(Session B's ``Gemma4DecodeHostCache``: no Core AI state, no in-graph indexed write — KV
caches are plain I/O, the current token's K/V is ``cat``-appended in-graph, the HOST writes
the returned ``*_cur`` columns into the caches between steps). The 6-layer host-cache core
runs on the ANE; the 35-layer MONOLITH OOMs the ANE first-run compile (jetsam). The only
remaining work for full ANE is **chunking** so each sub-graph fits — exactly what CoreML-LLM
does to ship gemma4 on this ANE at 34 tok/s.

This module splits the 35-layer host-cache core into a few chunks. Each chunk is the SAME
per-layer math as the monolith (it reuses ``models/macos/gemma4_bucketed.py``'s
``_hostcache_layer`` verbatim — composite RoPE + masked SDPA over ``cat``), only partitioned.
So a correct chunk *chain* is numerically identical to the proven 8/8 monolith; the only new
surface is the inter-chunk plumbing, which this file makes explicit and self-describing.

CHUNK TOPOLOGY (mirrors CoreML-LLM's ``compute_chunk_boundaries``; E2B 35 layers, producers
L13 sliding / L14 full → boundaries ``[(0,8),(8,15),(15,25),(25,35)]``):
  * chunk 1 (L0-7)   — own KV: 7 sliding slots + 1 full slot. hidden_in == inputs_embeds.
  * chunk 2 (L8-14)  — own KV: 5 sliding + 2 full. CONTAINS both producers (L13→sliding
                       slot 11, L14→full slot 2); the host extracts their ``*_cur`` columns
                       and feeds them to the consumer chunks.
  * chunk 3 (L15-24) — stateless, all KV-shared: reads ONLY the producer slots (past) + the
                       producer ``*_cur`` (this step).
  * chunk 4 (L25-34) — same as chunk 3; the final RMSNorm is applied here.

INTER-CHUNK DATA FLOW (host-managed, no state):
  * ``hidden``: chunk1 → chunk2 → chunk3 → chunk4 (each chunk's ``hidden`` output is the next
    chunk's ``hidden_in``).
  * own-slot ``*_cur``: each non-shared layer returns its current K/V column; the host writes
    it into the GLOBAL cache at position ``p`` (the ``Gemma4DecodeHostCache`` contract).
  * producer ``*_cur``: the producer layers (L13/L14) live in chunk 2; the host keeps chunk 2's
    producer columns and feeds them — plus the producer slot's host-written history (``past``)
    — to the consumer chunks (CoreML-LLM's chunk2→chunk3/4 kv13/kv14 aliases).

All shapes are STATIC (driven at a fixed bucket B), no Core AI state, no in-graph indexed
write → only MPSGraph-friendly ops. ``models/ios/gemma4_ane.py`` (the single-graph
iOS-primitive port with Core AI KV *state*) stays as the reference for iOS-primitive details;
THIS file is the path that actually executes on the device (host-cache, chunked).

Boundary: reuses ``gemma4_bucketed.py`` (Session B) by IMPORT only — it is not modified.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from coreai_models.models.macos.gemma4_bucketed import (
    MASK_NEG,
    build_hostcache_caches,
    build_hostcache_mask_full,
    build_hostcache_mask_sliding,
)
from coreai_models.models.macos.gemma4_text import (
    FULL,
    SLIDING,
    Gemma4TextConfig,
    Gemma4TextModel,
)

__all__ = [
    "MASK_NEG",
    "FORWARD_PARAMS",
    "Gemma4ChunkHostCache",
    "default_chunk_boundaries",
    "build_chunk_plan",
    "build_hostcache_caches",
    "build_hostcache_mask_full",
    "build_hostcache_mask_sliding",
]

# Canonical forward parameter order (== the forward signature). Used to build a positionally-aligned
# example-inputs tuple for k-means palettization (coreai-opt traces the model with positional args and
# .detach()s each, so None is forbidden — unused params are filled with harmless zero dummies).
FORWARD_PARAMS = (
    "hidden_in", "per_layer_inputs", "position_ids", "causal_mask_full", "causal_mask_sliding",
    "sliding_k", "sliding_v", "full_k", "full_v",
    # KV-shared consumer inputs. EXTERNAL producer → host pre-cat'd [past ++ cur] (no in-graph cat):
    "prod_sliding_kv_k", "prod_sliding_kv_v", "prod_full_kv_k", "prod_full_kv_v",
    # LOCAL producer (rare; producer + consumer in the same chunk) → producer slot past (cur is in-graph):
    "prod_sliding_k", "prod_sliding_v", "prod_full_k", "prod_full_v",
)

# The composite RMSNorm (coreai_torch.composite_ops.RMSNorm) is correct on the GPU but its MPSGraph→ANE
# lowering computes mean(x²) in fp16, which overflows / loses precision for gemma4's LARGE activations
# (per-chunk absmax 11→510 on device) → the hidden magnitude grows and the tokens go wrong. GEMMA4_CHUNK_FP32NORM=1
# swaps every weighted RMSNorm for this explicit fp32-reduction form (the GPU stays 8/8; the ANE gets a
# numerically safe reduction). Mirrors CoreML-LLM's ANE RMSNorm (which uses the ANE's fp32 LayerNorm kernel).
# DEFAULT ON: both are REQUIRED for ANE correctness (proven 8/8 on the iPhone 17 Pro ANE) and are
# GPU-numerically-equivalent (~2e-8 vs the monolith). Set the env to "0" only to A/B the old fp16 paths.
_FP32_NORM = os.environ.get("GEMMA4_CHUNK_FP32NORM", "1") == "1"

# GEMMA4_CHUNK_CONV=1 wraps every nn.Linear projection as Conv2d 1×1 (CoreML-LLM's Conv2dLinear). On the
# ANE the CONV engine accumulates matmuls in fp32, whereas Linear→MPSGraph→ANE appears to accumulate in
# fp16 — which (after the RMSNorm overflow is fixed) leaves a residual drift that compounds over 35 layers
# (per-chunk hidden drifts GPU vs ANE, growing chunk1→chunk6). Same [b,s,d] interface + shared weights.
_CONV = os.environ.get("GEMMA4_CHUNK_CONV", "1") == "1"


class _Conv2dLinear(nn.Module):
    """nn.Linear computed as Conv2d(kernel=1) — ANE conv engine accumulates in fp32. [b,s,in]→[b,s,out]."""

    def __init__(self, lin: nn.Linear) -> None:
        super().__init__()
        dt = lin.weight.dtype
        self.conv = nn.Conv2d(lin.in_features, lin.out_features, kernel_size=1, bias=lin.bias is not None, dtype=dt)
        with torch.no_grad():
            self.conv.weight.copy_(lin.weight.detach().unsqueeze(-1).unsqueeze(-1))
            if lin.bias is not None:
                self.conv.bias.copy_(lin.bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(2)   # [b, s, in] -> [b, in, 1, s]
        x = self.conv(x)                       # [b, out, 1, s]
        return x.squeeze(2).permute(0, 2, 1)   # -> [b, s, out]


def _patch_conv2d(layer) -> None:
    """Swap a DecoderLayer's nn.Linear projections (attn q/k/v/o, MLP, PLE) for Conv2d 1×1 (ANE fp32 MAC)."""
    a = layer.self_attn
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(a, name, _Conv2dLinear(getattr(a, name)))
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(layer.mlp, name, _Conv2dLinear(getattr(layer.mlp, name)))
    layer.per_layer_input_gate = _Conv2dLinear(layer.per_layer_input_gate)
    layer.per_layer_projection = _Conv2dLinear(layer.per_layer_projection)


class _Fp32RMSNorm(nn.Module):
    """gemma4 RMSNorm via the [x, -x] LayerNorm trick (CoreML-LLM's ANE recipe).

    A manual ``rsqrt(mean(x²))`` with a ``.float()`` cast is a NO-OP on the ANE — MPSGraph drops the cast
    and computes the sum-of-squares in fp16, which overflows / loses precision for gemma4's large
    activations. Instead, ``LayerNorm([x, -x])`` has zero mean so it equals RMSNorm, and the ANE runs
    LayerNorm with a HARDWARE fp32-accumulating kernel — the actual way to get fp32 reduction on the ANE.
    """

    def __init__(self, weight: torch.Tensor, eps: float) -> None:
        super().__init__()
        self.weight = weight  # share the checkpoint param (no copy)
        self.eps = eps
        self.dim = weight.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        doubled = torch.cat([x, -x], dim=-1)                       # zero-mean → LayerNorm == RMSNorm
        normed = nn.functional.layer_norm(doubled, (2 * self.dim,), weight=None, bias=None, eps=self.eps)
        normed, _ = torch.chunk(normed, 2, dim=-1)
        return normed * self.weight


def _patch_fp32_norms(layer, eps: float) -> None:
    """Replace a DecoderLayer's weighted composite RMSNorms with fp32-reduction norms (ANE-safe)."""
    for name in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                 "post_feedforward_layernorm", "post_per_layer_input_norm"):
        setattr(layer, name, _Fp32RMSNorm(getattr(layer, name).weight, eps))
    layer.self_attn.q_norm = _Fp32RMSNorm(layer.self_attn.q_norm.weight, eps)
    layer.self_attn.k_norm = _Fp32RMSNorm(layer.self_attn.k_norm.weight, eps)


def _ane_attention(attn, x, position_ids, past_k, past_v, mask, scale, cur_kv=None, kv_full=None):
    """Host-cache attention with a MANUAL decomposed softmax (ANE-correct mask handling).

    Same projections / Q-K-V norms / RoPE / o_proj as the macOS attention (reused from ``attn``), only
    the SDPA core is manual so the additive ``mask`` is honoured on the ANE (the fused composite SDPA's
    attn_mask is ignored by the ANE's native SDPA).

    KV source (three modes):
      * ``kv_full`` given  — a KV-SHARED layer whose producer K/V was concatenated HOST-SIDE into one
        ``(k_full, v_full)`` [1, nkv, B+1, hd] input. NO in-graph ``cat``. This is REQUIRED for the
        device ANE: an in-graph ``cat`` of TWO GRAPH INPUTS (producer past ++ producer cur) is
        miscompiled by MPSGraph→ANE (localised: chunk1 own-KV [input ++ computed] is ANE-correct, but
        the consumer chunks' [input ++ input] cat computes WRONG). CoreML-LLM likewise reads the whole
        producer cache slice rather than concatenating. Returns ``out`` only.
      * ``cur_kv`` given   — a KV-shared layer whose producer is IN THIS chunk (local cur). ``cat`` of
        an input (past) and a computed tensor (cur) — ANE-safe like the own-cache path. Returns ``out``.
      * neither            — own-cache layer: compute this token's K/V, ``cat`` with the input past
        (input ++ computed → ANE-safe), return ``(out, cur_k, cur_v)`` for the host to store.
    """
    b, s, _ = x.shape  # s == 1
    H, hd = attn.n_heads, attn.head_dim
    q = attn.q_proj(x).view(b, s, H, hd).permute(0, 2, 1, 3)   # [1, H, 1, hd]
    q = attn.q_norm(q)
    q = attn.rope(q, position_ids=position_ids, freqs=attn.inv_freq)

    own = cur_kv is None and kv_full is None
    if kv_full is not None:
        k_full, v_full = kv_full                                # host pre-cat'd; no in-graph cat
        cur_k = cur_v = None
    else:
        if own:
            cur_k, cur_v = attn._project_kv(x, position_ids)    # [1, nkv, 1, hd]
        else:
            cur_k, cur_v = cur_kv
        k_full = torch.cat([past_k, cur_k], dim=-2)             # [1, nkv, B+1, hd]
        v_full = torch.cat([past_v, cur_v], dim=-2)

    # Manual attention: GQA broadcasts the single kv head over H query heads.
    scores = torch.matmul(q, k_full.transpose(-1, -2)) * scale  # [1, H, 1, B+1]
    scores = scores + mask                                      # ADDITIVE mask (0 / MASK_NEG); honoured on ANE
    mx = scores.max(dim=-1, keepdim=True).values
    e = torch.exp(scores - mx)
    attn_w = e / e.sum(dim=-1, keepdim=True)
    out = torch.matmul(attn_w, v_full)                          # [1, H, 1, hd]
    out = out.permute(0, 2, 1, 3).reshape(b, s, H * hd)
    out = attn.o_proj(out)
    return (out, cur_k, cur_v) if own else out


def _ane_layer(layer, x, position_ids, per_layer_input, past_k, past_v, mask, scale, cur_kv=None, kv_full=None):
    """DecoderLayer norms / MLP / PLE around :func:`_ane_attention` (mirrors gemma4_bucketed._hostcache_layer)."""
    own = cur_kv is None and kv_full is None
    residual = x
    h = layer.input_layernorm(x)
    a = _ane_attention(layer.self_attn, h, position_ids, past_k, past_v, mask, scale, cur_kv, kv_full)
    if own:
        h, cur_k, cur_v = a
    else:
        h = a
    h = layer.post_attention_layernorm(h)
    x = residual + h

    residual = x
    h = layer.pre_feedforward_layernorm(x)
    h = layer.mlp(h)
    h = layer.post_feedforward_layernorm(h)
    x = residual + h

    residual = x
    h = layer.per_layer_input_gate(x)
    h = nn.functional.gelu(h, approximate="tanh")
    h = h * per_layer_input
    h = layer.per_layer_projection(h)
    h = layer.post_per_layer_input_norm(h)
    x = residual + h
    x = x * layer.layer_scalar
    return (x, cur_k, cur_v) if own else x


def default_chunk_boundaries(cfg: Gemma4TextConfig) -> list[tuple[int, int]]:
    """4 host-cache decode-chunk boundaries, mirroring CoreML-LLM's ``compute_chunk_boundaries``.

    chunk2 must end at ``full_producer + 1`` so it owns (and the host can extract) the shared
    producer KV; chunk1 splits the own-KV region ~in half; chunks 3/4 split the shared region.
    For E2B (35 layers, producers L13/L14) → ``[(0,8),(8,15),(15,25),(25,35)]``.
    """
    n = cfg.num_hidden_layers
    own_end = cfg.producer_idx(FULL) + 1
    c1_end = (own_end + 1) // 2
    c3_end = own_end + (n - own_end) // 2
    return [(0, c1_end), (c1_end, own_end), (own_end, c3_end), (c3_end, n)]


def subdivide_boundaries(boundaries: list[tuple[int, int]], max_layers: int) -> list[tuple[int, int]]:
    """Split any chunk longer than ``max_layers`` into even sub-chunks (the device-ANE OOM contingency).

    The chunk core + plan are boundary-agnostic (producers/consumers are resolved by global layer index,
    not by chunk position), so finer boundaries "just work": a producer landing in some sub-chunk emits its
    ``*_cur`` there and the later consumer sub-chunks read it. Use when a 10-layer shared chunk OOMs the ANE
    first-run compile (the de-risk's proven ANE size is ~6 layers).
    """
    out: list[tuple[int, int]] = []
    for s, e in boundaries:
        n = e - s
        if n <= max_layers:
            out.append((s, e))
            continue
        k = (n + max_layers - 1) // max_layers  # number of even sub-chunks
        base, rem, p = n // k, n % k, s
        for j in range(k):
            sz = base + (1 if j < rem else 0)
            out.append((p, p + sz))
            p += sz
    return out


class Gemma4ChunkHostCache(nn.Module):
    """One host-cache decode chunk over global layers ``[start, end)``.

    Same per-layer computation as ``Gemma4DecodeHostCache`` (reuses ``_hostcache_layer``),
    partitioned. The forward declares ALL possible inputs as ``None``-defaulted keyword args;
    a given chunk only references the subset its routing needs, so torch.export traces only
    those (export is called with ``kwargs=reference_inputs``). Use :meth:`build_export_spec`
    for the per-chunk reference inputs / I/O names, and :func:`build_chunk_plan` for the
    pipeline-level host orchestration (cache slicing, producer-cur routing, host-write map).

    I/O (only the names a chunk uses are present, in this canonical order):
      inputs:  hidden_in[1,1,h], per_layer_inputs[1,1,end-start,ld], position_ids[1,1],
               causal_mask_full[1,1,1,B+1]?, causal_mask_sliding[1,1,1,B+1]?,
               sliding_k[n_own_s,1,nkv,B,256]?, sliding_v?, full_k[n_own_f,1,nkv,B,512]?, full_v?,
               prod_sliding_k[1,nkv,B,256]?, prod_sliding_v?, prod_sliding_k_cur[1,nkv,1,256]?, prod_sliding_v_cur?,
               prod_full_k[1,nkv,B,512]?, prod_full_v?, prod_full_k_cur[1,nkv,1,512]?, prod_full_v_cur?
      outputs: hidden[1,1,h], sliding_k_cur[n_own_s,1,nkv,1,256]?, sliding_v_cur?,
               full_k_cur[n_own_f,1,nkv,1,512]?, full_v_cur?
    """

    coreai_externalize_specs: tuple = ()  # same opt-out as Gemma4DecodeHostCache

    def __init__(self, text_model: Gemma4TextModel, start: int, end: int) -> None:
        super().__init__()
        cfg = text_model.config
        self.cfg = cfg
        self.start, self.end = start, end
        # Hold ONLY this chunk's layers (so the traced graph carries only their weights).
        self.layers = nn.ModuleList([text_model.layers[i] for i in range(start, end)])
        # The final RMSNorm is applied once, by the LAST chunk only.
        self.final_norm = text_model.norm if end == cfg.num_hidden_layers else None
        if _FP32_NORM:  # ANE-safe fp32-reduction RMSNorms (the composite norm overflows fp16 on the ANE)
            for layer in self.layers:
                _patch_fp32_norms(layer, cfg.rms_norm_eps)
            if self.final_norm is not None:
                self.final_norm = _Fp32RMSNorm(self.final_norm.weight, cfg.rms_norm_eps)
        if _CONV:  # Conv2d 1×1 projections → ANE fp32 matmul accumulation (kills the residual depth drift)
            for layer in self.layers:
                _patch_conv2d(layer)

        route, n_sliding, n_full = cfg.stateful_routing()
        self.route = route
        self.prod_s_layer = cfg.producer_idx(SLIDING)
        self.prod_f_layer = cfg.producer_idx(FULL)
        self.prod_s_slot = n_sliding - 1  # sliding producer's global slot
        self.prod_f_slot = n_full - 1     # full producer's global slot

        # Own (written) slots, in layer order == cache slot order. Map global slot -> local idx.
        own_s, own_f = [], []
        for i in range(start, end):
            is_full, slot, write = route[i]
            if write:
                (own_f if is_full else own_s).append(slot)
        self.own_sliding_slots = own_s
        self.own_full_slots = own_f
        self.own_sliding_local = {s: j for j, s in enumerate(own_s)}
        self.own_full_local = {s: j for j, s in enumerate(own_f)}

        # Which attention types appear (→ which masks this chunk consumes).
        self.has_full = any(route[i][0] for i in range(start, end))
        self.has_sliding = any(not route[i][0] for i in range(start, end))
        # Shared (KV-consumer) layers present → this chunk reads the producer slot.
        self.has_shared_sliding = any(not route[i][2] and not route[i][0] for i in range(start, end))
        self.has_shared_full = any(not route[i][2] and route[i][0] for i in range(start, end))
        # Is the producer LAYER inside this chunk (→ producer cur is local), or earlier (→ input)?
        self.sliding_prod_in_chunk = start <= self.prod_s_layer < end
        self.full_prod_in_chunk = start <= self.prod_f_layer < end
        # EXTERNAL producer (the common case): the host pre-concatenates [producer slot history ++ producer
        # cur] into ONE input (prod_<type>_kv_*), so the consumer does NO in-graph cat (which MPSGraph→ANE
        # miscompiles for two graph inputs). LOCAL producer (rare; producer + consumer share a chunk): the
        # consumer reads the producer slot past as an input and cats it with the in-chunk computed cur.
        self.needs_ext_sliding_kv = self.has_shared_sliding and not self.sliding_prod_in_chunk
        self.needs_ext_full_kv = self.has_shared_full and not self.full_prod_in_chunk
        self.needs_local_sliding_past = self.has_shared_sliding and self.sliding_prod_in_chunk
        self.needs_local_full_past = self.has_shared_full and self.full_prod_in_chunk

    # ---- introspection: the canonical, present-only I/O name lists ----
    @property
    def input_names(self) -> tuple[str, ...]:
        names = ["hidden_in", "per_layer_inputs", "position_ids"]
        if self.has_full:
            names.append("causal_mask_full")
        if self.has_sliding:
            names.append("causal_mask_sliding")
        if self.own_sliding_slots:
            names += ["sliding_k", "sliding_v"]
        if self.own_full_slots:
            names += ["full_k", "full_v"]
        if self.needs_ext_sliding_kv:
            names += ["prod_sliding_kv_k", "prod_sliding_kv_v"]
        if self.needs_local_sliding_past:
            names += ["prod_sliding_k", "prod_sliding_v"]
        if self.needs_ext_full_kv:
            names += ["prod_full_kv_k", "prod_full_kv_v"]
        if self.needs_local_full_past:
            names += ["prod_full_k", "prod_full_v"]
        return tuple(names)

    @property
    def output_names(self) -> tuple[str, ...]:
        names = ["hidden"]
        if self.own_sliding_slots:
            names += ["sliding_k_cur", "sliding_v_cur"]
        if self.own_full_slots:
            names += ["full_k_cur", "full_v_cur"]
        return tuple(names)

    def forward(
        self,
        hidden_in,
        per_layer_inputs,
        position_ids,
        causal_mask_full=None,
        causal_mask_sliding=None,
        sliding_k=None,
        sliding_v=None,
        full_k=None,
        full_v=None,
        prod_sliding_kv_k=None,
        prod_sliding_kv_v=None,
        prod_full_kv_k=None,
        prod_full_kv_v=None,
        prod_sliding_k=None,
        prod_sliding_v=None,
        prod_full_k=None,
        prod_full_v=None,
    ):
        x = hidden_in
        local_prod: dict[str, tuple] = {}  # producer cur computed in THIS chunk
        out_sk, out_sv, out_fk, out_fv = [], [], [], []
        for li in range(self.end - self.start):
            i = self.start + li
            layer = self.layers[li]
            is_full, slot, write = self.route[i]
            mask = causal_mask_full if is_full else causal_mask_sliding
            ple_i = per_layer_inputs[:, :, li, :]
            if write:
                # Non-shared: owns this slot; read its host-written history + compute current K/V.
                local_slot = (self.own_full_local if is_full else self.own_sliding_local)[slot]
                pk_cache, pv_cache = (full_k, full_v) if is_full else (sliding_k, sliding_v)
                past_k = pk_cache[local_slot]
                past_v = pv_cache[local_slot]
                # Own-cache: cat(input past ++ in-graph computed cur) — ANE-safe.
                x, cur_k, cur_v = _ane_layer(layer, x, position_ids, ple_i, past_k, past_v, mask, 1.0)
                (out_fk if is_full else out_sk).append(cur_k)
                (out_fv if is_full else out_sv).append(cur_v)
                if i == self.prod_s_layer:
                    local_prod[SLIDING] = (cur_k, cur_v)
                elif i == self.prod_f_layer:
                    local_prod[FULL] = (cur_k, cur_v)
            else:
                # KV-shared. EXTERNAL producer → host pre-cat'd [past ++ cur] in one input, used directly
                # (NO in-graph cat — the device-ANE fix). LOCAL producer (rare) → cat the producer slot
                # past input with the in-chunk computed cur (input ++ computed = ANE-safe).
                lt = FULL if is_full else SLIDING
                if lt in local_prod:                      # producer ran in this chunk → cat with its cur
                    past_k, past_v = (prod_full_k, prod_full_v) if is_full else (prod_sliding_k, prod_sliding_v)
                    x = _ane_layer(layer, x, position_ids, ple_i, past_k, past_v, mask, 1.0,
                                   cur_kv=local_prod[lt])
                else:                                     # external producer → host pre-cat'd KV, no cat
                    kv_full = (prod_full_kv_k, prod_full_kv_v) if is_full else (prod_sliding_kv_k, prod_sliding_kv_v)
                    x = _ane_layer(layer, x, position_ids, ple_i, None, None, mask, 1.0, kv_full=kv_full)

        if self.final_norm is not None:
            x = self.final_norm(x)

        result = [x]
        if self.own_sliding_slots:
            result += [torch.stack(out_sk, dim=0), torch.stack(out_sv, dim=0)]
        if self.own_full_slots:
            result += [torch.stack(out_fk, dim=0), torch.stack(out_fv, dim=0)]
        return tuple(result)

    def build_export_spec(self, target_dtype: torch.dtype, bucket: int) -> dict:
        """Reference inputs for a FULLY STATIC per-chunk export (no state, no Dim), driven at bucket B."""
        cfg = self.cfg
        h, ld = cfg.hidden_size, cfg.hidden_size_per_layer_input
        nkv = cfg.num_key_value_heads
        hs, hf = cfg.head_dim, cfg.global_head_dim
        B = bucket
        n_own_s, n_own_f = len(self.own_sliding_slots), len(self.own_full_slots)

        ref: dict[str, torch.Tensor] = {
            "hidden_in": torch.zeros(1, 1, h, dtype=target_dtype),
            "per_layer_inputs": torch.zeros(1, 1, self.end - self.start, ld, dtype=target_dtype),
            "position_ids": torch.tensor([[0]], dtype=torch.int32),
        }
        if self.has_full:
            ref["causal_mask_full"] = torch.zeros(1, 1, 1, B + 1, dtype=target_dtype)
        if self.has_sliding:
            ref["causal_mask_sliding"] = torch.zeros(1, 1, 1, B + 1, dtype=target_dtype)
        if n_own_s:
            ref["sliding_k"] = torch.zeros(n_own_s, 1, nkv, B, hs, dtype=target_dtype)
            ref["sliding_v"] = torch.zeros(n_own_s, 1, nkv, B, hs, dtype=target_dtype)
        if n_own_f:
            ref["full_k"] = torch.zeros(n_own_f, 1, nkv, B, hf, dtype=target_dtype)
            ref["full_v"] = torch.zeros(n_own_f, 1, nkv, B, hf, dtype=target_dtype)
        if self.needs_ext_sliding_kv:  # host pre-cat'd [past(B) ++ cur(1)] = B+1
            ref["prod_sliding_kv_k"] = torch.zeros(1, nkv, B + 1, hs, dtype=target_dtype)
            ref["prod_sliding_kv_v"] = torch.zeros(1, nkv, B + 1, hs, dtype=target_dtype)
        if self.needs_local_sliding_past:
            ref["prod_sliding_k"] = torch.zeros(1, nkv, B, hs, dtype=target_dtype)
            ref["prod_sliding_v"] = torch.zeros(1, nkv, B, hs, dtype=target_dtype)
        if self.needs_ext_full_kv:
            ref["prod_full_kv_k"] = torch.zeros(1, nkv, B + 1, hf, dtype=target_dtype)
            ref["prod_full_kv_v"] = torch.zeros(1, nkv, B + 1, hf, dtype=target_dtype)
        if self.needs_local_full_past:
            ref["prod_full_k"] = torch.zeros(1, nkv, B, hf, dtype=target_dtype)
            ref["prod_full_v"] = torch.zeros(1, nkv, B, hf, dtype=target_dtype)

        # Order the dict canonically (== input_names) so the exported graph's input order is stable.
        ref = {k: ref[k] for k in self.input_names}
        return {
            "reference_inputs": ref,
            "dynamic_shapes": None,  # FULLY STATIC → one compile, reused every step
            "input_names": self.input_names,
            "output_names": self.output_names,
            "state_names": None,
        }

    def palettize_example_inputs(self, target_dtype: torch.dtype, bucket: int) -> tuple:
        """Positional example-inputs tuple (all forward params, in order) for k-means palettization.

        coreai-opt traces the model with POSITIONAL args and ``.detach()``s each element, so it cannot
        take None. We pass the chunk's real inputs where it uses them and natural-shape zero dummies in
        the gaps (the forward never references the unused ones). Weight k-means is data-free, so the
        dummies do not affect the result. The actual export still uses the kwargs subset from
        :meth:`build_export_spec`.
        """
        cfg = self.cfg
        h, ld, nkv = cfg.hidden_size, cfg.hidden_size_per_layer_input, cfg.num_key_value_heads
        hs, hf, B = cfg.head_dim, cfg.global_head_dim, bucket
        ns, nf = max(len(self.own_sliding_slots), 1), max(len(self.own_full_slots), 1)
        natural = {
            "hidden_in": torch.zeros(1, 1, h, dtype=target_dtype),
            "per_layer_inputs": torch.zeros(1, 1, self.end - self.start, ld, dtype=target_dtype),
            "position_ids": torch.tensor([[0]], dtype=torch.int32),
            "causal_mask_full": torch.zeros(1, 1, 1, B + 1, dtype=target_dtype),
            "causal_mask_sliding": torch.zeros(1, 1, 1, B + 1, dtype=target_dtype),
            "sliding_k": torch.zeros(ns, 1, nkv, B, hs, dtype=target_dtype),
            "sliding_v": torch.zeros(ns, 1, nkv, B, hs, dtype=target_dtype),
            "full_k": torch.zeros(nf, 1, nkv, B, hf, dtype=target_dtype),
            "full_v": torch.zeros(nf, 1, nkv, B, hf, dtype=target_dtype),
            "prod_sliding_kv_k": torch.zeros(1, nkv, B + 1, hs, dtype=target_dtype),
            "prod_sliding_kv_v": torch.zeros(1, nkv, B + 1, hs, dtype=target_dtype),
            "prod_full_kv_k": torch.zeros(1, nkv, B + 1, hf, dtype=target_dtype),
            "prod_full_kv_v": torch.zeros(1, nkv, B + 1, hf, dtype=target_dtype),
            "prod_sliding_k": torch.zeros(1, nkv, B, hs, dtype=target_dtype),
            "prod_sliding_v": torch.zeros(1, nkv, B, hs, dtype=target_dtype),
            "prod_full_k": torch.zeros(1, nkv, B, hf, dtype=target_dtype),
            "prod_full_v": torch.zeros(1, nkv, B, hf, dtype=target_dtype),
        }
        return tuple(natural[name] for name in FORWARD_PARAMS)


def build_chunk_plan(cfg: Gemma4TextConfig, boundaries: list[tuple[int, int]]) -> dict:
    """Pipeline-level host-orchestration plan for a chunked host-cache decode.

    Returns a dict describing how the host drives the chain: per-chunk I/O names, which GLOBAL
    cache slots each chunk owns (for slicing the input cache view and writing back ``*_cur``),
    which producer columns each chunk consumes, and where each producer's current column is
    produced. Drives the Python verify driver and (later) the Swift engine — one source of truth.

    Keys:
      ``prod_s_slot`` / ``prod_f_slot``  global cache slots of the sliding / full producers.
      ``producer_src``  {SLIDING|FULL: (chunk_idx, cur_output_name_k, cur_output_name_v, local_idx)}
                        where the producer's current column lives in that chunk's output.
      ``chunks``  list (one per boundary) of:
        name, start, end, input_names, output_names,
        own_sliding_slots, own_full_slots  (GLOBAL slots; host slices ``cache[slots]`` in / writes
            ``*_cur[j]`` back to ``slots[j]``),
        needs_prod_sliding_past, needs_prod_full_past   (read producer slot history),
        needs_ext_sliding_cur, needs_ext_full_cur       (producer cur comes from another chunk).
    """
    # A throwaway text model just to introspect routing/shapes — no weights needed for the plan.
    _, n_sliding, n_full = cfg.stateful_routing()
    prod_s_slot, prod_f_slot = n_sliding - 1, n_full - 1
    prod_s_layer, prod_f_layer = cfg.producer_idx(SLIDING), cfg.producer_idx(FULL)

    chunks = []
    producer_src: dict[str, tuple] = {}
    for ci, (start, end) in enumerate(boundaries):
        own_s, own_f = [], []
        route, _, _ = cfg.stateful_routing()
        for i in range(start, end):
            is_full, slot, write = route[i]
            if write:
                (own_f if is_full else own_s).append(slot)
        has_full = any(route[i][0] for i in range(start, end))
        has_sliding = any(not route[i][0] for i in range(start, end))
        has_shared_sliding = any(not route[i][2] and not route[i][0] for i in range(start, end))
        has_shared_full = any(not route[i][2] and route[i][0] for i in range(start, end))
        sliding_prod_in = start <= prod_s_layer < end
        full_prod_in = start <= prod_f_layer < end

        names_in = ["hidden_in", "per_layer_inputs", "position_ids"]
        if has_full:
            names_in.append("causal_mask_full")
        if has_sliding:
            names_in.append("causal_mask_sliding")
        if own_s:
            names_in += ["sliding_k", "sliding_v"]
        if own_f:
            names_in += ["full_k", "full_v"]
        ext_sliding_kv = has_shared_sliding and not sliding_prod_in
        ext_full_kv = has_shared_full and not full_prod_in
        local_sliding_past = has_shared_sliding and sliding_prod_in
        local_full_past = has_shared_full and full_prod_in
        if ext_sliding_kv:
            names_in += ["prod_sliding_kv_k", "prod_sliding_kv_v"]
        if local_sliding_past:
            names_in += ["prod_sliding_k", "prod_sliding_v"]
        if ext_full_kv:
            names_in += ["prod_full_kv_k", "prod_full_kv_v"]
        if local_full_past:
            names_in += ["prod_full_k", "prod_full_v"]

        names_out = ["hidden"]
        if own_s:
            names_out += ["sliding_k_cur", "sliding_v_cur"]
        if own_f:
            names_out += ["full_k_cur", "full_v_cur"]

        # If this chunk owns the producer slot, record where its current column is emitted.
        if prod_s_slot in own_s:
            producer_src[SLIDING] = (ci, "sliding_k_cur", "sliding_v_cur", own_s.index(prod_s_slot))
        if prod_f_slot in own_f:
            producer_src[FULL] = (ci, "full_k_cur", "full_v_cur", own_f.index(prod_f_slot))

        chunks.append({
            "name": f"chunk{ci + 1}",
            "start": start,
            "end": end,
            "input_names": tuple(names_in),
            "output_names": tuple(names_out),
            "own_sliding_slots": own_s,
            "own_full_slots": own_f,
            # EXTERNAL producer → host feeds prod_<type>_kv_* = cat(cache[prod_slot], producer_cur). LOCAL
            # producer → host feeds prod_<type>_k/v (slot past); the chunk cats with its in-graph cur.
            "needs_ext_sliding_kv": ext_sliding_kv,
            "needs_ext_full_kv": ext_full_kv,
            "needs_local_sliding_past": local_sliding_past,
            "needs_local_full_past": local_full_past,
        })

    return {
        "prod_s_slot": prod_s_slot,
        "prod_f_slot": prod_f_slot,
        "producer_src": producer_src,
        "chunks": chunks,
    }
