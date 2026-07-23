# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma 4 E2B decode-only core shaped for Apple's pipelined GPU engine.

Engine contract (``CoreAIPipelinedEngine`` + the per-token-inputs patch):
``(input_ids [1,1] static, position_ids [1,S] dynamic, ple_tokens [1,1,L,ld])
-> logits [1,1,V]`` with ONE growing KV state pair (``keyCache``/``valueCache``).

Differences vs :class:`~coreai_models.models.macos.gemma4_text.Gemma4DecodeStateful`
(the 4-state core driven by a hand-rolled host loop):

* ``embed_tokens`` (262144 x hidden, fp16) and the tied ``lm_head`` live IN-GRAPH —
  the engine feeds token ids and samples on-GPU. Only ``ple_tokens`` crosses from
  the host each step: the raw ``embed_tokens_per_layer`` row for the token
  (already scaled by sqrt(ld)), reshaped ``[1, 1, num_layers, ld]``, gathered from
  an mmap of the giant PLE table by the engine's ``PerTokenInputProvider``.
* ``per_layer_inputs`` are computed in-graph from ``inputs_embeds`` and the
  ``ple_tokens`` input (the projection + RMSNorm weights are small).
* ONE unified KV pair instead of dual sliding/full caches: the 15 non-shared
  layers stack into ``[n_slots, 1, n_kv, S, global_head_dim]``; sliding layers
  (head_dim 256) zero-pad K/V up to 512 on write and slice ``[..., :256]`` on
  read — the padded region is never read. Sliding layers ride the LINEAR cache
  with SDPA's window mask (no ring buffer); memory is ~60 KB/token.

The graph is S=1 static (one token per step), so prefill runs as pipelined S=1
steps (``COREAI_CHUNK_THRESHOLD=1``), exactly like the qwen3.5 decode-only port.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.models.macos.gemma4_text import (
    Gemma4ForCausalLM,
    Gemma4TextConfig,
)
from coreai_models.primitives.macos.cache import KVCache

# State tensor names surfaced to the Core AI runtime (engine reads them positionally:
# states[0] = key cache, states[1] = value cache).
PIPELINED_STATE_NAMES = ("keyCache", "valueCache")


def unified_routing(cfg: Gemma4TextConfig) -> tuple[list[tuple[int, bool]], int]:
    """Per-layer ``(slot, writes)`` into ONE padded KV pair.

    Sliding non-shared layers take slots ``[0, n_sliding)`` in layer order; full
    non-shared layers take ``[n_sliding, n_sliding + n_full)``. KV-shared layers
    read their type's producer slot (the last non-shared slot of that type).
    """
    route, n_sliding, _n_full = cfg.stateful_routing()
    unified = [
        (slot + n_sliding if is_full else slot, write) for is_full, slot, write in route
    ]
    return unified, n_sliding + _n_full


def build_pipelined_state(
    cfg: Gemma4TextConfig, max_seq_len: int, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Zero-initialised unified KV pair ``[n_slots, 1, n_kv, max_seq_len, hd_max]``."""
    _, n_slots = unified_routing(cfg)
    nkv = cfg.num_key_value_heads
    hd_max = cfg.global_head_dim
    return {
        "k_cache": torch.zeros(n_slots, 1, nkv, max_seq_len, hd_max, dtype=dtype),
        "v_cache": torch.zeros(n_slots, 1, nkv, max_seq_len, hd_max, dtype=dtype),
    }


def ple_tokens_from_model(
    model: Gemma4ForCausalLM, input_ids: torch.Tensor
) -> torch.Tensor:
    """The exact ``ple_tokens`` input for ``input_ids`` from the model's own table.

    ``embed_tokens_per_layer`` already scales by sqrt(ld), so this is just the
    gathered rows reshaped ``[b, s, num_layers, ld]`` — what a host provider must
    reproduce from its mmap dump (row * scale * sqrt(ld) for int8 dumps).
    """
    cfg = model.config
    b, s = input_ids.shape
    rows = model.model.embed_tokens_per_layer(input_ids)
    return rows.reshape(b, s, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input)


class Gemma4PipelinedForCausalLM(nn.Module):
    """Engine-shaped Gemma 4: in-graph embed/head, ple_tokens input, unified KV pair."""

    # Same opt-out as Gemma4DecodeStateful: the wrapper holds the whole text model
    # incl. modules not in the traced graph (embed_tokens_per_layer); externalizing
    # by class would mark those orphans and fail composite resolution.
    coreai_externalize_specs: tuple = ()

    def __init__(self, causal: Gemma4ForCausalLM) -> None:
        super().__init__()
        self.config = causal.config
        self.model = causal.model
        self.lm_head = causal.lm_head
        route, n_slots = unified_routing(self.config)
        self.route = route
        self.n_slots = n_slots
        self.hd_max = self.config.global_head_dim

    def _attn_step(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        kv: KVCache,
        slot: int,
        write: bool,
    ) -> torch.Tensor:
        """Attention against the unified padded cache.

        Mirrors ``Attention.forward_stateful`` (linear branch) with the dual-head-dim
        unification: K/V zero-padded to ``hd_max`` on write, sliced back on read.
        """
        b, s, _ = x.shape
        seq_len = position_ids.shape[-1]
        torch._check_is_size(s)
        torch._check_is_size(seq_len)
        offset = seq_len - s
        torch._check_is_size(offset)
        rope_pos = position_ids.narrow(-1, offset, s)

        hd = attn.head_dim
        q = attn.q_proj(x).view(b, s, attn.n_heads, hd).permute(0, 2, 1, 3)
        q = attn.q_norm(q)
        q = attn.rope(q, position_ids=rope_pos, freqs=attn.inv_freq)

        if write:
            k, v = attn._project_kv(x, rope_pos)
            if hd < self.hd_max:
                k = nn.functional.pad(k, (0, self.hd_max - hd))
                v = nn.functional.pad(v, (0, self.hd_max - hd))
            k, v = kv.update_and_fetch(slot, offset, k, v, seq_len=seq_len, query_len=s)
        else:
            # Read the producer's history (it ran earlier this step).
            k = kv._k_cache.narrow(0, slot, 1).narrow(-2, 0, seq_len).squeeze(0)
            v = kv._v_cache.narrow(0, slot, 1).narrow(-2, 0, seq_len).squeeze(0)
        if hd < self.hd_max:
            k = k.narrow(-1, 0, hd)
            v = v.narrow(-1, 0, hd)

        out = attn.sdpa(query=q, key=k, value=v)
        out = out.permute(0, 2, 1, 3).reshape(b, s, attn.n_heads * hd)
        return attn.o_proj(out)

    def _layer_step(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        per_layer_input: torch.Tensor,
        kv: KVCache,
        slot: int,
        write: bool,
    ) -> torch.Tensor:
        residual = x
        x = layer.input_layernorm(x)
        x = self._attn_step(layer.self_attn, x, position_ids, kv, slot, write)
        x = layer.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = layer.pre_feedforward_layernorm(x)
        x = layer.mlp(x)
        x = layer.post_feedforward_layernorm(x)
        x = residual + x

        residual = x
        x = layer.per_layer_input_gate(x)
        x = nn.functional.gelu(x, approximate="tanh")
        x = x * per_layer_input
        x = layer.per_layer_projection(x)
        x = layer.post_per_layer_input_norm(x)
        x = residual + x

        return x * layer.layer_scalar

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        ple_tokens: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        cfg = self.config
        b, s = input_ids.shape
        ld = cfg.hidden_size_per_layer_input

        x = m.embed_tokens(input_ids)

        # per_layer_inputs from the in-graph projection + the host-gathered rows.
        proj = m.per_layer_model_projection(x) * m.per_layer_model_projection_scale
        proj = proj.reshape(b, s, cfg.num_hidden_layers, ld)
        proj = m.per_layer_projection_norm(proj)
        per_layer_inputs = (proj + ple_tokens) * m.per_layer_input_scale

        kv = KVCache(k_cache, v_cache)
        for i, layer in enumerate(m.layers):
            slot, write = self.route[i]
            x = self._layer_step(
                layer, x, position_ids, per_layer_inputs[:, :, i, :], kv, slot, write
            )
        hidden = m.norm(x)

        logits = self.lm_head(hidden)
        sc = cfg.final_logit_softcapping
        if sc is not None:
            logits = torch.tanh(logits / sc) * sc
        return logits

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_past: int = 64,
    ) -> dict:
        """Reference inputs + dynamic shapes for the decode-only engine export."""
        cfg = self.config
        input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
        position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
        ple_tokens = torch.zeros(
            1, 1, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input,
            dtype=target_dtype,
        )
        state = build_pipelined_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)

        reference_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "ple_tokens": ple_tokens,
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
        }
        seq_pos = torch.export.Dim("seq_pos", min=2, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        dynamic_shapes = {
            "input_ids": None,   # static [1, 1]
            "position_ids": {1: seq_pos},
            "ple_tokens": None,  # static [1, 1, L, ld] — per-token host input
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids", "ple_tokens"),
            "output_names": ("logits",),
            "state_names": PIPELINED_STATE_NAMES,
        }


class Gemma4VLPipelinedForCausalLM(Gemma4PipelinedForCausalLM):
    """VL rider on the provider variant: ``image_embeds`` as a STATIC input.

    Qwen3-VL recipe minus M-RoPE/DeepStack (Gemma 4 E2B keeps the image span
    CAUSAL — verified vs the fp32 HF oracle mask dump — and uses standard
    contiguous positions, so ONLY the embedding lookup changes):

    * the host rewrites the prompt's 256/280 ``<image_soft_token>`` ids to
      EXTENSION ids ``V + slot`` (slot 0..n_soft-1 in soft-token order) and
      binds ``image_embeds [n_image_slots, hidden]`` (vision encoder output,
      raw — NO sqrt(h) embed scale, matching HF's post-scale splice) to the
      same buffer every encode;
    * in-graph: ``x = ids < V ? embed_tokens[ids] : image_embeds[ids - V]``;
    * PLE: HF gathers the PAD row (id 0) for image positions
      (``llm_input_ids[multimodal_mask] = pad_token_id``) — with per-token
      ``ple_tokens`` this is a HOST rule: feed the pad row whenever the step's
      id is an extension id. The per-layer PROJECTION branch reads the spliced
      ``x`` in-graph, exactly like HF (projection input at image slots IS the
      vision embedding).

    With no extension ids in the stream the graph degenerates to the text
    decoder (image_embeds is a dead 1-row gather per step).
    """

    def __init__(self, causal: Gemma4ForCausalLM, n_image_slots: int = 280,
                 pad_token_id: int = 0) -> None:
        super().__init__(causal)
        self.n_image_slots = n_image_slots
        self.pad_token_id = pad_token_id

    def _spliced_embed(
        self, input_ids: torch.Tensor, image_embeds: torch.Tensor
    ) -> torch.Tensor:
        V, N = self.config.vocab_size, self.n_image_slots
        b, s = input_ids.shape
        is_img = input_ids >= V
        slot = (input_ids - V).clamp(0, N - 1)
        e_txt = self.model.embed_tokens(input_ids.clamp(0, V - 1))
        e_img = image_embeds.index_select(0, slot.reshape(-1)).reshape(b, s, -1)
        return torch.where(is_img.unsqueeze(-1), e_img.to(e_txt.dtype), e_txt)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        ple_tokens: torch.Tensor,
        image_embeds: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        cfg = self.config
        b, s = input_ids.shape
        ld = cfg.hidden_size_per_layer_input

        x = self._spliced_embed(input_ids, image_embeds)

        proj = m.per_layer_model_projection(x) * m.per_layer_model_projection_scale
        proj = proj.reshape(b, s, cfg.num_hidden_layers, ld)
        proj = m.per_layer_projection_norm(proj)
        per_layer_inputs = (proj + ple_tokens) * m.per_layer_input_scale

        kv = KVCache(k_cache, v_cache)
        for i, layer in enumerate(m.layers):
            slot, write = self.route[i]
            x = self._layer_step(
                layer, x, position_ids, per_layer_inputs[:, :, i, :], kv, slot, write
            )
        hidden = m.norm(x)

        logits = self.lm_head(hidden)
        sc = cfg.final_logit_softcapping
        if sc is not None:
            logits = torch.tanh(logits / sc) * sc
        return logits

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_past: int = 64,
    ) -> dict:
        spec = super().build_export_spec(
            target_dtype, max_context_length, trace_kv_len, trace_past)
        spec["reference_inputs"]["image_embeds"] = torch.zeros(
            self.n_image_slots, self.config.hidden_size, dtype=target_dtype)
        # re-key so image_embeds sits between ple_tokens and the caches
        ref = spec["reference_inputs"]
        spec["reference_inputs"] = {
            "input_ids": ref["input_ids"], "position_ids": ref["position_ids"],
            "ple_tokens": ref["ple_tokens"], "image_embeds": ref["image_embeds"],
            "k_cache": ref["k_cache"], "v_cache": ref["v_cache"],
        }
        spec["dynamic_shapes"] = {
            **{k: v for k, v in spec["dynamic_shapes"].items()
               if k not in ("k_cache", "v_cache")},
            "image_embeds": None,  # static [n_image_slots, h] — bound once
            "k_cache": spec["dynamic_shapes"]["k_cache"],
            "v_cache": spec["dynamic_shapes"]["v_cache"],
        }
        spec["input_names"] = (
            "input_ids", "position_ids", "ple_tokens", "image_embeds")
        return spec


class Gemma4PipelinedTblForCausalLM(Gemma4PipelinedForCausalLM):
    """``tbl`` variant: the PLE table rides as a STATIC graph INPUT.

    Kills the per-token host dependency of the base variant: instead of a
    ``ple_tokens`` row filled by a host provider each step (which forces the
    decode loop to wait for the GPU-sampled token — a serialization tax measured
    ~13 ms/token on iPhone), the graph takes

    * ``ple_table  [V, L*ld] int8`` — the per-row-quantized PLE dump,
    * ``ple_scale  [V] f32``        — its per-row scales,

    bound to the SAME host buffers every encode (the engine's static-input hook),
    and gathers rows in-graph by ``input_ids``: row = q[id] * s[id] * sqrt(ld).
    No token ever needs to reach the CPU, so the engine's full 3-deep pipeline
    survives in decode. ``embed_tokens`` stays IN-GRAPH: a 3rd table input was
    measured strictly worse on iPhone (every statically-bound byte pays a
    per-encode residency tax, and owned buffers are dirty memory against the
    jetsam limit — the AOT escape handles the 2.0 GB constants anyway).
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        ple_table: torch.Tensor,
        ple_scale: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        cfg = self.config
        b, s = input_ids.shape
        ld = cfg.hidden_size_per_layer_input
        ids = input_ids.reshape(-1)

        x = m.embed_tokens(input_ids)

        # PLE rows from the int8 dump INPUT — the PerTokenInputProvider semantics
        # (q[id] f32 * scale[id] * sqrt(ld) -> compute dtype) moved in-graph.
        rows = ple_table.index_select(0, ids).to(torch.float32)
        scale = ple_scale.index_select(0, ids) * (float(ld) ** 0.5)
        ple_tokens = (rows * scale.unsqueeze(-1)).to(x.dtype)
        ple_tokens = ple_tokens.reshape(b, s, cfg.num_hidden_layers, ld)

        proj = m.per_layer_model_projection(x) * m.per_layer_model_projection_scale
        proj = proj.reshape(b, s, cfg.num_hidden_layers, ld)
        proj = m.per_layer_projection_norm(proj)
        per_layer_inputs = (proj + ple_tokens) * m.per_layer_input_scale

        kv = KVCache(k_cache, v_cache)
        for i, layer in enumerate(m.layers):
            slot, write = self.route[i]
            x = self._layer_step(
                layer, x, position_ids, per_layer_inputs[:, :, i, :], kv, slot, write
            )
        hidden = m.norm(x)

        logits = self.lm_head(hidden)
        sc = cfg.final_logit_softcapping
        if sc is not None:
            logits = torch.tanh(logits / sc) * sc
        return logits

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_past: int = 64,
        ple_table: torch.Tensor | None = None,
        ple_scale: torch.Tensor | None = None,
        trace_query: int = 1,
    ) -> dict:
        """Reference inputs + dynamic shapes; the PLE tables ride as STATIC inputs.

        Pass the REAL dump tensors so the quantizer's calibration forward sees
        ship-path values (zeros would work for tracing but not for calibration).

        ``trace_query`` selects the entrypoint's query width. ``1`` is the S=1
        decode function (default). A value > 1 traces a STATIC S=``trace_query``
        chunked-prefill function: ``input_ids`` is pinned ``[1, q]`` so the engine
        walks a 1024-token prompt in fixed q-sized chunks, capping the attention
        scratch at ``q * context`` instead of the O(p^2) blow-up a single dynamic
        S=p prefill allocates (which jetsams the phone at p>=512). The forward is
        already parametric in ``s`` — only this trace shape changes.
        """
        cfg = self.config
        pld = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input
        q = trace_query
        input_ids = torch.randint(1, cfg.vocab_size, (1, q), dtype=torch.int32)
        position_ids = torch.arange(trace_past + q, dtype=torch.int32).unsqueeze(0)
        if ple_table is None:
            ple_table = torch.zeros(cfg.vocab_size, pld, dtype=torch.int8)
        if ple_scale is None:
            ple_scale = torch.ones(cfg.vocab_size, dtype=torch.float32)
        state = build_pipelined_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)

        reference_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "ple_table": ple_table,
            "ple_scale": ple_scale,
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
        }
        # min = q+1: the static-S=q attention traces a `seq_len != q` guard
        # (offset = seq_len - q must be > 0 for the KV read narrow to certify), so
        # torch refuses seq_len == q. Matches the S=1 decode convention (min=2).
        # The compiled MPSGraph keeps S dynamic, so the engine's first chunk (which
        # arrives at seq_len == q, offset 0) still runs the same kernel — a full
        # causal q×q prefill — verified numerically on device.
        seq_pos = torch.export.Dim("seq_pos", min=q + 1, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        dynamic_shapes = {
            "input_ids": None,    # static [1, q] — S=1 decode or S=q prefill chunk
            "position_ids": {1: seq_pos},
            "ple_table": None,    # static [V, L*ld] — bound once, never copied
            "ple_scale": None,    # static [V]
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids", "ple_table", "ple_scale"),
            "output_names": ("logits",),
            "state_names": PIPELINED_STATE_NAMES,
        }


class Gemma4VLPipelinedTblForCausalLM(Gemma4PipelinedTblForCausalLM):
    """VL rider on the ``tbl`` variant (in-graph PLE gather + image_embeds).

    Same splice as :class:`Gemma4VLPipelinedForCausalLM` (see its docstring),
    but the PLE pad-row rule moves IN-GRAPH: the table gather indexes
    ``where(ids >= V, pad_token_id, ids)`` so image steps read the pad row —
    byte-identical PLE tables to the text ship, no host involvement, full
    pipeline depth in decode.
    """

    def __init__(self, causal: Gemma4ForCausalLM, n_image_slots: int = 280,
                 pad_token_id: int = 0) -> None:
        super().__init__(causal)
        self.n_image_slots = n_image_slots
        self.pad_token_id = pad_token_id

    _spliced_embed = Gemma4VLPipelinedForCausalLM._spliced_embed

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        ple_table: torch.Tensor,
        ple_scale: torch.Tensor,
        image_embeds: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        cfg = self.config
        b, s = input_ids.shape
        ld = cfg.hidden_size_per_layer_input
        V = cfg.vocab_size

        x = self._spliced_embed(input_ids, image_embeds)

        # PLE rows: image steps gather the PAD row (HF llm_input_ids rule).
        # Dequant in f16 end-to-end — int8 values are exact in f16 and the
        # scale rounding (~2^-11 rel) sits far below the dump's own int8
        # noise; the f32 [1, L*ld] intermediate of the text variant was
        # 35.8 KB of MPSGraph scratch, and the VL splice ops on top of it
        # overflowed the runtime's ~208 KB per-encode scratch heap on iOS
        # (allocateMTLBufferFromMTLHeap / ViewOp abort).
        ple_ids = torch.where(
            input_ids >= V,
            torch.full_like(input_ids, self.pad_token_id),
            input_ids,
        ).reshape(-1)
        rows = ple_table.index_select(0, ple_ids).to(x.dtype)
        scale = (ple_scale.index_select(0, ple_ids) * (float(ld) ** 0.5)).to(x.dtype)
        ple_tokens = rows * scale.unsqueeze(-1)
        ple_tokens = ple_tokens.reshape(b, s, cfg.num_hidden_layers, ld)

        proj = m.per_layer_model_projection(x) * m.per_layer_model_projection_scale
        proj = proj.reshape(b, s, cfg.num_hidden_layers, ld)
        proj = m.per_layer_projection_norm(proj)
        per_layer_inputs = (proj + ple_tokens) * m.per_layer_input_scale

        kv = KVCache(k_cache, v_cache)
        for i, layer in enumerate(m.layers):
            slot, write = self.route[i]
            x = self._layer_step(
                layer, x, position_ids, per_layer_inputs[:, :, i, :], kv, slot, write
            )
        hidden = m.norm(x)

        logits = self.lm_head(hidden)
        sc = cfg.final_logit_softcapping
        if sc is not None:
            logits = torch.tanh(logits / sc) * sc
        return logits

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_past: int = 64,
        ple_table: torch.Tensor | None = None,
        ple_scale: torch.Tensor | None = None,
    ) -> dict:
        spec = super().build_export_spec(
            target_dtype, max_context_length, trace_kv_len, trace_past,
            ple_table=ple_table, ple_scale=ple_scale)
        ref = spec["reference_inputs"]
        spec["reference_inputs"] = {
            "input_ids": ref["input_ids"], "position_ids": ref["position_ids"],
            "ple_table": ref["ple_table"], "ple_scale": ref["ple_scale"],
            "image_embeds": torch.zeros(
                self.n_image_slots, self.config.hidden_size, dtype=target_dtype),
            "k_cache": ref["k_cache"], "v_cache": ref["v_cache"],
        }
        spec["dynamic_shapes"] = {
            **{k: v for k, v in spec["dynamic_shapes"].items()
               if k not in ("k_cache", "v_cache")},
            "image_embeds": None,  # static [n_image_slots, h] — bound once
            "k_cache": spec["dynamic_shapes"]["k_cache"],
            "v_cache": spec["dynamic_shapes"]["v_cache"],
        }
        spec["input_names"] = (
            "input_ids", "position_ids", "ple_table", "ple_scale", "image_embeds")
        return spec
