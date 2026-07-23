# Community port — NOT an Apple model.
"""S=4 VERIFY companion for the Gemma-4 E2B mixed-bit transplant (spec-decode P5c).

Same weights/graph as :class:`Gemma4MixedbitPipelinedForCausalLM`, but:
  * ``input_ids`` STATIC [1, 4] — one forward verifies token t plus 3 MTP drafts
    (weights read ONCE for 4 tokens; the matvecs ride the M=4 kernels)
  * outputs ``(logits [1,4,V], activations [1,4,1536])`` — activations = the
    final-norm hidden, the drafter's chaining/kickoff tap (litert `verify` contract)
  * KV WRITES all 4 positions (offset = seq_len-4); rejected drafts' rows are
    overwritten by the next round's write at the re-anchored offset — the litert
    accept loop's exact state discipline.

The SDPA composite emits the lower-right causal (+ sliding-window) masks for
q_len 4, so intra-block causality and the 512-window both hold in-graph.
"""
from __future__ import annotations

import torch

from coreai_models.models.macos.gemma4_mixedbit_pipelined import (
    Gemma4MixedbitPipelinedForCausalLM,
)
from coreai_models.primitives.macos.cache import KVCache

VERIFY_S = 4


class Gemma4MixedbitVerifyForCausalLM(Gemma4MixedbitPipelinedForCausalLM):
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.model
        cfg = self.config
        b, s = input_ids.shape
        ld = cfg.hidden_size_per_layer_input

        x = m.embed_tokens(input_ids)
        ple_tokens = self._ple_tokens(input_ids)

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

        # The 4 freshly written K/V rows of the drafter's producer slots (11 = L13
        # sliding, 14 = L14 full) — the host appends these to its small drafter-input
        # buffers each round (no cross-graph state sharing needed).
        seq_len = position_ids.shape[-1]
        off = seq_len - s
        torch._check_is_size(off)

        def rows(cache: torch.Tensor, slot: int) -> torch.Tensor:
            return cache.narrow(0, slot, 1).narrow(-2, off, s).reshape(1, s, -1)

        k11_rows = rows(k_cache, 11)
        v11_rows = rows(v_cache, 11)
        k14_rows = rows(k_cache, 14)
        v14_rows = rows(v_cache, 14)
        return logits, hidden, k11_rows, v11_rows, k14_rows, v14_rows

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_past: int = 64,
    ) -> dict:
        from coreai_models.models.macos.gemma4_pipelined import build_pipelined_state

        cfg = self.config
        input_ids = torch.randint(1, cfg.vocab_size, (1, VERIFY_S), dtype=torch.int32)
        position_ids = torch.arange(trace_past + VERIFY_S, dtype=torch.int32).unsqueeze(0)
        state = build_pipelined_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)

        reference_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
        }
        seq_pos = torch.export.Dim("seq_pos", min=VERIFY_S + 1, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        dynamic_shapes = {
            "input_ids": None,  # STATIC [1, 4]
            "position_ids": {1: seq_pos},
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids"),
            "output_names": ("logits", "activations",
                             "k11_rows", "v11_rows", "k14_rows", "v14_rows"),
            "state_names": ("keyCache", "valueCache"),
        }
