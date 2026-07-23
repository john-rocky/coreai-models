# Community port — NOT an Apple model.
"""Gemma 4 E2B MIXED-BIT (mobile QAT transplant) decode core for the pipelined engine.

Rider on :class:`~coreai_models.models.macos.gemma4_pipelined.Gemma4PipelinedForCausalLM`
carrying Google's mobile QAT recipe (INT2/INT4/INT8 per-channel symmetric, extracted bit-exact
from the ``.litertlm`` release — see
coreai-models-community/knowledge/gemma4-mixedbit-qat-transplant.md). Differences vs the
shipped ``tbl`` variant:

* **No extra inputs at all** — the packed INT2 embed table and the packed INT4 PLE table ride
  as IN-GRAPH CONSTANTS (uint8 buffers), so the bundle runs on a stock engine / llm-benchmark
  with no ``staticInputBuffers`` binding and no per-token provider. Per-token reads stay tiny
  (row gathers).
* **Bit-unpack via byte-LUT gathers, not bitwise ops**: a gathered packed row (uint8) indexes a
  tiny constant LUT (``[256, codes-per-byte]`` fp16) with ``index_select`` — the exact op class
  the shipped tbl variant already proved on the GPU delegate (gather + cast + multiply). INT2:
  ``lut2 [256,4]`` (bits[1:0] first, signed); INT4: ``lut4 [256,2]`` (low nibble first, signed).
* PLE scales are per (row, table): ``ple_scale [V, 35] fp32`` (the extract's 35 per-table
  per-row scale vectors stacked), vs the tbl variant's single per-row scale.
* fp16 end-to-end after the fp32 scale gather (the VL-variant iOS scratch-heap lesson).

The FFN / attn q/o / lm_head matvecs are swapped for the transplant Metal-kernel modules by the
export script (``gemma4_metal_mlp_int2``); this file only owns the graph shape + table gathers.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.models.macos.gemma4_pipelined import (
    Gemma4PipelinedForCausalLM,
    build_pipelined_state,
)
from coreai_models.models.macos.gemma4_text import Gemma4ForCausalLM
from coreai_models.primitives.macos.cache import KVCache


def _int2_byte_lut() -> torch.Tensor:
    """[256, 4] fp16: byte -> its 4 signed INT2 codes (bits[1:0] first, two's complement)."""
    b = torch.arange(256, dtype=torch.int32)
    c = torch.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], dim=-1)
    return torch.where(c >= 2, c - 4, c).to(torch.float16)


def _int4_byte_lut() -> torch.Tensor:
    """[256, 2] fp16: byte -> its 2 signed INT4 codes (low nibble first, two's complement)."""
    b = torch.arange(256, dtype=torch.int32)
    c = torch.stack([b & 0xF, b >> 4], dim=-1)
    return torch.where(c >= 8, c - 16, c).to(torch.float16)


class PackedInt2Embedding(nn.Module):
    """In-graph INT2 embedding gather: dequant = lut2[bytes] * scale[id] * embed_scale.

    Drop-in for ``ScaledEmbedding`` (the sqrt(hidden) embed scale is folded into the gathered
    per-row scale). Table stays packed (4 codes/byte) — the per-token read is hidden/4 bytes.
    """

    def __init__(self, packed_u8: torch.Tensor, scale: torch.Tensor, hidden: int,
                 embed_scale: float) -> None:
        super().__init__()
        v = packed_u8.shape[0] if packed_u8.dim() == 2 else packed_u8.numel() // (hidden // 4)
        self.hidden = hidden
        self.embed_scale = float(embed_scale)
        self.register_buffer("packed", packed_u8.reshape(v, hidden // 4).contiguous())
        self.register_buffer("scale", scale.detach().float().contiguous())  # [V] fp32
        self.register_buffer("lut2", _int2_byte_lut())                      # [256, 4] fp16

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, s = input_ids.shape
        ids = input_ids.reshape(-1)
        rows = self.packed.index_select(0, ids)                       # [n, h/4] u8
        codes = self.lut2.index_select(0, rows.reshape(-1).to(torch.int32))  # [n*h/4, 4] f16
        sc = (self.scale.index_select(0, ids) * self.embed_scale).to(torch.float16)
        return codes.reshape(b, s, self.hidden) * sc.reshape(b, s, 1)


class Gemma4MixedbitPipelinedForCausalLM(Gemma4PipelinedForCausalLM):
    """Engine-shaped mixed-bit Gemma 4: in-graph packed INT4 PLE gather, no extra inputs.

    ``(input_ids [1,1] static, position_ids [1,S] dynamic) -> logits [1,1,V]`` with the ONE
    unified KV pair. ``causal.model.embed_tokens`` must already be a :class:`PackedInt2Embedding`
    (or equivalent) — the export script swaps it before wrapping.
    """

    def __init__(self, causal: Gemma4ForCausalLM, ple_packed_u8: torch.Tensor,
                 ple_scale: torch.Tensor) -> None:
        super().__init__(causal)
        cfg = self.config
        v = cfg.vocab_size_per_layer_input
        pld = cfg.num_hidden_layers * cfg.hidden_size_per_layer_input
        self.register_buffer("ple_packed", ple_packed_u8.reshape(v, pld // 2).contiguous())
        self.register_buffer("ple_scale", ple_scale.detach().float().contiguous())  # [V, L] fp32
        self.register_buffer("lut4", _int4_byte_lut())                              # [256, 2] fp16
        # sqrt(ld) — the ScaledEmbedding factor the PLE table rows carry in the reference model.
        self.ple_row_scale = float(cfg.hidden_size_per_layer_input) ** 0.5

    def _ple_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        b, s = input_ids.shape
        ld = cfg.hidden_size_per_layer_input
        ids = input_ids.reshape(-1)
        rows = self.ple_packed.index_select(0, ids)                   # [n, pld/2] u8
        codes = self.lut4.index_select(0, rows.reshape(-1).to(torch.int32))  # [n*pld/2, 2] f16
        codes = codes.reshape(b, s, cfg.num_hidden_layers, ld)
        sc = (self.ple_scale.index_select(0, ids) * self.ple_row_scale).to(torch.float16)
        return codes * sc.reshape(b, s, cfg.num_hidden_layers, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
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
        return logits

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_past: int = 64,
    ) -> dict:
        cfg = self.config
        input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
        position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
        state = build_pipelined_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)

        reference_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
        }
        seq_pos = torch.export.Dim("seq_pos", min=2, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        dynamic_shapes = {
            "input_ids": None,   # static [1, 1]
            "position_ids": {1: seq_pos},
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids"),
            "output_names": ("logits",),
            "state_names": ("keyCache", "valueCache"),
        }
