# Community port — NOT an Apple model. BSD-3-Clause (see LICENSE).

"""Gemma 4 *dense* (12B-class) decode-only core for Apple's pipelined GPU engine.

Engine contract (stock ``CoreAIPipelinedEngine`` — NO patch needed, exactly 2
states): ``(input_ids [1,1] static, position_ids [1,S] dynamic) -> logits
[1,1,V]`` with ONE growing KV pair (``keyCache`` / ``valueCache``). The dense
12B carries no PLE, so — unlike the E2B ``gemma4_pipelined`` core — there is no
``ple_tokens`` host input and no per-token provider: ``embed_tokens`` and the
tied ``lm_head`` live IN-GRAPH and the engine feeds only token ids + positions.

Unifying the dual attention into ONE KV pair
--------------------------------------------
Gemma 4 dense interleaves two attention shapes:

* sliding layers — ``num_key_value_heads`` (8) KV heads, head_dim 256, window;
* full layers — ``num_global_key_value_heads`` (1) KV head, head_dim 512, global,
  with ``attention_k_eq_v`` (value == raw k_proj output).

The pipelined engine grows exactly ONE KV state pair, so both ride a single
``[num_layers, 1, n_kv_max, S, hd_max]`` cache with ``n_kv_max = 8`` and
``hd_max = 512``. Every layer owns its own slot (no KV-sharing). On write a
layer's K/V are zero-padded up to ``(n_kv_max, hd_max)``; on read they are sliced
back to the layer's real ``(n_kv, head_dim)`` — the padded region is never read.
Sliding layers attend over the LINEAR cache with SDPA's window mask (no ring).
The graph is S=1 static, so prefill runs as pipelined S=1 steps
(``COREAI_CHUNK_THRESHOLD=1``).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.models.macos.gemma4_dense_text import (
    Gemma4DenseConfig,
    Gemma4DenseForCausalLM,
)
from coreai_models.primitives.macos.cache import KVCache

# states[0] = key cache, states[1] = value cache (read positionally by the runtime).
PIPELINED_STATE_NAMES = ("keyCache", "valueCache")


def build_pipelined_state(
    cfg: Gemma4DenseConfig, max_seq_len: int, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """Zero-initialised unified KV pair ``[num_layers, 1, n_kv_max, max_seq_len, hd_max]``."""
    n_slots = cfg.num_hidden_layers
    nkv = cfg.num_key_value_heads
    hd_max = cfg.global_head_dim
    return {
        "k_cache": torch.zeros(n_slots, 1, nkv, max_seq_len, hd_max, dtype=dtype),
        "v_cache": torch.zeros(n_slots, 1, nkv, max_seq_len, hd_max, dtype=dtype),
    }


class Gemma4DensePipelinedForCausalLM(nn.Module):
    """Engine-shaped Gemma 4 dense: in-graph embed/head, unified single KV pair."""

    # Hold the whole text model (incl. modules not in the traced graph); opt out of
    # composite externalization by class so no orphan modules break resolution.
    coreai_externalize_specs: tuple = ()

    def __init__(self, causal: Gemma4DenseForCausalLM) -> None:
        super().__init__()
        self.config = causal.config
        self.model = causal.model
        self.lm_head = causal.lm_head
        self.nkv_max = self.config.num_key_value_heads
        self.hd_max = self.config.global_head_dim

    def _attn_step(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        kv: KVCache,
        slot: int,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        seq_len = position_ids.shape[-1]
        torch._check_is_size(s)
        torch._check_is_size(seq_len)
        offset = seq_len - s
        torch._check_is_size(offset)
        rope_pos = position_ids.narrow(-1, offset, s)

        hd = attn.head_dim
        nkv = attn.n_kv_heads
        q = attn.q_proj(x).view(b, s, attn.n_heads, hd).permute(0, 2, 1, 3)
        q = attn.q_norm(q)
        q = attn.rope(q, position_ids=rope_pos, freqs=attn.inv_freq)

        k, v = attn._project_kv(x, rope_pos)  # [b, nkv, s, hd]
        # Unify into ONE [.., n_kv_max, .., hd_max] cache so all layers share a single
        # growing KV pair AND a single GQA ratio (n_heads : n_kv_max), avoiding the
        # full-layer broadcast + head-narrow that crash MPSGraph at 12B dims:
        #  * KV heads: full layers REPLICATE their n_kv real heads up to n_kv_max with
        #    ``repeat_interleave`` (each real head fills a contiguous block of slots), so the
        #    block GQA ``query_head -> slot // (n_heads // n_kv_max)`` lands on the right real
        #    head — correct for one global head (12B, GQA H:1) AND several (31B, GQA H:4);
        #  * head_dim: sliding layers zero-pad 256 -> hd_max (the pad is never read back).
        if nkv < self.nkv_max:
            k = k.repeat_interleave(self.nkv_max // nkv, dim=1)
            v = v.repeat_interleave(self.nkv_max // nkv, dim=1)
        if hd < self.hd_max:
            k = nn.functional.pad(k, (0, self.hd_max - hd))
            v = nn.functional.pad(v, (0, self.hd_max - hd))
        k, v = kv.update_and_fetch(slot, offset, k, v, seq_len=seq_len, query_len=s)
        if hd < self.hd_max:  # slice the padded head_dim back (KV heads are real)
            k = k.narrow(-1, 0, hd)
            v = v.narrow(-1, 0, hd)

        out = attn.sdpa(query=q, key=k, value=v)
        out = out.permute(0, 2, 1, 3).reshape(b, s, attn.n_heads * hd)
        return attn.o_proj(out)

    def _layer_step(
        self, layer: nn.Module, x: torch.Tensor, position_ids: torch.Tensor,
        kv: KVCache, slot: int,
    ) -> torch.Tensor:
        residual = x
        x = layer.input_layernorm(x)
        x = self._attn_step(layer.self_attn, x, position_ids, kv, slot)
        x = layer.post_attention_layernorm(x)
        x = residual + x

        residual = x
        x = layer.pre_feedforward_layernorm(x)
        x = layer.mlp(x)
        x = layer.post_feedforward_layernorm(x)
        x = residual + x
        return x * layer.layer_scalar

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        cfg = self.config
        x = m.embed_tokens(input_ids)
        kv = KVCache(k_cache, v_cache)
        for i, layer in enumerate(m.layers):
            x = self._layer_step(layer, x, position_ids, kv, i)
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
            "state_names": PIPELINED_STATE_NAMES,
        }
