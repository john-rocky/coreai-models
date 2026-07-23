# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""GLM-Image AR generator (``vision_language_encoder``) shaped for Core AI export.

The AR half of zai-org/GLM-Image: a dense GLM-4-9B decoder that autoregressively
emits *visual* tokens (the DiT prior). Distinct from the zoo's other GLM riders
(glm4_moe_lite = MoE+MLA): this is plain **dense GQA** with GLM-4 sandwich norm.

Confirmed arch (config text_config + checkpoint, greedy 513/513 torch parity in
coreai/_glmimg_ar_parity.py):
  - 40 layers, hidden 4096, 32 heads, **2 KV heads (GQA)**, head_dim 128,
    intermediate 13696, silu, rms_norm_eps 1e-5.
  - **sandwich norm** (input / post_self_attn / post_attention / post_mlp).
  - q/k/v **bias**, o_proj no bias. MLP fused ``gate_up_proj`` (gate=first half).
  - **partial rotary 0.5** (rotate first 64 of 128), non-interleaved rotate_half.
  - **3D M-RoPE** (mrope_section [8,12,12], theta 1e4). Not collapsible to a
    scalar position -> cos/sin are **pre-computed on the host** and fed as graph
    inputs (the FLUX precomputed-RoPE idiom; keeps rope-frequency ops out of the
    traced graph, which the Core AI optimizer otherwise corrupts).
  - **lm_head -> vision_vocab 16512** (not the full 168064). Generated ids are
    < 16512 and index embed_tokens directly (no offset).

Decode-graph contract (bespoke image-pipeline loop, NOT the chat engine):

  ``(input_ids [1,s] int32, position_ids [1,total] int32 ramp,
     cos [1,s,64], sin [1,s,64], keyCache, valueCache) -> logits [1,s,16512]``

The host computes the 3D raster positions (coreai/_glmimg_ar_positions.py) and
the mrope cos/sin, and drives the autoregressive loop.
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA

PIPELINED_STATE_NAMES = ("keyCache", "valueCache")


def _partial_rotary(cfg) -> float:
    rp = getattr(cfg, "rope_parameters", None) or {}
    return float(rp.get("partial_rotary_factor", 1.0))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class GlmImageARAttention(nn.Module):
    """Dense GQA with fused qkv (+bias), partial rotary from pre-computed cos/sin."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", dim // n_heads)
        self.rot = int(head_dim * _partial_rotary(config))  # 64

        # fused q/k/v with bias; o_proj no bias
        self.qkv_proj = nn.Linear(dim, (n_heads + 2 * n_kv_heads) * head_dim, bias=True)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.sdpa = SDPA(is_causal=True, scale=head_dim ** -0.5)

    def _apply_rope(self, x, cos, sin):
        # x [b, nh, s, hd]; cos/sin [b, 1, s, rot]
        rot = self.rot
        x_rot, x_pass = x[..., :rot], x[..., rot:]
        x_rot = (x_rot * cos) + (_rotate_half(x_rot) * sin)
        return torch.cat([x_rot, x_pass], dim=-1)

    def forward(self, x, position_ids, cos, sin, cache: KVCache) -> torch.Tensor:
        b, q_len, _ = x.shape
        nh, nkv = self.n_heads, self.n_kv_heads
        qkv = (
            self.qkv_proj(x)
            .reshape(b, q_len, nh + 2 * nkv, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        query = qkv.narrow(1, 0, nh)
        key = qkv.narrow(1, nh, nkv)
        value = qkv.narrow(1, nh + nkv, nkv)

        c = cos.unsqueeze(1)  # [b,1,s,rot]
        s = sin.unsqueeze(1)
        query = self._apply_rope(query, c, s)
        key = self._apply_rope(key, c, s)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(q_len)
        torch._check_is_size(seq_len)
        offset = seq_len - q_len
        torch._check_is_size(offset)

        key, value = cache.update_and_fetch(
            self.layer_idx, offset, key, value, seq_len=seq_len, query_len=q_len
        )
        out = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(b, q_len, nh * self.head_dim)
        )
        return self.o_proj(out)


class GlmImageARBlock(nn.Module):
    """GLM-4 sandwich-norm decoder block."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        eps = config.rms_norm_eps
        self.self_attn = GlmImageARAttention(config, layer_idx)
        self.mlp = MLP(hidden, config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.post_self_attn_layernorm = RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)
        self.post_mlp_layernorm = RMSNorm(hidden, eps=eps)

    def forward(self, x, position_ids, cos, sin, cache) -> torch.Tensor:
        r = x
        a = self.self_attn(self.input_layernorm(x), position_ids, cos, sin, cache)
        a = self.post_self_attn_layernorm(a)
        h = r + a
        r = h
        m = self.mlp(self.post_attention_layernorm(h))
        m = self.post_mlp_layernorm(m)
        return r + m


class GlmImageARModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GlmImageARBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class GlmImageARForCausalLM(nn.Module):
    """GLM-Image AR generator; see module docstring for the decode-graph contract."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = GlmImageARModel(config)
        # lm_head projects to the VISION vocab, not the full text vocab.
        self.vision_vocab_size = getattr(config, "vision_vocab_size", config.vocab_size)
        self.lm_head = nn.Linear(config.hidden_size, self.vision_vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,     # [1, s] int32
        position_ids: torch.Tensor,  # [1, total] int32 ramp (cache indexing)
        cos: torch.Tensor,           # [1, s, rot] pre-computed mrope
        sin: torch.Tensor,           # [1, s, rot]
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        x = m.embed_tokens(input_ids)
        cache = KVCache(k_cache, v_cache)
        for layer in m.layers:
            x = layer(x, position_ids, cos, sin, cache)
        x = m.norm(x)
        logits = self.lm_head(x)
        if logits.dtype == torch.bfloat16:
            logits = logits.to(torch.float16)
        return logits

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_pretrained_dir(
        cls,
        vle_dir: str,
        target_dtype: torch.dtype = torch.float16,
    ) -> "GlmImageARForCausalLM":
        """Load from a local GLM-Image ``vision_language_encoder`` folder
        (config.json + sharded safetensors; text keys under
        ``model.language_model.``, plus top-level ``lm_head.weight``)."""
        import json
        from types import SimpleNamespace

        from safetensors import safe_open

        # Build the text config from config.json directly (avoids depending on a
        # transformers build that ships the `glm_image` model_type).
        with open(os.path.join(vle_dir, "config.json")) as f:
            tc = json.load(f)["text_config"]
        cfg = SimpleNamespace(
            hidden_size=tc["hidden_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            intermediate_size=tc["intermediate_size"],
            rms_norm_eps=tc["rms_norm_eps"],
            vocab_size=tc["vocab_size"],
            vision_vocab_size=tc.get("vision_vocab_size", 16512),
            max_position_embeddings=tc.get("max_position_embeddings", 131072),
            rope_parameters=tc.get("rope_parameters", {}),
            head_dim=tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
        )

        P = "model.language_model."
        sd = {}
        for path in sorted(glob.glob(os.path.join(vle_dir, "*.safetensors"))):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith(P):
                        sd[key.removeprefix(P)] = f.get_tensor(key).to(target_dtype)
                    elif key == "lm_head.weight":
                        sd["lm_head.weight"] = f.get_tensor(key).to(target_dtype)

        model = cls(cfg).to(dtype=target_dtype)
        out = {}
        nl = cfg.num_hidden_layers
        for i in range(nl):
            a = f"layers.{i}.self_attn."
            qw = sd.pop(a + "q_proj.weight"); kw = sd.pop(a + "k_proj.weight"); vw = sd.pop(a + "v_proj.weight")
            qb = sd.pop(a + "q_proj.bias"); kb = sd.pop(a + "k_proj.bias"); vb = sd.pop(a + "v_proj.bias")
            out[f"model.{a}qkv_proj.weight"] = torch.cat([qw, kw, vw], dim=0)
            out[f"model.{a}qkv_proj.bias"] = torch.cat([qb, kb, vb], dim=0)
            out[f"model.{a}o_proj.weight"] = sd.pop(a + "o_proj.weight")
            # fused gate_up -> split into zoo MLP's separate gate/up (gate = first half)
            mp = f"layers.{i}.mlp."
            gate_up = sd.pop(mp + "gate_up_proj.weight")
            inter = gate_up.shape[0] // 2
            out[f"model.{mp}gate_proj.weight"] = gate_up[:inter]
            out[f"model.{mp}up_proj.weight"] = gate_up[inter:]
            out[f"model.{mp}down_proj.weight"] = sd.pop(mp + "down_proj.weight")
            for nm in ("input_layernorm", "post_self_attn_layernorm",
                       "post_attention_layernorm", "post_mlp_layernorm"):
                out[f"model.layers.{i}.{nm}.weight"] = sd.pop(f"layers.{i}.{nm}.weight")
        out["model.embed_tokens.weight"] = sd.pop("embed_tokens.weight")
        out["model.norm.weight"] = sd.pop("norm.weight")
        out["lm_head.weight"] = sd.pop("lm_head.weight")

        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        model.eval()
        return model

    # -- export -------------------------------------------------------------

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_query: int = 1,
    ) -> dict:
        cfg = self.config
        rot = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
                  * _partial_rotary(cfg))
        tq, tp = trace_query, 64
        input_ids = torch.randint(1, cfg.vision_vocab_size, (1, tq), dtype=torch.int32)
        position_ids = torch.arange(tp + tq, dtype=torch.int32).unsqueeze(0)
        cos = torch.zeros(1, tq, rot, dtype=target_dtype)
        sin = torch.zeros(1, tq, rot, dtype=target_dtype)
        k_cache = torch.zeros(
            cfg.num_hidden_layers, 1, cfg.num_key_value_heads, trace_kv_len,
            getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
            dtype=target_dtype)
        v_cache = torch.zeros_like(k_cache)

        reference_inputs = {
            "input_ids": input_ids, "position_ids": position_ids,
            "cos": cos, "sin": sin, "k_cache": k_cache, "v_cache": v_cache,
        }
        static_ids = tq == 1
        pos_min = max(2, tq) if static_ids else 2
        seq_pos = torch.export.Dim("seq_pos", min=pos_min, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        if static_ids:
            ids_shape, cs_shape = None, None
        else:
            sd = torch.export.Dim("seq_ids", min=1, max=max_context_length - 2)
            ids_shape = {1: sd}
            cs_shape = {1: sd}
        dynamic_shapes = {
            "input_ids": ids_shape,
            "position_ids": {1: seq_pos},
            "cos": cs_shape,
            "sin": cs_shape,
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids", "cos", "sin"),
            "output_names": ("logits",),
            "state_names": PIPELINED_STATE_NAMES,
        }
