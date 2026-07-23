# Community port — NOT an Apple model.
"""Qwen2.5-Omni *Thinker* text decoder for the Core AI authoring path.

The Thinker is a vanilla dense Qwen2.5-3B decoder (36L / hidden 2048 / GQA 16:2 /
head_dim 128 / fused-QKV **with bias** / SwiGLU / RMSNorm / RoPE theta 1e6 /
**UNTIED** lm_head).  Audio understanding = audio-encoder embeds injected at the
``<audio>`` placeholder positions, VLM-style — this mirrors ``qwen3_vl``'s
image-embed scatter exactly, swapping the vision tower for the Whisper-style
audio tower.

TMRoPE note
-----------
Qwen2.5-Omni uses TMRoPE (a 3-section t/h/w mRoPE).  For **audio + text only**
(no image / video) every token's three coordinates are expected to collapse to a
single temporal position (text is sequential; audio is purely temporal with no
spatial grid), so 3-section mRoPE == standard 1-D RoPE with the temporal
positions.  We therefore reuse ``qwen2.py``'s standard-RoPE attention block and
null out ``rope_scaling`` at load time, and the **host supplies the temporal
position_ids**.  THIS COLLAPSE MUST BE VALIDATED against the HF oracle
(``_smoke/gen_qwen2_5omni_ref.py`` checks ``get_rope_index`` returns t==h==w);
if audio tokens make the coordinates diverge, swap ``TransformerBlock`` for a
qwen2-attention + 3-section-rope variant.

Two-bundle design (like qwen3_vl)
---------------------------------
  - audio encoder ``.aimodel``:  mel [1,128,T] -> audio_embeds [N, 2048]
        (see ``qwen2_5_omni_audio.py``)
  - this decoder ``.aimodel``:  input_ids, position_ids, audio_embeds,
        k_cache, v_cache -> logits.  Audio placeholder ids = ``V + slot`` so the
        graph ``index_select``s row ``slot`` of ``audio_embeds`` (host rewrites
        the repeated ``audio_token_index`` into ``V, V+1, ...`` in order).
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn

from coreai_models.models.macos.qwen2 import TransformerBlock
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm

# KV-only stateful decode (no SSM/conv state, unlike qwen3.5): matches the
# pipelined engine contract used by qwen3_vl / gemma4_pipelined.
PIPELINED_STATE_NAMES = ("keyCache", "valueCache")


class _ThinkerModel(nn.Module):
    """embed + dense Qwen2.5 layers + final norm (weights live under ``model.``)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen2_5OmniThinkerForCausalLM(nn.Module):
    """Engine-shaped Qwen2.5-Omni Thinker: (audio + text) -> logits.

    Forward contract (mirrors ``Qwen3VLPipelinedForCausalLM`` minus the deepstack
    / 3-D grid rope): ``input_ids`` carries text token ids ``< V`` and audio
    placeholder ids ``>= V`` (``id - V`` selects the audio-embed row).
    """

    coreai_externalize_specs: tuple = ()

    def __init__(self, config, n_audio_tokens: int) -> None:
        super().__init__()
        self.config = config
        self.model = _ThinkerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        # Static width of the audio-embed input. Audio placeholder ids occupy
        # ``[V, V + n_audio_tokens)``.
        self.n_audio_tokens = n_audio_tokens

    def forward(
        self,
        input_ids: torch.Tensor,     # [1, s] int32; audio tokens = V + slot
        position_ids: torch.Tensor,  # [1, total] int32 temporal ramp (TMRoPE-collapsed)
        audio_embeds: torch.Tensor,  # [N, h] static input
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        cfg = self.config
        V = cfg.vocab_size
        N = self.n_audio_tokens
        b, s = input_ids.shape

        ids = input_ids
        is_aud = ids >= V                          # [1, s] bool
        slot = (ids - V).clamp(0, N - 1)           # [1, s] int32
        flat_slot = slot.reshape(-1)

        # token embedding: text table where id < V, else the audio-embed row.
        e_txt = m.embed_tokens(ids.clamp(0, V - 1))
        e_aud = audio_embeds.index_select(0, flat_slot).reshape(b, s, -1)
        x = torch.where(is_aud.unsqueeze(-1), e_aud.to(e_txt.dtype), e_txt)

        cache = KVCache(k_cache, v_cache)
        for layer in m.layers:
            x = layer(x, position_ids, cache)
        return self.lm_head(m.norm(x))

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_hf(
        cls,
        hf_id: str,
        n_audio_tokens: int,
        target_dtype: torch.dtype = torch.float16,
        max_context_length: int | None = None,
    ) -> "Qwen2_5OmniThinkerForCausalLM":
        """Load the Thinker text decoder from a Qwen2.5-Omni checkpoint.

        Thinker weights live under the ``thinker.`` prefix:
        ``thinker.model.*`` (decoder) + ``thinker.lm_head.weight`` (untied).
        ``thinker.audio_tower.*`` / ``thinker.visual.*`` are skipped here (the
        audio tower exports to its own bundle).
        """
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig

        model_dir = snapshot_download(
            hf_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
        raw = AutoConfig.from_pretrained(model_dir)
        tcfg = raw.thinker_config.text_config
        # qwen2 attention reads getattr(config, "head_dim", dim // n_heads); the
        # Omni text config stores head_dim=None, so make it concrete (= 128).
        if getattr(tcfg, "head_dim", None) in (None, 0):
            tcfg.head_dim = tcfg.hidden_size // tcfg.num_attention_heads
        # Collapse TMRoPE -> standard 1-D RoPE (see module docstring). The host
        # feeds temporal position_ids; with t==h==w mRoPE == 1-D RoPE.
        tcfg.rope_scaling = None
        if max_context_length is not None:
            tcfg.max_position_embeddings = max_context_length

        model = cls(tcfg, n_audio_tokens=n_audio_tokens).to(dtype=target_dtype)

        prefix = "thinker."
        sd: dict[str, torch.Tensor] = {}
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if not key.startswith(prefix):
                        continue
                    local = key[len(prefix):]  # model.* | lm_head.weight | audio_tower.* | visual.*
                    if local.startswith(("audio_tower", "visual")):
                        continue
                    sd[local] = f.get_tensor(key).to(target_dtype)

        # Fuse per-layer q/k/v (weight + bias) into qkv_proj (qwen2 recipe).
        for i in range(tcfg.num_hidden_layers):
            pre = f"model.layers.{i}.self_attn."
            qkv_w = [sd.pop(pre + p + ".weight") for p in ("q_proj", "k_proj", "v_proj")]
            qkv_b = [sd.pop(pre + p + ".bias") for p in ("q_proj", "k_proj", "v_proj")]
            sd[pre + "qkv_proj.weight"] = torch.cat(qkv_w, dim=0)
            sd[pre + "qkv_proj.bias"] = torch.cat(qkv_b, dim=0)

        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        missing = [k for k in missing if not (tcfg.tie_word_embeddings and k == "lm_head.weight")]
        if missing or unexpected:
            raise RuntimeError(f"thinker load mismatch: missing={missing} unexpected={unexpected}")
        if tcfg.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        model.eval()
        return model

    # -- export -------------------------------------------------------------
    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_query: int = 8,
        trace_past: int = 64,
    ) -> dict:
        cfg = self.config
        N, h = self.n_audio_tokens, cfg.hidden_size
        input_ids = torch.randint(1, cfg.vocab_size, (1, trace_query), dtype=torch.int32)
        position_ids = torch.arange(trace_past + trace_query, dtype=torch.int32).unsqueeze(0)
        k_cache = torch.zeros(
            cfg.num_hidden_layers, 1, cfg.num_key_value_heads, trace_kv_len, cfg.head_dim,
            dtype=target_dtype,
        )
        v_cache = torch.zeros_like(k_cache)
        reference_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "audio_embeds": torch.zeros(N, h, dtype=target_dtype),
            "k_cache": k_cache,
            "v_cache": v_cache,
        }
        seq_pos = torch.export.Dim("seq_pos", min=2, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        # trace_query==1 -> static [1,1] query (oracle gate variant); else dynamic.
        ids_shape = None if trace_query == 1 else {1: torch.export.Dim(
            "seq_ids", min=1, max=max_context_length - 2)}
        dynamic_shapes = {
            "input_ids": ids_shape,
            "position_ids": {1: seq_pos},
            "audio_embeds": None,
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids", "audio_embeds"),
            "output_names": ("logits",),
            "state_names": PIPELINED_STATE_NAMES,
        }
