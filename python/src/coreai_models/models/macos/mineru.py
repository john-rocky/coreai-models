# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""MinerU2.5-Pro (opendatalab/MinerU2.5-Pro-2605-1.2B, Apache-2.0) shaped for
Apple's pipelined GPU engine — the zoo's whole-page doc-OCR rider.

MinerU2.5 is a *stock* Qwen2-VL (``Qwen2VLForConditionalGeneration``): a
Qwen2-VL ViT vision tower + a Qwen2-0.5B text decoder, no custom code. The value
is the 2-stage host pipeline (layout detection -> per-region recognition ->
``json2md``); the model itself is a plain Qwen2-VL, so it rides the GLM-OCR /
Qwen3-VL contract (``glm_ocr.py`` / ``qwen3_vl.py``) with the Qwen2 specifics
swapped in.

Two graphs:

* ``MinerUVisionEncoder`` — fixed-grid Qwen2-VL ViT, runs ONCE per image as a
  plain ``.aimodel``: ``patches [n_patch, C*T*P*P] -> image_embeds [N, out_h]``.
  No deepstack (unlike Qwen3-VL); no learned position embedding (Qwen2-VL uses a
  2D rotary only) — only baked cos/sin. LayerNorm blocks, ``quick_gelu`` MLP
  (non-gated ``fc1 -> fc2``), and the standard Qwen2-VL ``PatchMerger``
  (``ln_q -> view -> Linear -> GELU -> Linear``).

* ``MinerUPipelinedForCausalLM`` — the Qwen2 text decoder on the UNMODIFIED
  pipelined-engine contract (dynamic query, ids+positions in, one KV pair), the
  multimodal state riding the static-input hook
  (apps/coreai-pipelined-static-inputs.patch):

  ``(input_ids [1,s] dyn, position_ids [1,total] dyn, image_embeds [N,h],
     rope_shift_start [1] i32, rope_shift_amount [1] i32,
     keyCache/valueCache) -> logits [1,s,V]``

  Identical rider to GLM-OCR (the host rewrites the prompt's image-placeholder
  ids to EXTENSION ids ``V + slot``; in-graph the token embedding is swapped for
  ``image_embeds[slot]`` and the 3D M-RoPE positions self-locate from
  ``(ids, position)``), minus deepstack.

Qwen2 vs GLM-OCR / Qwen3-VL (all from config.json + modeling_qwen2_vl.py):
  * decoder attention = separate q/k/v WITH bias, NO qk-norm, ``o_proj`` no bias;
  * decoder M-RoPE = SECTIONED [8,12,12] over the 32 half-dims, applied
    SPLIT-HALF (Qwen standard ``rotate_half``) — not GLM's interleaved pairs and
    not Qwen3-VL's ``j % 3`` interleave;
  * decoder block = standard 2-norm (input / post_attention), silu SwiGLU;
  * ``tie_word_embeddings = True`` (no ``lm_head`` weight in the checkpoint);
  * vision block norms are LayerNorm, attention carries qkv bias + NO q/k-norm,
    MLP is non-gated ``fc1 -> quick_gelu -> fc2``; merger = Qwen2-VL
    ``PatchMerger``; vision rope is split-half (baked constant).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA

PIPELINED_STATE_NAMES = ("keyCache", "valueCache")


def _rope_theta(cfg) -> float:
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    return float(getattr(cfg, "rope_theta", 1000000.0))


def _mrope_section(cfg) -> list[int]:
    rs = getattr(cfg, "rope_scaling", None) or getattr(cfg, "rope_parameters", None)
    if rs is None:
        raise ValueError("config has no rope_scaling / rope_parameters")
    if isinstance(rs, dict):
        return list(rs["mrope_section"])
    return list(rs.mrope_section)  # _Cfg namespace


def _quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """Qwen2-VL vision activation: x * sigmoid(1.702 * x)."""
    return x * torch.sigmoid(1.702 * x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen standard split-half rotate: [x1, x2] -> [-x2, x1]."""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


# ---------------------------------------------------------------------------
# Text decoder (Qwen2-0.5B)
# ---------------------------------------------------------------------------


class MinerUTextAttention(nn.Module):
    """Qwen2 attention: separate q/k/v (bias), NO qk-norm, sectioned split-half M-RoPE."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.sdpa = SDPA(is_causal=True)

        self.section = _mrope_section(config)  # [8, 12, 12]
        theta = _rope_theta(config)
        half = self.head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # [half]
        assert sum(self.section) == half, (self.section, half)

    def _cos_sin(self, pos_t, pos_h, pos_w):
        # pos_* : [1, s] -> per-axis freqs [1, s, half], sectioned then split-half tiled
        inv = self.inv_freq.to(torch.float32)
        ft = pos_t.float().unsqueeze(-1) * inv
        fh = pos_h.float().unsqueeze(-1) * inv
        fw = pos_w.float().unsqueeze(-1) * inv
        s0, s1, s2 = self.section
        freqs = torch.cat(
            [ft[..., :s0], fh[..., s0:s0 + s1], fw[..., s0 + s1:s0 + s1 + s2]], dim=-1
        )  # [1, s, half]
        emb = torch.cat([freqs, freqs], dim=-1)  # [1, s, head_dim]
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)  # [1, 1, s, head_dim]

    def _apply_rope(self, x, cos, sin):
        return (x.float() * cos + _rotate_half(x.float()) * sin).to(x.dtype)

    def forward(self, x, position_ids, pos_t, pos_h, pos_w, cache):
        b, q_len, _ = x.shape
        nh, nkv, hd = self.n_heads, self.n_kv_heads, self.head_dim

        query = self.q_proj(x).reshape(b, q_len, nh, hd).permute(0, 2, 1, 3)
        key = self.k_proj(x).reshape(b, q_len, nkv, hd).permute(0, 2, 1, 3)
        value = self.v_proj(x).reshape(b, q_len, nkv, hd).permute(0, 2, 1, 3)

        cos, sin = self._cos_sin(pos_t, pos_h, pos_w)
        query = self._apply_rope(query, cos, sin)
        key = self._apply_rope(key, cos, sin)

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
            .reshape(b, q_len, nh * hd)
        )
        return self.o_proj(out)


class MinerUTextBlock(nn.Module):
    """Standard Qwen2 2-norm decoder block."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.self_attn = MinerUTextAttention(config, layer_idx)
        self.mlp = MLP(hidden, config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(self, x, position_ids, pos_t, pos_h, pos_w, cache):
        r = self.self_attn(
            self.input_layernorm(x), position_ids, pos_t, pos_h, pos_w, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class MinerUTextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MinerUTextBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MinerUPipelinedForCausalLM(nn.Module):
    """Engine-shaped MinerU2.5 (Qwen2) text decoder; see module docstring."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, config, grid_h: int = 16, grid_w: int = 16) -> None:
        super().__init__()
        self.config = config
        self.model = MinerUTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight
        # merged-grid geometry (vision tokens after 2x2 merge)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_image_tokens = grid_h * grid_w

    def forward(
        self,
        input_ids: torch.Tensor,        # [1, s] int32; image tokens = V + slot
        position_ids: torch.Tensor,     # [1, total] int32 ramp
        image_embeds: torch.Tensor,     # [N, h] static input
        rope_shift_start: torch.Tensor, # [1] int32 static input
        rope_shift_amount: torch.Tensor,# [1] int32 static input
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        V = self.config.vocab_size
        N = self.n_image_tokens
        b, s = input_ids.shape

        seq_len = position_ids.shape[-1]
        torch._check_is_size(s)
        torch._check_is_size(seq_len)
        offset = seq_len - s
        torch._check_is_size(offset)
        p = position_ids.narrow(-1, offset, s)  # [1, s] int32

        ids = input_ids
        is_img = ids >= V                                  # [1, s] bool
        slot = (ids - V).clamp(0, N - 1)                   # [1, s] int32
        flat_slot = slot.reshape(-1)

        # --- token embedding: text table or image slot ---
        e_txt = m.embed_tokens(ids.clamp(0, V - 1))
        e_img = image_embeds.index_select(0, flat_slot).reshape(b, s, -1)
        x = torch.where(is_img.unsqueeze(-1), e_img.to(e_txt.dtype), e_txt)

        # --- 3D rope positions from (ids, p) ---
        shift = torch.where(
            p >= rope_shift_start, rope_shift_amount, torch.zeros_like(p))
        p_text = p - shift
        s0 = p - slot  # image start position (valid where is_img)
        row = torch.div(slot, self.grid_w, rounding_mode="floor")
        col = slot - row * self.grid_w
        pos_t = torch.where(is_img, s0, p_text)
        pos_h = torch.where(is_img, s0 + row, p_text)
        pos_w = torch.where(is_img, s0 + col, p_text)

        cache = KVCache(k_cache, v_cache)
        for layer in m.layers:
            x = layer(x, position_ids, pos_t, pos_h, pos_w, cache)

        hidden = m.norm(x)
        return self.lm_head(hidden)

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_id: str,
        target_dtype: torch.dtype = torch.float16,
        grid_h: int = 16,
        grid_w: int = 16,
    ) -> "MinerUPipelinedForCausalLM":
        """Load the Qwen2 text decoder from a MinerU2.5 checkpoint (keys under
        ``model.``)."""
        cfg, sd = _load_mineru_state_dict(hf_id, "model.", target_dtype)
        text_cfg = cfg.text_config
        model = cls(text_cfg, grid_h=grid_h, grid_w=grid_w).to(dtype=target_dtype)

        # Qwen2 decoder keys map 1:1 onto the re-authored tree under `model.`.
        out = {"model." + k: v for k, v in sd.items() if k != "lm_head.weight"}
        if not getattr(text_cfg, "tie_word_embeddings", True):
            if "lm_head.weight" in sd:
                out["lm_head.weight"] = sd["lm_head.weight"]

        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        missing = [k for k in missing if not k.endswith(("inv_freq", "lm_head.weight"))]
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        if getattr(text_cfg, "tie_word_embeddings", True):
            model.lm_head.weight = model.model.embed_tokens.weight
        model.eval()
        return model

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_query: int = 8,
        trace_past: int = 64,
        static_ids: bool | None = None,
    ) -> dict:
        cfg = self.config
        N, h = self.n_image_tokens, cfg.hidden_size
        input_ids = torch.randint(1, cfg.vocab_size, (1, trace_query), dtype=torch.int32)
        position_ids = torch.arange(
            trace_past + trace_query, dtype=torch.int32).unsqueeze(0)
        k_cache = torch.zeros(
            cfg.num_hidden_layers, 1, cfg.num_key_value_heads,
            trace_kv_len, cfg.head_dim, dtype=target_dtype)
        v_cache = torch.zeros_like(k_cache)

        reference_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "image_embeds": torch.zeros(N, h, dtype=target_dtype),
            "rope_shift_start": torch.tensor([1 << 30], dtype=torch.int32),
            "rope_shift_amount": torch.tensor([0], dtype=torch.int32),
            "k_cache": k_cache,
            "v_cache": v_cache,
        }
        if static_ids is None:
            static_ids = trace_query == 1
        pos_min = max(2, trace_query) if static_ids else 2
        seq_pos = torch.export.Dim("seq_pos", min=pos_min, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        if static_ids:
            ids_shape = None
        else:
            ids_shape = {1: torch.export.Dim("seq_ids", min=1, max=max_context_length - 2)}
        dynamic_shapes = {
            "input_ids": ids_shape,
            "position_ids": {1: seq_pos},
            "image_embeds": None,
            "rope_shift_start": None,
            "rope_shift_amount": None,
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": (
                "input_ids", "position_ids", "image_embeds",
                "rope_shift_start", "rope_shift_amount"),
            "output_names": ("logits",),
            "state_names": PIPELINED_STATE_NAMES,
        }


# ---------------------------------------------------------------------------
# Vision encoder (fixed grid) — Qwen2-VL ViT
# ---------------------------------------------------------------------------


class _VisionAttention(nn.Module):
    """Qwen2-VL ViT full attention: fused qkv (bias), NO q/k-norm, split-half rope."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.sdpa = SDPA(is_causal=False)

    def forward(self, x, cos, sin):
        # x [n, d]; cos/sin [1, 1, n, head_dim]
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3).unsqueeze(1)  # [3, 1, heads, n, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        def rot(t):
            t1 = t[..., : self.head_dim // 2]
            t2 = t[..., self.head_dim // 2:]
            return torch.cat((-t2, t1), dim=-1)

        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        out = self.sdpa(q, k, v)  # [1, heads, n, hd]
        out = out.permute(0, 2, 1, 3).reshape(n, self.num_heads * self.head_dim)
        return self.proj(out)


class _VisionMLP(nn.Module):
    """Non-gated Qwen2-VL ViT MLP: fc1 -> quick_gelu -> fc2."""

    def __init__(self, embed_dim: int, intermediate: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate, bias=True)
        self.fc2 = nn.Linear(intermediate, embed_dim, bias=True)

    def forward(self, x):
        return self.fc2(_quick_gelu(self.fc1(x)))


class _VisionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, intermediate: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = _VisionAttention(embed_dim, num_heads)
        self.mlp = _VisionMLP(embed_dim, intermediate)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class _VisionMerger(nn.Module):
    """Qwen2-VL PatchMerger: ln_q(pre-shuffle) -> view(merge^2) -> Linear -> GELU -> Linear."""

    def __init__(self, embed_dim: int, out_hidden: int, merge: int) -> None:
        super().__init__()
        self.hidden_size = embed_dim * (merge ** 2)
        self.ln_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, out_hidden),
        )

    def forward(self, x):
        # x [n_patch, embed_dim]; block-major patches -> concat 2x2 -> [n_merged, embed_dim*4]
        x = self.ln_q(x).view(-1, self.hidden_size)
        return self.mlp(x)


class MinerUVisionEncoder(nn.Module):
    """Fixed-grid Qwen2-VL vision tower.

    ``patches [n_patch, in_ch*T*P*P] -> image_embeds [N, out_h]`` with
    n_patch = (2*grid_h)*(2*grid_w) in block-major patch order and
    N = grid_h*grid_w merged tokens.
    """

    def __init__(self, vcfg, grid_h: int = 16, grid_w: int = 16) -> None:
        super().__init__()
        self.vcfg = vcfg
        self.grid_h, self.grid_w = grid_h, grid_w
        merge = vcfg.spatial_merge_size
        self.merge = merge
        embed_dim = vcfg.embed_dim
        num_heads = vcfg.num_heads
        intermediate = embed_dim * vcfg.mlp_ratio
        out_hidden = vcfg.hidden_size  # merger output = decoder hidden
        self.n_patches = (grid_h * merge) * (grid_w * merge)
        patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2

        self.patch_proj = nn.Linear(patch_dim, embed_dim, bias=False)  # Qwen2-VL: no bias
        self.blocks = nn.ModuleList(
            [_VisionBlock(embed_dim, num_heads, intermediate) for _ in range(vcfg.depth)])
        self.merger = _VisionMerger(embed_dim, out_hidden, merge)

        head_dim = embed_dim // num_heads
        self.register_buffer(
            "cos_const", torch.zeros(1, 1, self.n_patches, head_dim), persistent=False)
        self.register_buffer(
            "sin_const", torch.zeros(1, 1, self.n_patches, head_dim), persistent=False)

    def forward(self, patches: torch.Tensor):
        x = self.patch_proj(patches)
        cos = self.cos_const.to(x.dtype)
        sin = self.sin_const.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.merger(x)

    # -- constants ----------------------------------------------------------

    def _init_positional_constants(self) -> None:
        """Bake the block-major 2D rotary (Qwen2VLVisionRotaryEmbedding +
        rot_pos_emb) for the fixed grid, in fp32 — matches HF exactly."""
        vcfg = self.vcfg
        merge = self.merge
        H, W = self.grid_h * merge, self.grid_w * merge
        head_dim = vcfg.embed_dim // vcfg.num_heads
        rot_dim = head_dim // 2  # VisionRotaryEmbedding(dim=head_dim//2)
        inv = 1.0 / (10000.0 ** (torch.arange(0, rot_dim, 2).float() / rot_dim))  # [rot_dim/2]

        # rot_pos_emb: block-major (h, w) pos ids
        hpos = torch.arange(H).unsqueeze(1).expand(-1, W)
        hpos = hpos.reshape(H // merge, merge, W // merge, merge).permute(0, 2, 1, 3).flatten()
        wpos = torch.arange(W).unsqueeze(0).expand(H, -1)
        wpos = wpos.reshape(H // merge, merge, W // merge, merge).permute(0, 2, 1, 3).flatten()
        max_grid = max(H, W)
        full = torch.outer(torch.arange(max_grid).float(), inv)  # [max_grid, rot_dim/2]
        rpe = torch.cat([full[hpos], full[wpos]], dim=-1)  # [n, rot_dim]
        emb = torch.cat([rpe, rpe], dim=-1)  # [n, head_dim]
        self.cos_const.copy_(emb.cos().view(1, 1, self.n_patches, head_dim))
        self.sin_const.copy_(emb.sin().view(1, 1, self.n_patches, head_dim))

    @classmethod
    def from_hf(
        cls,
        hf_id: str,
        target_dtype: torch.dtype = torch.float16,
        grid_h: int = 16,
        grid_w: int = 16,
    ) -> "MinerUVisionEncoder":
        cfg, sd = _load_mineru_state_dict(hf_id, "visual.", torch.float32)
        vcfg = cfg.vision_config
        model = cls(vcfg, grid_h=grid_h, grid_w=grid_w).float()

        out = {}
        # Conv3d patch_embed -> linear over flattened patch (no bias in Qwen2-VL)
        conv = sd.pop("patch_embed.proj.weight")  # [d, C, T, P, P]
        out["patch_proj.weight"] = conv.reshape(conv.shape[0], -1)
        for k, v in sd.items():
            out[k] = v  # blocks.* and merger.* map 1:1

        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        missing = [k for k in missing if not k.endswith(("cos_const", "sin_const"))]
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        model._init_positional_constants()
        model = model.to(dtype=target_dtype)
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


class _Cfg:
    """Lightweight config namespace parsed straight from config.json — avoids a
    hard dependency on a transformers build that recognizes ``qwen2_vl``."""

    def __init__(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, _Cfg(v) if isinstance(v, dict) else v)


def _load_mineru_config(model_dir: str) -> "_Cfg":
    import json
    import os

    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    cfg = _Cfg(raw)
    tc = cfg.text_config
    if not hasattr(tc, "head_dim"):
        tc.head_dim = tc.hidden_size // tc.num_attention_heads
    if not hasattr(tc, "tie_word_embeddings"):
        tc.tie_word_embeddings = True
    vc = cfg.vision_config
    if not hasattr(vc, "in_channels"):
        vc.in_channels = getattr(vc, "in_chans", 3)
    # rope_scaling parsed as _Cfg -> expose mrope_section for the helpers
    return cfg


def _load_mineru_state_dict(hf_id: str, prefix: str, dtype: torch.dtype):
    """(config, {stripped_key: tensor}) for keys under ``prefix``.

    Accepts a local dir or an HF id. ``lm_head.weight`` (top-level, untied) is
    surfaced under the decoder prefix.
    """
    import glob
    import os

    from safetensors import safe_open

    if os.path.isdir(hf_id):
        model_dir = hf_id
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(
            hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"])
    cfg = _load_mineru_config(model_dir)
    sd = {}
    for path in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if key.startswith(prefix):
                    sd[key.removeprefix(prefix)] = f.get_tensor(key).to(dtype)
                elif prefix == "model." and key == "lm_head.weight":
                    sd["lm_head.weight"] = f.get_tensor(key).to(dtype)
    return cfg, sd
