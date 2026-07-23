# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""GLM-OCR (zai-org/GLM-OCR, 0.9B) shaped for Apple's pipelined GPU engine.

GLM-OCR = a GLM-4.V (Glm4v) family model scaled to OCR: a CogViT ViT vision
tower + a small GLM text decoder. It ports as the zoo's second on-device
doc-OCR (after Unlimited-OCR), reusing the Qwen3-VL rider contract
(``qwen3_vl.py``) with the GLM-specific pieces swapped in.

Two graphs:

* ``GlmOcrVisionEncoder`` — fixed-grid ViT, runs ONCE per image as a plain
  ``.aimodel``: ``patches [n_patch, C*T*P*P] -> image_embeds [N, out_h]``.
  No deepstack (unlike Qwen3-VL); no learned position embedding (Glm4v drops
  it) — only a baked 2D rotary. Merger: downsample conv (2x2) + GLM gated MLP.

* ``GlmOcrPipelinedForCausalLM`` — the GLM text decoder on the UNMODIFIED
  pipelined-engine contract (dynamic query, ids+positions in, one KV pair),
  the multimodal state riding the static-input hook
  (apps/coreai-pipelined-static-inputs.patch):

  ``(input_ids [1,s] dyn, position_ids [1,total] dyn, image_embeds [N,h],
     rope_shift_start [1] i32, rope_shift_amount [1] i32,
     keyCache/valueCache) -> logits [1,s,V]``

  The host rewrites the prompt's image-placeholder ids to EXTENSION ids
  ``V + slot`` (slot = 0..N-1 in patch order); the graph swaps the token
  embedding for ``image_embeds[slot]`` there, and self-locates the 3D M-RoPE
  positions from (ids, position) — identical to the Qwen3-VL rider, minus
  deepstack.

GLM-specific vs Qwen3-VL (all confirmed from transformers modeling_glm_ocr.py):
  * decoder block = GLM "sandwich" norm (input / post_self_attn / post_attn /
    post_mlp), NO qk-norm, fused ``gate_up_proj`` SwiGLU;
  * decoder M-RoPE = SECTIONED [16,24,24] over the 64 half-dims (chunk i -> axis
    i%3), applied INTERLEAVED (even/odd pairs) — not Qwen's split-half;
  * vision block norms are RMSNorm (not LayerNorm), attention carries q/k-norm,
    MLP is gated SiLU with bias; vision rope is split-half (baked constant).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA

PIPELINED_STATE_NAMES = ("keyCache", "valueCache")


def _rp_get(rp, key, default=None):
    """Fetch a key from a rope-params dict OR a namespace-like config object."""
    if rp is None:
        return default
    if isinstance(rp, dict):
        return rp.get(key, default)
    return getattr(rp, key, default)


def _rope_theta(cfg) -> float:
    rp = getattr(cfg, "rope_parameters", None)
    theta = _rp_get(rp, "rope_theta")
    return float(theta) if theta is not None else float(getattr(cfg, "rope_theta", 10000.0))


def _mrope_section(cfg) -> list[int]:
    rp = getattr(cfg, "rope_parameters", None)
    section = _rp_get(rp, "mrope_section")
    if section is not None:
        return list(section)
    return list(cfg.rope_scaling["mrope_section"])


def _interleave2(x: torch.Tensor) -> torch.Tensor:
    """[..., d] -> [..., 2d] duplicating each element (a,b -> a,a,b,b)."""
    return torch.stack((x, x), dim=-1).reshape(*x.shape[:-1], -1)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """GLM interleaved rotate: pairs (x0,x1) -> (-x1, x0). Export-safe reshape
    form of transformers `rotate_half_llm`."""
    xr = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0 = xr[..., 0]
    x1 = xr[..., 1]
    return torch.stack((-x1, x0), dim=-1).reshape(*x.shape)


# ---------------------------------------------------------------------------
# Text decoder
# ---------------------------------------------------------------------------


class GlmOcrTextAttention(nn.Module):
    """GLM attention (separate q/k/v, NO qk-norm) with sectioned interleaved M-RoPE."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.sdpa = SDPA(is_causal=True)

        self.section = _mrope_section(config)  # [16,24,24]
        theta = _rope_theta(config)
        half = self.head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # [half]
        assert sum(self.section) == half, (self.section, half)

    def _cos_sin(self, pos_t, pos_h, pos_w):
        # pos_* : [1, s] int32/float -> per-axis freqs [1, s, half]
        inv = self.inv_freq.to(torch.float32)
        ft = pos_t.float().unsqueeze(-1) * inv
        fh = pos_h.float().unsqueeze(-1) * inv
        fw = pos_w.float().unsqueeze(-1) * inv
        s0, s1, s2 = self.section
        freqs = torch.cat(
            [ft[..., :s0], fh[..., s0:s0 + s1], fw[..., s0 + s1:s0 + s1 + s2]], dim=-1
        )  # [1, s, half]
        cos = _interleave2(freqs.cos())  # [1, s, head_dim]
        sin = _interleave2(freqs.sin())
        return cos.unsqueeze(1), sin.unsqueeze(1)  # [1, 1, s, head_dim]

    def _apply_rope(self, x, cos, sin):
        return (x.float() * cos + _rotate_half_interleaved(x.float()) * sin).to(x.dtype)

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


class GlmOcrTextBlock(nn.Module):
    """GLM sandwich-norm decoder block.

    r = attn(input_ln(x));         x = x + post_self_attn_ln(r)
    r = mlp(post_attention_ln(x)); x = x + post_mlp_ln(r)
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.self_attn = GlmOcrTextAttention(config, layer_idx)
        self.mlp = MLP(hidden, config.intermediate_size)  # fused gate_up split in from_hf
        self.input_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(self, x, position_ids, pos_t, pos_h, pos_w, cache):
        r = self.self_attn(
            self.input_layernorm(x), position_ids, pos_t, pos_h, pos_w, cache)
        x = x + self.post_self_attn_layernorm(r)
        r = self.mlp(self.post_attention_layernorm(x))
        x = x + self.post_mlp_layernorm(r)
        return x


class GlmOcrTextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [GlmOcrTextBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class GlmOcrPipelinedForCausalLM(nn.Module):
    """Engine-shaped GLM-OCR text decoder; see module docstring for contract."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, config, grid_h: int = 16, grid_w: int = 16) -> None:
        super().__init__()
        self.config = config
        self.model = GlmOcrTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
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
    ) -> "GlmOcrPipelinedForCausalLM":
        cfg, sd = _load_glmocr_state_dict(hf_id, "model.language_model.", target_dtype)
        text_cfg = cfg.text_config
        model = cls(text_cfg, grid_h=grid_h, grid_w=grid_w).to(dtype=target_dtype)

        inter = text_cfg.intermediate_size
        n_layers = text_cfg.num_hidden_layers
        # drop the MTP layer (index == num_hidden_layers, num_nextn_predict_layers=1)
        sd = {k: v for k, v in sd.items() if not k.startswith(f"layers.{n_layers}.")}
        out = {}
        for i in range(n_layers):
            pre = f"layers.{i}."
            # fused gate_up_proj -> primitive MLP separate gate/up (gate = first half)
            gu = sd.pop(pre + "mlp.gate_up_proj.weight")
            out[f"model.{pre}mlp.gate_proj.weight"] = gu[:inter]
            out[f"model.{pre}mlp.up_proj.weight"] = gu[inter:]
        for k, v in sd.items():
            out[k if k == "lm_head.weight" else "model." + k] = v
        if getattr(text_cfg, "tie_word_embeddings", False):
            out.pop("lm_head.weight", None)

        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        missing = [k for k in missing if not k.endswith(("inv_freq", "lm_head.weight"))]
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        if getattr(text_cfg, "tie_word_embeddings", False):
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
# Vision encoder (fixed grid)
# ---------------------------------------------------------------------------


class _VisionAttention(nn.Module):
    """CogViT full attention: fused qkv (bias) + q/k RMSNorm + baked 2D rope."""

    def __init__(self, vcfg) -> None:
        super().__init__()
        self.num_heads = vcfg.num_heads
        self.head_dim = vcfg.hidden_size // vcfg.num_heads
        bias = vcfg.attention_bias
        self.qkv = nn.Linear(vcfg.hidden_size, vcfg.hidden_size * 3, bias=bias)
        self.proj = nn.Linear(vcfg.hidden_size, vcfg.hidden_size, bias=bias)
        self.q_norm = RMSNorm(self.head_dim, eps=vcfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=vcfg.rms_norm_eps)
        self.sdpa = SDPA(is_causal=False)

    def forward(self, x, cos, sin):
        # x [n, d]; cos/sin [1, 1, n, head_dim]
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(1, 2, 0, 3).unbind(0)  # each [heads, n, hd]
        q = self.q_norm(q).unsqueeze(0)  # [1, heads, n, hd]
        k = self.k_norm(k).unsqueeze(0)
        v = v.unsqueeze(0)

        def rot(t):
            t1 = t[..., : self.head_dim // 2]
            t2 = t[..., self.head_dim // 2:]
            return torch.cat((-t2, t1), dim=-1)

        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        out = self.sdpa(q, k, v)  # [1, heads, n, hd]
        out = out.permute(0, 2, 1, 3).reshape(n, self.num_heads * self.head_dim)
        return self.proj(out)


class _VisionBlock(nn.Module):
    def __init__(self, vcfg) -> None:
        super().__init__()
        self.norm1 = RMSNorm(vcfg.hidden_size, eps=vcfg.rms_norm_eps)
        self.norm2 = RMSNorm(vcfg.hidden_size, eps=vcfg.rms_norm_eps)
        self.attn = _VisionAttention(vcfg)
        self.mlp = MLP(vcfg.hidden_size, vcfg.intermediate_size, bias=vcfg.attention_bias)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class _VisionMerger(nn.Module):
    """GLM merger: proj -> LayerNorm -> GELU -> gated SiLU MLP (dim -> context -> dim)."""

    def __init__(self, dim: int, context_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=False)
        self.up_proj = nn.Linear(dim, context_dim, bias=False)
        self.down_proj = nn.Linear(context_dim, dim, bias=False)
        self.act1 = nn.GELU()
        self.act_fn = nn.functional.silu if hidden_act == "silu" else nn.functional.gelu

    def forward(self, x):
        x = self.proj(x)
        x = self.act1(self.post_projection_norm(x))
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GlmOcrVisionEncoder(nn.Module):
    """Fixed-grid GLM-OCR (CogViT) vision tower.

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
        self.n_patches = (grid_h * merge) * (grid_w * merge)
        patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2

        self.patch_proj = nn.Linear(patch_dim, vcfg.hidden_size, bias=True)
        self.blocks = nn.ModuleList([_VisionBlock(vcfg) for _ in range(vcfg.depth)])
        self.post_layernorm = RMSNorm(vcfg.hidden_size, eps=vcfg.rms_norm_eps)
        # downsample conv (2x2) baked as a linear over the flattened 2x2 block
        self.downsample = nn.Linear(
            vcfg.hidden_size * merge * merge, vcfg.out_hidden_size, bias=True)
        self.merger = _VisionMerger(
            dim=vcfg.out_hidden_size,
            context_dim=vcfg.out_hidden_size * vcfg.in_channels,
            hidden_act=vcfg.hidden_act,
        )

        head_dim = vcfg.hidden_size // vcfg.num_heads
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
        x = self.post_layernorm(x)
        # merge 2x2 blocks: [n_patch, h] -> [n_merged, merge*merge*h] -> downsample
        h = x.shape[-1]
        x = x.view(-1, self.merge, self.merge, h).permute(0, 3, 1, 2).reshape(-1, self.merge * self.merge * h)
        x = self.downsample(x)
        return self.merger(x)

    # -- constants ----------------------------------------------------------

    def _init_positional_constants(self) -> None:
        """Bake the block-major 2D rotary (GlmOcrVisionRotaryEmbedding +
        rot_pos_emb) for the fixed grid, in fp32 — matches HF exactly."""
        vcfg = self.vcfg
        merge = self.merge
        H, W = self.grid_h * merge, self.grid_w * merge
        head_dim = vcfg.hidden_size // vcfg.num_heads
        rot_dim = head_dim // 2  # GlmOcrVisionRotaryEmbedding(dim=head_dim//2)
        inv = 1.0 / (10000.0 ** (torch.arange(0, rot_dim, 2).float() / rot_dim))  # [rot_dim/2]

        # rot_pos_emb: block-major (h,w) pos ids
        hpos = torch.arange(H).unsqueeze(1).expand(-1, W)
        hpos = hpos.reshape(H // merge, merge, W // merge, merge).permute(0, 2, 1, 3).flatten()
        wpos = torch.arange(W).unsqueeze(0).expand(H, -1)
        wpos = wpos.reshape(H // merge, merge, W // merge, merge).permute(0, 2, 1, 3).flatten()
        # rotary_pos_emb_full[pos].flatten(1): per patch [h_freqs, w_freqs]
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
    ) -> "GlmOcrVisionEncoder":
        cfg, sd = _load_glmocr_state_dict(hf_id, "model.visual.", torch.float32)
        vcfg = cfg.vision_config
        model = cls(vcfg, grid_h=grid_h, grid_w=grid_w).float()

        out = {}
        # Conv3d patch_embed -> linear over flattened patch
        conv = sd.pop("patch_embed.proj.weight")  # [d, C, T, P, P]
        out["patch_proj.weight"] = conv.reshape(conv.shape[0], -1)
        out["patch_proj.bias"] = sd.pop("patch_embed.proj.bias")
        # Conv2d downsample -> linear over flattened (C, kh, kw)
        dsw = sd.pop("downsample.weight")  # [out, in, k, k]
        out["downsample.weight"] = dsw.reshape(dsw.shape[0], -1)
        out["downsample.bias"] = sd.pop("downsample.bias")
        for k, v in sd.items():
            out[k] = v

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
    hard dependency on a transformers build that recognizes ``glm_ocr``."""

    def __init__(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, _Cfg(v) if isinstance(v, dict) else v)


def _load_glmocr_config(model_dir: str) -> "_Cfg":
    import json
    import os

    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    cfg = _Cfg(raw)
    # defaults not always present in config.json
    vc = cfg.vision_config
    if not hasattr(vc, "in_channels"):
        vc.in_channels = 3
    if not hasattr(cfg.text_config, "tie_word_embeddings"):
        cfg.text_config.tie_word_embeddings = False
    return cfg


def _load_glmocr_state_dict(hf_id: str, prefix: str, dtype: torch.dtype):
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
    cfg = _load_glmocr_config(model_dir)
    sd = {}
    for path in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if key.startswith(prefix):
                    sd[key.removeprefix(prefix)] = f.get_tensor(key).to(dtype)
                elif prefix == "model.language_model." and key == "lm_head.weight":
                    sd["lm_head.weight"] = f.get_tensor(key).to(dtype)
    return cfg, sd
