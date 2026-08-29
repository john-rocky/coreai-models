# Community port — NOT an Apple model. BSD-3-Clause (see LICENSE).

"""Qwen3-VL shaped for Apple's pipelined GPU engine (text decoder) + a static
vision encoder — the zoo's first VLM rider.

Two graphs:

* ``Qwen3VLVisionEncoder`` — fixed-grid ViT, runs ONCE per image as a plain
  ``.aimodel``: ``patches [n_patch, C*T*P*P] -> (image_embeds [N, h],
  deepstack_embeds [3N, h])``. Everything positional (bilinear pos-embed
  interpolation, 2D rotary cos/sin) is a constant at a fixed grid.

* ``Qwen3VLPipelinedForCausalLM`` — the text decoder on the UNMODIFIED
  pipelined-engine contract (dynamic query, ids+positions in, one KV pair),
  with the multimodal state riding the static-input hook
  (apps/coreai-pipelined-static-inputs.patch):

  ``(input_ids [1,s] dyn, position_ids [1,total] dyn,
     image_embeds [N,h], deepstack_embeds [3N,h],
     rope_shift_start [1] i32, rope_shift_amount [1] i32,
     keyCache/valueCache) -> logits [1,s,V]``

  The host rewrites the prompt's ``<|image_pad|>`` ids to EXTENSION ids
  ``V + slot`` (slot = 0..N-1 in patch order). In-graph, per token:

  - embedding   = ids < V ? embed_tokens[ids] : image_embeds[ids - V]
  - deepstack   = layers 0..2 add deepstack_embeds[k*N + slot] at image tokens
  - M-RoPE      = interleaved 3D rope computed from (ids, position) alone:
      image tokens self-locate (slot = ids - V, image start s0 = pos - slot,
      t = s0, h = s0 + slot//W, w = s0 + slot%W); text tokens use
      pos - shift where shift = rope_shift_amount for pos >= rope_shift_start
      (the host sets these per conversation: image consumes only max(H,W)
      rope positions, so post-image text shifts by N - max(H,W); text-only
      chats bind shift_start = 1<<30). Since t==h==w for text, the three
      masked rotations collapse to standard RoPE there — the interleave
      pattern is just three constant 0/1 masks over head_dim picking which
      frequency comes from t, h or w (freq j: j%3==1 and j<3*sec_h -> h,
      j%3==2 and j<3*sec_w -> w, else t).

  Pure attention (no SSM): NO extra states, NO per-token host inputs — the
  engine's native chunked prefill and full pipeline depth both survive. With
  zero image embeds and shift_start at infinity the graph IS a plain Qwen3
  text decoder (llm-benchmark works out of the box).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA

PIPELINED_STATE_NAMES = ("keyCache", "valueCache")


def _rope_theta(cfg) -> float:
    rp = getattr(cfg, "rope_parameters", None)
    if rp and "rope_theta" in rp:
        return float(rp["rope_theta"])
    return float(cfg.rope_theta)


def _mrope_section(cfg) -> list[int]:
    rp = getattr(cfg, "rope_parameters", None)
    if rp and "mrope_section" in rp:
        return list(rp["mrope_section"])
    return list(cfg.rope_scaling["mrope_section"])


def mrope_masks(head_dim: int, section: list[int]) -> torch.Tensor:
    """[3, head_dim] 0/1 masks selecting which frequency rotates by t/h/w.

    Interleaved layout over the head_dim//2 frequencies (then tiled x2 for the
    rotate-half cos/sin layout): freq j comes from H if j % 3 == 1 and
    j < 3*section[1], from W if j % 3 == 2 and j < 3*section[2], else from T.
    """
    half = head_dim // 2
    m = torch.zeros(3, half)
    m[1, 1 : 3 * section[1] : 3] = 1.0
    m[2, 2 : 3 * section[2] : 3] = 1.0
    m[0] = 1.0 - m[1] - m[2]
    return torch.cat([m, m], dim=-1)  # [3, head_dim]


# ---------------------------------------------------------------------------
# Text decoder
# ---------------------------------------------------------------------------


class Qwen3VLTextAttention(nn.Module):
    """Qwen3 attention (fused qkv + fused qk-norm) with interleaved M-RoPE."""

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = config.head_dim

        self.qkv_proj = nn.Linear(
            dim, (n_heads + 2 * n_kv_heads) * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.qk_norm = RMSNorm(
            head_dim, eps=config.rms_norm_eps, n_heads=n_heads + n_kv_heads)
        self.sdpa = SDPA(is_causal=True)
        self.rope = initialize_rope(base=_rope_theta(config))

        masks = mrope_masks(head_dim, _mrope_section(config))
        # [1, 1, 1, head_dim] broadcast over [b, heads, s, head_dim]
        self.register_buffer("mask_t", masks[0].view(1, 1, 1, -1), persistent=False)
        self.register_buffer("mask_h", masks[1].view(1, 1, 1, -1), persistent=False)
        self.register_buffer("mask_w", masks[2].view(1, 1, 1, -1), persistent=False)

    def _mrope(self, x: torch.Tensor, pos_t, pos_h, pos_w) -> torch.Tensor:
        rt = self.rope(x, position_ids=pos_t)
        rh = self.rope(x, position_ids=pos_h)
        rw = self.rope(x, position_ids=pos_w)
        return rt * self.mask_t + rh * self.mask_h + rw * self.mask_w

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,  # [1, total] ramp (cache indexing)
        pos_t: torch.Tensor,         # [1, s] rope positions (current tokens)
        pos_h: torch.Tensor,
        pos_w: torch.Tensor,
        cache: KVCache,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        qkv = (
            self.qkv_proj(x)
            .reshape(batch_size, query_len, n_heads + 2 * n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        query_key = qkv.narrow(1, 0, n_heads + n_kv_heads)
        value = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)
        query_key = self.qk_norm(query_key)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)

        query_key = self._mrope(query_key, pos_t, pos_h, pos_w)
        query = query_key.narrow(1, 0, n_heads)
        key = query_key.narrow(1, n_heads, n_kv_heads)

        key, value = cache.update_and_fetch(
            self.layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
        )

        out = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, n_heads * self.head_dim)
        )
        return self.o_proj(out)


class Qwen3VLTextBlock(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.self_attn = Qwen3VLTextAttention(config, layer_idx)
        self.mlp = MLP(hidden, config.intermediate_size)
        self.input_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=config.rms_norm_eps)

    def forward(self, x, position_ids, pos_t, pos_h, pos_w, cache):
        r = self.self_attn(
            self.input_layernorm(x), position_ids, pos_t, pos_h, pos_w, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3VLTextModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3VLTextBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3VLPipelinedForCausalLM(nn.Module):
    """Engine-shaped Qwen3-VL text decoder; see module docstring for contract."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, config, grid_h: int = 14, grid_w: int = 14) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3VLTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        # merged-grid geometry (vision tokens after 2x2 merge)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_image_tokens = grid_h * grid_w
        self.n_deepstack = 3

    def forward(
        self,
        input_ids: torch.Tensor,        # [1, s] int32; image tokens = V + slot
        position_ids: torch.Tensor,     # [1, total] int32 ramp
        image_embeds: torch.Tensor,     # [N, h] static input
        deepstack_embeds: torch.Tensor, # [3N, h] static input
        rope_shift_start: torch.Tensor, # [1] int32 static input
        rope_shift_amount: torch.Tensor,# [1] int32 static input
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        cfg = self.config
        V = cfg.vocab_size
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

        img_f = is_img.unsqueeze(-1).to(x.dtype)  # [1, s, 1]

        cache = KVCache(k_cache, v_cache)
        for i, layer in enumerate(m.layers):
            x = layer(x, position_ids, pos_t, pos_h, pos_w, cache)
            if i < self.n_deepstack:
                ds = deepstack_embeds.index_select(0, flat_slot + i * N)
                x = x + ds.reshape(b, s, -1).to(x.dtype) * img_f

        hidden = m.norm(x)
        return self.lm_head(hidden)

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_id: str,
        target_dtype: torch.dtype = torch.float16,
        grid_h: int = 14,
        grid_w: int = 14,
    ) -> "Qwen3VLPipelinedForCausalLM":
        """Load the text decoder from a Qwen3-VL checkpoint
        (keys live under ``model.language_model.``)."""
        cfg, sd = _load_vl_state_dict(hf_id, "model.language_model.", target_dtype)
        text_cfg = cfg.text_config
        model = cls(text_cfg, grid_h=grid_h, grid_w=grid_w)
        model = model.to(dtype=target_dtype)

        # remap to the re-authored tree + fuse qkv / qk_norm (qwen3 recipe)
        out = {}
        n_layers = text_cfg.num_hidden_layers
        for i in range(n_layers):
            pre = f"layers.{i}.self_attn."
            qw = sd.pop(pre + "q_proj.weight")
            kw = sd.pop(pre + "k_proj.weight")
            vw = sd.pop(pre + "v_proj.weight")
            out[f"model.{pre}qkv_proj.weight"] = torch.cat([qw, kw, vw], dim=0)
            qn = sd.pop(pre + "q_norm.weight")
            kn = sd.pop(pre + "k_norm.weight")
            hd = text_cfg.head_dim
            nh = text_cfg.num_attention_heads
            nkv = text_cfg.num_key_value_heads
            fused = torch.cat(
                [
                    qn.view(1, 1, hd).expand(nh, 1, hd),
                    kn.view(1, 1, hd).expand(nkv, 1, hd),
                ],
                dim=0,
            )
            out[f"model.{pre}qk_norm.weight"] = fused.contiguous()
        for k, v in sd.items():
            # lm_head is a top-level module (untied models, e.g. 8B); everything
            # else lives under `model.`. Tied models have no lm_head in sd.
            out[k if k == "lm_head.weight" else "model." + k] = v
        if text_cfg.tie_word_embeddings:
            out.pop("lm_head.weight", None)
        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        missing = [k for k in missing if not k.endswith(
            ("mask_t", "mask_h", "mask_w", "lm_head.weight"))]
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        if text_cfg.tie_word_embeddings:
            # assign=True replaced the embed parameter object — re-tie the head.
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
        input_ids = torch.randint(
            1, cfg.vocab_size, (1, trace_query), dtype=torch.int32)
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
            "deepstack_embeds": torch.zeros(3 * N, h, dtype=target_dtype),
            "rope_shift_start": torch.tensor([1 << 30], dtype=torch.int32),
            "rope_shift_amount": torch.tensor([0], dtype=torch.int32),
            "k_cache": k_cache,
            "v_cache": v_cache,
        }
        if static_ids is None:
            static_ids = trace_query == 1
        # A static S=N query pins offset = seq_len - N >= 0, so torch.export
        # needs the position dim's min raised to N (first chunk: seq_len == N).
        pos_min = max(2, trace_query) if static_ids else 2
        seq_pos = torch.export.Dim(
            "seq_pos", min=pos_min, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        if static_ids:
            # static [1,S] query -> static [1,S,V] logits. S=1 carries the
            # python oracle gates (the python runtime cannot execute
            # dynamic-shaped outputs). S>1 is the chunked-prefill function:
            # MPSGraph's GPURegionRuntime crashes encoding externalized
            # composite region-calls when the ids dim is DYNAMIC (NSArrayM
            # nil insert, repro 2026-07-03), so prefill chunks ride a static
            # S with dynamic position/KV — the contract the engine already
            # runs for S=1 ship bundles and S=K spec-decode verify graphs.
            ids_shape = None
        else:
            ids_shape = {1: torch.export.Dim(
                "seq_ids", min=1, max=max_context_length - 2)}
        dynamic_shapes = {
            "input_ids": ids_shape,
            "position_ids": {1: seq_pos},
            "image_embeds": None,
            "deepstack_embeds": None,
            "rope_shift_start": None,
            "rope_shift_amount": None,
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": (
                "input_ids", "position_ids", "image_embeds", "deepstack_embeds",
                "rope_shift_start", "rope_shift_amount"),
            "output_names": ("logits",),
            "state_names": PIPELINED_STATE_NAMES,
        }


# ---------------------------------------------------------------------------
# Vision encoder (fixed grid)
# ---------------------------------------------------------------------------


class _VisionAttention(nn.Module):
    """Full bidirectional ViT attention with constant 2D rotary cos/sin."""

    def __init__(self, vcfg) -> None:
        super().__init__()
        self.num_heads = vcfg.num_heads
        self.head_dim = vcfg.hidden_size // vcfg.num_heads
        self.qkv = nn.Linear(vcfg.hidden_size, vcfg.hidden_size * 3, bias=True)
        self.proj = nn.Linear(vcfg.hidden_size, vcfg.hidden_size, bias=True)
        self.sdpa = SDPA(is_causal=False)

    def forward(self, x, cos, sin):
        # x [n, d]; cos/sin [1, 1, n, head_dim]
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3).unsqueeze(1)  # [3, 1, heads, n, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        def rot(t):
            t1 = t[..., : self.head_dim // 2]
            t2 = t[..., self.head_dim // 2 :]
            return torch.cat((-t2, t1), dim=-1)

        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        out = self.sdpa(q, k, v)  # [1, heads, n, hd]
        out = out.permute(0, 2, 1, 3).reshape(n, self.num_heads * self.head_dim)
        return self.proj(out)


class _VisionMLP(nn.Module):
    def __init__(self, vcfg) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(vcfg.hidden_size, vcfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(vcfg.intermediate_size, vcfg.hidden_size, bias=True)

    def forward(self, x):
        return self.linear_fc2(nn.functional.gelu(self.linear_fc1(x), approximate="tanh"))


class _VisionBlock(nn.Module):
    def __init__(self, vcfg) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(vcfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(vcfg.hidden_size, eps=1e-6)
        self.attn = _VisionAttention(vcfg)
        self.mlp = _VisionMLP(vcfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class _VisionMerger(nn.Module):
    """spatial_merge_size^2 patch concat -> MLP; norm pre- or post-shuffle."""

    def __init__(self, vcfg, use_postshuffle_norm: bool) -> None:
        super().__init__()
        self.hidden_size = vcfg.hidden_size * (vcfg.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else vcfg.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size, vcfg.out_hidden_size, bias=True)

    def forward(self, x):
        if self.use_postshuffle_norm:
            x = self.norm(x.reshape(-1, self.hidden_size))
        else:
            x = self.norm(x).reshape(-1, self.hidden_size)
        return self.linear_fc2(nn.functional.gelu(self.linear_fc1(x)))


class Qwen3VLVisionEncoder(nn.Module):
    """Fixed-grid Qwen3-VL vision tower.

    ``patches [n_patch, in_ch*T*P*P] -> (image_embeds [N, out_h],
    deepstack_embeds [3N, out_h])`` with n_patch = (2*grid_h)*(2*grid_w) in the
    processor's block-major patch order and N = grid_h*grid_w merged tokens.
    """

    def __init__(self, vcfg, grid_h: int = 14, grid_w: int = 14) -> None:
        super().__init__()
        self.vcfg = vcfg
        self.grid_h, self.grid_w = grid_h, grid_w
        merge = vcfg.spatial_merge_size
        self.n_patches = (grid_h * merge) * (grid_w * merge)
        patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2

        self.patch_proj = nn.Linear(patch_dim, vcfg.hidden_size, bias=True)
        self.blocks = nn.ModuleList(
            [_VisionBlock(vcfg) for _ in range(vcfg.depth)])
        self.merger = _VisionMerger(vcfg, use_postshuffle_norm=False)
        self.deepstack_merger_list = nn.ModuleList(
            [_VisionMerger(vcfg, use_postshuffle_norm=True)
             for _ in vcfg.deepstack_visual_indexes])
        self.deepstack_visual_indexes = list(vcfg.deepstack_visual_indexes)

        self.register_buffer(
            "pos_embed_const", torch.zeros(self.n_patches, vcfg.hidden_size),
            persistent=False)
        head_dim = vcfg.hidden_size // vcfg.num_heads
        self.register_buffer(
            "cos_const", torch.zeros(1, 1, self.n_patches, head_dim), persistent=False)
        self.register_buffer(
            "sin_const", torch.zeros(1, 1, self.n_patches, head_dim), persistent=False)

    def forward(self, patches: torch.Tensor):
        x = self.patch_proj(patches)
        x = x + self.pos_embed_const.to(x.dtype)
        cos = self.cos_const.to(x.dtype)
        sin = self.sin_const.to(x.dtype)
        deepstack = []
        k = 0
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin)
            if i in self.deepstack_visual_indexes:
                deepstack.append(self.deepstack_merger_list[k](x))
                k += 1
        image_embeds = self.merger(x)
        deepstack_embeds = torch.cat(deepstack, dim=0)
        return image_embeds, deepstack_embeds

    # -- constants ----------------------------------------------------------

    def _init_positional_constants(self, pos_embed_table: torch.Tensor) -> None:
        """Bake bilinear pos-embed interpolation + 2D rotary for the fixed grid.

        Mirrors HF ``fast_pos_embed_interpolate`` + ``rot_pos_emb`` (block-major
        patch order) exactly, in fp32.
        """
        vcfg = self.vcfg
        merge = vcfg.spatial_merge_size
        H, W = self.grid_h * merge, self.grid_w * merge
        side = int(vcfg.num_position_embeddings ** 0.5)
        table = pos_embed_table.float()

        h_idxs = torch.linspace(0, side - 1, H)
        w_idxs = torch.linspace(0, side - 1, W)
        h_f = h_idxs.int()
        w_f = w_idxs.int()
        h_c = (h_f + 1).clamp(max=side - 1)
        w_c = (w_f + 1).clamp(max=side - 1)
        dh = h_idxs - h_f
        dw = w_idxs - w_f

        idx = [
            (h_f[:, None] * side + w_f[None, :]).flatten(),
            (h_f[:, None] * side + w_c[None, :]).flatten(),
            (h_c[:, None] * side + w_f[None, :]).flatten(),
            (h_c[:, None] * side + w_c[None, :]).flatten(),
        ]
        wgt = [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
            ((1 - dh)[:, None] * dw[None, :]).flatten(),
            (dh[:, None] * (1 - dw)[None, :]).flatten(),
            (dh[:, None] * dw[None, :]).flatten(),
        ]
        pe = sum(table[i] * w[:, None] for i, w in zip(idx, wgt))  # [H*W, d] raster
        pe = (
            pe.view(H // merge, merge, W // merge, merge, -1)
            .permute(0, 2, 1, 3, 4)
            .reshape(H * W, -1)
        )  # block-major
        self.pos_embed_const.copy_(pe.to(self.pos_embed_const.dtype))

        # 2D rotary (block-major coords), HF Qwen3VLVisionRotaryEmbedding
        head_dim = vcfg.hidden_size // vcfg.num_heads
        rot_dim = head_dim // 2  # rotary covers half the head dims per axis pair
        inv = 1.0 / (10000.0 ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
        br = torch.arange(H // merge)
        bc = torch.arange(W // merge)
        ir = torch.arange(merge)
        ic = torch.arange(merge)
        rowg = (br[:, None, None, None] * merge + ir[None, None, :, None]).expand(
            H // merge, W // merge, merge, merge).reshape(-1)
        colg = (bc[None, :, None, None] * merge + ic[None, None, None, :]).expand(
            H // merge, W // merge, merge, merge).reshape(-1)
        coords = torch.stack([rowg, colg], dim=-1).float()       # [n, 2]
        freqs = coords[..., None] * inv[None, None, :]           # [n, 2, rot/2]
        freqs = freqs.flatten(1)                                  # [n, rot]
        emb = torch.cat([freqs, freqs], dim=-1)                  # [n, head_dim]
        self.cos_const.copy_(emb.cos().view(1, 1, self.n_patches, head_dim))
        self.sin_const.copy_(emb.sin().view(1, 1, self.n_patches, head_dim))

    @classmethod
    def from_hf(
        cls,
        hf_id: str,
        target_dtype: torch.dtype = torch.float16,
        grid_h: int = 14,
        grid_w: int = 14,
    ) -> "Qwen3VLVisionEncoder":
        cfg, sd = _load_vl_state_dict(hf_id, "model.visual.", torch.float32)
        vcfg = cfg.vision_config
        model = cls(vcfg, grid_h=grid_h, grid_w=grid_w).float()

        conv_w = sd.pop("patch_embed.proj.weight")  # [d, C, T, P, P]
        sd["patch_proj.weight"] = conv_w.reshape(conv_w.shape[0], -1)
        sd["patch_proj.bias"] = sd.pop("patch_embed.proj.bias")
        pos_table = sd.pop("pos_embed.weight")
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        missing = [k for k in missing if not k.endswith(
            ("pos_embed_const", "cos_const", "sin_const"))]
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        model._init_positional_constants(pos_table)
        model = model.to(dtype=target_dtype)
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def _load_vl_state_dict(hf_id: str, prefix: str, dtype: torch.dtype):
    """(config, {stripped_key: tensor}) for keys under ``prefix``."""
    import glob

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig

    model_dir = snapshot_download(
        hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"])
    cfg = AutoConfig.from_pretrained(model_dir)
    sd = {}
    for path in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if key.startswith(prefix):
                    sd[key.removeprefix(prefix)] = f.get_tensor(key).to(dtype)
                elif prefix == "model.language_model." and key == "lm_head.weight":
                    sd["lm_head.weight"] = f.get_tensor(key).to(dtype)
    return cfg, sd
