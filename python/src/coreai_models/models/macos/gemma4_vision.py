# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma 4 vision tower shaped as a FIXED-GRID one-shot encoder .aimodel.

``patches [n_patch, 3*P*P] -> image_embeds [n_soft, text_hidden]`` — the second
half of the Qwen3-VL-style VLM rider (see qwen3_vl.py): the decoder rides the
unmodified pipelined engine and swaps in these rows for ids >= V via the
static-input hook; this graph runs ONCE per image as a plain .aimodel.

Mirrors HF ``Gemma4VisionModel`` + ``Gemma4MultimodalEmbedder`` exactly, with
everything positional baked as constants for one canonical grid:

* patches come from the processor in row-major (y*W + x) order, values in
  [0, 1]; the model scales ``2*(p - 0.5)`` in-graph (processor does NOT
  normalize).
* learned pos-embed = ``table[0][x] + table[1][y]`` (two 10240-row coordinate
  tables, direct lookup, NO interpolation) — baked to ``[n_patch, h]``.
* 2D rotary (theta=100): head_dim 64 = [32 X-rope | 32 Y-rope], 16 freqs per
  axis over spatial_dim 32, standard rotate-half within each 32-half — baked
  cos/sin ``[1, 1, n_patch, 64]``.
* ``Gemma4ClippableLinear``: checkpoint-calibrated input/output clamps around
  every attn/MLP linear (QAT-style activation bounds). Clamps that load as
  +-inf collapse to plain linears at export.
* after the 16 bidirectional layers: 3x3 average pooling on the (y, x) grid
  (row-major both sides), then ``* sqrt(h)``, scale-free RMSNorm, and the
  768->1536 ``embed_vision`` projection.

The canonical SQUARE grid is 48x48 patches (768x768 px) -> 256 soft tokens
(= what the processor produces for any square image); the full-budget 280
needs a 10:7 grid (42x60). The decoder's image_embeds static input is sized
for max 280 rows; the host fills the first n_soft.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.models.macos.gemma4_text import ScaleFreeRMSNorm
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA


class _ClippedLinear(nn.Module):
    """HF ``Gemma4ClippableLinear``: clamp -> linear -> clamp.

    The four bounds are checkpoint buffers; if a bound pair loads as +-inf the
    corresponding clamp is dropped from the graph (export-friendliness).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.register_buffer("input_min", torch.tensor(-float("inf")))
        self.register_buffer("input_max", torch.tensor(float("inf")))
        self.register_buffer("output_min", torch.tensor(-float("inf")))
        self.register_buffer("output_max", torch.tensor(float("inf")))
        # Plain python flags so the branch is STATIC at torch.export trace
        # time (a tensor-valued isfinite() check is a data-dependent guard).
        # finalize_clips() refreshes them from the loaded buffer values.
        self.clip_input = False
        self.clip_output = False

    def finalize_clips(self) -> None:
        self.clip_input = bool(
            torch.isfinite(self.input_min) or torch.isfinite(self.input_max))
        self.clip_output = bool(
            torch.isfinite(self.output_min) or torch.isfinite(self.output_max))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.clip_input:
            x = torch.clamp(x, self.input_min.to(x.dtype), self.input_max.to(x.dtype))
        x = self.linear(x)
        if self.clip_output:
            x = torch.clamp(x, self.output_min.to(x.dtype), self.output_max.to(x.dtype))
        return x


class _VisionAttention(nn.Module):
    """Bidirectional MHA, scale=1.0, QK RMSNorm + scale-free V RMSNorm, 2D rope."""

    def __init__(self, vcfg: dict) -> None:
        super().__init__()
        self.n_heads = vcfg["num_attention_heads"]
        self.head_dim = vcfg["head_dim"]
        h = vcfg["hidden_size"]
        eps = vcfg["rms_norm_eps"]
        self.q_proj = _ClippedLinear(h, self.n_heads * self.head_dim)
        self.k_proj = _ClippedLinear(h, self.n_heads * self.head_dim)
        self.v_proj = _ClippedLinear(h, self.n_heads * self.head_dim)
        self.o_proj = _ClippedLinear(self.n_heads * self.head_dim, h)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        self.v_norm = ScaleFreeRMSNorm(eps=eps)
        # Gemma 4 vision attention scale is 1.0 (QK-norm bounds magnitudes).
        self.sdpa = SDPA(scale=1.0, is_causal=False)

    def _rope2d(self, t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Rotate the two 32-channel halves independently (X then Y axis)."""
        half = self.head_dim // 2
        outs = []
        for k in range(2):
            tk = t[..., k * half : (k + 1) * half]
            ck = cos[..., k * half : (k + 1) * half]
            sk = sin[..., k * half : (k + 1) * half]
            t1, t2 = tk[..., : half // 2], tk[..., half // 2 :]
            rot = torch.cat((-t2, t1), dim=-1)
            outs.append(tk * ck + rot * sk)
        return torch.cat(outs, dim=-1)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        n = x.shape[0]
        q = self.q_proj(x).view(n, self.n_heads, self.head_dim).permute(1, 0, 2)
        k = self.k_proj(x).view(n, self.n_heads, self.head_dim).permute(1, 0, 2)
        v = self.v_proj(x).view(n, self.n_heads, self.head_dim).permute(1, 0, 2)
        q = self.q_norm(q.unsqueeze(0))
        k = self.k_norm(k.unsqueeze(0))
        v = self.v_norm(v.unsqueeze(0))
        q = self._rope2d(q, cos, sin)
        k = self._rope2d(k, cos, sin)
        out = self.sdpa(query=q, key=k, value=v)  # [1, heads, n, hd]
        out = out.permute(0, 2, 1, 3).reshape(n, self.n_heads * self.head_dim)
        return self.o_proj(out)


class _VisionMLP(nn.Module):
    def __init__(self, vcfg: dict) -> None:
        super().__init__()
        h, inter = vcfg["hidden_size"], vcfg["intermediate_size"]
        self.gate_proj = _ClippedLinear(h, inter)
        self.up_proj = _ClippedLinear(h, inter)
        self.down_proj = _ClippedLinear(inter, h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = nn.functional.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


class _VisionBlock(nn.Module):
    """Gemma sandwich-norm block (input/post-attn + pre/post-FFN RMSNorms)."""

    def __init__(self, vcfg: dict) -> None:
        super().__init__()
        h, eps = vcfg["hidden_size"], vcfg["rms_norm_eps"]
        self.self_attn = _VisionAttention(vcfg)
        self.mlp = _VisionMLP(vcfg)
        self.input_layernorm = RMSNorm(h, eps=eps)
        self.post_attention_layernorm = RMSNorm(h, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(h, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(h, eps=eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.post_attention_layernorm(
            self.self_attn(self.input_layernorm(x), cos, sin))
        x = x + self.post_feedforward_layernorm(
            self.mlp(self.pre_feedforward_layernorm(x)))
        return x


class Gemma4VisionEncoder(nn.Module):
    """Fixed-grid Gemma 4 vision tower + multimodal embedder.

    ``patches [n_patch, 3*P*P] -> image_embeds [n_soft, text_hidden]`` with
    n_patch = grid_h*grid_w in the processor's row-major patch order and
    n_soft = n_patch // pooling_kernel_size^2.
    """

    coreai_externalize_specs: tuple = ()

    def __init__(self, vcfg: dict, text_hidden: int,
                 grid_h: int = 48, grid_w: int = 48) -> None:
        super().__init__()
        self.vcfg = vcfg
        k = vcfg["pooling_kernel_size"]
        if grid_h % k or grid_w % k:
            raise ValueError(f"grid {grid_h}x{grid_w} not divisible by pool {k}")
        self.grid_h, self.grid_w, self.pool_k = grid_h, grid_w, k
        self.n_patches = grid_h * grid_w
        self.n_soft = self.n_patches // (k * k)
        h = vcfg["hidden_size"]
        self.hidden_size = h
        self.root_hidden_size = float(h) ** 0.5

        self.input_proj = nn.Linear(3 * vcfg["patch_size"] ** 2, h, bias=False)
        self.layers = nn.ModuleList(
            [_VisionBlock(vcfg) for _ in range(vcfg["num_hidden_layers"])])
        # embed_vision (Gemma4MultimodalEmbedder): scale-free norm + projection
        self.embedding_pre_projection_norm = ScaleFreeRMSNorm(eps=vcfg["rms_norm_eps"])
        self.embedding_projection = nn.Linear(h, text_hidden, bias=False)

        self.register_buffer(
            "pos_embed_const", torch.zeros(self.n_patches, h), persistent=False)
        hd = vcfg["head_dim"]
        self.register_buffer(
            "cos_const", torch.zeros(1, 1, self.n_patches, hd), persistent=False)
        self.register_buffer(
            "sin_const", torch.zeros(1, 1, self.n_patches, hd), persistent=False)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # processor emits [0,1]; HF scales in model code (no processor norm)
        x = self.input_proj(2.0 * patches - 1.0)
        x = x + self.pos_embed_const.to(x.dtype)
        cos = self.cos_const.to(x.dtype)
        sin = self.sin_const.to(x.dtype)
        for blk in self.layers:
            x = blk(x, cos, sin)
        # 3x3 average pool on the (y, x) grid; row-major in and out
        k = self.pool_k
        x = x.view(self.grid_h // k, k, self.grid_w // k, k, self.hidden_size)
        x = x.mean(dim=(1, 3)).reshape(self.n_soft, self.hidden_size)
        x = x * self.root_hidden_size
        x = self.embedding_pre_projection_norm(x)
        return self.embedding_projection(x)

    # -- constants ------------------------------------------------------------

    def _init_positional_constants(self, pos_table: torch.Tensor) -> None:
        """Bake coordinate pos-embeds + 2D rotary for the fixed grid (fp32).

        Mirrors HF ``Gemma4VisionPatchEmbedder._position_embeddings`` (direct
        x/y table lookup) and ``Gemma4VisionRotaryEmbedding`` (independent
        per-axis freqs over spatial_dim = head_dim // 2, X axis first).
        """
        gh, gw = self.grid_h, self.grid_w
        ys, xs = torch.meshgrid(
            torch.arange(gh), torch.arange(gw), indexing="ij")
        xs = xs.reshape(-1)  # row-major: idx = y*gw + x
        ys = ys.reshape(-1)
        table = pos_table.float()
        pe = table[0][xs] + table[1][ys]  # [n_patch, h]
        self.pos_embed_const.copy_(pe.to(self.pos_embed_const.dtype))

        hd = self.vcfg["head_dim"]
        theta = float(self.vcfg["rope_theta"])
        spatial = hd // 2
        inv = 1.0 / (theta ** (torch.arange(0, spatial, 2).float() / spatial))
        cos_parts, sin_parts = [], []
        for coord in (xs, ys):  # HF iterates position_ids[..., i]; i=0 is X
            freqs = coord.float()[:, None] * inv[None, :]      # [n, spatial/2]
            emb = torch.cat([freqs, freqs], dim=-1)            # [n, spatial]
            cos_parts.append(emb.cos())
            sin_parts.append(emb.sin())
        cos = torch.cat(cos_parts, dim=-1).view(1, 1, self.n_patches, hd)
        sin = torch.cat(sin_parts, dim=-1).view(1, 1, self.n_patches, hd)
        self.cos_const.copy_(cos.to(self.cos_const.dtype))
        self.sin_const.copy_(sin.to(self.sin_const.dtype))

    # -- loading ----------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_id: str,
        target_dtype: torch.dtype = torch.float16,
        grid_h: int = 48,
        grid_w: int = 48,
    ) -> "Gemma4VisionEncoder":
        cfg, sd = _load_gemma4_vl_state_dict(
            hf_id, ("model.vision_tower.", "model.embed_vision."), torch.float32)
        vc = cfg["vision_config"]
        vcfg = {
            "num_hidden_layers": vc["num_hidden_layers"],
            "hidden_size": vc["hidden_size"],
            "num_attention_heads": vc["num_attention_heads"],
            "head_dim": vc["head_dim"],
            "intermediate_size": vc["intermediate_size"],
            "patch_size": vc["patch_size"],
            "pooling_kernel_size": vc["pooling_kernel_size"],
            "rms_norm_eps": vc["rms_norm_eps"],
            "rope_theta": vc["rope_parameters"]["rope_theta"],
        }
        if vc.get("standardize"):
            raise NotImplementedError("standardize=True vision checkpoints")
        model = cls(vcfg, text_hidden=cfg["text_config"]["hidden_size"],
                    grid_h=grid_h, grid_w=grid_w).float()

        out = {}
        for key, t in sd.items():
            k = key
            k = k.removeprefix("patch_embedder.")     # input_proj.*
            k = k.removeprefix("encoder.")            # layers.*
            out[k] = t
        pos_table = out.pop("position_embedding_table")  # [2, 10240, h]
        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        missing = [m for m in missing if not m.endswith(
            ("pos_embed_const", "cos_const", "sin_const"))]
        if missing or unexpected:
            raise RuntimeError(
                f"load mismatch: missing={missing} unexpected={unexpected}")
        model._init_positional_constants(pos_table)
        for m in model.modules():
            if isinstance(m, _ClippedLinear):
                m.finalize_clips()
        n_clip = sum(
            m.clip_input + m.clip_output
            for m in model.modules() if isinstance(m, _ClippedLinear))
        print(f"gemma4_vision: {n_clip} active clamp sites")
        model = model.to(dtype=target_dtype)
        # keep the baked rotary/pos constants fp32-derived but storage-cast
        model.eval()
        return model


def _load_gemma4_vl_state_dict(
    hf_id: str, prefixes: tuple[str, ...], dtype: torch.dtype
):
    """(config_dict, {stripped_key: tensor}) for keys under any of ``prefixes``."""
    import glob
    import json

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    model_dir = snapshot_download(
        hf_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )
    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    sd = {}
    for path in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                for prefix in prefixes:
                    if key.startswith(prefix):
                        sd[key.removeprefix(prefix)] = f.get_tensor(key).to(dtype)
                        break
    return cfg, sd
