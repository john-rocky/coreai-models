# Community port — NOT an Apple model.
"""Qwen2.5-Omni audio encoder, reworked to a **fixed shape** for Core AI / ANE.

The HF ``Qwen2_5OmniAudioEncoder.forward`` is data-dependent and will not
``torch.export`` to a single static graph:

  * mel is split into ``n_window*2 = 200``-frame chunks whose *count* depends on
    the clip length (``chunk_num = ceil(feature_lens / 200)``);
  * attention is **block-diagonal** (within-chunk only) via a ragged
    ``cu_seqlens`` mask built from those chunk lengths;
  * the whole audio is then ``avg_pool``'d (stride 2) + ``ln_post`` + ``proj``
    to ``[N, 2048]`` with ``N = ((mel-1)//2+1 - 2)//2 + 1``.

This module produces the *same* numbers with a single fixed-shape graph:

  mel ``[1, 128, 200*K]`` (host zero-pads to a whole number of chunks ``K``)
    -> reshape to ``[K, 128, 200]``                       (one chunk per batch row)
    -> conv1 / conv2 (stride 2) -> ``[K, 100, 1280]``     (post-CNN frames)
    -> + sinusoid positional embedding
    -> **batched** full self-attention over ``[K, 100, 1280]``  (= the original
       block-diagonal attention, one block per batch row; ANE-friendly)
    -> concat chunks -> ``[K*100, 1280]`` -> avg_pool -> ``[K*50, 1280]``
    -> ln_post -> proj -> ``[K*50 = N_max, 2048]``

The host trims ``[N_max, 2048]`` to the real ``N`` via ``feature_attention_mask``.

Faithfulness
------------
Per-frame ops (conv, layernorm, the q/k/v/o/fc linears) are identical whether run
on the ragged ``[sum, 1280]`` tensor or the batched ``[K, 100, 1280]`` one, and
the within-chunk attention is bit-identical to the block-diagonal mask.  Only the
**last, partial** chunk needs care:

  * a single ``attn_bias`` input ``[K, 1, 1, 100]`` (0 for valid post-CNN frames,
    a large negative for pad) reproduces ``cu_seqlens`` so valid frames never
    attend to pad.  No separate conv mask is needed: conv2's last *valid*
    post-CNN frame (col ``v-1``) only reads mel cols ``[2v-3, 2v-2, 2v-1]`` which
    are all inside the valid range, so a zero-padded mel never reaches a valid
    frame; the pad post-CNN frames are masked here and trimmed from the output.
  * the original drops the odd trailing real frame in ``avg_pool``; here the
    contaminated boundary token lands at output index ``>= N`` and is trimmed,
    so tokens ``[0, N)`` match exactly.

Two-bundle design: this is the audio encoder ``.aimodel``; the text decoder lives
in ``qwen2_5_omni_thinker.py``.  Audio placeholder ids ``V + slot`` in the decoder
``index_select`` row ``slot`` of this encoder's ``[N, 2048]`` output.
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn as nn


class Qwen2_5OmniAudioEncoderStatic(nn.Module):
    """Fixed-shape audio encoder wrapping the HF ``audio_tower`` weights.

    ``n_chunks`` (``K``) is baked at construction: the bundle accepts mel of a
    fixed width ``200*K`` and emits ``K*50`` audio tokens.  The host pads mel to
    ``K`` chunks and trims the output to the clip's real ``N``.
    """

    def __init__(self, tower, n_chunks: int) -> None:
        super().__init__()
        self.tower = tower
        self.n_chunks = int(n_chunks)
        self.chunk_mel = tower.n_window * 2          # 200 mel frames per chunk
        self.chunk_aftercnn = tower.n_window         # 100 post-CNN frames per chunk
        self.n_tokens = self.n_chunks * (tower.n_window // 2)  # K*50 output tokens
        self.num_heads = tower.config.encoder_attention_heads
        self.head_dim = tower.config.d_model // self.num_heads

    # -- per-layer (batched, within-chunk full attention) -------------------
    def _chunk_attention(self, layer, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        sa = layer.self_attn
        K, T, d = x.shape
        H, hd = self.num_heads, self.head_dim
        q = sa.q_proj(x).view(K, T, H, hd).transpose(1, 2)   # [K, H, T, hd]
        k = sa.k_proj(x).view(K, T, H, hd).transpose(1, 2)
        v = sa.v_proj(x).view(K, T, H, hd).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * sa.scaling   # [K, H, T, T]
        if attn_bias is not None:
            scores = scores + attn_bias                              # [K, 1, 1, T] -> mask pad keys
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(K, T, d)
        return sa.out_proj(out)

    def _encoder_layer(self, layer, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        residual = x
        x = layer.self_attn_layer_norm(x)
        x = self._chunk_attention(layer, x, attn_bias)
        x = residual + x
        residual = x
        x = layer.final_layer_norm(x)
        x = layer.fc2(layer.activation_fn(layer.fc1(x)))
        x = residual + x
        if x.dtype == torch.float16:                            # match the HF fp16 clamp
            clamp = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp, max=clamp)
        return x

    def forward(self, input_features: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        """input_features ``[1, 128, 200*K]`` (zero-padded) -> audio_embeds ``[K*50, 2048]``.

        ``attn_bias`` ``[K, 1, 1, 100]`` (0 valid / large-negative pad) masks the
        partial last chunk; pass zeros (or ``None``) when every chunk is full.
        """
        t = self.tower
        K, cm, ca = self.n_chunks, self.chunk_mel, self.chunk_aftercnn
        d = t.config.d_model

        # [1, 128, 200*K] -> [K, 128, 200]  (chunk k = mel cols [200k, 200k+200))
        x = input_features.reshape(t.num_mel_bins, K, cm).permute(1, 0, 2).contiguous()
        x = nn.functional.gelu(t.conv1(x))                      # [K, 1280, 200]
        x = nn.functional.gelu(t.conv2(x)).transpose(1, 2)      # [K, 100, 1280]
        pe = t.positional_embedding.positional_embedding[:ca, :].unsqueeze(0).to(x.dtype)
        x = x + pe                                              # broadcast over chunks
        for layer in t.layers:
            x = self._encoder_layer(layer, x, attn_bias)

        # pool over the WHOLE audio (concat chunks first), exactly like the original.
        # avg_pool(kernel=2, stride=2) over the (even) K*100 frame axis == mean of
        # adjacent pairs; reshape+mean is bit-identical and avoids the converter's
        # avg_pool2d path (which assumes a 4-D input).
        x = x.reshape(self.n_tokens, 2, d).mean(dim=1)         # [K*50, 1280]
        x = t.ln_post(x)
        x = t.proj(x)                                           # [K*50, 2048]
        return x

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_hf(
        cls,
        hf_id: str,
        n_chunks: int,
        target_dtype: torch.dtype = torch.float16,
    ) -> "Qwen2_5OmniAudioEncoderStatic":
        """Load only ``thinker.audio_tower.*`` into a bare HF encoder (~600M).

        Avoids materialising the 13 GB full Thinker just for the audio tower.
        """
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniAudioEncoder,
        )

        model_dir = snapshot_download(
            hf_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
        acfg = AutoConfig.from_pretrained(model_dir).thinker_config.audio_config
        acfg._attn_implementation = "eager"
        tower = Qwen2_5OmniAudioEncoder(acfg)

        prefix = "thinker.audio_tower."
        sd: dict[str, torch.Tensor] = {}
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith(prefix):
                        sd[key[len(prefix):]] = f.get_tensor(key).to(target_dtype)

        missing, unexpected = tower.load_state_dict(sd, strict=False, assign=True)
        # positional_embedding is a non-persistent (recomputed) buffer.
        missing = [m for m in missing if "positional_embedding" not in m]
        if missing or unexpected:
            raise RuntimeError(
                f"audio tower load mismatch: missing={missing} unexpected={unexpected}"
            )
        tower = tower.to(target_dtype).eval()
        return cls(tower, n_chunks).to(target_dtype).eval()


def eager_audio_features(
    tower,
    input_features: torch.Tensor,
    feature_attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Reference: HF ``get_audio_features`` reduced to the ``audio_tower`` call.

    Mirrors ``Qwen2_5OmniThinkerForConditionalGeneration.get_audio_features`` so
    the parity gate compares the *same weights* (the dynamic original vs the
    static rework), isolating the shape rework from any load drift.
    """
    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
    feats = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
    aftercnn_lens, _ = tower._get_feat_extract_output_lengths(audio_feature_lengths)
    out = tower(feats, feature_lens=audio_feature_lengths, aftercnn_lens=aftercnn_lens)
    return out.last_hidden_state
