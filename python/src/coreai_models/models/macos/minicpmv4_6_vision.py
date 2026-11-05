# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""MiniCPM-V-4.6 vision encoder for Core AI export.

Re-authors ``transformers``' ``minicpmv4_6`` vision tower and merger for one fixed
square input. The HF module computes its position ids and window indices from
``target_sizes`` at run time; here they are constants of the grid, so the graph holds
no ``bucketize``, ``argsort`` or per-window loop:

  pixel_values [1, 3, H, W]
    -> patch embedding (14x14 conv) + position embeddings for the grid
    -> encoder layers 0..insert_layer_id        [1, (H/14) * (W/14), hidden]
    -> window-attention merger, 2x2 -> 1        [1, (H/28) * (W/28), hidden]
    -> remaining encoder layers
    -> post layer norm
    -> downsample MLP, 2x2 -> 1, into text_hidden
  -> image_features [1, (H/56) * (W/56), text_hidden]

Module names mirror the checkpoint, so loading strips ``model.`` and slices the
position table. A 448x448 input gives 64 image features, one source image with no
slices in the HF processor's terms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from coreai_models.models.base import read_checkpoint_tensors

_CHECKPOINT_PREFIXES = ("model.vision_tower.", "model.merger.")

# Both spatial merges of this architecture group 2x2 blocks of tokens.
_MERGE_SIZE = 2


def _group_windows(x: torch.Tensor, grid: int) -> torch.Tensor:
    """``[1, grid**2, D]`` -> ``[(grid/2)**2, 4, D]``: one row per 2x2 window, in raster order."""
    merged = grid // _MERGE_SIZE
    dim = x.shape[-1]
    return (
        x.reshape(merged, _MERGE_SIZE, merged, _MERGE_SIZE, dim)
        .permute(0, 2, 1, 3, 4)
        .reshape(merged * merged, _MERGE_SIZE * _MERGE_SIZE, dim)
    )


def _position_ids(grid: int, table_side: int) -> torch.Tensor:
    """Rows of the ``table_side**2`` position table that a ``grid x grid`` input reads.

    The bucketing of the HF ``MiniCPMV4_6VisionEmbeddings``, for one fixed grid.
    """
    boundaries = torch.arange(1 / table_side, 1.0, 1 / table_side)
    fractions = torch.arange(0, 1 - 1e-6, 1 / grid)
    buckets = torch.bucketize(fractions, boundaries, right=True)
    return (buckets[:, None] * table_side + buckets).flatten()


class VisionAttention(nn.Module):
    """Multi-head self-attention over ``[B, S, hidden]``. All projections carry a bias."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        out = F.scaled_dot_product_attention(
            self._heads(self.q_proj(x)),
            self._heads(self.k_proj(x)),
            self._heads(self.v_proj(x)),
            is_causal=False,
        )
        return self.out_proj(out.transpose(1, 2).reshape(batch, seq, self.hidden_size))


class VisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class VisionEncoderLayer(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int, layer_norm_eps: float
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = VisionAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class VisionEncoder(nn.Module):
    """Holds the layers. ``VisionTower.forward`` runs them, to insert the merger between two."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_heads: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                VisionEncoderLayer(hidden_size, intermediate_size, num_heads, layer_norm_eps)
                for _ in range(num_hidden_layers)
            ]
        )


class VisionEmbeddings(nn.Module):
    """Patch embedding plus the position embeddings of one fixed grid.

    ``position_embedding`` holds only the ``grid**2`` rows the grid reads, in raster order;
    ``MiniCPMV46VisionModel.from_pretrained`` slices them out of the checkpoint's table.
    """

    def __init__(self, hidden_size: int, patch_size: int, grid: int) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size, padding="valid"
        )
        self.position_embedding = nn.Parameter(torch.zeros(grid * grid, hidden_size))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        return patches + self.position_embedding


class WindowAttentionMerger(nn.Module):
    """Merges each 2x2 window of tokens into one, partway through the encoder.

    Attention runs within each window; the windows are then flattened through an MLP, with
    the window's mean as the residual. The HF module reorders tokens into windows and back
    around the attention, then groups them again for the merge. Grouping once is the same.
    """

    def __init__(
        self, hidden_size: int, intermediate_size: int, num_heads: int, layer_norm_eps: float
    ) -> None:
        super().__init__()
        window_tokens = _MERGE_SIZE * _MERGE_SIZE
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = VisionAttention(hidden_size, num_heads)
        self.pre_norm = nn.LayerNorm(hidden_size * window_tokens, eps=layer_norm_eps)
        self.linear_1 = nn.Linear(hidden_size * window_tokens, intermediate_size * window_tokens)
        self.linear_2 = nn.Linear(intermediate_size * window_tokens, hidden_size)

    def forward(self, x: torch.Tensor, grid: int) -> torch.Tensor:
        windows = _group_windows(x, grid)
        windows = windows + self.self_attn(self.layer_norm1(windows))
        flat = windows.reshape(windows.shape[0], -1)
        merged = self.linear_2(F.gelu(self.linear_1(self.pre_norm(flat)), approximate="tanh"))
        return (merged + windows.mean(dim=1)).unsqueeze(0)


class VisionTower(nn.Module):
    """SigLIP-style encoder with the window-attention merger after ``insert_layer_id``."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        patch_size: int,
        layer_norm_eps: float,
        insert_layer_id: int,
        grid: int,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.insert_layer_id = insert_layer_id
        self.embeddings = VisionEmbeddings(hidden_size, patch_size, grid)
        self.encoder = VisionEncoder(
            hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, layer_norm_eps
        )
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.vit_merger = WindowAttentionMerger(
            hidden_size, intermediate_size, num_attention_heads, layer_norm_eps
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values)
        for index, layer in enumerate(self.encoder.layers):
            x = layer(x)
            if index == self.insert_layer_id:
                x = self.vit_merger(x, self.grid)
        return self.post_layernorm(x)


class DownsampleMLP(nn.Module):
    """Flattened 2x2 window -> text hidden. Uses exact GELU, unlike the tower's tanh form."""

    def __init__(self, hidden_size: int, text_hidden_size: int, layer_norm_eps: float) -> None:
        super().__init__()
        window_hidden = hidden_size * _MERGE_SIZE * _MERGE_SIZE
        self.pre_norm = nn.LayerNorm(window_hidden, eps=layer_norm_eps)
        self.linear_1 = nn.Linear(window_hidden, window_hidden)
        self.linear_2 = nn.Linear(window_hidden, text_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.gelu(self.linear_1(self.pre_norm(x))))


class Merger(nn.Module):
    def __init__(self, hidden_size: int, text_hidden_size: int, layer_norm_eps: float) -> None:
        super().__init__()
        self.mlp = nn.ModuleList([DownsampleMLP(hidden_size, text_hidden_size, layer_norm_eps)])

    def forward(self, x: torch.Tensor, grid: int) -> torch.Tensor:
        windows = _group_windows(x, grid)
        return self.mlp[0](windows.reshape(windows.shape[0], -1))


class MiniCPMV46VisionModel(nn.Module):
    """End-to-end vision pipeline for one fixed square input.

    Input:  ``pixel_values [1, 3, image_size, image_size]``, float32, normalized.
    Output: ``image_features [1, N, text_hidden_size]`` with ``N = (image_size // 56) ** 2``,
    in the dtype of the weights.
    """

    # Layer norm epsilon of the HF `MiniCPMV4_6DownsampleMLP`, which is not in the config.
    _DOWNSAMPLE_LAYER_NORM_EPS = 1e-6

    def __init__(
        self,
        image_size: int = 448,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        patch_size: int = 14,
        layer_norm_eps: float = 1e-6,
        insert_layer_id: int = 6,
        text_hidden_size: int = 1024,
    ) -> None:
        super().__init__()
        grid = image_size // patch_size
        if image_size % patch_size != 0 or grid % (_MERGE_SIZE * _MERGE_SIZE) != 0:
            raise ValueError(
                f"image_size={image_size} needs a patch grid divisible by "
                f"{_MERGE_SIZE * _MERGE_SIZE}, with patch_size={patch_size}"
            )
        self.merged_grid = grid // _MERGE_SIZE
        self.vision_tower = VisionTower(
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            patch_size,
            layer_norm_eps,
            insert_layer_id,
            grid,
        )
        self.merger = Merger(hidden_size, text_hidden_size, self._DOWNSAMPLE_LAYER_NORM_EPS)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        dtype = self.merger.mlp[0].linear_2.weight.dtype
        x = self.vision_tower(pixel_values.to(dtype))
        return self.merger(x, self.merged_grid).unsqueeze(0)

    def load_checkpoint_tensors(self, tensors: dict[str, torch.Tensor], table_side: int) -> None:
        """Assign tensors named as in the checkpoint, slicing the position table to this grid."""
        grid = self.vision_tower.grid
        table_key = "model.vision_tower.embeddings.position_embedding.weight"
        state_dict = {k.removeprefix("model."): v for k, v in tensors.items() if k != table_key}
        state_dict["vision_tower.embeddings.position_embedding"] = tensors[table_key][
            _position_ids(grid, table_side)
        ]
        self.load_state_dict(state_dict, assign=True)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        image_size: int = 448,
        dtype: torch.dtype = torch.float32,
    ) -> "MiniCPMV46VisionModel":
        """Load the vision weights of a MiniCPM-V-4.6 checkpoint for one square input size."""
        model_dir = snapshot_download(
            model_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"]
        )
        config = AutoConfig.from_pretrained(model_dir)
        vision = config.vision_config
        with torch.device("meta"):
            model = cls(
                image_size=image_size,
                hidden_size=vision.hidden_size,
                intermediate_size=vision.intermediate_size,
                num_hidden_layers=vision.num_hidden_layers,
                num_attention_heads=vision.num_attention_heads,
                patch_size=vision.patch_size,
                layer_norm_eps=vision.layer_norm_eps,
                insert_layer_id=config.insert_layer_id,
                text_hidden_size=config.text_config.hidden_size,
            )
        model.load_checkpoint_tensors(
            read_checkpoint_tensors(model_dir, _CHECKPOINT_PREFIXES, dtype),
            table_side=vision.image_size // vision.patch_size,
        )
        return model.eval()
