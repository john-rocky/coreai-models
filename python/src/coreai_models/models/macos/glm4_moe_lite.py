# GLM-4.7-Flash text decoder (HF model_type `glm4_moe_lite`, i.e.
# zai-org/GLM-4.7-Flash) for the Core AI authoring path.
#
# Community port — NOT an Apple model. This is the zoo's first model with
# Multi-head Latent Attention (MLA, DeepSeek-V3 style). Authored to be
# exportable via coreai_models.export.macos.export_to_coreai; decoupled from
# transformers (weights load straight from the HF safetensors, config is a
# local dataclass) so the export env needs no upstream glm4_moe_lite support.
#
# Two novel pieces vs the existing zoo decoders:
#
#  1. MLA attention (every one of the 47 layers), authored in the NAIVE /
#     MATERIALIZED form: the latent caches are projected up to per-head
#     q_nope[192]/q_rope[64], k_nope[192]/k_rope[64] and v[256]; decoupled RoPE
#     is applied to the 64-dim rope slices only; q=concat(nope,rope),
#     k=concat(nope,rope) build full per-head [20, 256] tensors that run through
#     the proven SDPA composite (scale 256**-0.5). The KV cache stores the
#     materialized per-head K/V (full MHA size, num_key_value_heads == 20) — the
#     absorbed-MLA form that keeps the [512] latent in cache is a follow-up.
#     RoPE is GLM's interleaved variant, applied manually from precomputed
#     cos/sin (qwen3_5.py precedent) so the convention is bit-exact vs the HF
#     oracle and the dot-product layout question never arises.
#
#  2. Sparse-MoE FFN (layers 1..46; layer 0 is a plain dense MLP at
#     intermediate_size, first_k_dense_replace=1). Routing is `noaux_tc` which,
#     at n_group == topk_group == 1, reduces EXACTLY to sigmoid scoring +
#     selection-only bias correction (the LFM2.5-MoE routing): the top-k indices
#     come from sigmoid(logits) + e_score_correction_bias, the gathered weights
#     come from the RAW sigmoid, then norm_topk_prob (÷sum, +1e-20 guard) ×
#     routed_scaling_factor (1.8). Plus a non-gated shared expert (a plain MLP
#     added directly, unlike Qwen3.5-MoE's sigmoid-gated shared expert). Experts
#     run through the SwitchGLU/GatherMM composite (the data-dependent expert
#     gather Apple's qwen3_moe/gpt_oss use).
#
# Checkpoint mapping (zai-org original layout — experts stored UNPACKED, one
# tensor per expert, like LFM2.5-MoE; NOT packed like Qwen3.5-MoE):
#   model.embed_tokens.weight                       -> same
#   lm_head.weight (untied, at ckpt root)           -> same
#   ...self_attn.{q_a_proj,q_b_proj,kv_a_proj_with_mqa,kv_b_proj,o_proj}.weight -> same
#   ...self_attn.{q_a_layernorm,kv_a_layernorm}.weight                         -> same
#   ...{input,post_attention}_layernorm.weight      -> same
#   dense layer 0:
#     ...mlp.{gate,up,down}_proj.weight             -> same
#   moe layers 1..46:
#     ...mlp.gate.weight              [E, d]        -> ...mlp.gate.weight (router)
#     ...mlp.gate.e_score_correction_bias [E]       -> ...mlp.expert_bias (buffer)
#     ...mlp.experts.{e}.gate_proj.weight [I,d] -> stack_e -> ...switch_mlp.gate_proj.weight [1,E,I,d]
#     ...mlp.experts.{e}.up_proj.weight   [I,d] -> stack_e -> ...switch_mlp.up_proj.weight   [1,E,I,d]
#     ...mlp.experts.{e}.down_proj.weight [d,I] -> stack_e -> ...switch_mlp.down_proj.weight [1,E,d,I]
#     ...mlp.shared_experts.{gate,up,down}_proj.weight -> ...mlp.shared_expert.{...}_proj.weight
#   model.layers.47.* / *.mtp.* / *.nextn.*         -> skipped (MTP head)
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU


@dataclass
class Glm4MoeLiteConfig:
    """Subset of GLM-4.7-Flash config needed for authoring (values from config.json)."""

    hidden_size: int = 2048
    num_hidden_layers: int = 47
    vocab_size: int = 154880
    intermediate_size: int = 10240  # dense layer-0 FFN width
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    # MLA attention
    num_attention_heads: int = 20
    num_key_value_heads: int = 20  # materialized full MHA (== num_attention_heads)
    q_lora_rank: int = 768
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    attention_bias: bool = False
    rope_theta: float = 1e6
    rope_interleave: bool = True
    # MoE FFN
    moe_intermediate_size: int = 1536
    n_routed_experts: int = 64
    n_shared_experts: int = 1
    num_experts_per_tok: int = 4
    routed_scaling_factor: float = 1.8
    norm_topk_prob: bool = True
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 202752

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim  # 256

    @property
    def head_dim(self) -> int:
        # KV cache head dim: K is qk_head_dim, V is v_head_dim; both 256 here so a
        # single value backs KVCache.create_cache_tensors.
        return self.qk_head_dim

    def is_moe(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_k_dense_replace


# --------------------------------------------------------------------------- #
# Decoupled interleaved RoPE (manual, from precomputed cos/sin) — matches HF
# Glm4MoeLite apply_rotary_pos_emb_interleave bit-for-bit.
# --------------------------------------------------------------------------- #
def _deinterleave(x: torch.Tensor) -> torch.Tensor:
    """[..., a0,a1,a2,a3,...] -> [..., a0,a2,...,a1,a3,...] (even dims, then odd).

    HF expresses this as ``view(..., d//2, 2).transpose(-1,-2).reshape(..., d)``;
    the strided-slice + cat form is identical and lowers as the same op the
    interleaved RoPE composite already uses (``x[..., ::2]`` / ``x[..., 1::2]``).
    """
    return torch.cat([x[..., 0::2], x[..., 1::2]], dim=-1)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope_interleave(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
):
    """q: [b,H,s,64], k: [b,1,s,64]; cos,sin: [b,s,64]. GLM interleaved RoPE.

    De-interleave the rope slice, then standard rotate-half RoPE. cos/sin are
    computed in fp32 and cast to the query dtype (HF casts the rotary tables to
    the activation dtype; no-op in fp32, required in fp16 so the cat keeps one
    dtype).
    """
    cos = cos.unsqueeze(1).to(q.dtype)  # [b,1,s,64]
    sin = sin.unsqueeze(1).to(q.dtype)
    q = _deinterleave(q)
    k = _deinterleave(k)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


# --------------------------------------------------------------------------- #
# MLA attention (naive / materialized): latent down/up projections -> per-head
# q/k/v -> decoupled RoPE on the 64-dim slice -> SDPA over the full 256 dim.
# --------------------------------------------------------------------------- #
class Glm4MoeLiteMLA(nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope = config.qk_nope_head_dim
        self.qk_rope = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim  # 256
        self.v_head_dim = config.v_head_dim  # 256
        bias = config.attention_bias

        self.q_a_proj = nn.Linear(d, self.q_lora_rank, bias=bias)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(d, self.kv_lora_rank + self.qk_rope, bias=bias)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope + self.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, d, bias=bias)
        self.sdpa = SDPA(scale=self.qk_head_dim**-0.5, is_causal=True)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        offset: int = 0,
        seq_len: int | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        H = self.num_heads

        # Query: down to q_lora, norm, up to per-head qk_head_dim, split nope/rope.
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(b, s, H, self.qk_head_dim).transpose(1, 2)  # [b,H,s,256]
        q_nope, q_rope = torch.split(q, [self.qk_nope, self.qk_rope], dim=-1)

        # Key/Value: shared compressed latent + a single decoupled rope head.
        compressed = self.kv_a_proj_with_mqa(x)  # [b,s,512+64]
        kv_c, k_rope = torch.split(compressed, [self.kv_lora_rank, self.qk_rope], dim=-1)
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_c))
        kv = kv.view(b, s, H, self.qk_nope + self.v_head_dim).transpose(1, 2)  # [b,H,s,448]
        k_nope, value = torch.split(kv, [self.qk_nope, self.v_head_dim], dim=-1)
        k_rope = k_rope.view(b, 1, s, self.qk_rope)  # single shared rope head

        q_rope, k_rope = apply_rope_interleave(q_rope, k_rope, cos, sin)
        k_rope = k_rope.expand(b, H, s, self.qk_rope)  # broadcast to all heads

        query = torch.cat([q_nope, q_rope], dim=-1)  # [b,H,s,256]
        key = torch.cat([k_nope, k_rope], dim=-1)  # [b,H,s,256]

        if kv_cache is not None:
            key, value = kv_cache.update_and_fetch(
                layer_idx, offset, key, value, seq_len=seq_len, query_len=s
            )
        out = self.sdpa(query, key, value)  # [b,H,s,256]
        out = out.transpose(1, 2).reshape(b, s, H * self.v_head_dim)  # [b,s,5120]
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Sparse-MoE FFN — noaux_tc routing (n_group==1 -> sigmoid + selection-only
# bias) + non-gated shared expert, on the SwitchGLU composite.
# --------------------------------------------------------------------------- #
class Glm4MoeLiteMoE(nn.Module):
    """Scores stay fp32 through the weighted accumulate (matches Apple's
    qwen3_moe authoring; HF accumulates per-expert in the model dtype, so this is
    equal-or-better against the fp32 oracle)."""

    def __init__(self, config: Glm4MoeLiteConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.gate = nn.Linear(d, config.n_routed_experts, bias=False)
        self.register_buffer(
            "expert_bias", torch.zeros(config.n_routed_experts, dtype=torch.float32)
        )
        self.switch_mlp = SwitchGLU(
            d, config.moe_intermediate_size, config.n_routed_experts, bias=False
        )
        self.shared_expert = MLP(d, config.moe_intermediate_size * config.n_shared_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sig = torch.sigmoid(self.gate(x).float())  # [b,s,E] raw routing weights
        scores_for_choice = sig + self.expert_bias  # selection-only bias correction
        _, indices = torch.topk(scores_for_choice, self.top_k, dim=-1)  # [b,s,k]
        weights = torch.gather(sig, -1, indices)  # weights from RAW sigmoid
        if self.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling_factor
        y = self.switch_mlp(x, indices.to(torch.uint16))  # [b,s,k,d] (x.dtype)
        y = (y * weights.unsqueeze(-1)).sum(dim=-2)  # fp32 weighted accumulate
        shared = self.shared_expert(x)  # non-gated, added directly
        return (y + shared.to(torch.float32)).to(x.dtype)


# --------------------------------------------------------------------------- #
# Decoder layer — MLA + (dense MLP for layer 0, sparse MoE for layers 1..46)
# --------------------------------------------------------------------------- #
class Glm4MoeLiteDecoderLayer(nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig, layer_idx: int) -> None:
        super().__init__()
        d = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Glm4MoeLiteMLA(config)
        self.input_layernorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(d, eps=config.rms_norm_eps)
        if config.is_moe(layer_idx):
            self.mlp: nn.Module = Glm4MoeLiteMoE(config)
        else:
            self.mlp = MLP(d, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        offset: int = 0,
        seq_len: int | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        normed = self.input_layernorm(x)
        if kv_cache is None:
            r = self.self_attn(normed, cos, sin)
        else:
            r = self.self_attn(normed, cos, sin, kv_cache, offset, seq_len, layer_idx)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# --------------------------------------------------------------------------- #
# Decoder stack
# --------------------------------------------------------------------------- #
class Glm4MoeLiteModel(nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Glm4MoeLiteDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        rd = config.qk_rope_head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_buffers(self, device: str = "cpu") -> None:
        """Recreate the non-persistent inv_freq buffer after meta-device
        construction + load_state_dict (it isn't in the state dict)."""
        rd = self.config.qk_rope_head_dim
        self.inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, rd, 2, device=device).float() / rd)
        )

    def rope_cos_sin(self, position_ids: torch.Tensor):
        # position_ids [b,s] -> cos/sin [b,s,qk_rope_head_dim].
        freqs = position_ids[..., None].float() * self.inv_freq  # [b,s,rd/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [b,s,rd]
        return emb.cos(), emb.sin()

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Prefill (no KV cache; causal SDPA over the query block)."""
        h = self.embed_tokens(input_ids)
        cos, sin = self.rope_cos_sin(position_ids)
        for layer in self.layers:
            h = layer(h, cos, sin)
        return self.norm(h)

    def forward_stateful(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        return self.forward_stateful_core(
            self.embed_tokens(input_ids), position_ids, kv_cache
        )

    def forward_stateful_core(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
    ) -> torch.Tensor:
        """Stateful prefill/decode on pre-computed embeddings. ``input_ids``
        carries the query tokens ([b, query_len]); ``position_ids`` carries the
        full positions ([b, seq_len]) so offset = seq_len - query_len."""
        query_len = inputs_embeds.shape[1]
        seq_len = position_ids.shape[1]
        offset = seq_len - query_len
        q_pos = position_ids.narrow(1, offset, query_len)
        cos, sin = self.rope_cos_sin(q_pos)
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, kv_cache=kv_cache, offset=offset,
                      seq_len=seq_len, layer_idx=i)
        return self.norm(h)


# State names surfaced via export_to_coreai(state_names=...) and the runtime.
DECODE_STATE_NAMES = ("keyCache", "valueCache")


def build_decode_state(
    config: Glm4MoeLiteConfig,
    max_seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Allocate the (zeroed) materialized-MLA KV cache for a fresh sequence.
    Every layer is attention, so the cache has num_hidden_layers slots; K and V
    are full per-head [num_key_value_heads, head_dim] (head_dim == qk_head_dim
    == v_head_dim == 256)."""
    n = config.num_hidden_layers
    nkv = config.num_key_value_heads
    hd = config.qk_head_dim
    return {
        "k_cache": torch.zeros(n, 1, nkv, max_seq_len, hd, dtype=dtype),
        "v_cache": torch.zeros(n, 1, nkv, max_seq_len, hd, dtype=dtype),
    }


class Glm4MoeLiteStatefulForCausalLM(nn.Module):
    """Stateful text decoder: one graph for prefill and decode.

    forward inputs: input_ids [b, query_len], position_ids [b, seq_len], plus the
    two KV-cache tensors from ``build_decode_state`` (mutated in place, surfaced
    as Core AI states via ``DECODE_STATE_NAMES``); returns logits for the query
    tokens (or just the last token when ``last_token_only`` is set).
    """

    def __init__(self, config: Glm4MoeLiteConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Glm4MoeLiteModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.last_token_only = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        kv = KVCache(k_cache, v_cache)
        h = self.model.forward_stateful(input_ids, position_ids, kv)
        if self.last_token_only:
            h = h[:, -1:, :]
        return self.lm_head(h)


# --------------------------------------------------------------------------- #
# Config + weight loading (safetensors-direct; no transformers glm4_moe_lite)
# --------------------------------------------------------------------------- #
def glm4_moe_lite_config_from_dict(raw: dict) -> Glm4MoeLiteConfig:
    rope = raw.get("rope_parameters") or {}
    return Glm4MoeLiteConfig(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        vocab_size=raw["vocab_size"],
        intermediate_size=raw["intermediate_size"],
        rms_norm_eps=raw.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw.get("num_key_value_heads", raw["num_attention_heads"]),
        q_lora_rank=raw["q_lora_rank"],
        kv_lora_rank=raw["kv_lora_rank"],
        qk_nope_head_dim=raw["qk_nope_head_dim"],
        qk_rope_head_dim=raw["qk_rope_head_dim"],
        v_head_dim=raw["v_head_dim"],
        attention_bias=bool(raw.get("attention_bias", False)),
        rope_theta=rope.get("rope_theta", raw.get("rope_theta", 1e6)),
        rope_interleave=bool(raw.get("rope_interleave", True)),
        moe_intermediate_size=raw["moe_intermediate_size"],
        n_routed_experts=raw["n_routed_experts"],
        n_shared_experts=raw.get("n_shared_experts", 1),
        num_experts_per_tok=raw["num_experts_per_tok"],
        routed_scaling_factor=raw.get("routed_scaling_factor", 1.0),
        norm_topk_prob=raw.get("norm_topk_prob", True),
        first_k_dense_replace=raw.get("first_k_dense_replace", 1),
        max_position_embeddings=raw.get("max_position_embeddings", 202752),
    )


# experts.{e}.{gate,up,down}_proj -> the SwitchGLU projection it feeds
_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
# weights we never load (the multi-token-prediction head)
_SKIP_RE = re.compile(r"(^model\.layers\.47\.)|(\.mtp\.)|(\.nextn)|(^mtp\.)")


def glm4_moe_lite_from_hf(
    huggingface_model_id: str,
    target_dtype: torch.dtype = torch.float16,
):
    """Load GLM-4.7-Flash from the HF checkpoint into the authored module.

    Experts are stored one tensor per expert; they are stacked into the
    [1, E, out, in] SwitchLinear layout. The router bias buffer and the shared
    expert are renamed to the authored module names. The untied lm_head lives at
    the checkpoint root; the MTP head (layer 47 / *.mtp.* / *.nextn.*) is skipped.
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    model_dir = snapshot_download(
        huggingface_model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )
    with open(os.path.join(model_dir, "config.json")) as f:
        config = glm4_moe_lite_config_from_dict(json.load(f))

    with torch.device("meta"):
        model = Glm4MoeLiteStatefulForCausalLM(config)
    model.to(dtype=target_dtype)

    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    E = config.n_routed_experts
    # expert_stack[(mlp_prefix, proj)] = {expert_idx: tensor}
    expert_stack: dict[tuple[str, str], dict[int, torch.Tensor]] = {}
    sd: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if _SKIP_RE.search(key):
                    continue
                m = _EXPERT_RE.match(key)
                if m is not None:
                    mlp_prefix, e_idx, proj = m.group(1), int(m.group(2)), m.group(3)
                    t = f.get_tensor(key)
                    if t.dtype != target_dtype:
                        t = t.to(target_dtype)
                    expert_stack.setdefault((mlp_prefix, proj), {})[e_idx] = t
                    continue
                # router selection bias stays fp32 (matches the registered buffer)
                if key.endswith(".mlp.gate.e_score_correction_bias"):
                    local = key.replace(".mlp.gate.e_score_correction_bias", ".mlp.expert_bias")
                    sd[local] = f.get_tensor(key).to(torch.float32)
                    continue
                local = key.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
                t = f.get_tensor(key)
                if t.dtype != target_dtype and t.is_floating_point():
                    t = t.to(target_dtype)
                sd[local] = t

    # Stack per-expert tensors into the SwitchLinear [1,E,out,in] layout, freeing
    # each group's raw tensors as we go (the MoE weights are ~90% of this 30B
    # model — holding both the raw per-expert dict and the stacked copy at once
    # is ~2x and OOMs a 128 GB host even at fp16).
    for key in list(expert_stack.keys()):
        per_expert = expert_stack.pop(key)
        mlp_prefix, proj = key
        if len(per_expert) != E:
            raise RuntimeError(
                f"{mlp_prefix}.switch_mlp.{proj}: got {len(per_expert)} experts, expected {E}"
            )
        stacked = torch.stack([per_expert.pop(e) for e in range(E)], dim=0)  # [E,out,in]
        sd[f"{mlp_prefix}.switch_mlp.{proj}.weight"] = stacked.unsqueeze(0).contiguous()
        del per_expert, stacked

    model.load_state_dict(sd, assign=True, strict=False)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.reset_buffers()

    meta_params = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_params:
        raise RuntimeError(f"Parameters not loaded: {meta_params[:8]} ... ({len(meta_params)} total)")
    model.eval()
    return model
