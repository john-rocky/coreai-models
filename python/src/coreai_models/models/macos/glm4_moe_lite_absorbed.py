# Community port — NOT an Apple model.
"""Absorbed-MLA variant of the GLM-4.7-Flash (``glm4_moe_lite``) attention.

This is the **speed/KV follow-up** to the naive-materialized MLA in
``glm4_moe_lite.py`` (which ships at 52.4 tok/s with the MoE gather kernel). The
naive form caches the materialized per-head K/V (``[20,256]×2 = 10240 elems/token``)
and re-runs ``kv_b_proj`` (512→8960) every layer every token. The **absorbed** form
caches only the compressed latent and attends over it (MQA), so the KV shrinks
~17.8× and the per-token kv_b up-projection disappears.

LITE factorization (derived from the naive weights; see ``ABSORBED_MLA_STATE.md``):
  * Cache ``c_kv = kv_a_layernorm(kv_a_proj(x)[:512])`` (the [512] latent) and
    ``k_rope = RoPE(kv_a_proj(x)[512:576])`` (the [64] decoupled rope key, shared
    across heads).  576 elems/token vs 10240.
  * **Q-side**: lift the per-head nope query into the latent space with the K
    up-projection — ``ql[h] = q_nope[h] @ W_UK[h]`` (∈ℝ⁵¹²). Then the nope score is
    the MQA dot ``ql[h]·c_kv`` (every head shares ``c_kv``). q_b_proj is KEPT (it
    still produces q_nope/q_rope); we only add the small W_UK lift.
  * **Rope**: ``q_rope[h]·k_rope`` over [64], k_rope shared.
  * **V-side**: the attention runs over the LATENT value ``ctx[h] = Σ a·c_kv`` (∈ℝ⁵¹²),
    then ``out[h] = W_UV[h] @ ctx[h]`` (∈ℝ²⁵⁶) and the original o_proj. o_proj is KEPT.

``W_UK[h]``/``W_UV[h]`` are the k_nope / value slices of ``kv_b_proj`` reshaped to
``[H, 192, 512]`` / ``[H, 256, 512]`` — no NEW big matrices (unlike the folded
``A_q[768→512]`` / ``A_o[512→2048]`` form, which re-expands the low rank and only
wins past ~560 tok of context). Lite is mathematically identical to naive (a
reassociation) and reads strictly ≤ naive bytes/token at every context.

The score then attends a **[512 latent ++ 64 rope = 576]** query against ONE shared
key (c_kv ++ k_rope) and ONE shared value (c_kv [512]) — K is 576-dim, V is 512-dim
(rope is score-only). That asymmetric MQA shape, head_dim 576 > 512, is what the
custom flash-decode kernel (``mla_metal_sdpa.py``) computes; this module is the
pure-torch reference (prefill with a causal mask + stateful decode over the latent
cache) that gates the math first.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from coreai_models.models.macos.glm4_moe_lite import (
    Glm4MoeLiteConfig,
    Glm4MoeLiteMLA,
    Glm4MoeLiteModel,
    Glm4MoeLiteStatefulForCausalLM,
    apply_rope_interleave,
    glm4_moe_lite_from_hf,
)
from coreai_models.primitives._ops import mutable_slice_update


# --------------------------------------------------------------------------- #
# Latent KV cache: the combined ``[c_kv (kv_lora) ++ k_rope (qk_rope)]`` = [576]
# per token, stored as TWO EQUAL halves of [576/2 = 288] (kvCacheA = combined[:288],
# kvCacheB = combined[288:]). 5-D like KVCache (n_layers, 1, 1-shared-head, seq, dim).
#
# Why two EQUAL halves (not one [576] state, nor 512+64):
#   * The engine's LLM decode harness requires >= 2 KV-cache states (a single state
#     is rejected: "Expected at least 2 states (KV cache), got 1").
#   * Two states of DIFFERENT last-dim (latent 512 + rope 64) made the engine
#     mis-size the smaller one at ctx > ~256 ("MPSNDArray.mm:893 buffer not large
#     enough"), crashing decode at p >= 512.
# Two IDENTICAL-shape [288] halves satisfy the count AND are sized uniformly (the
# split point is arbitrary — the kernel concatenates A++B back to the [576] it reads
# as latent[:512] ++ rope[512:]). No storage waste (576 total, same as the latent).
# --------------------------------------------------------------------------- #
class AbsorbedKVCache:
    """Stores the combined MLA latent ++ shared rope key as TWO EQUAL ``[.,288]``
    halves (``A`` = combined[:288], ``B`` = combined[288:]) instead of the
    materialized per-head K/V. One shared (MQA) head; once reassembled the
    ``[:kv_lora]`` slice is both the nope-key and the value, ``[kv_lora:]`` the
    score-only rope key."""

    def __init__(self, kv_a: torch.Tensor, kv_b: torch.Tensor) -> None:
        self._a = kv_a  # [n_layers, 1, 1, max_seq, (kv_lora+qk_rope)//2]
        self._b = kv_b

    @classmethod
    def seq_len_dim(cls) -> int:
        return 3

    @staticmethod
    def _update_layer(cache: torch.Tensor, update: torch.Tensor, layer_idx: int,
                      offset: int) -> None:
        """Write ``update`` [1, 1, 1, q, dim] into cache[layer_idx, :, :, offset:offset+q, :]."""
        device = cache.device
        li = torch.tensor((layer_idx,), dtype=torch.int32, device=device)
        lie = torch.tensor((layer_idx + 1,), dtype=torch.int32, device=device)
        z = torch.tensor((0,), dtype=torch.int32, device=device)
        q = update.size(-2)   # seq dim of [b, 1, q, dim] (NOT the feature dim at -1)
        mutable_slice_update(
            x=cache,
            update=update.unsqueeze(0),
            begin=torch.cat([li, z, z, torch.tensor((offset,), dtype=torch.int32, device=device), z]),
            end=torch.cat([
                lie,
                torch.tensor((cache.size(1),), dtype=torch.int32, device=device),
                torch.tensor((cache.size(2),), dtype=torch.int32, device=device),
                torch.tensor((offset + q,), dtype=torch.int32, device=device),
                torch.tensor((cache.size(4),), dtype=torch.int32, device=device),
            ]),
        )

    def update_and_fetch(
        self, layer_idx: int, offset: int, kv: torch.Tensor,
        seq_len: int | None = None, query_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``kv`` [b,1,q,kv_lora+qk_rope] (= cat(c_kv, k_rope)) -> grown halves
        (A [b,1,seq_len,288], B [b,1,seq_len,288])."""
        if query_len is None:
            query_len = kv.shape[-2]
        if seq_len is None:
            seq_len = offset + query_len
        torch._check_is_size(query_len)
        torch._check_is_size(offset)
        torch._check_is_size(seq_len)
        torch._check(layer_idx < self._a.size(0))
        half = kv.shape[-1] // 2
        self._update_layer(self._a, kv[..., :half], layer_idx, offset)
        self._update_layer(self._b, kv[..., half:], layer_idx, offset)
        a = self._a.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len).squeeze(0)
        b = self._b.narrow(0, layer_idx, 1).narrow(-2, 0, seq_len).squeeze(0)
        return a, b


DECODE_STATE_NAMES_ABSORBED = ("kvCacheA", "kvCacheB")


def build_absorbed_decode_state(
    config: Glm4MoeLiteConfig, max_seq_len: int, dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Allocate the (zeroed) latent KV cache as two equal halves of the combined
    ``c_kv [kv_lora] ++ k_rope [qk_rope]`` ([576] -> 2x[288]) per token per layer."""
    n = config.num_hidden_layers
    tot = config.kv_lora_rank + config.qk_rope_head_dim
    assert tot % 2 == 0, f"combined latent dim {tot} must be even to split into two halves"
    half = tot // 2
    return {"kv_a": torch.zeros(n, 1, 1, max_seq_len, half, dtype=dtype),
            "kv_b": torch.zeros(n, 1, 1, max_seq_len, half, dtype=dtype)}


# --------------------------------------------------------------------------- #
# Attention core: q_full [b,H,S,576] vs shared c_kv[512] / k_rope[64] -> ctx[b,H,S,512].
# Pure-torch reference (the metal kernel swaps this for the flash-decode kernel).
# --------------------------------------------------------------------------- #
class EagerMLACore(nn.Module):
    """Reference scores + softmax + latent-value readout for absorbed MLA.

    ``q_full`` is ``[ql (kv_lora) ++ q_rope (qk_rope)]`` per head; ``kv`` is the
    combined ``[c_kv (kv_lora) ++ k_rope (qk_rope)]`` shared (MQA) state — the latent
    slice is both the nope-key and the value, the rope slice is a score-only key.
    q=1 decode attends every grown key; a prefill block (S>1) gets the position-based
    causal mask (query abs pos = ``q_offset + i``, key pos = j; mask j>query)."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, kv_lora: int, qk_rope: int, scale: float) -> None:
        super().__init__()
        self.kv_lora = kv_lora
        self.qk_rope = qk_rope
        self.scale = scale

    def forward(self, q_full: torch.Tensor, ka: torch.Tensor, kb: torch.Tensor,
                q_offset: int = 0) -> torch.Tensor:
        # q_full [b,H,S,576]; ka,kb [b,1,Sg,288] -> combined [c_kv(512) | k_rope(64)].
        ql = q_full[..., : self.kv_lora]          # [b,H,S,512]
        qr = q_full[..., self.kv_lora :]          # [b,H,S,64]
        kv = torch.cat([ka, kb], dim=-1)          # [b,1,Sg,576]
        ck = kv[:, 0, :, : self.kv_lora].float()   # [b,Sg,512] latent (nope-key + value)
        kr = kv[:, 0, :, self.kv_lora :].float()   # [b,Sg,64]  rope key (score-only)
        s_nope = torch.einsum("bhsl,bjl->bhsj", ql.float(), ck)
        s_rope = torch.einsum("bhsd,bjd->bhsj", qr.float(), kr)
        scores = (s_nope + s_rope) * self.scale    # [b,H,S,Sg]
        S, Sg = scores.shape[2], scores.shape[3]
        if S > 1:                                  # prefill block: causal over abs positions
            qpos = torch.arange(S, device=scores.device) + q_offset
            kpos = torch.arange(Sg, device=scores.device)
            mask = kpos[None, :] > qpos[:, None]   # [S, Sg]
            scores = scores.masked_fill(mask[None, None], float("-inf"))
        attn = torch.softmax(scores, dim=-1)       # [b,H,S,Sg] fp32
        ctx = torch.einsum("bhsj,bjl->bhsl", attn, ck)  # [b,H,S,512]
        return ctx.to(q_full.dtype)


class Glm4MoeLiteMLAAbsorbed(nn.Module):
    """Absorbed-MLA attention (lite form). Keeps q_a/q_b/kv_a/o_proj from the naive
    module; adds W_UK (q-lift) + W_UV (latent-value readout) sliced from kv_b_proj;
    attends over the [512]+[64] latent cache via ``self.mla_core``."""

    def __init__(self, config: Glm4MoeLiteConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.qk_nope = config.qk_nope_head_dim
        self.qk_rope = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora = config.kv_lora_rank
        d = config.hidden_size
        bias = config.attention_bias
        from coreai_models.primitives.macos.rms_norm import RMSNorm

        # Kept-as-naive projections (loaded weights are copied/shared by from_naive).
        self.q_a_proj = nn.Linear(d, config.q_lora_rank, bias=bias)
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(d, self.kv_lora + self.qk_rope, bias=bias)
        self.kv_a_layernorm = RMSNorm(self.kv_lora, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, d, bias=bias)
        # Absorbed lifts (sliced from kv_b_proj by from_naive); kept fp16 (small).
        self.register_buffer("W_UK", torch.zeros(self.num_heads, self.qk_nope, self.kv_lora))
        self.register_buffer("W_UV", torch.zeros(self.num_heads, self.v_head_dim, self.kv_lora))
        self.mla_core = EagerMLACore(self.kv_lora, self.qk_rope, self.qk_head_dim ** -0.5)

    @classmethod
    def from_naive(cls, naive: Glm4MoeLiteMLA, config: Glm4MoeLiteConfig) -> "Glm4MoeLiteMLAAbsorbed":
        m = cls(config)
        m.q_a_proj = naive.q_a_proj
        m.q_a_layernorm = naive.q_a_layernorm
        m.q_b_proj = naive.q_b_proj
        m.kv_a_proj_with_mqa = naive.kv_a_proj_with_mqa
        m.kv_a_layernorm = naive.kv_a_layernorm
        m.o_proj = naive.o_proj
        H, nope, v, lat = m.num_heads, m.qk_nope, m.v_head_dim, m.kv_lora
        w = naive.kv_b_proj.weight.detach().view(H, nope + v, lat)  # [H, 448, 512]
        m.W_UK = w[:, :nope, :].contiguous()    # k_nope slice -> q-lift
        m.W_UV = w[:, nope:, :].contiguous()    # value slice  -> latent readout
        return m

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                kv_cache: AbsorbedKVCache | None = None, offset: int = 0,
                seq_len: int | None = None, layer_idx: int = 0) -> torch.Tensor:
        b, s, _ = x.shape
        H = self.num_heads

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(b, s, H, self.qk_head_dim).transpose(1, 2)        # [b,H,s,256]
        q_nope, q_rope = torch.split(q, [self.qk_nope, self.qk_rope], dim=-1)
        ql = torch.einsum("bhsn,hnl->bhsl", q_nope, self.W_UK.to(q_nope.dtype))  # [b,H,s,512]

        compressed = self.kv_a_proj_with_mqa(x)                      # [b,s,512+64]
        c_kv, k_rope = torch.split(compressed, [self.kv_lora, self.qk_rope], dim=-1)
        c_kv = self.kv_a_layernorm(c_kv)                             # [b,s,512]
        k_rope = k_rope.view(b, 1, s, self.qk_rope)                  # shared rope head

        q_rope, k_rope = apply_rope_interleave(q_rope, k_rope, cos, sin)  # [b,H,s,64],[b,1,s,64]
        c_kv = c_kv.view(b, 1, s, self.kv_lora)
        kv = torch.cat([c_kv, k_rope], dim=-1)                       # [b,1,s,576] (latent ++ rope)

        if kv_cache is not None:
            ka, kb = kv_cache.update_and_fetch(
                layer_idx, offset, kv, seq_len=seq_len, query_len=s)
        else:                                                        # prefill, no cache: split now
            half = kv.shape[-1] // 2
            ka, kb = kv[..., :half], kv[..., half:]

        q_full = torch.cat([ql, q_rope], dim=-1)                     # [b,H,s,576]
        ctx = self.mla_core(q_full, ka, kb, q_offset=offset)         # [b,H,s,512]
        out = torch.einsum("bhsl,hol->bhso", ctx, self.W_UV.to(ctx.dtype))  # [b,H,s,256]
        out = out.transpose(1, 2).reshape(b, s, H * self.v_head_dim)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Whole-model absorbification + stateful wrapper (reuses Glm4MoeLiteModel/MoE).
# --------------------------------------------------------------------------- #
def absorbify_glm4_model(causal_lm: Glm4MoeLiteStatefulForCausalLM) -> Glm4MoeLiteStatefulForCausalLM:
    """Swap every layer's naive ``self_attn`` for the absorbed form, in place
    (the naive ``kv_b_proj`` is dropped → no extra peak memory). Returns the same
    object, now driven by ``Glm4MoeLiteAbsorbedStatefulForCausalLM`` for the cache."""
    cfg = causal_lm.config
    for layer in causal_lm.model.layers:
        layer.self_attn = Glm4MoeLiteMLAAbsorbed.from_naive(layer.self_attn, cfg)
    return causal_lm


class Glm4MoeLiteAbsorbedStatefulForCausalLM(nn.Module):
    """Stateful decoder over the latent KV cache. forward(input_ids [b,q],
    position_ids [b,seq], kv_a [n,1,1,max,288], kv_b [n,1,1,max,288]) -> logits. Reuses
    the absorbed ``Glm4MoeLiteModel`` (with absorbed attns) and lm_head."""

    def __init__(self, model: Glm4MoeLiteModel, lm_head: nn.Linear, config: Glm4MoeLiteConfig) -> None:
        super().__init__()
        self.model = model
        self.lm_head = lm_head
        self.config = config
        self.last_token_only = False

    @classmethod
    def from_causal_lm(cls, causal_lm: Glm4MoeLiteStatefulForCausalLM) -> "Glm4MoeLiteAbsorbedStatefulForCausalLM":
        return cls(causal_lm.model, causal_lm.lm_head, causal_lm.config)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor,
                kv_a: torch.Tensor, kv_b: torch.Tensor) -> torch.Tensor:
        kv = AbsorbedKVCache(kv_a, kv_b)
        h = self.model.forward_stateful_core(self.model.embed_tokens(input_ids), position_ids, kv)
        if self.last_token_only:
            h = h[:, -1:, :]
        return self.lm_head(h)


def glm4_moe_lite_absorbed_from_hf(
    huggingface_model_id: str, target_dtype: torch.dtype = torch.float16,
) -> Glm4MoeLiteStatefulForCausalLM:
    """Load GLM-4.7-Flash via the naive loader, then absorbify in place. Returns the
    (absorbed) ``Glm4MoeLiteStatefulForCausalLM`` — wrap with
    ``Glm4MoeLiteAbsorbedStatefulForCausalLM.from_causal_lm`` for stateful decode."""
    model = glm4_moe_lite_from_hf(huggingface_model_id, target_dtype=target_dtype)
    return absorbify_glm4_model(model)
