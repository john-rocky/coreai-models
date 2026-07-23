# Community port — NOT an Apple model.
"""Streaming-quantize loader for the gather_qmm MoE path — unblocks models whose fp16 weights
exceed host RAM (e.g. Qwen3-Coder-Next 80B-A3B ≈ 159 GB fp16 on a 137 GB Mac).

The standard ``from_hf_memory_efficient`` builds the FULL fp16 state-dict in RAM (the routed-expert
tensors are the bulk) → OOM for an 80B MoE. Here, instead, the routed experts are read **one MoE
layer at a time** from the checkpoint via the safetensors weight-map (``safe_open`` reads only the
requested tensor), **immediately sym8-quantized** into the ``MetalSwitchGLU`` buffers (≈ half the
fp16 size), and the fp16 freed before the next layer. Peak RAM = one layer's experts (~few GB) +
the accumulating quantized experts (~half) + the small non-expert weights — well under the full
fp16 footprint. Non-expert weights load fp16 as usual (small) and are int8-quantized later by the
export's ``quantize_pytorch_model``.

Two expert layouts are auto-detected per layer:
  * packed (Qwen3.6 ``qwen3_5_moe``): ``…mlp.experts.gate_up_proj`` [E, 2·I, d] (split at I → gate /
    up) and ``…mlp.experts.down_proj`` [E, d, I].
  * per-expert (Qwen3-Coder-Next ``qwen3_next``): ``…mlp.experts.{e}.{gate,up,down}_proj.weight``
    ([I, d] / [I, d] / [d, I]), stacked over e → the same [E, …] slabs.
The non-expert key mapping mirrors ``from_hf_memory_efficient``; pass ``prefix`` to match the
checkpoint (``model.language_model.`` for Qwen3.6's multimodal ckpt, ``model.`` for Coder-Next).
"""
from __future__ import annotations

import glob
import json
import os

import torch
import torch.nn as nn

from coreai_models.models.base import _is_layer_key_beyond
from coreai_models.primitives.macos.switch import SwiGLU
from coreai_models.models.macos.moe_metal import (
    MetalSwitchGLU, build_gather_qmm_int8sym, build_gather_qmm_int4aff, build_gather_qmm_int4km,
    build_gather_qmm_int8km,
)


def _kernel(scheme: str):
    return {"sym8": build_gather_qmm_int8sym, "aff4": build_gather_qmm_int4aff,
            "km8": build_gather_qmm_int8km, "km4": build_gather_qmm_int4km}[scheme]()


def _split_gdn_qkvz(w_qkvz: torch.Tensor, config) -> tuple[torch.Tensor, torch.Tensor]:
    """De-interleave HF-canonical packed ``in_proj_qkvz`` [2*key_dim + 2*value_dim, d] into the
    qwen3_5 model body's contiguous ``in_proj_qkv`` [conv_dim, d] = cat(q, k, v) and
    ``in_proj_z`` [value_dim, d]. HF groups by KEY head: per kgroup the rows are
    [q(dk), k(dk), v(r*dv), z(r*dv)] (r = num_v // num_k). Pure row reorg → bit-exact
    (verified against modeling_qwen3_next.fix_query_key_value_ordering, max|Δ|=0)."""
    nk, dk, dv = config.linear_num_key_heads, config.linear_key_head_dim, config.linear_value_head_dim
    d, r = config.hidden_size, config.linear_num_value_heads // config.linear_num_key_heads
    qr = w_qkvz.reshape(nk, 2 * dk + 2 * r * dv, d)
    qkv = torch.cat([
        qr[:, 0:dk, :].reshape(nk * dk, d),                        # q (16 key heads)
        qr[:, dk:2 * dk, :].reshape(nk * dk, d),                   # k
        qr[:, 2 * dk:2 * dk + r * dv, :].reshape(nk * r * dv, d),  # v (value heads, GVA-paired)
    ], dim=0).contiguous()
    z = qr[:, 2 * dk + r * dv:, :].reshape(nk * r * dv, d).contiguous()
    return qkv, z


def _split_gdn_ba(w_ba: torch.Tensor, config) -> tuple[torch.Tensor, torch.Tensor]:
    """De-interleave packed ``in_proj_ba`` [2*num_v, d] (per kgroup [b(r), a(r)]) into the model's
    contiguous ``in_proj_b`` [num_v, d] and ``in_proj_a`` [num_v, d]."""
    nk, d = config.linear_num_key_heads, config.hidden_size
    r = config.linear_num_value_heads // nk
    br = w_ba.reshape(nk, 2 * r, d)
    return (br[:, 0:r, :].reshape(nk * r, d).contiguous(),
            br[:, r:2 * r, :].reshape(nk * r, d).contiguous())


def _weight_map(model_dir: str) -> tuple[dict[str, str], list[str]]:
    """Return (key -> abs safetensors path) and the file list. Handles sharded (index.json) and
    single-file checkpoints."""
    idx = glob.glob(os.path.join(model_dir, "*.safetensors.index.json"))
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if idx:
        wm = json.load(open(idx[0]))["weight_map"]
        return {k: os.path.join(model_dir, v) for k, v in wm.items()}, files
    # single file: map every key to it
    from safetensors import safe_open
    wm = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():  # noqa: SIM118
                wm[k] = path
    return wm, files


def from_hf_streaming_metal_moe(
    model_cls,
    config,
    hf_id: str,
    scheme: str = "sym8",
    prefix: str = "model.language_model.",
    num_layers: int | None = None,
    target_dtype: torch.dtype = torch.float16,
):
    """Build ``model_cls(config)`` with routed experts streamed → ``MetalSwitchGLU`` (scheme), the
    fp16 NEVER fully materialized. ``config`` is already built. Returns ``(model, kernel)``; run the
    export's int8 ``quantize_pytorch_model`` on the result (it skips the MetalSwitchGLU experts)."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    model_dir = snapshot_download(hf_id, allow_patterns=[
        "*.safetensors", "*.safetensors.index.json", "config.json"])
    wm, _files = _weight_map(model_dir)

    model = model_cls(config, model_device="meta")
    model.to(dtype=target_dtype)
    kernel = _kernel(scheme)
    inter = config.moe_intermediate_size
    n_layers = config.num_hidden_layers

    open_cache: dict[str, object] = {}

    def get(key: str) -> torch.Tensor:
        path = wm[key]
        if path not in open_cache:
            open_cache[path] = safe_open(path, framework="pt", device="cpu").__enter__()
        t = open_cache[path].get_tensor(key)
        if t.dtype != target_dtype and t.is_floating_point():
            t = t.to(target_dtype)
        return t

    # --- non-expert weights -> sd (small) ---
    sd: dict[str, torch.Tensor] = {}
    for key in wm:
        if key == "lm_head.weight" and not config.tie_word_embeddings:
            sd[key] = get(key)
            continue
        if not key.startswith(prefix):
            continue
        local = "model." + key[len(prefix):]
        if num_layers is not None and _is_layer_key_beyond(local, num_layers):
            continue
        if ".mlp.experts." in local:          # routed experts -> streamed below, NOT into sd
            continue
        # GatedDeltaNet: Coder-Next packs the input projections HF-canonically (grouped by key
        # head). The reused qwen3_5 body has separate contiguous in_proj_{qkv,z,b,a}, so split the
        # packed tensors here. Qwen3.6's checkpoint is already split -> these keys are absent there.
        if local.endswith(".linear_attn.in_proj_qkvz.weight"):
            base = local[: -len("in_proj_qkvz.weight")]
            qkv, z = _split_gdn_qkvz(get(key), config)
            sd[base + "in_proj_qkv.weight"], sd[base + "in_proj_z.weight"] = qkv, z
            continue
        if local.endswith(".linear_attn.in_proj_ba.weight"):
            base = local[: -len("in_proj_ba.weight")]
            bb, aa = _split_gdn_ba(get(key), config)
            sd[base + "in_proj_b.weight"], sd[base + "in_proj_a.weight"] = bb, aa
            continue
        sd[local] = get(key)

    model.load_state_dict(sd, assign=True, strict=False)
    del sd

    # --- routed experts: one MoE layer at a time -> sym8 MetalSwitchGLU, free fp16 ---
    # Two checkpoint layouts are supported (auto-detected per layer):
    #   packed     : ...mlp.experts.gate_up_proj [E,2I,d] + ...experts.down_proj [E,d,I]  (Qwen3.6 qwen3_5_moe)
    #   per-expert : ...mlp.experts.{e}.{gate,up,down}_proj.weight [I,d]/[I,d]/[d,I]       (Qwen3-Coder-Next qwen3_next)
    # Both produce the same [E,I,d]/[E,I,d]/[E,d,I] stacks; from_weight_stacks then sym8-quantizes.
    E = config.num_experts
    n_moe = 0
    for li, layer in enumerate(model.model.layers):
        gk_packed = f"{prefix}layers.{li}.mlp.experts.gate_up_proj"
        e0_key = f"{prefix}layers.{li}.mlp.experts.0.gate_proj.weight"
        if gk_packed in wm:
            gate_up = get(gk_packed)            # [E, 2I, d]
            down = get(f"{prefix}layers.{li}.mlp.experts.down_proj")  # [E, d, I]
            gate_w = gate_up[:, :inter, :].contiguous()
            up_w = gate_up[:, inter:, :].contiguous()
            del gate_up
        elif e0_key in wm:
            # stack the E per-expert slabs: gate/up_proj [I,d] -> [E,I,d]; down_proj [d,I] -> [E,d,I]
            gate_w = torch.stack(
                [get(f"{prefix}layers.{li}.mlp.experts.{e}.gate_proj.weight") for e in range(E)]).contiguous()
            up_w = torch.stack(
                [get(f"{prefix}layers.{li}.mlp.experts.{e}.up_proj.weight") for e in range(E)]).contiguous()
            down = torch.stack(
                [get(f"{prefix}layers.{li}.mlp.experts.{e}.down_proj.weight") for e in range(E)])  # [E, d, I]
        else:
            continue  # dense layer (no routed experts) — keep as loaded
        holder = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        holder.switch_mlp = MetalSwitchGLU.from_weight_stacks(
            gate_w, up_w, down.contiguous(), SwiGLU(), kernel, scheme)
        del gate_w, up_w, down
        n_moe += 1

    for f in open_cache.values():
        try:
            f.__exit__(None, None, None)
        except Exception:
            pass

    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    if hasattr(model.model, "reset_buffers"):
        model.model.reset_buffers()

    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"Parameters not loaded (non-expert): {meta[:6]} ... ({len(meta)})")
    print(f"streaming-metal: {n_moe} MoE layers -> {scheme} gather (fp16 never fully materialized)",
          flush=True)
    return model, kernel
