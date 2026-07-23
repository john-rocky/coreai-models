# Community port — NOT an Apple model.
"""Gemma-4 E2B MTP drafter (Section-11 transplant) for the spec-decode host loop.

Google ships an EAGLE-class trained draft head inside the ``.litertlm`` (litert-lm
never enables it). Wiring fully decoded in
coreai-models-community/knowledge/gemma4-mixedbit-qat-transplant.md ("MTP drafter"
+ "P5b DONE" sections); measured acceptance vs the real tflite = 2.506 tokens/step.

Graph (all verified against the Section-11 op dump + llm_litert_mtp_drafter.cc):

  x = pre_proj( concat(emb_norm(token), hidden) )          # [1,1,256], emb FIRST
  4 layers (gemma post-norm residual, per-layer skip_scale), NO own KV:
    layers 0-2 cross-attend the MAIN unified cache slot 11 (= L13 sliding, hd 256,
    window 512); layer 3 reads slot 14 (= L14 full, hd 512). q-norm (weighted RMS),
    RoPE(main inv-freqs), SDPA scale 1.0.
  fin = final_norm(x)
  logits = softcap_30( int4_head(fin) )                    # [262144, 256] Metal int4
  proj   = post_proj(fin)                                  # [1536, 256] chained hidden

Host contract (same as the decode bundle so state buffers can be SHARED):
  inputs  input_ids [1,1] i32, hidden [1,1,1536] f16, position_ids [1, seq] i32
          (seq = p: cache rows 0..p-1 are valid, query sits at p-1 = the position
          of the hidden's token — litert keeps this FIXED across the G=3 chain)
  states  keyCache/valueCache — the MAIN model's unified pipelined KV pair
          [n_slots, 1, 1, max_seq, 512] (read-only here; bind the same NDArrays)
  outputs logits [1,1,262144], proj_hidden [1,1,1536]
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA

DRAFTER_DIM = 256
HIDDEN = 1536
N_HEADS = 4
VOCAB = 262144
SLIDING_SLOT = 11   # main unified-cache slot of L13 (sliding producer)
FULL_SLOT = 14      # main unified-cache slot of L14 (full producer)
WINDOW = 512        # confirmed from the Section-11 mask constants


def _act_q(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Static symmetric int8 fake-quant (zp=0) — the drafter's QAT operating point.

    The Section-11 QUANTIZE ops carry FIXED calibrated per-tensor scales; the
    drafter was TRAINED behind them, so reproducing them is part of the transplant
    (fp32/fp16 activations measurably lose draft acceptance).
    """
    return torch.clamp(torch.round(x / scale), -128.0, 127.0) * scale


class _DrafterLayer(nn.Module):
    """One drafter block: cross-attn into a main-cache slot + gated FFN, post-norms."""

    def __init__(self, head_dim: int, slot: int, window: int, eps: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.slot = slot
        self.q_proj = nn.Linear(DRAFTER_DIM, N_HEADS * head_dim, bias=False)
        self.o_proj = nn.Linear(N_HEADS * head_dim, DRAFTER_DIM, bias=False)
        self.gate_proj = nn.Linear(DRAFTER_DIM, 2048, bias=False)
        self.up_proj = nn.Linear(DRAFTER_DIM, 2048, bias=False)
        self.down_proj = nn.Linear(2048, DRAFTER_DIM, bias=False)
        self.pre_attention_norm = RMSNorm(DRAFTER_DIM, eps=eps)
        self.post_attention_norm = RMSNorm(DRAFTER_DIM, eps=eps)
        self.pre_ffw_norm = RMSNorm(DRAFTER_DIM, eps=eps)
        self.post_ffw_norm = RMSNorm(DRAFTER_DIM, eps=eps)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.rope = RoPE(scale=1.0)
        # masking comes from the explicit host-built mask input (window folded in)
        self.sdpa = SDPA(scale=1.0, is_causal=False, window_size=0)
        self.window = window
        self.register_buffer("skip_scale", torch.ones(1))
        # static activation-quant scales (QAT operating point), filled by load_transplant
        self.register_buffer("aq_q", torch.ones(1))
        self.register_buffer("aq_attn_vec", torch.ones(1))
        self.register_buffer("aq_gating", torch.ones(1))
        self.register_buffer("aq_down", torch.ones(1))
        # main-model inv_freq for this slot's layer type (fp32 attribute, not buffer)
        self.inv_freq: torch.Tensor | None = None

    def forward(
        self,
        x: torch.Tensor,
        rope_pos: torch.Tensor,   # [1, 1] int32 — the query position (p-1)
        k_slot: torch.Tensor,     # [1, 1, MAX, 512] FIXED length (padded slot rows)
        v_slot: torch.Tensor,
        mask: torch.Tensor,       # [1, 1, 1, MAX] additive fp mask (0 valid / -inf)
    ) -> torch.Tensor:
        b, s, _ = x.shape
        hd = self.head_dim

        residual = x
        h = _act_q(self.pre_attention_norm(x), self.aq_q)
        q = self.q_proj(h).view(b, s, N_HEADS, hd).permute(0, 2, 1, 3)
        q = self.q_norm(q)
        q = self.rope(q, position_ids=rope_pos, freqs=self.inv_freq)

        k = k_slot
        v = v_slot
        if hd < k.shape[-1]:
            k = k.narrow(-1, 0, hd)
            v = v.narrow(-1, 0, hd)

        # Fixed-length K + explicit BOOLEAN mask (the Section-11 tflite's own design;
        # dynamic narrows on inputs crash the iOS MPSGraph ViewOp path, and the SDPA
        # composite only handles bool masks exactly — additive fp masks diverge).
        out = self.sdpa(query=q, key=k, value=v, attn_mask=mask > 0.5)
        out = out.permute(0, 2, 1, 3).reshape(b, s, N_HEADS * hd)
        out = self.o_proj(_act_q(out, self.aq_attn_vec))
        x = residual + self.post_attention_norm(out)

        residual = x
        h = _act_q(self.pre_ffw_norm(x), self.aq_gating)
        ffn = self.down_proj(_act_q(
            nn.functional.gelu(self.gate_proj(h), approximate="tanh") * self.up_proj(h),
            self.aq_down))
        x = residual + self.post_ffw_norm(ffn)
        return x * self.skip_scale


class Gemma4MtpDrafter(nn.Module):
    """Decode-shaped drafter. ``embed_tokens`` and ``lm_head`` are attached by the
    export script (PackedInt2Embedding sharing the main table bytes; Metal int4 head)."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, eps: float = 1e-6, softcap: float = 30.0) -> None:
        super().__init__()
        self.pre_proj = nn.Linear(2 * HIDDEN, DRAFTER_DIM, bias=False)
        self.post_proj = nn.Linear(DRAFTER_DIM, HIDDEN, bias=False)
        self.layers = nn.ModuleList([
            _DrafterLayer(256, SLIDING_SLOT, WINDOW, eps),
            _DrafterLayer(256, SLIDING_SLOT, WINDOW, eps),
            _DrafterLayer(256, SLIDING_SLOT, WINDOW, eps),
            _DrafterLayer(512, FULL_SLOT, 0, eps),
        ])
        self.final_norm = RMSNorm(DRAFTER_DIM, eps=eps)
        self.softcap = softcap
        self.register_buffer("aq_pre_proj", torch.ones(1))
        self.register_buffer("aq_post_proj", torch.ones(1))
        self.embed_tokens: nn.Module | None = None
        self.lm_head: nn.Module | None = None

    def forward(
        self,
        input_ids: torch.Tensor,     # [1, 1] int32
        hidden: torch.Tensor,        # [1, 1, 1536] f16 (main final-norm output)
        pos: torch.Tensor,           # [1, 1] int32 — query position p-1 (FIXED over the chain)
        mask_sliding: torch.Tensor,  # [1,1,1,MAX] fp16 1=attend/0=not (window 512 at pos)
        mask_full: torch.Tensor,     # [1,1,1,MAX] fp16 1=attend/0=not (causal 0..pos)
        k11: torch.Tensor,           # main slot 11 (= L13 sliding) [1,1,MAX,512] FIXED
        v11: torch.Tensor,
        k14: torch.Tensor,           # main slot 14 (= L14 full)    [1,1,MAX,512] FIXED
        v14: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed_tokens(input_ids)                     # normalized (x sqrt(1536))
        act = torch.cat([emb, hidden], dim=-1)                 # emb FIRST (litert concat)
        x = self.pre_proj(_act_q(act, self.aq_pre_proj))
        for layer in self.layers:
            if layer.slot == SLIDING_SLOT:
                x = layer(x, pos, k11, v11, mask_sliding)
            else:
                x = layer(x, pos, k14, v14, mask_full)
        fin = self.final_norm(x)

        logits = self.lm_head(fin)                             # head input NOT act-quantized
        logits = torch.tanh(logits / self.softcap) * self.softcap
        proj = self.post_proj(_act_q(fin, self.aq_post_proj))
        # In-graph greedy pick so chained drafting never round-trips to the host:
        # the next step's input_ids / the verify ids slot binds this output directly.
        amax = torch.argmax(logits, dim=-1).to(torch.int32)   # [1, 1] int32
        return logits, proj, amax


# ---- transplant loading (extraction artifacts -> module weights) ---------------------------------

def _dq_int8(flat_u8: torch.Tensor, scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    codes = flat_u8.reshape(rows, cols).view(torch.int8).float()
    return codes * scale.float().unsqueeze(1)


def _dq_int4(flat_u8: torch.Tensor, scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    p = flat_u8.reshape(rows, cols // 2).to(torch.int16)
    c = torch.stack([p & 0xF, p >> 4], dim=-1).reshape(rows, cols)
    codes = torch.where(c >= 8, c - 16, c).float()
    return codes * scale.float().unsqueeze(1)


def load_transplant(model: "Gemma4MtpDrafter", extract_root: str,
                    dtype: torch.dtype = torch.float32,
                    fp_head: bool = True) -> None:
    """Fill the module from the P1/P5c extraction artifacts (dequantized weights).

    ``fp_head=True`` also materializes the 262144x256 head as a plain Linear
    (parity/oracle use); the export script instead attaches the Metal int4 module.
    """
    import json
    from pathlib import Path

    from safetensors import safe_open

    root = Path(extract_root)
    fw = safe_open(str(root / "gemma4e2b_mixedbit_weights.safetensors"), framework="pt")
    fx = safe_open(str(root / "gemma4e2b_drafter_extras.safetensors"), framework="pt")
    sc = json.loads((root / "gemma4e2b_drafter_extras.json").read_text())
    aq = json.loads((root / "gemma4e2b_drafter_act_quant.json").read_text())

    def aq_scale(substr: str) -> torch.Tensor:
        hits = [v["scale"] for k, v in aq.items() if substr in k]
        assert len(hits) == 1, (substr, hits)
        return torch.tensor([hits[0]], dtype=torch.float32)

    def w8(key: str, rows: int, cols: int) -> torch.Tensor:
        return _dq_int8(fw.get_tensor(key), fw.get_tensor(key + ".scale"), rows, cols).to(dtype)

    def w4(key: str, rows: int, cols: int) -> torch.Tensor:
        return _dq_int4(fw.get_tensor(key), fw.get_tensor(key + ".scale"), rows, cols).to(dtype)

    with torch.no_grad():
        model.pre_proj.weight.copy_(w8("drafter.dot_general", DRAFTER_DIM, 2 * HIDDEN))
        model.post_proj.weight.copy_(_dq_int8(
            fx.get_tensor("post_proj.weight_int8").flatten().view(torch.uint8),
            fx.get_tensor("post_proj.scale"), HIDDEN, DRAFTER_DIM).to(dtype))
        model.final_norm.weight.copy_(fx.get_tensor("final_norm").to(dtype))
        model.aq_pre_proj.copy_(aq_scale("mtp_pre_proj"))
        model.aq_post_proj.copy_(aq_scale("mtp_post_proj"))
        for i, layer in enumerate(model.layers):
            P = f"drafter.layer_{i:02d}."
            X = f"layer_{i}."
            hd = layer.head_dim
            layer.q_proj.weight.copy_(w8(P + "attn.q", N_HEADS * hd, DRAFTER_DIM))
            layer.o_proj.weight.copy_(w8(P + "attn.o", DRAFTER_DIM, N_HEADS * hd))
            layer.gate_proj.weight.copy_(w4(P + "mlp.gating1", 2048, DRAFTER_DIM))
            layer.up_proj.weight.copy_(w4(P + "mlp.gating2", 2048, DRAFTER_DIM))
            layer.down_proj.weight.copy_(w4(P + "mlp.down", DRAFTER_DIM, 2048))
            for norm in ("pre_attention_norm", "post_attention_norm",
                         "pre_ffw_norm", "post_ffw_norm", "q_norm"):
                getattr(layer, norm).weight.copy_(fx.get_tensor(X + norm).to(dtype))
            layer.skip_scale.copy_(torch.tensor([sc[X + "skip_scale"]], dtype=dtype))
            layer.aq_q.copy_(aq_scale(f"layer_{i}/layer_{i}.pre_q/attn.pre_q"))
            layer.aq_attn_vec.copy_(aq_scale(f"layer_{i}/layer_{i}.post_qkv/attn.post_qkv"))
            layer.aq_gating.copy_(aq_scale(f"layer_{i}/layer_{i}.post_qkv/mlp/gating_einsum1"))
            layer.aq_down.copy_(aq_scale(f"layer_{i}/layer_{i}.post_qkv/mlp/linear"))
            freq_key = "rope_freq_128" if hd == 256 else "rope_freq_256"
            layer.inv_freq = fx.get_tensor(freq_key).float()
        if fp_head:
            head = torch.nn.Linear(DRAFTER_DIM, VOCAB, bias=False)
            head.weight.copy_(w4("drafter.lm_head", VOCAB, DRAFTER_DIM))
            model.lm_head = head.to(dtype)
