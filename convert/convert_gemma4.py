#!/usr/bin/env python3
"""Convert google/gemma-4-{E2B,E4B}-it safetensors into a fastgemma .fgm file.

Text tower only — the vision and audio towers are dropped, as are the k/v
projections of KV-shared layers, which the checkpoint still ships but the model
never reads (transformers lists them in _keys_to_ignore_on_load_unexpected).

Every linear weight is Hadamard-rotated along its K axis ("rotconv"). H is
orthogonal, so W' = W H^T combined with a runtime fast Walsh-Hadamard transform
of the activation reproduces the original product exactly, but both the weight
groups and the activation rows become outlier-free — which is what makes int4
weights and int8 activations safe. Measured on outlier-heavy weights this cuts
int4 relative error from 0.194 to 0.078.

Large tensors are quantised in row chunks and streamed to disk so peak RSS stays
a few hundred MB rather than the ~9 GB a full float32 PLE table would need.

Usage:
    python3 convert_gemma4.py --src /models/g4e2b --out /models/g4e2b.fgm
"""

import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fgm import FgmWriter  # noqa: E402
import quant as Q  # noqa: E402

TEXT = "model.language_model."
HAD = 128  # Hadamard block size; every K in this model is a multiple of it


# --------------------------------------------------------------- safetensors
class SafeTensors:
    """Minimal zero-copy safetensors reader (mmap, no torch)."""

    DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32,
          "I8": np.int8, "U8": np.uint8}

    def __init__(self, path):
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def __contains__(self, k):
        return k in self.header

    def shape(self, k):
        return self.header[k]["shape"]

    def _raw(self, k):
        e = self.header[k]
        s, t = e["data_offsets"]
        v = self.mm[self.base + s : self.base + t].view(self.DT[e["dtype"]])
        return v.reshape(e["shape"]) if e["shape"] else v.reshape(())

    def get(self, k):
        e = self.header[k]
        a = self._raw(k)
        return Q.bf16_to_f32(np.asarray(a)) if e["dtype"] == "BF16" else np.asarray(a, np.float32)

    def rows(self, k, r0, r1):
        """float32 copy of rows [r0, r1) — avoids materialising huge tensors."""
        e = self.header[k]
        a = self._raw(k)[r0:r1]
        return Q.bf16_to_f32(np.asarray(a)) if e["dtype"] == "BF16" else np.asarray(a, np.float32)


# ------------------------------------------------------------------ topology
class Topo:
    def __init__(self, cfg):
        t = cfg["text_config"]
        self.H = t["hidden_size"]
        self.L = t["num_hidden_layers"]
        self.I = t["intermediate_size"]
        self.dw = t.get("use_double_wide_mlp", False)
        self.nh = t["num_attention_heads"]
        self.nkv = t["num_key_value_heads"]
        self.hd = t["head_dim"]
        self.ghd = t.get("global_head_dim") or t["head_dim"]
        self.types = t["layer_types"]
        self.first_shared = self.L - t.get("num_kv_shared_layers", 0)
        self.ple_dim = t["hidden_size_per_layer_input"]
        self.vocab = t["vocab_size"]
        self.ple_vocab = t["vocab_size_per_layer_input"]
        self.eps = t["rms_norm_eps"]
        self.softcap = t.get("final_logit_softcapping")
        self.window = t["sliding_window"]
        self.rope = t["rope_parameters"]
        prev = self.types[: self.first_shared]
        self.store_full = sorted({len(prev) - 1 - prev[::-1].index(ty) for ty in set(prev)})

    def is_shared(self, i):
        return i >= self.first_shared

    def head_dim(self, i):
        return self.ghd if self.types[i] == "full_attention" else self.hd

    def inter(self, i):
        return self.I * (2 if (self.dw and self.is_shared(i)) else 1)


def rope_inv_freq(topo, layer_type):
    p = topo.rope[layer_type]
    base = p["rope_theta"]
    hd = topo.ghd if layer_type == "full_attention" else topo.hd
    if p.get("rope_type") == "proportional":
        n_rot = int(p.get("partial_rotary_factor", 1.0) * hd // 2)
        f = 1.0 / (base ** (np.arange(0, 2 * n_rot, 2, dtype=np.float64) / hd))
        nope = hd // 2 - n_rot
        if nope > 0:
            f = np.concatenate([f, np.zeros(nope, dtype=np.float64)])
        return (f / p.get("factor", 1.0)).astype(np.float32)
    return (1.0 / (base ** (np.arange(0, hd, 2, dtype=np.float64) / hd))).astype(np.float32)


# ------------------------------------------------------------------ converter
class Converter:
    def __init__(self, src, out, bits, rotate):
        self.cfg = json.load(open(os.path.join(src, "config.json")))
        self.topo = Topo(self.cfg)
        self.st = SafeTensors(os.path.join(src, "model.safetensors"))
        self.w = FgmWriter(out)
        self.w.start_body(header_reserve=1 << 21)
        self.bits = bits
        self.rotate = rotate
        self.err = {}
        self.t0 = time.time()
        self.out = out

    # -- helpers ---------------------------------------------------------
    def f32(self, name, arr):
        return self.w.add(name, np.ascontiguousarray(arr, np.float32), "f32", shape=arr.shape)

    def linear(self, name, key, bits, probe=False):
        """Quantise + rotate + AMX-pack a Linear weight [out, in], streamed."""
        out, k = self.st.shape(key)
        rot = HAD if (self.rotate and k % HAD == 0) else 0
        dt = "q8c" if bits == 8 else "q4g"
        g = k // Q.GROUP
        scales = np.empty((out, g), np.float16) if bits == 4 else np.empty(out, np.float32)
        # chunk rows so each chunk is a whole number of 16-row AMX n-blocks
        step = max(16, (1 << 24) // max(k, 1) // 16 * 16)
        self.w.begin(name, dt, [out, k], meta={"hadamard": rot})
        probed = None
        for r0 in range(0, out, step):
            r1 = min(out, r0 + step)
            blk = self.st.rows(key, r0, r1)
            if rot:
                blk = Q.apply_hadamard_k(blk, rot)
            if probe and probed is None:
                probed = (Q.rel_err(blk, dt), blk.shape)
            if bits == 8:
                blob, sc = Q.quant_q8c(blk)
            else:
                blob, sc = Q.quant_q4g(blk)   # sc is [g, rows]
                sc = np.ascontiguousarray(sc.T)
            scales[r0:r1] = sc
            self.w.append(blob)
        e = self.w.end()
        if bits == 4:  # store transposed [g, out]: a tile's 16 scales are contiguous
            self.w.add(name + ".scale", np.ascontiguousarray(scales.T), "f16", shape=(g, out))
        else:
            self.w.add(name + ".scale", scales, "f32", shape=(out,))
        if probe:
            self.err[name] = probed[0]
        return e

    def table(self, name, key, bits):
        """Row-major gather table (embeddings / PLE), streamed."""
        rows, k = self.st.shape(key)
        g = k // Q.GROUP
        dt = "q8r" if bits == 8 else "q4r"
        scales = np.empty((rows, g), np.float16) if bits == 4 else np.empty(rows, np.float32)
        step = max(1, (1 << 26) // max(k, 1))
        self.w.begin(name, dt, [rows, k])
        for r0 in range(0, rows, step):
            r1 = min(rows, r0 + step)
            blk = self.st.rows(key, r0, r1)
            if bits == 8:
                blob, sc = Q.quant_q8_rows(blk)
            else:
                blob, sc = Q.quant_q4_rows(blk)
            scales[r0:r1] = sc
            self.w.append(blob)
        self.w.end()
        self.w.add(name + ".scale", scales, "f16" if bits == 4 else "f32", shape=scales.shape)

    # -- main ------------------------------------------------------------
    def run(self):
        t, b = self.topo, self.bits
        print(f"  H={t.H} L={t.L} I={t.I} dw={t.dw} nkv={t.nkv} hd={t.hd}/{t.ghd} "
              f"first_shared={t.first_shared} rotate={self.rotate}")
        print(f"  store_full_length_kv layers: {t.store_full}")
        print(f"  bits: ffn={b['ffn']} attn={b['attn']} ple={b['ple']} emb={b['emb']}")

        print("  embed_tokens (gather rows + AMX-tiled lm_head)...")
        self.table("embed_tokens", TEXT + "embed_tokens.weight", b["emb"])
        self.linear("lm_head", TEXT + "embed_tokens.weight", b["emb"], probe=True)

        print(f"  embed_tokens_per_layer {self.st.shape(TEXT + 'embed_tokens_per_layer.weight')} "
              f"-> q{b['ple']} ...")
        self.table("embed_tokens_per_layer", TEXT + "embed_tokens_per_layer.weight", b["ple"])

        self.linear("per_layer_model_projection", TEXT + "per_layer_model_projection.weight", 8)
        self.f32("per_layer_projection_norm", self.st.get(TEXT + "per_layer_projection_norm.weight"))
        self.f32("norm", self.st.get(TEXT + "norm.weight"))

        def K(i, n):
            return f"{TEXT}layers.{i}.{n}"

        for i in range(t.L):
            shared = t.is_shared(i)
            for n in ("input_layernorm", "post_attention_layernorm",
                      "pre_feedforward_layernorm", "post_feedforward_layernorm",
                      "post_per_layer_input_norm"):
                self.f32(f"l{i}.{n}", self.st.get(K(i, n + ".weight")))
            self.f32(f"l{i}.layer_scalar", self.st.get(K(i, "layer_scalar")).reshape(1))
            self.f32(f"l{i}.q_norm", self.st.get(K(i, "self_attn.q_norm.weight")))

            self.linear(f"l{i}.q_proj", K(i, "self_attn.q_proj.weight"), b["attn"])
            self.linear(f"l{i}.o_proj", K(i, "self_attn.o_proj.weight"), b["attn"])
            if not shared:
                self.f32(f"l{i}.k_norm", self.st.get(K(i, "self_attn.k_norm.weight")))
                self.linear(f"l{i}.k_proj", K(i, "self_attn.k_proj.weight"), b["attn"])
                self.linear(f"l{i}.v_proj", K(i, "self_attn.v_proj.weight"), b["attn"])

            pr = i in (0, t.L // 2, t.L - 1)
            self.linear(f"l{i}.gate_proj", K(i, "mlp.gate_proj.weight"), b["ffn"], probe=pr)
            self.linear(f"l{i}.up_proj", K(i, "mlp.up_proj.weight"), b["ffn"])
            self.linear(f"l{i}.down_proj", K(i, "mlp.down_proj.weight"), b["ffn"], probe=pr)
            self.linear(f"l{i}.per_layer_input_gate", K(i, "per_layer_input_gate.weight"), 8)
            self.linear(f"l{i}.per_layer_projection", K(i, "per_layer_projection.weight"), 8)
            if i % 5 == 0 or i == t.L - 1:
                print(f"    layer {i:2d} {'shared' if shared else 'kv    '} "
                      f"hd={t.head_dim(i)} inter={t.inter(i)}  [{time.time() - self.t0:5.1f}s]")

        for ty in sorted(set(t.types)):
            self.f32(f"rope.{ty}.inv_freq", rope_inv_freq(t, ty))

        cfg = {
            "hidden_size": t.H, "num_hidden_layers": t.L, "intermediate_size": t.I,
            "use_double_wide_mlp": t.dw, "num_attention_heads": t.nh,
            "num_key_value_heads": t.nkv, "head_dim": t.hd, "global_head_dim": t.ghd,
            "layer_types": t.types, "first_shared_layer": t.first_shared,
            "store_full_length_kv": t.store_full,
            "hidden_size_per_layer_input": t.ple_dim, "vocab_size": t.vocab,
            "vocab_size_per_layer_input": t.ple_vocab, "rms_norm_eps": t.eps,
            "final_logit_softcapping": t.softcap, "sliding_window": t.window,
            "embed_scale": math.sqrt(t.H), "ple_embed_scale": math.sqrt(t.ple_dim),
            "ple_model_projection_scale": t.H ** -0.5, "ple_input_scale": 2.0 ** -0.5,
            "hadamard": HAD if self.rotate else 0,
            "quant": dict(self.bits),
        }
        self.w.meta.update({"convert_s": round(time.time() - self.t0, 1),
                            "quant_rel_err": {k: round(v, 5) for k, v in self.err.items()}})
        self.w.close(cfg)
        sz = os.path.getsize(self.out)
        print(f"\n  wrote {self.out}  {sz / 1e9:.2f} GB in {time.time() - self.t0:.0f}s")
        for k, v in self.err.items():
            print(f"    rel_err {k:28s} {v:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffn-bits", type=int, default=4)
    ap.add_argument("--attn-bits", type=int, default=8)
    ap.add_argument("--ple-bits", type=int, default=4)
    ap.add_argument("--emb-bits", type=int, default=4)
    ap.add_argument("--no-rotate", action="store_true")
    a = ap.parse_args()
    print(f"converting {a.src} -> {a.out}")
    Converter(a.src, a.out,
              {"ffn": a.ffn_bits, "attn": a.attn_bits, "ple": a.ple_bits, "emb": a.emb_bits},
              not a.no_rotate).run()
