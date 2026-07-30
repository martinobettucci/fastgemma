#!/usr/bin/env python3
"""Validate the fastgemma forward pass against HF transformers.

Loads the text tower only (bf16 — float32 would need ~18 GB), runs the same
token ids, and compares logits. Our int4 weights carry ~9% relative error by
construction, so exact agreement is not the bar; what this catches is
*structural* error — a wrong norm, a missed scale, RoPE on the wrong axis,
KV read from the wrong layer — all of which destroy top-1 agreement and
cosine similarity, not merely perturb them.

Usage:
    python3 compare_hf.py --src /models/g4e2b --logits /tmp/fgm_logits.bin \
        --tokens 2,2364,573,3287
"""

import argparse
import gc
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--tokens", default="2,2364,573,3287")
    ap.add_argument("--dump-hidden", default=None,
                    help="also dump HF per-layer hidden states to this .npz")
    a = ap.parse_args()

    import torch
    from transformers import Gemma4ForConditionalGeneration

    ids = [int(t) for t in a.tokens.split(",")]
    print(f"tokens: {ids}")

    print("loading HF model (bf16, text tower only)...")
    full = Gemma4ForConditionalGeneration.from_pretrained(
        a.src, dtype=torch.bfloat16, device_map="cpu",
    )
    lm = full.model.language_model
    lm_head = full.lm_head
    del full.model.vision_tower, full.model.audio_tower
    gc.collect()

    with torch.no_grad():
        inp = torch.tensor([ids], dtype=torch.long)
        out = lm(input_ids=inp, use_cache=False, output_hidden_states=a.dump_hidden is not None)
        hs = out.last_hidden_state
        logits = lm_head(hs).float()[0, -1]
        cap = lm.config.final_logit_softcapping
        if cap:
            logits = cap * torch.tanh(logits / cap)
        ref = logits.numpy().astype(np.float64)
        if a.dump_hidden:
            np.savez(a.dump_hidden,
                     **{f"h{i}": h[0].float().numpy() for i, h in enumerate(out.hidden_states)})

    got = np.fromfile(a.logits, dtype=np.float32).astype(np.float64)
    assert got.size == ref.size, f"size mismatch {got.size} vs {ref.size}"

    cos = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref)))
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    ta = np.argsort(-ref)
    tb = np.argsort(-got)

    def topk_overlap(k):
        return len(set(ta[:k].tolist()) & set(tb[:k].tolist())) / k

    # rank correlation over the head of the distribution
    def softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    pr, pg = softmax(ref), softmax(got)
    kl = float((pr * (np.log(pr + 1e-12) - np.log(pg + 1e-12))).sum())

    print()
    print(f"  cosine similarity   {cos:.6f}")
    print(f"  relative L2 error   {rel:.4f}")
    print(f"  KL(hf || fastgemma) {kl:.5f} nats")
    print(f"  top-1  agree        {ta[0] == tb[0]}   (hf={ta[0]} fgm={tb[0]})")
    for k in (5, 10, 50):
        print(f"  top-{k:<3d} overlap      {topk_overlap(k) * 100:.0f}%")
    print()
    print(f"  hf   top5: {[(int(i), round(float(ref[i]), 2)) for i in ta[:5]]}")
    print(f"  fgm  top5: {[(int(i), round(float(got[i]), 2)) for i in tb[:5]]}")

    ok = cos > 0.98 and ta[0] == tb[0] and topk_overlap(5) >= 0.8
    print()
    print("VERDICT:", "PASS - structurally correct" if ok else "FAIL - structural bug")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
