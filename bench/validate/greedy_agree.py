#!/usr/bin/env python3
"""Greedy-agreement validation: does fastgemma pick the same token as HF?

Single-token top-1 comparison is not a usable signal on this model — HF's own
top-5 logits sit within ~0.4 of each other and bf16 reduction order varies with
thread scheduling, so HF disagrees with *itself* run to run. This measures the
statistic that actually matters for tool-call exactness: over many positions,
how often does the engine's argmax match, and how often is the reference's
choice still in our top-k.

It also runs the HF reference twice and reports HF-vs-HF agreement as the noise
floor, so our number can be read against what "identical" even means here.

    ./target/release/fgm-bench dump model.fgm <tokens-csv> /tmp/logits.bin
    python3 greedy_agree.py --src /models/g4e2b --logits /tmp/logits.bin \
        --tokens <same-csv>
"""

import argparse
import gc
import numpy as np


def topk_sets(x, k):
    return np.argpartition(-x, k, axis=-1)[:, :k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--logits", required=True)
    ap.add_argument("--tokens", help="comma-separated ids; omit to use --text")
    ap.add_argument("--text", help="real text to tokenize (preferred: random ids "
                                   "give a near-uniform next-token distribution, "
                                   "so argmax is unstable by construction)")
    ap.add_argument("--label", default="fastgemma")
    a = ap.parse_args()

    import torch

    # The reference MUST be single-threaded. torch's multithreaded bf16 path is
    # not deterministic on this model: at 4 threads HF agrees with itself on only
    # ~86% of positions with max|logit diff| ~10.7, which would silently inflate
    # the apparent noise floor. At 1 thread it is bit-identical run to run.
    torch.set_num_threads(1)
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    if a.text is not None:
        tok = AutoTokenizer.from_pretrained(a.src)
        ids = tok(a.text, add_special_tokens=True)["input_ids"]
        print(f"tokenized {len(a.text)} chars -> {len(ids)} tokens")
        print("ids:", ",".join(map(str, ids)))
    else:
        ids = [int(t) for t in a.tokens.split(",")]
    m = len(ids)

    full = Gemma4ForConditionalGeneration.from_pretrained(
        a.src, dtype=torch.bfloat16, device_map="cpu"
    )
    lm, head = full.model.language_model, full.lm_head
    del full.model.vision_tower, full.model.audio_tower
    gc.collect()
    cap = lm.config.final_logit_softcapping

    def hf_logits():
        with torch.no_grad():
            hs = lm(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state
            lg = head(hs).float()[0]
            if cap:
                lg = cap * torch.tanh(lg / cap)
            return lg.numpy().astype(np.float32)

    ref = hf_logits()
    ref2 = hf_logits()   # sanity: must be identical now that torch is 1-thread
    got = np.fromfile(a.logits, dtype=np.float32).reshape(m, -1)
    assert got.shape == ref.shape, f"shape {got.shape} vs {ref.shape}"

    def report(name, x, y):
        ax, ay = x.argmax(-1), y.argmax(-1)
        agree = float((ax == ay).mean())
        # is y's pick inside x's top-k?
        in_k = {}
        for k in (1, 3, 5, 10):
            tk = topk_sets(x, k)
            in_k[k] = float(np.mean([ay[i] in tk[i] for i in range(len(ay))]))
        gap = np.sort(x, -1)[:, -1] - np.sort(x, -1)[:, -2]
        dis = ax != ay
        cos = float(np.mean([
            x[i] @ y[i] / (np.linalg.norm(x[i]) * np.linalg.norm(y[i]) + 1e-12)
            for i in range(len(ax))
        ]))
        print(f"  {name}")
        print(f"     greedy agreement   {agree * 100:5.1f}%  ({int(agree * m)}/{m} positions)")
        for k in (1, 3, 5, 10):
            print(f"     pick in hf top-{k:<2d}   {in_k[k] * 100:5.1f}%")
        print(f"     mean cosine        {cos:.5f}")
        if dis.any():
            print(f"     median top1-top2 logit gap: {np.median(gap):.2f} overall, "
                  f"{np.median(gap[dis]):.2f} where they disagree")
        return agree

    print(f"\npositions: {m}\n")
    print("determinism check (same HF model twice, 1 thread — expect 100%):")
    report("hf vs hf", ref, ref2)
    print()
    print("engine under test:")
    report(f"hf vs {a.label}", ref, got)


if __name__ == "__main__":
    main()
