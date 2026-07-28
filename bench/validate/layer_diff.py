#!/usr/bin/env python3
"""Layer-by-layer divergence between fastgemma and HF.

A structural bug shows up as a step change in relative error at one layer; pure
quantisation noise shows up as a smooth accumulation. This prints the per-layer
curve so the two are distinguishable.

    FGM_DUMP=/tmp/fgm_hidden.bin ./target/release/fgm-bench model.fgm 2,2364,573,3287
    python3 layer_diff.py --src /models/g4e2b --dump /tmp/fgm_hidden.bin
"""

import argparse
import gc
import struct
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tokens", default="2,2364,573,3287")
    a = ap.parse_args()

    with open(a.dump, "rb") as f:
        nl, m, hs = struct.unpack("<III", f.read(12))
        mine = np.frombuffer(f.read(), dtype=np.float32).reshape(nl, m, hs).astype(np.float64)
    print(f"fastgemma dump: {nl} layers x {m} tokens x {hs}")

    import torch
    from transformers import Gemma4ForConditionalGeneration

    ids = [int(t) for t in a.tokens.split(",")]
    full = Gemma4ForConditionalGeneration.from_pretrained(
        a.src, dtype=torch.bfloat16, device_map="cpu"
    )
    lm = full.model.language_model
    del full.model.vision_tower, full.model.audio_tower
    gc.collect()

    with torch.no_grad():
        out = lm(input_ids=torch.tensor([ids]), use_cache=False, output_hidden_states=True)
    hidden = [h[0].float().numpy().astype(np.float64) for h in out.hidden_states]
    print(f"hf hidden_states: {len(hidden)} entries of {hidden[0].shape}")

    # HF records layer *inputs*: hidden[0] = embeddings, hidden[l+1] = output of
    # layer l for l <= L-2, and hidden[-1] is last_hidden_state (post final norm).
    # Our dump is layer outputs 0..L-1 followed by our own post-norm output, so
    # hidden[l+1] lines up for l < L-1 and hidden[-1] lines up with our last entry.
    pairs = [(l, l + 1) for l in range(nl - 2)]
    pairs.append((nl - 1, len(hidden) - 1))   # post-norm vs last_hidden_state
    print("aligning: hf[l+1] <-> layer l ; hf[-1] <-> our post-norm output\n")
    print(f"  {'layer':>5} {'rel L2':>9} {'cos':>9} {'|hf|':>10} {'|fgm|':>10}")
    prev = 0.0
    for l, hi in pairs:
        ref = hidden[hi]
        got = mine[l]
        rel = np.linalg.norm(got - ref) / (np.linalg.norm(ref) + 1e-12)
        cos = float((got.ravel() @ ref.ravel())
                    / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))
        jump = " <-- JUMP" if rel > prev * 1.8 + 0.02 and l > 0 else ""
        tag = "post" if hi == len(hidden) - 1 else str(l)
        print(f"  {tag:>5} {rel:>9.4f} {cos:>9.5f} {np.linalg.norm(ref):>10.1f} "
              f"{np.linalg.norm(got):>10.1f}{jump}")
        prev = rel


if __name__ == "__main__":
    main()
