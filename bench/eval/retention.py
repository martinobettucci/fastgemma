#!/usr/bin/env python3
"""Long-context retention at the target 8k.

The other half of the behavioural gate. Tool choice can look fine on short
prompts while the engine quietly loses the middle of a long context -- a ring
buffer off by one, a sliding-window boundary, a position that maps to the wrong
slot. None of that shows up in a 200-token test, and all of it shows up here.

A fact is planted at a controlled depth in filler text of a target token length,
then asked for at the end. Scored on whether the generated text contains the
planted value. Depths sweep the whole context because the failure modes are
positional: a ring bug loses the oldest positions, a sliding-window bug loses
everything outside the last window, and a chunk-boundary bug loses whatever
landed at a multiple of the prefill chunk.

Run:
  python3 bench/eval/retention.py [--ctx 8192] [--model PATH]
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from promptfmt import Fmt, BOS, TURN_O, TURN_C  # noqa: E402

SYSTEM = "You are a helpful assistant. Answer using only the notes provided."

# Filler with no numbers in it, so the answer cannot be guessed from context.
FILLER = (
    "The maintenance log records routine inspections of the facility. "
    "Technicians walk the corridors, check the seals on each door, and note "
    "anything unusual in the daily book. Most days nothing of consequence "
    "happens. The lights hum, the ventilation runs, and the floors are swept "
    "before the evening shift begins. "
)

FACTS = [
    ("access code for the east wing", "XJ-4471"),
    ("serial number of the backup generator", "QW-8823"),
    ("badge number of the night supervisor", "RT-1096"),
    ("part number of the replacement filter", "ZB-5570"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/user/models/g4e2b-g256.fgm")
    ap.add_argument("--tokenizer", default="/home/user/models/g4e2b/tokenizer.json")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ngen", type=int, default=24)
    ap.add_argument("--depths", default="0.02,0.15,0.35,0.5,0.65,0.85,0.98")
    a = ap.parse_args()

    fmt = Fmt(a.tokenizer)
    filler_ids = fmt.enc(FILLER)
    depths = [float(d) for d in a.depths.split(",")]

    cases, prompts = [], []
    for di, depth in enumerate(depths):
        label, value = FACTS[di % len(FACTS)]
        fact = f"Note: the {label} is {value}. "
        fact_ids = fmt.enc(fact)
        # question and framing are outside the filler budget
        question = f"Using the notes above, what is the {label}? Answer with the code only."
        head = fmt.enc("system\n" + SYSTEM)
        tail = fmt.enc("user\n") + fmt.enc(question)
        budget = a.ctx - len(head) - len(tail) - len(fact_ids) - 16
        assert budget > 0, "context too small for the framing"
        reps = budget // len(filler_ids) + 1
        body = (filler_ids * reps)[:budget]
        at = int(len(body) * depth)
        body = body[:at] + fact_ids + body[at:]

        ids = ([BOS, TURN_O] + head + body + [TURN_C]
               + [TURN_O] + tail + [TURN_C] + [TURN_O] + fmt.enc("model\n"))
        cases.append({"depth": depth, "label": label, "value": value, "len": len(ids)})
        prompts.append(ids)

    pf = HERE / ".retention.txt"
    pf.write_text("\n".join(",".join(str(t) for t in p) for p in prompts) + "\n")
    print(f"{len(cases)} depths at ~{cases[0]['len']} tokens each")

    r = subprocess.run(
        [str(ROOT / "target/release/fgm-bench"), "genfile", a.model, str(pf), str(a.ngen)],
        capture_output=True, text=True, env=dict(os.environ),
    )
    if r.returncode != 0:
        sys.exit(f"fgm-bench failed ({r.returncode}):\n{r.stderr[-3000:]}")
    for line in r.stderr.splitlines():
        if " prompts," in line:
            print("  " + line)

    # keep empty lines: a prompt whose first generated token is EOS
    # legitimately produces no ids, and dropping it silently
    # misaligns every case after it
    outs = r.stdout.split("\n")
    while outs and outs[-1] == "":
        outs.pop()
    assert len(outs) == len(cases), f"{len(outs)} outputs for {len(cases)} cases"

    hits = 0
    print(f"\n  {'depth':>6} {'tokens':>7}  {'found':>5}  answer")
    for case, line in zip(cases, outs):
        ids = [int(x) for x in line.split(",")] if line.strip() else []
        text = fmt.dec(ids).strip()
        ok = case["value"].lower() in text.lower().replace(" ", "")
        ok = ok or case["value"].lower() in text.lower()
        hits += ok
        print(f"  {case['depth']:>6.2f} {case['len']:>7}  {'yes' if ok else 'NO ':>5}  "
              f"{text[:70]!r}")
    n = len(cases)
    print(f"\n  retention at {a.ctx} tokens: {hits}/{n} = {100*hits/n:.0f}%")
    json.dump({"ctx": a.ctx, "hits": hits, "n": n}, open(HERE / ".retention.json", "w"))
    if hits < n:
        sys.exit(1)


if __name__ == "__main__":
    main()
