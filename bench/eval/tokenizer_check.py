#!/usr/bin/env python3
"""Check the Rust tokenizer against HF `tokenizers`, id for id.

Round-tripping text through encode/decode is a necessary test and a weak one:
an encoder that emitted one byte-fallback token per byte would pass it, while
producing 4x the tokens and a prompt the model has never seen in that shape.
The only test that catches that is comparing the *ids* against the reference
implementation on text that exercises the cases the config actually declares --
merges, byte fallback, control tokens, and the space normalisation.

    python3 bench/eval/tokenizer_check.py [--tokenizer PATH]
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

CASES = [
    "",
    "a",
    " ",
    "Hello, world!",
    "The capital of France is",
    "The quick brown fox jumps over the lazy dog.",
    "  leading and  doubled   spaces ",
    "trailing space ",
    "\nnewlines\n\nand\ttabs\t\t",
    "unicode: éàü ß ñ",
    "日本語のテキストです",
    "emoji 🎉🚀 and combining é",
    "numbers 0123456789 and 3.14159 and -42",
    "punctuation!?;:'\"()[]{}<>/\\|@#$%^&*~`",
    "CamelCaseAndsnake_case_and-kebab-case",
    "a" * 200,
    "<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>,days:3}<tool_call|>",
    "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n",
    "def f(x):\n    return x ** 2 + 1  # a comment\n",
    "Mixed 日本語 and English and 🎉 in one line, with  spaces.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/home/user/models/g4e2b/tokenizer.json")
    ap.add_argument("--bin", default=str(ROOT / "target/release/fgm-tokcheck"))
    a = ap.parse_args()

    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("pip install tokenizers")
    ref = Tokenizer.from_file(a.tokenizer)

    payload = json.dumps(CASES)
    r = subprocess.run([a.bin, a.tokenizer], input=payload, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{a.bin} failed:\n{r.stderr[-2000:]}")
    ours = json.loads(r.stdout)

    bad = 0
    for text, mine in zip(CASES, ours):
        want = ref.encode(text, add_special_tokens=False).ids
        if mine != want:
            bad += 1
            print(f"MISMATCH {text!r}")
            print(f"  ref  ({len(want)}): {want}")
            print(f"  ours ({len(mine)}): {mine}")
            # first divergence, which is usually the whole story
            for i, (x, y) in enumerate(zip(want, mine)):
                if x != y:
                    print(f"  first differs at {i}: {ref.decode([x])!r} vs {ref.decode([y])!r}")
                    break
        else:
            print(f"ok  {len(want):4d} tok  {text[:48]!r}")

    print(f"\n{len(CASES) - bad}/{len(CASES)} cases match the reference")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
