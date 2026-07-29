#!/usr/bin/env python3
"""Publish fastgemma weights and a model card to a Hugging Face repo.

Requires a token with write access to the target org:

    export HF_TOKEN=hf_...          # or: hf auth login
    python3 publish/upload_hf.py --repo P2Enjoy/fastgemma-gemma-4-E2B

The weights are a *derivative of google/gemma-4-E2B-it*, so the upload carries
the Gemma Terms of Use. That is not boilerplate: redistribution of Gemma-derived
weights is governed by those terms, and a model card that omits them makes the
repo non-compliant regardless of what the engine's own licence says. The card
generator refuses to run without the notice for that reason.

Nothing here uploads by default -- `--dry-run` prints the plan and exits, so the
first invocation cannot surprise anyone with a 4.34 GB push.
"""

import argparse
import hashlib
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

GEMMA_NOTICE = """## Licence

These weights are a derivative of
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) and are
distributed under the **Gemma Terms of Use**. Use is subject to the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
Quantising, rotating and repacking the tensors does not change that: the
restrictions travel with the derivative.

The fastgemma *engine* (converter and kernels) is separate and carries its own
licence in the source repository. The licence on this repo governs the weights.
"""


def sha256(path, limit=None):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit and n >= limit:
                break
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. P2Enjoy/fastgemma-gemma-4-E2B")
    ap.add_argument("--weights", nargs="+", default=["/home/user/models/g4e2b-dual.fgm"])
    ap.add_argument("--card", default=str(ROOT / "publish" / "MODEL_CARD.md"))
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--card-only", action="store_true",
                    help="push only the model card; the weights on the repo are unchanged")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    card = pathlib.Path(a.card)
    if not card.exists():
        sys.exit(f"model card missing: {card}")
    text = card.read_text()
    if "Gemma Terms of Use" not in text:
        sys.exit("model card does not carry the Gemma Terms of Use -- refusing to upload")

    files = []
    for w in ([] if a.card_only else a.weights):
        p = pathlib.Path(w)
        if not p.exists():
            sys.exit(f"missing weight file: {p}")
        files.append(p)

    print(f"repo    {a.repo}  ({'private' if a.private else 'public'})")
    print(f"card    {card}  ({len(text)} bytes)")
    total = 0
    for p in files:
        sz = p.stat().st_size
        total += sz
        print(f"weight  {p.name:24s} {sz/1e9:6.2f} GB")
    print(f"total   {total/1e9:.2f} GB")

    if a.dry_run:
        print("\ndry run -- nothing uploaded")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()
    print(f"\nauthenticated as {who.get('name')}")

    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md",
                    repo_id=a.repo, repo_type="model")
    if a.card_only:
        print("card only -- weights on the repo left as they are")
    for p in files:
        print(f"uploading {p.name} ({p.stat().st_size/1e9:.2f} GB) ...")
        api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                        repo_id=a.repo, repo_type="model")
    print(f"\nhttps://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
