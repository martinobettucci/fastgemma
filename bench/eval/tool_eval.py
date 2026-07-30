#!/usr/bin/env python3
"""Behavioural acceptance test for tool calling.

This is the accuracy gate, and it deliberately is not logit agreement with a
reference implementation. The engine trades numerical precision for speed on
purpose -- int4 weights, int8 activations, a Hadamard rotation, a different
softmax evaluation order -- so its numbers will not match another engine's and
were never going to. What must hold is behavioural: given a request and 12 tool
declarations, pick the right tool and fill in the right arguments.

Scored per case:
  parses   the output is a structurally well-formed Gemma 4 tool call
  tool     the selected tool is the expected one
  args     every expected argument is present with the expected value
  exact    all of the above

Run:
  python3 bench/eval/tool_eval.py [--constrained] [--model PATH]
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from promptfmt import Fmt  # noqa: E402

SYSTEM = ("You are a helpful assistant with access to tools. "
          "When the user's request can be served by a tool, call it.")


def parse_call(s):
    """Parse Gemma 4's native call syntax; mirror of grammar::parse_call."""
    m = re.search(r"<\|tool_call>call:(\w+)\{", s)
    if not m:
        return None
    name = m.group(1)
    body = s[m.end():]
    end = body.find("}<tool_call|>")
    if end < 0:
        # tolerate a missing close token so an unconstrained run is still
        # scored on tool and argument choice rather than only on structure
        end = body.rfind("}")
        if end < 0:
            return None
    body, closed = body[:end], body[end:].startswith("}<tool_call|>")
    args, rest = {}, body.strip()
    while rest:
        c = rest.find(":")
        if c < 0:
            break
        key = rest[:c].strip()
        rest = rest[c + 1:].lstrip()
        if rest.startswith('<|"|>'):
            rest = rest[5:]
            e = rest.find('<|"|>')
            if e < 0:
                break
            val, rest = rest[:e], rest[e + 5:]
        else:
            e = rest.find(",")
            e = len(rest) if e < 0 else e
            val, rest = rest[:e].strip(), rest[e:]
        args[key] = val
        rest = rest.lstrip()
        if rest.startswith(","):
            rest = rest[1:].lstrip()
    return {"name": name, "args": args, "closed": closed}


def norm(v):
    v = v.strip().strip('"').strip("'").lower()
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/user/models/g4e2b-g256.fgm")
    ap.add_argument("--tokenizer", default="/home/user/models/g4e2b/tokenizer.json")
    ap.add_argument("--constrained", action="store_true")
    ap.add_argument("--ngen", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    tools = json.load(open(HERE / "tools.json"))
    cases = json.load(open(HERE / "cases.json"))
    if a.limit:
        cases = cases[: a.limit]
    fmt = Fmt(a.tokenizer)

    prompts = [fmt.prompt(SYSTEM, tools, c["user"]) for c in cases]
    pf = HERE / ".prompts.txt"
    pf.write_text("\n".join(",".join(str(t) for t in p) for p in prompts) + "\n")
    print(f"{len(cases)} cases, {len(tools)} tools, "
          f"prompt {len(prompts[0])} tokens, constrained={a.constrained}")

    env = dict(os.environ)
    if a.constrained:
        env["FGM_GRAMMAR"] = a.tokenizer
        env["FGM_TOOLSPEC"] = str(HERE / "tools.json")
    r = subprocess.run(
        [str(ROOT / "target/release/fgm-bench"), "genfile", a.model, str(pf), str(a.ngen)],
        capture_output=True, text=True, env=env,
    )
    if r.returncode != 0:
        sys.exit(f"fgm-bench failed ({r.returncode}):\n{r.stderr[-3000:]}")
    for line in r.stderr.splitlines():
        if line.startswith(("constrained", "forced", "24 prompts")) or " prompts," in line:
            print("  " + line)

    # keep empty lines: a prompt whose first generated token is EOS
    # legitimately produces no ids, and dropping it silently
    # misaligns every case after it
    outs = r.stdout.split("\n")
    while outs and outs[-1] == "":
        outs.pop()
    assert len(outs) == len(cases), f"{len(outs)} outputs for {len(cases)} cases"

    tot = {"parses": 0, "tool": 0, "args": 0, "exact": 0}
    arg_hits = arg_total = 0
    failures = []
    for case, line in zip(cases, outs):
        ids = [int(x) for x in line.split(",")] if line.strip() else []
        text = fmt.dec(ids)
        got = parse_call(text)
        ok_parse = got is not None and got["closed"]
        ok_tool = got is not None and got["name"] == case["tool"]
        hits = 0
        if got is not None:
            for k, v in case["args"].items():
                if k in got["args"] and norm(got["args"][k]) == norm(v):
                    hits += 1
        arg_hits += hits
        arg_total += len(case["args"])
        ok_args = ok_tool and hits == len(case["args"])
        tot["parses"] += ok_parse
        tot["tool"] += ok_tool
        tot["args"] += ok_args
        tot["exact"] += ok_parse and ok_args
        if not (ok_parse and ok_args):
            failures.append((case, text[:220]))

    n = len(cases)
    print(f"\n  well-formed call     {tot['parses']:>3}/{n}  {100*tot['parses']/n:5.1f}%")
    print(f"  correct tool         {tot['tool']:>3}/{n}  {100*tot['tool']/n:5.1f}%")
    print(f"  all arguments right  {tot['args']:>3}/{n}  {100*tot['args']/n:5.1f}%")
    print(f"  argument-level       {arg_hits:>3}/{arg_total}  {100*arg_hits/arg_total:5.1f}%")
    print(f"  EXACT (gate)         {tot['exact']:>3}/{n}  {100*tot['exact']/n:5.1f}%")
    if failures:
        print(f"\n  {len(failures)} failure(s):")
        for case, text in failures[:8]:
            print(f"   - want {case['tool']}{case['args']}")
            print(f"     got  {text!r}")


if __name__ == "__main__":
    main()
