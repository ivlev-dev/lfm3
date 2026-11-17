#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "The quick brown fox jumps over the lazy dog.",
    "def fibonacci(n):",
    "Столица Франции —",
]


def ids(output: str) -> list[int]:
    matches = re.findall(r"ids:\s*\[([^\]]*)\]", output)
    if not matches:
        raise SystemExit(f"no ids in output:\n{output}")
    return [int(x) for x in matches[-1].split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/LFM3-1.7B-F16.gguf")
    ap.add_argument("--greedy", default="/tmp/greedy")
    ap.add_argument("--ours", default="./target/release/lfm-bench")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prompt", action="append")
    args = ap.parse_args()
    prompts = args.prompt or PROMPTS
    if not os.path.exists(args.greedy):
        raise SystemExit(
            f"{args.greedy} not found: build tools/oracle/greedy.cpp (see README)"
        )
    ok = 0
    for prompt in prompts:
        llama = ids(
            subprocess.run(
                [args.greedy, args.model, prompt, str(args.tokens)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        ours = ids(
            subprocess.run(
                [
                    args.ours,
                    "ref-gen",
                    "--model",
                    args.model,
                    "--prompt",
                    prompt,
                    "--tokens",
                    str(args.tokens),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        diff = next((i for i in range(min(len(llama), len(ours))) if llama[i] != ours[i]), None)
        if diff is None and len(llama) == len(ours):
            ok += 1
            print(f"OK   {prompt!r} ({len(llama)} ids)")
        else:
            print(f"DIFF {prompt!r} len {len(llama)}/{len(ours)} first diff {diff}")
            if diff is not None:
                lo = max(0, diff - 2)
                print(f"  llama {llama[lo:diff + 4]}")
                print(f"  ours  {ours[lo:diff + 4]}")
    print(f"parity {ok}/{len(prompts)}")
    return 0 if ok == len(prompts) else 1


if __name__ == "__main__":
    sys.exit(main())
