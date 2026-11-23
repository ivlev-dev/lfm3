#!/usr/bin/env python3
"""Detokenise greedy ids from q4k_host (or lfm-bench ref-gen) with the HF
tokenizer, i.e. exactly what `lfm-bench ref-gen` prints as `text:`.

    python3 q4k_detok.py --ids-file build/gen_q4k.log
    python3 q4k_detok.py 4741 20 309 278 5205
"""
import argparse
import json
import re
import sys

DEFAULT_TOK = "models/hf/tokenizer.json"


def bytes_to_unicode():
    # GPT-2 byte <-> printable unicode table (the ByteLevel decoder mapping)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--ids-file", default=None,
                    help="file with a line starting with 'ids:'")
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    a = ap.parse_args()

    tok = json.load(open(a.tokenizer))
    vocab = tok["model"]["vocab"]
    inv = {v: k for k, v in vocab.items()}
    for t in tok.get("added_tokens", []):
        inv[t["id"]] = t["content"]

    ids = list(a.ids)
    if a.ids_file:
        with open(a.ids_file) as f:
            for line in f:
                if line.startswith("ids:"):
                    for x in line.split()[1:]:
                        if x.lstrip("-").isdigit():
                            ids.append(int(x))
                        else:
                            break
                    break

    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    out = bytearray()
    for i in ids:
        s = inv.get(i, "")
        for ch in s:
            if ch in u2b:
                out.append(u2b[ch])
            else:
                out += ch.encode("utf-8", "replace")
    print("ids:", " ".join(str(i) for i in ids))
    print("text:", out.decode("utf-8", "replace"))


if __name__ == "__main__":
    main()
