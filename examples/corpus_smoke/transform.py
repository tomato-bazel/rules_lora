#!/usr/bin/env python3
"""Smoke transform for `lora_corpus`.

Contract (per `lora_corpus` rule docs):
  * Repeated  --input <path>       once per file in `source`.
  * Repeated  --corpus-dep <path>  once per validated upstream JSONL.
  * Single    --output <path>      where to write the SFT JSONL.

Behavior: read every `--input` file line-by-line, treat each line as a
user prompt, emit a `messages_v1` row with a fixed assistant reply.
Optionally also pass-through every line from each `--corpus-dep` JSONL
(deps are already in messages_v1 form, so we re-emit verbatim).

This is *only* a smoke fixture; production transforms emit their own
structured supervision.
"""

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--corpus-dep", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []

    # Pass-through dep rows (they're already messages_v1).
    for dep in args.corpus_dep:
        with open(dep) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(line)

    # Emit one messages_v1 row per non-empty input line.
    for path in args.input:
        with open(path) as f:
            for raw in f:
                prompt = raw.strip()
                if not prompt:
                    continue
                row = {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "ok"},
                    ]
                }
                rows.append(json.dumps(row))

    with open(args.output, "w") as out:
        out.write("\n".join(rows) + ("\n" if rows else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
