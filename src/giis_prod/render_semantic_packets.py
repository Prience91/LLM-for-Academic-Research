from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .io import read_jsonl

_BLOCK = re.compile(r"(?ms)^\[E(\d+)\]\s*\n(.*?)(?=^\[E\d+\]\s*$|\Z)")
_MARKER = re.compile(r"\[E(\d+)\]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queue", type=Path)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    rows = list(read_jsonl(args.queue))
    for ordinal, row in enumerate(
        rows[args.start - 1 : args.start - 1 + args.count], start=args.start
    ):
        blocks = {number: text.strip() for number, text in _BLOCK.findall(row["evidence"])}
        cited = sorted(
            set(_MARKER.findall(row["response_a"]["answer"]))
            | set(_MARKER.findall(row["response_b"]["answer"])),
            key=int,
        )
        print(f"\n===== ITEM {ordinal} | {row['pair_id']} | {row['source_mode']} =====")
        print(f"DOCUMENTS: {', '.join(row['document_ids'])}")
        print("\n--- CITED EVIDENCE ---")
        for number in cited:
            print(f"\n[E{number}]\n{blocks.get(number, '[missing block]')}")
        for label in ("a", "b"):
            response = row[f"response_{label}"]
            print(f"\n--- RESPONSE {label.upper()} ---")
            print(f"QUESTION: {response['question']}")
            print(f"ANSWER:\n{response['answer']}")


if __name__ == "__main__":
    main()
