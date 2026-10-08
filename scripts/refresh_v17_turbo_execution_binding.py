#!/usr/bin/env python3
"""Bind this checkout after intentional deployment changes; never mutate frozen run plans."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "config/v17_turbo_heldout_handoff_binding.json"
binding = json.loads(path.read_text())
if binding.get("routing_revision") != "split_review_v2":
    raise SystemExit("Unexpected routing revision")
for relative in binding["bindings"]:
    source = (root / relative).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise SystemExit("Missing or unsafe binding source: " + relative)
    binding["bindings"][relative] = hashlib.sha256(source.read_bytes()).hexdigest()
binding["fingerprint"] = hashlib.sha256(json.dumps(binding["bindings"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
path.write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
print(binding["fingerprint"])
