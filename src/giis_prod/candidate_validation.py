from __future__ import annotations

import re
from typing import Any

_MARKER = re.compile(r"\[E\d+\]")
_MERMAID_BLOCK = re.compile(r"```mermaid\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_NODE = re.compile(r'^\s*([A-Za-z][A-Za-z0-9_]*)\s*\[(?:"?)(.*?)(?:"?)\]\s*$')
_EDGE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*--(?:>|[^\n]*?-->)\s*"
    r"([A-Za-z][A-Za-z0-9_]*)\s*$"
)

ANSWER_STYLES = {
    "prose",
    "sectioned",
    "table",
    "comparison",
    "causal_chain",
    "concept_diagram",
}


def _validate_table(answer: str) -> tuple[bool, str]:
    rows = [line.strip() for line in answer.splitlines() if line.strip().startswith("|")]
    if len(rows) < 3:
        return False, "table_requires_header_separator_and_data"
    widths = [len([cell for cell in row.strip("|").split("|")]) for row in rows]
    if len(set(widths)) != 1 or widths[0] < 2:
        return False, "table_column_mismatch"
    separator = rows[1].replace("|", "").replace(":", "").replace("-", "").strip()
    if separator:
        return False, "table_missing_markdown_separator"
    for row in rows[2:]:
        if not _MARKER.search(row):
            return False, "table_row_missing_evidence_marker"
    return True, "ok"


def _validate_causal_chain(answer: str) -> tuple[bool, str]:
    steps = [
        line.strip()
        for line in answer.splitlines()
        if "->" in line or "→" in line or re.match(r"^\s*\d+[.)]\s+", line)
    ]
    if len(steps) < 2:
        return False, "causal_chain_requires_at_least_two_steps"
    if any(not _MARKER.search(step) for step in steps):
        return False, "causal_chain_step_missing_evidence_marker"
    return True, "ok"


def _validate_concept_diagram(answer: str) -> tuple[bool, str]:
    blocks = _MERMAID_BLOCK.findall(answer)
    if len(blocks) != 1:
        return False, "concept_diagram_requires_one_mermaid_block"
    body = blocks[0].strip()
    lowered = body.casefold()
    forbidden = ("click ", "http://", "https://", "javascript:", "%%", "<script", "<iframe")
    if any(value in lowered for value in forbidden):
        return False, "unsafe_mermaid_directive"
    lines = [line.strip().rstrip(";") for line in body.splitlines() if line.strip()]
    if not lines or not re.fullmatch(
        r"flowchart\s+(?:TD|TB|LR|RL)", lines[0], re.IGNORECASE
    ):
        return False, "unsupported_mermaid_header"
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    for line in lines[1:]:
        node = _NODE.fullmatch(line)
        if node:
            nodes[node.group(1)] = node.group(2)
            continue
        edge = _EDGE.fullmatch(line)
        if edge:
            edges.append((edge.group(1), edge.group(2)))
            continue
        return False, "unsupported_mermaid_statement"
    if not 2 <= len(nodes) <= 20 or not edges:
        return False, "concept_diagram_size_or_edge_failure"
    if any(source not in nodes or target not in nodes for source, target in edges):
        return False, "concept_diagram_undefined_node"
    if any(not _MARKER.search(label) for label in nodes.values()):
        return False, "concept_diagram_node_missing_evidence_marker"
    return True, "ok"


def validate_answer_style(example: dict[str, Any]) -> tuple[bool, str]:
    style = str(example.get("answer_style") or "prose")
    answer = str(example.get("answer") or "")
    if style not in ANSWER_STYLES:
        return False, "invalid_answer_style"
    if style == "table":
        return _validate_table(answer)
    if style == "comparison":
        if "|" in answer:
            return _validate_table(answer)
        if len(set(_MARKER.findall(answer))) < 2:
            return False, "comparison_requires_multiple_evidence_markers"
    if style == "causal_chain":
        return _validate_causal_chain(answer)
    if style == "concept_diagram":
        return _validate_concept_diagram(answer)
    if style == "sectioned" and len(re.findall(r"^#{2,4}\s+\S", answer, re.MULTILINE)) < 2:
        return False, "sectioned_answer_requires_multiple_headings"
    return True, "ok"
