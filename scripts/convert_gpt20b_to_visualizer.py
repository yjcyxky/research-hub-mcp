"""
Convert GPT-oss outputs to visualizer-friendly JSONL.

Usage:
    python scripts/convert_gpt20b_to_visualizer.py \
        --input examples/text2table_ade_corpus_gpt_20b.jsonl \
        --output examples/round1/text2table_ade_corpus_gpt_20b_fixed.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _strip_think_blocks(table_text: str) -> str:
    """Remove <think>...</think> blocks and return trailing content."""
    if "</think>" in table_text:
        return table_text.split("</think>")[-1].strip()
    return table_text.strip()


def _parse_tsv(table_text: str) -> Tuple[List[str], List[List[str]]]:
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if not lines:
        return [], []
    headers = [h.strip() for h in lines[0].split("\t")]
    rows = [[c.strip() for c in line.split("\t")] for line in lines[1:]]
    return headers, rows


def _rows_to_relations(headers: List[str], rows: List[List[str]]) -> List[dict]:
    relations = []
    for row in rows:
        entities = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
        relations.append({"type": "extracted", "entities": entities})
    return relations


def convert_record(record: Dict) -> Dict:
    pred_table = record.get("pred_table") or {}
    raw_table = pred_table.get("raw_table") or pred_table.get("raw_markdown") or ""
    cleaned_table = _strip_think_blocks(raw_table)
    headers, rows = _parse_tsv(cleaned_table)

    if headers and rows:
        structured_rows = []
        for idx, row in enumerate(rows):
            cells = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
            structured_rows.append({"cells": cells, "row_idx": idx})
        record["pred_table"] = {
            "headers": headers,
            "rows": structured_rows,
            "raw_table": cleaned_table,
        }
        record["pred_relations"] = _rows_to_relations(headers, rows)
    else:
        # Leave as-is if we cannot parse
        record["pred_table"]["raw_table"] = cleaned_table
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GPT-oss outputs to visualizer-friendly JSONL.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to source JSONL (GPT-oss output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write converted JSONL.",
    )
    args = parser.parse_args()

    lines = args.input.read_text(encoding="utf-8").splitlines()
    output_lines: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        fixed = convert_record(record)
        output_lines.append(json.dumps(fixed, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(output_lines)} records to {args.output}")


if __name__ == "__main__":
    main()
