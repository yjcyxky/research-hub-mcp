#!/usr/bin/env python3
"""
Table Extraction Evaluation Script

Evaluates text2table pipeline on relation/attribute extraction tasks.
Supports multiple evaluation modes:
1. Relation extraction (e.g., drug-disease, drug-ADE)
2. Attribute table extraction (e.g., drug-dose-duration-frequency)

Datasets:
- ADE Corpus V2 (Drug-ADE relations)
- Drug Combination Extraction (N-ary relations)
- Custom annotation format

Usage:
    # Evaluate on ADE corpus
    python benches/re/table_extraction_eval.py --dataset ade_corpus \
        --labels drug ade --server-url http://localhost:8000/v1

    # Evaluate with custom data
    python benches/re/table_extraction_eval.py --dataset custom \
        --data-file /path/to/data.jsonl --labels drug dose duration

    # Dump debug output
    python benches/re/table_extraction_eval.py --dataset ade_corpus \
        --dump-jsonl /tmp/re_debug.jsonl --dump-limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _append_jsonl(output_path: Path, rows: List[object]) -> None:
    """Append rows (dict or objects with to_dict) to a JSONL file."""
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for row in rows:
            payload = row.to_dict() if hasattr(row, "to_dict") else row
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")


def _relation_to_dict(rel: Relation) -> dict:
    return {"type": rel.relation_type, "entities": rel.entities}


def _build_dump_record(result: "BatchResult", sample: EvalSample, labels: List[str]) -> dict:
    """Combine prediction result with gold info for dumping."""
    pred_table = ExtractedTable.from_tsv(result.table, expected_headers=labels)
    pred_relations = table_to_relations(pred_table, "extracted")
    return {
        "index": result.index,
        "id": result.id,
        "text": result.text,
        "gold_table": sample.gold_table.to_dict(),
        "pred_table": pred_table.to_dict(),
        "gold_relations": [_relation_to_dict(r) for r in sample.gold_relations],
        "pred_relations": [_relation_to_dict(r) for r in pred_relations],
        "entities": result.entities,
        "thinking": result.thinking,
        "metadata": sample.metadata,
        "status": result.status,
        "error": result.error,
    }

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TableCell:
    """Represents a cell in the extracted table."""
    value: str
    column: str  # column header/label
    row_idx: int = 0

    def normalized(self) -> str:
        """Normalize cell value for comparison."""
        return self.value.lower().strip()

    def __hash__(self) -> int:
        return hash((self.normalized(), self.column.lower(), self.row_idx))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TableCell):
            return False
        return (self.normalized() == other.normalized() and
                self.column.lower() == other.column.lower() and
                self.row_idx == other.row_idx)


@dataclass
class TableRow:
    """Represents a row in the extracted table."""
    cells: Dict[str, str]  # column -> value mapping
    row_idx: int = 0

    def get(self, column: str, default: str = "") -> str:
        """Get cell value by column name (case-insensitive)."""
        column_lower = column.lower()
        for k, v in self.cells.items():
            if k.lower() == column_lower:
                return v
        return default

    def normalized_cells(self) -> Dict[str, str]:
        """Return normalized cell values."""
        return {
            k.lower(): v.lower().strip()
            for k, v in self.cells.items()
            if v
            and v.strip()
            and k.lower() != "confidence"
        }

    def __hash__(self) -> int:
        items = tuple(sorted(self.normalized_cells().items()))
        return hash(items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TableRow):
            return False
        return self.normalized_cells() == other.normalized_cells()


@dataclass
class ExtractedTable:
    """Represents an extracted table with headers and rows."""
    headers: List[str]
    rows: List[TableRow]
    raw_table: str = ""

    @classmethod
    def from_tsv(cls, tsv: str, expected_headers: Optional[List[str]] = None) -> "ExtractedTable":
        """Parse a TSV table into ExtractedTable."""
        lines = [line.strip() for line in tsv.strip().split("\n") if line.strip()]
        if not lines:
            return cls(headers=[], rows=[], raw_table=tsv)
        headers = [h.strip() for h in lines[0].split("\t")]
        rows: List[TableRow] = []
        for idx, line in enumerate(lines[1:]):
            cells = [c.strip() for c in line.split("\t")]
            if len(cells) >= len(headers):
                cell_dict = {headers[i]: cells[i] if i < len(cells) else "" for i in range(len(headers))}
                rows.append(TableRow(cells=cell_dict, row_idx=idx))
        return cls(headers=headers, rows=rows, raw_table=tsv)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "headers": self.headers,
            "rows": [{"cells": row.cells, "row_idx": row.row_idx} for row in self.rows],
            "raw_table": self.raw_table,
        }


@dataclass
class Relation:
    """Represents a relation/tuple extracted from text."""
    relation_type: str  # e.g., "drug-ade", "drug-dose"
    entities: Dict[str, str]  # role -> entity text mapping
    source_text: str = ""

    def key(self) -> Tuple[str, ...]:
        """Generate a comparable key for this relation."""
        items = sorted((k.lower(), v.lower().strip()) for k, v in self.entities.items() if v.strip())
        return (self.relation_type.lower(),) + tuple(items)

    def __hash__(self) -> int:
        return hash(self.key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Relation):
            return False
        return self.key() == other.key()


@dataclass
class EvalSample:
    """A single evaluation sample."""
    text: str
    gold_table: ExtractedTable
    pred_table: Optional[ExtractedTable] = None
    gold_relations: List[Relation] = field(default_factory=list)
    pred_relations: List[Relation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetMetrics:
    """Metrics for evaluation."""
    name: str
    # Table-level metrics
    table_precision: float = 0.0
    table_recall: float = 0.0
    table_f1: float = 0.0
    # Cell-level metrics
    cell_precision: float = 0.0
    cell_recall: float = 0.0
    cell_f1: float = 0.0
    cell_tp: int = 0
    cell_fp: int = 0
    cell_fn: int = 0
    # Row-level metrics (exact match)
    row_precision: float = 0.0
    row_recall: float = 0.0
    row_f1: float = 0.0
    row_tp: int = 0
    row_fp: int = 0
    row_fn: int = 0
    # Relation-level metrics
    rel_precision: float = 0.0
    rel_recall: float = 0.0
    rel_f1: float = 0.0
    rel_tp: int = 0
    rel_fp: int = 0
    rel_fn: int = 0
    # Counts
    total_gold_rows: int = 0
    total_pred_rows: int = 0
    total_gold_cells: int = 0
    total_pred_cells: int = 0
    examples: int = 0


def table_to_relations(table: ExtractedTable, relation_type: str) -> List[Relation]:
    """Convert table rows to relations."""
    relations = []
    for row in table.rows:
        if row.cells:
            entities = {k: v for k, v in row.cells.items() if k.lower() != "confidence"}
            relations.append(Relation(
                relation_type=relation_type,
                entities=entities,
            ))
    return relations


def compute_cell_metrics(
    gold_table: ExtractedTable,
    pred_table: ExtractedTable,
    target_columns: Optional[List[str]] = None,
) -> Tuple[int, int, int]:
    """Compute TP, FP, FN at cell level."""
    def extract_cells(table: ExtractedTable, columns: Optional[List[str]]) -> set:
        cells = set()
        for row in table.rows:
            for col, val in row.cells.items():
                if val and val.strip() and val.strip().lower() not in ("n/a", "-", ""):
                    if columns is None or col.lower() in [c.lower() for c in columns]:
                        cells.add((col.lower(), val.lower().strip()))
        return cells

    gold_cells = extract_cells(gold_table, target_columns)
    pred_cells = extract_cells(pred_table, target_columns)

    tp = len(gold_cells & pred_cells)
    fp = len(pred_cells - gold_cells)
    fn = len(gold_cells - pred_cells)

    return tp, fp, fn


def compute_row_metrics(
    gold_table: ExtractedTable,
    pred_table: ExtractedTable,
    key_columns: Optional[List[str]] = None,
) -> Tuple[int, int, int]:
    """Compute TP, FP, FN at row level (exact match)."""
    def row_key(row: TableRow, columns: Optional[List[str]]) -> tuple:
        if columns:
            items = [(c.lower(), row.get(c, "").lower().strip())
                     for c in columns if row.get(c, "").strip()]
        else:
            items = [(k.lower(), v.lower().strip())
                     for k, v in row.cells.items() if v.strip()]
        return tuple(sorted(items))

    gold_keys = {row_key(row, key_columns) for row in gold_table.rows if row_key(row, key_columns)}
    pred_keys = {row_key(row, key_columns) for row in pred_table.rows if row_key(row, key_columns)}

    tp = len(gold_keys & pred_keys)
    fp = len(pred_keys - gold_keys)
    fn = len(gold_keys - pred_keys)

    return tp, fp, fn


def compute_relation_metrics(
    gold_relations: List[Relation],
    pred_relations: List[Relation],
) -> Tuple[int, int, int]:
    """Compute TP, FP, FN at relation level."""
    gold_set = set(gold_relations)
    pred_set = set(pred_relations)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    return tp, fp, fn


def calculate_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Calculate precision, recall, F1."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ============================================================================
# Dataset Loaders
# ============================================================================

def load_ade_corpus(
    split: str = "test",
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[EvalSample]:
    """Load ADE Corpus V2 from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install 'datasets': pip install datasets")
        sys.exit(1)

    logger.info("Loading ADE Corpus V2...")

    # Load the RE split
    dataset = load_dataset(
        "ade-benchmark-corpus/ade_corpus_v2",
        "Ade_corpus_v2_drug_ade_relation",
        split=split,
        cache_dir=cache_dir,
    )

    samples = []
    seen_texts = {}  # Group by text to build tables

    for example in dataset:
        text = example.get("text", "")
        drug = example.get("drug", "")
        effect = example.get("effect", "")

        if not text or not drug or not effect:
            continue

        if text not in seen_texts:
            seen_texts[text] = {"drugs": [], "effects": [], "relations": []}

        seen_texts[text]["drugs"].append(drug)
        seen_texts[text]["effects"].append(effect)
        seen_texts[text]["relations"].append(Relation(
            relation_type="drug-ade",
            entities={"drug": drug, "ade": effect},
            source_text=text,
        ))

    # Convert to samples with tables
    for text, data in seen_texts.items():
        headers = ["drug", "ade"]
        rows = []
        for i, (drug, effect) in enumerate(zip(data["drugs"], data["effects"])):
            rows.append(TableRow(cells={"drug": drug, "ade": effect}, row_idx=i))

        gold_table = ExtractedTable(headers=headers, rows=rows)
        samples.append(EvalSample(
            text=text,
            gold_table=gold_table,
            gold_relations=data["relations"],
        ))

        if max_examples and len(samples) >= max_examples:
            break

    logger.info(f"Loaded {len(samples)} samples with {sum(len(s.gold_table.rows) for s in samples)} relations")
    return samples


def load_drug_combo_dataset(
    split: str = "test",
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[EvalSample]:
    """Load Drug Combination Extraction dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install 'datasets': pip install datasets")
        sys.exit(1)

    logger.info("Loading Drug Combination Extraction dataset...")

    dataset = load_dataset(
        "allenai/drug-combo-extraction",
        split=split,
        cache_dir=cache_dir,
    )

    samples = []

    for example in dataset:
        text = example.get("sentence", example.get("text", ""))
        drugs = example.get("drug_entities", example.get("drugs", []))
        label = example.get("label", example.get("relation_label", ""))

        if not text:
            continue

        # Build table with drug combinations
        headers = ["drug1", "drug2", "combination_type"]
        rows = []

        if isinstance(drugs, list) and len(drugs) >= 2:
            for i in range(len(drugs) - 1):
                for j in range(i + 1, len(drugs)):
                    rows.append(TableRow(
                        cells={
                            "drug1": drugs[i] if isinstance(drugs[i], str) else str(drugs[i]),
                            "drug2": drugs[j] if isinstance(drugs[j], str) else str(drugs[j]),
                            "combination_type": label if label else "combination",
                        },
                        row_idx=len(rows),
                    ))

        gold_table = ExtractedTable(headers=headers, rows=rows)
        gold_relations = [
            Relation(
                relation_type="drug-combo",
                entities=row.cells,
                source_text=text,
            ) for row in rows
        ]

        samples.append(EvalSample(
            text=text,
            gold_table=gold_table,
            gold_relations=gold_relations,
        ))

        if max_examples and len(samples) >= max_examples:
            break

    logger.info(f"Loaded {len(samples)} samples")
    return samples


def load_custom_dataset(
    data_file: str,
    max_examples: Optional[int] = None,
) -> List[EvalSample]:
    """
    Load custom dataset from JSONL file.

    Expected format per line:
    {
        "text": "source text",
        "table": {
            "headers": ["col1", "col2"],
            "rows": [{"col1": "val1", "col2": "val2"}, ...]
        },
        // OR alternatively:
        "relations": [
            {"type": "drug-dose", "entities": {"drug": "aspirin", "dose": "100mg"}}
        ]
    }
    """
    logger.info(f"Loading custom dataset from: {data_file}")

    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse line: {line[:100]}")
                continue

            text = data.get("text", "")
            if not text:
                continue

            # Parse table if present
            gold_table = ExtractedTable(headers=[], rows=[])
            if "table" in data:
                table_data = data["table"]
                headers = table_data.get("headers", [])
                rows = []
                for idx, row_data in enumerate(table_data.get("rows", [])):
                    if isinstance(row_data, dict):
                        rows.append(TableRow(cells=row_data, row_idx=idx))
                    elif isinstance(row_data, list):
                        cells = {headers[i]: row_data[i] for i in range(min(len(headers), len(row_data)))}
                        rows.append(TableRow(cells=cells, row_idx=idx))
                gold_table = ExtractedTable(headers=headers, rows=rows)

            # Parse relations if present
            gold_relations = []
            if "relations" in data:
                for rel_data in data["relations"]:
                    gold_relations.append(Relation(
                        relation_type=rel_data.get("type", "unknown"),
                        entities=rel_data.get("entities", {}),
                        source_text=text,
                    ))

            samples.append(EvalSample(
                text=text,
                gold_table=gold_table,
                gold_relations=gold_relations,
                metadata=data.get("metadata", {}),
            ))

            if max_examples and len(samples) >= max_examples:
                break

    logger.info(f"Loaded {len(samples)} samples from custom file")
    return samples


# ============================================================================
# Text2Table Integration
# ============================================================================

def run_text2table_predictions(
    samples: List[EvalSample],
    labels: List[str],
    server_url: str,
    gliner_url: Optional[str] = None,
    gliner_model: str = "Ihor/gliner-biomed-large-v1.0",
    model: Optional[str] = None,
    threshold: float = 0.5,
    gliner_soft_threshold: Optional[float] = None,
    enable_row_validation: bool = False,
    row_validation_mode: str = "substring",
    disable_gliner: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    show_progress: bool = True,
    concurrency: int = 1,
    dump_jsonl: Optional[Path] = None,
    flush_every: int = 20,
) -> List[EvalSample]:
    """Run text2table pipeline on samples."""
    import asyncio

    from python.rust_research_py.text2table import AsyncText2Table, BatchItem, BatchResult, DEFAULT_USER_PROMPT

    logger.info(f"Running text2table with labels: {labels}")
    logger.info(f"Server URL: {server_url}")
    logger.info("Concurrency: %d", concurrency)

    extractor = AsyncText2Table(
        labels=labels,
        gliner_model_name=gliner_model,
        model_name=model,
        threshold=threshold,
        gliner_soft_threshold=gliner_soft_threshold,
        server_url=server_url,
        gliner_url=gliner_url,
        use_gliner=not disable_gliner,
        enable_row_validation=enable_row_validation,
        row_validation_mode=row_validation_mode,
    )

    batch_items = [BatchItem(text=sample.text, index=idx) for idx, sample in enumerate(samples)]

    progress = None
    buffer: List[BatchResult] = []
    if dump_jsonl:
        dump_jsonl.parent.mkdir(parents=True, exist_ok=True)
        dump_jsonl.write_text("", encoding="utf-8")
    if flush_every <= 0:
        flush_every = 20
    if show_progress:
        try:
            from tqdm import tqdm

            progress = tqdm(total=len(batch_items), desc="Extracting tables")
        except ImportError:
            progress = None

    async def handle_result(res: BatchResult) -> None:
        if progress:
            progress.update(1)
        if dump_jsonl:
            buffer.append(res)
            if len(buffer) >= flush_every:
                # Enrich with gold/pred relations before writing
                enriched = [
                    _build_dump_record(r, samples[r.index], labels) for r in buffer
                ]
                _append_jsonl(dump_jsonl, enriched)
                buffer.clear()

    async def _run_all() -> List[BatchResult]:
        try:
            results = await extractor.run_many(
                batch_items,
                user_prompt=DEFAULT_USER_PROMPT,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                concurrency=max(1, concurrency),
                on_result=handle_result if progress else None,
            )
            return results
        finally:
            await extractor.close()
            if progress:
                progress.close()
            if dump_jsonl and buffer:
                enriched = [
                    _build_dump_record(r, samples[r.index], labels) for r in buffer
                ]
                _append_jsonl(dump_jsonl, enriched)

    results = asyncio.run(_run_all())

    for result in results:
        idx = result.index
        sample = samples[idx]
        if result.status == "ok":
            sample.pred_table = ExtractedTable.from_tsv(result.table, expected_headers=labels)
            sample.pred_relations = table_to_relations(sample.pred_table, "extracted")
        else:
            logger.warning("Extraction failed for index %d: %s", idx, result.error)
            sample.pred_table = ExtractedTable(headers=labels, rows=[])
            sample.pred_relations = []

    return samples


# ============================================================================
# Evaluation
# ============================================================================

def calculate_metrics(
    samples: List[EvalSample],
    name: str,
    key_columns: Optional[List[str]] = None,
) -> DatasetMetrics:
    """Calculate all metrics for samples."""
    metrics = DatasetMetrics(name=name, examples=len(samples))

    total_cell_tp, total_cell_fp, total_cell_fn = 0, 0, 0
    total_row_tp, total_row_fp, total_row_fn = 0, 0, 0
    total_rel_tp, total_rel_fp, total_rel_fn = 0, 0, 0

    for sample in samples:
        gold_table = sample.gold_table
        pred_table = sample.pred_table or ExtractedTable(headers=[], rows=[])

        # Cell-level
        cell_tp, cell_fp, cell_fn = compute_cell_metrics(gold_table, pred_table, key_columns)
        total_cell_tp += cell_tp
        total_cell_fp += cell_fp
        total_cell_fn += cell_fn

        # Row-level
        row_tp, row_fp, row_fn = compute_row_metrics(gold_table, pred_table, key_columns)
        total_row_tp += row_tp
        total_row_fp += row_fp
        total_row_fn += row_fn

        # Relation-level
        rel_tp, rel_fp, rel_fn = compute_relation_metrics(
            sample.gold_relations, sample.pred_relations
        )
        total_rel_tp += rel_tp
        total_rel_fp += rel_fp
        total_rel_fn += rel_fn

        # Counts
        metrics.total_gold_rows += len(gold_table.rows)
        metrics.total_pred_rows += len(pred_table.rows)
        metrics.total_gold_cells += sum(
            len([v for v in row.cells.values() if v.strip()]) for row in gold_table.rows
        )
        metrics.total_pred_cells += sum(
            len([v for v in row.cells.values() if v.strip()]) for row in pred_table.rows
        )

    # Calculate PRF scores
    metrics.cell_tp, metrics.cell_fp, metrics.cell_fn = total_cell_tp, total_cell_fp, total_cell_fn
    metrics.cell_precision, metrics.cell_recall, metrics.cell_f1 = calculate_prf(
        total_cell_tp, total_cell_fp, total_cell_fn
    )

    metrics.row_tp, metrics.row_fp, metrics.row_fn = total_row_tp, total_row_fp, total_row_fn
    metrics.row_precision, metrics.row_recall, metrics.row_f1 = calculate_prf(
        total_row_tp, total_row_fp, total_row_fn
    )

    metrics.rel_tp, metrics.rel_fp, metrics.rel_fn = total_rel_tp, total_rel_fp, total_rel_fn
    metrics.rel_precision, metrics.rel_recall, metrics.rel_f1 = calculate_prf(
        total_rel_tp, total_rel_fp, total_rel_fn
    )

    return metrics


def dump_samples_jsonl(
    samples: List[EvalSample],
    output_path: str,
    limit: Optional[int] = None,
) -> None:
    """Dump samples to JSONL for visualization."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            if limit is not None and i >= limit:
                break

            record = {
                "text": sample.text,
                "gold_table": sample.gold_table.to_dict() if sample.gold_table else None,
                "pred_table": sample.pred_table.to_dict() if sample.pred_table else None,
                "gold_relations": [
                    {"type": r.relation_type, "entities": r.entities}
                    for r in sample.gold_relations
                ],
                "pred_relations": [
                    {"type": r.relation_type, "entities": r.entities}
                    for r in sample.pred_relations
                ],
                "metadata": sample.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Dumped {min(len(samples), limit or len(samples))} samples to {output_path}")


def metrics_to_dict(metrics: DatasetMetrics) -> Dict[str, Any]:
    """Convert metrics to dictionary."""
    return {
        "name": metrics.name,
        "examples": metrics.examples,
        "cell_level": {
            "precision": round(metrics.cell_precision, 4),
            "recall": round(metrics.cell_recall, 4),
            "f1": round(metrics.cell_f1, 4),
            "tp": metrics.cell_tp,
            "fp": metrics.cell_fp,
            "fn": metrics.cell_fn,
        },
        "row_level": {
            "precision": round(metrics.row_precision, 4),
            "recall": round(metrics.row_recall, 4),
            "f1": round(metrics.row_f1, 4),
            "tp": metrics.row_tp,
            "fp": metrics.row_fp,
            "fn": metrics.row_fn,
        },
        "relation_level": {
            "precision": round(metrics.rel_precision, 4),
            "recall": round(metrics.rel_recall, 4),
            "f1": round(metrics.rel_f1, 4),
            "tp": metrics.rel_tp,
            "fp": metrics.rel_fp,
            "fn": metrics.rel_fn,
        },
        "counts": {
            "gold_rows": metrics.total_gold_rows,
            "pred_rows": metrics.total_pred_rows,
            "gold_cells": metrics.total_gold_cells,
            "pred_cells": metrics.total_pred_cells,
        },
    }


# ============================================================================
# Dataset configurations
# ============================================================================

DATASET_CONFIGS = {
    "ade_corpus": {
        "loader": load_ade_corpus,
        "description": "ADE Corpus V2 - Drug-ADE relation extraction",
        "default_labels": ["drug", "ade"],
        "relation_type": "drug-ade",
    },
    "drug_combo": {
        "loader": load_drug_combo_dataset,
        "description": "Drug Combination Extraction - N-ary drug relations",
        "default_labels": ["drug1", "drug2", "combination_type"],
        "relation_type": "drug-combo",
    },
    "custom": {
        "loader": load_custom_dataset,
        "description": "Custom JSONL dataset",
        "default_labels": None,
        "relation_type": "custom",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate text2table pipeline on relation/table extraction tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate on ADE corpus
    python table_extraction_eval.py --dataset ade_corpus \\
        --server-url http://localhost:8000/v1

    # Evaluate with custom labels
    python table_extraction_eval.py --dataset ade_corpus \\
        --labels drug adverse_effect --server-url http://localhost:8000/v1

    # Use custom dataset
    python table_extraction_eval.py --dataset custom \\
        --data-file /path/to/data.jsonl --labels drug dose duration frequency

    # Dump debug output
    python table_extraction_eval.py --dataset ade_corpus \\
        --dump-jsonl /tmp/re_debug.jsonl --dump-limit 20
        """,
    )

    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        default="ade_corpus",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        help="Path to custom JSONL data file (for custom dataset)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Maximum examples to evaluate",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        help="HuggingFace datasets cache directory",
    )

    # Label options
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Labels/columns to extract (overrides dataset defaults)",
    )
    parser.add_argument(
        "--key-columns",
        nargs="+",
        help="Key columns for row matching (default: all columns)",
    )

    # text2table options
    parser.add_argument(
        "--server-url",
        type=str,
        required=False,
        help="vLLM server URL (required)",
        default="http://localhost:8000/v1",
    )
    parser.add_argument(
        "--gliner-url",
        type=str,
        help="GLiNER service URL (optional, falls back to local)",
    )
    parser.add_argument(
        "--gliner-model",
        type=str,
        default="Ihor/gliner-biomed-large-v1.0",
        help="GLiNER model name",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="vLLM model name (uses server default if not specified)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="GLiNER confidence threshold",
    )
    parser.add_argument(
        "--gliner-soft-threshold",
        type=float,
        default=None,
        help="Lower GLiNER threshold for recall-only candidates (marked low-confidence)",
    )
    parser.add_argument(
        "--disable-gliner",
        action="store_true",
        help="Skip GLiNER and use LLM only",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens for table generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature",
    )
    parser.add_argument(
        "--enable-row-validation",
        action="store_true",
        help="Drop rows not supported by the source text (substring check)",
    )
    parser.add_argument(
        "--row-validation-mode",
        type=str,
        default="substring",
        choices=["substring", "llm"],
        help="Row validation strategy: substring or llm",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent in-flight requests when extracting tables",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--dump-jsonl",
        type=str,
        help="Path to dump sample JSONL for debugging/visualization",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=20,
        help="When dumping JSONL, flush to disk every N samples",
    )
    parser.add_argument(
        "--dump-limit",
        type=int,
        help="Maximum samples to dump",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.dataset == "custom" and not args.data_file:
        parser.error("--data-file is required for custom dataset")

    # Get dataset config
    config = DATASET_CONFIGS[args.dataset]

    # Determine labels
    labels = args.labels or config.get("default_labels")
    if not labels:
        parser.error("--labels is required for this dataset")

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Labels: {labels}")

    # Load samples
    if args.dataset == "custom":
        samples = config["loader"](args.data_file, args.max_examples)
    else:
        samples = config["loader"](
            split=args.split,
            max_examples=args.max_examples,
            cache_dir=args.dataset_cache_dir,
        )

    if not samples:
        logger.error("No samples loaded!")
        sys.exit(1)

    # Run predictions
    samples = run_text2table_predictions(
        samples=samples,
        labels=labels,
        server_url=args.server_url,
        gliner_url=args.gliner_url,
        gliner_model=args.gliner_model,
        gliner_soft_threshold=args.gliner_soft_threshold,
        model=args.model,
        threshold=args.threshold,
        enable_row_validation=args.enable_row_validation,
        row_validation_mode=args.row_validation_mode,
        disable_gliner=args.disable_gliner,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        show_progress=not args.no_progress,
        concurrency=max(1, args.concurrency),
        dump_jsonl=Path(args.dump_jsonl) if args.dump_jsonl else None,
        flush_every=args.flush_every,
    )

    # Calculate metrics
    metrics = calculate_metrics(samples, args.dataset, args.key_columns)

    # Output results
    results = {
        "dataset": args.dataset,
        "labels": labels,
        "settings": {
            "server_url": args.server_url,
            "gliner_model": args.gliner_model,
            "model": args.model,
            "threshold": args.threshold,
            "disable_gliner": args.disable_gliner,
        },
        "metrics": metrics_to_dict(metrics),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(json.dumps(results, indent=2, ensure_ascii=False))

    # Dump samples if requested
    if args.dump_jsonl:
        dump_samples_jsonl(samples, args.dump_jsonl, args.dump_limit)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Examples: {metrics.examples}")
    logger.info(f"Cell-level F1: {metrics.cell_f1:.4f}")
    logger.info(f"Row-level F1:  {metrics.row_f1:.4f}")
    logger.info(f"Relation F1:   {metrics.rel_f1:.4f}")


if __name__ == "__main__":
    main()
