#!/usr/bin/env python3
"""
GLiNER Biomedical NER Evaluation Script

Evaluates GLiNER models on biomedical disease NER datasets:
- ncbi/ncbi_disease (NCBI Disease Corpus)
- tner/bc5cdr (BC5CDR with disease entities)

Uses datasets available in parquet format for compatibility with datasets>=4.0.

Usage:
    python benches/ner/biomed_gliner_eval.py --device mps --datasets bc5cdr_disease
    python benches/ner/biomed_gliner_eval.py --datasets ncbi_disease bc5cdr_disease \
        --dump-jsonl /tmp/ner_debug.jsonl --dump-limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Represents an entity span with start, end offsets and label."""

    start: int
    end: int
    label: str
    text: str = ""

    def __hash__(self) -> int:
        return hash((self.start, self.end, self.label))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Span):
            return False
        return self.start == other.start and self.end == other.end and self.label == other.label


@dataclass
class EvalSample:
    """A single evaluation sample with text and entity spans."""

    text: str
    gold_spans: list[Span] = field(default_factory=list)
    pred_spans: list[Span] = field(default_factory=list)


@dataclass
class DatasetMetrics:
    """Metrics for a single dataset."""

    name: str
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    predicted: int
    gold: int
    examples: int


@dataclass
class EvalResults:
    """Complete evaluation results."""

    per_dataset: list[DatasetMetrics]
    overall: DatasetMetrics


# Disease entity type labels (case-insensitive matching)
DISEASE_LABELS = {
    "disease",
    "disease or syndrome",
    "diseaseorsyndrome",
    "disorder",
    "disease_disorder",
    "specific disease",
    "specificdisease",
    "composite mention",
    "compositemention",
    "modifier",
}


def normalize_label(label: str) -> str | None:
    """
    Normalize entity type label.
    Returns 'Disease' for disease-related labels, None for others.
    """
    label_lower = label.lower().strip()
    if label_lower in DISEASE_LABELS:
        return "Disease"
    return None


def parse_offset(offset: Any) -> tuple[int, int] | None:
    """
    Parse various offset formats to (start, end) tuple.

    Supports:
    - list/tuple [start, end]
    - dict {"start": x, "end": y} or {"offset": x, "length": y}
    """
    if isinstance(offset, (list, tuple)) and len(offset) >= 2:
        return (int(offset[0]), int(offset[1]))
    if isinstance(offset, dict):
        if "start" in offset and "end" in offset:
            return (int(offset["start"]), int(offset["end"]))
        if "offset" in offset and "length" in offset:
            start = int(offset["offset"])
            return (start, start + int(offset["length"]))
    return None


def extract_text_from_passages(passages: list[dict[str, Any]]) -> str:
    """
    Extract full text from BigBio passages structure.

    Each passage has "text" field which can be a list or string.
    We concatenate all passage texts with spaces.
    """
    texts = []
    for passage in passages:
        text = passage.get("text", "")
        if isinstance(text, list):
            texts.extend(text)
        else:
            texts.append(str(text))
    return " ".join(texts)


def extract_entities_from_bigbio_kb(
    example: dict[str, Any], target_label: str = "Disease"
) -> tuple[str, list[Span]]:
    """
    Extract text and disease entity spans from BigBio KB schema.

    BigBio KB schema has:
    - passages: list of {"text": [...], "offsets": [...], ...}
    - entities: list of {"text": [...], "offsets": [[start, end], ...], "type": ...}
    """
    # Extract full text from passages
    passages = example.get("passages", [])
    text = extract_text_from_passages(passages)

    # Extract entities
    spans = []
    entities = example.get("entities", [])

    for entity in entities:
        entity_type = entity.get("type", "")
        normalized = normalize_label(entity_type)

        if normalized != target_label:
            continue

        offsets = entity.get("offsets", [])
        texts = entity.get("text", [])

        # Each entity can have multiple discontinuous spans
        for i, offset in enumerate(offsets):
            parsed = parse_offset(offset)
            if parsed is None:
                continue

            start, end = parsed
            entity_text = texts[i] if i < len(texts) else ""

            spans.append(
                Span(
                    start=start,
                    end=end,
                    label=target_label,
                    text=entity_text,
                )
            )

    return text, spans


def extract_entities_from_source(
    example: dict[str, Any], target_label: str = "Disease"
) -> tuple[str, list[Span]]:
    """
    Extract text and disease entity spans from source schema.

    Source schema may have:
    - text field directly
    - tokens with ner_tags (BIO format)
    - mentions list
    """
    text = ""
    spans = []

    # Try direct text field
    if "text" in example:
        text = example["text"]
        if isinstance(text, list):
            text = " ".join(text)

    # Try passages field (some source schemas have this too)
    elif "passages" in example:
        text = extract_text_from_passages(example["passages"])

    # Try title + abstract combination
    elif "title" in example and "abstract" in example:
        title = example["title"]
        abstract = example["abstract"]
        if isinstance(title, list):
            title = " ".join(title)
        if isinstance(abstract, list):
            abstract = " ".join(abstract)
        text = f"{title} {abstract}".strip()

    # Try entities field (similar to KB but at top level)
    if "entities" in example:
        entities = example["entities"]
        for entity in entities:
            entity_type = entity.get("type", "")
            normalized = normalize_label(entity_type)
            if normalized != target_label:
                continue

            offsets = entity.get("offsets", [])
            texts = entity.get("text", [])

            for i, offset in enumerate(offsets):
                parsed = parse_offset(offset)
                if parsed is None:
                    continue
                start, end = parsed
                entity_text = texts[i] if i < len(texts) else ""
                spans.append(Span(start=start, end=end, label=target_label, text=entity_text))

    # Try mentions field
    elif "mentions" in example:
        mentions = example["mentions"]
        for mention in mentions:
            mention_type = mention.get("type", mention.get("label", ""))
            normalized = normalize_label(mention_type)
            if normalized != target_label:
                continue

            # Various offset formats
            if "start" in mention and "end" in mention:
                start, end = int(mention["start"]), int(mention["end"])
            elif "offset" in mention:
                offset = mention["offset"]
                parsed = parse_offset(offset)
                if parsed is None:
                    continue
                start, end = parsed
            elif "offsets" in mention:
                offsets = mention["offsets"]
                if offsets:
                    parsed = parse_offset(offsets[0])
                    if parsed is None:
                        continue
                    start, end = parsed
                else:
                    continue
            else:
                continue

            mention_text = mention.get("text", "")
            if isinstance(mention_text, list):
                mention_text = mention_text[0] if mention_text else ""

            spans.append(Span(start=start, end=end, label=target_label, text=mention_text))

    return text, spans


def bio_to_spans(
    tokens: list[str],
    tags: list[str | int],
    target_label: str = "Disease",
    tag_mapping: dict[int, str] | None = None,
) -> tuple[str, list[Span]]:
    """
    Convert BIO-tagged tokens to text and spans.

    Args:
        tokens: List of tokens
        tags: List of BIO tags (string or int)
        target_label: Target label to filter for
        tag_mapping: Optional mapping from int to string tags
    """
    # Reconstruct text with simple spacing
    text = " ".join(tokens)

    # Convert numeric tags to string if needed
    if tag_mapping and tags and isinstance(tags[0], int):
        tags = [tag_mapping.get(t, "O") for t in tags]

    spans = []
    current_start = 0
    entity_start = None
    entity_label = None
    prev_end = 0

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        token_start = current_start
        token_end = current_start + len(token)

        # Handle string tags
        tag_str = str(tag) if not isinstance(tag, str) else tag

        if tag_str.startswith("B-"):
            # Start new entity
            if entity_start is not None:
                # Close previous entity
                normalized = normalize_label(entity_label) if entity_label else None
                if normalized == target_label:
                    spans.append(
                        Span(
                            start=entity_start,
                            end=prev_end,
                            label=target_label,
                            text=text[entity_start:prev_end],
                        )
                    )
            entity_start = token_start
            entity_label = tag_str[2:]

        elif tag_str.startswith("I-"):
            # Continue entity (if matching)
            if entity_start is None or (entity_label and tag_str[2:] != entity_label):
                # Orphan I- tag, treat as B-
                if entity_start is not None:
                    normalized = normalize_label(entity_label) if entity_label else None
                    if normalized == target_label:
                        spans.append(
                            Span(
                                start=entity_start,
                                end=prev_end,
                                label=target_label,
                                text=text[entity_start:prev_end],
                            )
                        )
                entity_start = token_start
                entity_label = tag_str[2:]
        else:
            # O tag - close any open entity
            if entity_start is not None:
                normalized = normalize_label(entity_label) if entity_label else None
                if normalized == target_label:
                    spans.append(
                        Span(
                            start=entity_start,
                            end=prev_end,
                            label=target_label,
                            text=text[entity_start:prev_end],
                        )
                    )
                entity_start = None
                entity_label = None

        prev_end = token_end
        current_start = token_end + 1  # +1 for space

    # Close final entity
    if entity_start is not None:
        normalized = normalize_label(entity_label) if entity_label else None
        if normalized == target_label:
            spans.append(
                Span(
                    start=entity_start,
                    end=prev_end,
                    label=target_label,
                    text=text[entity_start:prev_end],
                )
            )

    return text, spans


def parse_example(
    example: dict[str, Any],
    schema: str,
    target_label: str = "Disease",
    tag_mapping: dict[int, str] | None = None,
) -> tuple[str, list[Span]]:
    """
    Parse a dataset example to extract text and entity spans.

    Args:
        example: Dataset example dict
        schema: "bigbio_kb", "source", or "tner"
        target_label: Label to use for normalized disease entities
        tag_mapping: Optional mapping from int to string tags (for tner format)

    Returns:
        Tuple of (text, list of Span)
    """
    # TNER format: tokens + tags (numeric)
    if schema == "tner" and "tokens" in example and "tags" in example:
        tokens = example["tokens"]
        tags = example["tags"]
        return bio_to_spans(tokens, tags, target_label, tag_mapping)

    # Check for BIO-tagged format (tokens + ner_tags)
    if "tokens" in example and "ner_tags" in example:
        tokens = example["tokens"]
        tags = example["ner_tags"]
        return bio_to_spans(tokens, tags, target_label, tag_mapping)

    # BigBio KB schema
    if schema == "bigbio_kb" or "passages" in example:
        return extract_entities_from_bigbio_kb(example, target_label)

    # Source schema
    return extract_entities_from_source(example, target_label)


def pad_text_if_needed(text: str, spans: list[Span]) -> str:
    """
    Ensure text is long enough to contain all spans.
    """
    if not spans:
        return text

    max_end = max(span.end for span in spans)
    if len(text) < max_end:
        text = text + " " * (max_end - len(text))
    return text


def load_tner_dataset_from_json(
    repo_id: str,
    split: str = "test",
    cache_dir: str | None = None,
) -> Any:
    """
    Load a TNER dataset from JSON files in HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "tner/bc5cdr")
        split: Dataset split ("train", "valid", "test")
        cache_dir: Cache directory

    Returns:
        Dataset object
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install 'datasets' package: pip install datasets")
        sys.exit(1)

    # Map split names to file names
    split_file_map = {
        "train": "train.json",
        "validation": "valid.json",
        "valid": "valid.json",
        "test": "test.json",
    }

    file_name = split_file_map.get(split, f"{split}.json")
    data_url = f"hf://datasets/{repo_id}/dataset/{file_name}"

    logger.info(f"Loading from: {data_url}")

    dataset = load_dataset(
        "json",
        data_files=data_url,
        split="train",  # JSON loader always uses "train" split
        cache_dir=cache_dir,
    )

    return dataset


def parse_conll_data(content: str) -> list[dict[str, Any]]:
    """
    Parse CoNLL format data into tokens and tags.

    Args:
        content: Raw CoNLL format text (tab-separated, blank lines between sentences)

    Returns:
        List of examples with tokens and tags
    """
    examples = []
    current_tokens: list[str] = []
    current_tags: list[str] = []

    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            # End of sentence
            if current_tokens:
                examples.append({
                    "tokens": current_tokens,
                    "tags": current_tags,
                })
                current_tokens = []
                current_tags = []
        else:
            parts = line.split("\t")
            if len(parts) >= 2:
                current_tokens.append(parts[0])
                current_tags.append(parts[1])

    # Add last sentence if any
    if current_tokens:
        examples.append({
            "tokens": current_tokens,
            "tags": current_tags,
        })

    return examples


def load_ncbi_disease_from_source(
    split: str = "test",
    cache_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Load NCBI Disease dataset from the original source.

    First tries HuggingFace, then falls back to downloading CoNLL files from GitHub.

    Args:
        split: Dataset split
        cache_dir: Cache directory

    Returns:
        List of examples with tokens and tags
    """
    import urllib.request
    from pathlib import Path

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install 'datasets' package: pip install datasets")
        sys.exit(1)

    # Try loading from ncbi/ncbi_disease with trust_remote_code
    # This will work with older datasets versions or if the script is trusted
    try:
        logger.info("Attempting to load ncbi/ncbi_disease with trust_remote_code...")
        dataset = load_dataset(
            "ncbi/ncbi_disease",
            split=split,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        # Convert to list of dicts with string tags
        examples = []
        tag_mapping = {0: "O", 1: "B-Disease", 2: "I-Disease"}
        for example in dataset:
            examples.append({
                "tokens": example["tokens"],
                "tags": [tag_mapping.get(t, "O") for t in example["ner_tags"]],
            })
        return examples
    except Exception as e:
        logger.warning(f"Could not load ncbi/ncbi_disease from HuggingFace: {e}")

    # Fallback: Download from GitHub
    logger.info("Attempting to download NCBI Disease from GitHub (CoNLL format)...")

    # Map split names
    split_map = {
        "train": "train",
        "validation": "devel",
        "valid": "devel",
        "test": "test",
    }
    file_name = split_map.get(split, split)
    url = f"https://raw.githubusercontent.com/spyysalo/ncbi-disease/master/conll/{file_name}.tsv"

    try:
        # Check cache first
        if cache_dir:
            cache_path = Path(cache_dir) / "ncbi_disease" / f"{file_name}.tsv"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if cache_path.exists():
                logger.info(f"Loading from cache: {cache_path}")
                with open(cache_path, "r", encoding="utf-8") as f:
                    content = f.read()
            else:
                logger.info(f"Downloading from: {url}")
                req = urllib.request.Request(url)
                req.add_header("User-Agent", "Mozilla/5.0")
                with urllib.request.urlopen(req, timeout=30) as response:
                    content = response.read().decode("utf-8")
                # Save to cache
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(content)
        else:
            logger.info(f"Downloading from: {url}")
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0")
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read().decode("utf-8")

        examples = parse_conll_data(content)
        logger.info(f"Loaded {len(examples)} examples from CoNLL file")
        return examples

    except Exception as e:
        logger.error(f"Failed to download NCBI Disease from GitHub: {e}")

    # Final fallback
    logger.error(
        "Could not load ncbi_disease dataset. "
        "Please check your internet connection or try: pip install 'datasets<4.0'"
    )
    return []


def load_dataset_samples(
    dataset_name: str,
    config_name: str | None = None,
    split: str = "test",
    trust_remote_code: bool = False,
    cache_dir: str | None = None,
    max_examples: int | None = None,
    target_label: str = "Disease",
    schema: str = "auto",
    tag_mapping: dict[int, str] | None = None,
    loader_type: str = "auto",
) -> list[EvalSample]:
    """
    Load and parse a dataset into EvalSample objects.

    Args:
        dataset_name: HuggingFace dataset name
        config_name: Dataset configuration name (optional)
        split: Dataset split
        trust_remote_code: Whether to trust remote code
        cache_dir: Cache directory for datasets
        max_examples: Maximum number of examples to load
        target_label: Target entity label
        schema: Schema type ("auto", "bigbio_kb", "source", "tner")
        tag_mapping: Mapping from int to string tags for TNER format
        loader_type: How to load ("auto", "json", "standard", "ncbi_source")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Please install 'datasets' package: pip install datasets")
        sys.exit(1)

    logger.info(f"Loading dataset: {dataset_name}" + (f" with config: {config_name}" if config_name else ""))

    dataset = None

    # Determine loader type
    if loader_type == "auto":
        if dataset_name.startswith("tner/"):
            loader_type = "json"
        elif dataset_name == "ncbi/ncbi_disease":
            loader_type = "ncbi_source"
        else:
            loader_type = "standard"

    try:
        if loader_type == "json":
            # Load from JSON files in HuggingFace Hub
            dataset = load_tner_dataset_from_json(dataset_name, split, cache_dir)
        elif loader_type == "ncbi_source":
            # Load NCBI disease from source
            examples = load_ncbi_disease_from_source(split, cache_dir)
            if not examples:
                logger.error("No data loaded for ncbi_disease")
                return []
            # Create a simple iterable
            dataset = examples
        else:
            # Standard loading
            kwargs = {
                "split": split,
                "cache_dir": cache_dir,
            }
            if config_name:
                kwargs["name"] = config_name
            if trust_remote_code:
                kwargs["trust_remote_code"] = True

            dataset = load_dataset(dataset_name, **kwargs)
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise

    # Auto-detect schema
    if schema == "auto":
        if config_name and "bigbio" in config_name:
            schema = "bigbio_kb"
        elif loader_type == "json" or loader_type == "ncbi_source":
            schema = "tner"
        elif hasattr(dataset, "features") and "tags" in dataset.features:
            schema = "tner"
        elif hasattr(dataset, "features") and "ner_tags" in dataset.features:
            schema = "source"
        else:
            schema = "source"

    logger.info(f"Using schema: {schema}")

    samples = []
    num_examples = len(dataset)
    if max_examples is not None:
        num_examples = min(num_examples, max_examples)

    for i in range(num_examples):
        example = dataset[i]
        text, spans = parse_example(example, schema, target_label, tag_mapping)

        if not text:
            continue

        # Ensure text is long enough for all spans
        text = pad_text_if_needed(text, spans)

        samples.append(EvalSample(text=text, gold_spans=spans, pred_spans=[]))

    logger.info(f"Loaded {len(samples)} samples with {sum(len(s.gold_spans) for s in samples)} gold entities")

    return samples


def run_gliner_predictions(
    samples: list[EvalSample],
    model_name: str,
    labels: list[str],
    threshold: float = 0.5,
    device: str = "cpu",
    model_cache_dir: str | None = None,
    show_progress: bool = True,
) -> list[EvalSample]:
    """
    Run GLiNER predictions on samples.
    """
    try:
        from gliner import GLiNER
    except ImportError:
        logger.error("Please install 'gliner' package: pip install gliner")
        sys.exit(1)

    logger.info(f"Loading GLiNER model: {model_name}")

    # Load model
    model = GLiNER.from_pretrained(
        model_name,
        cache_dir=model_cache_dir,
    )

    # Move to device
    if device == "cuda":
        import torch

        if torch.cuda.is_available():
            model = model.to("cuda")
        else:
            logger.warning("CUDA not available, falling back to CPU")
            device = "cpu"
    elif device == "mps":
        import torch

        if torch.backends.mps.is_available():
            model = model.to("mps")
        else:
            logger.warning("MPS not available, falling back to CPU")
            device = "cpu"

    logger.info(f"Running predictions on {len(samples)} samples with threshold={threshold}")

    # Progress bar
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(samples, desc="Predicting")
        except ImportError:
            iterator = samples
    else:
        iterator = samples

    for sample in iterator:
        try:
            predictions = model.predict_entities(
                sample.text,
                labels,
                threshold=threshold,
            )

            for pred in predictions:
                sample.pred_spans.append(
                    Span(
                        start=pred["start"],
                        end=pred["end"],
                        label=pred["label"],
                        text=pred.get("text", sample.text[pred["start"] : pred["end"]]),
                    )
                )
        except Exception as e:
            logger.warning(f"Prediction failed for sample: {e}")
            continue

    return samples


def compute_metrics(samples: list[EvalSample]) -> tuple[int, int, int]:
    """
    Compute TP, FP, FN using strict span+label matching.
    """
    tp = 0
    fp = 0
    fn = 0

    for sample in samples:
        gold_set = set(sample.gold_spans)
        pred_set = set(sample.pred_spans)

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    return tp, fp, fn


def calculate_dataset_metrics(samples: list[EvalSample], name: str) -> DatasetMetrics:
    """
    Calculate metrics for a dataset.
    """
    tp, fp, fn = compute_metrics(samples)

    predicted = tp + fp
    gold = tp + fn

    precision = tp / predicted if predicted > 0 else 0.0
    recall = tp / gold if gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return DatasetMetrics(
        name=name,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        predicted=predicted,
        gold=gold,
        examples=len(samples),
    )


def dump_samples_jsonl(samples: list[EvalSample], output_path: str, limit: int | None = None) -> None:
    """
    Dump samples to JSONL file for debugging.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            if limit is not None and i >= limit:
                break

            record = {
                "text": sample.text,
                "gold": [
                    {"start": s.start, "end": s.end, "label": s.label, "text": s.text} for s in sample.gold_spans
                ],
                "pred": [
                    {"start": s.start, "end": s.end, "label": s.label, "text": s.text} for s in sample.pred_spans
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Dumped {min(len(samples), limit or len(samples))} samples to {output_path}")


def metrics_to_dict(metrics: DatasetMetrics) -> dict[str, Any]:
    """Convert DatasetMetrics to dictionary."""
    return {
        "name": metrics.name,
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "f1": round(metrics.f1, 4),
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "predicted": metrics.predicted,
        "gold": metrics.gold,
        "examples": metrics.examples,
    }


# Tag mappings for different datasets
# NCBI Disease: 0=O, 1=B-Disease, 2=I-Disease
NCBI_DISEASE_TAG_MAPPING = {
    0: "O",
    1: "B-Disease",
    2: "I-Disease",
}

# TNER BC5CDR: 0=O, 1=B-Chemical, 2=B-Disease, 3=I-Disease, 4=I-Chemical
TNER_BC5CDR_TAG_MAPPING = {
    0: "O",
    1: "B-Chemical",
    2: "B-Disease",
    3: "I-Disease",
    4: "I-Chemical",
}

# Dataset configurations
DATASET_CONFIGS = {
    "ncbi_disease": {
        "dataset_name": "ncbi/ncbi_disease",
        "config_name": None,
        "description": "NCBI Disease Corpus - disease name recognition",
        "schema": "tner",  # Use tner schema since we convert to tokens/tags
        "tag_mapping": None,  # Tags are already strings from CoNLL
        "loader_type": "ncbi_source",
    },
    "bc5cdr_disease": {
        "dataset_name": "tner/bc5cdr",
        "config_name": None,
        "description": "BC5CDR Corpus - disease entities only (filtering chemicals)",
        "schema": "tner",
        "tag_mapping": TNER_BC5CDR_TAG_MAPPING,
        "loader_type": "json",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GLiNER models on biomedical disease NER datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate on bc5cdr with MPS device (uses default biomed model)
    python biomed_gliner_eval.py --device mps --datasets bc5cdr_disease

    # Evaluate on both datasets with debug output
    python biomed_gliner_eval.py --datasets ncbi_disease bc5cdr_disease \\
        --dump-jsonl /tmp/ner_debug.jsonl --dump-limit 20

    # Use different model with different threshold
    python biomed_gliner_eval.py --model-name urchade/gliner_medium-v2.1 --threshold 0.4 \\
        --datasets ncbi_disease

Recommended models for biomedical NER:
    - Ihor/gliner-biomed-large-v1.0 (default, best for disease NER)
    - urchade/gliner_medium-v2.1 (general purpose, less accurate)
        """,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Ihor/gliner-biomed-large-v1.0",
        help="GLiNER model ID from HuggingFace (default: Ihor/gliner-biomed-large-v1.0 for biomedical NER)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Prediction threshold (default: 0.5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default="cpu",
        help="Device for inference (default: cpu)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS.keys()),
        default=["ncbi_disease"],
        help="Datasets to evaluate (default: ncbi_disease)",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=str,
        default=None,
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default=None,
        help="Model cache directory",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum examples per dataset (default: all)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading datasets",
    )
    parser.add_argument(
        "--dump-jsonl",
        type=str,
        default=None,
        help="Path to dump sample JSONL for debugging",
    )
    parser.add_argument(
        "--dump-limit",
        type=int,
        default=None,
        help="Maximum samples to dump (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (default: test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: print to stdout)",
    )

    args = parser.parse_args()

    # Target label for disease entities
    target_label = "Disease"
    labels = [target_label]

    all_samples: list[EvalSample] = []
    per_dataset_metrics: list[DatasetMetrics] = []

    # Process each dataset
    for dataset_key in args.datasets:
        config = DATASET_CONFIGS[dataset_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {dataset_key}")
        logger.info(f"Description: {config['description']}")
        logger.info(f"{'='*60}")

        # Load samples
        samples = load_dataset_samples(
            dataset_name=config["dataset_name"],
            config_name=config.get("config_name"),
            split=args.split,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.dataset_cache_dir,
            max_examples=args.max_examples,
            target_label=target_label,
            schema=config.get("schema", "auto"),
            tag_mapping=config.get("tag_mapping"),
            loader_type=config.get("loader_type", "auto"),
        )

        # Check for empty gold
        total_gold = sum(len(s.gold_spans) for s in samples)
        if total_gold == 0:
            logger.error(f"WARNING: No gold entities found in {dataset_key}!")
            logger.error("This may indicate a parsing issue. Check entity types and offsets.")

        # Run predictions
        samples = run_gliner_predictions(
            samples=samples,
            model_name=args.model_name,
            labels=labels,
            threshold=args.threshold,
            device=args.device,
            model_cache_dir=args.model_cache_dir,
            show_progress=not args.no_progress,
        )

        # Calculate metrics
        metrics = calculate_dataset_metrics(samples, dataset_key)
        per_dataset_metrics.append(metrics)
        all_samples.extend(samples)

        logger.info(f"Results for {dataset_key}:")
        logger.info(f"  Precision: {metrics.precision:.4f}")
        logger.info(f"  Recall:    {metrics.recall:.4f}")
        logger.info(f"  F1:        {metrics.f1:.4f}")
        logger.info(f"  TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn}")
        logger.info(f"  Predicted: {metrics.predicted}, Gold: {metrics.gold}, Examples: {metrics.examples}")

    # Calculate overall metrics
    overall_metrics = calculate_dataset_metrics(all_samples, "overall")

    # Build results
    results = EvalResults(per_dataset=per_dataset_metrics, overall=overall_metrics)

    # Output JSON
    output_dict = {
        "model": args.model_name,
        "threshold": args.threshold,
        "device": args.device,
        "per_dataset": [metrics_to_dict(m) for m in results.per_dataset],
        "overall": metrics_to_dict(results.overall),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved to: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(json.dumps(output_dict, indent=2, ensure_ascii=False))

    # Dump samples if requested
    if args.dump_jsonl:
        dump_samples_jsonl(all_samples, args.dump_jsonl, args.dump_limit)

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("OVERALL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Threshold: {args.threshold}")
    logger.info(f"Precision: {overall_metrics.precision:.4f}")
    logger.info(f"Recall:    {overall_metrics.recall:.4f}")
    logger.info(f"F1:        {overall_metrics.f1:.4f}")
    logger.info(
        f"Total: TP={overall_metrics.tp}, FP={overall_metrics.fp}, FN={overall_metrics.fn}"
    )


if __name__ == "__main__":
    main()
