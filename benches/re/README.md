# Table/Relation Extraction Evaluation

This module provides evaluation scripts for table extraction and relation extraction tasks using the `text2table` pipeline.

## Overview

The evaluation supports two main task types:

1. **Relation Extraction (RE)**: Extract relationships between entities (e.g., drug-ADE, drug-disease)
2. **Attribute Table Extraction**: Extract entity attributes into structured tables (e.g., drug-dose-duration-frequency)

## Datasets

### Built-in Datasets

| Dataset | Description | Labels | Download |
|---------|-------------|--------|----------|
| `ade_corpus` | ADE Corpus V2 - Drug-ADE relations | drug, ade | Auto (HuggingFace) |
| `drug_combo` | Drug Combination Extraction | drug1, drug2, combination_type | Auto (HuggingFace) |
| `custom` | Custom JSONL format | User-defined | Manual |

### Additional Recommended Datasets

For medication attribute extraction, consider:

- **[n2c2 2018 Track 2](https://portal.dbmi.hms.harvard.edu/projects/n2c2-2018-t2/)**: Drug-dose-route-duration-frequency relations (requires DUA)
- **[MADE 1.0](https://pmc.ncbi.nlm.nih.gov/articles/PMC6860017/)**: Medication and adverse drug event extraction
- **[DDI-2013](https://sbmi.uth.edu/ccb/resources/ddi.htm)**: Drug-drug interaction corpus

## Usage

### Basic Evaluation

```bash
# Evaluate on ADE corpus
python benches/re/table_extraction_eval.py \
    --dataset ade_corpus \
    --server-url http://localhost:8000/v1

# With GLiNER service
python benches/re/table_extraction_eval.py \
    --dataset ade_corpus \
    --server-url http://localhost:8000/v1 \
    --gliner-url http://localhost:9001

# Custom labels
python benches/re/table_extraction_eval.py \
    --dataset ade_corpus \
    --labels drug adverse_effect \
    --server-url http://localhost:8000/v1
```

### Custom Dataset Evaluation

```bash
python benches/re/table_extraction_eval.py \
    --dataset custom \
    --data-file /path/to/your/data.jsonl \
    --labels drug dose duration frequency \
    --server-url http://localhost:8000/v1
```

### Debug Output

```bash
# Dump samples for visualization
python benches/re/table_extraction_eval.py \
    --dataset ade_corpus \
    --dump-jsonl /tmp/re_debug.jsonl \
    --dump-limit 20 \
    --server-url http://localhost:8000/v1
```

## JSONL Data Format

### Input Format (Custom Dataset)

```json
{"text": "Patient received 100mg aspirin daily for 2 weeks.", "table": {"headers": ["drug", "dose", "frequency", "duration"], "rows": [{"drug": "aspirin", "dose": "100mg", "frequency": "daily", "duration": "2 weeks"}]}}
```

Or with relations:
```json
{"text": "Aspirin caused headache.", "relations": [{"type": "drug-ade", "entities": {"drug": "aspirin", "ade": "headache"}}]}
```

### Output Format (for Visualizer)

```json
{
  "text": "source text",
  "gold_table": {
    "headers": ["col1", "col2"],
    "rows": [{"cells": {"col1": "val1", "col2": "val2"}, "row_idx": 0}],
    "raw_markdown": "| col1 | col2 |\n|---|---|\n| val1 | val2 |"
  },
  "pred_table": { ... },
  "gold_relations": [{"type": "relation-type", "entities": {"role1": "entity1"}}],
  "pred_relations": [...],
  "metadata": {}
}
```

## Metrics

The evaluation computes metrics at three levels:

### Cell-level Metrics
- Measures individual cell value matches
- Useful for partial credit evaluation

### Row-level Metrics
- Exact match of entire rows
- Primary metric for table extraction

### Relation-level Metrics
- Matches based on relation type and entities
- Standard RE evaluation

## Visualization

Load the output JSONL file in the visualizer:

```bash
cd visualizer
npm run dev
# Open http://localhost:5173/re
```

The visualizer shows:
- Side-by-side gold vs predicted tables
- True positives (green), false positives (red), false negatives (orange)
- Raw markdown output for debugging
- Overall precision, recall, F1 scores

## Integration with text2table

The evaluation script directly uses the `text2table` pipeline:

```python
from text2table import Text2Table

extractor = Text2Table(
    labels=["drug", "dose", "duration"],
    server_url="http://localhost:8000/v1",
    gliner_url="http://localhost:9001",  # optional
)

table_md, entities = extractor.run(text)
```

## Example: Drug Attribute Extraction

For your specific use case (drug-dose-duration):

```bash
# Prepare custom data
echo '{"text": "Rats were administered 50mg/kg metformin for 14 days.", "table": {"headers": ["drug", "dose", "num_of_animal", "duration"], "rows": [{"drug": "metformin", "dose": "50mg/kg", "num_of_animal": "rats", "duration": "14 days"}]}}' > /tmp/test_data.jsonl

# Run evaluation
python benches/re/table_extraction_eval.py \
    --dataset custom \
    --data-file /tmp/test_data.jsonl \
    --labels drug dose num_of_animal duration \
    --server-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-8B \
    --dump-jsonl /tmp/drug_attr_results.jsonl
```

## References

- [ADE Corpus V2](https://huggingface.co/datasets/ade-benchmark-corpus/ade_corpus_v2)
- [Drug Combination Extraction](https://huggingface.co/datasets/allenai/drug-combo-extraction)
- [n2c2 2018 Challenge](https://pubmed.ncbi.nlm.nih.gov/31584655/)
