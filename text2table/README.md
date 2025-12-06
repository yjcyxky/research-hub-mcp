# text2table

`text2table` converts raw text into Markdown tables by first extracting entities with **GLiNER** and then asking **Qwen/Qwen3-30B-A3B-Instruct-2507** to render a table with user-supplied headers.

## Requirements

- Python 3.10+
- `torch`, `transformers`, `accelerate`
- `gliner`
- `click`
- Sufficient GPU/CPU memory for Qwen 30B (consider 4/8-bit loading if adapting).

Set `HUGGINGFACE_HUB_TOKEN` if the models require authentication and optionally `HUGGINGFACE_HUB_CACHE` for cache placement.

## Usage

```bash
python -m text2table.cli run \
  --text-file examples/test_ner.txt \
  --label "Disease" \
  --label "Drug" \
  --label "Drug dosage" \
  --label "Drug frequency" \
  --label "Lab test" \
  --label "Lab test value" \
  --label "Demographic information" \
  --label "Animal Model" \
  --label "Number of animal model" \
  --prompt "请基于上述信息，输出包含labels为表头信息的表格"
```

Options:
- `--gliner-model`: defaults to `Ihor/gliner-biomed-large-v1.0`
- `--qwen-model`: defaults to `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `--threshold`: GLiNER score threshold (default `0.5`)
- `--max-new-tokens`, `--temperature`, `--top-p`: table generation controls
- `--output`: save the Markdown table
- `--emit-entities` / `--dump-entities-json`: inspect extracted entities

## Pipeline

1. Load GLiNER and extract entities for the provided labels.
2. Build a compact prompt containing headers, deduplicated entities, and the source text.
3. Ask Qwen to emit **only** a Markdown table with headers in the given order.
