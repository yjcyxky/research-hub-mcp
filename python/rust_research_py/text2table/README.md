# text2table

`text2table` converts raw text into TSV tables by optionally extracting entities with **GLiNER** (service or local) and then asking a **vLLM (OpenAI-compatible)** endpoint (e.g., `Qwen/Qwen3-30B-A3B-Instruct-2507` or the model id your server exposes) to render a table with user-supplied headers. The LLM is always called via the service; GLiNER can run locally or via HTTP.

## Requirements

Service mode only:
- Python 3.10+
- Running vLLM OpenAI-compatible endpoint (set via `TEXT2TABLE_VLLM_URL` or `--server-url`)
- Optional GLiNER HTTP service for entity extraction (set via `TEXT2TABLE_GLINER_URL` or `--gliner-url`)
- `openai`, `httpx`, `click`

## Starting the vLLM Server

```bash
python text2table/server.py --trust-remote-code --model Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --max-model-len 16384
```

> **Note**: Enabling FP8 quantization + limiting `max-model-len` allows 30B models to run via vLLM on NVIDIA GPUs with 48GB VRAM.

Expected GLiNER service contract (if you choose service mode):
- `POST /extract` with JSON payload `{"text": "...", "labels": [...], "threshold": 0.5, "model": "optional-name"}`
- Response either as a list of entity dicts or an object containing an `entities` array.

Local GLiNER mode:
- Install `gliner` and ensure GPU/CPU fits your chosen model.
- Optional `--gliner-cache-dir` and `--gliner-device` control weights location and device.

Set `HUGGINGFACE_HUB_TOKEN` on the vLLM host if the model requires authentication. If your services are secured, pass `--api-key` (vLLM) and/or `--gliner-api-key` (GLiNER).

## Usage

```bash
python -m text2table.cli run \
  --server-url http://localhost:8000/v1 \
  --gliner-url http://localhost:9001 \  # omit to use local GLiNER
  --model <server-model-id-if-not-default> \
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
- `--server-url` / `TEXT2TABLE_VLLM_URL`: vLLM OpenAI-compatible endpoint (required)
- `--gliner-url` / `TEXT2TABLE_GLINER_URL`: GLiNER service URL (if omitted and not disabled, uses local GLiNER)
- `--disable-gliner`: skip entity extraction and let the LLM infer directly from text
- `--gliner-model`, `--model` (optional): model hints forwarded to the services; if `--model` is omitted, the server's default model is used.
- `--gliner-cache-dir`, `--gliner-device`: control local GLiNER cache/device when not using the service
- `--threshold`: GLiNER score threshold (default `0.5`)
- `--request-timeout`, `--pool-size`, `--max-retries` (retries after the first call), `--backoff-factor`, `--max-backoff`: HTTP client controls for production use
- `--max-new-tokens`, `--temperature`, `--top-p`: table generation controls
- `--enable-thinking`: ask the model to think in `<think>...</think>` before the final table
- `--max-reasoning-tokens`: when using vLLM + thinking, how many reasoning tokens to request (default `2048`)
- `--output`: save the TSV table
- `--emit-entities` / `--dump-entities-json`: inspect extracted entities
- `--gliner-soft-threshold`: pull lower-confidence GLiNER candidates as soft hints (marked “low confidence - verify”)
- `--enable-row-validation`: drop generated rows not supported by the source text; combine with `--row-validation-mode substring|llm` (llm re-asks the model to verify each row)
- Output format is TSV (tab-separated), and the last column is always `confidence` in [0,1] (default 1.0).

Batch/concurrent processing via the async pipeline:

```bash
text2table run-batch \
  --input-jsonl samples.jsonl \
  --label Drug --label ADE \
  --server-url http://localhost:8000/v1 \
  --gliner-url http://localhost:9001 \
  --concurrency 4 \
  --dump-jsonl outputs/results.jsonl \
  --flush-every 20
```

`run-batch` fans out work through `AsyncText2Table` with a bounded concurrency (`--concurrency`). When `--dump-jsonl` is set, results are streamed to disk and flushed every `--flush-every` rows to avoid losing progress on long runs.

## Async usage

The package ships an asyncio-native pipeline via `AsyncText2Table`, allowing easy `asyncio.gather` fan-out while reusing connection pools:

```python
import asyncio
from text2table import AsyncText2Table, DEFAULT_USER_PROMPT

async def main():
    t2t = AsyncText2Table(
        labels=["Drug", "ADE"],
        server_url="http://localhost:8000/v1",
        gliner_url="http://localhost:9001",
        gliner_soft_threshold=0.3,  # pull extra low-conf candidates as soft hints
        enable_row_validation=True,  # drop rows not found in the source text
    )
    table, entities = await t2t.run(
        "Example text body.",
        user_prompt=DEFAULT_USER_PROMPT,
    )
    print(table)
    await t2t.close()

asyncio.run(main())
```

For concurrent batches, pass a list of `BatchItem` to `AsyncText2Table.run_many` and set `concurrency` (bounded semaphore):

```python
from text2table import AsyncText2Table, BatchItem, DEFAULT_USER_PROMPT

items = [
    BatchItem(text="Text A", id="a"),
    BatchItem(text="Text B", id="b"),
]

async def main():
    t2t = AsyncText2Table(labels=["Drug", "ADE"], server_url="http://localhost:8000/v1")
    results = await t2t.run_many(items, user_prompt=DEFAULT_USER_PROMPT, concurrency=4)
    await t2t.close()
    for res in results:
        print(res.id, res.table)

asyncio.run(main())
```

## Pipeline

1. (Optional) Call the GLiNER service to extract entities for the provided labels.
2. Build a compact prompt containing headers, deduplicated entities (or instructions to infer), and the source text.
3. Call the vLLM service to emit **only** a TSV table with headers in the given order, optionally including `<think>` reasoning.
