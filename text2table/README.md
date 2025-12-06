# text2table

`text2table` converts raw text into Markdown tables by optionally extracting entities with **GLiNER** (service or local) and then asking a **vLLM (OpenAI-compatible)** endpoint (e.g., `Qwen/Qwen3-30B-A3B-Instruct-2507` or the model id your server exposes) to render a table with user-supplied headers. The LLM is always called via the service; GLiNER can run locally or via HTTP.

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

> **Note**: 启用 FP8 量化 + 限制 `max-model-len` 可使 30B 模型通过 vLLM 在 48GB 显存的 NVIDIA GPU 上运行。

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
  --qwen-model <server-model-id-if-not-default> \
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
- `--gliner-model`, `--qwen-model` (optional): model hints forwarded to the services; if `--qwen-model` is omitted, the server's default model is used.
- `--gliner-cache-dir`, `--gliner-device`: control local GLiNER cache/device when not using the service
- `--threshold`: GLiNER score threshold (default `0.5`)
- `--request-timeout`, `--pool-size`, `--max-retries` (retries after the first call), `--backoff-factor`, `--max-backoff`: HTTP client controls for production use
- `--max-new-tokens`, `--temperature`, `--top-p`: table generation controls
- `--enable-thinking`: ask the model to think in `<think>...</think>` before the final table
- `--max-reasoning-tokens`: when using vLLM + thinking, how many reasoning tokens to request (default `2048`)
- `--output`: save the Markdown table
- `--emit-entities` / `--dump-entities-json`: inspect extracted entities

## Pipeline

1. (Optional) Call the GLiNER service to extract entities for the provided labels.
2. Build a compact prompt containing headers, deduplicated entities (or instructions to infer), and the source text.
3. Call the vLLM service to emit **only** a Markdown table with headers in the given order, optionally including `<think>` reasoning.
