"""
Command line interface for the text2table pipeline.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import click

CLI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLI_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from text2table import (  # noqa: E402
    DEFAULT_USER_PROMPT,
    AsyncText2Table,
    BatchItem,
    BatchResult,
    Text2Table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _read_text(text: Optional[str], text_file: Optional[Path]) -> str:
    if text_file:
        return text_file.read_text(encoding="utf-8")
    if text:
        return text
    raise click.UsageError("Provide either --text or --text-file.")


def _load_labels(cli_labels: Sequence[str], labels_file: Optional[Path]) -> List[str]:
    labels: List[str] = [label.strip() for label in cli_labels if label.strip()]
    if labels_file:
        file_labels = [
            line.strip() for line in labels_file.read_text(encoding="utf-8").splitlines()
        ]
        labels.extend([label for label in file_labels if label])
    if not labels:
        raise click.UsageError("At least one label is required via --label or --labels-file.")
    return labels


def _echo_entities(entities: Iterable[dict]) -> None:
    for entity in entities:
        label = entity.get("label")
        text = entity.get("text")
        score = entity.get("score")
        score_display = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        logger.info("Entity: %s -> %s (score=%s)", label, text, score_display)


def _append_jsonl(output_path: Path, rows: List[dict]) -> None:
    """Append a batch of rows to a JSONL file, creating parents as needed."""
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for row in rows:
            payload = row.to_dict() if hasattr(row, "to_dict") else row
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")


def _load_jsonl_records(
    input_path: Path, text_field: str, id_field: Optional[str]
) -> Iterable[Tuple[int, str, Optional[object], dict]]:
    """Yield (index, text, record_id, raw_record) from a JSONL file."""
    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise click.UsageError(
                    f"Line {idx + 1} in {input_path} is not valid JSON: {exc}"
                )

            if text_field not in payload:
                logger.warning(
                    "Skipping line %d: missing text field '%s'", idx + 1, text_field
                )
                continue

            text = str(payload[text_field])
            if not text.strip():
                logger.warning(
                    "Skipping line %d: text field '%s' is empty", idx + 1, text_field
                )
                continue

            record_id = payload.get(id_field) if id_field else None
            yield idx, text, record_id, payload


@click.group(
    name="text2table",
    help="Extract entities via a GLiNER service (optional) and render tables via a vLLM OpenAI-compatible endpoint.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli() -> None:
    """CLI entrypoint."""


@cli.command(
    name="run",
    help="Extract entities from text and render a TSV table.",
)
@click.option(
    "--text",
    type=str,
    help="Raw text to process.",
)
@click.option(
    "--text-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a text/markdown file to process.",
)
@click.option(
    "--label",
    "labels",
    multiple=True,
    help="Label to extract (repeat for multiple labels).",
)
@click.option(
    "--labels-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File with one label per line.",
)
@click.option(
    "--prompt",
    default=None,
    help="Optional custom prompt appended to the generation request.",
)
@click.option(
    "--threshold",
    default=0.5,
    show_default=True,
    help="Confidence threshold for GLiNER (when enabled).",
)
@click.option(
    "--gliner-model",
    default="Ihor/gliner-biomed-large-v1.0",
    show_default=True,
    help="GLiNER model name or path (used for local mode or as a hint to the service).",
)
@click.option(
    "--gliner-soft-threshold",
    type=float,
    default=None,
    show_default=False,
    help="Lower GLiNER threshold for recall-only candidates (marked as low-confidence hints).",
)
@click.option(
    "--gliner-soft-threshold",
    type=float,
    default=None,
    show_default=False,
    help="Lower GLiNER threshold for recall-only candidates (marked as low-confidence hints).",
)
@click.option(
    "--model",
    "--llm-model",
    "model",
    default=None,
    show_default=False,
    help="Model name to request from the vLLM server; if omitted, use the server's default model.",
)
@click.option(
    "--max-new-tokens",
    default=512,
    show_default=True,
    help="Maximum tokens to generate for the table.",
)
@click.option(
    "--temperature",
    default=0.2,
    show_default=True,
    help="Sampling temperature for table generation.",
)
@click.option(
    "--top-p",
    default=0.9,
    show_default=True,
    help="Top-p sampling for table generation.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional path to write the generated table (TSV).",
)
@click.option(
    "--emit-entities",
    is_flag=True,
    help="Log extracted entities to stderr.",
)
@click.option(
    "--dump-entities-json",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional path to write extracted entities as JSON.",
)
@click.option(
    "--enable-thinking",
    is_flag=True,
    default=False,
    help="Enable thinking mode: model will output reasoning process before final result.",
)
@click.option(
    "--max-reasoning-tokens",
    type=int,
    default=2048,
    show_default=True,
    help="Maximum reasoning tokens to request in thinking mode (vLLM only).",
)
@click.option(
    "--server-url",
    type=str,
    envvar="TEXT2TABLE_VLLM_URL",
    help="vLLM server URL (OpenAI-compatible, e.g., http://localhost:8000/v1). Required for generation.",
)
@click.option(
    "--api-key",
    type=str,
    default="dummy-key",
    show_default=True,
    help="API key for the vLLM service if it requires authentication.",
)
@click.option(
    "--gliner-url",
    type=str,
    envvar="TEXT2TABLE_GLINER_URL",
    help="GLiNER service base URL (e.g., http://localhost:9001). If omitted, falls back to local GLiNER.",
)
@click.option(
    "--gliner-api-key",
    type=str,
    default=None,
    help="Optional bearer token for the GLiNER service.",
)
@click.option(
    "--disable-gliner",
    is_flag=True,
    default=False,
    help="Skip GLiNER extraction and rely on the LLM to infer entities directly from the source text.",
)
@click.option(
    "--gliner-cache-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Cache directory for local GLiNER weights (local mode).",
)
@click.option(
    "--gliner-device",
    type=str,
    default=None,
    help="Target device for local GLiNER (cpu, cuda, etc.).",
)
@click.option(
    "--request-timeout",
    type=float,
    default=120.0,
    show_default=True,
    help="HTTP timeout (seconds) for both GLiNER and vLLM service calls.",
)
@click.option(
    "--pool-size",
    type=int,
    default=10,
    show_default=True,
    help="Connection pool size for service HTTP clients.",
)
@click.option(
    "--max-retries",
    type=int,
    default=3,
    show_default=True,
    help="Retry attempts per request (in addition to the initial call).",
)
@click.option(
    "--backoff-factor",
    type=float,
    default=1.5,
    show_default=True,
    help="Exponential backoff factor between retries.",
)
@click.option(
    "--max-backoff",
    type=float,
    default=10.0,
    show_default=True,
    help="Maximum backoff time (seconds) between retries.",
)
@click.option(
    "--enable-row-validation",
    is_flag=True,
    default=False,
    help="Validate each generated row against the source text (substring or LLM) and drop unsupported rows.",
)
@click.option(
    "--row-validation-mode",
    type=click.Choice(["substring", "llm"]),
    default="substring",
    show_default=True,
    help="Row validation strategy.",
)
def run_command(
    text: Optional[str],
    text_file: Optional[Path],
    labels: Sequence[str],
    labels_file: Optional[Path],
    prompt: Optional[str],
    threshold: float,
    gliner_model: str,
    gliner_soft_threshold: Optional[float],
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output: Optional[Path],
    emit_entities: bool,
    dump_entities_json: Optional[Path],
    enable_thinking: bool,
    max_reasoning_tokens: int,
    server_url: Optional[str],
    api_key: str,
    gliner_url: Optional[str],
    gliner_api_key: Optional[str],
    disable_gliner: bool,
    gliner_cache_dir: Optional[Path],
    gliner_device: Optional[str],
    request_timeout: float,
    pool_size: int,
    max_retries: int,
    backoff_factor: float,
    max_backoff: float,
    enable_row_validation: bool,
    row_validation_mode: str,
) -> None:
    labels_list = _load_labels(labels, labels_file)
    source_text = _read_text(text, text_file)

    extractor = Text2Table(
        labels=labels_list,
        gliner_model_name=gliner_model,
        model_name=model,
        threshold=threshold,
        gliner_soft_threshold=gliner_soft_threshold,
        enable_thinking=enable_thinking,
        max_reasoning_tokens=max_reasoning_tokens,
        server_url=server_url,
        gliner_url=gliner_url,
        use_gliner=not disable_gliner,
        api_key=api_key,
        gliner_api_key=gliner_api_key,
        gliner_cache_dir=gliner_cache_dir,
        gliner_device=gliner_device,
        request_timeout=request_timeout,
        pool_size=pool_size,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        max_backoff=max_backoff,
        enable_row_validation=enable_row_validation,
        row_validation_mode=row_validation_mode,
    )
    
    if not extractor.server_url:
        raise click.UsageError(
            "vLLM server URL is required. Provide --server-url or TEXT2TABLE_VLLM_URL."
        )
    logger.info("Using vLLM server at: %s", extractor.server_url)
    if disable_gliner:
        logger.info("GLiNER is disabled; relying on vLLM only.")
    elif extractor.gliner_url:
        logger.info("Using GLiNER service at: %s", extractor.gliner_url)
    else:
        logger.info("Using local GLiNER model: %s", extractor.gliner_model_name)

    try:
        if enable_thinking:
            thinking, table, entities = extractor.run_with_thinking(
                source_text,
                user_prompt=prompt or DEFAULT_USER_PROMPT,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            logger.info("Thinking mode: thinking_len=%d, table_len=%d", 
                       len(thinking) if thinking else 0, len(table) if table else 0)
            if thinking:
                click.echo("=== Thinking Process ===", err=True)
                click.echo(thinking, err=True)
                click.echo("=== Final Result ===\n", err=True)
        else:
            table, entities = extractor.run(
                source_text,
                user_prompt=prompt or DEFAULT_USER_PROMPT,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            logger.info("Normal mode: table_len=%d", len(table) if table else 0)

        if not table or not table.strip():
            logger.warning("Generated table is empty! Output length: %d", len(table) if table else 0)
            click.echo("Warning: No table was generated. Check logs for details.", err=True)
        else:
            logger.info("Successfully generated table (length: %d)", len(table))
            click.echo(table)
    except Exception as e:
        logger.error("Error during processing: %s", e, exc_info=True)
        click.echo(f"Error: {e}", err=True)
        raise

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(table, encoding="utf-8")
        logger.info("Table saved to %s", output)

    if emit_entities:
        _echo_entities(entities)

    if dump_entities_json:
        dump_entities_json.parent.mkdir(parents=True, exist_ok=True)
        dump_entities_json.write_text(json.dumps(entities, indent=2), encoding="utf-8")
        logger.info("Entities saved to %s", dump_entities_json)


@cli.command(
    name="run-batch",
    help="Process multiple texts from a JSONL file, optionally with async concurrency.",
)
@click.option(
    "--input-jsonl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file containing one record per line.",
)
@click.option(
    "--text-field",
    default="text",
    show_default=True,
    help="Field name in the JSONL records that contains the source text.",
)
@click.option(
    "--id-field",
    default=None,
    show_default=False,
    help="Optional field name to treat as an ID and propagate to outputs.",
)
@click.option(
    "--dump-jsonl",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional JSONL path to write results incrementally (one line per input).",
)
@click.option(
    "--flush-every",
    type=int,
    default=20,
    show_default=True,
    help="When using --dump-jsonl, flush to disk after this many processed rows.",
)
@click.option(
    "--concurrency",
    type=int,
    default=4,
    show_default=True,
    help="Max in-flight texts to process concurrently (uses AsyncText2Table).",
)
@click.option(
    "--label",
    "labels",
    multiple=True,
    help="Label to extract (repeat for multiple labels).",
)
@click.option(
    "--labels-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File with one label per line.",
)
@click.option(
    "--prompt",
    default=None,
    help="Optional custom prompt appended to the generation request.",
)
@click.option(
    "--threshold",
    default=0.5,
    show_default=True,
    help="Confidence threshold for GLiNER (when enabled).",
)
@click.option(
    "--gliner-model",
    default="Ihor/gliner-biomed-large-v1.0",
    show_default=True,
    help="GLiNER model name or path (used for local mode or as a hint to the service).",
)
@click.option(
    "--model",
    "--llm-model",
    "model",
    default=None,
    show_default=False,
    help="Model name to request from the vLLM server; if omitted, use the server's default model.",
)
@click.option(
    "--max-new-tokens",
    default=512,
    show_default=True,
    help="Maximum tokens to generate for the table.",
)
@click.option(
    "--temperature",
    default=0.2,
    show_default=True,
    help="Sampling temperature for table generation.",
)
@click.option(
    "--top-p",
    default=0.9,
    show_default=True,
    help="Top-p sampling for table generation.",
)
@click.option(
    "--emit-entities",
    is_flag=True,
    help="Log extracted entities to stderr.",
)
@click.option(
    "--enable-thinking",
    is_flag=True,
    default=False,
    help="Enable thinking mode: model will output reasoning process before final result.",
)
@click.option(
    "--max-reasoning-tokens",
    type=int,
    default=2048,
    show_default=True,
    help="Maximum reasoning tokens to request in thinking mode (vLLM only).",
)
@click.option(
    "--server-url",
    type=str,
    envvar="TEXT2TABLE_VLLM_URL",
    help="vLLM server URL (OpenAI-compatible, e.g., http://localhost:8000/v1). Required for generation.",
)
@click.option(
    "--api-key",
    type=str,
    default="dummy-key",
    show_default=True,
    help="API key for the vLLM service if it requires authentication.",
)
@click.option(
    "--gliner-url",
    type=str,
    envvar="TEXT2TABLE_GLINER_URL",
    help="GLiNER service base URL (e.g., http://localhost:9001). If omitted, falls back to local GLiNER.",
)
@click.option(
    "--gliner-api-key",
    type=str,
    default=None,
    help="Optional bearer token for the GLiNER service.",
)
@click.option(
    "--disable-gliner",
    is_flag=True,
    default=False,
    help="Skip GLiNER extraction and rely on the LLM to infer entities directly from the source text.",
)
@click.option(
    "--gliner-cache-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Cache directory for local GLiNER weights (local mode).",
)
@click.option(
    "--gliner-device",
    type=str,
    default=None,
    help="Target device for local GLiNER (cpu, cuda, etc.).",
)
@click.option(
    "--request-timeout",
    type=float,
    default=120.0,
    show_default=True,
    help="HTTP timeout (seconds) for both GLiNER and vLLM service calls.",
)
@click.option(
    "--pool-size",
    type=int,
    default=10,
    show_default=True,
    help="Connection pool size for service HTTP clients.",
)
@click.option(
    "--max-retries",
    type=int,
    default=3,
    show_default=True,
    help="Retry attempts per request (in addition to the initial call).",
)
@click.option(
    "--backoff-factor",
    type=float,
    default=1.5,
    show_default=True,
    help="Exponential backoff factor between retries.",
)
@click.option(
    "--max-backoff",
    type=float,
    default=10.0,
    show_default=True,
    help="Maximum backoff time (seconds) between retries.",
)
@click.option(
    "--enable-row-validation",
    is_flag=True,
    default=False,
    help="Validate each generated row against the source text (substring or LLM) and drop unsupported rows.",
)
@click.option(
    "--row-validation-mode",
    type=click.Choice(["substring", "llm"]),
    default="substring",
    show_default=True,
    help="Row validation strategy.",
)
def run_batch_command(
    input_jsonl: Path,
    text_field: str,
    id_field: Optional[str],
    dump_jsonl: Optional[Path],
    flush_every: int,
    concurrency: int,
    labels: Sequence[str],
    labels_file: Optional[Path],
    prompt: Optional[str],
    threshold: float,
    gliner_model: str,
    gliner_soft_threshold: Optional[float],
    model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    emit_entities: bool,
    enable_thinking: bool,
    max_reasoning_tokens: int,
    server_url: Optional[str],
    api_key: str,
    gliner_url: Optional[str],
    gliner_api_key: Optional[str],
    disable_gliner: bool,
    gliner_cache_dir: Optional[Path],
    gliner_device: Optional[str],
    request_timeout: float,
    pool_size: int,
    max_retries: int,
    backoff_factor: float,
    max_backoff: float,
    enable_row_validation: bool,
    row_validation_mode: str,
) -> None:
    """Batch runner that fans out work via AsyncText2Table with bounded concurrency."""
    labels_list = _load_labels(labels, labels_file)
    if flush_every <= 0:
        raise click.UsageError("--flush-every must be a positive integer.")
    if concurrency <= 0:
        raise click.UsageError("--concurrency must be a positive integer.")

    records = list(_load_jsonl_records(input_jsonl, text_field, id_field))
    if not records:
        logger.warning("No valid records found in %s", input_jsonl)
        return

    extractor = AsyncText2Table(
        labels=labels_list,
        gliner_model_name=gliner_model,
        model_name=model,
        threshold=threshold,
        gliner_soft_threshold=gliner_soft_threshold,
        enable_thinking=enable_thinking,
        max_reasoning_tokens=max_reasoning_tokens,
        server_url=server_url,
        gliner_url=gliner_url,
        use_gliner=not disable_gliner,
        api_key=api_key,
        gliner_api_key=gliner_api_key,
        gliner_cache_dir=gliner_cache_dir,
        gliner_device=gliner_device,
        request_timeout=request_timeout,
        pool_size=pool_size,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        max_backoff=max_backoff,
        enable_row_validation=enable_row_validation,
        row_validation_mode=row_validation_mode,
    )

    if not extractor.server_url:
        raise click.UsageError(
            "vLLM server URL is required. Provide --server-url or TEXT2TABLE_VLLM_URL."
        )
    logger.info("Using vLLM server at: %s", extractor.server_url)
    if disable_gliner:
        logger.info("GLiNER is disabled; relying on vLLM only.")
    elif extractor.gliner_url:
        logger.info("Using GLiNER service at: %s", extractor.gliner_url)
    else:
        logger.info("Using local GLiNER model: %s", extractor.gliner_model_name)

    if dump_jsonl:
        dump_jsonl.parent.mkdir(parents=True, exist_ok=True)
        dump_jsonl.write_text("", encoding="utf-8")
        logger.info("Streaming results to %s (flush every %d rows)", dump_jsonl, flush_every)

    batch_items = [
        BatchItem(text=text_value, id=record_id, index=idx) for idx, text_value, record_id, _ in records
    ]

    async def _run_all() -> None:
        buffer: List[BatchResult] = []
        processed = 0

        async def handle_result(result: BatchResult) -> None:
            nonlocal processed, buffer
            processed += 1
            if result.status == "ok":
                logger.info(
                    "Processed record %s (index %d), table_len=%d",
                    result.id,
                    result.index,
                    len(result.table or ""),
                )
                if emit_entities and result.entities:
                    _echo_entities(result.entities)
            else:
                logger.warning(
                    "Record %s (index %d) failed: %s",
                    result.id,
                    result.index,
                    result.error,
                )

            if not dump_jsonl:
                header = f"Record {result.index}"
                if result.id is not None:
                    header += f" (id={result.id})"
                click.echo(header)
                if result.status == "ok":
                    click.echo(result.table or "")
                else:
                    click.echo(f"ERROR: {result.error}", err=True)
                click.echo()
            else:
                buffer.append(result)
                if len(buffer) >= flush_every:
                    _append_jsonl(dump_jsonl, buffer)
                    buffer.clear()

        try:
            await extractor.run_many(
                batch_items,
                user_prompt=prompt or DEFAULT_USER_PROMPT,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                concurrency=concurrency,
                on_result=handle_result,
            )
        finally:
            if dump_jsonl and buffer:
                _append_jsonl(dump_jsonl, buffer)
            await extractor.close()
            if not dump_jsonl:
                logger.info("Processed %d records (no dump-jsonl provided)", processed)
            else:
                logger.info("Processed %d records and wrote to %s", processed, dump_jsonl)

    asyncio.run(_run_all())


if __name__ == "__main__":
    cli()
