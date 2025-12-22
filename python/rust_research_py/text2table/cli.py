"""
Unified command line interface for the text2table pipeline.

Supports single text and batch processing with automatic format detection.
"""

import asyncio
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import click

CLI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLI_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rust_research_py.text2table import (  # noqa: E402
    DEFAULT_USER_PROMPT,
    AsyncText2Table,
    BatchItem,
    BatchResult,
    Text2Table,
)
from rust_research_py.text2table.config import (  # noqa: E402
    detect_input_format,
    detect_output_format,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _load_labels(cli_labels: Sequence[str], labels_file: Optional[Path]) -> List[str]:
    """Load labels from CLI arguments and/or file."""
    labels: List[str] = [label.strip() for label in cli_labels if label.strip()]
    if labels_file:
        file_labels = [
            line.strip() for line in labels_file.read_text(encoding="utf-8").splitlines()
        ]
        labels.extend([label for label in file_labels if label])
    if not labels:
        raise click.UsageError("At least one label is required via --label or --labels-file.")
    return labels


def _load_single_text(path: Path) -> str:
    """Load text from a single .txt file."""
    return path.read_text(encoding="utf-8")


def _load_tsv(
    path: Path, text_column: str, id_column: Optional[str]
) -> Iterable[Tuple[int, str, Optional[str], Dict]]:
    """Load records from TSV file."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            if text_column not in row:
                logger.warning("Skipping row %d: missing text column '%s'", idx, text_column)
                continue
            text = row[text_column].strip()
            if not text:
                logger.warning("Skipping row %d: empty text", idx)
                continue
            record_id = row.get(id_column) if id_column else None
            yield idx, text, record_id, row


def _load_csv(
    path: Path, text_column: str, id_column: Optional[str]
) -> Iterable[Tuple[int, str, Optional[str], Dict]]:
    """Load records from CSV file."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if text_column not in row:
                logger.warning("Skipping row %d: missing text column '%s'", idx, text_column)
                continue
            text = row[text_column].strip()
            if not text:
                logger.warning("Skipping row %d: empty text", idx)
                continue
            record_id = row.get(id_column) if id_column else None
            yield idx, text, record_id, row


def _load_jsonl(
    path: Path, text_column: str, id_column: Optional[str]
) -> Iterable[Tuple[int, str, Optional[str], Dict]]:
    """Load records from JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise click.UsageError(f"Line {idx + 1}: invalid JSON: {exc}")

            if text_column not in payload:
                logger.warning("Skipping line %d: missing text field '%s'", idx + 1, text_column)
                continue

            text = str(payload[text_column]).strip()
            if not text:
                logger.warning("Skipping line %d: empty text", idx + 1)
                continue

            record_id = payload.get(id_column) if id_column else None
            yield idx, text, record_id, payload


def _output_tsv(results: List[dict], output: Optional[Path]) -> None:
    """Output results as TSV."""
    if not results:
        return

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            for r in results:
                if r.get("table"):
                    f.write(r["table"])
                    f.write("\n")
        logger.info("Results saved to %s", output)
    else:
        for r in results:
            if r.get("table"):
                click.echo(r["table"])


def _output_csv(results: List[dict], output: Path) -> None:
    """Output results as CSV."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        for r in results:
            if r.get("table"):
                f.write(r["table"].replace("\t", ","))
                f.write("\n")
    logger.info("Results saved to %s", output)


def _output_jsonl(results: List[dict], output: Path) -> None:
    """Output results as JSONL."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")
    logger.info("Results saved to %s", output)


def _append_jsonl_batch(output_path: Path, rows: List[BatchResult]) -> None:
    """Append batch results to JSONL file."""
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for row in rows:
            payload = row.to_dict() if hasattr(row, "to_dict") else {
                "id": row.id,
                "index": row.index,
                "status": row.status,
                "table": row.table,
                "entities": row.entities,
                "error": row.error,
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")


@click.command()
@click.argument(
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
)
@click.option(
    "--text", "-t",
    type=str,
    help="Direct text input (instead of file)",
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output file path (format detected from extension: .tsv, .csv, .jsonl)",
)
@click.option(
    "--label", "-l",
    "labels",
    multiple=True,
    help="Column label to extract (repeat for multiple)",
)
@click.option(
    "--labels-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File with one label per line",
)
# Core server options
@click.option(
    "--server-url",
    envvar="TEXT2TABLE_VLLM_URL",
    required=True,
    help="vLLM server URL (e.g., http://localhost:8000/v1)",
)
@click.option(
    "--gliner-url",
    envvar="TEXT2TABLE_GLINER_URL",
    help="GLiNER service URL (optional)",
)
# Model options
@click.option(
    "--model",
    default=None,
    help="Model name for vLLM server",
)
@click.option(
    "--gliner-model",
    default="Ihor/gliner-biomed-large-v1.0",
    show_default=True,
    help="GLiNER model name",
)
@click.option(
    "--disable-gliner",
    is_flag=True,
    help="Skip GLiNER, rely on LLM only",
)
# Processing options
@click.option(
    "--threshold",
    default=0.5,
    type=float,
    show_default=True,
    help="GLiNER confidence threshold",
)
@click.option(
    "--gliner-soft-threshold",
    default=0.3,
    type=float,
    show_default=True,
    help="GLiNER soft threshold for recall candidates",
)
@click.option(
    "--max-new-tokens",
    default=4096,
    type=int,
    show_default=True,
    help="Maximum tokens to generate",
)
@click.option(
    "--request-timeout",
    default=600,
    type=int,
    show_default=True,
    help="HTTP timeout in seconds",
)
@click.option(
    "--temperature",
    default=0.2,
    type=float,
    show_default=True,
    help="Sampling temperature",
)
@click.option(
    "--top-p",
    default=0.9,
    type=float,
    show_default=True,
    help="Top-p sampling",
)
# Thinking mode
@click.option(
    "--enable-thinking",
    is_flag=True,
    help="Enable reasoning mode",
)
@click.option(
    "--max-reasoning-tokens",
    default=2048,
    type=int,
    show_default=True,
    help="Max reasoning tokens",
)
# Validation
@click.option(
    "--enable-row-validation",
    is_flag=True,
    help="Validate rows against source text",
)
@click.option(
    "--row-validation-mode",
    type=click.Choice(["substring", "llm"]),
    default="substring",
    show_default=True,
    help="Row validation strategy",
)
# Batch options
@click.option(
    "--text-column",
    default="text",
    show_default=True,
    help="Column containing text (batch mode)",
)
@click.option(
    "--id-column",
    default=None,
    help="Column for record ID (batch mode)",
)
@click.option(
    "--concurrency",
    default=4,
    type=int,
    show_default=True,
    help="Concurrent requests (batch mode)",
)
@click.option(
    "--flush-every",
    default=20,
    type=int,
    show_default=True,
    help="Flush to disk interval (batch mode)",
)
# HTTP client options
@click.option(
    "--api-key",
    default="dummy-key",
    help="API key for vLLM",
)
@click.option(
    "--gliner-api-key",
    default=None,
    help="API key for GLiNER service",
)
@click.option(
    "--pool-size",
    default=10,
    type=int,
    show_default=True,
    help="HTTP connection pool size",
)
@click.option(
    "--max-retries",
    default=3,
    type=int,
    show_default=True,
    help="Max retry attempts",
)
@click.option(
    "--backoff-factor",
    default=1.5,
    type=float,
    show_default=True,
    help="Exponential backoff factor",
)
@click.option(
    "--max-backoff",
    default=10.0,
    type=float,
    show_default=True,
    help="Max backoff time in seconds",
)
# Optional prompt
@click.option(
    "--prompt",
    default=None,
    help="Custom prompt (appended to generation)",
)
# Debug
@click.option(
    "--emit-entities",
    is_flag=True,
    help="Log extracted entities",
)
@click.option(
    "--dump-entities-json",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Save entities to JSON file (single mode)",
)
def main(
    input_file: Optional[Path],
    text: Optional[str],
    output: Optional[Path],
    labels: Sequence[str],
    labels_file: Optional[Path],
    server_url: str,
    gliner_url: Optional[str],
    model: Optional[str],
    gliner_model: str,
    disable_gliner: bool,
    threshold: float,
    gliner_soft_threshold: float,
    max_new_tokens: int,
    request_timeout: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    max_reasoning_tokens: int,
    enable_row_validation: bool,
    row_validation_mode: str,
    text_column: str,
    id_column: Optional[str],
    concurrency: int,
    flush_every: int,
    api_key: str,
    gliner_api_key: Optional[str],
    pool_size: int,
    max_retries: int,
    backoff_factor: float,
    max_backoff: float,
    prompt: Optional[str],
    emit_entities: bool,
    dump_entities_json: Optional[Path],
) -> None:
    """Extract structured tables from text.

    INPUT_FILE can be:
    - A .txt file (single text mode)
    - A .tsv/.csv file (batch mode)
    - A .jsonl file (batch mode)

    Or use --text for direct text input.

    OUTPUT format is determined by extension:
    - .tsv: Tab-separated (default)
    - .csv: Comma-separated
    - .jsonl: JSON Lines

    Examples:
        t2t input.txt --label "Gene" --label "Disease" -o output.tsv
        t2t batch.jsonl --text-column "abstract" -o results.jsonl
        t2t --text "Some text..." --label "Entity" --server-url http://localhost:8000/v1
    """
    # Determine input mode
    if text:
        mode = "single"
        input_texts = [(0, text, None, {})]
    elif input_file:
        try:
            mode = detect_input_format(input_file)
        except ValueError as e:
            raise click.UsageError(str(e))

        if mode == "single":
            input_texts = [(0, _load_single_text(input_file), None, {})]
        else:
            ext = input_file.suffix.lower()
            if ext == ".tsv":
                input_texts = list(_load_tsv(input_file, text_column, id_column))
            elif ext == ".csv":
                input_texts = list(_load_csv(input_file, text_column, id_column))
            elif ext == ".jsonl":
                input_texts = list(_load_jsonl(input_file, text_column, id_column))
            else:
                raise click.UsageError(f"Unsupported format: {ext}")
    else:
        raise click.UsageError("Either INPUT_FILE or --text is required")

    if not input_texts:
        logger.warning("No valid input texts found")
        return

    # Load labels
    labels_list = _load_labels(labels, labels_file)

    # Determine output format
    out_format = detect_output_format(output)

    # Log configuration
    logger.info("Using vLLM server at: %s", server_url)
    if disable_gliner:
        logger.info("GLiNER disabled; relying on vLLM only")
    elif gliner_url:
        logger.info("Using GLiNER service at: %s", gliner_url)
    else:
        logger.info("Using local GLiNER model: %s", gliner_model)
    logger.info("Mode: %s, Input records: %d", mode, len(input_texts))

    # Process based on mode
    if mode == "single":
        _run_single(
            text=input_texts[0][1],
            labels=labels_list,
            output=output,
            out_format=out_format,
            server_url=server_url,
            gliner_url=gliner_url,
            model=model,
            gliner_model=gliner_model,
            disable_gliner=disable_gliner,
            threshold=threshold,
            gliner_soft_threshold=gliner_soft_threshold,
            max_new_tokens=max_new_tokens,
            request_timeout=request_timeout,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
            max_reasoning_tokens=max_reasoning_tokens,
            enable_row_validation=enable_row_validation,
            row_validation_mode=row_validation_mode,
            api_key=api_key,
            gliner_api_key=gliner_api_key,
            pool_size=pool_size,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            max_backoff=max_backoff,
            prompt=prompt,
            emit_entities=emit_entities,
            dump_entities_json=dump_entities_json,
        )
    else:
        _run_batch(
            input_texts=input_texts,
            labels=labels_list,
            output=output,
            out_format=out_format,
            server_url=server_url,
            gliner_url=gliner_url,
            model=model,
            gliner_model=gliner_model,
            disable_gliner=disable_gliner,
            threshold=threshold,
            gliner_soft_threshold=gliner_soft_threshold,
            max_new_tokens=max_new_tokens,
            request_timeout=request_timeout,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
            max_reasoning_tokens=max_reasoning_tokens,
            enable_row_validation=enable_row_validation,
            row_validation_mode=row_validation_mode,
            concurrency=concurrency,
            flush_every=flush_every,
            api_key=api_key,
            gliner_api_key=gliner_api_key,
            pool_size=pool_size,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            max_backoff=max_backoff,
            prompt=prompt,
            emit_entities=emit_entities,
        )


def _run_single(
    text: str,
    labels: List[str],
    output: Optional[Path],
    out_format: str,
    server_url: str,
    gliner_url: Optional[str],
    model: Optional[str],
    gliner_model: str,
    disable_gliner: bool,
    threshold: float,
    gliner_soft_threshold: float,
    max_new_tokens: int,
    request_timeout: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    max_reasoning_tokens: int,
    enable_row_validation: bool,
    row_validation_mode: str,
    api_key: str,
    gliner_api_key: Optional[str],
    pool_size: int,
    max_retries: int,
    backoff_factor: float,
    max_backoff: float,
    prompt: Optional[str],
    emit_entities: bool,
    dump_entities_json: Optional[Path],
) -> None:
    """Run single text extraction."""
    extractor = Text2Table(
        labels=labels,
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
        request_timeout=request_timeout,
        pool_size=pool_size,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        max_backoff=max_backoff,
        enable_row_validation=enable_row_validation,
        row_validation_mode=row_validation_mode,
    )

    try:
        if enable_thinking:
            thinking, table, entities = extractor.run_with_thinking(
                text,
                user_prompt=prompt or DEFAULT_USER_PROMPT,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            if thinking:
                click.echo("=== Thinking Process ===", err=True)
                click.echo(thinking, err=True)
                click.echo("=== Final Result ===\n", err=True)
        else:
            table, entities = extractor.run(
                text,
                user_prompt=prompt or DEFAULT_USER_PROMPT,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        if not table or not table.strip():
            logger.warning("Generated table is empty")
            click.echo("Warning: No table was generated.", err=True)
        else:
            logger.info("Generated table (length: %d)", len(table))
            results = [{"table": table, "entities": entities}]

            if out_format == "jsonl" and output:
                _output_jsonl(results, output)
            elif out_format == "csv" and output:
                _output_csv(results, output)
            else:
                _output_tsv(results, output)

        if emit_entities:
            for entity in entities:
                label = entity.get("label")
                txt = entity.get("text")
                score = entity.get("score", "n/a")
                logger.info("Entity: %s -> %s (score=%s)", label, txt, score)

        if dump_entities_json:
            dump_entities_json.parent.mkdir(parents=True, exist_ok=True)
            dump_entities_json.write_text(json.dumps(entities, indent=2), encoding="utf-8")
            logger.info("Entities saved to %s", dump_entities_json)

    except Exception as e:
        logger.error("Error during processing: %s", e, exc_info=True)
        raise


def _run_batch(
    input_texts: List[Tuple[int, str, Optional[str], Dict]],
    labels: List[str],
    output: Optional[Path],
    out_format: str,
    server_url: str,
    gliner_url: Optional[str],
    model: Optional[str],
    gliner_model: str,
    disable_gliner: bool,
    threshold: float,
    gliner_soft_threshold: float,
    max_new_tokens: int,
    request_timeout: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    max_reasoning_tokens: int,
    enable_row_validation: bool,
    row_validation_mode: str,
    concurrency: int,
    flush_every: int,
    api_key: str,
    gliner_api_key: Optional[str],
    pool_size: int,
    max_retries: int,
    backoff_factor: float,
    max_backoff: float,
    prompt: Optional[str],
    emit_entities: bool,
) -> None:
    """Run batch extraction."""
    extractor = AsyncText2Table(
        labels=labels,
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
        request_timeout=request_timeout,
        pool_size=pool_size,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        max_backoff=max_backoff,
        enable_row_validation=enable_row_validation,
        row_validation_mode=row_validation_mode,
    )

    batch_items = [
        BatchItem(text=text_value, id=record_id, index=idx)
        for idx, text_value, record_id, _ in input_texts
    ]

    # Initialize output file if JSONL
    if output and out_format == "jsonl":
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")
        logger.info("Streaming results to %s (flush every %d rows)", output, flush_every)

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
                    for entity in result.entities:
                        logger.info(
                            "Entity: %s -> %s",
                            entity.get("label"),
                            entity.get("text"),
                        )
            else:
                logger.warning(
                    "Record %s (index %d) failed: %s",
                    result.id,
                    result.index,
                    result.error,
                )

            if output and out_format == "jsonl":
                buffer.append(result)
                if len(buffer) >= flush_every:
                    _append_jsonl_batch(output, buffer)
                    buffer.clear()
            elif not output:
                # Print to stdout
                header = f"Record {result.index}"
                if result.id is not None:
                    header += f" (id={result.id})"
                click.echo(header)
                if result.status == "ok":
                    click.echo(result.table or "")
                else:
                    click.echo(f"ERROR: {result.error}", err=True)
                click.echo()

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
            # Flush remaining buffer
            if output and out_format == "jsonl" and buffer:
                _append_jsonl_batch(output, buffer)
            await extractor.close()
            logger.info("Processed %d records", processed)

    asyncio.run(_run_all())

    # For non-JSONL output in batch mode, collect all results first
    # This is handled via streaming above for JSONL


# Legacy aliases for backward compatibility
cli = main


if __name__ == "__main__":
    main()
