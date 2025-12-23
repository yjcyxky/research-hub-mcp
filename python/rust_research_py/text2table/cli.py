"""
Unified command line interface for the text2table pipeline.

All input is batch processing from TSV/JSON/JSONL files.
Input records are converted to text via key:value format for LLM processing.
Output tables use user-specified labels as headers.
"""

import asyncio
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def _format_record_as_text(record: Dict[str, Any]) -> str:
    """Convert a record dict to key:value text format for LLM processing."""
    parts: List[str] = []
    for key, value in record.items():
        if value is not None and str(value).strip():
            parts.append(f"{key}: {value}")
    return "\n".join(parts)


def _detect_input_format(path: Path) -> str:
    """Detect input format from file extension."""
    ext = path.suffix.lower()
    if ext == ".tsv":
        return "tsv"
    elif ext == ".csv":
        return "csv"
    elif ext in (".json", ".jsonl"):
        return "jsonl"
    else:
        raise ValueError(f"Unsupported input format: {ext}. Supported: .tsv, .csv, .json, .jsonl")


def _detect_output_format(path: Optional[Path]) -> str:
    """Detect output format from file extension."""
    if path is None:
        return "tsv"  # stdout default
    ext = path.suffix.lower()
    if ext in (".json", ".jsonl"):
        return "jsonl"
    elif ext == ".csv":
        return "csv"
    else:
        return "tsv"


def _load_records(
    path: Path,
    input_format: str,
    text_column: Optional[str],
    id_column: Optional[str],
) -> List[Dict[str, Any]]:
    """Load records from input file. Returns list of dicts with 'id', 'text', and 'original' fields."""
    records: List[Dict[str, Any]] = []

    if input_format == "tsv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                record_id = row.get(id_column) if id_column else str(idx)
                # Use text_column if specified, else convert all columns to key:value format
                if text_column and text_column in row:
                    text = row[text_column].strip()
                else:
                    text = _format_record_as_text(row)
                if text:
                    records.append({
                        "id": record_id,
                        "text": text,
                        "original": row,
                    })
    elif input_format == "csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                record_id = row.get(id_column) if id_column else str(idx)
                if text_column and text_column in row:
                    text = row[text_column].strip()
                else:
                    text = _format_record_as_text(row)
                if text:
                    records.append({
                        "id": record_id,
                        "text": text,
                        "original": row,
                    })
    elif input_format == "jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise click.UsageError(f"Line {idx + 1}: invalid JSON: {exc}")

                record_id = payload.get(id_column) if id_column else str(idx)
                # Use text_column if specified and exists, else convert entire record to text
                if text_column and text_column in payload:
                    text = str(payload[text_column]).strip()
                else:
                    text = _format_record_as_text(payload)

                if text:
                    records.append({
                        "id": record_id,
                        "text": text,
                        "original": payload,
                    })

    return records


def _output_results_tsv(results: List[Dict[str, Any]], output: Optional[Path]) -> None:
    """Output results as TSV (table only)."""
    lines = []
    for r in results:
        if r.get("table"):
            lines.append(r["table"])

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Results saved to %s", output)
    else:
        for line in lines:
            click.echo(line)


def _output_results_csv(results: List[Dict[str, Any]], output: Path) -> None:
    """Output results as CSV (table only)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in results:
        if r.get("table"):
            lines.append(r["table"].replace("\t", ","))
    output.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Results saved to %s", output)


def _output_results_jsonl(results: List[Dict[str, Any]], output: Optional[Path]) -> None:
    """Output results as JSONL with full details."""
    lines = []
    for r in results:
        # JSONL includes detailed info: table, source text, entities, etc.
        payload = {
            "id": r.get("id"),
            "status": r.get("status", "ok"),
            "source_text": r.get("text"),
            "table": r.get("table"),
            "entities": r.get("entities"),
            "thinking": r.get("thinking"),
            "original": r.get("original"),
            "error": r.get("error"),
        }
        lines.append(json.dumps(payload, ensure_ascii=False))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Results saved to %s", output)
    else:
        for line in lines:
            click.echo(line)


@click.command()
@click.argument(
    "input_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
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
    default=None,
    help="Column containing text (if not specified, all columns are used as key:value pairs)",
)
@click.option(
    "--id-column",
    default=None,
    help="Column for record ID",
)
@click.option(
    "--concurrency",
    default=4,
    type=int,
    show_default=True,
    help="Concurrent requests",
)
@click.option(
    "--flush-every",
    default=20,
    type=int,
    show_default=True,
    help="Flush to disk interval",
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
def main(
    input_file: Path,
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
    text_column: Optional[str],
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
) -> None:
    """Extract structured tables from records in TSV/JSON/JSONL files.

    All input is batch processing (1 to N records).
    Input records are converted to text via key:value format.
    Output tables use user-specified labels as headers.

    INPUT_FILE must be:
    - A .tsv/.csv file (batch of records)
    - A .json/.jsonl file (batch of records)

    OUTPUT format is determined by extension:
    - .tsv: Tab-separated table output
    - .csv: Comma-separated table output
    - .jsonl: JSON Lines with full details (table, source_text, entities, etc.)

    Examples:
        t2t batch.jsonl --text-column "abstract" -o results.jsonl
        t2t records.tsv -o results.jsonl  # Uses all columns as key:value pairs
        t2t data.csv --label "Gene" --label "Disease" -o output.tsv
    """
    # Detect input format
    try:
        input_format = _detect_input_format(input_file)
    except ValueError as e:
        raise click.UsageError(str(e))

    # Load records from input file
    records = _load_records(input_file, input_format, text_column, id_column)

    if not records:
        logger.warning("No valid records found in input file")
        return

    # Load labels
    labels_list = _load_labels(labels, labels_file)

    # Determine output format
    out_format = _detect_output_format(output)

    # Log configuration
    logger.info("Using vLLM server at: %s", server_url)
    if disable_gliner:
        logger.info("GLiNER disabled; relying on vLLM only")
    elif gliner_url:
        logger.info("Using GLiNER service at: %s", gliner_url)
    else:
        logger.info("Using local GLiNER model: %s", gliner_model)
    logger.info("Processing %d records with concurrency %d", len(records), concurrency)

    # Create extractor
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
        request_timeout=request_timeout,
        pool_size=pool_size,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        max_backoff=max_backoff,
        enable_row_validation=enable_row_validation,
        row_validation_mode=row_validation_mode,
    )

    # Prepare batch items
    batch_items = [
        BatchItem(text=rec["text"], id=rec["id"], index=idx, metadata=rec.get("original"))
        for idx, rec in enumerate(records)
    ]

    # Initialize output file for streaming (JSONL only)
    if output and out_format == "jsonl":
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")

    # Run processing
    all_results: List[Dict[str, Any]] = []

    async def _run_all() -> None:
        buffer: List[Dict[str, Any]] = []
        processed = 0

        async def handle_result(result: BatchResult) -> None:
            nonlocal processed, buffer
            processed += 1

            # Build result dict
            res_dict: Dict[str, Any] = {
                "id": result.id,
                "status": result.status,
                "text": result.text,
                "table": result.table,
                "entities": result.entities,
                "thinking": result.thinking,
                "original": result.metadata,
                "error": result.error,
            }

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

            all_results.append(res_dict)

            # Streaming output for JSONL
            if output and out_format == "jsonl":
                buffer.append(res_dict)
                if len(buffer) >= flush_every:
                    _flush_jsonl_buffer(output, buffer)
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
            # Flush remaining buffer
            if output and out_format == "jsonl" and buffer:
                _flush_jsonl_buffer(output, buffer)
            await extractor.close()
            logger.info("Processed %d records", processed)

    asyncio.run(_run_all())

    # Sort results by original order
    id_to_idx = {rec["id"]: idx for idx, rec in enumerate(records)}
    all_results.sort(key=lambda r: id_to_idx.get(r["id"], 0))

    # Output results (for non-streaming output)
    if out_format == "jsonl":
        if not output:
            # Stdout output for JSONL
            _output_results_jsonl(all_results, None)
    elif out_format == "csv":
        if output:
            _output_results_csv(all_results, output)
    else:
        _output_results_tsv(all_results, output)


def _flush_jsonl_buffer(output_path: Path, buffer: List[Dict[str, Any]]) -> None:
    """Append buffer to JSONL file."""
    if not buffer:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for r in buffer:
            payload = {
                "id": r.get("id"),
                "status": r.get("status", "ok"),
                "source_text": r.get("text"),
                "table": r.get("table"),
                "entities": r.get("entities"),
                "thinking": r.get("thinking"),
                "original": r.get("original"),
                "error": r.get("error"),
            }
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")


# Entry point
cli = main


if __name__ == "__main__":
    main()
