"""
Command line interface for the text2table pipeline.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import click

CLI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLI_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from text2table import DEFAULT_USER_PROMPT, Text2Table  # noqa: E402

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


@click.group(
    name="text2table",
    help="Extract entities with GLiNER and render tables with Qwen.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli() -> None:
    """CLI entrypoint."""


@cli.command(
    name="run",
    help="Extract entities from text and render a Markdown table.",
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
    help="Confidence threshold for GLiNER.",
)
@click.option(
    "--gliner-model",
    default="Ihor/gliner-biomed-large-v1.0",
    show_default=True,
    help="GLiNER model name or path.",
)
@click.option(
    "--qwen-model",
    default="Qwen/Qwen3-30B-A3B-Instruct-2507",
    show_default=True,
    help="Qwen model name or path.",
)
@click.option(
    "--cache-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Optional cache directory for model weights.",
)
@click.option(
    "--device",
    default=None,
    help="Target device for inference (cpu, cuda, or auto). Defaults to auto selection.",
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
    help="Optional path to write the generated table (Markdown).",
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
def run_command(
    text: Optional[str],
    text_file: Optional[Path],
    labels: Sequence[str],
    labels_file: Optional[Path],
    prompt: Optional[str],
    threshold: float,
    gliner_model: str,
    qwen_model: str,
    cache_dir: Optional[Path],
    device: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output: Optional[Path],
    emit_entities: bool,
    dump_entities_json: Optional[Path],
    enable_thinking: bool,
) -> None:
    labels_list = _load_labels(labels, labels_file)
    source_text = _read_text(text, text_file)

    extractor = Text2Table(
        labels=labels_list,
        gliner_model_name=gliner_model,
        qwen_model_name=qwen_model,
        threshold=threshold,
        cache_dir=cache_dir,
        device=device,
        enable_thinking=enable_thinking,
    )

    if enable_thinking:
        thinking, table, entities = extractor.run_with_thinking(
            source_text,
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
            source_text,
            user_prompt=prompt or DEFAULT_USER_PROMPT,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    click.echo(table)

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


if __name__ == "__main__":
    cli()
