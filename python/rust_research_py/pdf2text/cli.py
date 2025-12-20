"""
Command line interface for the simplified pdf2text toolkit.

Supported workflows:
- Convert PDFs to structured JSON (and optional Markdown) using GROBID
- Extract figures/tables with scipdf
- Render existing JSON outputs into Markdown
- Manage a local GROBID server
"""

import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import click
import requests

CLI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CLI_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rust_research_py.pdf2text.grobid import GrobidServer, ensure_grobid_server
from rust_research_py.pdf2text.pdf2text import (
    extract_fulltext,
    list_pdfs,
    save_markdown_from_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group(
    name="pdf2text",
    help="PDF to structured text and Markdown utilities.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
def cli() -> None:
    """CLI entrypoint."""


@cli.command(name="pdf", help="Extract structured JSON (and optional Markdown) from PDFs.")
@click.option(
    "--pdf-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Directory containing PDF files.",
)
@click.option(
    "--pdf-file",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Single PDF file to process.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Output directory for extracted content.",
)
@click.option(
    "--grobid-url",
    default=None,
    help="GROBID server URL (e.g., http://localhost:8070). If omitted, will auto-start or use the public endpoint.",
)
@click.option(
    "--no-auto-start",
    is_flag=True,
    help="Do not attempt to auto-start a local GROBID server when no URL is provided.",
)
@click.option(
    "--no-figures",
    is_flag=True,
    help="Do not extract figures into the bundle.",
    show_default=True,
)
@click.option(
    "--no-tables",
    is_flag=True,
    help="Do not extract tables into the bundle.",
    show_default=True,
)
@click.option(
    "--copy-pdf",
    is_flag=True,
    help="Copy the source PDF into the output bundle.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing outputs.",
)
@click.option(
    "--no-markdown",
    is_flag=True,
    help="Do not render Markdown alongside JSON output.",
)
def pdf_command(
    pdf_dir: Optional[Path],
    pdf_file: Optional[Path],
    output_dir: Path,
    grobid_url: Optional[str],
    no_auto_start: bool,
    no_figures: bool,
    no_tables: bool,
    copy_pdf: bool,
    overwrite: bool,
    no_markdown: bool,
) -> None:
    """Convert PDFs to JSON/Markdown outputs."""
    if not pdf_dir and not pdf_file:
        pdf_dir = Path(".")
        logger.info("No input specified, defaulting to current directory.")
    if pdf_dir and pdf_file:
        raise click.UsageError("Specify either --pdf-dir or --pdf-file, not both.")

    pdfs = list_pdfs(pdf_dir) if pdf_dir else [pdf_file]
    if not pdfs:
        raise click.ClickException(f"No PDF files found in {pdf_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    grobid_endpoint = grobid_url
    auto_start = not no_auto_start
    if grobid_endpoint is None and auto_start:
        grobid_endpoint = ensure_grobid_server()
    elif grobid_endpoint is None:
        grobid_endpoint = "https://kermitt2-grobid.hf.space"
        logger.info(
            "Using public GROBID service: %s. Provide --grobid-url to use your own instance.",
            grobid_endpoint,
        )

    successful = 0
    failed = []

    for pdf in pdfs:
        try:
            json_result = extract_fulltext(
                pdf,
                output_dir,
                grobid_url=grobid_endpoint,
                auto_start_grobid=False,
                overwrite=overwrite,
                generate_markdown=not no_markdown,
                copy_pdf=copy_pdf,
                extract_figures=not no_figures,
                extract_tables=not no_tables,
            )

            if json_result:
                successful += 1
        except Exception as exc:
            logger.error("Error processing %s: %s", pdf, exc)
            failed.append((pdf, str(exc)))

    logger.info("Processing complete. Successful: %d | Failed: %d", successful, len(failed))
    if failed:
        for pdf, err in failed:
            logger.error("  %s -> %s", pdf, err)


def _iter_json_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise click.ClickException("Input file must be a JSON produced by pdf2text.")
        return [path]
    return sorted(path.rglob("*.json"))


@cli.command(
    name="markdown",
    help="Render Markdown from existing pdf2text JSON outputs.",
)
@click.argument(
    "input_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    help="Optional directory for Markdown output. Defaults to the JSON file directory.",
)
def markdown_command(input_path: Path, output_dir: Optional[Path]) -> None:
    """Convert extracted JSON files to Markdown."""
    json_files = list(_iter_json_files(input_path))
    if not json_files:
        raise click.ClickException(f"No JSON files found under {input_path}")

    base_is_file = input_path.is_file()
    created = 0
    for json_file in json_files:
        md_target = None
        if output_dir:
            if base_is_file:
                md_target = output_dir / json_file.with_suffix(".md").name
            else:
                md_target = output_dir / json_file.relative_to(input_path).with_suffix(".md")
        save_markdown_from_json(json_file, md_target)
        created += 1

    logger.info("Generated %d Markdown file(s).", created)


@cli.group(name="grobid", help="GROBID server management commands.")
def grobid_group() -> None:
    """Container-backed GROBID server helpers."""


@grobid_group.command(name="start", help="Start a GROBID server.")
@click.option("--port", "-p", default=8070, type=click.IntRange(min=1024, max=65535), show_default=True)
@click.option("--host", "-h", default="0.0.0.0", show_default=True)
@click.option("--memory", "-m", default="4g", show_default=True, help="Java heap memory allocation.")
@click.option(
    "--runtime",
    type=click.Choice(["auto", "singularity", "apptainer", "podman", "docker", "java"]),
    default="docker",
    show_default=True,
)
@click.option("--image", default="grobid/grobid:0.8.0", show_default=True, help="Container image to use.")
def grobid_start(port: int, host: str, memory: str, runtime: str, image: str) -> None:
    """Start a local GROBID instance."""
    try:
        from rust_research_py.pdf2text.grobid import ContainerRuntime

        if runtime == "auto":
            runtime_enum = ContainerRuntime.detect()
            if runtime_enum == ContainerRuntime.NONE:
                raise click.ClickException(
                    "No container runtime found. Please install Singularity, Podman, or Docker."
                )
        else:
            runtime_map = {
                "singularity": ContainerRuntime.SINGULARITY,
                "apptainer": ContainerRuntime.APPTAINER,
                "podman": ContainerRuntime.PODMAN,
                "docker": ContainerRuntime.DOCKER,
                "java": ContainerRuntime.NONE,
            }
            runtime_enum = runtime_map[runtime]

        server = GrobidServer(
            port=port,
            host=host,
            memory=memory,
            runtime=runtime_enum,
            container_image=image,
        )

        logger.info("Starting GROBID server with %s...", runtime_enum.value)
        if server.start_server():
            url = f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}"
            click.secho(f"✓ GROBID server started at {url}", fg="green")
            click.echo(f"  Health check: {url}/api/isalive")
            click.echo(f"  API docs: {url}/api")
            click.echo("\nPress Ctrl+C to stop the server...")

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                click.echo("\nStopping GROBID server...")
                server.stop_server()
                click.secho("✓ Server stopped", fg="green")
        else:
            raise click.ClickException("Failed to start GROBID server")
    except Exception as exc:
        logger.error("Error: %s", exc)
        raise click.ClickException(str(exc))


@grobid_group.command(name="stop", help="Stop a running GROBID server.")
@click.option("--port", "-p", default=8070, type=int, show_default=True)
def grobid_stop(port: int) -> None:
    """Stop a local GROBID container."""
    try:
        server = GrobidServer(port=port)
        if server.stop_server():
            click.secho(f"✓ GROBID server on port {port} stopped", fg="green")
        else:
            click.echo(f"No GROBID server found on port {port}")
    except Exception as exc:
        logger.error("Error: %s", exc)
        raise click.ClickException(str(exc))


@grobid_group.command(name="status", help="Check GROBID server status.")
@click.option("--port", "-p", default=8070, type=int, show_default=True)
@click.option("--host", default="localhost", show_default=True)
def grobid_status(port: int, host: str) -> None:
    """Check whether a GROBID server is reachable."""
    try:
        url = f"http://{host}:{port}"
        try:
            response = requests.get(f"{url}/api/isalive", timeout=5)
            if response.text.strip() == "true":
                click.secho(f"✓ GROBID server is running at {url}", fg="green")
                try:
                    version_response = requests.get(f"{url}/api/version", timeout=5)
                    click.echo(f"  Version: {version_response.text.strip()}")
                except Exception:
                    pass
            else:
                click.echo(f"✗ GROBID server at {url} is not responding correctly", fg="yellow")
        except requests.RequestException:
            click.echo(f"✗ No GROBID server found at {url}", fg="red")
    except Exception as exc:
        logger.error("Error: %s", exc)
        raise click.ClickException(str(exc))


@grobid_group.command(name="list-images", help="List cached container images.")
def grobid_list_images() -> None:
    """List cached container images for GROBID."""
    try:
        cache_dirs = [
            Path.home() / ".cache" / "biomedgps" / "grobid",
            Path.home() / ".singularity" / "cache",
            Path.home() / ".apptainer" / "cache",
        ]

        sif_files = []
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                sif_files.extend(cache_dir.glob("*.sif"))

        if sif_files:
            click.echo("Singularity/Apptainer images:")
            for sif in sif_files:
                size = sif.stat().st_size / (1024 * 1024)
                click.echo(f"  - {sif.name} ({size:.1f} MB)")

        for runtime_cmd in ("docker", "podman"):
            if shutil.which(runtime_cmd):
                result = subprocess.run(
                    [
                        runtime_cmd,
                        "images",
                        "--filter",
                        "reference=*grobid*",
                        "--format",
                        "table {{.Repository}}:{{.Tag}}\t{{.Size}}",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    click.echo(f"\n{runtime_cmd.capitalize()} images:")
                    for line in result.stdout.strip().split("\n"):
                        if line and not line.startswith("REPOSITORY"):
                            click.echo(f"  - {line}")

        if not sif_files:
            click.echo("No cached images found.")
    except Exception as exc:
        logger.error("Error: %s", exc)
        raise click.ClickException(str(exc))


@grobid_group.command(name="clean", help="Remove cached container images.")
@click.option("--force", "-f", is_flag=True, help="Remove without confirmation.")
def grobid_clean(force: bool) -> None:
    """Remove cached GROBID container images."""
    try:
        cache_dirs = [
            Path.home() / ".cache" / "biomedgps" / "grobid",
            Path.home() / ".grobid",
        ]

        files_to_remove = []
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                if cache_dir.name == "grobid":
                    files_to_remove.extend(cache_dir.glob("*.sif"))
                else:
                    files_to_remove.append(cache_dir)

        if not files_to_remove:
            click.echo("No cached files found.")
            return

        click.echo("Files/directories to remove:")
        total_size = 0.0
        for item in files_to_remove:
            if item.is_file():
                size = item.stat().st_size / (1024 * 1024)
                total_size += size
                click.echo(f"  - {item} ({size:.1f} MB)")
            else:
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) / (1024 * 1024)
                total_size += size
                click.echo(f"  - {item}/ ({size:.1f} MB)")

        click.echo(f"\nTotal size: {total_size:.1f} MB")

        if not force:
            if not click.confirm("Do you want to remove these files?"):
                click.echo("Cancelled.")
                return

        for item in files_to_remove:
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

        click.secho(f"✓ Removed {len(files_to_remove)} items ({total_size:.1f} MB)", fg="green")
    except Exception as exc:
        logger.error("Error: %s", exc)
        raise click.ClickException(str(exc))


if __name__ == "__main__":
    cli()
