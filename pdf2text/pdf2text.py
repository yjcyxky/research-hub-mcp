"""
Lightweight PDF extraction helpers.

This module focuses on converting PDFs into structured JSON using GROBID,
optionally extracting figures/tables, and rendering the extracted structure
into Markdown files that sit next to the JSON output. Paths stored in JSON
and Markdown are normalized to be relative to the output bundle.
"""

import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pdf2text.grobid import GrobidServer, ensure_grobid_server

logger = logging.getLogger(__name__)


class PDFListConfig(BaseModel):
    """Configuration for listing PDF files."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    path: Path = Field(description="Directory path to search for PDFs")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: Path) -> Path:
        resolved = value.resolve()
        if not resolved.exists():
            raise ValueError(f"Path does not exist: {resolved}")
        if not resolved.is_dir():
            raise ValueError(f"Path is not a directory: {resolved}")
        return resolved


class ExtractionConfig(BaseModel):
    """Configuration for PDF extraction."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    pdf_file: Path = Field(description="PDF file path")
    output_dir: Path = Field(description="Output directory")
    grobid_url: Optional[str] = Field(default=None, description="GROBID server URL")
    auto_start_grobid: bool = Field(
        default=True, description="Auto-start a local GROBID server when needed"
    )
    overwrite: bool = Field(default=False, description="Overwrite existing files")
    generate_markdown: bool = Field(
        default=True, description="Render extracted JSON into Markdown"
    )
    copy_pdf: bool = Field(
        default=False, description="Copy the source PDF into the output bundle"
    )
    extract_figures: bool = Field(
        default=False, description="Extract figures into a figures/ folder"
    )
    extract_tables: bool = Field(
        default=False, description="Extract tables into a tables/ folder"
    )

    @field_validator("pdf_file")
    @classmethod
    def validate_pdf_file(cls, value: Path) -> Path:
        resolved = value.resolve()
        if not resolved.exists():
            raise ValueError(f"PDF file does not exist: {resolved}")
        if resolved.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {resolved}")
        return resolved

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, value: Path) -> Path:
        resolved = value.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


def list_pdfs(path: Union[str, Path]) -> List[Path]:
    """
    Recursively list all PDF files under the provided directory.
    """
    config = PDFListConfig(path=Path(path))
    pdfs: List[Path] = []

    for root, dirs, files in os.walk(config.path):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            if file.lower().endswith(".pdf") and not file.startswith("."):
                pdfs.append(Path(root) / file)

    logger.info("Found %d PDF files in %s", len(pdfs), config.path)
    return pdfs


def gen_dest_path(pdf_file: Union[str, Path], output_dir: Union[str, Path]) -> Tuple[Path, str]:
    """
    Build the destination directory and stem for a PDF output bundle.
    """
    pdf_path = Path(pdf_file)
    output_path = Path(output_dir)
    basename = pdf_path.stem
    return output_path / basename, basename


def _format_authors(authors: Any) -> Optional[str]:
    """Best-effort formatting for author lists from scipdf output."""
    if not authors:
        return None

    names: List[str] = []
    if isinstance(authors, str):
        names.append(authors.strip())
    elif isinstance(authors, list):
        for author in authors:
            name: Optional[str] = None
            if isinstance(author, str):
                name = author.strip()
            elif isinstance(author, dict):
                # scipdf can return {"name": "..."} or split name parts
                if author.get("name"):
                    name = str(author["name"]).strip()
                else:
                    parts = [
                        author.get("first") or author.get("given") or author.get("forename"),
                        author.get("middle"),
                        author.get("last") or author.get("surname") or author.get("family"),
                    ]
                    name = " ".join(part for part in parts if part)
            if name:
                names.append(name)

    formatted = ", ".join(n for n in names if n)
    return formatted or None


def _render_table_content(content: Any) -> str:
    """Render table content to Markdown-friendly text."""
    if content is None:
        return ""

    if isinstance(content, list):
        if content and all(isinstance(row, (list, tuple)) for row in content):
            header = [str(cell) for cell in content[0]]
            rows = [header]
            for row in content[1:]:
                rows.append([str(cell) for cell in row])

            separator = "| " + " | ".join("---" for _ in header) + " |"
            formatted_rows = ["| " + " | ".join(header) + " |", separator]
            for row in rows[1:]:
                formatted_rows.append("| " + " | ".join(row) + " |")
            return "\n".join(formatted_rows)

        return "\n".join(str(item) for item in content)

    if isinstance(content, dict):
        return json.dumps(content, indent=2, ensure_ascii=False)

    return str(content)


def build_markdown_content(article: Dict[str, Any]) -> str:
    """
    Create a Markdown representation from the scipdf JSON structure.
    """
    lines: List[str] = []

    title = article.get("title")
    if title:
        lines.append(f"# {title.strip()}")

    metadata_parts: List[str] = []
    authors_line = _format_authors(article.get("authors"))
    if authors_line:
        metadata_parts.append(f"**Authors:** {authors_line}")
    if article.get("doi"):
        metadata_parts.append(f"**DOI:** {article['doi']}")
    if article.get("pmid"):
        metadata_parts.append(f"**PMID:** {article['pmid']}")
    if metadata_parts:
        lines.append(" | ".join(metadata_parts))

    abstract = article.get("abstract")
    if abstract:
        lines.append("## Abstract")
        lines.append(str(abstract).strip())

    sections = article.get("sections", []) or []
    for idx, section in enumerate(sections, start=1):
        heading = section.get("heading") or f"Section {idx}"
        text = (section.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"## {heading}")
        lines.append(text)

    figures = article.get("figures", []) or []
    if figures:
        lines.append("## Figures")
        for idx, figure in enumerate(figures, start=1):
            label = (
                figure.get("label")
                or figure.get("name")
                or figure.get("id")
                or f"Figure {idx}"
            )
            caption = (figure.get("caption") or figure.get("description") or "").strip()
            page = figure.get("page")
            file_path = figure.get("file") or figure.get("file_path") or figure.get("path")

            lines.append(f"### {label}")
            if file_path:
                lines.append(f"![{label}]({Path(file_path).as_posix()})")
            if caption:
                lines.append(caption)
            meta_parts = []
            if page:
                meta_parts.append(f"Page {page}")
            if file_path:
                meta_parts.append(f"File: {file_path}")
            if meta_parts:
                lines.append(f"_({' | '.join(meta_parts)})_")

    tables = article.get("tables", []) or []
    if tables:
        lines.append("## Tables")
        for idx, table in enumerate(tables, start=1):
            label = (
                table.get("label")
                or table.get("name")
                or table.get("id")
                or f"Table {idx}"
            )
            caption = (table.get("caption") or "").strip()
            content = table.get("content") or table.get("text")
            page = table.get("page")
            table_path = table.get("file") or table.get("file_path") or table.get("path")

            lines.append(f"### {label}")
            if table_path:
                lines.append(f"![{label}]({Path(table_path).as_posix()})")
            if caption:
                lines.append(caption)
            if content:
                lines.append(_render_table_content(content))
            if page:
                lines.append(f"_Page {page}_")

    markdown = "\n\n".join(lines).strip()
    return f"{markdown}\n" if markdown else ""


def save_markdown_from_json(
    json_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Generate a Markdown file from an extracted JSON file.
    """
    json_path = Path(json_file).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as handle:
        article = json.load(handle)

    markdown = build_markdown_content(article)

    md_path = Path(output_file) if output_file else json_path.with_suffix(".md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)

    logger.info("Saved Markdown to %s", md_path)
    return md_path


def _relativize_path(path: Union[str, Path], base: Path) -> Optional[str]:
    """Return path relative to base if possible."""
    if not path:
        return None
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        if not path_obj.is_absolute():
            return path_obj.as_posix()
    return None


def _merge_asset_data(
    article: Dict[str, Any],
    asset_data: Dict[str, Any],
    output_path: Path,
    include_figures: bool = True,
    include_tables: bool = True,
) -> None:
    """Merge figure/table metadata into the article with relative paths."""
    figures_meta = asset_data.get("figures") or []
    tables_meta = asset_data.get("tables") or []

    figure_files = asset_data.get("figure_files") or []
    table_files = asset_data.get("table_files") or []

    merged_figures: List[Dict[str, Any]] = []
    if include_figures:
        source_figures = article.get("figures") or figures_meta
        for idx, figure in enumerate(source_figures):
            entry = dict(figure)
            path_candidate = (
                entry.get("file")
                or entry.get("file_path")
                or entry.get("path")
                or entry.get("url")
            )
            if not path_candidate and idx < len(figure_files):
                path_candidate = figure_files[idx]

            rel_path = _relativize_path(path_candidate, output_path) if path_candidate else None
            if rel_path:
                entry["file_path"] = rel_path
            merged_figures.append(entry)

        if not merged_figures and figure_files:
            merged_figures = [
                {"label": f"Figure {idx + 1}", "file_path": _relativize_path(path, output_path)}
                for idx, path in enumerate(figure_files)
            ]

    merged_tables: List[Dict[str, Any]] = []
    if include_tables:
        source_tables = article.get("tables") or tables_meta
        for idx, table in enumerate(source_tables):
            entry = dict(table)
            path_candidate = (
                entry.get("file")
                or entry.get("file_path")
                or entry.get("path")
                or entry.get("url")
            )
            if not path_candidate and idx < len(table_files):
                path_candidate = table_files[idx]

            rel_path = _relativize_path(path_candidate, output_path) if path_candidate else None
            if rel_path:
                entry["file_path"] = rel_path
            merged_tables.append(entry)

        if not merged_tables and table_files:
            merged_tables = [
                {"label": f"Table {idx + 1}", "file_path": _relativize_path(path, output_path)}
                for idx, path in enumerate(table_files)
            ]

    if merged_figures:
        article["figures"] = merged_figures
    if merged_tables:
        article["tables"] = merged_tables


def extract_fulltext(
    pdf_file: Union[str, Path],
    output_dir: Union[str, Path],
    grobid_url: Optional[str] = None,
    auto_start_grobid: bool = True,
    overwrite: bool = False,
    generate_markdown: bool = True,
    copy_pdf: bool = False,
    extract_figures: bool = False,
    extract_tables: bool = False,
) -> Optional[Path]:
    """
    Extract structured text from a PDF and optionally render Markdown.
    """
    config = ExtractionConfig(
        pdf_file=Path(pdf_file),
        output_dir=Path(output_dir),
        grobid_url=grobid_url,
        auto_start_grobid=auto_start_grobid,
        overwrite=overwrite,
        generate_markdown=generate_markdown,
        copy_pdf=copy_pdf,
        extract_figures=extract_figures,
        extract_tables=extract_tables,
    )

    if config.grobid_url is None and config.auto_start_grobid:
        config.grobid_url = ensure_grobid_server()
    elif config.grobid_url is None:
        config.grobid_url = "https://kermitt2-grobid.hf.space"
        logger.info("Using public GROBID endpoint: %s", config.grobid_url)

    output_path, basename = gen_dest_path(config.pdf_file, config.output_dir)
    json_output = output_path / f"{basename}.json"
    md_output = output_path / f"{basename}.md"
    output_path.mkdir(parents=True, exist_ok=True)

    if json_output.exists() and not config.overwrite:
        logger.info("Output file already exists, skipping: %s", json_output)
        return None

    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            warnings.filterwarnings("ignore", message=".*Implicitly cleaning up.*")
            warnings.filterwarnings("ignore", message=".*unclosed.*")
            import scipdf

        logger.info(
            "Processing PDF: %s with GROBID URL: %s", config.pdf_file, config.grobid_url
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            warnings.filterwarnings("ignore", message=".*Implicitly cleaning up.*")
            article_dict = scipdf.parse_pdf_to_dict(
                str(config.pdf_file), grobid_url=config.grobid_url
            )

        asset_data: Dict[str, Any] = {}
        if config.extract_figures or config.extract_tables or config.copy_pdf:
            asset_data = extract_assets(
                config.pdf_file,
                output_path,
                basename,
                overwrite=config.overwrite,
                copy_pdf=config.copy_pdf,
                run_extraction=(config.extract_figures or config.extract_tables),
            )
            _merge_asset_data(
                article_dict,
                asset_data,
                output_path,
                include_figures=config.extract_figures,
                include_tables=config.extract_tables,
            )
            if config.copy_pdf:
                article_dict["pdf_file"] = f"{basename}.pdf"

        with open(json_output, "w", encoding="utf-8") as handle:
            json.dump(article_dict, handle, indent=4, ensure_ascii=False)

        if config.generate_markdown:
            markdown = build_markdown_content(article_dict)
            md_output.parent.mkdir(parents=True, exist_ok=True)
            with open(md_output, "w", encoding="utf-8") as handle:
                handle.write(markdown)
            logger.info("Saved Markdown to %s", md_output)

        logger.info("Successfully extracted to: %s", json_output)
        return json_output

    except Exception as exc:
        logger.error("Error processing %s: %s", config.pdf_file, exc)
        return None


def extract_assets(
    pdf_file: Union[str, Path],
    output_path: Path,
    basename: str,
    overwrite: bool = False,
    copy_pdf: bool = False,
    run_extraction: bool = True,
) -> Dict[str, Any]:
    """
    Extract figures and tables using scipdf.parse_figures.

    Returns metadata about the extracted assets (figure/table lists and file paths).
    """
    import warnings

    pdf_path = Path(pdf_file).resolve()
    asset_info: Dict[str, Any] = {}

    data_dir = output_path / "data"
    figures_dir = output_path / "figures"
    tables_dir = output_path / "tables"
    datafile = data_dir / f"{basename}.json"

    if run_extraction:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            warnings.filterwarnings("ignore", message=".*Implicitly cleaning up.*")
            warnings.filterwarnings("ignore", message=".*unclosed.*")
            import scipdf

        if datafile.exists() and not overwrite:
            logger.info("Figures/tables already extracted, skipping: %s", datafile)
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            figures_dir.mkdir(parents=True, exist_ok=True)
            tables_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                temp_pdf_dir = Path(tmpdir)
                temp_pdf = temp_pdf_dir / f"{basename}.pdf"
                shutil.copy2(pdf_path, temp_pdf)

                logger.info("Extracting figures/tables from %s to %s", pdf_path, output_path)
                scipdf.parse_figures(str(temp_pdf_dir), output_folder=str(output_path))

    if copy_pdf:
        copied_pdf = output_path / f"{basename}.pdf"
        if not copied_pdf.exists() or overwrite:
            shutil.copy2(pdf_path, copied_pdf)
        asset_info["pdf_file"] = copied_pdf

    if datafile.exists():
        try:
            with open(datafile, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
                asset_info.update(meta if isinstance(meta, dict) else {})
        except Exception as exc:
            logger.warning("Failed to load figure/table metadata from %s: %s", datafile, exc)

    figure_files = []
    if figures_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
            figure_files.extend(sorted(figures_dir.rglob(ext)))
    table_files = []
    if tables_dir.exists():
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
            table_files.extend(sorted(tables_dir.rglob(ext)))

    asset_info["figure_files"] = figure_files
    asset_info["table_files"] = table_files

    return asset_info


def extract_figures(
    pdf_file: Union[str, Path],
    output_dir: Union[str, Path],
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Public helper to extract figures/tables into the expected bundle layout.
    """
    pdf_path = Path(pdf_file).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    output_path, basename = gen_dest_path(pdf_path, output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    asset_data = extract_assets(
        pdf_path,
        output_path,
        basename,
        overwrite=overwrite,
        copy_pdf=False,
    )
    return output_path


__all__ = [
    "GrobidServer",
    "build_markdown_content",
    "extract_figures",
    "extract_fulltext",
    "gen_dest_path",
    "list_pdfs",
    "save_markdown_from_json",
]
