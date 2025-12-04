"""
pdf2text

Utilities for turning PDFs into structured JSON, Markdown, and extracted assets.
"""

from pdf2text.grobid import GrobidServer, ensure_grobid_server
from pdf2text.pdf2text import (
    build_markdown_content,
    extract_figures,
    extract_fulltext,
    gen_dest_path,
    list_pdfs,
    save_markdown_from_json,
)

__all__ = [
    "build_markdown_content",
    "extract_figures",
    "extract_fulltext",
    "gen_dest_path",
    "list_pdfs",
    "save_markdown_from_json",
    "GrobidServer",
    "ensure_grobid_server",
]
