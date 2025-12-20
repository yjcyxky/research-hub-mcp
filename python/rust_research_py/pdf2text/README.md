# PDF2Text Module

Lightweight utilities for turning PDFs into structured JSON, Markdown, and extracted figures/tables with GROBID + scipdf. RAG and embedding helpers have been removed to keep the surface focused on conversion outputs.

## Features
- PDF → structured JSON via GROBID (auto-start support)
- Optional Markdown rendering that mirrors the JSON contents
- Optional copying of the source PDF into the bundle
- Figure/table extraction through scipdf with relative paths recorded in JSON/MD
- CLI helpers plus a small Python API

## Installation

```bash
# Create an environment with Java for GROBID
conda create -n pdf2text python=3.10 openjdk=11
conda activate pdf2text

pip install -e .
pip install scipdf  # required for PDF parsing
```

## Quick Start

```bash
# Convert a directory of PDFs to JSON + Markdown (+figures/tables)
python -m pdf2text.cli pdf --pdf-dir ./pdfs --output-dir ./output

# Skip Markdown or assets when needed
python -m pdf2text.cli pdf --pdf-dir ./pdfs --output-dir ./output --no-markdown --no-figures --no-tables

# Copy the original PDF into the bundle
python -m pdf2text.cli pdf --pdf-dir ./pdfs --output-dir ./output --copy-pdf

# Render Markdown from existing JSON outputs
python -m pdf2text.cli markdown ./output
```

GROBID is auto-started when no URL is provided. To use your own instance:

```bash
python -m pdf2text.cli pdf --pdf-dir ./pdfs --output-dir ./output --grobid-url http://localhost:8070
```

## Python API

```python
from pdf2text import extract_fulltext, extract_figures, save_markdown_from_json

# Extract JSON + Markdown for a single PDF (with figures/tables + PDF copy)
extract_fulltext("paper.pdf", "./output", extract_figures=True, extract_tables=True, copy_pdf=True)

# Extract figures/tables into the same output bundle
extract_figures("paper.pdf", "./output")

# Convert an existing JSON output to Markdown
save_markdown_from_json("./output/paper/paper.json")
```

## GROBID Management

```bash
# Start a local GROBID server (Docker/Podman/Singularity)
python -m pdf2text.cli grobid start

# Check status or stop
python -m pdf2text.cli grobid status
python -m pdf2text.cli grobid stop
```

## Output Structure

```
output/
├── paper/
│   ├── paper.json    # structured text and metadata (paths are relative)
│   ├── paper.md      # Markdown rendering of the JSON (optional)
│   ├── paper.pdf     # optional copy of the source PDF
│   ├── figures/      # extracted figure images (optional)
│   └── tables/       # extracted table images (optional)
```
