"""
Shared configuration for text2table operations.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Text2TableConfig:
    """Shared configuration for text2table operations."""

    # Required
    server_url: str = ""

    # Labels
    labels: List[str] = field(default_factory=list)

    # GLiNER settings
    gliner_url: Optional[str] = None
    gliner_model: str = "Ihor/gliner-biomed-large-v1.0"
    gliner_api_key: Optional[str] = None
    gliner_cache_dir: Optional[Path] = None
    gliner_device: Optional[str] = None
    use_gliner: bool = True
    threshold: float = 0.5
    gliner_soft_threshold: float = 0.3

    # LLM settings
    model: Optional[str] = None
    api_key: str = "dummy-key"
    max_new_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.9

    # Thinking mode
    enable_thinking: bool = False
    max_reasoning_tokens: int = 2048

    # Row validation
    enable_row_validation: bool = False
    row_validation_mode: str = "substring"

    # HTTP client settings
    request_timeout: int = 600
    pool_size: int = 10
    max_retries: int = 3
    backoff_factor: float = 1.5
    max_backoff: float = 10.0

    # Batch settings
    concurrency: int = 4
    flush_every: int = 20

    @classmethod
    def from_env(cls) -> "Text2TableConfig":
        """Load config from environment variables."""
        return cls(
            server_url=os.environ.get("TEXT2TABLE_VLLM_URL", ""),
            gliner_url=os.environ.get("TEXT2TABLE_GLINER_URL"),
        )


def detect_input_format(path: Path) -> str:
    """Detect input format from file extension.

    Returns:
        'single' for .txt files
        'batch' for .tsv, .csv, .jsonl files
    """
    ext = path.suffix.lower()
    if ext == ".txt":
        return "single"
    elif ext in (".tsv", ".csv", ".jsonl"):
        return "batch"
    else:
        raise ValueError(f"Unsupported input format: {ext}. Supported: .txt, .tsv, .csv, .jsonl")


def detect_output_format(path: Optional[Path]) -> str:
    """Detect output format from file extension.

    Returns:
        'tsv' for .tsv or no extension
        'csv' for .csv
        'jsonl' for .jsonl
    """
    if path is None:
        return "tsv"  # stdout default
    ext = path.suffix.lower()
    if ext == ".jsonl":
        return "jsonl"
    elif ext == ".csv":
        return "csv"
    else:
        return "tsv"  # default for .tsv and unknown
