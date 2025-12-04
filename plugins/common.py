"""Shared plugin primitives for publisher-specific downloaders."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DownloadResult:
    """Result of a PDF download operation."""

    success: bool
    file_path: Optional[Path] = None
    file_size: int = 0
    error: Optional[str] = None
    doi: Optional[str] = None
    publisher: Optional[str] = None


class BasePlugin(abc.ABC):
    """Base class for publisher-specific download plugins."""

    publisher: str

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    @abc.abstractmethod
    def build_download_url(self, doi: str) -> Optional[str]:
        """Return a direct download URL if this plugin supports the DOI."""

    @abc.abstractmethod
    async def download(
        self,
        url: str,
        output_dir: str | Path = ".",
        filename: Optional[str] = None,
        wait_time: float = 5.0,
        doi: Optional[str] = None,
    ) -> DownloadResult:
        """Download a PDF using a prepared URL."""
