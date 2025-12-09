"""Shared plugin primitives for publisher-specific downloaders."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


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
    def is_supported_doi(self, doi: str) -> bool:
        """
        Check if this plugin supports the given DOI.
        
        Args:
            doi: Normalized DOI string
            
        Returns:
            True if this plugin can handle this DOI
        """

    @abc.abstractmethod
    def build_download_url(self, doi: str) -> Optional[str]:
        """
        Build a download URL from a DOI.
        
        Args:
            doi: Normalized DOI string
            
        Returns:
            Download URL or None if DOI is not supported
        """

    @abc.abstractmethod
    async def download(
        self,
        url: str,
        output_dir: Union[str, Path] = ".",
        filename: Optional[str] = None,
        wait_time: float = 5.0,
        doi: Optional[str] = None,
    ) -> DownloadResult:
        """Download a PDF using a prepared URL."""
