from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Type, Union

from plugins.common import BasePlugin, DownloadResult

# Import plugins lazily to avoid circular imports
# Plugins will be registered via PLUGIN_REGISTRY


@dataclass
class PublisherDetection:
    """Publisher detection result with prepared URLs."""

    publisher: str
    doi: str
    urls: list[str]


# Plugin registry - populated by importing plugin modules
PLUGIN_REGISTRY: Dict[str, Type[BasePlugin]] = {}


def normalize_doi(doi: str) -> str:
    """Normalize DOI strings (strip prefixes like https://doi.org/)."""
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    doi = doi.removeprefix("doi:")
    return doi.strip()


def format_filename_from_doi(doi: str) -> str:
    """Create a safe default filename from a DOI (without extension)."""
    normalized = normalize_doi(doi)
    safe = normalized.replace("/", "_")
    safe = re.sub(r'[<>:"|?*]', "", safe)
    return safe


def detect_publisher_patterns(doi: str) -> Optional[PublisherDetection]:
    """
    Detect publisher by DOI and return prepared candidate URLs.
    
    Iterates through registered plugins to find one that supports the DOI.

    Args:
        doi: DOI string in any common format.

    Returns:
        PublisherDetection with publisher key and candidate URLs, or None if unknown.
    """
    normalized_doi = normalize_doi(doi)
    
    # Try each registered plugin
    for publisher, plugin_cls in PLUGIN_REGISTRY.items():
        # Create a temporary instance to check support
        # Use minimal initialization (no browser, etc.)
        try:
            plugin = plugin_cls()
            if plugin.is_supported_doi(normalized_doi):
                url = plugin.build_download_url(normalized_doi)
                if url:
                    urls = [url]
                    
                    # Special handling for Hindawi: add alternative URL pattern
                    if publisher == "hindawi":
                        parts = normalized_doi.split("/")
                        if len(parts) >= 2:
                            article_id = parts[1]
                            # Add alternative URL pattern
                            alt_url = f"https://downloads.hindawi.com/journals/{article_id}.pdf"
                            if alt_url != url:
                                urls.append(alt_url)
                    
                    return PublisherDetection(
                        publisher=publisher,
                        doi=normalized_doi,
                        urls=urls,
                    )
        except Exception:
            # If plugin initialization fails, skip it
            continue
    
    return None


async def download_with_detected_plugin(
    doi: str,
    output_dir: Union[str, Path] = ".",
    filename: Optional[str] = None,
    wait_time: float = 5.0,
    plugin_options: Optional[Dict[str, dict]] = None,
) -> DownloadResult:
    """
    Detect the publisher for a DOI, route to the correct plugin, and download.

    Args:
        doi: DOI string.
        output_dir: Directory to save the PDF.
        filename: Optional filename override.
        wait_time: Wait time passed through to the plugin downloader.
        plugin_options: Optional mapping of publisher -> kwargs for plugin init.

    Returns:
        DownloadResult indicating success or failure.
    """
    detection = detect_publisher_patterns(doi)
    if not detection:
        return DownloadResult(
            success=False,
            error=f"Unsupported or unknown publisher pattern for DOI: {doi}",
            doi=normalize_doi(doi),
        )

    plugin_cls = PLUGIN_REGISTRY.get(detection.publisher)
    if not plugin_cls:
        return DownloadResult(
            success=False,
            error=f"No plugin registered for publisher '{detection.publisher}'",
            doi=detection.doi,
            publisher=detection.publisher,
        )

    plugin_kwargs = (plugin_options or {}).get(detection.publisher, {})
    default_filename = filename or format_filename_from_doi(detection.doi)

    try:
        plugin = plugin_cls(**plugin_kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        return DownloadResult(
            success=False,
            error=f"Failed to initialize plugin '{detection.publisher}': {exc}",
            doi=detection.doi,
            publisher=detection.publisher,
        )

    last_error: Optional[str] = None

    try:
        async with plugin as downloader:
            for url in detection.urls:
                result = await downloader.download(
                    url=url,
                    output_dir=output_dir,
                    filename=default_filename,
                    wait_time=wait_time,
                    doi=detection.doi,
                )
                if result.success:
                    result.publisher = detection.publisher
                    return result
                last_error = result.error or "Unknown download failure"
    except Exception as exc:  # pragma: no cover - defensive
        last_error = str(exc)

    return DownloadResult(
        success=False,
        error=last_error
        or f"All candidate URLs failed for publisher '{detection.publisher}'",
        doi=detection.doi,
        publisher=detection.publisher,
    )
