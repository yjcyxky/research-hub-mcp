from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Type

from plugins.common import BasePlugin, DownloadResult
from plugins.wiley_pdf_downloader import WileyPDFDownloader


@dataclass
class PublisherDetection:
    """Publisher detection result with prepared URLs."""

    publisher: str
    doi: str
    urls: list[str]


PLUGIN_REGISTRY: Dict[str, Type[BasePlugin]] = {
    "wiley": WileyPDFDownloader,
}

PUBLISHER_PREFIXES: Dict[str, tuple[str, ...]] = {
    "wiley": ("10.1002", "10.1111", "10.1113", "10.1046", "10.1034"),
    "mdpi": ("10.3390",),
    "pnas": ("10.1073",),
    "frontiers": ("10.3389",),
    "plos": ("10.1371",),
    "nature": ("10.1038",),
    "hindawi": ("10.1155",),
}


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

    Args:
        doi: DOI string in any common format.

    Returns:
        PublisherDetection with publisher key and candidate URLs, or None if unknown.
    """
    normalized_doi = normalize_doi(doi)
    prefix = normalized_doi.split("/")[0] if "/" in normalized_doi else normalized_doi

    detections: list[PublisherDetection] = []

    # Wiley/Wiley Online Library
    if prefix in PUBLISHER_PREFIXES["wiley"]:
        detections.append(
            PublisherDetection(
                publisher="wiley",
                doi=normalized_doi,
                urls=[f"https://onlinelibrary.wiley.com/doi/pdfdirect/{normalized_doi}"],
            )
        )

    # MDPI (10.3390)
    if prefix in PUBLISHER_PREFIXES["mdpi"]:
        parts = normalized_doi.split("/")
        if len(parts) >= 2:
            article_id = parts[1].lower()
            match = re.match(r"([a-z]+)(\d+)", article_id)
            if match:
                journal = match.group(1)
                numbers = match.group(2)
                if len(numbers) >= 7:
                    vol = numbers[:2].lstrip("0") or "1"
                    issue = numbers[2:4].lstrip("0") or "1"
                    art = numbers[4:].lstrip("0") or "1"
                    detections.append(
                        PublisherDetection(
                            publisher="mdpi",
                            doi=normalized_doi,
                            urls=[
                                f"https://www.mdpi.com/{journal}/{vol}/{issue}/{art}/pdf"
                            ],
                        )
                    )

    # PNAS (10.1073)
    if prefix in PUBLISHER_PREFIXES["pnas"]:
        parts = normalized_doi.split("/")
        if len(parts) >= 2:
            detections.append(
                PublisherDetection(
                    publisher="pnas",
                    doi=normalized_doi,
                    urls=[f"https://www.pnas.org/content/pnas/{parts[1]}.full.pdf"],
                )
            )

    # Frontiers (10.3389)
    if prefix in PUBLISHER_PREFIXES["frontiers"]:
        detections.append(
            PublisherDetection(
                publisher="frontiers",
                doi=normalized_doi,
                urls=[f"https://www.frontiersin.org/articles/{normalized_doi}/pdf"],
            )
        )

    # PLOS (10.1371)
    if prefix in PUBLISHER_PREFIXES["plos"]:
        detections.append(
            PublisherDetection(
                publisher="plos",
                doi=normalized_doi,
                urls=[
                    f"https://journals.plos.org/plosone/article/file?id={normalized_doi}&type=printable"
                ],
            )
        )

    # Nature (10.1038)
    if prefix in PUBLISHER_PREFIXES["nature"]:
        parts = normalized_doi.split("/")
        if len(parts) >= 2:
            detections.append(
                PublisherDetection(
                    publisher="nature",
                    doi=normalized_doi,
                    urls=[f"https://www.nature.com/articles/{parts[1]}.pdf"],
                )
            )

    # Hindawi (10.1155)
    if prefix in PUBLISHER_PREFIXES["hindawi"]:
        parts = normalized_doi.split("/")
        if len(parts) >= 2:
            detections.append(
                PublisherDetection(
                    publisher="hindawi",
                    doi=normalized_doi,
                    urls=[
                        f"https://downloads.hindawi.com/journals/{parts[1][:2]}/{parts[1][2:]}.pdf",
                        f"https://downloads.hindawi.com/journals/{parts[1]}.pdf",
                    ],
                )
            )

    return detections[0] if detections else None


async def download_with_detected_plugin(
    doi: str,
    output_dir: str | Path = ".",
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
