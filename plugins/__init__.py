"""Research Hub MCP Plugins."""

from .common import BasePlugin, DownloadResult
from .wiley_pdf_downloader import WileyPDFDownloader
from .utils import (
    PublisherDetection,
    detect_publisher_patterns,
    download_with_detected_plugin,
    format_filename_from_doi,
)

__all__ = [
    "BasePlugin",
    "DownloadResult",
    "PublisherDetection",
    "format_filename_from_doi",
    "WileyPDFDownloader",
    "detect_publisher_patterns",
    "download_with_detected_plugin",
]
