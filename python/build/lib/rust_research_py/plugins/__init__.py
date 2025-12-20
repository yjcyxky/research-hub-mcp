"""Research Hub MCP Plugins."""

from .downloader.biorxiv_pdf_downloader import BioRxivPDFDownloader
from .common import BasePlugin, DownloadResult
from .downloader.frontiers_pdf_downloader import FrontiersPDFDownloader
from .downloader.hindawi_pdf_downloader import HindawiPDFDownloader
from .downloader.mdpi_pdf_downloader import MDPIPDFDownloader
from .downloader.nature_pdf_downloader import NaturePDFDownloader
from .downloader.oxford_pdf_downloader import OxfordPDFDownloader
from .downloader.plos_pdf_downloader import PLOSPDFDownloader
from .downloader.pnas_pdf_downloader import PNASPDFDownloader
from .downloader.springer_pdf_downloader import SpringerPDFDownloader
from .utils import PLUGIN_REGISTRY
from .downloader.wiley_pdf_downloader import WileyPDFDownloader
from .utils import (
    PublisherDetection,
    detect_publisher_patterns,
    download_with_detected_plugin,
    format_filename_from_doi,
)

# Register plugins
PLUGIN_REGISTRY["nature"] = NaturePDFDownloader
PLUGIN_REGISTRY["wiley"] = WileyPDFDownloader
PLUGIN_REGISTRY["mdpi"] = MDPIPDFDownloader
PLUGIN_REGISTRY["frontiers"] = FrontiersPDFDownloader
PLUGIN_REGISTRY["pnas"] = PNASPDFDownloader
PLUGIN_REGISTRY["plos"] = PLOSPDFDownloader
PLUGIN_REGISTRY["hindawi"] = HindawiPDFDownloader
PLUGIN_REGISTRY["springer"] = SpringerPDFDownloader
PLUGIN_REGISTRY["biorxiv"] = BioRxivPDFDownloader
PLUGIN_REGISTRY["oxford"] = OxfordPDFDownloader

__all__ = [
    "BasePlugin",
    "DownloadResult",
    "PublisherDetection",
    "format_filename_from_doi",
    "NaturePDFDownloader",
    "WileyPDFDownloader",
    "MDPIPDFDownloader",
    "FrontiersPDFDownloader",
    "PNASPDFDownloader",
    "PLOSPDFDownloader",
    "HindawiPDFDownloader",
    "SpringerPDFDownloader",
    "BioRxivPDFDownloader",
    "OxfordPDFDownloader",
    "detect_publisher_patterns",
    "download_with_detected_plugin",
]


