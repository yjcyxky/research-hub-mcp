#!/usr/bin/env python3
"""
Hindawi PDF Downloader Plugin

Downloads PDF files from downloads.hindawi.com URLs.
Hindawi is an open access publisher, so downloads are typically direct HTTP.

Usage:
    from plugins.hindawi_pdf_downloader import HindawiPDFDownloader

    async with HindawiPDFDownloader() as downloader:
        result = await downloader.download(
            url="https://downloads.hindawi.com/journals/xx/xxxxxx.pdf",
            output_dir="./downloads"
        )

CLI Usage:
    python -m plugins.hindawi_pdf_downloader <url> [--output-dir ./downloads] [--headless]
"""

import asyncio
import re
import sys
from pathlib import Path
from typing import Optional, Union
from urllib.request import Request, urlopen

try:
    from playwright.async_api import Browser, async_playwright  # type: ignore
except ImportError:
    Browser = None  # type: ignore
    async_playwright = None  # type: ignore

# Allow running as a script without installing the package
PLUGIN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PLUGIN_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plugins.common import BasePlugin, DownloadResult  # noqa: E402
from plugins.utils import normalize_doi  # noqa: E402


class HindawiPDFDownloader(BasePlugin):
    """
    Downloads PDF files from Hindawi.

    Hindawi is an open access publisher, so downloads are typically direct HTTP.
    """

    publisher = "hindawi"

    SUPPORTED_URL_PATTERN = re.compile(
        r"^https?://downloads\.hindawi\.com/journals/[^/]+/([^/?#]+)\.pdf$"
    )

    # Hindawi DOI prefixes
    SUPPORTED_PREFIXES = ("10.1155",)

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        headless: bool = False,
        timeout: int = 60000,
        user_agent: Optional[str] = None,
    ):
        """
        Initialize the downloader.

        Args:
            headless: Run browser in headless mode (default: False for debugging)
            timeout: Navigation timeout in milliseconds (default: 60000)
            user_agent: Custom user agent string (optional)
        """
        self.headless = headless
        self.timeout = timeout
        self.user_agent = user_agent or self.DEFAULT_USER_AGENT
        self._playwright = None
        self._browser: Optional[Browser] = None

    async def __aenter__(self):
        """Async context manager entry."""
        if async_playwright is None:
            raise ImportError(
                "playwright is not installed. Install it with: pip install playwright"
            )
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    @staticmethod
    def normalize_doi(doi: str) -> str:
        """Normalize common DOI formats to the canonical form."""
        return normalize_doi(doi)

    def is_supported_doi(self, doi: str) -> bool:
        """
        Check if this plugin supports the given DOI.

        Args:
            doi: Normalized DOI string

        Returns:
            True if this plugin can handle this DOI
        """
        normalized_doi = self.normalize_doi(doi)
        prefix = (
            normalized_doi.split("/")[0] if "/" in normalized_doi else normalized_doi
        )
        return prefix in self.SUPPORTED_PREFIXES

    @classmethod
    def is_supported_url(cls, url: str) -> bool:
        """Check if the URL is a supported Hindawi PDF URL."""
        return cls.SUPPORTED_URL_PATTERN.match(url) is not None

    def build_download_url(self, doi: str) -> Optional[str]:
        """
        Build Hindawi PDF URLs from DOI.

        Args:
            doi: Normalized DOI string (e.g., "10.1155/2023/1234567")

        Returns:
            List of candidate PDF URLs or None if DOI format is not supported
        """
        if not self.is_supported_doi(doi):
            return None

        normalized_doi = self.normalize_doi(doi)
        parts = normalized_doi.split("/")
        if len(parts) >= 2:
            article_id = parts[1]
            # Try two URL patterns based on Hindawi's structure
            # Pattern 1: Split article ID into journal prefix and article number
            if len(article_id) >= 2:
                journal_prefix = article_id[:2]
                article_number = article_id[2:]
                url1 = f"https://downloads.hindawi.com/journals/{journal_prefix}/{article_number}.pdf"
                url2 = f"https://downloads.hindawi.com/journals/{article_id}.pdf"
                # Return the first pattern as primary, but both will be tried
                return url1
        return None

    @staticmethod
    def sanitize_filename(article_id: str) -> str:
        """Convert article ID to a safe filename."""
        filename = article_id.replace("/", "_")
        filename = re.sub(r'[<>:"|?*]', "", filename)
        return f"{filename}.pdf"

    async def _download_direct_http(
        self, url: str, output_file: Path
    ) -> Optional[bytes]:
        """
        Attempt direct HTTP download (Hindawi is open access).

        Args:
            url: PDF URL
            output_file: Target file path

        Returns:
            PDF content bytes or None if download failed
        """
        try:
            def _sync_download():
                req = Request(url, headers={"User-Agent": self.user_agent})
                with urlopen(req, timeout=30) as response:
                    if response.status == 200:
                        content = response.read()
                        if len(content) > 4 and content[:4] == b"%PDF":
                            return content
                    return None

            content = await asyncio.to_thread(_sync_download)
            return content
        except Exception:
            return None

    async def download(
        self,
        url: str,
        output_dir: Union[str, Path] = ".",
        filename: Optional[str] = None,
        wait_time: float = 5.0,
        doi: Optional[str] = None,
    ) -> DownloadResult:
        """
        Download a PDF from Hindawi.

        Args:
            url: The Hindawi PDF URL
            output_dir: Directory to save the PDF (default: current directory)
            filename: Custom filename (optional, derived from article ID if not provided)
            wait_time: Time to wait for PDF to load in seconds (default: 5.0)
            doi: DOI string if already known (optional)

        Returns:
            DownloadResult with success status and file information
        """
        if not self.is_supported_url(url):
            return DownloadResult(
                success=False,
                error="Unsupported URL format. Expected: https://downloads.hindawi.com/journals/.../...pdf",
                doi=doi,
                publisher=self.publisher,
            )

        normalized_doi = self.normalize_doi(doi or "hindawi_download")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if filename:
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            output_file = output_path / filename
        else:
            match = self.SUPPORTED_URL_PATTERN.match(url)
            article_id = match.group(1) if match else "hindawi_download"
            output_file = output_path / self.sanitize_filename(article_id)

        # Try the provided URL first
        pdf_content = await self._download_direct_http(url, output_file)

        # If that fails and we have a DOI, try alternative URL patterns
        if not pdf_content and doi:
            parts = normalized_doi.split("/")
            if len(parts) >= 2:
                article_id = parts[1]
                # Try alternative URL pattern
                alt_url = f"https://downloads.hindawi.com/journals/{article_id}.pdf"
                if alt_url != url:
                    pdf_content = await self._download_direct_http(alt_url, output_file)

        if pdf_content and pdf_content[:4] == b"%PDF":
            output_file.write_bytes(pdf_content)
            return DownloadResult(
                success=True,
                file_path=output_file,
                file_size=len(pdf_content),
                doi=normalized_doi if doi else None,
                publisher=self.publisher,
            )
        else:
            return DownloadResult(
                success=False,
                error="Failed to download PDF from Hindawi.",
                doi=normalized_doi if doi else None,
                publisher=self.publisher,
            )


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download PDF from Hindawi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s https://downloads.hindawi.com/journals/xx/xxxxxx.pdf
    %(prog)s https://downloads.hindawi.com/journals/xx/xxxxxx.pdf --output-dir ./papers
        """,
    )
    parser.add_argument("url", help="Hindawi PDF URL to download")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=".",
        help="Output directory (default: current directory)",
    )
    parser.add_argument(
        "--filename",
        "-f",
        help="Custom output filename (optional)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode",
    )

    args = parser.parse_args()

    if not HindawiPDFDownloader.is_supported_url(args.url):
        print("Error: Unsupported URL format")
        print("Expected: https://downloads.hindawi.com/journals/.../...pdf")
        return 1

    print(f"Downloading: {args.url}")
    print(f"Output directory: {args.output_dir}")

    async with HindawiPDFDownloader(headless=args.headless) as downloader:
        result = await downloader.download(
            url=args.url,
            output_dir=args.output_dir,
            filename=args.filename,
        )

    if result.success:
        print(f"Success: {result.file_path} ({result.file_size:,} bytes)")
        if result.doi:
            print(f"DOI: {result.doi}")
        return 0
    else:
        print(f"Error: {result.error}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))

