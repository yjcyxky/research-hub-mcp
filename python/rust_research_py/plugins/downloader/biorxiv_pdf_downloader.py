#!/usr/bin/env python3
"""
bioRxiv PDF Downloader Plugin

Downloads PDF files from bioRxiv (biorxiv.org) and medRxiv (medrxiv.org).
These are preprint servers for biology and health sciences.

Usage:
    from plugins.biorxiv_pdf_downloader import BioRxivPDFDownloader

    async with BioRxivPDFDownloader() as downloader:
        result = await downloader.download(
            url="https://www.biorxiv.org/content/10.1101/2025.07.31.667880v1.full.pdf",
            output_dir="./downloads"
        )

CLI Usage:
    python -m plugins.biorxiv_pdf_downloader <url> [--output-dir ./downloads] [--headless]
    python -m plugins.biorxiv_pdf_downloader --doi 10.1101/2025.07.31.667880 --output-dir ./papers
"""

import asyncio
import base64
import re
import sys
import tempfile
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

from rust_research_py.plugins.common import BasePlugin, DownloadResult  # noqa: E402


class BioRxivPDFDownloader(BasePlugin):
    """
    Downloads PDF files from bioRxiv and medRxiv preprint servers.

    Handles both direct HTTP downloads (typically available for preprints) and
    browser-based downloads as fallback.
    """

    publisher = "biorxiv"

    # Match bioRxiv and medRxiv PDF URLs
    SUPPORTED_URL_PATTERN = re.compile(
        r"^https?://www\.(biorxiv|medrxiv)\.org/content/([^/?#]+(?:/[^/?#]+)?)(?:v\d+)?(?:\.full)?(?:\.pdf)?$"
    )

    # bioRxiv and medRxiv DOI prefix
    SUPPORTED_PREFIXES = (
        "10.1101",  # bioRxiv and medRxiv
    )

    DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/142.0.0.0 Safari/537.36"
    )

    # Both servers share the same DOI prefix 10.1101
    SERVERS = ("biorxiv", "medrxiv")

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
            raise ImportError("playwright is not installed. Install it with: pip install playwright")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    @classmethod
    def extract_doi_from_url(cls, url: str) -> Optional[str]:
        """
        Extract DOI from a bioRxiv/medRxiv PDF URL.

        Args:
            url: The bioRxiv/medRxiv PDF URL

        Returns:
            The DOI string or None if not found
        """
        match = cls.SUPPORTED_URL_PATTERN.match(url)
        if match:
            doi_path = match.group(2)
            # Clean up the DOI (remove version, .full, .pdf suffixes)
            doi_path = re.sub(r"v\d+$", "", doi_path)
            doi_path = re.sub(r"\.full$", "", doi_path)
            doi_path = re.sub(r"\.pdf$", "", doi_path)
            return doi_path
        return None

    @staticmethod
    def normalize_doi(doi: str) -> str:
        """Normalize common DOI formats to the canonical form."""
        doi = doi.strip()
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        doi = doi.removeprefix("doi:")
        return doi.strip()

    def is_supported_doi(self, doi: str) -> bool:
        """
        Check if this plugin supports the given DOI.

        Args:
            doi: Normalized DOI string

        Returns:
            True if this plugin can handle this DOI
        """
        normalized_doi = self.normalize_doi(doi)
        prefix = normalized_doi.split("/")[0] if "/" in normalized_doi else normalized_doi
        return prefix in self.SUPPORTED_PREFIXES

    @classmethod
    def is_supported_url(cls, url: str) -> bool:
        """Check if the URL is a supported bioRxiv/medRxiv PDF URL."""
        return cls.SUPPORTED_URL_PATTERN.match(url) is not None

    def build_download_url(self, doi: str) -> Optional[str]:
        """
        Build a bioRxiv PDF URL from DOI (returns first server URL for compatibility).

        Args:
            doi: Normalized DOI string (e.g., "10.1101/2025.07.31.667880")

        Returns:
            PDF URL or None if DOI format is not supported
        """
        urls = self.build_download_urls(doi)
        return urls[0] if urls else None

    def build_download_urls(self, doi: str) -> list[str]:
        """
        Build PDF URLs for both bioRxiv and medRxiv from DOI.

        Args:
            doi: Normalized DOI string (e.g., "10.1101/2025.07.31.667880")

        Returns:
            List of PDF URLs to try (bioRxiv first, then medRxiv)
        """
        if not self.is_supported_doi(doi):
            return []

        normalized_doi = self.normalize_doi(doi)
        # Try both servers - bioRxiv first, then medRxiv
        # URL format: https://www.{server}.org/content/{DOI}v1.full.pdf
        return [
            f"https://www.{server}.org/content/{normalized_doi}v1.full.pdf"
            for server in self.SERVERS
        ]

    @staticmethod
    def sanitize_filename(doi: str) -> str:
        """Convert DOI to a safe filename."""
        filename = doi.replace("/", "_")
        filename = re.sub(r'[<>:"|?*]', "", filename)
        return f"{filename}.pdf"

    async def _download_direct_http(
        self,
        url: str,
        output_file: Path,
    ) -> Optional[bytes]:
        """
        Attempt direct HTTP download (preprints are typically open access).

        Args:
            url: PDF URL
            output_file: Target file path

        Returns:
            PDF content bytes or None if download failed
        """
        try:
            # Use asyncio to run blocking HTTP request in thread pool
            def _sync_download():
                req = Request(url, headers={"User-Agent": self.user_agent})
                with urlopen(req, timeout=30) as response:
                    if response.status == 200:
                        content = response.read()
                        # Verify it's a PDF
                        if len(content) > 4 and content[:4] == b"%PDF":
                            return content
                    return None

            content = await asyncio.to_thread(_sync_download)
            return content
        except Exception:
            return None

    async def _download_with_cdp_fetch(
        self,
        url: str,
        output_file: Path,
        wait_time: float,
    ) -> Optional[bytes]:
        """
        Download PDF using CDP Fetch interception.
        Works in non-headless mode where PDF is displayed in browser.
        """
        pdf_content = None
        context = await self._browser.new_context(user_agent=self.user_agent)
        page = await context.new_page()

        try:
            # Set up CDP session for request interception
            client = await context.new_cdp_session(page)

            # Enable Fetch domain to intercept responses
            await client.send(
                "Fetch.enable",
                {"patterns": [{"urlPattern": "*", "requestStage": "Response"}]},
            )

            async def handle_fetch(params):
                nonlocal pdf_content
                request_id = params["requestId"]

                try:
                    # Get response body
                    body_result = await client.send(
                        "Fetch.getResponseBody", {"requestId": request_id}
                    )
                    body_data = body_result.get("body", "")
                    is_base64 = body_result.get("base64Encoded", False)

                    if is_base64:
                        data = base64.b64decode(body_data)
                    else:
                        data = (
                            body_data.encode()
                            if isinstance(body_data, str)
                            else body_data
                        )

                    # Check if it's a PDF (magic bytes)
                    if len(data) > 100 and data[:4] == b"%PDF":
                        pdf_content = data

                except Exception:
                    pass

                # Continue the request
                try:
                    await client.send("Fetch.continueRequest", {"requestId": request_id})
                except Exception:
                    pass

            client.on(
                "Fetch.requestPaused",
                lambda params: asyncio.create_task(handle_fetch(params)),
            )

            # Navigate to the URL
            await page.goto(url, timeout=self.timeout)

            # Wait for PDF to load
            await asyncio.sleep(wait_time)

        finally:
            await context.close()

        return pdf_content

    async def _download_with_download_handler(
        self,
        url: str,
        output_file: Path,
        wait_time: float,
    ) -> Optional[bytes]:
        """
        Download PDF by handling browser download events.
        Works in headless mode where PDF triggers a download.
        """
        # Create a temporary directory for downloads
        with tempfile.TemporaryDirectory() as temp_dir:
            context = await self._browser.new_context(
                user_agent=self.user_agent,
                accept_downloads=True,
            )
            page = await context.new_page()

            try:
                download = None
                try:
                    async with page.expect_download(timeout=self.timeout) as download_info:
                        try:
                            await page.goto(url, timeout=self.timeout, wait_until="commit")
                        except Exception as e:
                            # "Download is starting" is expected when navigation triggers a download
                            if "Download is starting" not in str(e):
                                raise
                    download = await download_info.value
                except Exception:
                    download = None

                if not download:
                    await asyncio.sleep(wait_time)
                    return None

                suggested_name = download.suggested_filename or "download.pdf"
                temp_file = Path(temp_dir) / suggested_name
                try:
                    await download.save_as(temp_file)
                except Exception:
                    # If save_as fails (e.g., canceled), bail out
                    await context.close()
                    return None

                if temp_file.exists():
                    return temp_file.read_bytes()

            finally:
                await context.close()

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
        Download a PDF from bioRxiv/medRxiv.

        Args:
            url: The bioRxiv/medRxiv PDF URL
            output_dir: Directory to save the PDF (default: current directory)
            filename: Custom filename (optional, derived from DOI if not provided)
            wait_time: Time to wait for PDF to load in seconds (default: 5.0)
            doi: DOI string if already known (optional)

        Returns:
            DownloadResult with success status and file information
        """
        # Normalize URL - ensure proper PDF suffix
        if not url.endswith(".pdf"):
            # Add version and .full.pdf if needed
            if "v" not in url.split("/")[-1]:
                url = url.rstrip("/") + "v1"
            if ".full" not in url:
                url = url + ".full"
            url = url + ".pdf"

        # Extract DOI from URL if not provided
        if not doi:
            doi = self.extract_doi_from_url(url)

        # Validate URL
        if not self.is_supported_url(url):
            return DownloadResult(
                success=False,
                error="Unsupported URL format. Expected: https://www.biorxiv.org/content/... or https://www.medrxiv.org/content/...",
                doi=doi,
                publisher=self.publisher,
            )

        extracted_doi = self.extract_doi_from_url(url) or "biorxiv_download"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine output filename
        if filename:
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            output_file = output_path / filename
        else:
            output_file = output_path / self.sanitize_filename(extracted_doi)

        # Try direct HTTP download first (preprints are typically open access)
        pdf_content = await self._download_direct_http(url, output_file)

        # If direct download failed, try browser-based download
        if not pdf_content and self._browser:
            try:
                if self.headless:
                    # In headless mode, PDF triggers download
                    pdf_content = await self._download_with_download_handler(
                        url, output_file, wait_time
                    )

                    # If download handler didn't work, try CDP fetch as fallback
                    if not pdf_content:
                        pdf_content = await self._download_with_cdp_fetch(
                            url, output_file, wait_time
                        )
                else:
                    # In non-headless mode, PDF is displayed in browser
                    pdf_content = await self._download_with_cdp_fetch(
                        url, output_file, wait_time
                    )
            except Exception as e:
                return DownloadResult(
                    success=False,
                    error=f"Browser download failed: {str(e)}",
                    doi=doi,
                    publisher=self.publisher,
                )

        if pdf_content and pdf_content[:4] == b"%PDF":
            output_file.write_bytes(pdf_content)
            return DownloadResult(
                success=True,
                file_path=output_file,
                file_size=len(pdf_content),
                doi=doi,
                publisher=self.publisher,
            )
        else:
            return DownloadResult(
                success=False,
                error="Failed to download PDF from bioRxiv/medRxiv.",
                doi=doi,
                publisher=self.publisher,
            )


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download PDF from bioRxiv/medRxiv (tries both servers automatically)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s https://www.biorxiv.org/content/10.1101/2025.07.31.667880v1.full.pdf
    %(prog)s --doi 10.1101/2025.07.31.667880 --output-dir ./papers
        """,
    )
    parser.add_argument("url", nargs="?", help="bioRxiv/medRxiv PDF URL to download")
    parser.add_argument(
        "--doi",
        "-d",
        help="DOI to download (alternative to URL, will try both biorxiv and medrxiv)",
    )
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
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=60000,
        help="Navigation timeout in milliseconds (default: 60000)",
    )
    parser.add_argument(
        "--wait",
        "-w",
        type=float,
        default=5.0,
        help="Wait time for PDF to load in seconds (default: 5.0)",
    )

    args = parser.parse_args()

    # Determine URL(s) from arguments
    url = args.url
    doi = args.doi
    urls_to_try = []

    if url:
        # Single URL provided
        if not url.endswith(".pdf"):
            if "v" not in url.split("/")[-1]:
                url = url.rstrip("/") + "v1"
            if ".full" not in url:
                url = url + ".full"
            url = url + ".pdf"
        urls_to_try = [url]
    elif doi:
        # Build URLs from DOI - try both servers
        downloader = BioRxivPDFDownloader()
        urls_to_try = downloader.build_download_urls(doi)
        if not urls_to_try:
            print(f"Error: Unsupported DOI format: {doi}")
            return 1
    else:
        print("Error: Either URL or --doi must be provided")
        return 1

    print(f"Will try {len(urls_to_try)} URL(s): {', '.join(urls_to_try)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'headless' if args.headless else 'visible browser'}")

    async with BioRxivPDFDownloader(
        headless=args.headless,
        timeout=args.timeout,
    ) as downloader:
        for url in urls_to_try:
            print(f"Trying: {url}")
            result = await downloader.download(
                url=url,
                output_dir=args.output_dir,
                filename=args.filename,
                wait_time=args.wait,
                doi=doi,
            )
            if result.success:
                print(f"Success: {result.file_path} ({result.file_size:,} bytes)")
                if result.doi:
                    print(f"DOI: {result.doi}")
                return 0
            else:
                print(f"Failed: {result.error}")

    print("Error: All download attempts failed")
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
