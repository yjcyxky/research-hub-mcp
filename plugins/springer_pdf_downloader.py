#!/usr/bin/env python3
"""
Springer PDF Downloader Plugin

Downloads PDF files from Springer Link (link.springer.com).
Supports DOIs from various publishers hosted on Springer platform including:
- EMBO (10.15252)
- Springer Nature articles (10.1007)
- And other Springer-hosted content

Usage:
    from plugins.springer_pdf_downloader import SpringerPDFDownloader

    async with SpringerPDFDownloader() as downloader:
        result = await downloader.download(
            url="https://link.springer.com/content/pdf/10.15252/embr.202357306.pdf",
            output_dir="./downloads"
        )

CLI Usage:
    python -m plugins.springer_pdf_downloader <url> [--output-dir ./downloads] [--headless]
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

from plugins.common import BasePlugin, DownloadResult  # noqa: E402


class SpringerPDFDownloader(BasePlugin):
    """
    Downloads PDF files from Springer Link (link.springer.com).

    Handles both direct HTTP downloads (for open access articles) and
    browser-based downloads (for subscription-protected content).
    """

    publisher = "springer"

    SUPPORTED_URL_PATTERN = re.compile(
        r"^https?://link\.springer\.com/content/pdf/([^/?#]+(?:/[^/?#]+)?)(?:\.pdf)?$"
    )

    # Springer-hosted publisher DOI prefixes
    # EMBO journals (hosted on Springer Link)
    # Springer Nature journals
    # BMC journals (hosted on Springer)
    SUPPORTED_PREFIXES = (
        "10.15252",  # EMBO (EMBO Journal, EMBO Reports, EMBO Molecular Medicine, etc.)
        "10.1007",   # Springer Nature
        "10.1186",   # BMC (BioMed Central)
        "10.1140",   # European Physical Journal (Springer)
        "10.1023",   # Springer (legacy)
        "10.1365",   # Springer Gabler
        "10.1617",   # RILEM (Springer)
    )

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
        Extract DOI from a Springer Link PDF URL.

        Args:
            url: The Springer Link PDF URL

        Returns:
            The DOI string or None if not found
        """
        match = cls.SUPPORTED_URL_PATTERN.match(url)
        if match:
            doi_path = match.group(1)
            # Remove .pdf extension if present
            if doi_path.endswith(".pdf"):
                doi_path = doi_path[:-4]
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
        """Check if the URL is a supported Springer Link PDF URL."""
        return cls.SUPPORTED_URL_PATTERN.match(url) is not None

    def build_download_url(self, doi: str) -> Optional[str]:
        """
        Build a Springer Link PDF URL from DOI.

        Args:
            doi: Normalized DOI string (e.g., "10.15252/embr.202357306")

        Returns:
            PDF URL or None if DOI format is not supported
        """
        if not self.is_supported_doi(doi):
            return None

        normalized_doi = self.normalize_doi(doi)
        # Springer Link PDF URL format: https://link.springer.com/content/pdf/{DOI}.pdf
        return f"https://link.springer.com/content/pdf/{normalized_doi}.pdf"

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
        Attempt direct HTTP download (for open access articles).

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
        Download a PDF from Springer Link.

        Args:
            url: The Springer Link PDF URL
            output_dir: Directory to save the PDF (default: current directory)
            filename: Custom filename (optional, derived from DOI if not provided)
            wait_time: Time to wait for PDF to load in seconds (default: 5.0)
            doi: DOI string if already known (optional)

        Returns:
            DownloadResult with success status and file information
        """
        # Normalize URL - ensure it has .pdf suffix
        if not url.endswith(".pdf"):
            url = url.rstrip("/") + ".pdf"

        # Extract DOI from URL if not provided
        if not doi:
            doi = self.extract_doi_from_url(url)

        # Validate URL
        if not self.is_supported_url(url):
            return DownloadResult(
                success=False,
                error="Unsupported URL format. Expected: https://link.springer.com/content/pdf/...",
                doi=doi,
                publisher=self.publisher,
            )

        extracted_doi = self.extract_doi_from_url(url) or "springer_download"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine output filename
        if filename:
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            output_file = output_path / filename
        else:
            output_file = output_path / self.sanitize_filename(extracted_doi)

        # Try direct HTTP download first (for open access articles)
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
                error="Failed to download PDF. The article may require subscription access.",
                doi=doi,
                publisher=self.publisher,
            )


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download PDF from Springer Link",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s https://link.springer.com/content/pdf/10.15252/embr.202357306.pdf
    %(prog)s https://link.springer.com/content/pdf/10.1007/s00018-024-05321-8.pdf --output-dir ./papers
    %(prog)s --doi 10.15252/embr.202357306 --output-dir ./papers --headless
        """,
    )
    parser.add_argument("url", nargs="?", help="Springer Link PDF URL to download")
    parser.add_argument(
        "--doi",
        "-d",
        help="DOI to download (alternative to URL)",
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

    # Determine URL from arguments
    url = args.url
    doi = args.doi

    if not url and doi:
        # Build URL from DOI
        downloader = SpringerPDFDownloader()
        url = downloader.build_download_url(doi)
        if not url:
            print(f"Error: Unsupported DOI format: {doi}")
            return 1
    elif not url:
        print("Error: Either URL or --doi must be provided")
        return 1

    # Normalize URL
    if not url.endswith(".pdf"):
        url = url.rstrip("/") + ".pdf"

    if not SpringerPDFDownloader.is_supported_url(url):
        print("Error: Unsupported URL format")
        print("Expected: https://link.springer.com/content/pdf/...")
        return 1

    print(f"Downloading: {url}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'headless' if args.headless else 'visible browser'}")

    async with SpringerPDFDownloader(
        headless=args.headless,
        timeout=args.timeout,
    ) as downloader:
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
        print(f"Error: {result.error}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
