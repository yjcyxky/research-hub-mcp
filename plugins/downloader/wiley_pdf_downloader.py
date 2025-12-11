#!/usr/bin/env python3
"""
Wiley Online Library PDF Downloader Plugin

Downloads PDF files from onlinelibrary.wiley.com/doi/pdfdirect URLs
using Playwright to handle authentication and redirects.

Usage:
    from plugins.wiley_pdf_downloader import WileyPDFDownloader

    async with WileyPDFDownloader() as downloader:
        result = await downloader.download(
            url="https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/jcsm.70098",
            output_dir="./downloads"
        )

CLI Usage:
    python -m plugins.wiley_pdf_downloader <url> [--output-dir ./downloads] [--headless]
"""

import asyncio
import base64
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union
from urllib.parse import unquote

from playwright.async_api import Browser, async_playwright # type: ignore

# Allow running as a script without installing the package
PLUGIN_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PLUGIN_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plugins.common import BasePlugin, DownloadResult  # noqa: E402
from plugins.utils import normalize_doi  # noqa: E402


class WileyPDFDownloader(BasePlugin):
    """
    Downloads PDF files from Wiley Online Library.

    Handles the authentication flow and extracts the original PDF
    using Chrome DevTools Protocol (CDP) Fetch interception.
    """

    publisher = "wiley"

    SUPPORTED_URL_PATTERN = re.compile(
        r"^https?://onlinelibrary\.wiley\.com/doi/pdfdirect/(.+)$"
    )

    # Wiley Online Library DOI prefixes
    SUPPORTED_PREFIXES = (
        "10.1002",  # John Wiley & Sons
        "10.1111",
        "10.1113",
        "10.1046",
        "10.1034",
        # Additional Wiley/Wiley-Blackwell prefixes
        "10.3322",  # Wiley (American Cancer Society)
        "10.2966",
        "10.1892",
        "10.1359",
        "10.2755",
        "10.1348",
        "10.1506",
        "10.3162",
        "10.2746",
        "10.1516",
        "10.1301",
        "10.3405",
        "10.1196",
        "10.3170",
        "10.3401",
        "10.1581",
        "10.1576",
        "10.1256",
        "10.1526",
        "10.1897",
        "10.5054",
        "10.4004",
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
    def extract_doi(cls, url: str) -> Optional[str]:
        """
        Extract DOI from a Wiley pdfdirect URL.

        Args:
            url: The Wiley PDF URL

        Returns:
            The DOI string or None if not found
        """
        match = cls.SUPPORTED_URL_PATTERN.match(url)
        if match:
            return unquote(match.group(1))
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
        """Check if the URL is a supported Wiley pdfdirect URL."""
        return cls.SUPPORTED_URL_PATTERN.match(url) is not None

    def build_download_url(self, doi: str) -> Optional[str]:
        """
        Build a Wiley pdfdirect URL from DOI.

        Args:
            doi: Normalized DOI string

        Returns:
            PDF URL or None if DOI format is not supported
        """
        if not self.is_supported_doi(doi):
            return None

        normalized_doi = self.normalize_doi(doi)
        return f"https://onlinelibrary.wiley.com/doi/pdfdirect/{normalized_doi}"

    @staticmethod
    def sanitize_filename(doi: str) -> str:
        """Convert DOI to a safe filename."""
        # Replace / with _ and remove other problematic characters
        filename = doi.replace("/", "_")
        filename = re.sub(r'[<>:"|?*]', "", filename)
        return f"{filename}.pdf"

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
        Download a PDF from Wiley Online Library.

        Args:
            url: The Wiley pdfdirect URL
            output_dir: Directory to save the PDF (default: current directory)
            filename: Custom filename (optional, derived from DOI if not provided)
            wait_time: Time to wait for PDF to load in seconds (default: 5.0)
            doi: DOI string if already known (optional)

        Returns:
            DownloadResult with success status and file information
        """
        # Validate URL
        if not self.is_supported_url(url):
            return DownloadResult(
                success=False,
                error="Unsupported URL format. Expected: https://onlinelibrary.wiley.com/doi/pdfdirect/...",
                doi=doi,
                publisher=self.publisher,
            )

        doi = normalize_doi(doi or self.extract_doi(url) or "wiley_download")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine output filename
        if filename:
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            output_file = output_path / filename
        else:
            output_file = output_path / self.sanitize_filename(doi)

        # Ensure browser is initialized
        if not self._browser:
            return DownloadResult(
                success=False,
                error="Browser not initialized. Use 'async with WileyPDFDownloader() as downloader:'",
                doi=doi,
                publisher=self.publisher,
            )

        pdf_content = None

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
                    error="Failed to capture PDF content",
                    doi=doi,
                    publisher=self.publisher,
                )

        except Exception as e:
            return DownloadResult(
                success=False,
                error=str(e),
                doi=doi,
                publisher=self.publisher,
            )


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download PDF from Wiley Online Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/jcsm.70098
    %(prog)s https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/jcsm.70098 --output-dir ./papers
    %(prog)s https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/jcsm.70098 --filename my_paper.pdf --headless
        """,
    )
    parser.add_argument("url", help="Wiley pdfdirect URL to download")
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

    if not WileyPDFDownloader.is_supported_url(args.url):
        print("Error: Unsupported URL format")
        print("Expected: https://onlinelibrary.wiley.com/doi/pdfdirect/...")
        return 1

    print(f"Downloading: {args.url}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'headless' if args.headless else 'visible browser'}")

    async with WileyPDFDownloader(
        headless=args.headless,
        timeout=args.timeout,
    ) as downloader:
        result = await downloader.download(
            url=args.url,
            output_dir=args.output_dir,
            filename=args.filename,
            wait_time=args.wait,
        )

    if result.success:
        print(f"Success: {result.file_path} ({result.file_size:,} bytes)")
        print(f"DOI: {result.doi}")
        return 0
    else:
        print(f"Error: {result.error}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
