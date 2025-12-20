#!/usr/bin/env python3
"""
PNAS PDF Downloader Plugin

Downloads PDF files from pnas.org URLs.
Supports both direct HTTP download (for open access) and browser-based download.

Usage:
    from plugins.pnas_pdf_downloader import PNASPDFDownloader

    async with PNASPDFDownloader() as downloader:
        result = await downloader.download(
            url="https://www.pnas.org/content/pnas/12345678.full.pdf",
            output_dir="./downloads"
        )

CLI Usage:
    python -m plugins.pnas_pdf_downloader <url> [--output-dir ./downloads] [--headless]
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
from rust_research_py.plugins.utils import normalize_doi  # noqa: E402


class PNASPDFDownloader(BasePlugin):
    """
    Downloads PDF files from PNAS (Proceedings of the National Academy of Sciences).

    Handles both direct HTTP downloads (for open access articles) and
    browser-based downloads (for subscription-protected content).
    """

    publisher = "pnas"

    SUPPORTED_URL_PATTERN = re.compile(
        r"^https?://(www\.)?pnas\.org/content/pnas/([^/?#]+)\.full\.pdf$"
    )

    # PNAS DOI prefixes
    SUPPORTED_PREFIXES = ("10.1073",)

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
        """Check if the URL is a supported PNAS PDF URL."""
        return cls.SUPPORTED_URL_PATTERN.match(url) is not None

    def build_download_url(self, doi: str) -> Optional[str]:
        """
        Build a PNAS PDF URL from DOI.

        Args:
            doi: Normalized DOI string (e.g., "10.1073/pnas.1234567890")

        Returns:
            PDF URL or None if DOI format is not supported
        """
        if not self.is_supported_doi(doi):
            return None

        normalized_doi = self.normalize_doi(doi)
        parts = normalized_doi.split("/")
        if len(parts) >= 2:
            article_id = parts[1]
            return f"https://www.pnas.org/content/pnas/{article_id}.full.pdf"
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
        Attempt direct HTTP download (for open access articles).

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

    async def _download_with_cdp_fetch(
        self, url: str, output_file: Path, wait_time: float
    ) -> Optional[bytes]:
        """Download PDF using CDP Fetch interception."""
        pdf_content = None
        context = await self._browser.new_context(user_agent=self.user_agent)
        page = await context.new_page()

        try:
            client = await context.new_cdp_session(page)
            await client.send(
                "Fetch.enable",
                {"patterns": [{"urlPattern": "*", "requestStage": "Response"}]},
            )

            async def handle_fetch(params):
                nonlocal pdf_content
                request_id = params["requestId"]

                try:
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

                    if len(data) > 100 and data[:4] == b"%PDF":
                        pdf_content = data
                except Exception:
                    pass

                try:
                    await client.send("Fetch.continueRequest", {"requestId": request_id})
                except Exception:
                    pass

            client.on(
                "Fetch.requestPaused",
                lambda params: asyncio.create_task(handle_fetch(params)),
            )

            await page.goto(url, timeout=self.timeout)
            await asyncio.sleep(wait_time)

        finally:
            await context.close()

        return pdf_content

    async def _download_with_download_handler(
        self, url: str, output_file: Path, wait_time: float
    ) -> Optional[bytes]:
        """Download PDF by handling browser download events."""
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
        Download a PDF from PNAS.

        Args:
            url: The PNAS PDF URL
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
                error="Unsupported URL format. Expected: https://www.pnas.org/content/pnas/.../full.pdf",
                doi=doi,
                publisher=self.publisher,
            )

        normalized_doi = self.normalize_doi(doi or "pnas_download")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if filename:
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            output_file = output_path / filename
        else:
            match = self.SUPPORTED_URL_PATTERN.match(url)
            article_id = match.group(2) if match else "pnas_download"
            output_file = output_path / self.sanitize_filename(article_id)

        pdf_content = await self._download_direct_http(url, output_file)

        if not pdf_content and self._browser:
            try:
                if self.headless:
                    pdf_content = await self._download_with_download_handler(
                        url, output_file, wait_time
                    )
                    if not pdf_content:
                        pdf_content = await self._download_with_cdp_fetch(
                            url, output_file, wait_time
                        )
                else:
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
                doi=normalized_doi if doi else None,
                publisher=self.publisher,
            )
        else:
            return DownloadResult(
                success=False,
                error="Failed to download PDF. The article may require subscription access.",
                doi=normalized_doi if doi else None,
                publisher=self.publisher,
            )


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download PDF from PNAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s https://www.pnas.org/content/pnas/12345678.full.pdf
    %(prog)s https://www.pnas.org/content/pnas/12345678.full.pdf --output-dir ./papers
        """,
    )
    parser.add_argument("url", help="PNAS PDF URL to download")
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

    if not PNASPDFDownloader.is_supported_url(args.url):
        print("Error: Unsupported URL format")
        print("Expected: https://www.pnas.org/content/pnas/.../full.pdf")
        return 1

    print(f"Downloading: {args.url}")
    print(f"Output directory: {args.output_dir}")

    async with PNASPDFDownloader(
        headless=args.headless, timeout=args.timeout
    ) as downloader:
        result = await downloader.download(
            url=args.url,
            output_dir=args.output_dir,
            filename=args.filename,
            wait_time=args.wait,
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

