#!/usr/bin/env python3
"""
Oxford Academic (OUP) PDF Downloader Plugin

Downloads PDF files from Oxford Academic (academic.oup.com).
Uses Playwright throughout to:
1. Navigate to https://doi.org/{DOI} (auto-redirects to article page)
2. Extract citation_pdf_url from page source
3. Navigate to PDF URL and capture the download via CDP Fetch interception

CLI Usage:
    python -m plugins.oxford_pdf_downloader --doi 10.1093/bib/bbaf505 --output-dir ./papers
"""

import asyncio
import base64
import re
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional, Union

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


class MetaTagParser(HTMLParser):
    """Parser to extract citation_pdf_url from HTML meta tags."""
    
    def __init__(self):
        super().__init__()
        self.pdf_url: Optional[str] = None
    
    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() == "meta":
            attrs_dict = dict(attrs)
            name = attrs_dict.get("name", "")
            if name.lower() == "citation_pdf_url":
                self.pdf_url = attrs_dict.get("content")


class OxfordPDFDownloader(BasePlugin):
    """
    Downloads PDF files from Oxford Academic (academic.oup.com).
    Uses Playwright browser throughout for reliable downloads.
    """

    publisher = "oxford"

    # Oxford University Press DOI prefix
    SUPPORTED_PREFIXES = ("10.1093",)

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
        """Initialize the downloader."""
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

    @staticmethod
    def normalize_doi(doi: str) -> str:
        """Normalize common DOI formats to the canonical form."""
        doi = doi.strip()
        doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
        doi = doi.removeprefix("doi:")
        return doi.strip()

    def is_supported_doi(self, doi: str) -> bool:
        """Check if this plugin supports the given DOI."""
        normalized_doi = self.normalize_doi(doi)
        prefix = normalized_doi.split("/")[0] if "/" in normalized_doi else normalized_doi
        return prefix in self.SUPPORTED_PREFIXES

    @classmethod
    def is_supported_url(cls, url: str) -> bool:
        """Check if the URL is a supported Oxford Academic URL."""
        return "academic.oup.com" in url

    def build_download_url(self, doi: str) -> Optional[str]:
        """Build DOI resolution URL."""
        if not self.is_supported_doi(doi):
            return None
        return f"https://doi.org/{self.normalize_doi(doi)}"

    @staticmethod
    def sanitize_filename(doi: str) -> str:
        """Convert DOI to a safe filename."""
        filename = doi.replace("/", "_")
        filename = re.sub(r'[<>:"|?*]', "", filename)
        return f"{filename}.pdf"

    async def download(
        self,
        url: str,
        output_dir: Union[str, Path] = ".",
        filename: Optional[str] = None,
        wait_time: float = 8.0,
        doi: Optional[str] = None,
    ) -> DownloadResult:
        """
        Download a PDF from Oxford Academic.

        For Oxford Academic, the URL from build_download_url is a DOI URL (https://doi.org/...).
        This method:
        1. Navigates to the DOI URL (auto-redirects to article page)
        2. Extracts citation_pdf_url from page source
        3. Downloads PDF using CDP Fetch interception to bypass anti-bot

        Args:
            url: DOI URL (https://doi.org/...) built by build_download_url
            output_dir: Directory to save the PDF
            filename: Custom filename (optional)
            wait_time: Time to wait for PDF to load (default: 8.0)
            doi: DOI string if already known (optional)

        Returns:
            DownloadResult with success status and file information
        """
        if not self._browser:
            return DownloadResult(
                success=False,
                error="Browser not initialized",
                doi=doi,
                publisher=self.publisher,
            )

        normalized_doi = self.normalize_doi(doi) if doi else None

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine output filename
        if filename:
            if not filename.endswith(".pdf"):
                filename += ".pdf"
            output_file = output_path / filename
        elif normalized_doi:
            output_file = output_path / self.sanitize_filename(normalized_doi)
        else:
            output_file = output_path / "oxford_download.pdf"

        # Step 1: Navigate to DOI URL (auto-redirects to article page)
        print(f"Navigating to: {url}")

        context = await self._browser.new_context(
            user_agent=self.user_agent,
            accept_downloads=True,
        )

        try:
            page = await context.new_page()

            # Navigate to DOI URL (will redirect to article page)
            await page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
            print(f"Redirected to: {page.url}")

            # Wait for page to fully load
            await asyncio.sleep(3)

            # Step 2: Get page source and extract citation_pdf_url
            html_content = await page.content()
            parser = MetaTagParser()
            parser.feed(html_content)
            pdf_url = parser.pdf_url

            if not pdf_url:
                await context.close()
                return DownloadResult(
                    success=False,
                    error=f"Could not find citation_pdf_url in article page: {page.url}",
                    doi=normalized_doi,
                    publisher=self.publisher,
                )

            print(f"Found PDF URL: {pdf_url}")
            await context.close()

            # Step 3: Download PDF using CDP Fetch interception
            pdf_content = await self._download_with_cdp_fetch(pdf_url, wait_time)

            # If CDP fetch fails, try download handler as fallback
            if not pdf_content or len(pdf_content) < 5 or pdf_content[:4] != b"%PDF":
                print("CDP fetch failed, trying download handler...")
                pdf_content = await self._download_with_download_handler(pdf_url, wait_time)

            if pdf_content and len(pdf_content) > 4 and pdf_content[:4] == b"%PDF":
                output_file.write_bytes(pdf_content)
                print(f"Downloaded: {len(pdf_content)} bytes")
                return DownloadResult(
                    success=True,
                    file_path=output_file,
                    file_size=len(pdf_content),
                    doi=normalized_doi,
                    publisher=self.publisher,
                )
            else:
                return DownloadResult(
                    success=False,
                    error="Failed to download PDF from Oxford Academic.",
                    doi=normalized_doi,
                    publisher=self.publisher,
                )

        except Exception as e:
            return DownloadResult(
                success=False,
                error=f"Download failed: {str(e)}",
                doi=normalized_doi,
                publisher=self.publisher,
            )

    async def _download_with_cdp_fetch(
        self,
        pdf_url: str,
        wait_time: float,
        context=None,
        page=None,
    ) -> Optional[bytes]:
        """
        Download PDF using CDP Fetch interception.
        Intercepts HTTP responses to capture the PDF content directly.
        """
        if not self._browser:
            return None

        pdf_content = None
        close_context = context is None

        if context is None:
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

            # Navigate to the PDF URL
            try:
                await page.goto(pdf_url, timeout=self.timeout, wait_until="commit")
            except Exception as e:
                # Navigation might fail because PDF is being rendered or downloaded
                if "net::ERR" not in str(e) and "timeout" not in str(e).lower():
                    print(f"Navigation note: {e}")

            # Wait for PDF to be intercepted
            await asyncio.sleep(wait_time)

        finally:
            if close_context:
                await context.close()

        return pdf_content

    async def _download_with_download_handler(
        self,
        pdf_url: str,
        wait_time: float,
        context=None,
        page=None,
    ) -> Optional[bytes]:
        """
        Download PDF by handling browser download events.
        Works in headless mode where PDF triggers a download.
        """
        if not self._browser:
            return None

        close_context = context is None

        with tempfile.TemporaryDirectory() as temp_dir:
            if context is None:
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
                            await page.goto(pdf_url, timeout=self.timeout, wait_until="commit")
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
                    if close_context:
                        await context.close()
                    return None

                if temp_file.exists():
                    return temp_file.read_bytes()

            finally:
                if close_context:
                    await context.close()

        return None


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download PDF from Oxford Academic (academic.oup.com)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --doi 10.1093/bib/bbaf505 --output-dir ./papers
    %(prog)s --doi 10.1093/bib/bbaf505 --headless
        """,
    )
    parser.add_argument("--doi", "-d", required=True, help="DOI to download (e.g., 10.1093/bib/bbaf505)")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory (default: current directory)")
    parser.add_argument("--filename", "-f", help="Custom output filename (optional)")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--timeout", "-t", type=int, default=60000, help="Timeout in ms (default: 60000)")
    parser.add_argument("--wait", "-w", type=float, default=8.0, help="Wait time in seconds (default: 8.0)")

    args = parser.parse_args()

    downloader = OxfordPDFDownloader(headless=args.headless, timeout=args.timeout)

    if not downloader.is_supported_doi(args.doi):
        print(f"Error: Unsupported DOI prefix. Expected: 10.1093/...")
        return 1

    url = downloader.build_download_url(args.doi)
    if not url:
        print(f"Error: Could not build URL for DOI: {args.doi}")
        return 1

    print(f"DOI: {args.doi}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'headless' if args.headless else 'visible browser'}")

    async with downloader:
        result = await downloader.download(
            url=url,
            output_dir=args.output_dir,
            filename=args.filename,
            wait_time=args.wait,
            doi=args.doi,
        )

    if result.success:
        print(f"Success: {result.file_path} ({result.file_size:,} bytes)")
        return 0
    else:
        print(f"Error: {result.error}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
