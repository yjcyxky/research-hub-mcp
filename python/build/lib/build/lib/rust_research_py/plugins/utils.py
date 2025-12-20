from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Type, Union

from rust_research_py.plugins.common import BasePlugin, DownloadResult

try:
    from playwright.async_api import async_playwright  # type: ignore
except ImportError:
    async_playwright = None  # type: ignore

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


async def execute_download_by_cdp(
    url: str,
    output_file: Union[str, Path],
    headless: bool = True,
    timeout: int = 60000,
    wait_time: float = 5.0,
    user_agent: Optional[str] = None,
) -> DownloadResult:
    """
    Download a file using Playwright with multiple strategies for robustness.

    This function tries multiple approaches in order:
    1. Direct download handling (for URLs that trigger browser downloads)
    2. CDP Fetch interception (for content that loads in page)
    3. Direct HTTP fetch as final fallback

    Args:
        url: URL of the file to download
        output_file: Path where the file should be saved
        headless: Run browser in headless mode (default: True)
        timeout: Navigation timeout in milliseconds (default: 60000)
        wait_time: Time to wait after page load in seconds (default: 5.0)
        user_agent: Custom user agent string (optional)

    Returns:
        DownloadResult indicating success or failure
    """
    if async_playwright is None:
        return DownloadResult(
            success=False,
            error="playwright is not installed. Install it with: pip install playwright",
        )

    default_user_agent = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
    user_agent = user_agent or default_user_agent

    output_path = Path(output_file)
    output_dir = output_path.parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    playwright = None
    file_content: Optional[bytes] = None
    last_error: Optional[str] = None

    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=headless)

        # Strategy 1: Try download handler first (handles direct download URLs)
        file_content, last_error = await _try_download_handler(
            browser, url, output_path, user_agent, timeout, wait_time
        )

        # Strategy 2: If download handler didn't work, try CDP Fetch interception
        if file_content is None:
            file_content, cdp_error = await _try_cdp_fetch(
                browser, url, user_agent, timeout, wait_time
            )
            if cdp_error and not last_error:
                last_error = cdp_error

        await browser.close()

    except Exception as e:
        last_error = f"Browser error: {str(e)}"

    finally:
        if playwright:
            await playwright.stop()

    # Strategy 3: If browser methods failed, try direct HTTP fetch
    if file_content is None:
        file_content, http_error = await _try_http_fetch(url, user_agent, timeout)
        if http_error and not last_error:
            last_error = http_error

    # Save the file if we got content
    if file_content:
        # Validate PDF content
        if not _is_valid_pdf(file_content):
            return DownloadResult(
                success=False,
                error="Downloaded content is not a valid PDF file",
            )

        try:
            output_path.write_bytes(file_content)
            file_size = len(file_content)
            return DownloadResult(
                success=True,
                file_path=output_path,
                file_size=file_size,
            )
        except Exception as e:
            return DownloadResult(
                success=False,
                error=f"Failed to save file: {str(e)}",
            )

    return DownloadResult(
        success=False,
        error=last_error or "No file content captured via any download method",
    )


async def _try_download_handler(
    browser,
    url: str,
    output_path: Path,
    user_agent: str,
    timeout: int,
    wait_time: float,
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Try to download using Playwright's download handler.

    This handles URLs that trigger browser download behavior (Content-Disposition: attachment).
    """
    file_content: Optional[bytes] = None
    error: Optional[str] = None

    try:
        context = await browser.new_context(
            user_agent=user_agent,
            accept_downloads=True,
        )
        page = await context.new_page()

        try:
            # Set up download handler before navigation
            download_started = asyncio.Event()
            download_obj = None

            async def handle_download(download):
                nonlocal download_obj
                download_obj = download
                download_started.set()

            page.on("download", handle_download)

            # Navigate - this may fail with "Download is starting" which is expected
            navigation_error = None
            try:
                await page.goto(url, timeout=timeout, wait_until="commit")
            except Exception as e:
                navigation_error = str(e)
                # "Download is starting" means navigation triggered a download - this is expected
                if "download" not in navigation_error.lower():
                    # Other errors might still allow download to proceed
                    pass

            # Wait for download to start (with timeout)
            try:
                await asyncio.wait_for(download_started.wait(), timeout=wait_time + 5)
            except asyncio.TimeoutError:
                # Download didn't start, not a direct download URL
                if navigation_error and "download" not in navigation_error.lower():
                    error = navigation_error

            # If download started, save it
            if download_obj is not None:
                try:
                    # Wait for download to complete
                    temp_path = await download_obj.path()
                    if temp_path:
                        file_content = Path(temp_path).read_bytes()
                except Exception as e:
                    # Try save_as as fallback
                    try:
                        await download_obj.save_as(output_path)
                        if output_path.exists():
                            file_content = output_path.read_bytes()
                    except Exception as save_error:
                        error = f"Download save failed: {save_error}"

        finally:
            await context.close()

    except Exception as e:
        error = f"Download handler error: {str(e)}"

    return file_content, error


async def _try_cdp_fetch(
    browser,
    url: str,
    user_agent: str,
    timeout: int,
    wait_time: float,
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Try to download using CDP Fetch interception.

    This captures response bodies at the network level, useful for content
    that loads in the page rather than triggering a download.
    """
    file_content: Optional[bytes] = None
    error: Optional[str] = None
    captured_contents: list[bytes] = []

    try:
        context = await browser.new_context(user_agent=user_agent)
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

                    # Check if it's PDF content
                    if _is_valid_pdf(data):
                        captured_contents.append(data)
                    elif len(data) > 10000:
                        # Large content might be a PDF without proper magic bytes at start
                        # (e.g., if there's some wrapper)
                        # Check for PDF signature anywhere in first 1024 bytes
                        if b"%PDF" in data[:1024]:
                            captured_contents.append(data)

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

            # Navigate to the URL - handle potential download trigger gracefully
            try:
                await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
                # Wait for content to load
                await asyncio.sleep(wait_time)
            except Exception as e:
                # Navigation might fail but CDP could still have captured content
                if "download" in str(e).lower():
                    # Download was triggered, CDP might have captured it
                    await asyncio.sleep(2)
                else:
                    error = f"Navigation error: {str(e)}"

            # Get the largest captured PDF content
            if captured_contents:
                file_content = max(captured_contents, key=len)

        finally:
            await context.close()

    except Exception as e:
        error = f"CDP fetch error: {str(e)}"

    return file_content, error


async def _try_http_fetch(
    url: str,
    user_agent: str,
    timeout: int,
) -> tuple[Optional[bytes], Optional[str]]:
    """
    Try direct HTTP fetch as a fallback.

    This is simpler but may not work for sites with anti-bot protection.
    """
    file_content: Optional[bytes] = None
    error: Optional[str] = None

    try:
        import aiohttp

        headers = {
            "User-Agent": user_agent,
            "Accept": "application/pdf,*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }

        connector = aiohttp.TCPConnector(ssl=False)
        timeout_obj = aiohttp.ClientTimeout(total=timeout / 1000)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout_obj
        ) as session:
            async with session.get(url, headers=headers, allow_redirects=True) as response:
                if response.status == 200:
                    data = await response.read()
                    if _is_valid_pdf(data):
                        file_content = data
                    else:
                        error = f"HTTP response is not a valid PDF (status: {response.status})"
                else:
                    error = f"HTTP request failed with status: {response.status}"

    except ImportError:
        error = "aiohttp not available for HTTP fallback"
    except Exception as e:
        error = f"HTTP fetch error: {str(e)}"

    return file_content, error


def _is_valid_pdf(data: bytes) -> bool:
    """Check if the data looks like a valid PDF file."""
    if not data or len(data) < 100:
        return False

    # Check for PDF magic bytes at the start
    if data[:4] == b"%PDF":
        return True

    # Some PDFs might have a BOM or whitespace at the start
    # Check first 1024 bytes for PDF signature
    return b"%PDF" in data[:1024]
