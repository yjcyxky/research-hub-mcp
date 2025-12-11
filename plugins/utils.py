from __future__ import annotations

import asyncio
import base64
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Type, Union

from plugins.common import BasePlugin, DownloadResult

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
    Download a file using Playwright with CDP (Chrome DevTools Protocol) Fetch interception.
    
    This is a robust download function that uses CDP to intercept network requests
    and capture file content, bypassing many anti-bot measures.
    
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
        "Chrome/142.0.0.0 Safari/537.36"
    )
    user_agent = user_agent or default_user_agent
    
    output_path = Path(output_file)
    output_dir = output_path.parent
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    playwright = None
    file_content = None
    
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=headless)
        
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
                nonlocal file_content
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
                    
                    # Check if it's a PDF (magic bytes) or other binary content
                    # Accept any content that looks like a file (not HTML)
                    content_type = params.get("responseHeaders", {}).get("content-type", "")
                    is_binary = (
                        len(data) > 100 and data[:4] == b"%PDF"  # PDF
                        or "application/pdf" in content_type.lower()
                        or "application/octet-stream" in content_type.lower()
                        or "application/zip" in content_type.lower()
                        or len(data) > 1000  # Large files are likely binary
                    )
                    
                    if is_binary and (file_content is None or len(data) > len(file_content or b"")):
                        file_content = data
                
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
            await page.goto(url, timeout=timeout, wait_until="domcontentloaded")
            
            # Wait for content to load
            await asyncio.sleep(wait_time)
            
            # If CDP fetch didn't capture content, try download handler as fallback
            if file_content is None:
                # Try download handler approach
                with tempfile.TemporaryDirectory() as temp_dir:
                    download_context = await browser.new_context(
                        user_agent=user_agent,
                        accept_downloads=True,
                    )
                    download_page = await download_context.new_page()
                    
                    try:
                        download = None
                        try:
                            async with download_page.expect_download(timeout=timeout) as download_info:
                                try:
                                    await download_page.goto(url, timeout=timeout, wait_until="commit")
                                except Exception:
                                    # "Download is starting" is expected when navigation triggers a download
                                    pass
                            
                            download = await download_info.value
                            await download.save_as(output_path)
                            file_content = output_path.read_bytes()
                        except Exception:
                            # Download handler failed, try to get content from page
                            try:
                                # Wait a bit more for content
                                await asyncio.sleep(wait_time)
                                # Try to get content via page content
                                content = await download_page.content()
                                if len(content) > 1000:
                                    # Might be HTML, not what we want
                                    pass
                            except Exception:
                                pass
                    finally:
                        await download_context.close()
        
        finally:
            await context.close()
            await browser.close()
    
    except Exception as e:
        return DownloadResult(
            success=False,
            error=f"CDP download failed: {str(e)}",
        )
    
    finally:
        if playwright:
            await playwright.stop()
    
    # Save the file if we got content
    if file_content:
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
        error="No file content captured via CDP or download handler",
    )
