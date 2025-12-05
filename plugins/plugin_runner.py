#!/usr/bin/env python3
"""Lightweight CLI wrapper to run download plugins from Rust fallback paths."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plugins.utils import download_with_detected_plugin, normalize_doi


async def run() -> int:
    parser = argparse.ArgumentParser(description="Plugin-backed PDF downloader")
    parser.add_argument("--doi", required=True, help="DOI to download")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write the downloaded PDF",
    )
    parser.add_argument(
        "--filename",
        help="Optional filename for the downloaded PDF (extension is appended if missing)",
    )
    parser.add_argument(
        "--wait-time",
        type=float,
        default=5.0,
        help="Seconds to wait for plugin downloads before timing out",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser-based plugins in headless mode",
    )
    args = parser.parse_args()

    doi = normalize_doi(args.doi)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = await download_with_detected_plugin(
            doi=doi,
            output_dir=output_dir,
            filename=args.filename,
            wait_time=args.wait_time,
            plugin_options={"wiley": {"headless": args.headless}},
        )
        payload = {
            "success": result.success,
            "file_path": str(result.file_path) if result.file_path else None,
            "file_size": result.file_size,
            "error": result.error,
            "publisher": result.publisher,
            "doi": result.doi,
        }
    except Exception as exc:  # pragma: no cover - defensive
        payload = {
            "success": False,
            "file_path": None,
            "file_size": 0,
            "error": str(exc),
            "publisher": None,
            "doi": doi,
        }

    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()
    return 0


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
