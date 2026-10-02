#!/usr/bin/env python3
"""Download locations and a staleness-checked download for the GenCC export and MONDO."""

import logging
import time
from pathlib import Path

import httpx2

logger = logging.getLogger(__name__)

GENCC_URL = "https://search.thegencc.org/download/action/submissions-export-tsv"
MONDO_OBO_URL = "https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo"

_MAX_AGE_SECONDS = 7 * 24 * 3600  # 1 week


def download_if_stale(url: str, path: Path) -> None:
    """Download a file if it doesn't exist or is older than 1 week."""
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < _MAX_AGE_SECONDS:
            logger.info(f"Using cached {path} (age: {age / 3600:.0f}h)")
            return
        logger.info(f"Re-downloading stale {path} (age: {age / 3600:.0f}h)")
    else:
        logger.info(f"Downloading {url}")

    with httpx2.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(url)
        response.raise_for_status()
        path.write_bytes(response.content)
    logger.info(f"Downloaded {len(response.content):,} bytes → {path}")
