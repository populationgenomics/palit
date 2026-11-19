"""Downloads of the reference files that stages read next to the run database."""

import logging
import time
from pathlib import Path

import httpx2

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_AGE_SECONDS = 7 * 24 * 3600


def download_if_stale(url: str, path: Path) -> None:
    """Download *url* to *path* unless *path* exists and is less than a week old."""
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < MAX_DOWNLOAD_AGE_SECONDS:
            logger.info("Using cached %s (age: %.0fh)", path, age / 3600)
            return
        logger.info("Re-downloading stale %s (age: %.0fh)", path, age / 3600)
    else:
        logger.info("Downloading %s", url)

    with httpx2.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(url)
        response.raise_for_status()
        path.write_bytes(response.content)
    logger.info("Downloaded %s bytes to %s", f"{len(response.content):,}", path)
