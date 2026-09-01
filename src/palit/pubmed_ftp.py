#!/usr/bin/env python3
"""Fetch primitives for the NCBI PubMed/MEDLINE FTP baseline + update files.

NCBI publishes PubMed citations (with abstracts) as gzipped `PubmedArticleSet`
XML under https://ftp.ncbi.nlm.nih.gov/pubmed/ over HTTPS (the FTP *protocol*
was retired in 2022; the paths are unchanged). The annual baseline
(`pubmed{YY}n0001..N`, released each December) is a full re-snapshot of all of
PubMed; daily *update* files numbered above the baseline range carry new,
revised, and deleted citations. A record indexed long after its create date
appears in whatever update file first adds it — which is why the update-file
stream, applied incrementally by file number, catches late-indexed "straggler"
papers at constant per-run cost.

This module only fetches and verifies files and parses the directory listing /
README. The apply-to-ledger orchestration (which file numbers to apply, parsing
each into `Paper` records, and upserting) lives in `palit.ledger`, which owns the
ledger upsert.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx2
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

logger = logging.getLogger(__name__)

FTP_BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed"
BASELINE_DIR = "baseline"
UPDATEFILES_DIR = "updatefiles"

# pubmed26n1335.xml.gz -> (year=26, number=1335). The `.md5` siblings are excluded
# by anchoring on the `.gz` end.
_FILE_RE = re.compile(r"pubmed(\d{2})n(\d{4})\.xml\.gz(?![.\w])")
# README: "...consists of files pubmed26n0001.xml through pubmed26n1334.xml."
_BASELINE_RANGE_RE = re.compile(
    r"pubmed(\d{2})n(\d{4})\.xml\s+through\s+pubmed(\d{2})n(\d{4})\.xml"
)
# "MD5(pubmed26n1497.xml.gz)= 5dfab57821dba99506ba2ee6c08fade9"
_MD5_RE = re.compile(r"([0-9a-fA-F]{32})")

# Stream downloads in 1 MiB chunks; update files are tens of MB compressed.
_CHUNK_SIZE = 1 << 20


@dataclass(frozen=True, order=True)
class RemoteFile:
    """A `pubmed{YY}n{NNNN}.xml.gz` file on the NCBI FTP host.

    Ordered by (year, number) so a sorted list applies in publication order and
    survives the annual numbering reset (a new year sorts after the old one).
    """

    year: int
    number: int

    @property
    def name(self) -> str:
        return f"pubmed{self.year:02d}n{self.number:04d}.xml.gz"


def make_client() -> httpx2.Client:
    """Build an HTTP client for the NCBI FTP host with a courteous User-Agent."""
    return httpx2.Client(
        headers={"User-Agent": "palit-pubmed-ingest (https://github.com/populationgenomics)"},
        timeout=httpx2.Timeout(120.0, connect=30.0),
        follow_redirects=True,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=2, max=60),
    retry=retry_if_exception_type((httpx2.HTTPStatusError, httpx2.TransportError)),
    reraise=True,
)
def _get_text(client: httpx2.Client, url: str) -> str:
    """GET a small text resource (directory listing, README, .md5) with retries."""
    response = client.get(url)
    response.raise_for_status()
    return response.text


def list_remote_files(client: httpx2.Client, subdir: str) -> list[RemoteFile]:
    """List the `.xml.gz` files in an FTP subdirectory, sorted by (year, number)."""
    listing = _get_text(client, f"{FTP_BASE}/{subdir}/")
    files = {RemoteFile(int(y), int(n)) for y, n in _FILE_RE.findall(listing)}
    return sorted(files)


def read_baseline_max(client: httpx2.Client) -> RemoteFile:
    """Read the highest baseline file number from the baseline README.

    The README always describes the current annual baseline, e.g. "the complete
    baseline consists of files pubmed26n0001.xml through pubmed26n1334.xml". The
    returned file marks where the update-file stream begins (the first update
    file is the next number). We never load the baseline itself.
    """
    readme = _get_text(client, f"{FTP_BASE}/{BASELINE_DIR}/README.txt")
    match = _BASELINE_RANGE_RE.search(readme)
    if match is None:
        raise ValueError("Could not parse baseline file range from baseline/README.txt")
    _, _, end_year, end_number = match.groups()
    return RemoteFile(int(end_year), int(end_number))


def parse_md5(text: str) -> str:
    """Extract the 32-hex-char checksum from an NCBI `.md5` file's contents."""
    match = _MD5_RE.search(text)
    if match is None:
        raise ValueError(f"Could not parse MD5 checksum from: {text!r}")
    return match.group(1).lower()


def _md5_file(path: Path) -> str:
    """Compute the MD5 of a local file, streaming to bound memory."""
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=2, max=60),
    retry=retry_if_exception_type((httpx2.HTTPStatusError, httpx2.TransportError, ValueError)),
    reraise=True,
)
def download_verified(
    client: httpx2.Client, subdir: str, remote: RemoteFile, dest_dir: Path
) -> Path:
    """Download a `.xml.gz` file and verify it against its `.md5` sibling.

    Streams the gzip to `dest_dir/<name>` and checks the MD5 published alongside
    it. A checksum mismatch raises `ValueError`, which the retry wraps — a partial
    or corrupted transfer is re-fetched rather than silently ingested.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / remote.name
    url = f"{FTP_BASE}/{subdir}/{remote.name}"

    with client.stream("GET", url) as response:
        response.raise_for_status()
        with dest.open("wb") as f:
            for chunk in response.iter_bytes(_CHUNK_SIZE):
                f.write(chunk)

    expected = parse_md5(_get_text(client, f"{url}.md5"))
    actual = _md5_file(dest)
    if actual != expected:
        dest.unlink(missing_ok=True)
        raise ValueError(f"MD5 mismatch for {remote.name}: expected {expected}, got {actual}")

    logger.debug("Downloaded and verified %s (%d bytes)", remote.name, dest.stat().st_size)
    return dest
