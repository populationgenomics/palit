#!/usr/bin/env python3

"""Download papers workflow: Automated PMC/preprint download and PDF matching."""

import json
import logging
import random
import sqlite3
import tempfile
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from urllib.parse import urlparse

import boto3
import httpx2
import typer
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from pypdf import PdfReader, PdfWriter
from rich.console import Console
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from palit.panelapp_client import PanelAppClient
from palit.papers import doi_to_path
from palit.progress import LoggingProgress as Progress

console = Console()
app = typer.Typer(help="Download papers workflow: Automated PMC/preprint download and PDF matching")

logger = logging.getLogger(__name__)


def read_dois_from_db(
    db_path: Path, skip_hgnc_ids: list[int] | None = None, expansion_only: bool = False
) -> list[str]:
    """Read DOIs from database where papers require manual download.

    Args:
        db_path: Path to database
        skip_hgnc_ids: Optional list of HGNC IDs to skip papers for
        expansion_only: Only include papers with source_type='expansion'
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Build base WHERE clause
        where_clauses = ["p.download_status = 'manual_required'"]
        if expansion_only:
            where_clauses.append("p.source_type = 'expansion'")

        base_where = " AND ".join(where_clauses)

        if skip_hgnc_ids:
            placeholders = ",".join("?" * len(skip_hgnc_ids))
            query = f"""
                SELECT DISTINCT p.doi
                FROM papers p
                WHERE {base_where}
                AND p.doi NOT IN (
                    SELECT DISTINCT gm.paper_doi
                    FROM gene_mentions gm
                    WHERE gm.hgnc_id IN ({placeholders})
                )
                ORDER BY p.doi
            """
            cursor.execute(query, skip_hgnc_ids)
        else:
            query = f"""
                SELECT doi FROM papers p
                WHERE {base_where}
                ORDER BY doi
            """
            cursor.execute(query)

        dois = [row[0] for row in cursor.fetchall()]

    if not dois:
        logger.warning("No papers with download_status='manual_required' in database")
    elif skip_hgnc_ids:
        logger.info(f"Excluding papers for HGNC IDs: {skip_hgnc_ids}")

    return dois


def check_existing_files(doi: str, download_dir: Path) -> list[str]:
    """Check which file types exist for a DOI."""
    existing = []
    if doi_to_path(doi, download_dir, ".pdf").exists():
        existing.append("pdf")
    if doi_to_path(doi, download_dir, ".json").exists():
        existing.append("json")
    return existing


def get_green_hgnc_ids_from_panel(panel_date: str) -> list[int]:
    """Get HGNC IDs of genes with GREEN (3) confidence rating from target panels.

    Args:
        panel_date: Date in YYYY-MM-DD format for panel state

    Returns:
        List of HGNC IDs with confidence level 3 (GREEN) in target panels
    """
    panelapp_client = PanelAppClient(panel_date)
    target_panel_data = panelapp_client.get_target_panels_genes()

    green_hgnc_ids = [
        hgnc_id
        for hgnc_id, confidence in target_panel_data.gene_confidence.items()
        if confidence == 3
    ]

    logger.info(f"Found {len(green_hgnc_ids)} GREEN genes in target panels for {panel_date}")
    return green_hgnc_ids


@app.command("open-browser")
def open_browser(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Database path to read DOIs from papers assessed as relevant",
    ),
    target_dir: Path = typer.Option(
        Path("data/papers"), "--target-dir", "-t", help="Directory to check for existing PDFs"
    ),
    browser_delay: float = typer.Option(
        1.0, "--browser-delay", help="Delay in seconds between opening browser tabs"
    ),
    skip_hgnc_ids: str = typer.Option(
        None,
        "--skip-hgnc-ids",
        help="Comma-separated list of HGNC IDs to skip (e.g., 700,994)",
    ),
    exclude_green: bool = typer.Option(
        False,
        "--exclude-green",
        help="Exclude papers for genes with existing GREEN (3) confidence in target panels",
    ),
    panel_date: str = typer.Option(
        None,
        "--panel-date",
        help="Panel state date (YYYY-MM-DD), required when using --exclude-green",
    ),
    expansion_only: bool = typer.Option(
        False,
        "--expansion-only",
        help="Only include papers from literature expansion (source_type='expansion')",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print URLs that would be opened without actually opening browser",
    ),
) -> None:
    """Open DOI links in browser for manual PDF download."""

    # Validate parameters
    if exclude_green and not panel_date:
        console.print("[red]Error: --panel-date is required when using --exclude-green[/red]")
        raise typer.Exit(1)

    # Parse skip HGNC IDs from command line
    skip_id_list = [int(x.strip()) for x in skip_hgnc_ids.split(",")] if skip_hgnc_ids else []

    # Add GREEN genes from panel if --exclude-green is set
    if exclude_green:
        green_ids = get_green_hgnc_ids_from_panel(panel_date)
        skip_id_list.extend(green_ids)
        console.print(
            f"[cyan]Excluding {len(green_ids)} GREEN genes from panel at {panel_date}[/cyan]"
        )

    dois = read_dois_from_db(db_path, skip_id_list, expansion_only)
    if not dry_run:
        console.print(f"[bold]Found {len(dois)} papers marked for download in database[/bold]")
    if expansion_only:
        console.print("[cyan]Filtering to expansion papers only (source_type='expansion')[/cyan]")

    # Filter out papers that already have PDFs
    dois_needing_download = []
    existing_pdfs = 0

    for doi in dois:
        existing = check_existing_files(doi, target_dir)
        if "pdf" not in existing:
            dois_needing_download.append(doi)
        else:
            existing_pdfs += 1

    if existing_pdfs > 0:
        console.print(
            f"[dim]Skipping {existing_pdfs} papers that already have PDFs in {target_dir}[/dim]"
        )

    if not dois_needing_download:
        console.print("[green]All papers already have PDFs - no downloads needed![/green]")
        return

    # If dry-run, just print URLs and exit
    if dry_run:
        for doi in dois_needing_download:
            console.print(f"https://doi.org/{doi}")
        return

    console.print(
        f"\n[yellow]Opening {len(dois_needing_download)} DOI links in browser for manual PDF download...[/yellow]"
    )
    console.print(f"[dim]Browser delay: {browser_delay} seconds between tabs[/dim]\n")

    for i, doi in enumerate(dois_needing_download, 1):
        url = f"https://doi.org/{doi}"
        console.print(f"[{i}/{len(dois_needing_download)}] Opening: {url}")
        webbrowser.open(url)

        if i < len(dois_needing_download) and browser_delay > 0:
            time.sleep(browser_delay)

    console.print(
        f"\n[bold green]Opened {len(dois_needing_download)} DOI links in browser[/bold green]"
    )
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("  1. Download PDFs manually to data/papers/")
    console.print("  2. Convert PDFs: [dim]uv run palit docling convert[/dim]")
    console.print("  3. Register papers: [dim]uv run palit download-papers register[/dim]")


@app.command("register")
def register_papers(
    papers_dir: Path = typer.Option(
        default=Path("data/papers"), help="Directory containing converted JSON files"
    ),
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Database path",
    ),
) -> None:
    """Register converted papers by updating download_status for DOIs with JSON files."""

    if not papers_dir.exists():
        console.print(f"[red]Papers directory not found: {papers_dir}[/red]")
        raise typer.Exit(1)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Find papers that need registration
        cursor.execute("""
            SELECT doi, download_status FROM papers
            WHERE download_status IN ('scheduled', 'manual_required')
        """)
        candidates = cursor.fetchall()

        if not candidates:
            console.print("[green]No papers pending registration.[/green]")
            return

        registered_count = 0
        missing_json = 0

        for row in candidates:
            doi = row["doi"]
            json_path = doi_to_path(doi, papers_dir, ".json")

            if not json_path.exists():
                missing_json += 1
                continue

            cursor.execute(
                "UPDATE papers SET download_status = 'downloaded' WHERE doi = ?",
                (doi,),
            )
            logger.info(f"Registered DOI {doi} (was: {row['download_status']})")
            registered_count += 1

        conn.commit()

    console.print("\n[bold]Registration Summary:[/bold]")
    console.print(f"  Registered papers: {registered_count}")
    console.print(f"  Still missing JSON: {missing_json}")

    if registered_count > 0:
        console.print(
            "\n[green]Registration complete! Papers ready for evidence extraction.[/green]"
        )


PMC_OA_BUCKET = "pmc-oa-opendata"
_S3_CONFIG = Config(signature_version=UNSIGNED, region_name="us-east-1")


def _make_s3_client():  # type: ignore[no-untyped-def]
    """Create an anonymous S3 client for the PMC OA bucket."""
    return boto3.client("s3", config=_S3_CONFIG)


def concatenate_pdfs(pdf_paths: list[Path], output_path: Path) -> None:
    """Concatenate multiple PDFs into a single file."""
    writer = PdfWriter()

    for pdf_path in pdf_paths:
        writer.append(str(pdf_path))

    with open(output_path, "wb") as output_file:
        writer.write(output_file)


def _filter_supplements_by_page_count(
    supplement_pdfs: list[Path],
    *,
    max_pages_per_supplement: int = 20,
    max_total_supplement_pages: int = 50,
) -> list[Path]:
    """Filter supplement PDFs by page count to avoid timeouts on huge table dumps."""
    accepted: list[Path] = []
    total_pages = 0

    for pdf_path in supplement_pdfs:
        try:
            pages = len(PdfReader(pdf_path).pages)
        except Exception as e:
            logger.warning("Cannot read page count for %s, skipping: %s", pdf_path.name, e)
            continue

        if pages > max_pages_per_supplement:
            logger.info(
                "Skipping supplement %s: %d pages (limit %d)",
                pdf_path.name,
                pages,
                max_pages_per_supplement,
            )
            continue

        if total_pages + pages > max_total_supplement_pages:
            logger.info(
                "Skipping supplement %s: %d pages would exceed total budget (%d/%d)",
                pdf_path.name,
                pages,
                total_pages,
                max_total_supplement_pages,
            )
            continue

        accepted.append(pdf_path)
        total_pages += pages

    return accepted


@dataclass
class DownloadResult:
    """Result of attempting to download a single paper."""

    doi: str
    status: str  # 'downloaded' | 'manual_required' | 'error'
    message: str


def _resolve_latest_version(s3, pmcid: str) -> int | None:  # type: ignore[no-untyped-def]
    """Find the latest version number for a PMCID in the OA bucket."""
    response = s3.list_objects_v2(Bucket=PMC_OA_BUCKET, Prefix=f"{pmcid}.", Delimiter="/")
    prefixes = response.get("CommonPrefixes", [])
    if not prefixes:
        return None

    versions: list[int] = []
    for p in prefixes:
        prefix_str = p["Prefix"].rstrip("/")
        version_str = prefix_str.rsplit(".", 1)[-1]
        try:
            versions.append(int(version_str))
        except ValueError:
            continue

    return max(versions) if versions else None


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=2, max=30),
    retry=retry_if_exception_type(ClientError),
    reraise=True,
)
def _download_s3_object(s3, key: str) -> bytes:  # type: ignore[no-untyped-def]
    """Download a single object from the PMC OA bucket with retries."""
    response = s3.get_object(Bucket=PMC_OA_BUCKET, Key=key)
    data: bytes = response["Body"].read()
    return data


def _s3_url_to_key(s3_url: str) -> str:
    """Convert an S3 URL from metadata JSON to a plain key.

    Strips the 's3://bucket/' prefix and any '?md5=...' query parameter.
    """
    path = s3_url.split(f"s3://{PMC_OA_BUCKET}/")[-1]
    return path.split("?")[0]


# Cloudflare fronts the preprint servers and rate-limits per host. Bounding
# in-flight requests per host rather than by worker-pool size lets a batch that
# mixes bioRxiv, medRxiv and Research Square still use the whole pool.
_MAX_IN_FLIGHT_PER_HOST = 2

# The default python-httpx2 User-Agent scores poorly with Cloudflare's bot
# management. Identify the tool rather than impersonating a browser.
_USER_AGENT = f"palit/{package_version('palit')} (open-access preprint retrieval)"

# A rate-limit window lasts far longer than a plain exponential backoff waits,
# so the 429 loop is both longer-lived and separately bounded.
_MAX_429_ATTEMPTS = 6
_BASE_429_WAIT_S = 15.0
_MAX_429_WAIT_S = 300.0


def _is_retryable_http(exc: BaseException) -> bool:
    """Transport failures and server errors are worth retrying; other 4xx are final.

    429 never reaches here: it is handled in-band against ``Retry-After``, and
    exhausting that loop is terminal for the run.
    """
    if isinstance(exc, httpx2.TransportError):
        return True
    if isinstance(exc, httpx2.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


class PreprintDownloader:
    """Fetches preprint PDFs from behind Cloudflare's per-host rate limiter.

    One shared client keeps the connection pool and Cloudflare's session cookies
    alive for the whole run, so each PDF is not evaluated as a fresh visitor. A
    per-host semaphore bounds concurrency so a wide worker pool cannot overrun a
    single server. 429s honour ``Retry-After``; 5xx and transport errors back
    off through tenacity.
    """

    def __init__(self, timeout: float) -> None:
        self._http = httpx2.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )
        self._semaphores: dict[str, threading.Semaphore] = {}
        self._semaphores_lock = threading.Lock()

    def close(self) -> None:
        self._http.close()

    def _semaphore_for(self, url: str) -> threading.Semaphore:
        host = urlparse(url).netloc
        with self._semaphores_lock:
            semaphore = self._semaphores.get(host)
            if semaphore is None:
                semaphore = threading.Semaphore(_MAX_IN_FLIGHT_PER_HOST)
                self._semaphores[host] = semaphore
            return semaphore

    def fetch(self, url: str) -> httpx2.Response:
        """GET ``url``, waiting out rate limits and retrying transient failures."""
        with self._semaphore_for(url):
            return self._fetch_with_retry(url)

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=2, max=60),
        retry=retry_if_exception(_is_retryable_http),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _fetch_with_retry(self, url: str) -> httpx2.Response:
        for attempt in range(_MAX_429_ATTEMPTS):
            response = self._http.get(url)
            if response.status_code != 429:
                response.raise_for_status()
                return response
            delay = _retry_after_delay(response, attempt)
            logger.warning(f"Rate limited on {url}; waiting {delay:.0f}s before retry")
            time.sleep(delay)
        # Still limited after the full budget — surface the 429 so the caller
        # records an error and the next run picks the paper up again.
        response.raise_for_status()
        raise RuntimeError("unreachable: 429 loop fell through without raising")


def _retry_after_delay(response: httpx2.Response, attempt: int) -> float:
    """Seconds to wait after a 429, preferring the server's own guidance.

    RFC 9110 also permits an HTTP-date in ``Retry-After``; the preprint servers
    send delta-seconds, and anything else falls back to our own jittered
    backoff rather than failing the run over a header.
    """
    retry_after = response.headers.get("Retry-After", "")
    if retry_after.isdigit():
        return min(float(retry_after), _MAX_429_WAIT_S)
    return min(_BASE_429_WAIT_S * 2.0**attempt, _MAX_429_WAIT_S) + random.uniform(0, 5)


def download_pmc_paper(
    doi: str,
    pmcid: str | None,
    db_path: Path,
    target_dir: Path,
    timeout: float,
) -> DownloadResult:
    """Download a single paper from PMC using its PMCID via the pmc-oa-opendata S3 bucket."""
    if not pmcid:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE papers SET download_status = 'manual_required' WHERE doi = ?",
                (doi,),
            )
        return DownloadResult(doi, "manual_required", "No PMCID in source_metadata")

    try:
        s3 = _make_s3_client()

        # Find latest version in S3 bucket
        version = _resolve_latest_version(s3, pmcid)
        if version is None:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE papers SET download_status = 'manual_required' WHERE doi = ?",
                    (doi,),
                )
            return DownloadResult(doi, "manual_required", f"{pmcid} not in PMC OA bucket")

        # Fetch per-article metadata JSON
        metadata_key = f"metadata/{pmcid}.{version}.json"
        metadata = json.loads(_download_s3_object(s3, metadata_key))

        pdf_url = metadata.get("pdf_url")
        if not pdf_url:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE papers SET download_status = 'manual_required' WHERE doi = ?",
                    (doi,),
                )
            return DownloadResult(
                doi, "manual_required", f"No pdf_url in metadata for {pmcid}.{version}"
            )

        # Identify main PDF and supplement PDFs from metadata
        main_pdf_key = _s3_url_to_key(pdf_url)
        supplement_keys = [
            _s3_url_to_key(url)
            for url in metadata.get("media_urls", [])
            if url.lower().split("?")[0].endswith(".pdf")
        ]

        # Download all PDFs
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            main_path = tmpdir_path / "main.pdf"
            main_path.write_bytes(_download_s3_object(s3, main_pdf_key))

            supplement_paths: list[Path] = []
            for i, key in enumerate(supplement_keys):
                path = tmpdir_path / f"supplement_{i:03d}.pdf"
                path.write_bytes(_download_s3_object(s3, key))
                supplement_paths.append(path)

            supplement_paths = _filter_supplements_by_page_count(supplement_paths)
            all_pdfs = [main_path, *supplement_paths]

            pdf_path = doi_to_path(doi, target_dir, ".pdf")
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            concatenate_pdfs(all_pdfs, pdf_path)

        # Update status to downloaded
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE papers SET download_status = 'downloaded' WHERE doi = ?",
                (doi,),
            )

        n_supps = len(supplement_paths)
        return DownloadResult(
            doi,
            "downloaded",
            f"Downloaded ({pmcid}.{version}) - {len(all_pdfs)} PDFs ({n_supps} supplements)",
        )

    except Exception as e:
        return DownloadResult(doi, "error", str(e))


@app.command("attempt-pmc")
def attempt_pmc(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Database path",
    ),
    target_dir: Path = typer.Option(
        Path("data/papers"), "--target-dir", "-t", help="Directory to save PDFs"
    ),
    timeout: float = typer.Option(30.0, "--timeout", help="HTTP request timeout in seconds"),
    max_workers: int = typer.Option(5, "--max-workers", "-w", help="Number of parallel downloads"),
) -> None:
    """Attempt automated PMC download for scheduled papers.

    For each paper with download_status='scheduled', PMCID in source_metadata, and missing PDF:
    1. Query OA API for PDF download link using stored PMCID
    2. Download PDF if available
    3. Update status to 'downloaded' or 'manual_required'
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    target_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT doi, json_extract(source_metadata, '$.pmcid') AS pmcid
            FROM papers
            WHERE download_status = 'scheduled'
              AND source = 'pubmed'
            ORDER BY doi
        """)
        scheduled = cursor.fetchall()

    if not scheduled:
        console.print("[yellow]No papers with download_status='scheduled'[/yellow]")
        return

    # Filter out papers that already have PDFs
    papers_needing_download = [
        (doi, pmcid)
        for doi, pmcid in scheduled
        if not doi_to_path(doi, target_dir, ".pdf").exists()
    ]

    if not papers_needing_download:
        console.print("[green]All scheduled papers already have PDFs![/green]")
        return

    console.print(
        f"[cyan]Attempting PMC download for {len(papers_needing_download)} papers "
        f"({max_workers} workers)[/cyan]"
    )

    downloaded = 0
    manual_required = 0
    errors: list[tuple[str, str]] = []

    with Progress() as progress:
        task = progress.add_task("Downloading...", total=len(papers_needing_download))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_pmc_paper, doi, pmcid, db_path, target_dir, timeout): doi
                for doi, pmcid in papers_needing_download
            }

            for future in as_completed(futures):
                result = future.result()
                if result.status == "downloaded":
                    downloaded += 1
                    logger.info(f"DOI {result.doi}: {result.message}")
                elif result.status == "manual_required":
                    manual_required += 1
                    logger.info(f"DOI {result.doi}: {result.message}")
                else:  # error
                    errors.append((result.doi, result.message))
                    logger.error(f"DOI {result.doi}: {result.message}")
                progress.advance(task)

    # Summary
    console.print("\n[bold]PMC Download Summary:[/bold]")
    console.print(f"  Downloaded: {downloaded}")
    console.print(f"  Manual required: {manual_required}")
    if errors:
        console.print(f"  Errors: {len(errors)}")
        for doi, error in errors[:5]:
            console.print(f"     {doi}: {error}")
        if len(errors) > 5:
            console.print(f"     ... and {len(errors) - 5} more errors")


def _build_preprint_url(source: str, doi: str, version: int) -> str:
    """Construct the direct PDF download URL for a preprint."""
    if source in ("biorxiv", "medrxiv"):
        return f"https://www.{source}.org/content/{doi}v{version}.full.pdf"
    # researchsquare: DOI like "10.21203/rs.3.rs-7989161" → article ID "rs-7989161"
    article_id = doi.removeprefix("10.21203/rs.3.")
    return f"https://www.researchsquare.com/article/{article_id}/v{version}.pdf"


def download_preprint_paper(
    doi: str,
    source: str,
    version: int,
    db_path: Path,
    target_dir: Path,
    downloader: PreprintDownloader,
) -> DownloadResult:
    """Download a single preprint PDF by constructing its URL from metadata."""
    url = _build_preprint_url(source, doi, version)

    try:
        response = downloader.fetch(url)

        content_type = response.headers.get("content-type", "")
        if "application/pdf" not in content_type:
            return DownloadResult(
                doi, "error", f"Not a PDF (content-type: {content_type}, url: {url})"
            )

        pdf_path = doi_to_path(doi, target_dir, ".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(response.content)

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE papers SET download_status = 'downloaded' WHERE doi = ?",
                (doi,),
            )

        size_kb = len(response.content) / 1024
        return DownloadResult(doi, "downloaded", f"Downloaded from {source} ({size_kb:.0f} KB)")

    except Exception as e:
        return DownloadResult(doi, "error", f"{url}: {e}")


@app.command("download-preprints")
def download_preprints_cmd(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Database path",
    ),
    target_dir: Path = typer.Option(
        Path("data/papers"), "--target-dir", "-t", help="Directory to save PDFs"
    ),
    timeout: float = typer.Option(30.0, "--timeout", help="HTTP request timeout in seconds"),
    max_workers: int = typer.Option(
        5,
        "--max-workers",
        "-w",
        help=f"Parallel downloads (capped at {_MAX_IN_FLIGHT_PER_HOST} per host)",
    ),
) -> None:
    """Download PDFs for scheduled preprint papers (bioRxiv, medRxiv, Research Square).

    Constructs PDF URLs from DOI + version in source_metadata and downloads directly.
    All preprints are open access. Any download failure aborts with non-zero exit;
    papers that fail keep download_status='scheduled', so a rerun retries only those.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    target_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT doi, source,
                   json_extract(source_metadata, '$.version') AS version
            FROM papers
            WHERE download_status = 'scheduled'
              AND source IN ('biorxiv', 'medrxiv', 'researchsquare')
            ORDER BY doi
        """)
        scheduled = cursor.fetchall()

    if not scheduled:
        console.print("[yellow]No preprint papers with download_status='scheduled'[/yellow]")
        return

    # Filter out papers that already have PDFs
    papers_needing_download = [
        (doi, source, int(version))
        for doi, source, version in scheduled
        if not doi_to_path(doi, target_dir, ".pdf").exists()
    ]

    if not papers_needing_download:
        console.print("[green]All scheduled preprints already have PDFs![/green]")
        return

    console.print(
        f"[cyan]Downloading {len(papers_needing_download)} preprint PDFs "
        f"({max_workers} workers)[/cyan]"
    )

    downloaded = 0
    errors: list[tuple[str, str]] = []
    downloader = PreprintDownloader(timeout)

    try:
        with Progress() as progress:
            task = progress.add_task("Downloading...", total=len(papers_needing_download))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        download_preprint_paper,
                        doi,
                        source,
                        version,
                        db_path,
                        target_dir,
                        downloader,
                    ): doi
                    for doi, source, version in papers_needing_download
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result.status == "downloaded":
                        downloaded += 1
                        logger.info(f"DOI {result.doi}: {result.message}")
                    else:
                        errors.append((result.doi, result.message))
                        logger.error(f"DOI {result.doi}: {result.message}")
                    progress.advance(task)
    finally:
        downloader.close()

    console.print("\n[bold]Preprint Download Summary:[/bold]")
    console.print(f"  Downloaded: {downloaded}")

    if errors:
        console.print(f"  [red]Errors: {len(errors)}[/red]")
        for doi, error in errors:
            pdf_path = doi_to_path(doi, target_dir, ".pdf")
            console.print(f"    {doi}: {error}")
            console.print(f"      Save to: {pdf_path}")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
