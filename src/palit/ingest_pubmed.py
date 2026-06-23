#!/usr/bin/env python3
"""Download and ingest PubMed papers into database."""

import gzip
import logging
import sqlite3
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import typer
from rich.console import Console

from palit import ledger as ledger_ops
from palit import pubmed_ftp
from palit.papers import serialize_source_metadata
from palit.progress import LoggingProgress as Progress
from palit.pubmed_xml import extract_papers_from_xml

console = Console()
app = typer.Typer(help="Download and ingest PubMed papers")

logger = logging.getLogger(__name__)

# Configuration
MIN_FILE_SIZE = 1000  # Minimum expected file size in bytes
MAX_RETRIES = 5
RETRY_DELAY = 2  # Seconds between retries


def process_xml_file(
    xml_path: Path,
    start_date: str,
    end_date: str,
    output_db: Path,
) -> None:
    """Extract papers from one efetch day-dump into a standalone database.

    Used by the periodic backfill-diff backstop, which materialises the current
    PubMed index for a window into a scratch database. Routine monthly ingestion
    no longer writes the run database directly: it upserts into the ledger and
    seeds the run database from the ledger's actionable set.
    """
    logger.info(f"Processing XML file: {xml_path}")
    logger.debug(f"Date range: {start_date} to {end_date}")
    logger.debug(f"Output database: {output_db}")

    # Connect to database (schema must exist)
    conn = sqlite3.connect(output_db)

    try:
        with gzip.open(xml_path, "rb") as f:
            logger.debug("Parsing XML file...")
            xml_content = f.read()

        # For initial papers, use the filename as source_details
        source_type = "initial"
        source_details = xml_path.name
        all_papers, stats = extract_papers_from_xml(xml_content, source_type, source_details)
        logger.debug(f"Found {stats.extracted} papers (from {stats.total_articles} articles)")

        papers = [paper for paper in all_papers if start_date <= paper.source_date <= end_date]
        logger.debug(f"Found {len(papers)} papers in date range")

        # Insert papers into database
        if papers:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO papers
                (doi, pmid, title, abstract, authors, journal, source, source_date,
                 source_metadata, source_type, source_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doi) DO UPDATE SET
                    pmid = excluded.pmid,
                    title = excluded.title,
                    abstract = excluded.abstract,
                    authors = excluded.authors,
                    journal = excluded.journal,
                    source_date = excluded.source_date,
                    source_metadata = excluded.source_metadata,
                    source_type = excluded.source_type,
                    source_details = excluded.source_details
                -- Only update within the same source (e.g., newer PubMed file
                -- overwrites older PubMed file). Never cross-source: a preprint
                -- ingested first must not be overwritten by PubMed, and vice versa.
                WHERE excluded.source = papers.source
                  AND excluded.source_type = 'initial'
                  AND papers.source_type = 'initial'
                  AND excluded.source_details > papers.source_details
            """,
                [
                    (
                        p.doi,
                        p.pmid,
                        p.title,
                        p.abstract,
                        p.authors,
                        p.journal,
                        p.source,
                        p.source_date,
                        serialize_source_metadata(p.source_metadata),
                        p.source_type,
                        p.source_details,
                    )
                    for p in papers
                ],
            )

            # Backfill PMIDs into papers ingested from other sources (e.g., preprints).
            # The ON CONFLICT above skips cross-source updates, but we still want the
            # PMID that PubMed provides.
            cursor.executemany(
                "UPDATE papers SET pmid = ? WHERE doi = ? AND pmid IS NULL",
                [(p.pmid, p.doi) for p in papers if p.pmid is not None],
            )

            conn.commit()

        logger.info(f"Inserted {len(papers)} papers into database")

    finally:
        conn.close()


@dataclass
class DownloadResult:
    """Result of downloading a single day's papers."""

    day: date
    success: bool
    file_path: Path | None
    file_size: int
    attempts: int
    error: str | None = None


def download_day(day: date, output_dir: Path) -> DownloadResult:
    """Download PubMed papers for a single day with retry logic.

    Args:
        day: Date to download
        output_dir: Directory to save XML files

    Returns:
        DownloadResult with success status and metadata
    """
    output_file = output_dir / f"pubmed_{day}.xml.gz"

    # Always re-fetch, overwriting any cached dump. This is the thin live recency
    # guard over the current ingest window: live efetch is the freshest view of the
    # newest papers, where the FTP update files can sporadically lag. Late-indexed
    # stragglers with older create dates are caught by the FTP update-file sync
    # instead, so this fetch stays bounded to the run's own date window.
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Build esearch and efetch pipeline. We use efetch, as the daily update files at
            # https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/ can sometimes lag behind significantly.
            # We use CRDT (Create Date) rather than EDAT (Entrez Date) because empirically CRDT
            # matches PubMedPubDate[@PubStatus="entrez"] in the XML (which we extract as entrez_date),
            # while EDAT can return papers on a different day. See:
            # https://www.nlm.nih.gov/pubs/techbull/nd08/nd08_pm_new_date_field.html
            date_str = f"{day.year}/{day.month:02d}/{day.day:02d}"
            esearch_cmd = [
                "esearch",
                "-db",
                "pubmed",
                "-query",
                "hasabstract",
                "-datetype",
                "CRDT",
                "-mindate",
                date_str,
                "-maxdate",
                date_str,
            ]

            efetch_cmd = ["efetch", "-format", "xml"]

            # Run pipeline
            esearch_proc = subprocess.Popen(
                esearch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            efetch_proc = subprocess.Popen(
                efetch_cmd,
                stdin=esearch_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if esearch_proc.stdout:
                esearch_proc.stdout.close()  # Allow esearch to receive SIGPIPE

            # Get XML output and compress with gzip
            xml_data, _stderr = efetch_proc.communicate()

            # Write compressed data
            with gzip.open(output_file, "wb") as f:
                f.write(xml_data)

            # Check file size
            if output_file.exists():
                file_size = output_file.stat().st_size
                if file_size >= MIN_FILE_SIZE:
                    return DownloadResult(
                        day=day,
                        success=True,
                        file_path=output_file,
                        file_size=file_size,
                        attempts=attempt,
                    )
                else:
                    logger.warning(
                        f"{day}: File too small ({file_size} bytes), "
                        f"attempt {attempt}/{MAX_RETRIES}"
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)

        except (OSError, subprocess.SubprocessError) as e:
            # Catch expected retryable errors (network, subprocess, file I/O)
            logger.warning(f"{day}: Download failed (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        # Let unexpected exceptions propagate to parent

    # All retries exhausted - skip this day
    file_size = output_file.stat().st_size if output_file.exists() else 0
    logger.warning(
        f"{day}: Skipping after {MAX_RETRIES} attempts. "
        f"Final file size: {file_size} bytes (minimum required: {MIN_FILE_SIZE})"
    )
    return DownloadResult(
        day=day,
        success=False,
        file_path=None,
        file_size=file_size,
        attempts=MAX_RETRIES,
        error=f"File too small after {MAX_RETRIES} attempts",
    )


def extract_papers(
    xml_files: list[Path],
    start_date: str,
    end_date: str,
    db_path: Path,
    parallel_jobs: int,
) -> None:
    """Extract papers from efetch day-dumps into a standalone DB, in parallel.

    Used by the backfill-diff backstop (see `process_xml_file`); routine monthly
    ingestion goes through the ledger.

    Args:
        xml_files: List of XML files to process
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        db_path: Path to database
        parallel_jobs: Number of parallel jobs
    """
    with Progress(console=console) as progress:
        task = progress.add_task("Extracting papers...", total=len(xml_files))

        with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = {
                executor.submit(process_xml_file, xml_file, start_date, end_date, db_path): xml_file
                for xml_file in xml_files
            }

            for future in as_completed(futures):
                future.result()
                progress.advance(task)


def _live_window_to_ledger(
    ledger_conn: sqlite3.Connection,
    start_dt: date,
    end_dt: date,
    output_dir: Path,
    parallel_jobs: int,
    today: str,
) -> None:
    """Fetch the current window via live efetch and upsert it into the ledger.

    The recency guard for the newest papers: each day in [start, end] is fetched
    fresh (overwriting any cached dump) and upserted into the ledger. Disposition
    is untouched -- a brand-new paper lands as actionable, a refreshed one just
    has its metadata updated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    days: list[date] = []
    current = start_dt
    while current <= end_dt:
        days.append(current)
        current += timedelta(days=1)

    results: list[DownloadResult] = []
    with Progress(console=console) as progress:
        task = progress.add_task("Fetching live days...", total=len(days))
        with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = {executor.submit(download_day, day, output_dir): day for day in days}
            for future in as_completed(futures):
                results.append(future.result())
                progress.advance(task)

    # A day with no abstract-bearing papers yields an empty (too-small) file; that
    # is expected, not an error.
    empty_days = sum(1 for r in results if not r.success)
    if empty_days:
        logger.info("No data for %d/%d days in the live window", empty_days, len(days))

    xml_files = sorted(r.file_path for r in results if r.success and r.file_path is not None)
    total_inserted = total_updated = 0
    with Progress(console=console) as progress:
        task = progress.add_task("Upserting live window into ledger...", total=len(xml_files))
        for xml_file in xml_files:
            with gzip.open(xml_file, "rb") as f:
                xml_content = f.read()
            papers, _stats = extract_papers_from_xml(xml_content, "initial", xml_file.name)
            inserted, updated = ledger_ops.upsert_papers(ledger_conn, papers, today)
            total_inserted += inserted
            total_updated += updated
            progress.advance(task)
    console.print(
        f"[green]✓[/green] Live window: {total_inserted} new, {total_updated} refreshed in ledger"
    )


def _ensure_run_db(db_path: Path) -> None:
    """Create the run database from schema.sql if it does not yet exist."""
    if db_path.exists():
        return
    console.print("[yellow]Run database not found, creating from schema.sql...[/yellow]")
    schema_file = Path("schema.sql")
    if not schema_file.exists():
        console.print(f"[red]Error: schema.sql not found at {schema_file.absolute()}[/red]")
        raise typer.Exit(1)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema_file.read_text())
    finally:
        conn.close()
    console.print(f"[green]✓[/green] Created run database at {db_path}")


@app.callback(invoke_without_command=True)
def main(
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    ledger: Path = typer.Option(
        ledger_ops.DEFAULT_LEDGER_PATH, "--ledger", help="Ledger database path"
    ),
    db_path: Path = typer.Option(Path("data/db.sqlite"), "--db-path", help="Run database path"),
    output_dir: Path = typer.Option(
        Path("data/pubmed_xml"), "--output-dir", help="Scratch dir for live efetch day-dumps"
    ),
    ftp_work_dir: Path = typer.Option(
        Path("data/pubmed_updatefiles"), "--ftp-work-dir", help="Scratch dir for FTP update files"
    ),
    closure_horizon_months: int = typer.Option(
        ledger_ops.DEFAULT_CLOSURE_HORIZON_MONTHS,
        "--closure-horizon-months",
        help="A CRDT month older than this is finalised and dropped from the actionable set",
    ),
    apply_ftp_sync: bool = typer.Option(
        True, "--sync-ftp/--no-sync-ftp", help="Apply new FTP update files to the ledger first"
    ),
    parallel_jobs: int = typer.Option(8, "--jobs", "-j", help="Number of parallel download jobs"),
) -> None:
    """Ingest PubMed papers for a window through the ledger.

    1. Apply new NCBI FTP update files to the ledger (late-indexed stragglers).
    2. Fetch the current window live and upsert it into the ledger (recency guard).
    3. Seed the run database from the ledger's actionable set within the closure
       horizon -- new papers to assess plus relevant-not-downloaded carry-overs.
    """
    try:
        start_dt = date.fromisoformat(start_date)
        end_dt = date.fromisoformat(end_date)
    except ValueError as e:
        console.print(f"[red]Invalid date format: {e}[/red]")
        raise typer.Exit(1) from e

    if start_dt > end_dt:
        console.print("[red]Start date must be before or equal to end date[/red]")
        raise typer.Exit(1)

    if not ledger.exists():
        console.print(
            f"[red]Ledger not found at {ledger}. Create it first: "
            f"`palit ledger init --ledger {ledger}`.[/red]"
        )
        raise typer.Exit(1)

    floor = ledger_ops.subtract_months(end_dt, closure_horizon_months).isoformat()
    today = date.today().isoformat()
    console.print(f"[cyan]Ingesting PubMed via ledger {ledger}: {start_date} to {end_date}[/cyan]")
    console.print(f"[cyan]Closure horizon floor (CRDT): {floor}[/cyan]\n")

    ledger_conn = ledger_ops.connect(ledger)
    client = pubmed_ftp.make_client()
    try:
        if apply_ftp_sync:
            console.print("[bold]Step 1: Applying FTP update files to the ledger...[/bold]")
            ledger_ops.sync_ftp(
                ledger_conn, min_crdt=floor, work_dir=ftp_work_dir, client=client, today=today
            )
        else:
            console.print("[yellow]Step 1: Skipping FTP sync (--no-sync-ftp)[/yellow]")

        console.print("\n[bold]Step 2: Live recency guard over the current window...[/bold]")
        _live_window_to_ledger(ledger_conn, start_dt, end_dt, output_dir, parallel_jobs, today)
    finally:
        client.close()
        ledger_conn.close()

    console.print("\n[bold]Step 3: Seeding run database from the ledger's actionable set...[/bold]")
    _ensure_run_db(db_path)
    n = ledger_ops.seed_run_db_from_ledger(ledger, db_path, horizon_floor=floor, end_date=end_date)
    console.print(f"[green]✓[/green] Seeded {n} actionable papers into {db_path}")
    console.print(f"\n[green]✓ Ingestion complete! Run database ready at: {db_path}[/green]")


if __name__ == "__main__":
    app()
