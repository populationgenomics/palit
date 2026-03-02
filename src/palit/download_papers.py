#!/usr/bin/env python3

"""Download papers workflow: Automated PMC/preprint download and PDF matching."""

import logging
import re
import sqlite3
import tarfile
import tempfile
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
import requests
import typer
from defusedxml import ElementTree as ET
from pypdf import PdfWriter
from requests.adapters import HTTPAdapter
from rich.console import Console
from rich.progress import Progress
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from urllib3.util.retry import Retry

from palit.panelapp_client import PanelAppClient
from palit.papers import doi_to_path

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


def extract_supplements_from_nxml(nxml_path: Path, archive_dir: Path) -> list[Path]:
    """Parse NXML file and extract ordered list of PDF supplement paths.

    Args:
        nxml_path: Path to the .nxml file
        archive_dir: Directory containing extracted archive files

    Returns:
        List of paths to PDF supplements in order from NXML
    """
    supplements = []

    try:
        tree = ET.parse(nxml_path)
        root = tree.getroot()

        # Find all supplementary-material elements (in document order)
        for supp in root.iter():
            if supp.tag.endswith("supplementary-material"):
                # Find media element with href
                for media in supp.iter():
                    if media.tag.endswith("media"):
                        href = None
                        # Check xlink:href attribute (with or without namespace)
                        for attr_name, attr_value in media.attrib.items():
                            if "href" in attr_name:
                                href = attr_value
                                break

                        if href and href.lower().endswith(".pdf"):
                            # Look for this file in the archive
                            matching_files = list(archive_dir.rglob(href))
                            if matching_files:
                                supplements.append(matching_files[0])
                                logger.debug(f"Found supplement PDF: {href}")

    except Exception as e:
        logger.warning(f"Failed to parse NXML for supplements: {e}")

    return supplements


def concatenate_pdfs(pdf_paths: list[Path], output_path: Path) -> None:
    """Concatenate multiple PDFs into a single file.

    Args:
        pdf_paths: List of PDF paths to concatenate (in order)
        output_path: Path to write the concatenated PDF
    """
    writer = PdfWriter()

    for pdf_path in pdf_paths:
        writer.append(str(pdf_path))

    with open(output_path, "wb") as output_file:
        writer.write(output_file)


@dataclass
class DownloadResult:
    """Result of attempting to download a single paper."""

    doi: str
    status: str  # 'downloaded' | 'manual_required' | 'error'
    message: str


def process_tgz_archive(tgz_content: bytes, output_path: Path) -> tuple[bool, str]:
    """Extract TGZ archive, find main PDF, concatenate with supplements.

    Args:
        tgz_content: Raw bytes of the TGZ archive
        output_path: Path to write the final concatenated PDF

    Returns:
        Tuple of (success, message)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        tgz_path = tmpdir_path / "archive.tar.gz"
        tgz_path.write_bytes(tgz_content)

        # Extract archive
        try:
            with tarfile.open(tgz_path, "r:gz") as tar:
                tar.extractall(tmpdir_path)
        except Exception as e:
            return False, f"Failed to extract TGZ: {e}"

        # Find .nxml files
        nxml_files = list(tmpdir_path.rglob("*.nxml"))

        if len(nxml_files) == 0:
            return False, "No NXML file found in archive"
        elif len(nxml_files) > 1:
            return False, f"Multiple NXML files found: {[f.name for f in nxml_files]}"

        nxml_file = nxml_files[0]
        main_pdf = nxml_file.with_suffix(".pdf")

        if not main_pdf.exists():
            return False, f"Main PDF not found (expected {main_pdf.name})"

        # Parse NXML for supplements
        supplement_pdfs = extract_supplements_from_nxml(nxml_file, tmpdir_path)

        # Concatenate: main PDF + supplements
        all_pdfs = [main_pdf, *supplement_pdfs]

        try:
            concatenate_pdfs(all_pdfs, output_path)
            return True, f"Concatenated {len(all_pdfs)} PDFs ({len(supplement_pdfs)} supplements)"
        except Exception as e:
            return False, f"Failed to concatenate PDFs: {e}"


def download_pmc_paper(
    doi: str,
    pmcid: str | None,
    db_path: Path,
    target_dir: Path,
    timeout: float,
) -> DownloadResult:
    """Download a single paper from PMC using its PMCID.

    Creates its own requests session (not thread-safe) and updates the database.
    """
    if not pmcid:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE papers SET download_status = 'manual_required' WHERE doi = ?",
                (doi,),
            )
        return DownloadResult(doi, "manual_required", "No PMCID in source_metadata")

    # Create session with retry strategy (requests.Session is not thread-safe)
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        # Get PDF link from OA API
        oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
        response = session.get(oa_url, timeout=timeout)
        response.raise_for_status()
        oa_data = response.text

        # Look for TGZ link (includes main paper + supplements)
        tgz_match = re.search(r'<link[^>]*format="tgz"[^>]*href="([^"]+)"[^>]*/>', oa_data)

        if not tgz_match:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE papers SET download_status = 'manual_required' WHERE doi = ?",
                    (doi,),
                )
            return DownloadResult(doi, "manual_required", f"No TGZ link in OA ({pmcid})")

        tgz_url = tgz_match.group(1)

        # Replace ftp:// with https:// if needed
        if tgz_url.startswith("ftp://"):
            tgz_url = tgz_url.replace("ftp://", "https://")

        # Download TGZ
        tgz_response = session.get(tgz_url, timeout=timeout)
        tgz_response.raise_for_status()

        # Process TGZ: extract, find main PDF, concatenate with supplements
        pdf_path = doi_to_path(doi, target_dir, ".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        success, message = process_tgz_archive(tgz_response.content, pdf_path)

        if not success:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE papers SET download_status = 'manual_required' WHERE doi = ?",
                    (doi,),
                )
            return DownloadResult(doi, "manual_required", f"TGZ processing failed: {message}")

        # Update status to downloaded
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE papers SET download_status = 'downloaded' WHERE doi = ?",
                (doi,),
            )

        return DownloadResult(doi, "downloaded", f"Downloaded ({pmcid}) - {message}")

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


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    reraise=True,
)
def _download_preprint_pdf(client: httpx.Client, url: str, timeout: float) -> bytes:
    """Download a preprint PDF with retries."""
    response = client.get(url, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "application/pdf" not in content_type:
        raise ValueError(f"Not a PDF (content-type: {content_type})")
    return response.content


def download_preprint_paper(
    doi: str,
    source: str,
    version: int,
    db_path: Path,
    target_dir: Path,
    timeout: float,
) -> DownloadResult:
    """Download a single preprint PDF by constructing its URL from metadata."""
    url = _build_preprint_url(source, doi, version)

    try:
        with httpx.Client() as client:
            content = _download_preprint_pdf(client, url, timeout)

        pdf_path = doi_to_path(doi, target_dir, ".pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(content)

        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE papers SET download_status = 'downloaded' WHERE doi = ?",
                (doi,),
            )

        size_kb = len(content) / 1024
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
    max_workers: int = typer.Option(5, "--max-workers", "-w", help="Number of parallel downloads"),
) -> None:
    """Download PDFs for scheduled preprint papers (bioRxiv, medRxiv, Research Square).

    Constructs PDF URLs from DOI + version in source_metadata and downloads directly.
    All preprints are open access. Any download failure aborts with non-zero exit.
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

    with Progress() as progress:
        task = progress.add_task("Downloading...", total=len(papers_needing_download))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    download_preprint_paper, doi, source, version, db_path, target_dir, timeout
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

    console.print("\n[bold]Preprint Download Summary:[/bold]")
    console.print(f"  Downloaded: {downloaded}")

    if errors:
        console.print(f"  [red]Errors: {len(errors)}[/red]")
        for doi, error in errors:
            console.print(f"    {doi}: {error}")
        raise typer.Exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
