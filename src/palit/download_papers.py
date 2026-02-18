#!/usr/bin/env python3

"""Download papers workflow: Automated PMC download and PDF matching."""

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

import requests
import typer
from defusedxml import ElementTree as ET
from pypdf import PdfWriter
from requests.adapters import HTTPAdapter
from rich.console import Console
from rich.progress import Progress
from urllib3.util.retry import Retry

from palit.panelapp_client import PanelAppClient

console = Console()
app = typer.Typer(help="Download papers workflow: Automated PMC download and PDF matching")

logger = logging.getLogger(__name__)


def batch_sql_update(
    cursor: sqlite3.Cursor, query_template: str, pmid_list: list[str], batch_size: int = 900
) -> int:
    """Execute a SQL UPDATE query in batches to handle large PMID lists.

    Args:
        cursor: SQLite cursor
        query_template: SQL template with {placeholders} to be filled with "?, ?, ..."
        pmid_list: List of PMIDs to process
        batch_size: Maximum number of parameters per batch (default 900 to stay under SQLite's 999 limit)

    Returns:
        Total number of rows affected across all batches
    """
    total_affected = 0

    for i in range(0, len(pmid_list), batch_size):
        batch = pmid_list[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        query = query_template.format(placeholders=placeholders)
        cursor.execute(query, batch)
        total_affected += cursor.rowcount

    return total_affected


def read_pmids_from_db(
    db_path: Path, skip_hgnc_ids: list[int] | None = None, expansion_only: bool = False
) -> list[str]:
    """Read PMIDs from database where papers require manual download.

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
            # Exclude PMIDs that are associated with skip_hgnc_ids
            placeholders = ",".join("?" * len(skip_hgnc_ids))
            query = f"""
                SELECT DISTINCT p.pmid
                FROM papers p
                WHERE {base_where}
                AND p.pmid NOT IN (
                    SELECT DISTINCT gm.pmid
                    FROM gene_mentions gm
                    WHERE gm.hgnc_id IN ({placeholders})
                )
                ORDER BY p.pmid
            """
            cursor.execute(query, skip_hgnc_ids)
        else:
            query = f"""
                SELECT pmid FROM papers p
                WHERE {base_where}
                ORDER BY pmid
            """
            cursor.execute(query)

        pmids = [str(row[0]) for row in cursor.fetchall()]

    if not pmids:
        logger.warning("No papers with download_status='manual_required' in database")
    elif skip_hgnc_ids:
        logger.info(f"Excluding papers for HGNC IDs: {skip_hgnc_ids}")

    return pmids


def read_pmids_and_titles_from_db(
    db_path: Path, skip_hgnc_ids: list[int] | None = None
) -> dict[str, str]:
    """Read PMIDs and titles from database where papers have non-NULL download_status.

    Args:
        db_path: Path to database
        skip_hgnc_ids: Optional list of HGNC IDs to skip papers for

    Returns:
        Dict mapping PMID (as string) to title
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        if skip_hgnc_ids:
            # Exclude PMIDs that are associated with skip_hgnc_ids
            placeholders = ",".join("?" * len(skip_hgnc_ids))
            query = f"""
                SELECT DISTINCT p.pmid, p.title
                FROM papers p
                WHERE p.download_status IS NOT NULL
                AND p.pmid NOT IN (
                    SELECT DISTINCT gm.pmid
                    FROM gene_mentions gm
                    WHERE gm.hgnc_id IN ({placeholders})
                )
                ORDER BY p.pmid
            """
            cursor.execute(query, skip_hgnc_ids)
        else:
            cursor.execute("""
                SELECT pmid, title FROM papers
                WHERE download_status IS NOT NULL
                ORDER BY pmid
            """)

        pmid_to_title = {str(row[0]): row[1] for row in cursor.fetchall()}

    if not pmid_to_title:
        logger.warning("No papers with non-NULL download_status in database")
    elif skip_hgnc_ids:
        logger.info(f"Excluding papers for HGNC IDs: {skip_hgnc_ids}")

    return pmid_to_title


def check_existing_files(pmid: str, download_dir: Path) -> list[str]:
    """Check which file types exist for a PMID."""
    existing = []
    if (download_dir / f"{pmid}.pdf").exists():
        existing.append("pdf")
    if (download_dir / f"{pmid}.json").exists():
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
        help="Database path to read PMIDs from papers assessed as relevant",
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
    """Open PubMed links in browser for manual PDF download."""

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

    pmids = read_pmids_from_db(db_path, skip_id_list, expansion_only)
    if not dry_run:
        console.print(f"[bold]Found {len(pmids)} PMIDs marked for download in database[/bold]")
    if expansion_only:
        console.print("[cyan]Filtering to expansion papers only (source_type='expansion')[/cyan]")

    # Filter out PMIDs that already have PDFs
    pmids_needing_download = []
    existing_pdfs = 0

    for pmid in pmids:
        existing = check_existing_files(pmid, target_dir)
        if "pdf" not in existing:
            pmids_needing_download.append(pmid)
        else:
            existing_pdfs += 1

    if existing_pdfs > 0:
        console.print(
            f"[dim]Skipping {existing_pdfs} PMIDs that already have PDFs in {target_dir}[/dim]"
        )

    if not pmids_needing_download:
        console.print("[green]All PMIDs already have PDFs - no downloads needed![/green]")
        return

    # If dry-run, just print URLs and exit
    if dry_run:
        for pmid in pmids_needing_download:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            console.print(url)

        return

    console.print(
        f"\n[yellow]Opening {len(pmids_needing_download)} PubMed links in browser for manual PDF download...[/yellow]"
    )
    console.print(f"[dim]Browser delay: {browser_delay} seconds between tabs[/dim]\n")

    for i, pmid in enumerate(pmids_needing_download, 1):
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        console.print(f"[{i}/{len(pmids_needing_download)}] Opening: {url}")
        webbrowser.open(url)

        if i < len(pmids_needing_download) and browser_delay > 0:
            time.sleep(browser_delay)

    console.print(
        f"\n[bold green]✅ Opened {len(pmids_needing_download)} PubMed links in browser[/bold green]"
    )
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("  1. Download PDFs manually to data/papers with PMID.pdf naming")
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
    """Register converted papers by updating download_status for PMIDs with JSON files."""

    if not papers_dir.exists():
        console.print(f"[red]Papers directory not found: {papers_dir}[/red]")
        raise typer.Exit(1)

    # Find all JSON files that match PMID pattern (numeric filenames)
    json_files = [f for f in papers_dir.glob("*.json") if f.stem.isdigit()]

    if not json_files:
        console.print(f"[yellow]No PMID JSON files found in {papers_dir}[/yellow]")
        return

    console.print(f"[cyan]Found {len(json_files)} PMID JSON files in {papers_dir}[/cyan]")

    registered_count = 0
    not_in_db = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        for json_path in json_files:
            pmid = json_path.stem

            # Check if PMID exists in database
            cursor.execute("SELECT download_status FROM papers WHERE pmid = ?", (pmid,))
            row = cursor.fetchone()

            if not row:
                not_in_db.append(pmid)
                continue

            current_status = row["download_status"]

            # Only update if status is 'scheduled' or 'manual_required'
            if current_status in ("scheduled", "manual_required"):
                cursor.execute(
                    "UPDATE papers SET download_status = 'manual_downloaded' WHERE pmid = ?",
                    (pmid,),
                )
                logger.info(f"✅ Registered PMID {pmid} (was: {current_status})")
                registered_count += 1
            else:
                logger.debug(f"Skipped PMID {pmid} (status already: {current_status})")

        conn.commit()

    console.print("\n[bold]Registration Summary:[/bold]")
    console.print(f"  ✅ Registered papers: {registered_count}")
    console.print(
        f"  • Skipped (already registered): {len(json_files) - registered_count - len(not_in_db)}"
    )

    if not_in_db:
        console.print(f"  ⚠️ Not in database: {len(not_in_db)}")
        for pmid in sorted(not_in_db):
            console.print(f"     • {pmid}")

    if registered_count > 0:
        console.print(
            "\n[green]✅ Registration complete! Papers ready for evidence extraction.[/green]"
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
class PmcDownloadResult:
    """Result of attempting to download a single paper from PMC."""

    pmid: str
    status: str  # 'pmc_downloaded' | 'manual_required' | 'error'
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


def download_single_pmid(
    pmid: str,
    db_path: Path,
    target_dir: Path,
    timeout: float,
    tool: str,
    email: str,
) -> PmcDownloadResult:
    """Download a single paper from PMC.

    Creates its own requests session (not thread-safe) and updates the database.
    """
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
        # Step 1: Get PMCID from idconv API
        idconv_url = f"https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?ids={pmid}&idtype=pmid&format=json&tool={tool}&email={email}"
        response = session.get(idconv_url, timeout=timeout)
        response.raise_for_status()
        idconv_data = response.json()

        # Extract PMCID
        records = idconv_data.get("records", [])
        if not records or not records[0].get("pmcid"):
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE papers SET download_status = 'manual_required' WHERE pmid = ?",
                    (pmid,),
                )
            return PmcDownloadResult(pmid, "manual_required", "No PMCID found")

        pmcid = records[0]["pmcid"]

        # Step 2: Get PDF link from OA API
        oa_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}"
        response = session.get(oa_url, timeout=timeout)
        response.raise_for_status()
        oa_data = response.text

        # Step 3: Look for TGZ link (includes main paper + supplements)
        tgz_match = re.search(r'<link[^>]*format="tgz"[^>]*href="([^"]+)"[^>]*/>', oa_data)

        if not tgz_match:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE papers SET download_status = 'manual_required' WHERE pmid = ?",
                    (pmid,),
                )
            return PmcDownloadResult(pmid, "manual_required", f"No TGZ link in OA ({pmcid})")

        tgz_url = tgz_match.group(1)

        # Replace ftp:// with https:// if needed
        if tgz_url.startswith("ftp://"):
            tgz_url = tgz_url.replace("ftp://", "https://")

        # Download TGZ
        tgz_response = session.get(tgz_url, timeout=timeout)
        tgz_response.raise_for_status()

        # Process TGZ: extract, find main PDF, concatenate with supplements
        pdf_path = target_dir / f"{pmid}.pdf"
        success, message = process_tgz_archive(tgz_response.content, pdf_path)

        if not success:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE papers SET download_status = 'manual_required' WHERE pmid = ?",
                    (pmid,),
                )
            return PmcDownloadResult(pmid, "manual_required", f"TGZ processing failed: {message}")

        # Update status to pmc_downloaded
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE papers SET download_status = 'pmc_downloaded' WHERE pmid = ?",
                (pmid,),
            )

        return PmcDownloadResult(pmid, "pmc_downloaded", f"Downloaded ({pmcid}) - {message}")

    except Exception as e:
        return PmcDownloadResult(pmid, "error", str(e))


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
    email: str = typer.Option(
        "panelapp-support@mcri.edu.au", "--email", help="Email for NCBI API identification"
    ),
    tool: str = typer.Option(
        "panelapp-literature-search", "--tool", help="Tool name for NCBI API identification"
    ),
    max_workers: int = typer.Option(5, "--max-workers", "-w", help="Number of parallel downloads"),
) -> None:
    """Attempt automated PMC download for scheduled papers.

    For each paper with download_status='scheduled' and missing PDF:
    1. Query idconv API for PMCID
    2. Query OA API for PDF download link
    3. Download PDF if available
    4. Update status to 'pmc_downloaded' or 'manual_required'
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    target_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get papers scheduled for download that don't have PDFs
        cursor.execute("""
            SELECT pmid
            FROM papers
            WHERE download_status = 'scheduled'
            ORDER BY pmid
        """)
        scheduled_pmids = [row[0] for row in cursor.fetchall()]

    if not scheduled_pmids:
        console.print("[yellow]No papers with download_status='scheduled'[/yellow]")
        return

    # Filter out PMIDs that already have PDFs
    pmids_needing_download = [
        pmid for pmid in scheduled_pmids if not (target_dir / f"{pmid}.pdf").exists()
    ]

    if not pmids_needing_download:
        console.print("[green]All scheduled papers already have PDFs![/green]")
        return

    console.print(
        f"[cyan]Attempting PMC download for {len(pmids_needing_download)} papers "
        f"({max_workers} workers)[/cyan]"
    )

    pmc_downloaded = 0
    manual_required = 0
    errors: list[tuple[str, str]] = []

    with Progress() as progress:
        task = progress.add_task("Downloading...", total=len(pmids_needing_download))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    download_single_pmid, pmid, db_path, target_dir, timeout, tool, email
                ): pmid
                for pmid in pmids_needing_download
            }

            for future in as_completed(futures):
                result = future.result()
                if result.status == "pmc_downloaded":
                    pmc_downloaded += 1
                    logger.info(f"✅ PMID {result.pmid}: {result.message}")
                elif result.status == "manual_required":
                    manual_required += 1
                    logger.info(f"PMID {result.pmid}: {result.message}")
                else:  # error
                    errors.append((result.pmid, result.message))
                    logger.error(f"❌ PMID {result.pmid}: {result.message}")
                progress.advance(task)

    # Summary
    console.print("\n[bold]PMC Download Summary:[/bold]")
    console.print(f"  ✅ PMC downloaded: {pmc_downloaded}")
    console.print(f"  📝 Manual required: {manual_required}")
    if errors:
        console.print(f"  ❌ Errors: {len(errors)}")
        for pmid, error in errors[:5]:  # Show first 5 errors
            console.print(f"     PMID {pmid}: {error}")
        if len(errors) > 5:
            console.print(f"     ... and {len(errors) - 5} more errors")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
