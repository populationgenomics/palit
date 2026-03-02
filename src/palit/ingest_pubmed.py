#!/usr/bin/env python3
"""Download and ingest PubMed papers into database."""

import gzip
import io
import logging
import sqlite3
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import typer
from lxml import etree
from rich.console import Console
from rich.progress import Progress

from palit.papers import Paper, PubmedMetadata, SkipReason, serialize_source_metadata

console = Console()
app = typer.Typer(help="Download and ingest PubMed papers")

logger = logging.getLogger(__name__)

# Configuration
MIN_FILE_SIZE = 1000  # Minimum expected file size in bytes
MAX_RETRIES = 5
RETRY_DELAY = 2  # Seconds between retries


# Paper extraction functions


def extract_text_content(element: etree._Element | None) -> str:
    """Extract all text content from an XML element, including nested elements."""
    if element is None:
        return ""

    # Get all text content, including from nested elements
    text_parts = []
    if element.text:
        text_parts.append(element.text)

    for child in element:
        text_parts.append(extract_text_content(child))
        if child.tail:
            text_parts.append(child.tail)

    return "".join(text_parts).strip()


def extract_authors(article_elem: etree._Element) -> str:
    """Extract authors from AuthorList element."""
    author_list = article_elem.find(".//MedlineCitation/Article/AuthorList")
    if author_list is None:
        return ""

    authors = []
    for author in author_list.findall("Author"):
        last_name_elem = author.find("LastName")
        first_name_elem = author.find("ForeName")

        if last_name_elem is not None and last_name_elem.text:
            if first_name_elem is not None and first_name_elem.text:
                authors.append(f"{last_name_elem.text}, {first_name_elem.text}")
            else:
                authors.append(last_name_elem.text)

    return "; ".join(authors)


def extract_journal(article_elem: etree._Element) -> str:
    """Extract journal title from Journal element."""
    journal_elem = article_elem.find(".//MedlineCitation/Article/Journal/Title")
    if journal_elem is not None and journal_elem.text is not None:
        return str(journal_elem.text).strip()
    return ""


def extract_date(date_element: etree._Element | None) -> str | None:
    """Extract date from PubMedPubDate element."""
    if date_element is None:
        return None

    year_elem = date_element.find("Year")
    month_elem = date_element.find("Month")
    day_elem = date_element.find("Day")

    if year_elem is None or month_elem is None or day_elem is None:
        return None

    try:
        year = int(year_elem.text)
        month = int(month_elem.text)
        day = int(day_elem.text)
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, TypeError):
        return None


def extract_paper(
    article_elem: etree._Element,
    source_type: str,
    source_details: str,
    require_abstract: bool = True,
) -> Paper | SkipReason:
    """Extract paper data from PubmedArticle element.

    Args:
        article_elem: The XML element containing the paper data
        source_type: Type of source (e.g., "initial", "expansion")
        source_details: Details about the source
        require_abstract: If True, skip papers without abstracts. Default True.
    """
    # Extract article IDs (DOI, PMID, PMCID) from ArticleIdList
    article_ids: dict[str, str] = {}
    for article_id in article_elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = article_id.get("IdType")
        if id_type and article_id.text:
            article_ids[id_type] = article_id.text.strip()

    doi = article_ids.get("doi")
    if not doi:
        return SkipReason.NO_DOI

    # Extract entrez date from PubmedData/History/PubMedPubDate[@PubStatus="entrez"]
    entrez_date_elem = article_elem.find('.//PubmedData/History/PubMedPubDate[@PubStatus="entrez"]')
    source_date = extract_date(entrez_date_elem)
    if not source_date:
        return SkipReason.NO_DATE

    # Extract title from MedlineCitation/Article/ArticleTitle
    title_elem = article_elem.find(".//MedlineCitation/Article/ArticleTitle")
    title = extract_text_content(title_elem)
    if not title:
        return SkipReason.NO_TITLE

    # Extract abstract from MedlineCitation/Article/Abstract/AbstractText elements
    abstract_elems = article_elem.findall(".//MedlineCitation/Article/Abstract/AbstractText")
    abstract_parts = [extract_text_content(elem) for elem in abstract_elems]
    abstract = " ".join(abstract_parts).strip()

    if require_abstract and not abstract:
        return SkipReason.NO_ABSTRACT

    # Extract authors and journal
    authors = extract_authors(article_elem)
    journal = extract_journal(article_elem)

    pmid_str = article_ids.get("pubmed")
    pmid = int(pmid_str) if pmid_str else None

    return Paper(
        doi=doi,
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=authors,
        journal=journal,
        source="pubmed",
        source_date=source_date,
        source_metadata=PubmedMetadata(pmcid=article_ids.get("pmc")),
        source_type=source_type,
        source_details=source_details,
    )


@dataclass
class ExtractionStats:
    """Statistics from paper extraction."""

    total_articles: int
    extracted: int
    skipped: dict[SkipReason, int]


def extract_papers_from_xml(
    xml_content: bytes,
    source_type: str,
    source_details: str,
    require_abstract: bool = True,
    min_year: int | None = None,
) -> tuple[list[Paper], ExtractionStats]:
    """Extract papers from XML bytes content.

    Args:
        xml_content: XML content as bytes
        source_type: Type of source (e.g., "initial", "expansion")
        source_details: Details about the source (e.g., filename, gene name)
        require_abstract: If True, skip papers without abstracts. Default True.
        min_year: If provided, only include papers with source_date >= this year

    Returns:
        Tuple of (list of Paper objects, extraction statistics)
    """
    # Parse XML
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    tree = etree.parse(io.BytesIO(xml_content), parser)
    root = tree.getroot()
    if root is None:
        return [], ExtractionStats(total_articles=0, extracted=0, skipped={})

    # Find all PubmedArticle elements
    papers = []
    skipped: dict[SkipReason, int] = {}
    article_elements = root.findall(".//PubmedArticle")

    for article_elem in article_elements:
        result = extract_paper(article_elem, source_type, source_details, require_abstract)
        if isinstance(result, SkipReason):
            skipped[result] = skipped.get(result, 0) + 1
            continue

        # Filter by year if min_year is specified
        if min_year is not None:
            paper_year = int(result.source_date[:4])
            if paper_year < min_year:
                continue

        papers.append(result)

    stats = ExtractionStats(
        total_articles=len(article_elements),
        extracted=len(papers),
        skipped=skipped,
    )
    if skipped:
        skip_summary = ", ".join(
            f"{r.value}: {n}" for r, n in sorted(skipped.items(), key=lambda x: -x[1])
        )
        logger.info(
            f"Skipped {sum(skipped.values())}/{len(article_elements)} articles "
            f"from {source_details} ({skip_summary})"
        )

    return papers, stats


def process_xml_file(xml_path: Path, start_date: str, end_date: str, output_db: Path) -> None:
    """Process XML file and extract papers within date range."""
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
                -- Only update if both old and new records are from initial
                -- and the new file is lexicographically later (e.g., pubmed_2025-09-16.xml > pubmed_2025-09-15.xml)
                -- This ensures newer PubMed updates override older ones, but expansion searches
                -- never overwrite existing records
                WHERE excluded.source_type = 'initial'
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
            conn.commit()

        logger.info(f"Inserted {len(papers)} papers into database")

    finally:
        conn.close()


@dataclass
class DownloadResult:
    """Result of downloading a single day's papers."""

    day: int
    success: bool
    file_path: Path | None
    file_size: int
    attempts: int
    error: str | None = None


def download_day(year: int, month: int, day: int, output_dir: Path) -> DownloadResult:
    """Download PubMed papers for a single day with retry logic.

    Args:
        year: Year to download
        month: Month to download
        day: Day to download
        output_dir: Directory to save XML files

    Returns:
        DownloadResult with success status and metadata
    """
    output_file = output_dir / f"pubmed_{year}-{month:02d}-{day:02d}.xml.gz"

    # Skip if file already exists and is large enough
    if output_file.exists():
        file_size = output_file.stat().st_size
        if file_size >= MIN_FILE_SIZE:
            logger.debug(f"Day {day:02d}: File already exists ({file_size} bytes), skipping")
            return DownloadResult(
                day=day,
                success=True,
                file_path=output_file,
                file_size=file_size,
                attempts=0,  # 0 attempts means it was already cached
            )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Build esearch and efetch pipeline. We use efetch, as the daily update files at
            # https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/ can sometimes lag behind significantly.
            # We use CRDT (Create Date) rather than EDAT (Entrez Date) because empirically CRDT
            # matches PubMedPubDate[@PubStatus="entrez"] in the XML (which we extract as entrez_date),
            # while EDAT can return papers on a different day. See:
            # https://www.nlm.nih.gov/pubs/techbull/nd08/nd08_pm_new_date_field.html
            esearch_cmd = [
                "esearch",
                "-db",
                "pubmed",
                "-query",
                "hasabstract",
                "-datetype",
                "CRDT",
                "-mindate",
                f"{year}/{month:02d}/{day:02d}",
                "-maxdate",
                f"{year}/{month:02d}/{day:02d}",
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
                        f"Day {day:02d}: File too small ({file_size} bytes), "
                        f"attempt {attempt}/{MAX_RETRIES}"
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_DELAY)

        except (OSError, subprocess.SubprocessError) as e:
            # Catch expected retryable errors (network, subprocess, file I/O)
            logger.warning(f"Day {day:02d}: Download failed (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        # Let unexpected exceptions propagate to parent

    # All retries exhausted - skip this day
    file_size = output_file.stat().st_size if output_file.exists() else 0
    logger.warning(
        f"Day {day:02d}: Skipping after {MAX_RETRIES} attempts. "
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
    xml_files: list[Path], start_date: str, end_date: str, db_path: Path, parallel_jobs: int
) -> None:
    """Extract papers from XML files in parallel.

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


@app.callback(invoke_without_command=True)
def main(
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(
        Path("data/pubmed_xml"), "--output-dir", help="Directory for XML files"
    ),
    db_path: Path = typer.Option(Path("data/db.sqlite"), "--db-path", help="Database path"),
    parallel_jobs: int = typer.Option(8, "--jobs", "-j", help="Number of parallel jobs"),
) -> None:
    """Download PubMed papers and ingest into database.

    This command:
    1. Downloads PubMed XML files for the specified date range (one file per day)
    2. Retries downloads for files smaller than expected size
    3. Extracts papers from XML files into the database in parallel
    """

    # Parse dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        console.print(f"[red]Invalid date format: {e}[/red]")
        raise typer.Exit(1) from e

    if start_dt > end_dt:
        console.print("[red]Start date must be before or equal to end date[/red]")
        raise typer.Exit(1)

    # Extract date components (assuming same month for simplicity)
    year = start_dt.year
    month = start_dt.month
    start_day = start_dt.day
    end_day = end_dt.day

    if start_dt.month != end_dt.month:
        console.print(
            "[red]Error: Cross-month downloads not yet supported. "
            "Please run separately for each month.[/red]"
        )
        raise typer.Exit(1)

    console.print(f"[cyan]Downloading PubMed papers: {start_date} to {end_date}[/cyan]")
    console.print(f"[cyan]Database: {db_path}[/cyan]")
    console.print("")

    # Create database if it doesn't exist
    if not db_path.exists():
        console.print("[yellow]Database not found, creating from schema.sql...[/yellow]")
        schema_file = Path("schema.sql")
        if not schema_file.exists():
            console.print(f"[red]Error: schema.sql not found at {schema_file.absolute()}[/red]")
            raise typer.Exit(1)

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(schema_file.read_text())
        conn.close()
        console.print(f"[green]✓[/green] Created database at {db_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download XML files in parallel
    console.print("[bold]Step 1: Downloading PubMed XML files...[/bold]")

    days = list(range(start_day, end_day + 1))
    download_results = []

    with Progress(console=console) as progress:
        task = progress.add_task("Downloading...", total=len(days))

        with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = {
                executor.submit(download_day, year, month, day, output_dir): day for day in days
            }

            for future in as_completed(futures):
                download_result = future.result()
                download_results.append(download_result)
                progress.advance(task)

    # Report download results
    successful = [r for r in download_results if r.success]
    skipped = [r for r in download_results if not r.success]
    console.print(f"[green]✓[/green] Downloaded {len(successful)}/{len(days)} files successfully")
    if skipped:
        skipped_days = ", ".join(f"day {r.day}" for r in skipped)
        console.print(f"[yellow]⚠[/yellow] Skipped (no data after retries): {skipped_days}")

    # Step 2: Extract papers from downloaded files
    console.print("\n[bold]Step 2: Extracting papers to database...[/bold]")

    xml_files = sorted(output_dir.glob("*.xml.gz"))
    if not xml_files:
        console.print("[red]No XML files found to extract[/red]")
        raise typer.Exit(1)

    extract_papers(xml_files, start_date, end_date, db_path, parallel_jobs)

    console.print(f"[green]✓[/green] Extracted papers from {len(xml_files)} files")
    console.print(f"\n[green]✓ Ingestion complete! Database ready at: {db_path}[/green]")


if __name__ == "__main__":
    app()
