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

console = Console()
app = typer.Typer(help="Download and ingest PubMed papers")

logger = logging.getLogger(__name__)

# Configuration
MIN_FILE_SIZE = 1000  # Minimum expected file size in bytes
MAX_RETRIES = 5
RETRY_DELAY = 2  # Seconds between retries


# Article extraction functions (merged from extract_articles.py)


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
            name_parts = [last_name_elem.text]
            if first_name_elem is not None and first_name_elem.text:
                name_parts.append(first_name_elem.text)
            authors.append(" ".join(name_parts))

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


@dataclass
class Article:
    """Article data extracted from PubMed XML."""

    pmid: int
    entrez_date: str
    title: str
    abstract: str
    authors: str
    journal: str
    source_type: str
    source_details: str


def extract_article(
    article_elem: etree._Element,
    source_type: str,
    source_details: str,
    require_abstract: bool = True,
) -> Article | None:
    """Extract article data from PubmedArticle element.

    Args:
        article_elem: The XML element containing the article data
        source_type: Type of source (e.g., "initial", "expansion")
        source_details: Details about the source
        require_abstract: If True, skip articles without abstracts. Default True.
    """
    # Extract PMID from MedlineCitation/PMID
    pmid_elem = article_elem.find(".//MedlineCitation/PMID")
    if pmid_elem is None or not pmid_elem.text:
        return None

    try:
        pmid = int(pmid_elem.text)
    except ValueError:
        return None

    # Extract entrez date from PubmedData/History/PubMedPubDate[@PubStatus="entrez"]
    entrez_date_elem = article_elem.find('.//PubmedData/History/PubMedPubDate[@PubStatus="entrez"]')
    entrez_date = extract_date(entrez_date_elem)
    if not entrez_date:
        return None

    # Extract title from MedlineCitation/Article/ArticleTitle
    title_elem = article_elem.find(".//MedlineCitation/Article/ArticleTitle")
    title = extract_text_content(title_elem)

    # Extract abstract from MedlineCitation/Article/Abstract/AbstractText elements
    abstract_elems = article_elem.findall(".//MedlineCitation/Article/Abstract/AbstractText")
    abstract_parts = [extract_text_content(elem) for elem in abstract_elems]
    abstract = " ".join(abstract_parts).strip()

    # Extract authors and journal
    authors = extract_authors(article_elem)
    journal = extract_journal(article_elem)

    # Skip articles without title
    if not title:
        return None

    # Skip articles without abstract only if required
    if require_abstract and not abstract:
        return None

    return Article(
        pmid=pmid,
        entrez_date=entrez_date,
        title=title,
        abstract=abstract,
        authors=authors,
        journal=journal,
        source_type=source_type,
        source_details=source_details,
    )


def extract_articles_from_xml(
    xml_content: bytes,
    source_type: str,
    source_details: str,
    require_abstract: bool = True,
    min_year: int | None = None,
) -> list[Article]:
    """Extract articles from XML bytes content.

    Args:
        xml_content: XML content as bytes
        source_type: Type of source (e.g., "initial", "expansion")
        source_details: Details about the source (e.g., filename, gene name)
        require_abstract: If True, skip articles without abstracts. Default True.
        min_year: If provided, only include articles with entrez_date >= this year

    Returns:
        List of Article objects extracted from XML
    """
    # Parse XML
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    tree = etree.parse(io.BytesIO(xml_content), parser)
    root = tree.getroot()
    if root is None:
        return []

    # Find all PubmedArticle elements
    articles = []
    article_elements = root.findall(".//PubmedArticle")

    for article_elem in article_elements:
        article_data = extract_article(article_elem, source_type, source_details, require_abstract)
        if article_data:
            # Filter by year if min_year is specified
            if min_year is not None:
                article_year = int(article_data.entrez_date[:4])
                if article_year < min_year:
                    continue

            articles.append(article_data)

    return articles


def process_xml_file(xml_path: Path, start_date: str, end_date: str, output_db: Path) -> None:
    """Process XML file and extract articles within date range."""
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
        all_articles = extract_articles_from_xml(xml_content, source_type, source_details)
        logger.debug(f"Found {len(all_articles)} total articles")

        articles = [
            article
            for article in all_articles
            if article.entrez_date and start_date <= article.entrez_date <= end_date
        ]
        logger.debug(f"Found {len(articles)} articles in date range")

        # Insert articles into database
        if articles:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO papers
                (pmid, entrez_date, title, abstract, authors, journal, source_type, source_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pmid) DO UPDATE SET
                    entrez_date = excluded.entrez_date,
                    title = excluded.title,
                    abstract = excluded.abstract,
                    authors = excluded.authors,
                    journal = excluded.journal,
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
                        a.pmid,
                        a.entrez_date,
                        a.title,
                        a.abstract,
                        a.authors,
                        a.journal,
                        a.source_type,
                        a.source_details,
                    )
                    for a in articles
                ],
            )
            conn.commit()

        logger.info(f"Inserted {len(articles)} articles into database")

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


def extract_articles(
    xml_files: list[Path], start_date: str, end_date: str, db_path: Path, parallel_jobs: int
) -> None:
    """Extract articles from XML files in parallel.

    Args:
        xml_files: List of XML files to process
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
        db_path: Path to database
        parallel_jobs: Number of parallel jobs
    """
    with Progress(console=console) as progress:
        task = progress.add_task("Extracting articles...", total=len(xml_files))

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
    3. Extracts articles from XML files into the database in parallel
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
        try:
            subprocess.run(
                ["sqlite3", str(db_path)],
                stdin=open(schema_file),
                capture_output=True,
                check=True,
                text=True,
            )
            console.print(f"[green]✓[/green] Created database at {db_path}")
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Failed to create database: {e.stderr}[/red]")
            raise typer.Exit(1) from e

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

    # Step 2: Extract articles from downloaded files
    console.print("\n[bold]Step 2: Extracting articles to database...[/bold]")

    xml_files = sorted(output_dir.glob("*.xml.gz"))
    if not xml_files:
        console.print("[red]No XML files found to extract[/red]")
        raise typer.Exit(1)

    extract_articles(xml_files, start_date, end_date, db_path, parallel_jobs)

    console.print(f"[green]✓[/green] Extracted articles from {len(xml_files)} files")
    console.print(f"\n[green]✓ Ingestion complete! Database ready at: {db_path}[/green]")


if __name__ == "__main__":
    app()
