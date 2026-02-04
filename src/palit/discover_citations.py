#!/usr/bin/env python3
"""Discover and download papers referenced in evidence extractions."""

import json
import logging
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path

import httpx
import typer
from rich.console import Console
from rich.progress import track

from palit.ingest_pubmed import Article, extract_articles_from_xml

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

console = Console()
app = typer.Typer(help="Discover papers referenced in evidence extractions")

logger = logging.getLogger(__name__)


@dataclass
class ReferencedSource:
    """A paper cited as a source for previously reported cases."""

    title: str
    context: str
    gene_symbol: str
    citing_pmid: int


def extract_referenced_sources_from_db(db_path: Path) -> list[ReferencedSource]:
    """Extract all previously reported sources from disease entities in the database.

    Args:
        db_path: Path to SQLite database

    Returns:
        List of ReferencedSource objects
    """
    logger.info("Extracting previously reported sources from database")

    sources: list[ReferencedSource] = []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get all papers with evidence extraction
        cursor.execute(
            """
            SELECT DISTINCT
                p.pmid,
                gm.paper_gene_symbol,
                p.evidence_extraction_json
            FROM papers p
            JOIN gene_mentions gm ON p.pmid = gm.pmid
            WHERE p.evidence_extraction_json IS NOT NULL
            """
        )

        for row in cursor.fetchall():
            pmid, gene_symbol, evidence_json = row

            try:
                evidence = json.loads(evidence_json)
            except json.JSONDecodeError:
                logger.warning(f"Skipping PMID {pmid}: invalid JSON in evidence_extraction")
                continue

            # Look for gene evaluations
            gene_evaluations = evidence.get("gene_evaluations", [])
            for gene_eval in gene_evaluations:
                if gene_eval.get("gene") != gene_symbol:
                    continue

                # Look for disease entities with previously_reported_sources
                disease_entities = gene_eval.get("disease_entities", [])
                for entity in disease_entities:
                    previously_reported = entity.get("previously_reported_sources", [])
                    for prev_source in previously_reported:
                        title = prev_source.get("title", "").strip()
                        context = prev_source.get("context", "")

                        if title and title != "NR":
                            sources.append(
                                ReferencedSource(
                                    title=title,
                                    context=context,
                                    gene_symbol=gene_symbol,
                                    citing_pmid=pmid,
                                )
                            )

    logger.info(f"Found {len(sources)} previously reported sources across all genes")
    return sources


def search_pubmed_by_title(title: str) -> int | None:
    """Search PubMed for a paper by title using esearch.

    Uses a single flexible search with PubMed's built-in fuzzy matching.

    Args:
        title: Paper title to search for

    Returns:
        PMID if found (top result), None otherwise
    """
    try:
        params = {
            "db": "pubmed",
            "retmode": "json",
            "field": "title",
            "term": title,
        }

        # We use the ESearch endpoint directly instead of the `esearch` CLI, as the
        # latter applies some "helpful" pre-processing for stopwords etc., making most
        # title searches fail.
        response = httpx.get(ESEARCH_URL, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        esearch_result = data.get("esearchresult", {})
        count_str = esearch_result.get("count")
        if count_str is None:
            return None

        try:
            count = int(count_str)
        except ValueError:
            logger.warning(f"Unexpected count in PubMed response for title: {title}...")
            return None

        if count == 0:
            logger.debug(f"No results for title: {title}...")
            return None

        id_list = esearch_result.get("idlist", [])
        if not id_list:
            return None

        if count > 1:
            logger.warning(
                f"Multiple results ({count}) for title, using top result PMID {id_list[0]}: {title}..."
            )
        else:
            logger.debug(f"Found PMID {id_list[0]} for title: {title}...")

        try:
            return int(id_list[0])
        except (TypeError, ValueError):
            logger.warning(f"Invalid PMID received for title: {title}...")
            return None

    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning(f"Search failed for title: {title}... ({e})")
        return None


def fetch_paper_metadata(pmid: int, gene_symbol: str, citing_pmid: int | str) -> Article | None:
    """Fetch paper metadata from PubMed using efetch.

    Args:
        pmid: PubMed ID
        gene_symbol: Gene symbol for source_details
        citing_pmid: PMID of paper citing this reference, or "manual" for manually added papers

    Returns:
        Article object if successful, None otherwise
    """
    try:
        # Use efetch to get XML
        efetch_cmd = ["efetch", "-db", "pubmed", "-id", str(pmid), "-format", "xml"]

        result = subprocess.run(
            efetch_cmd,
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(f"efetch failed for PMID {pmid}")
            return None

        # Parse articles using existing function
        # Allow articles without abstracts for cited papers (case reports, etc.)
        articles = extract_articles_from_xml(
            result.stdout,
            source_type="expansion",
            source_details=f"referenced:{gene_symbol}:{citing_pmid}",
            require_abstract=False,
        )

        if not articles:
            logger.warning(f"No article data extracted for PMID {pmid}")
            return None

        return articles[0]

    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to fetch metadata for PMID {pmid}: {e}")
        return None


def check_pmid_exists(db_path: Path, pmid: int) -> bool:
    """Check if a PMID already exists in the database.

    Args:
        db_path: Path to SQLite database
        pmid: PubMed ID to check

    Returns:
        True if PMID exists, False otherwise
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM papers WHERE pmid = ?", (pmid,))
        return cursor.fetchone() is not None


def store_referenced_paper(db_path: Path, article: Article) -> bool:
    """Store a referenced paper in the database.

    Args:
        db_path: Path to SQLite database
        article: Article object to store
        gene_symbol: Gene symbol this paper was referenced for
        citing_pmid: PMID of the paper that cited this one

    Returns:
        True if paper was newly inserted, False if already existed
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO papers
            (pmid, title, abstract, authors, journal, entrez_date, source_type, source_details, download_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'scheduled')
            ON CONFLICT(pmid) DO NOTHING
            """,
            (
                article.pmid,
                article.title,
                article.abstract,
                article.authors,
                article.journal,
                article.entrez_date,
                article.source_type,
                article.source_details,
            ),
        )

        inserted = cursor.rowcount > 0
        conn.commit()

        return inserted


@app.command()
def discover(
    db_path: Path = typer.Option(default=Path("data/db.sqlite"), help="Path to SQLite database"),
) -> None:
    """Discover and schedule referenced papers for download.

    Scans all genes with evidence, extracts previously_reported_sources from
    disease entities, searches PubMed for the papers, and adds missing papers
    as expansion papers.
    """

    if not db_path.exists():
        console.print(f"[red]Error: Database not found: {db_path}[/red]")
        raise typer.Exit(code=1)

    # Extract all referenced sources
    sources = extract_referenced_sources_from_db(db_path)

    if not sources:
        console.print("[yellow]No referenced sources found in database[/yellow]")
        return

    # Deduplicate by title (case-insensitive)
    unique_sources: dict[str, ReferencedSource] = {}
    for source in sources:
        title_key = source.title.lower()
        if title_key not in unique_sources:
            unique_sources[title_key] = source

    console.print(f"Found {len(unique_sources)} unique referenced paper titles")

    # Search for each title and store new papers
    new_papers = 0
    already_exists = 0
    not_found = 0
    failures_by_gene: dict[str, list[str]] = {}

    for source in track(unique_sources.values(), description="Searching PubMed..."):
        # Search for PMID
        pmid = search_pubmed_by_title(source.title)

        if pmid is None:
            logger.info(
                f"Could not find PMID for '{source.title}' "
                f"(gene: {source.gene_symbol}, cited by: {source.citing_pmid})"
            )
            not_found += 1
            if source.gene_symbol not in failures_by_gene:
                failures_by_gene[source.gene_symbol] = []
            failures_by_gene[source.gene_symbol].append(source.title)
            continue

        # Check if already in database
        if check_pmid_exists(db_path, pmid):
            logger.debug(f"PMID {pmid} already in database, skipping")
            already_exists += 1
            continue

        # Fetch metadata
        article = fetch_paper_metadata(pmid, source.gene_symbol, source.citing_pmid)
        if article is None:
            logger.warning(f"Failed to fetch metadata for PMID {pmid}")
            not_found += 1
            if source.gene_symbol not in failures_by_gene:
                failures_by_gene[source.gene_symbol] = []
            failures_by_gene[source.gene_symbol].append(
                f"{source.title} (PMID {pmid} found but metadata fetch failed)"
            )
            continue

        # Store in database
        inserted = store_referenced_paper(db_path, article)
        if inserted:
            logger.info(
                f"Added PMID {pmid} for gene {source.gene_symbol} "
                f"(referenced by PMID {source.citing_pmid})"
            )
            new_papers += 1
        else:
            already_exists += 1

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Unique titles searched: {len(unique_sources)}")
    console.print(f"  [green]New papers added: {new_papers}[/green]")
    console.print(f"  [blue]Already in database: {already_exists}[/blue]")
    console.print(f"  [yellow]Not found in PubMed: {not_found}[/yellow]")

    # Detailed failure report
    if failures_by_gene:
        console.print("\n[bold yellow]Failed to find the following papers:[/bold yellow]")
        for gene in sorted(failures_by_gene.keys()):
            console.print(f"\n[bold]{gene}:[/bold]")
            for title in failures_by_gene[gene]:
                console.print(f"  - {title}")

        console.print("\n[dim]Hint: To add papers manually, look up PMIDs and run:[/dim]")
        console.print(
            "[dim]  uv run palit discover-citations add --gene GENE_SYMBOL PMID1 PMID2 ...[/dim]"
        )

    if new_papers > 0:
        console.print(
            "\n[green]Run 'download-papers' and 'extract-evidence' to process the new papers[/green]"
        )


@app.command()
def add(
    pmids: list[int] = typer.Argument(..., help="PMIDs to add to the database"),
    gene: str = typer.Option(..., "--gene", "-g", help="Gene symbol to associate with"),
    db_path: Path = typer.Option(default=Path("data/db.sqlite"), help="Path to SQLite database"),
) -> None:
    """Manually add papers to the database by PMID.

    Papers will be added with source_type='expansion' and
    source_details='referenced:{gene}:manual' to match the citation discovery format.
    """

    if not db_path.exists():
        console.print(f"[red]Error: Database not found: {db_path}[/red]")
        raise typer.Exit(code=1)

    if not pmids:
        console.print("[red]Error: At least one PMID must be provided[/red]")
        raise typer.Exit(code=1)

    console.print(f"Adding {len(pmids)} paper(s) for gene {gene}")

    added = 0
    skipped = 0
    failed = 0

    for pmid in pmids:
        # Check if already exists
        if check_pmid_exists(db_path, pmid):
            logger.info(f"PMID {pmid} already in database, skipping")
            skipped += 1
            continue

        # Fetch metadata using "manual" as the citing PMID
        article = fetch_paper_metadata(pmid, gene, citing_pmid="manual")

        if article is None:
            logger.error(f"Failed to fetch metadata for PMID {pmid}")
            failed += 1
            continue

        # Store in database
        inserted = store_referenced_paper(db_path, article)
        if inserted:
            logger.info(f"Added PMID {pmid} for gene {gene}")
            added += 1
        else:
            skipped += 1

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  [green]Papers added: {added}[/green]")
    console.print(f"  [blue]Already in database: {skipped}[/blue]")
    console.print(f"  [red]Failed to fetch: {failed}[/red]")

    if added > 0:
        console.print(
            "\n[green]Run 'download-papers' and 'extract-evidence' to process the new papers[/green]"
        )


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
