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

from palit.hgnc import HgncResolver
from palit.ingest_pubmed import extract_papers_from_xml
from palit.papers import Paper, serialize_source_metadata

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

console = Console()
app = typer.Typer(help="Discover papers referenced in evidence extractions")

logger = logging.getLogger(__name__)


@dataclass
class ReferencedSource:
    """A paper cited as a source for previously reported cases."""

    title: str
    context: str
    hgnc_id: int
    citing_doi: str


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
                p.doi,
                gm.hgnc_id,
                p.evidence_extraction_json
            FROM papers p
            JOIN gene_mentions gm ON p.doi = gm.paper_doi
            WHERE p.evidence_extraction_json IS NOT NULL
            """
        )

        for row in cursor.fetchall():
            doi, hgnc_id, evidence_json = row

            try:
                evidence = json.loads(evidence_json)
            except json.JSONDecodeError:
                logger.warning(f"Skipping DOI {doi}: invalid JSON in evidence_extraction")
                continue

            # Look for gene evaluations
            gene_evaluations = evidence.get("gene_evaluations", [])
            for gene_eval in gene_evaluations:
                if gene_eval.get("hgnc_id") != hgnc_id:
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
                                    hgnc_id=hgnc_id,
                                    citing_doi=doi,
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


def fetch_paper_metadata(pmid: int, hgnc_id: int, citing_doi: str) -> Paper | None:
    """Fetch paper metadata from PubMed using efetch.

    Args:
        pmid: PubMed ID
        hgnc_id: HGNC ID for source_details
        citing_doi: DOI of paper citing this reference, or "manual" for manually added papers

    Returns:
        Paper object if successful (and has a DOI), None otherwise
    """
    try:
        efetch_cmd = ["efetch", "-db", "pubmed", "-id", str(pmid), "-format", "xml"]

        result = subprocess.run(
            efetch_cmd,
            capture_output=True,
            timeout=30,
        )

        if result.returncode != 0:
            logger.warning(f"efetch failed for PMID {pmid}")
            return None

        # Allow papers without abstracts for cited papers (case reports, etc.)
        papers, _stats = extract_papers_from_xml(
            result.stdout,
            source_type="expansion",
            source_details=f"referenced:{hgnc_id}:{citing_doi}",
            require_abstract=False,
        )

        if not papers:
            logger.warning(f"No paper data extracted for PMID {pmid} (may lack DOI)")
            return None

        return papers[0]

    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to fetch metadata for PMID {pmid}: {e}")
        return None


def check_doi_exists(db_path: Path, doi: str) -> bool:
    """Check if a DOI already exists in the database.

    Args:
        db_path: Path to SQLite database
        doi: DOI to check

    Returns:
        True if DOI exists, False otherwise
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM papers WHERE doi = ?", (doi,))
        return cursor.fetchone() is not None


def store_referenced_paper(db_path: Path, paper: Paper) -> bool:
    """Store a referenced paper in the database.

    Args:
        db_path: Path to SQLite database
        paper: Paper object to store

    Returns:
        True if paper was newly inserted, False if already existed
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO papers
            (doi, pmid, title, abstract, authors, journal, source, source_date, source_metadata,
             source_type, source_details, download_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'scheduled')
            ON CONFLICT(doi) DO NOTHING
            """,
            (
                paper.doi,
                paper.pmid,
                paper.title,
                paper.abstract,
                paper.authors,
                paper.journal,
                paper.source,
                paper.source_date,
                serialize_source_metadata(paper.source_metadata),
                paper.source_type,
                paper.source_details,
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
    hgnc_resolver = HgncResolver.from_file()
    new_papers = 0
    already_exists = 0
    not_found = 0
    failures_by_gene: dict[int, list[str]] = {}

    for source in track(unique_sources.values(), description="Searching PubMed..."):
        gene_display = hgnc_resolver.get_symbol(source.hgnc_id)

        # Search for PMID (PubMed title search still returns PMIDs)
        pmid = search_pubmed_by_title(source.title)

        if pmid is None:
            logger.info(
                f"Could not find PMID for '{source.title}' "
                f"(gene: {gene_display}, cited by: {source.citing_doi})"
            )
            not_found += 1
            failures_by_gene.setdefault(source.hgnc_id, []).append(source.title)
            continue

        # Fetch metadata (efetch returns XML from which we extract DOI)
        paper = fetch_paper_metadata(pmid, source.hgnc_id, source.citing_doi)
        if paper is None:
            logger.warning(f"Failed to fetch metadata for PMID {pmid}")
            not_found += 1
            failures_by_gene.setdefault(source.hgnc_id, []).append(
                f"{source.title} (PMID {pmid} found but metadata fetch failed)"
            )
            continue

        # Check if already in database (by DOI)
        if check_doi_exists(db_path, paper.doi):
            logger.debug(f"DOI {paper.doi} already in database, skipping")
            already_exists += 1
            continue

        # Store in database
        inserted = store_referenced_paper(db_path, paper)
        if inserted:
            logger.info(
                f"Added DOI {paper.doi} for {gene_display} (HGNC:{source.hgnc_id}) "
                f"(referenced by DOI {source.citing_doi})"
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
        for hgnc_id in sorted(failures_by_gene.keys()):
            gene_display = hgnc_resolver.get_symbol(hgnc_id)
            console.print(f"\n[bold]{gene_display} (HGNC:{hgnc_id}):[/bold]")
            for title in failures_by_gene[hgnc_id]:
                console.print(f"  - {title}")

        console.print("\n[dim]Hint: To add papers manually, look up PMIDs and run:[/dim]")
        console.print(
            f"[dim]  uv run palit discover-citations add --db-path {db_path} --hgnc-id HGNC_ID PMID1 PMID2 ...[/dim]"
        )
        console.print("[dim]  (Papers without a DOI in PubMed will be skipped)[/dim]")

    if new_papers > 0:
        console.print(
            "\n[green]Run 'download-papers' and 'extract-evidence' to process the new papers[/green]"
        )


@app.command()
def add(
    pmids: list[int] = typer.Argument(..., help="PMIDs to add to the database"),
    hgnc_id: int = typer.Option(..., "--hgnc-id", "-g", help="HGNC ID to associate with"),
    db_path: Path = typer.Option(default=Path("data/db.sqlite"), help="Path to SQLite database"),
) -> None:
    """Manually add papers to the database by PMID.

    Papers will be added with source_type='expansion' and
    source_details='referenced:{hgnc_id}:manual' to match the citation discovery format.
    """

    if not db_path.exists():
        console.print(f"[red]Error: Database not found: {db_path}[/red]")
        raise typer.Exit(code=1)

    if not pmids:
        console.print("[red]Error: At least one PMID must be provided[/red]")
        raise typer.Exit(code=1)

    hgnc_resolver = HgncResolver.from_file()
    gene_display = hgnc_resolver.get_symbol(hgnc_id)
    console.print(f"Adding {len(pmids)} paper(s) for {gene_display} (HGNC:{hgnc_id})")

    added = 0
    skipped = 0
    failed = 0

    for pmid in pmids:
        # Fetch metadata using "manual" as the citing source
        paper = fetch_paper_metadata(pmid, hgnc_id, citing_doi="manual")

        if paper is None:
            logger.error(f"Failed to fetch metadata for PMID {pmid}")
            failed += 1
            continue

        # Check if already exists (by DOI)
        if check_doi_exists(db_path, paper.doi):
            logger.info(f"DOI {paper.doi} already in database, skipping")
            skipped += 1
            continue

        # Store in database
        inserted = store_referenced_paper(db_path, paper)
        if inserted:
            logger.info(f"Added DOI {paper.doi} for {gene_display} (HGNC:{hgnc_id})")
            added += 1
        else:
            skipped += 1

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  [green]Papers added: {added}[/green]")
    console.print(f"  [blue]Already in database: {skipped}[/blue]")
    if failed > 0:
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
