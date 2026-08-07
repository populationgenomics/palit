#!/usr/bin/env python3
"""Discover and download papers referenced in evidence extractions."""

import json
import logging
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import typer
from rich.console import Console
from rich.progress import track

from palit.hgnc import HgncResolver
from palit.papers import (
    CrossrefMetadata,
    Paper,
    format_crossref_authors,
    parse_crossref_date,
    serialize_source_metadata,
    strip_xml_tags,
)
from palit.pubmed_xml import extract_papers_from_xml

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# PubMed reads these as boolean operators even in lowercase, so a title using them as
# ordinary words is mangled: "UV-sensitive syndrome but not Cockayne syndrome" becomes
# a NOT clause excluding the very paper being searched for. They carry no search signal
# in a title, so dropping them costs nothing.
BOOLEAN_OPERATORS = re.compile(r"\b(?:AND|OR|NOT)\b", re.IGNORECASE)

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


def strip_boolean_operators(title: str) -> str:
    """Remove the words PubMed would parse as boolean operators from a title."""
    return " ".join(BOOLEAN_OPERATORS.sub(" ", title).split())


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
            "term": strip_boolean_operators(title),
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


def fetch_paper_metadata_by_doi(doi: str, hgnc_id: int, citing_doi: str) -> Paper | None:
    """Fetch paper metadata from CrossRef by DOI.

    Args:
        doi: Digital Object Identifier
        hgnc_id: HGNC ID for source_details
        citing_doi: DOI of paper citing this reference, or "manual" for manually added papers

    Returns:
        Paper object if successful, None otherwise
    """
    url = f"https://api.crossref.org/works/{quote(doi, safe='')}"
    try:
        response = httpx.get(url, timeout=30.0)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning(f"CrossRef fetch failed for DOI {doi}: {e}")
        return None

    record: dict[str, Any] = data["message"]

    titles = record.get("title", [])
    if not titles:
        logger.warning(f"No title in CrossRef response for DOI {doi}")
        return None

    abstract_raw = record.get("abstract", "")
    abstract = strip_xml_tags(abstract_raw) if abstract_raw else ""

    container_titles = record.get("container-title", [])
    if container_titles:
        journal = container_titles[0]
    else:
        # Preprints lack container-title; fall back to institution name (e.g. "medRxiv")
        institutions = record.get("institution", [])
        journal = institutions[0]["name"] if institutions else ""

    date_obj = record.get("published") or record.get("created")
    if not date_obj:
        logger.warning(f"No date in CrossRef response for DOI {doi}")
        return None

    return Paper(
        doi=doi,
        pmid=None,
        title=titles[0],
        abstract=abstract,
        authors=format_crossref_authors(record.get("author", [])),
        journal=journal,
        source="crossref",
        source_date=parse_crossref_date(date_obj),
        source_metadata=CrossrefMetadata(),
        source_type="expansion",
        source_details=f"referenced:{hgnc_id}:{citing_doi}",
    )


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

        console.print("\nHint: To add papers manually, look up PMIDs or DOIs and run:")
        console.print(
            f"  uv run palit discover-citations add --db-path {db_path} --hgnc-id HGNC_ID PMID1 DOI1 ..."
        )
        console.print("[dim]  (PMID-added papers without a DOI in PubMed will be skipped)[/dim]")

    if new_papers > 0:
        console.print(
            "\n[green]Run 'download-papers' and 'extract-evidence' to process the new papers[/green]"
        )


def _classify_identifier(identifier: str) -> tuple[str, str | int]:
    """Classify a CLI argument as a DOI or PMID.

    DOIs always contain '/', PMIDs are pure integers.

    Returns:
        ("doi", doi_string) or ("pmid", pmid_int)
    """
    if "/" in identifier:
        return ("doi", identifier)
    try:
        return ("pmid", int(identifier))
    except ValueError:
        raise typer.BadParameter(
            f"'{identifier}' is neither a DOI (must contain '/') nor a PMID (must be an integer)"
        ) from None


@app.command()
def add(
    identifiers: list[str] = typer.Argument(..., help="PMIDs or DOIs to add to the database"),
    hgnc_id: int = typer.Option(..., "--hgnc-id", "-g", help="HGNC ID to associate with"),
    db_path: Path = typer.Option(default=Path("data/db.sqlite"), help="Path to SQLite database"),
) -> None:
    """Manually add papers to the database by PMID or DOI.

    Accepts both PMIDs (integers) and DOIs (strings containing '/').
    PMIDs are resolved via PubMed efetch; DOIs are resolved via CrossRef.
    Papers will be added with source_type='expansion' and
    source_details='referenced:{hgnc_id}:manual' to match the citation discovery format.
    """

    if not db_path.exists():
        console.print(f"[red]Error: Database not found: {db_path}[/red]")
        raise typer.Exit(code=1)

    if not identifiers:
        console.print("[red]Error: At least one PMID or DOI must be provided[/red]")
        raise typer.Exit(code=1)

    classified = [_classify_identifier(ident) for ident in identifiers]

    hgnc_resolver = HgncResolver.from_file()
    gene_display = hgnc_resolver.get_symbol(hgnc_id)
    console.print(f"Adding {len(classified)} paper(s) for {gene_display} (HGNC:{hgnc_id})")

    added = 0
    skipped = 0
    failed = 0

    for id_type, id_value in classified:
        if id_type == "pmid":
            assert isinstance(id_value, int)
            paper = fetch_paper_metadata(id_value, hgnc_id, citing_doi="manual")
            display_id = f"PMID {id_value}"
        else:
            assert isinstance(id_value, str)
            # For DOIs we can check existence before making the API call
            if check_doi_exists(db_path, id_value):
                logger.info(f"DOI {id_value} already in database, skipping")
                skipped += 1
                continue
            paper = fetch_paper_metadata_by_doi(id_value, hgnc_id, citing_doi="manual")
            display_id = f"DOI {id_value}"

        if paper is None:
            logger.error(f"Failed to fetch metadata for {display_id}")
            failed += 1
            continue

        if check_doi_exists(db_path, paper.doi):
            logger.info(f"DOI {paper.doi} already in database, skipping")
            skipped += 1
            continue

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
