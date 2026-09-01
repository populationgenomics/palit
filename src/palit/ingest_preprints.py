#!/usr/bin/env python3
"""Download and ingest preprints from bioRxiv, medRxiv, and Research Square."""

import enum
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import httpx2
import typer
from rich.console import Console
from rich.progress import TaskID
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from palit import ledger as ledger_ops
from palit.papers import (
    Paper,
    ResearchSquareMetadata,
    RxivMetadata,
    format_crossref_authors,
    parse_crossref_date,
    serialize_source_metadata,
    strip_xml_tags,
)
from palit.progress import LoggingProgress as Progress

console = Console()
app = typer.Typer(help="Download and ingest preprints from bioRxiv, medRxiv, and Research Square")

logger = logging.getLogger(__name__)
logging.getLogger("httpx2").setLevel(logging.WARNING)

CROSSREF_PAGE_SIZE = 1000  # Crossref allows up to 1000 results per page
_RS_VERSION_SUFFIX = re.compile(r"/v\d+$")


class RxivServer(enum.Enum):
    """bioRxiv/medRxiv API configuration."""

    BIORXIV = ("biorxiv", "https://api.biorxiv.org", "bioRxiv")
    MEDRXIV = ("medrxiv", "https://api.medrxiv.org", "medRxiv")

    def __init__(self, source: str, base_url: str, journal: str) -> None:
        self.source = source
        self.base_url = base_url
        self.journal = journal


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=2, max=60),
    retry=retry_if_exception_type((httpx2.HTTPStatusError, httpx2.TransportError)),
    reraise=True,
)
def _fetch_rxiv_page(client: httpx2.Client, url: str) -> dict[str, Any]:
    """Fetch a single page from the bioRxiv/medRxiv API with retries."""
    response = client.get(url, timeout=90)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def _parse_rxiv_paper(record: dict[str, Any], server: RxivServer, source_details: str) -> Paper:
    """Parse a single API record into a Paper."""
    published = record.get("published", "NA")

    return Paper(
        doi=record["doi"],
        pmid=None,
        title=record["title"],
        abstract=record["abstract"],
        authors=record["authors"],
        journal=server.journal,
        source=server.source,
        source_date=record["date"],
        source_metadata=RxivMetadata(
            version=int(record["version"]),
            category=record["category"],
            license=record.get("license"),
            jatsxml_url=record.get("jatsxml"),
            published_doi=published if published and published != "NA" else None,
        ),
        source_type="initial",
        source_details=source_details,
    )


def fetch_rxiv_papers(
    server: RxivServer,
    start_date: str,
    end_date: str,
    progress: Progress,
    task: TaskID,
) -> list[Paper]:
    """Fetch all papers from a bioRxiv/medRxiv server for a date range.

    Paginates until the API reports no further posts. The page size is chosen by
    the server and differs between bioRxiv and medRxiv, so it is never assumed.
    A preprint revised inside the range is returned once per posted version;
    only the highest version of each is kept.
    """
    source_details = f"{server.source}/{start_date}/{end_date}"
    highest: dict[str, Paper] = {}
    versions: dict[str, int] = {}
    fetched = 0
    cursor = 0

    with httpx2.Client() as client:
        while True:
            url = f"{server.base_url}/details/{server.source}/{start_date}/{end_date}/{cursor}/json"

            data = _fetch_rxiv_page(client, url)

            # Past the last page the API answers "no posts found" and omits the
            # collection entirely; that is the only reliable end-of-results signal.
            collection = data.get("collection", [])
            if not collection:
                break

            progress.update(task, total=int(data["messages"][0]["total"]))

            for record in collection:
                doi = record["doi"]
                version = int(record["version"])
                if versions.get(doi, 0) < version:
                    versions[doi] = version
                    highest[doi] = _parse_rxiv_paper(record, server, source_details)

            fetched += len(collection)
            progress.update(task, completed=fetched)
            cursor += len(collection)

    papers = list(highest.values())
    logger.info(
        f"Fetched {fetched} records from {server.journal}, "
        f"{len(papers)} unique papers after version dedup"
    )
    return papers


def _normalize_rs_doi(doi: str) -> str:
    """Strip version suffix from Research Square DOI.

    '10.21203/rs.3.rs-7989161/v1' -> '10.21203/rs.3.rs-7989161'
    """
    return _RS_VERSION_SUFFIX.sub("", doi)


def _parse_crossref_paper(record: dict[str, Any], source_details: str) -> Paper:
    """Parse a Crossref work record into a Paper for Research Square."""
    versioned_doi = record["DOI"]
    doi = _normalize_rs_doi(versioned_doi)

    version_match = _RS_VERSION_SUFFIX.search(versioned_doi)
    version = int(version_match.group()[2:]) if version_match else 1

    return Paper(
        doi=doi,
        pmid=None,
        title=record["title"][0],
        abstract=strip_xml_tags(record["abstract"]),
        authors=format_crossref_authors(record.get("author", [])),
        journal="Research Square",
        source="researchsquare",
        source_date=parse_crossref_date(record["posted"]),
        source_metadata=ResearchSquareMetadata(version=version, versioned_doi=versioned_doi),
        source_type="initial",
        source_details=source_details,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=2, max=60),
    retry=retry_if_exception_type((httpx2.HTTPStatusError, httpx2.TransportError)),
    reraise=True,
)
def _fetch_crossref_page(client: httpx2.Client, url: str, params: dict[str, str]) -> dict[str, Any]:
    """Fetch a single page from the Crossref API with retries."""
    response = client.get(url, params=params, timeout=90)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def fetch_rs_papers(
    start_date: str,
    end_date: str,
    mailto: str,
    progress: Progress,
    task: TaskID,
) -> list[Paper]:
    """Fetch Research Square preprints from Crossref API for a date range.

    Uses Crossref cursor-based deep paging (1000 results per page) until all
    results are fetched. Cursor paging is required because offset paging is
    capped at 10000 results, which these date ranges routinely exceed.
    Deduplicates multiple versions by normalized DOI, keeping the highest version.
    """
    source_details = f"researchsquare/{start_date}/{end_date}"
    papers: list[Paper] = []

    url = "https://api.crossref.org/works"
    params = {
        "filter": (
            "type:posted-content,prefix:10.21203,"
            f"from-posted-date:{start_date},until-posted-date:{end_date},"
            "has-abstract:true"
        ),
        "rows": str(CROSSREF_PAGE_SIZE),
        "mailto": mailto,
        "cursor": "*",
    }

    with httpx2.Client() as client:
        while True:
            data = _fetch_crossref_page(client, url, params)
            message = data["message"]
            items: list[dict[str, Any]] = message.get("items", [])

            if not items:
                break

            total_results: int = message["total-results"]
            progress.update(task, total=total_results)

            for record in items:
                papers.append(_parse_crossref_paper(record, source_details))

            progress.update(task, completed=len(papers))
            params["cursor"] = message["next-cursor"]

    # Deduplicate by normalized DOI, keeping highest version
    seen: dict[str, Paper] = {}
    for paper in papers:
        existing = seen.get(paper.doi)
        paper_meta = paper.source_metadata
        existing_meta = existing.source_metadata if existing else None
        assert isinstance(paper_meta, ResearchSquareMetadata)
        if not existing_meta or (
            isinstance(existing_meta, ResearchSquareMetadata)
            and paper_meta.version > existing_meta.version
        ):
            seen[paper.doi] = paper

    deduped = list(seen.values())
    logger.info(
        f"Fetched {len(papers)} records from Research Square, "
        f"{len(deduped)} unique papers after version dedup"
    )
    return deduped


def insert_papers(papers: list[Paper], db_path: Path) -> int:
    """Insert papers into the database.

    Returns the number of rows inserted or updated.
    """
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO papers
            (doi, pmid, title, abstract, authors, journal, source, source_date,
             source_metadata, source_type, source_details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doi) DO UPDATE SET
                title = excluded.title,
                abstract = excluded.abstract,
                authors = excluded.authors,
                journal = excluded.journal,
                source_date = excluded.source_date,
                source_metadata = excluded.source_metadata,
                source_type = excluded.source_type,
                source_details = excluded.source_details
            -- Only update within the same source. Never cross-source.
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
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


VALID_SERVERS: dict[str, str] = {
    "biorxiv": "bioRxiv",
    "medrxiv": "medRxiv",
    "researchsquare": "Research Square",
}


def _fetch_server(
    server: str, start_date: str, end_date: str, mailto: str, progress: Progress, task: TaskID
) -> list[Paper]:
    """Fetch papers from a single preprint server."""
    if server == "biorxiv":
        return fetch_rxiv_papers(RxivServer.BIORXIV, start_date, end_date, progress, task)
    elif server == "medrxiv":
        return fetch_rxiv_papers(RxivServer.MEDRXIV, start_date, end_date, progress, task)
    elif server == "researchsquare":
        return fetch_rs_papers(start_date, end_date, mailto, progress, task)
    else:
        raise ValueError(f"Unknown server: {server}")


@app.callback(invoke_without_command=True)
def main(
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    db_path: Path = typer.Option(Path("data/db.sqlite"), "--db-path", help="Database path"),
    servers: list[str] = typer.Option(
        list(VALID_SERVERS), "--server", help="Preprint servers to ingest from"
    ),
    mailto: str = typer.Option(
        "panelapp-support@mcri.edu.au", "--mailto", help="Contact email for Crossref polite pool"
    ),
    ledger: Path = typer.Option(
        ledger_ops.DEFAULT_LEDGER_PATH, "--ledger", help="Ledger database path"
    ),
) -> None:
    """Ingest preprints from bioRxiv, medRxiv, and/or Research Square."""
    for s in servers:
        if s not in VALID_SERVERS:
            console.print(f"[red]Unknown server: {s}. Valid: {', '.join(VALID_SERVERS)}[/red]")
            raise typer.Exit(1)

    if not ledger.exists():
        console.print(
            f"[red]Ledger not found at {ledger}. Create it first: "
            f"`palit ledger init --ledger {ledger}`.[/red]"
        )
        raise typer.Exit(1)

    # Skip preprints whose disposition the ledger has already settled (majority
    # not-relevant, or downloaded). Relevant-not-downloaded carry-overs come back
    # through seed-run-db during ingest-pubmed, so they need no handling here.
    settled = ledger_ops.settled_dois(ledger)
    console.print(f"[cyan]Skipping {len(settled)} DOIs already settled in the ledger[/cyan]")

    console.print(
        f"[cyan]Ingesting preprints: {start_date} to {end_date} "
        f"from {', '.join(VALID_SERVERS[s] for s in servers)}[/cyan]"
    )
    console.print(f"[cyan]Database: {db_path}[/cyan]")

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
        console.print(f"[green]Created database at {db_path}[/green]")

    total_inserted = 0
    with Progress(console=console) as progress:
        for srv in servers:
            display = VALID_SERVERS[srv]
            task = progress.add_task(display, total=None)
            papers = _fetch_server(srv, start_date, end_date, mailto, progress, task)

            if not papers:
                progress.update(task, total=0, completed=0)
                continue

            if settled:
                before = len(papers)
                papers = [p for p in papers if p.doi not in settled]
                filtered = before - len(papers)
                if filtered:
                    logger.info(f"{display}: skipped {filtered} DOIs already settled in ledger")

            inserted = insert_papers(papers, db_path)
            total_inserted += inserted

    console.print(
        f"\n[green]Done. {total_inserted} total papers inserted/updated in {db_path}[/green]"
    )


if __name__ == "__main__":
    app()
