#!/usr/bin/env python3
"""Download and ingest bioRxiv/medRxiv preprints into database."""

import enum
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from palit.papers import Paper

console = Console()
app = typer.Typer(help="Download and ingest bioRxiv/medRxiv preprints")

logger = logging.getLogger(__name__)

PAGE_SIZE = 100  # API returns max 100 results per page


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
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
    reraise=True,
)
def _fetch_page(client: httpx.Client, url: str) -> dict[str, Any]:
    """Fetch a single page from the bioRxiv/medRxiv API with retries."""
    response = client.get(url, timeout=30)
    response.raise_for_status()
    result: dict[str, Any] = response.json()
    return result


def _parse_paper(record: dict[str, Any], server: RxivServer, source_details: str) -> Paper:
    """Parse a single API record into a Paper."""
    source_metadata: dict[str, object] = {
        "version": record["version"],
        "category": record["category"],
    }
    license_val = record.get("license")
    if license_val:
        source_metadata["license"] = license_val
    jatsxml = record.get("jatsxml")
    if jatsxml:
        source_metadata["jatsxml_url"] = jatsxml

    published = record.get("published", "NA")
    if published and published != "NA":
        source_metadata["published_doi"] = published

    return Paper(
        doi=record["doi"],
        pmid=None,
        title=record["title"],
        abstract=record["abstract"],
        authors=record["authors"],
        journal=server.journal,
        source=server.source,
        source_date=record["date"],
        source_metadata=source_metadata,
        source_type="initial",
        source_details=source_details,
    )


def fetch_papers(server: RxivServer, start_date: str, end_date: str) -> list[Paper]:
    """Fetch all papers from a bioRxiv/medRxiv server for a date range.

    Paginates through the API (100 results per page) until all results are fetched.
    """
    source_details = f"{server.source}/{start_date}/{end_date}"
    papers: list[Paper] = []
    cursor = 0

    with httpx.Client() as client:
        while True:
            url = f"{server.base_url}/details/{server.source}/{start_date}/{end_date}/{cursor}/json"
            logger.info(f"Fetching {server.journal} cursor={cursor}")

            data = _fetch_page(client, url)
            collection = data.get("collection", [])

            if not collection:
                break

            for record in collection:
                papers.append(_parse_paper(record, server, source_details))

            # messages[0].count tells us how many results were in this page
            messages = data.get("messages", [])
            count = messages[0]["count"] if messages else 0

            if count < PAGE_SIZE:
                break

            cursor += PAGE_SIZE

    logger.info(f"Fetched {len(papers)} papers from {server.journal}")
    return papers


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
                    json.dumps(p.source_metadata),
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


@app.callback(invoke_without_command=True)
def main(
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    db_path: Path = typer.Option(Path("data/db.sqlite"), "--db-path", help="Database path"),
    server: str = typer.Option("all", "--server", help="Server: biorxiv, medrxiv, or all"),
) -> None:
    """Ingest preprints from bioRxiv and/or medRxiv into database."""
    # Resolve which servers to fetch from
    server_lower = server.lower()
    if server_lower == "all":
        servers = list(RxivServer)
    elif server_lower == "biorxiv":
        servers = [RxivServer.BIORXIV]
    elif server_lower == "medrxiv":
        servers = [RxivServer.MEDRXIV]
    else:
        console.print(f"[red]Unknown server: {server}. Use biorxiv, medrxiv, or all.[/red]")
        raise typer.Exit(1)

    console.print(
        f"[cyan]Ingesting preprints: {start_date} to {end_date} "
        f"from {', '.join(s.journal for s in servers)}[/cyan]"
    )
    console.print(f"[cyan]Database: {db_path}[/cyan]")

    if not db_path.exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        raise typer.Exit(1)

    # Fetch and insert from each server
    total_inserted = 0
    for rxiv_server in servers:
        console.print(f"\n[bold]Fetching from {rxiv_server.journal}...[/bold]")
        papers = fetch_papers(rxiv_server, start_date, end_date)

        if not papers:
            console.print(f"  No papers found from {rxiv_server.journal}")
            continue

        # Count papers with published DOIs
        published_count = sum(
            1 for p in papers if p.source_metadata.get("published_doi") is not None
        )

        inserted = insert_papers(papers, db_path)
        total_inserted += inserted
        console.print(
            f"  [green]Fetched {len(papers)} papers "
            f"({published_count} already published), inserted/updated {inserted}[/green]"
        )

    console.print(
        f"\n[green]Done. {total_inserted} total papers inserted/updated in {db_path}[/green]"
    )


if __name__ == "__main__":
    app()
