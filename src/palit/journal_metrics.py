#!/usr/bin/env python3
"""Fetch OpenAlex figures for the journals of a database's papers.

The report orders the papers awaiting manual download by their journal's two-year
mean citedness in OpenAlex, an open analogue of the journal impact factor. This
stage looks up every journal_issn of the database's papers that has no
journal_metrics row yet and stores what OpenAlex gives for it, so the report reads
the figures from the database and its order does not change between generations.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import httpx2
import typer
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

app = typer.Typer(help="Fetch OpenAlex citedness figures for the journals of a database's papers")

logger = logging.getLogger(__name__)
logging.getLogger("httpx2").setLevel(logging.WARNING)

OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"
# OpenAlex ORs at most 100 values in one filter.
ISSNS_PER_REQUEST = 100
# Above the most sources one request can match: one ISSN can match several sources.
SOURCES_PER_PAGE = 200
_USER_AGENT = f"palit/{package_version('palit')} (journal metrics)"


@dataclass(frozen=True)
class JournalMetric:
    """What OpenAlex gives for one ISSN; all None when it has no journal with it."""

    issn: str
    openalex_source_id: str | None
    openalex_display_name: str | None
    two_year_mean_citedness: float | None


def unfetched_issns(conn: sqlite3.Connection) -> list[str]:
    """The distinct journal ISSNs of the database's papers that have no metrics row."""
    rows = conn.execute(
        """
        SELECT DISTINCT p.journal_issn FROM papers p
        WHERE p.journal_issn IS NOT NULL
          AND NOT EXISTS (SELECT 1 FROM journal_metrics m WHERE m.issn = p.journal_issn)
        ORDER BY p.journal_issn
        """
    )
    return [issn for (issn,) in rows]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=2, max=60),
    retry=retry_if_exception_type((httpx2.HTTPStatusError, httpx2.TransportError)),
    reraise=True,
)
def fetch_sources(client: httpx2.Client, issns: list[str]) -> list[dict[str, Any]]:
    """The OpenAlex sources listing any of *issns* (at most ISSNS_PER_REQUEST)."""
    response = client.get(
        OPENALEX_SOURCES_URL,
        params={
            "filter": "issn:" + "|".join(issns),
            "select": "id,display_name,issn,works_count,summary_stats",
            "per-page": str(SOURCES_PER_PAGE),
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if data["meta"]["count"] > SOURCES_PER_PAGE:
        raise RuntimeError(
            f"OpenAlex matched {data['meta']['count']} sources for {len(issns)} ISSNs, "
            f"more than the {SOURCES_PER_PAGE} of one page"
        )
    results: list[dict[str, Any]] = data["results"]
    return results


def metrics_for(issns: list[str], sources: list[dict[str, Any]]) -> list[JournalMetric]:
    """Each of *issns* with the figures of the source in *sources* that lists it.

    OpenAlex holds a few journals twice, as sources sharing an ISSN; the source with
    the most works is the journal's main record. ISSNs are compared without regard
    to the case of a check digit X.
    """
    by_issn: dict[str, dict[str, Any]] = {}
    for source in sources:
        for issn in source["issn"]:
            held = by_issn.get(issn.upper())
            if held is None or source["works_count"] > held["works_count"]:
                by_issn[issn.upper()] = source

    metrics = []
    for issn in issns:
        journal = by_issn.get(issn.upper())
        if journal is None:
            metrics.append(JournalMetric(issn, None, None, None))
            continue
        metrics.append(
            JournalMetric(
                issn=issn,
                openalex_source_id=journal["id"],
                openalex_display_name=journal["display_name"],
                two_year_mean_citedness=journal["summary_stats"]["2yr_mean_citedness"],
            )
        )
    return metrics


def store_metrics(conn: sqlite3.Connection, metrics: list[JournalMetric], fetched_at: str) -> None:
    conn.executemany(
        """
        INSERT INTO journal_metrics
            (issn, openalex_source_id, openalex_display_name, two_year_mean_citedness,
             fetched_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                m.issn,
                m.openalex_source_id,
                m.openalex_display_name,
                m.two_year_mean_citedness,
                fetched_at,
            )
            for m in metrics
        ],
    )
    conn.commit()


def fetch_journal_metrics(db_path: Path, client: httpx2.Client) -> list[JournalMetric]:
    """Look up the journals without metrics in OpenAlex and store what it gives.

    Each request's results are stored before the next, so an interrupted run
    resumes with the journals still missing.
    """
    with sqlite3.connect(db_path) as conn:
        issns = unfetched_issns(conn)
        logger.info("%d journals without OpenAlex figures", len(issns))
        fetched: list[JournalMetric] = []
        for i in range(0, len(issns), ISSNS_PER_REQUEST):
            batch = issns[i : i + ISSNS_PER_REQUEST]
            metrics = metrics_for(batch, fetch_sources(client, batch))
            store_metrics(conn, metrics, datetime.now(UTC).isoformat(timespec="seconds"))
            fetched.extend(metrics)
    return fetched


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        Path("data/db.sqlite"),
        "--db-path",
        help="Path to SQLite database",
    ),
) -> None:
    """Fetch OpenAlex citedness figures for the journals of the database's papers.

    Only journals without a stored row are looked up, so a re-run fetches nothing
    unless papers from new journals have arrived since.
    """
    with httpx2.Client(headers={"User-Agent": _USER_AGENT}) as client:
        fetched = fetch_journal_metrics(db_path, client)
    unknown = sum(m.openalex_source_id is None for m in fetched)
    without_figure = sum(
        m.openalex_source_id is not None and m.two_year_mean_citedness is None for m in fetched
    )
    logger.info(
        "Stored figures for %d journals: %d unknown to OpenAlex, %d without a citedness figure",
        len(fetched),
        unknown,
        without_figure,
    )


if __name__ == "__main__":
    app()
