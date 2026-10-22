"""Tests for journal-metrics: matching ISSNs to OpenAlex sources and storing figures."""

import sqlite3
from pathlib import Path
from typing import Any

import httpx2

from palit.journal_metrics import (
    ISSNS_PER_REQUEST,
    JournalMetric,
    fetch_journal_metrics,
    metrics_for,
)

ROOT = Path(__file__).resolve().parents[1]


def _source(
    source_id: str, issns: list[str], works: int, citedness: float | None
) -> dict[str, Any]:
    return {
        "id": f"https://openalex.org/{source_id}",
        "display_name": f"Journal {source_id}",
        "issn": issns,
        "works_count": works,
        "summary_stats": {"2yr_mean_citedness": citedness},
    }


def test_each_issn_takes_the_figures_of_the_source_listing_it() -> None:
    sources = [_source("S1", ["0002-9297", "1537-6605"], 100, 7.3)]
    assert metrics_for(["1537-6605", "1111-1111"], sources) == [
        JournalMetric("1537-6605", "https://openalex.org/S1", "Journal S1", 7.3),
        JournalMetric("1111-1111", None, None, None),
    ]


def test_an_issn_two_sources_share_takes_the_source_with_more_works() -> None:
    sources = [
        _source("S1", ["1234-5678"], 50, 1.0),
        _source("S2", ["1234-5678"], 900, 4.0),
        _source("S3", ["1234-5678"], 10, 9.0),
    ]
    (metric,) = metrics_for(["1234-5678"], sources)
    assert metric.openalex_source_id == "https://openalex.org/S2"


def test_issns_match_whatever_the_case_of_the_check_digit() -> None:
    (metric,) = metrics_for(["1234-567x"], [_source("S1", ["1234-567X"], 1, 2.0)])
    assert (metric.issn, metric.two_year_mean_citedness) == ("1234-567x", 2.0)


def _run_db(tmp_path: Path, issns: list[str | None]) -> Path:
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, journal_issn) VALUES (?, 't', 'pubmed', ?)",
            [(f"10.1/{i}", issn) for i, issn in enumerate(issns)],
        )
    return db_path


def test_fetch_looks_up_each_journal_once_in_batches_and_stores_every_answer(
    tmp_path: Path,
) -> None:
    issns = [f"0000-{i:04d}" for i in range(ISSNS_PER_REQUEST + 1)]
    db_path = _run_db(tmp_path, [*issns, issns[0], None])
    requested: list[list[str]] = []

    def openalex(request: httpx2.Request) -> httpx2.Response:
        batch = request.url.params["filter"].removeprefix("issn:").split("|")
        requested.append(batch)
        # OpenAlex knows every journal but the first.
        known = [_source(issn, [issn], 1, 1.5) for issn in batch if issn != issns[0]]
        return httpx2.Response(200, json={"meta": {"count": len(known)}, "results": known})

    with httpx2.Client(transport=httpx2.MockTransport(openalex)) as client:
        fetched = fetch_journal_metrics(db_path, client)
        # A second run finds every journal looked up.
        assert fetch_journal_metrics(db_path, client) == []

    assert [len(batch) for batch in requested] == [ISSNS_PER_REQUEST, 1]
    assert len(fetched) == len(issns)
    with sqlite3.connect(db_path) as conn:
        rows = dict(conn.execute("SELECT issn, two_year_mean_citedness FROM journal_metrics"))
        (unknown,) = conn.execute(
            "SELECT openalex_source_id FROM journal_metrics WHERE issn = ?", (issns[0],)
        ).fetchone()
    assert set(rows) == set(issns)
    assert rows[issns[0]] is None and unknown is None
    assert rows[issns[1]] == 1.5
