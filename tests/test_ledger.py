"""Tests for the PubMed ingestion ledger: disposition partition, upsert, sync."""

import gzip
import json
import sqlite3
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from palit import ledger, pubmed_ftp
from palit.papers import Paper, PubmedMetadata
from palit.pubmed_ftp import RemoteFile

# --- helpers ----------------------------------------------------------------


def _make_paper(doi: str, pmid: int | None, source_date: str, title: str = "T") -> Paper:
    return Paper(
        doi=doi,
        pmid=pmid,
        title=title,
        abstract="abstract",
        authors="Smith, Jane",
        journal="Test Journal",
        source="pubmed",
        source_date=source_date,
        source_metadata=PubmedMetadata(pmcid=None),
        source_type="initial",
        source_details="test",
    )


def _new_ledger(tmp_path: Path) -> Path:
    path = tmp_path / "ledger.sqlite"
    ledger.create_ledger(path)
    return path


def _new_run_db(tmp_path: Path, name: str = "run.sqlite") -> Path:
    path = tmp_path / name
    conn = sqlite3.connect(path)
    try:
        conn.executescript(Path("schema.sql").read_text())
    finally:
        conn.close()
    return path


def _assessment(*relevant: bool) -> str:
    return json.dumps([{"relevant": r, "associations": []} for r in relevant])


def _articleset_gz(articles: list[tuple[str, int, str]]) -> bytes:
    """Build a gzipped PubmedArticleSet from (doi, pmid, YYYY-MM-DD) tuples."""
    items = []
    for doi, pmid, day in articles:
        y, m, d = day.split("-")
        items.append(
            f"""
            <PubmedArticle>
              <MedlineCitation>
                <Article>
                  <ArticleTitle>Title {doi}</ArticleTitle>
                  <Abstract><AbstractText>Abstract {doi}</AbstractText></Abstract>
                  <AuthorList><Author><LastName>Smith</LastName>
                    <ForeName>Jane</ForeName></Author></AuthorList>
                  <Journal><Title>Test Journal</Title></Journal>
                </Article>
              </MedlineCitation>
              <PubmedData>
                <History><PubMedPubDate PubStatus="entrez">
                  <Year>{y}</Year><Month>{m}</Month><Day>{d}</Day>
                </PubMedPubDate></History>
                <ArticleIdList>
                  <ArticleId IdType="pubmed">{pmid}</ArticleId>
                  <ArticleId IdType="doi">{doi}</ArticleId>
                </ArticleIdList>
              </PubmedData>
            </PubmedArticle>
            """
        )
    xml = f"<PubmedArticleSet>{''.join(items)}</PubmedArticleSet>".encode()
    return gzip.compress(xml)


# --- date math --------------------------------------------------------------


@pytest.mark.parametrize(
    ("d", "months", "expected"),
    [
        (date(2026, 6, 23), 6, date(2025, 12, 23)),
        (date(2026, 3, 31), 1, date(2026, 2, 28)),  # day clamp
        (date(2024, 3, 31), 1, date(2024, 2, 29)),  # leap-year clamp
        (date(2026, 1, 15), 1, date(2025, 12, 15)),  # year boundary
        (date(2027, 1, 31), 1, date(2026, 12, 31)),  # December (month==12 branch)
    ],
)
def test_subtract_months(d: date, months: int, expected: date) -> None:
    assert ledger.subtract_months(d, months) == expected


# --- relevance derivation ---------------------------------------------------


def test_relevant_from_assessment() -> None:
    assert ledger.relevant_from_assessment(None) is None
    assert ledger.relevant_from_assessment("") is None
    assert ledger.relevant_from_assessment(_assessment(True, True)) is None  # not 3
    assert ledger.relevant_from_assessment(_assessment(True, True, False)) == 1
    assert ledger.relevant_from_assessment(_assessment(False, False, True)) == 0


# --- fetch-upsert -----------------------------------------------------------


def test_upsert_is_idempotent_and_preserves_disposition(tmp_path: Path) -> None:
    ledger_path = _new_ledger(tmp_path)
    conn = ledger.connect(ledger_path)
    try:
        inserted, updated = ledger.upsert_papers(
            conn, [_make_paper("10.1/a", 1, "2026-01-15", title="First")], "2026-01-20"
        )
        assert (inserted, updated) == (1, 0)
        # A run owns disposition; simulate it.
        conn.execute("UPDATE ledger SET relevant = 1, download_status = 'scheduled'")
        conn.commit()

        # Re-fetch with a refreshed title + a later last_seen.
        inserted, updated = ledger.upsert_papers(
            conn, [_make_paper("10.1/a", 1, "2026-01-15", title="Refreshed")], "2026-02-01"
        )
        assert (inserted, updated) == (0, 1)
        row = conn.execute(
            "SELECT title, last_seen, relevant, download_status FROM ledger WHERE doi='10.1/a'"
        ).fetchone()
        assert row == ("Refreshed", "2026-02-01", 1, "scheduled")  # disposition untouched
    finally:
        conn.close()


# --- settled / actionable partition -----------------------------------------


def test_seed_run_db_partitions_settled_and_actionable(tmp_path: Path) -> None:
    ledger_path = _new_ledger(tmp_path)
    conn = ledger.connect(ledger_path)
    try:
        papers = [
            _make_paper("10.1/never", 1, "2026-01-10"),  # actionable: never assessed
            _make_paper("10.1/carry", 2, "2026-01-11"),  # actionable: relevant, not downloaded
            _make_paper("10.1/notrel", 3, "2026-01-12"),  # settled: not relevant
            _make_paper("10.1/done", 4, "2026-01-13"),  # settled: downloaded
            _make_paper("10.1/old", 5, "2025-01-01"),  # actionable but outside horizon
        ]
        ledger.upsert_papers(conn, papers, "2026-01-20")
        conn.execute(
            "UPDATE ledger SET relevance_assessment_json=?, relevant=1 WHERE doi='10.1/carry'",
            (_assessment(True, True, True),),
        )
        conn.execute(
            "UPDATE ledger SET relevance_assessment_json=?, relevant=0 WHERE doi='10.1/notrel'",
            (_assessment(False, False, False),),
        )
        conn.execute(
            "UPDATE ledger SET relevance_assessment_json=?, relevant=1, "
            "download_status='downloaded' WHERE doi='10.1/done'",
            (_assessment(True, True, True),),
        )
        conn.commit()
    finally:
        conn.close()

    run_db = _new_run_db(tmp_path)
    floor = ledger.subtract_months(date(2026, 1, 31), 6).isoformat()  # 2025-07-31
    n = ledger.seed_run_db_from_ledger(
        ledger_path, run_db, horizon_floor=floor, end_date="2026-01-31"
    )
    assert n == 2

    run = sqlite3.connect(run_db)
    try:
        dois = {r[0] for r in run.execute("SELECT doi FROM papers")}
        assert dois == {"10.1/never", "10.1/carry"}  # settled + out-of-horizon excluded
        # Carry-over keeps its assessment so it resumes at download (not re-assessed);
        # the never-assessed paper carries none so it gets assessed.
        carry = run.execute(
            "SELECT relevance_assessment_json, download_status FROM papers WHERE doi='10.1/carry'"
        ).fetchone()
        assert carry[0] is not None and carry[1] is None
        never = run.execute(
            "SELECT relevance_assessment_json FROM papers WHERE doi='10.1/never'"
        ).fetchone()
        assert never[0] is None
    finally:
        run.close()


def test_seed_run_db_preserves_existing_rows_and_backfills_pmid(tmp_path: Path) -> None:
    ledger_path = _new_ledger(tmp_path)
    conn = ledger.connect(ledger_path)
    try:
        ledger.upsert_papers(conn, [_make_paper("10.1/shared", 999, "2026-01-10")], "2026-01-20")
    finally:
        conn.close()

    # A preprint row for the same DOI is already present (ingested first), without a PMID.
    run_db = _new_run_db(tmp_path)
    run = sqlite3.connect(run_db)
    try:
        run.execute(
            "INSERT INTO papers (doi, pmid, title, source, source_type) "
            "VALUES ('10.1/shared', NULL, 'Preprint title', 'biorxiv', 'initial')"
        )
        run.commit()
    finally:
        run.close()

    ledger.seed_run_db_from_ledger(
        ledger_path, run_db, horizon_floor="2025-07-31", end_date="2026-01-31"
    )
    run = sqlite3.connect(run_db)
    try:
        title, pmid = run.execute(
            "SELECT title, pmid FROM papers WHERE doi='10.1/shared'"
        ).fetchone()
        assert title == "Preprint title"  # INSERT OR IGNORE did not clobber the preprint
        assert pmid == 999  # but the ledger's PMID was backfilled
    finally:
        run.close()


def test_settled_dois(tmp_path: Path) -> None:
    ledger_path = _new_ledger(tmp_path)
    conn = ledger.connect(ledger_path)
    try:
        ledger.upsert_papers(
            conn,
            [_make_paper(f"10.1/{k}", i, "2026-01-10") for i, k in enumerate("abcd")],
            "2026-01-20",
        )
        conn.execute("UPDATE ledger SET relevant=0 WHERE doi='10.1/a'")
        conn.execute("UPDATE ledger SET download_status='downloaded' WHERE doi='10.1/b'")
        conn.execute("UPDATE ledger SET relevant=1 WHERE doi='10.1/c'")  # actionable
        conn.commit()
    finally:
        conn.close()
    assert ledger.settled_dois(ledger_path) == {"10.1/a", "10.1/b"}


# --- write-back -------------------------------------------------------------


def test_writeback_updates_existing_and_inserts_expansion(tmp_path: Path) -> None:
    ledger_path = _new_ledger(tmp_path)
    conn = ledger.connect(ledger_path)
    try:
        ledger.upsert_papers(conn, [_make_paper("10.1/known", 1, "2026-01-10")], "2026-01-20")
    finally:
        conn.close()

    run_db = _new_run_db(tmp_path)
    run = sqlite3.connect(run_db)
    try:
        # Known paper assessed not-relevant.
        run.execute(
            "INSERT INTO papers (doi, pmid, title, source, source_type, "
            "relevance_assessment_json) VALUES ('10.1/known', 1, 'Known', 'pubmed', 'initial', ?)",
            (_assessment(False, False, False),),
        )
        # An expansion paper the ledger has never seen, downloaded.
        run.execute(
            "INSERT INTO papers (doi, pmid, title, source, source_type, "
            "relevance_assessment_json, download_status) "
            "VALUES ('10.1/expansion', 2, 'Exp', 'crossref', 'expansion', ?, 'downloaded')",
            (_assessment(True, True, True),),
        )
        run.commit()
    finally:
        run.close()

    written = ledger.writeback(ledger_path, run_db, run_id="run_x")
    assert written == 2

    conn = ledger.connect(ledger_path)
    try:
        known = conn.execute(
            "SELECT relevant, reported_run FROM ledger WHERE doi='10.1/known'"
        ).fetchone()
        assert known == (0, "run_x")
        exp = conn.execute(
            "SELECT relevant, download_status, source, title FROM ledger WHERE doi='10.1/expansion'"
        ).fetchone()
        assert exp == (1, "downloaded", "crossref", "Exp")  # inserted with metadata
    finally:
        conn.close()


# --- seed from existing run databases ---------------------------------------


def _new_pmid_era_db(tmp_path: Path, name: str) -> Path:
    """A pre-DOI-migration run DB: PMID-keyed, no `doi` column, `entrez_date`."""
    path = tmp_path / name
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            "CREATE TABLE papers (pmid INTEGER PRIMARY KEY, entrez_date DATE, title TEXT, "
            "relevance_assessment_json JSON, download_status TEXT);"
        )
    finally:
        conn.close()
    return path


def test_more_processed_precedence() -> None:
    assert ledger._more_processed("downloaded", None) == "downloaded"
    assert ledger._more_processed(None, "downloaded") == "downloaded"
    assert ledger._more_processed("scheduled", "manual_required") == "manual_required"
    assert ledger._more_processed("scheduled", None) == "scheduled"


def test_seed_cross_era_resolve_discard_and_latest_wins(tmp_path: Path) -> None:
    ledger_path = _new_ledger(tmp_path)
    conn = ledger.connect(ledger_path)
    try:
        # Simulate a synced ledger: in-window rows carry the PMID<->DOI mapping.
        ledger.upsert_papers(
            conn,
            [
                _make_paper("10.1/x", 100, "2026-03-10"),
                _make_paper("10.1/y", 200, "2026-01-20"),
            ],
            "2026-03-20",
        )
    finally:
        conn.close()

    # PMID-era January run (no DOIs): X not-relevant, Y not-relevant, W unknown PMID.
    jan = _new_pmid_era_db(tmp_path, "db_2026_january_h1.sqlite")
    conn = sqlite3.connect(jan)
    try:
        conn.executemany(
            "INSERT INTO papers (pmid, entrez_date, title, relevance_assessment_json, "
            "download_status) VALUES (?, ?, ?, ?, ?)",
            [
                (100, "2026-01-15", "X", _assessment(False, False, False), "scheduled"),
                (200, "2026-01-20", "Y", _assessment(False, False, False), None),
                (999, "2026-01-25", "W", _assessment(True, True, True), None),  # not in ledger
            ],
        )
        conn.commit()
    finally:
        conn.close()

    # DOI-era March run: X re-assessed relevant + downloaded; Z is brand new.
    mar = _new_run_db(tmp_path, "db_2026_march.sqlite")
    conn = sqlite3.connect(mar)
    try:
        conn.executemany(
            "INSERT INTO papers (doi, pmid, title, source, source_date, source_type, "
            "relevance_assessment_json, download_status) VALUES (?, ?, ?, 'pubmed', ?, 'initial', ?, ?)",
            [
                ("10.1/x", 100, "X", "2026-03-10", _assessment(True, True, True), "downloaded"),
                ("10.1/z", 300, "Z", "2026-03-12", _assessment(False, False, False), None),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    # Oldest -> newest, so March overrides January for X.
    stats = ledger.seed_from_run_dbs(ledger_path, [jan, mar], min_crdt="2025-12-01")
    assert (stats.pmid_resolved, stats.pmid_discarded, stats.doi_era_rows) == (2, 1, 2)
    assert stats.dispositions == 3  # x, y, z (w discarded)

    conn = ledger.connect(ledger_path)
    try:
        # X: latest (March) verdict + downloaded both win.
        assert conn.execute(
            "SELECT relevant, download_status FROM ledger WHERE doi='10.1/x'"
        ).fetchone() == (1, "downloaded")
        # Y: resolved from PMID, not-relevant.
        assert conn.execute("SELECT relevant FROM ledger WHERE doi='10.1/y'").fetchone() == (0,)
        # Z: brand-new DOI-era row inserted with metadata.
        assert conn.execute("SELECT relevant, title FROM ledger WHERE doi='10.1/z'").fetchone() == (
            0,
            "Z",
        )
        # W's PMID never resolved to a DOI -> nothing inserted.
        assert conn.execute("SELECT count(*) FROM ledger WHERE pmid=999").fetchone()[0] == 0
    finally:
        conn.close()


def test_seed_respects_crdt_floor(tmp_path: Path) -> None:
    ledger_path = _new_ledger(tmp_path)
    run = _new_run_db(tmp_path, "db_2026_april.sqlite")
    conn = sqlite3.connect(run)
    try:
        conn.executemany(
            "INSERT INTO papers (doi, pmid, title, source, source_date, source_type, "
            "relevance_assessment_json) VALUES (?, ?, ?, 'pubmed', ?, 'initial', ?)",
            [
                ("10.1/recent", 1, "R", "2026-04-10", _assessment(False, False, False)),
                ("10.1/ancient", 2, "A", "2012-04-10", _assessment(False, False, False)),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    stats = ledger.seed_from_run_dbs(ledger_path, [run], min_crdt="2025-12-01")
    assert stats.doi_era_rows == 1  # the 2012 expansion paper is below the floor
    conn = ledger.connect(ledger_path)
    try:
        assert {r[0] for r in conn.execute("SELECT doi FROM ledger")} == {"10.1/recent"}
    finally:
        conn.close()


# --- FTP sync (fixture-driven, no network) ----------------------------------


@pytest.fixture
def _patch_ftp(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    """Patch the FTP primitives; tests set `state['files']` / `state['baseline']`."""
    state: dict[str, Any] = {"baseline": RemoteFile(26, 1334), "files": []}

    def fake_baseline(client: httpx.Client | None) -> RemoteFile:
        baseline: RemoteFile = state["baseline"]
        return baseline

    def fake_list(client: httpx.Client | None, subdir: str) -> list[RemoteFile]:
        files: list[RemoteFile] = sorted(state["files"])
        return files

    def fake_download(
        client: httpx.Client | None, subdir: str, remote: RemoteFile, dest_dir: Path
    ) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        path: Path = dest_dir / remote.name
        path.write_bytes(state["content"])
        return path

    monkeypatch.setattr(pubmed_ftp, "read_baseline_max", fake_baseline)
    monkeypatch.setattr(pubmed_ftp, "list_remote_files", fake_list)
    monkeypatch.setattr(pubmed_ftp, "download_verified", fake_download)
    yield state


def test_sync_ftp_inits_cursor_filters_crdt_and_is_idempotent(
    tmp_path: Path, _patch_ftp: dict
) -> None:
    ledger_path = _new_ledger(tmp_path)
    conn = ledger.connect(ledger_path)
    # One in-window article, one below the CRDT floor.
    _patch_ftp["content"] = _articleset_gz(
        [("10.1/in", 100, "2026-01-15"), ("10.1/out", 101, "2025-01-15")]
    )
    _patch_ftp["files"] = [RemoteFile(26, 1335), RemoteFile(26, 1336)]
    try:
        stats = ledger.sync_ftp(
            conn,
            min_crdt="2025-07-31",
            work_dir=tmp_path / "uf",
            client=None,
            today="2026-02-01",
        )
        assert stats.files_applied == 2
        assert stats.last_applied_file_number == 1336
        dois = {r[0] for r in conn.execute("SELECT doi FROM ledger")}
        assert dois == {"10.1/in"}  # out-of-window CRDT filtered out
        # Cursor persisted: a second sync with no new files applies nothing.
        stats2 = ledger.sync_ftp(
            conn, min_crdt="2025-07-31", work_dir=tmp_path / "uf", client=None, today="2026-02-02"
        )
        assert stats2.files_applied == 0
    finally:
        conn.close()


def test_sync_ftp_handles_annual_rollover(tmp_path: Path, _patch_ftp: dict) -> None:
    ledger_path = _new_ledger(tmp_path)
    conn = ledger.connect(ledger_path)
    _patch_ftp["content"] = _articleset_gz([("10.1/y26", 200, "2026-06-15")])
    _patch_ftp["files"] = [RemoteFile(26, 1335)]
    try:
        ledger.sync_ftp(
            conn, min_crdt="2025-07-31", work_dir=tmp_path / "uf", client=None, today="2026-06-20"
        )
        # New annual baseline appears; numbering resets under year 27.
        _patch_ftp["baseline"] = RemoteFile(27, 1400)
        _patch_ftp["files"] = [RemoteFile(26, 1335), RemoteFile(27, 1401)]
        _patch_ftp["content"] = _articleset_gz([("10.1/y27", 201, "2026-12-15")])
        stats = ledger.sync_ftp(
            conn, min_crdt="2026-06-01", work_dir=tmp_path / "uf", client=None, today="2026-12-20"
        )
        assert stats.baseline_year == 27
        assert stats.last_applied_file_number == 1401
        assert stats.files_applied == 1  # only the post-baseline year-27 file
        dois = {r[0] for r in conn.execute("SELECT doi FROM ledger")}
        assert dois == {"10.1/y26", "10.1/y27"}
    finally:
        conn.close()
