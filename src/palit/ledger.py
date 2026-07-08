#!/usr/bin/env python3
"""The PubMed ingestion ledger: dedup/disposition memory across all runs.

The ledger is a single canonical database (default
``data/pubmed_ingestion_ledger.sqlite``) holding one row per PubMed DOI ever
fetched, with refreshed bibliographic metadata and the terminal disposition the
owning run wrote back. It replaces the old per-run buffer window + single
``--previous-db`` set-difference: papers already *settled* (majority not-relevant,
or downloaded) are never reconsidered, while *actionable* papers (never assessed,
or relevant-but-not-downloaded) are re-emitted into each new run's database.

Operations (§10.3 of specs/pubmed_robust_ingestion.md):

1. fetch-upsert  -- refresh metadata + last_seen from a fetched corpus; never
                    touches disposition (the run owns that).
2. seed-run-db   -- copy the actionable set into a fresh run database.
3. write-back    -- fold a finished run's dispositions back into the ledger.

The FTP-updatefile sync (``sync_ftp``) is the source for late-indexed stragglers;
the thin live-recent fetch lives in ``ingest_pubmed`` (which owns the live efetch)
and feeds the ledger through ``upsert_papers``.
"""

import gzip
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console

from palit import pubmed_ftp
from palit.papers import Paper, serialize_source_metadata
from palit.progress import LoggingProgress as Progress
from palit.pubmed_xml import extract_papers_from_xml
from palit.relevance import compute_relevance_majority_vote

console = Console()
app = typer.Typer(help="PubMed ingestion ledger: dedup/disposition memory across runs")

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = Path("data/pubmed_ingestion_ledger.sqlite")
LEDGER_SCHEMA_PATH = Path("ledger_schema.sql")
DEFAULT_CLOSURE_HORIZON_MONTHS = 6

# A previously-seen DOI is either settled (never reconsider) or actionable
# (re-include in the next run). The denormalized `relevant` flag makes both a
# pure indexed filter. These predicates are the single source of truth for the
# partition and are shared by seed-run-db and the dedup queries.
SETTLED_WHERE = "relevant = 0 OR download_status = 'downloaded'"
ACTIONABLE_WHERE = (
    "relevant IS NULL OR (relevant = 1 AND COALESCE(download_status, '') <> 'downloaded')"
)


# --- date helpers -----------------------------------------------------------


def subtract_months(d: date, months: int) -> date:
    """Return the date `months` calendar months before `d`, clamping the day.

    Used for the closure horizon: a CRDT month older than this is finalised and
    dropped from the actionable set. Clamps to the last valid day of the target
    month (e.g. 2026-03-31 minus 1 month -> 2026-02-28).
    """
    total = (d.year * 12 + (d.month - 1)) - months
    year, month = divmod(total, 12)
    month += 1
    # Days in the target month (day 1 of the following month minus a day).
    if month == 12:
        last_day = 31
    else:
        last_day = (date(year, month + 1, 1) - date(year, month, 1)).days
    return date(year, month, min(d.day, last_day))


# --- connection / schema ----------------------------------------------------


def connect(ledger_path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    """Open the ledger database (read-only optional)."""
    if readonly:
        return sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    return sqlite3.connect(ledger_path)


def create_ledger(ledger_path: Path) -> None:
    """Create an empty ledger database from ledger_schema.sql."""
    if not LEDGER_SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Ledger schema not found at {LEDGER_SCHEMA_PATH.absolute()}")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ledger_path)
    try:
        conn.executescript(LEDGER_SCHEMA_PATH.read_text())
    finally:
        conn.close()


def _require_ledger(ledger_path: Path) -> None:
    """Fail fast with an actionable message if the ledger has not been created."""
    if not ledger_path.exists():
        raise FileNotFoundError(
            f"Ledger not found at {ledger_path}. Create it first: "
            f"`palit ledger init --ledger {ledger_path}`."
        )


# --- disposition helpers ----------------------------------------------------


def relevant_from_assessment(assessment_json: str | None) -> int | None:
    """Derive the majority-vote `relevant` flag (0/1) from the assessment array.

    Returns None when there is no assessment, or when the array does not hold
    exactly three results (the majority vote is undefined) -- the caller then
    drops the assessment so the row stays cleanly actionable rather than stuck
    as assessed-but-relevance-unknown.
    """
    if not assessment_json:
        return None
    assessments = json.loads(assessment_json)
    if len(assessments) != 3:
        return None
    return 1 if compute_relevance_majority_vote(assessments)["relevant"] else 0


def settled_dois(ledger_path: Path) -> set[str]:
    """Return the DOIs the ledger has settled (never reconsider).

    A DOI is settled if it was assessed majority not-relevant, or already
    downloaded. ingest-preprints uses this to skip re-fetching preprints whose
    disposition is final; relevant-not-downloaded carry-overs come back through
    seed-run-db instead (it is source-agnostic).
    """
    _require_ledger(ledger_path)
    conn = connect(ledger_path, readonly=True)
    try:
        rows = conn.execute(f"SELECT doi FROM ledger WHERE {SETTLED_WHERE}")
        return {row[0] for row in rows}
    finally:
        conn.close()


# --- fetch-upsert (operation 1) ---------------------------------------------


def upsert_papers(conn: sqlite3.Connection, papers: list[Paper], today: str) -> tuple[int, int]:
    """Upsert fetched metadata into the ledger without touching disposition.

    On insert: a fresh row with NULL disposition and first_seen = last_seen =
    today. On conflict: refresh title/abstract/authors/journal/source_date/
    source_metadata, coalesce in a newly-available PMID, and bump last_seen. The
    disposition columns are deliberately left for the run write-back to own, so a
    late-arriving abstract simply updates the row (abstract-lag handled).

    Returns (inserted, updated).
    """
    if not papers:
        return 0, 0
    before = int(conn.execute("SELECT count(*) FROM ledger").fetchone()[0])
    conn.executemany(
        """
        INSERT INTO ledger
            (doi, pmid, title, abstract, authors, journal, source, source_date,
             source_metadata, first_seen, last_seen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doi) DO UPDATE SET
            pmid = COALESCE(excluded.pmid, ledger.pmid),
            title = excluded.title,
            abstract = excluded.abstract,
            authors = excluded.authors,
            journal = excluded.journal,
            source = excluded.source,
            source_date = excluded.source_date,
            source_metadata = excluded.source_metadata,
            last_seen = excluded.last_seen
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
                today,
                today,
            )
            for p in papers
        ],
    )
    conn.commit()
    after = int(conn.execute("SELECT count(*) FROM ledger").fetchone()[0])
    inserted = after - before
    return inserted, len(papers) - inserted


# --- FTP updatefile sync ----------------------------------------------------


@dataclass
class FtpSyncStats:
    """Result of an FTP updatefile sync."""

    files_applied: int
    papers_inserted: int
    papers_updated: int
    last_applied_file_number: int
    baseline_year: int


def _read_sync_state(conn: sqlite3.Connection) -> tuple[int, int] | None:
    """Return (baseline_year, last_applied_file_number) or None if uninitialised."""
    row = conn.execute(
        "SELECT baseline_year, last_applied_file_number FROM ftp_sync_state WHERE id = 1"
    ).fetchone()
    return (int(row[0]), int(row[1])) if row else None


def _write_sync_state(conn: sqlite3.Connection, baseline_year: int, last_applied: int) -> None:
    conn.execute(
        """
        INSERT INTO ftp_sync_state (id, baseline_year, last_applied_file_number, updated_at)
        VALUES (1, ?, ?, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
            baseline_year = excluded.baseline_year,
            last_applied_file_number = excluded.last_applied_file_number,
            updated_at = excluded.updated_at
        """,
        (baseline_year, last_applied),
    )
    conn.commit()


def sync_ftp(
    conn: sqlite3.Connection,
    *,
    min_crdt: str,
    work_dir: Path,
    client: httpx.Client,
    today: str,
    max_files: int | None = None,
    keep_files: bool = False,
) -> FtpSyncStats:
    """Apply new NCBI update files to the ledger, incrementally by file number.

    Initialises the sync cursor from the baseline README on first run (we record
    the baseline's top file number and never load the baseline itself). Detects
    the annual rollover (a higher YY prefix appears) and resumes from the new
    year's first post-baseline update file. Each applied file is downloaded,
    md5-verified, parsed, filtered to source_date >= min_crdt, upserted, and the
    cursor advanced + committed -- so an interrupted sync resumes cleanly.

    `<DeleteCitation>` entries are ignored: the parser only reads PubmedArticle
    elements, and a deleted relevant-not-downloaded record ages out of the
    actionable set via the closure horizon.
    """
    state = _read_sync_state(conn)
    if state is None:
        baseline = pubmed_ftp.read_baseline_max(client)
        baseline_year, last_applied = baseline.year, baseline.number
        _write_sync_state(conn, baseline_year, last_applied)
        logger.info(
            "Initialised FTP sync from baseline %s (year %d, applying updates from #%d)",
            baseline.name,
            baseline_year,
            last_applied + 1,
        )
    else:
        baseline_year, last_applied = state

    remote = pubmed_ftp.list_remote_files(client, pubmed_ftp.UPDATEFILES_DIR)
    if not remote:
        raise ValueError("No update files listed on the FTP host")

    # Annual rollover: a new baseline year has appeared. Switch to it and resume
    # from its first post-baseline update file (the old year's tail is covered by
    # the periodic backfill-diff backstop, not here).
    max_year = max(f.year for f in remote)
    if max_year > baseline_year:
        baseline = pubmed_ftp.read_baseline_max(client)
        logger.info(
            "Annual rollover detected: year %d -> %d; resuming from #%d",
            baseline_year,
            baseline.year,
            baseline.number + 1,
        )
        baseline_year, last_applied = baseline.year, baseline.number
        _write_sync_state(conn, baseline_year, last_applied)

    to_apply = sorted(f for f in remote if f.year == baseline_year and f.number > last_applied)
    if max_files is not None:
        to_apply = to_apply[:max_files]

    if not to_apply:
        logger.info("FTP sync up to date (last applied #%d, year %d)", last_applied, baseline_year)
        return FtpSyncStats(0, 0, 0, last_applied, baseline_year)

    logger.info(
        "Applying %d update file(s) #%d..#%d (CRDT floor %s)",
        len(to_apply),
        to_apply[0].number,
        to_apply[-1].number,
        min_crdt,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    total_inserted = total_updated = 0
    with Progress(console=console) as progress:
        task = progress.add_task("Applying update files...", total=len(to_apply))
        for remote_file in to_apply:
            path = pubmed_ftp.download_verified(
                client, pubmed_ftp.UPDATEFILES_DIR, remote_file, work_dir
            )
            with gzip.open(path, "rb") as f:
                xml_content = f.read()
            papers, _stats = extract_papers_from_xml(xml_content, "initial", remote_file.name)
            in_window = [p for p in papers if p.source_date >= min_crdt]
            inserted, updated = upsert_papers(conn, in_window, today)
            total_inserted += inserted
            total_updated += updated
            last_applied = remote_file.number
            _write_sync_state(conn, baseline_year, last_applied)
            if not keep_files:
                path.unlink(missing_ok=True)
            progress.advance(task)

    logger.info(
        "FTP sync applied %d files: %d new, %d refreshed (last #%d)",
        len(to_apply),
        total_inserted,
        total_updated,
        last_applied,
    )
    return FtpSyncStats(len(to_apply), total_inserted, total_updated, last_applied, baseline_year)


# --- seed run DB (operation 2) ----------------------------------------------


def seed_run_db_from_ledger(
    ledger_path: Path,
    run_db_path: Path,
    *,
    horizon_floor: str,
    end_date: str,
) -> int:
    """Copy the actionable set within [horizon_floor, end_date] into a run DB.

    New papers (relevance NULL) arrive without a disposition and get assessed;
    relevant-not-downloaded carry-overs arrive with their assessment +
    download_status so they resume directly at the download stage -- even if they
    were not in this run's fetch window. Uses INSERT OR IGNORE so rows already in
    the run database (e.g. preprints ingested first) are preserved, then
    backfills any PMID the ledger has into those pre-existing rows.
    """
    _require_ledger(ledger_path)
    conn = sqlite3.connect(run_db_path)
    try:
        conn.execute("ATTACH DATABASE ? AS ledger", (str(ledger_path),))
        before = int(conn.execute("SELECT count(*) FROM papers").fetchone()[0])
        conn.execute(
            f"""
            INSERT OR IGNORE INTO papers
                (doi, pmid, title, abstract, authors, journal, source, source_date,
                 source_metadata, source_type, source_details, download_status,
                 relevance_assessment_json)
            SELECT doi, pmid, title, abstract, authors, journal, source, source_date,
                   source_metadata, 'initial', 'ledger', download_status,
                   relevance_assessment_json
            FROM ledger.ledger
            WHERE source_date >= ? AND source_date <= ?
              AND ({ACTIONABLE_WHERE})
            """,
            (horizon_floor, end_date),
        )
        # Backfill PMIDs the ledger knows into rows already present (e.g. preprints).
        conn.execute(
            """
            UPDATE papers
            SET pmid = (SELECT l.pmid FROM ledger.ledger l WHERE l.doi = papers.doi)
            WHERE pmid IS NULL
              AND EXISTS (
                  SELECT 1 FROM ledger.ledger l
                  WHERE l.doi = papers.doi AND l.pmid IS NOT NULL
              )
            """
        )
        conn.commit()
        after = int(conn.execute("SELECT count(*) FROM papers").fetchone()[0])
        conn.execute("DETACH DATABASE ledger")
        inserted = after - before
        logger.info(
            "Seeded %d actionable papers into %s (CRDT %s..%s)",
            inserted,
            run_db_path,
            horizon_floor,
            end_date,
        )
        return inserted
    finally:
        conn.close()


# --- write-back (operation 3) -----------------------------------------------


def writeback(ledger_path: Path, run_db_path: Path, run_id: str) -> int:
    """Fold a finished run's dispositions back into the ledger.

    Upserts every run-DB paper's relevance_assessment_json + derived `relevant` +
    download_status + reported_run. Papers absent from the ledger (expansion /
    discovered-citation papers, keyed by DOI) are inserted with their metadata so
    a later PubMed fetch recognises them as already-processed. Papers already in
    the ledger keep their fetched metadata (canonical/fresher) and only have
    their disposition updated.

    Returns the number of run-DB papers written back.
    """
    _require_ledger(ledger_path)
    today = date.today().isoformat()
    run_conn = sqlite3.connect(f"file:{run_db_path}?mode=ro", uri=True)
    try:
        rows = run_conn.execute(
            """
            SELECT doi, pmid, title, abstract, authors, journal, source, source_date,
                   source_metadata, relevance_assessment_json, download_status
            FROM papers
            WHERE title IS NOT NULL
            """
        ).fetchall()
    finally:
        run_conn.close()

    ledger_conn = connect(ledger_path)
    try:
        written = 0
        for (
            doi,
            pmid,
            title,
            abstract,
            authors,
            journal,
            source,
            source_date,
            source_metadata,
            assessment_json,
            download_status,
        ) in rows:
            relevant = relevant_from_assessment(assessment_json)
            # Keep the invariant: an assessment we cannot majority-vote is dropped
            # so the ledger row stays cleanly actionable rather than stuck.
            stored_assessment = assessment_json if relevant is not None else None
            ledger_conn.execute(
                """
                INSERT INTO ledger
                    (doi, pmid, title, abstract, authors, journal, source, source_date,
                     source_metadata, relevance_assessment_json, relevant, download_status,
                     reported_run, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doi) DO UPDATE SET
                    relevance_assessment_json = excluded.relevance_assessment_json,
                    relevant = excluded.relevant,
                    download_status = excluded.download_status,
                    reported_run = excluded.reported_run,
                    last_seen = excluded.last_seen
                """,
                (
                    doi,
                    pmid,
                    title,
                    abstract,
                    authors,
                    journal,
                    source,
                    source_date,
                    source_metadata,
                    stored_assessment,
                    relevant,
                    download_status,
                    run_id,
                    today,
                    today,
                ),
            )
            written += 1
        ledger_conn.commit()
        logger.info("Wrote back %d dispositions from %s (run-id %s)", written, run_db_path, run_id)
        return written
    finally:
        ledger_conn.close()


# --- seed from existing run databases (migration) ---------------------------


@dataclass
class _Disposition:
    """Accumulated disposition for one DOI across the seeded run databases."""

    assessment: str | None
    relevant: int | None
    download_status: str | None


@dataclass
class SeedStats:
    """Result of seeding the ledger from existing run databases."""

    sources: int
    dispositions: int  # distinct DOIs written
    doi_era_rows: int
    pmid_resolved: int
    pmid_discarded: int


# Download progress, most-processed first. A DOI downloaded in any run is settled,
# so the highest-ranked status across runs wins.
_DOWNLOAD_RANK: dict[str | None, int] = {
    None: 0,
    "scheduled": 1,
    "manual_required": 2,
    "downloaded": 3,
}

# Pre-DOI-migration databases recorded full-text retrieval with separate statuses
# for the PMC vs manual paths; both mean "downloaded" in the current schema.
_DOWNLOAD_STATUS_ALIASES = {
    "pmc_downloaded": "downloaded",
    "manual_downloaded": "downloaded",
}


def _normalize_download_status(value: str | None) -> str | None:
    """Map a (possibly legacy) download_status onto the canonical ledger set."""
    if value is None:
        return None
    return _DOWNLOAD_STATUS_ALIASES.get(value, value)


def _more_processed(a: str | None, b: str | None) -> str | None:
    return a if _DOWNLOAD_RANK[a] >= _DOWNLOAD_RANK[b] else b


def _merge_disposition(
    disp: dict[str, _Disposition],
    doi: str,
    assessment_json: str | None,
    download_status: str | None,
) -> None:
    """Fold one source row's disposition into the per-DOI accumulator.

    Sources are processed oldest -> newest, so a later assessment overwrites an
    earlier one (latest month wins); download_status takes the most-processed
    value across all runs.
    """
    relevant = relevant_from_assessment(assessment_json)
    stored = assessment_json if relevant is not None else None
    download_status = _normalize_download_status(download_status)
    cur = disp.get(doi)
    if cur is None:
        disp[doi] = _Disposition(stored, relevant, download_status)
        return
    if relevant is not None:
        cur.assessment = stored
        cur.relevant = relevant
    cur.download_status = _more_processed(cur.download_status, download_status)


def seed_from_run_dbs(ledger_path: Path, source_dbs: list[Path], *, min_crdt: str) -> SeedStats:
    """Seed the ledger from existing run databases (chronological order).

    Source databases must be ordered oldest -> newest so the latest month's
    relevance verdict wins on conflict. DOI-era databases (current schema) are
    seeded by DOI with their metadata; PMID-era databases (pre-DOI-migration)
    carry only their dispositions, resolved to a DOI via the already-synced
    ledger's PMID index -- a PMID with no in-window DOI is discarded. Only papers
    within the closure horizon (date >= min_crdt) are seeded.

    Run after `ledger init` + `ledger sync` so the PMID index is populated; this
    is a one-time bootstrap, intended to run before any write-back.
    """
    _require_ledger(ledger_path)
    today = date.today().isoformat()
    conn = connect(ledger_path)
    try:
        pmid_to_doi: dict[int, str] = {
            int(pmid): doi
            for doi, pmid in conn.execute("SELECT doi, pmid FROM ledger WHERE pmid IS NOT NULL")
        }
        if not pmid_to_doi:
            logger.warning(
                "Ledger has no PMID index -- run `palit ledger sync` first, or PMID-era "
                "dispositions will all be discarded."
            )

        disp: dict[str, _Disposition] = {}
        meta: dict[str, tuple[Any, ...]] = {}
        doi_era_rows = pmid_resolved = pmid_discarded = 0

        for db in source_dbs:
            src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                cols = {r[1] for r in src.execute("PRAGMA table_info(papers)")}
                if "doi" in cols:
                    for row in src.execute(
                        "SELECT doi, pmid, title, abstract, authors, journal, source, "
                        "source_date, source_metadata, relevance_assessment_json, download_status "
                        "FROM papers WHERE doi IS NOT NULL AND source_date >= ?",
                        (min_crdt,),
                    ):
                        doi, aj, ds = row[0], row[9], row[10]
                        _merge_disposition(disp, doi, aj, ds)
                        meta[doi] = row[1:9]  # pmid..source_metadata
                        doi_era_rows += 1
                else:
                    for pmid, aj, ds in src.execute(
                        "SELECT pmid, relevance_assessment_json, download_status "
                        "FROM papers WHERE pmid IS NOT NULL AND entrez_date >= ?",
                        (min_crdt,),
                    ):
                        resolved = pmid_to_doi.get(int(pmid))
                        if resolved is None:
                            pmid_discarded += 1
                            continue
                        _merge_disposition(disp, resolved, aj, ds)
                        pmid_resolved += 1
            finally:
                src.close()
            logger.info("Read dispositions from %s", db.name)

        # DOI-era rows carry metadata: insert if absent, else refresh disposition
        # only (the synced metadata is canonical/fresher than the source DB's).
        for doi, m in meta.items():
            d = disp[doi]
            conn.execute(
                """
                INSERT INTO ledger
                    (doi, pmid, title, abstract, authors, journal, source, source_date,
                     source_metadata, relevance_assessment_json, relevant, download_status,
                     first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doi) DO UPDATE SET
                    relevance_assessment_json = excluded.relevance_assessment_json,
                    relevant = excluded.relevant,
                    download_status = excluded.download_status,
                    last_seen = excluded.last_seen
                """,
                (doi, *m, d.assessment, d.relevant, d.download_status, today, today),
            )
        # PMID-resolved DOIs already exist in the ledger (resolved from it); update
        # disposition only.
        for doi, d in disp.items():
            if doi in meta:
                continue
            conn.execute(
                "UPDATE ledger SET relevance_assessment_json = ?, relevant = ?, "
                "download_status = ?, last_seen = ? WHERE doi = ?",
                (d.assessment, d.relevant, d.download_status, today, doi),
            )
        conn.commit()
        logger.info(
            "Seed complete: %d DOIs (%d DOI-era rows, %d PMID-resolved, %d PMID discarded)",
            len(disp),
            doi_era_rows,
            pmid_resolved,
            pmid_discarded,
        )
        return SeedStats(len(source_dbs), len(disp), doi_era_rows, pmid_resolved, pmid_discarded)
    finally:
        conn.close()


# --- CLI --------------------------------------------------------------------


@app.command()
def init(
    ledger: Path = typer.Option(DEFAULT_LEDGER_PATH, "--ledger", help="Ledger database path"),
) -> None:
    """Create an empty ledger database from ledger_schema.sql."""
    if ledger.exists():
        console.print(f"[yellow]Ledger already exists at {ledger}[/yellow]")
        raise typer.Exit(1)
    create_ledger(ledger)
    console.print(f"[green]✓[/green] Created ledger at {ledger}")


@app.command()
def sync(
    ledger: Path = typer.Option(DEFAULT_LEDGER_PATH, "--ledger", help="Ledger database path"),
    min_crdt: str = typer.Option(
        None,
        "--min-crdt",
        help="CRDT floor (YYYY-MM-DD) for records to keep. Default: today minus the "
        "closure horizon.",
    ),
    closure_horizon_months: int = typer.Option(
        DEFAULT_CLOSURE_HORIZON_MONTHS, "--closure-horizon-months", help="Closure horizon"
    ),
    work_dir: Path = typer.Option(
        Path("data/pubmed_updatefiles"), "--work-dir", help="Scratch dir for downloaded files"
    ),
    max_files: int = typer.Option(
        None, "--max-files", help="Apply at most this many update files (for testing/catch-up)"
    ),
    keep_files: bool = typer.Option(
        False, "--keep-files", help="Keep downloaded update files instead of deleting them"
    ),
) -> None:
    """Apply new NCBI update files to the ledger (incremental by file number)."""
    _require_ledger(ledger)
    floor = min_crdt or subtract_months(date.today(), closure_horizon_months).isoformat()
    conn = connect(ledger)
    client = pubmed_ftp.make_client()
    try:
        stats = sync_ftp(
            conn,
            min_crdt=floor,
            work_dir=work_dir,
            client=client,
            today=date.today().isoformat(),
            max_files=max_files,
            keep_files=keep_files,
        )
    finally:
        client.close()
        conn.close()
    console.print(
        f"[green]✓[/green] Applied {stats.files_applied} file(s): "
        f"{stats.papers_inserted} new, {stats.papers_updated} refreshed "
        f"(cursor: year {stats.baseline_year}, #{stats.last_applied_file_number})"
    )


@app.command()
def seed(
    source_dbs: list[Path] = typer.Option(
        ..., "--db", help="Source run database (repeatable; pass oldest -> newest)"
    ),
    ledger: Path = typer.Option(DEFAULT_LEDGER_PATH, "--ledger", help="Ledger database path"),
    min_crdt: str = typer.Option(
        None, "--min-crdt", help="CRDT floor (YYYY-MM-DD). Default: today minus the closure horizon"
    ),
    closure_horizon_months: int = typer.Option(
        DEFAULT_CLOSURE_HORIZON_MONTHS, "--closure-horizon-months", help="Closure horizon"
    ),
) -> None:
    """Seed the ledger from existing run databases (run after init + sync)."""
    _require_ledger(ledger)
    for db in source_dbs:
        if not db.exists():
            console.print(f"[red]Source DB not found: {db}[/red]")
            raise typer.Exit(1)
    floor = min_crdt or subtract_months(date.today(), closure_horizon_months).isoformat()
    stats = seed_from_run_dbs(ledger, source_dbs, min_crdt=floor)
    console.print(
        f"[green]✓[/green] Seeded {stats.dispositions} DOIs from {stats.sources} databases "
        f"({stats.doi_era_rows} DOI-era rows, {stats.pmid_resolved} PMID-resolved, "
        f"{stats.pmid_discarded} PMID discarded; CRDT floor {floor})"
    )


@app.command(name="seed-run-db")
def seed_run_db_command(
    run_db: Path = typer.Argument(..., help="Run database to seed (must already exist)"),
    end_date: str = typer.Argument(..., help="Run window end date (YYYY-MM-DD)"),
    ledger: Path = typer.Option(DEFAULT_LEDGER_PATH, "--ledger", help="Ledger database path"),
    closure_horizon_months: int = typer.Option(
        DEFAULT_CLOSURE_HORIZON_MONTHS, "--closure-horizon-months", help="Closure horizon"
    ),
) -> None:
    """Copy the actionable set within the closure horizon into a run database."""
    _require_ledger(ledger)
    if not run_db.exists():
        console.print(f"[red]Run DB not found: {run_db}[/red]")
        raise typer.Exit(1)
    floor = subtract_months(date.fromisoformat(end_date), closure_horizon_months).isoformat()
    n = seed_run_db_from_ledger(ledger, run_db, horizon_floor=floor, end_date=end_date)
    console.print(f"[green]✓[/green] Seeded {n} actionable papers into {run_db}")


@app.command(name="writeback")
def writeback_command(
    db_path: Path = typer.Option(..., "--db-path", help="Finished run database"),
    run_id: str = typer.Option(..., "--run-id", help="Run/report id recorded for traceability"),
    ledger: Path = typer.Option(DEFAULT_LEDGER_PATH, "--ledger", help="Ledger database path"),
) -> None:
    """Fold a finished run's dispositions back into the ledger."""
    _require_ledger(ledger)
    if not db_path.exists():
        console.print(f"[red]Run DB not found: {db_path}[/red]")
        raise typer.Exit(1)
    n = writeback(ledger, db_path, run_id)
    console.print(f"[green]✓[/green] Wrote back {n} dispositions from {db_path}")


if __name__ == "__main__":
    app()
