-- PubMed ingestion ledger schema.
--
-- Standalone database (default: data/pubmed_ingestion_ledger.sqlite) recording
-- the terminal disposition of every PubMed paper across all runs, plus the
-- refreshed bibliographic metadata needed to re-seed a run database. This is the
-- dedup/disposition memory that replaces the per-run buffer window and the
-- single --previous-db set-difference: papers already settled (not relevant, or
-- downloaded) are skipped, while actionable papers (never assessed, or relevant
-- but not yet downloaded) are re-emitted into each new run.
--
-- This is intentionally a different schema from the per-run schema.sql; run
-- databases never carry the ledger table.

-- WAL mode for concurrent read/write, matching the run databases.
PRAGMA journal_mode=WAL;

CREATE TABLE ledger (
    doi TEXT PRIMARY KEY,
    pmid INTEGER,

    -- Bibliographic metadata, refreshed on every fetch (live efetch or FTP
    -- updatefile). A late-attached abstract or a backfilled PMID simply updates
    -- the row; these columns are sufficient to seed a run database and retry a
    -- download without re-fetching from PubMed.
    title TEXT NOT NULL,
    abstract TEXT,
    authors TEXT,
    journal TEXT,
    source TEXT NOT NULL,             -- 'pubmed' for fetched records; other sources
                                      -- only enter via run write-back of expansion /
                                      -- discovered-citation papers (keyed by DOI)
    source_date DATE,                 -- entrez date (CRDT)
    source_metadata JSON,             -- carries PMCID for the attempt-pmc download path

    -- Disposition, owned by the run and folded back at run end. Untouched by the
    -- fetch-upsert step.
    relevance_assessment_json JSON,   -- array of 3 parsed assessments; NULL = never assessed
    relevant INTEGER,                 -- derived majority vote: NULL = unassessed, else 0/1
    download_status TEXT CHECK(download_status IN
                              ('scheduled', 'manual_required', 'downloaded')),
    reported_run TEXT,                -- run/report id that consumed it (traceability)

    -- Bookkeeping.
    first_seen DATE NOT NULL,         -- first fetch that introduced this DOI
    last_seen DATE NOT NULL           -- most recent fetch that contained it
);

-- The denormalized `relevant` flag makes settled/actionable a pure indexed
-- filter:
--   settled (never reconsider):  relevant = 0 OR download_status = 'downloaded'
--   actionable (re-include):     relevant IS NULL
--                                OR (relevant = 1 AND COALESCE(download_status,'') <> 'downloaded')
CREATE INDEX idx_ledger_pmid ON ledger(pmid) WHERE pmid IS NOT NULL;
CREATE INDEX idx_ledger_relevant ON ledger(relevant);
-- The actionable query bounds by source_date (the closure horizon), so index it.
CREATE INDEX idx_ledger_source_date ON ledger(source_date);

-- Incremental FTP-updatefile sync cursor.
--
-- PubMed publishes an annual baseline (pubmed{YY}n0001..N) each December, then
-- daily update files numbered above the baseline range. The incremental key is
-- the file number: we apply every update file numbered higher than the last we
-- applied. Annual rollover restarts the numbering under a new YY prefix; on
-- rollover we switch baseline_year and resume from the first post-baseline
-- update file (we never load the full new baseline -- update files alone carry
-- late-indexed records across the year boundary).
--
-- Single row (id = 1), present once the ledger has been initialised against a
-- baseline.
CREATE TABLE ftp_sync_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    baseline_year INTEGER NOT NULL,           -- e.g. 26 for pubmed26nXXXX
    last_applied_file_number INTEGER NOT NULL,
    updated_at TEXT NOT NULL                  -- ISO-8601 timestamp of the last sync
);
