"""Tests for seeding PanelApp's cited publications into the run corpus, and for
taking a snapshot's citations into the baseline."""

import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from palit import panelapp_publications
from palit.panelapp_client import PanelGeneData
from palit.panelapp_publications import (
    CITATION_SOURCE_DETAILS_PREFIX,
    CITATION_SOURCE_TYPE,
    SEED_SOURCE_TYPE,
    collect_missing_publications,
    held_identifiers,
    ingest_snapshot_citations,
    snapshot_publications,
    store_publications,
)
from palit.papers import CrossrefMetadata, Paper, PubmedMetadata

MENDELIOME = 137
SKELETAL = 258


def _make_db(tmp_path: Path, papers: list[tuple[str, int | None]], genes: list[int]) -> Path:
    """Build a minimal run database with the columns the seeder reads."""
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE papers (doi TEXT PRIMARY KEY, pmid INTEGER);
            CREATE TABLE gene_mentions (hgnc_id INTEGER, paper_doi TEXT, source TEXT);
            """
        )
        conn.executemany("INSERT INTO papers (doi, pmid) VALUES (?, ?)", papers)
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_doi, source) VALUES (?, ?, 'recent_evidence')",
            [(hgnc_id, "10.1000/seed") for hgnc_id in genes],
        )
    return db_path


class _FakeClient:
    """PanelApp client stub returning fixed entity and review publications."""

    def __init__(
        self,
        entity_publications: dict[int, list[str]],
        review_publications: dict[int, list[str]] | None = None,
    ) -> None:
        self._entity_publications = entity_publications
        self._review_publications = review_publications or {}

    def get_panel_data(self, panel_id: int) -> dict[str, Any]:
        return {
            "genes": [
                {
                    "gene_data": {"hgnc_id": f"HGNC:{hgnc_id}"},
                    "publications": publications,
                }
                for hgnc_id, publications in self._entity_publications.items()
            ]
        }

    def holds_gene(self, panel_id: int, hgnc_id: int) -> bool:
        return hgnc_id in self._entity_publications

    def gene_reviews(self, panel_id: int, hgnc_id: int) -> list[dict[str, Any]]:
        publications = self._review_publications.get(hgnc_id)
        return [{"publications": publications}] if publications else []


def _panel_data(gene_panels: dict[int, set[int]], panel_ids: list[int]) -> PanelGeneData:
    return PanelGeneData(
        panel_ids=panel_ids,
        gene_panel_confidence={
            hgnc_id: dict.fromkeys(panels, 3) for hgnc_id, panels in gene_panels.items()
        },
    )


def test_collects_publications_absent_from_the_corpus(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path, [("10.1000/held", 11111111)], [3604])
    client = _FakeClient({3604: ["11111111", "22222222"]})
    panel_data = _panel_data({3604: {MENDELIOME}}, [MENDELIOME])

    missing = collect_missing_publications(held_identifiers(db_path), client, panel_data, [3604])

    assert missing.pmid_owner == {22222222: 3604}
    assert missing.publications_cited == 2
    assert missing.already_present == 1
    assert missing.genes_on_target_panel == 1


def test_unions_entity_and_review_publications(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path, [], [3604])
    client = _FakeClient({3604: ["11111111"]}, review_publications={3604: ["22222222"]})
    panel_data = _panel_data({3604: {MENDELIOME}}, [MENDELIOME])

    missing = collect_missing_publications(held_identifiers(db_path), client, panel_data, [3604])

    assert missing.pmid_owner == {11111111: 3604, 22222222: 3604}


def test_attributes_a_shared_publication_to_the_lowest_hgnc_id(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path, [], [3604, 10260])
    client = _FakeClient({3604: ["19999999"], 10260: ["19999999"]})
    panel_data = _panel_data({3604: {MENDELIOME}, 10260: {MENDELIOME}}, [MENDELIOME])

    missing = collect_missing_publications(
        held_identifiers(db_path), client, panel_data, [3604, 10260]
    )

    assert missing.pmid_owner == {19999999: 3604}


def test_skips_genes_absent_from_every_target_panel(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path, [], [3604])
    client = _FakeClient({3604: ["11111111"]})
    # Gene sits only on a panel outside the target set.
    panel_data = _panel_data({3604: {SKELETAL}}, [MENDELIOME])

    missing = collect_missing_publications(held_identifiers(db_path), client, panel_data, [3604])

    assert missing.pmid_owner == {}
    assert missing.genes_on_target_panel == 0


def test_matches_held_dois_case_insensitively(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path, [("10.1000/mixedcase", None)], [3604])
    client = _FakeClient({3604: ["10.1000/MixedCase", "10.1000/other"]})
    panel_data = _panel_data({3604: {MENDELIOME}}, [MENDELIOME])

    missing = collect_missing_publications(held_identifiers(db_path), client, panel_data, [3604])

    assert missing.doi_owner == {"10.1000/other": 3604}
    assert missing.already_present == 1


def test_keeps_the_first_cited_spelling_of_a_doi(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path, [], [3604, 10260])
    client = _FakeClient({3604: ["10.1000/abc"], 10260: ["10.1000/AbC"]})
    panel_data = _panel_data({3604: {MENDELIOME}, 10260: {MENDELIOME}}, [MENDELIOME])

    missing = collect_missing_publications(
        held_identifiers(db_path), client, panel_data, [3604, 10260]
    )

    assert missing.doi_owner == {"10.1000/abc": 3604}


PANEL_DATE = "2026-10-15"


def _entry(publications: list[str], reviews: list[list[str] | None] | None) -> dict[str, Any]:
    """A snapshot entity; *reviews* None for an entry without reviews, as an STR has."""
    entry: dict[str, Any] = {"publications": publications}
    if reviews is not None:
        entry["evaluations"] = [{"publications": review} for review in reviews]
    return entry


def _snapshot_panel(
    genes: list[dict[str, Any]],
    strs: list[dict[str, Any]] | None = None,
    regions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {"genes": genes, "strs": strs or [], "regions": regions or []}


def test_snapshot_publications_takes_entries_reviews_and_strs_on_every_panel() -> None:
    panels = {
        MENDELIOME: _snapshot_panel(
            [_entry(["11111111", "10.1000/Entity"], reviews=[["22222222"], None])],
            strs=[_entry(["33333333"], reviews=None)],
        ),
        SKELETAL: _snapshot_panel(
            # Reviews PanelApp could not serve are null in the snapshot.
            [{"publications": ["44444444"], "evaluations": None}],
            regions=[_entry(["15555555"], reviews=None)],
        ),
    }

    cited = snapshot_publications(panels)

    assert cited.pmids == {11111111, 22222222, 33333333, 44444444}
    assert cited.dois == {"10.1000/Entity"}


def _make_baseline(tmp_path: Path, papers: list[tuple[str, int | None]]) -> Path:
    """A database with the papers columns the intake reads and writes."""
    db_path = tmp_path / "baseline.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE papers (
                doi TEXT PRIMARY KEY, pmid INTEGER, title TEXT NOT NULL, abstract TEXT,
                authors TEXT, journal TEXT, source TEXT NOT NULL, source_date DATE,
                source_metadata JSON, source_type TEXT, source_details TEXT,
                download_status TEXT
            );
            CREATE TABLE gene_mentions (hgnc_id INTEGER, paper_doi TEXT, source TEXT);
            """
        )
        conn.executemany(
            "INSERT INTO papers (doi, pmid, title, source) VALUES (?, ?, 'Held', 'pubmed')",
            papers,
        )
    return db_path


def _paper(doi: str, pmid: int | None) -> Paper:
    return Paper(
        doi=doi,
        pmid=pmid,
        title=f"Paper {doi}",
        abstract="",
        authors="",
        journal="",
        source="pubmed" if pmid is not None else "crossref",
        source_date="2001-01-01",
        source_metadata=PubmedMetadata() if pmid is not None else CrossrefMetadata(),
        source_type="expansion",
        source_details="placeholder",
    )


class _FakeSources:
    """PubMed and CrossRef stand-ins resolving fixed identifiers, recording requests."""

    def __init__(self, pmid_dois: dict[int, str], resolvable_dois: set[str]) -> None:
        self._pmid_dois = pmid_dois
        self._resolvable_dois = resolvable_dois
        self.pmid_requests: list[int] = []
        self.doi_requests: list[str] = []

    def fetch_papers_by_pmids(self, pmids: Sequence[int], source_details: str) -> list[Paper]:
        self.pmid_requests.extend(pmids)
        return [_paper(self._pmid_dois[p], p) for p in pmids if p in self._pmid_dois]

    def fetch_paper_metadata_by_doi(self, doi: str, source_details: str) -> Paper | None:
        # CrossRef resolves a DOI in any letter case and the paper keeps the DOI as asked.
        self.doi_requests.append(doi)
        resolvable = {d.lower() for d in self._resolvable_dois}
        return _paper(doi, None) if doi.lower() in resolvable else None


@pytest.fixture
def sources(monkeypatch: pytest.MonkeyPatch) -> _FakeSources:
    fake = _FakeSources(
        pmid_dois={22222222: "10.1000/pmid222", 33333333: "10.1000/pmid333"},
        resolvable_dois={"10.1000/new"},
    )
    monkeypatch.setattr(panelapp_publications, "fetch_papers_by_pmids", fake.fetch_papers_by_pmids)
    monkeypatch.setattr(
        panelapp_publications, "fetch_paper_metadata_by_doi", fake.fetch_paper_metadata_by_doi
    )
    return fake


def test_ingest_adds_only_publications_the_database_lacks(
    tmp_path: Path, sources: _FakeSources
) -> None:
    db_path = _make_baseline(tmp_path, [("10.1000/held-by-pmid", 11111111), ("10.1000/held", None)])
    panels = {
        MENDELIOME: _snapshot_panel(
            [_entry(["11111111", "22222222", "10.1000/HELD"], reviews=[["10.1000/new"]])]
        ),
        SKELETAL: _snapshot_panel([], strs=[_entry(["22222222"], reviews=None)]),
    }

    stats = ingest_snapshot_citations(db_path, panels, PANEL_DATE)

    # PMID 22222222 is cited on both panels and fetched once; held identifiers never are.
    assert sources.pmid_requests == [22222222]
    assert sources.doi_requests == ["10.1000/new"]
    assert (stats.publications_cited, stats.already_present, stats.added) == (4, 2, 2)
    with sqlite3.connect(db_path) as conn:
        added = conn.execute(
            "SELECT doi, pmid, source_type, source_details, download_status FROM papers "
            "WHERE source_details IS NOT NULL ORDER BY doi"
        ).fetchall()
        mentions = conn.execute("SELECT COUNT(*) FROM gene_mentions").fetchone()[0]
    source_details = f"{CITATION_SOURCE_DETAILS_PREFIX}:{PANEL_DATE}"
    assert source_details == "panelapp-snapshot:2026-10-15"
    assert added == [
        ("10.1000/new", None, CITATION_SOURCE_TYPE, source_details, "scheduled"),
        ("10.1000/pmid222", 22222222, CITATION_SOURCE_TYPE, source_details, "scheduled"),
    ]
    assert mentions == 0


def test_ingest_is_idempotent(tmp_path: Path, sources: _FakeSources) -> None:
    db_path = _make_baseline(tmp_path, [])
    panels = {MENDELIOME: _snapshot_panel([_entry(["22222222", "10.1000/new"], reviews=[])])}

    ingest_snapshot_citations(db_path, panels, PANEL_DATE)
    sources.pmid_requests.clear()
    sources.doi_requests.clear()
    rerun = ingest_snapshot_citations(db_path, panels, PANEL_DATE)

    assert (rerun.publications_cited, rerun.already_present, rerun.added) == (2, 2, 0)
    assert sources.pmid_requests == []
    assert sources.doi_requests == []


def test_ingest_dedupes_dois_case_insensitively(tmp_path: Path, sources: _FakeSources) -> None:
    db_path = _make_baseline(tmp_path, [])
    panels = {
        MENDELIOME: _snapshot_panel([_entry(["10.1000/NEW"], reviews=[])]),
        SKELETAL: _snapshot_panel([_entry(["10.1000/new"], reviews=[])]),
    }

    stats = ingest_snapshot_citations(db_path, panels, PANEL_DATE)

    # One spelling is fetched and stored: the first in sort order.
    assert sources.doi_requests == ["10.1000/NEW"]
    assert (stats.publications_cited, stats.added) == (1, 1)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT doi FROM papers").fetchall() == [("10.1000/NEW",)]


def test_ingest_stores_a_doi_as_panelapp_cites_it(tmp_path: Path, sources: _FakeSources) -> None:
    db_path = _make_baseline(tmp_path, [])
    panels = {MENDELIOME: _snapshot_panel([_entry(["https://doi.org/10.1000/New"], [])])}

    ingest_snapshot_citations(db_path, panels, PANEL_DATE)

    assert sources.doi_requests == ["10.1000/New"]
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT doi FROM papers").fetchall() == [("10.1000/New",)]


def test_ingest_stores_a_paper_cited_by_pmid_and_doi_once(
    tmp_path: Path, sources: _FakeSources
) -> None:
    db_path = _make_baseline(tmp_path, [])
    # PMID 33333333 resolves to 10.1000/pmid333, which another entry cites by DOI.
    panels = {MENDELIOME: _snapshot_panel([_entry(["33333333", "10.1000/PMID333"], reviews=[])])}

    stats = ingest_snapshot_citations(db_path, panels, PANEL_DATE)

    assert sources.doi_requests == []
    assert (stats.publications_cited, stats.added, stats.already_held) == (2, 1, 1)


def test_ingest_counts_unresolved_identifiers_and_lists_them_at_debug(
    tmp_path: Path, sources: _FakeSources, caplog: pytest.LogCaptureFixture
) -> None:
    db_path = _make_baseline(tmp_path, [])
    panels = {MENDELIOME: _snapshot_panel([_entry(["99999", "10.1000/gone"], reviews=[])])}

    with caplog.at_level(logging.DEBUG, logger=panelapp_publications.__name__):
        stats = ingest_snapshot_citations(db_path, panels, PANEL_DATE)

    assert (stats.added, stats.unresolved) == (0, 2)
    # An unresolved identifier is routine, not a failure: counted at INFO, named at DEBUG.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    debug = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("99999" in message for message in debug)
    assert any("10.1000/gone" in message for message in debug)


def test_ingest_matches_a_resolved_doi_held_in_another_case(
    tmp_path: Path, sources: _FakeSources
) -> None:
    # PMID 22222222 resolves to 10.1000/pmid222, which the database holds as 10.1000/PMID222.
    db_path = _make_baseline(tmp_path, [("10.1000/PMID222", None)])
    panels = {MENDELIOME: _snapshot_panel([_entry(["22222222"], reviews=[])])}

    stats = ingest_snapshot_citations(db_path, panels, PANEL_DATE)

    assert sources.pmid_requests == [22222222]
    assert (stats.added, stats.already_held, stats.unresolved) == (0, 1, 0)
    # The held paper lacked a PMID and now has it, so a rerun does not fetch it again.
    assert stats.pmids_linked == 1
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT doi, pmid FROM papers").fetchall() == [
            ("10.1000/PMID222", 22222222)
        ]
    sources.pmid_requests.clear()
    assert ingest_snapshot_citations(db_path, panels, PANEL_DATE).already_present == 1
    assert sources.pmid_requests == []


def test_ingest_leaves_the_pmid_of_a_held_paper_alone(
    tmp_path: Path, sources: _FakeSources
) -> None:
    # The held row has a PMID of its own (PubMed merged records, say); it is kept.
    db_path = _make_baseline(tmp_path, [("10.1000/pmid222", 12345678)])
    panels = {MENDELIOME: _snapshot_panel([_entry(["22222222"], reviews=[])])}

    stats = ingest_snapshot_citations(db_path, panels, PANEL_DATE)

    assert (stats.already_held, stats.pmids_linked) == (1, 0)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT pmid FROM papers").fetchall() == [(12345678,)]


def test_store_publications_stamps_seeding_provenance(
    tmp_path: Path, sources: _FakeSources
) -> None:
    db_path = _make_baseline(tmp_path, [])

    outcome = store_publications(
        db_path,
        held_identifiers(db_path).dois,
        {22222222: "panelapp:3604"},
        {"10.1000/new": "panelapp:10260"},
        SEED_SOURCE_TYPE,
    )

    assert (outcome.added, outcome.already_held, outcome.unresolved) == (2, 0, 0)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT doi, source_type, source_details FROM papers ORDER BY doi"
        ).fetchall()
    assert rows == [
        ("10.1000/new", "expansion", "panelapp:10260"),
        ("10.1000/pmid222", "expansion", "panelapp:3604"),
    ]
