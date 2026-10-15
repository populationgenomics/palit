"""Tests for seeding PanelApp's cited publications into the run corpus, and for
taking a snapshot's citations into the baseline."""

import json
import logging
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from palit import panelapp_publications
from palit.discover_citations import PubmedFetchError
from palit.hgnc import HgncResolver
from palit.panelapp_client import PanelPublications
from palit.panelapp_publications import (
    CITATION_SOURCE_DETAILS_PREFIX,
    CITATION_SOURCE_TYPE,
    SEED_SOURCE_TYPE,
    HeldIdentifiers,
    PaperWriter,
    SeedStats,
    collect_missing_publications,
    fetch_publications,
    gene_citations,
    held_identifiers,
    ingest_snapshot_citations,
    seed_panelapp_publications,
    snapshot_publications,
)
from palit.papers import CrossrefMetadata, Paper, PubmedMetadata

ROOT = Path(__file__).resolve().parents[1]

MENDELIOME = 137
SKELETAL = 258
OTHER_PANEL = 9001


def _held(pmids: set[int], dois: list[str]) -> HeldIdentifiers:
    return HeldIdentifiers(pmids=pmids, dois={doi.lower(): doi for doi in dois})


def _cited(pmids: set[int] | None = None, dois: set[str] | None = None) -> PanelPublications:
    return PanelPublications(pmids=pmids or set(), dois=dois or set())


def test_collects_publications_absent_from_the_database() -> None:
    missing = collect_missing_publications(
        _held({11111111}, []), {3604: _cited({11111111, 22222222})}
    )

    assert missing.pmid_owner == {22222222: 3604}
    assert missing.publications_cited == 2
    assert missing.already_present == 1


def test_attributes_a_shared_publication_to_the_lowest_hgnc_id() -> None:
    missing = collect_missing_publications(
        _held(set(), []), {10260: _cited({19999999}), 3604: _cited({19999999})}
    )

    assert missing.pmid_owner == {19999999: 3604}


def test_matches_held_dois_case_insensitively() -> None:
    missing = collect_missing_publications(
        _held(set(), ["10.1000/mixedcase"]),
        {3604: _cited(dois={"10.1000/MixedCase", "10.1000/other"})},
    )

    assert missing.doi_owner == {"10.1000/other": 3604}
    assert missing.already_present == 1


def test_keeps_the_first_cited_spelling_of_a_doi() -> None:
    missing = collect_missing_publications(
        _held(set(), []),
        {3604: _cited(dois={"10.1000/abc"}), 10260: _cited(dois={"10.1000/AbC"})},
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


def test_fetch_publications_stamps_seeding_provenance(
    tmp_path: Path, sources: _FakeSources
) -> None:
    db_path = _make_baseline(tmp_path, [])
    writer = PaperWriter(db_path, held_identifiers(db_path).dois, SEED_SOURCE_TYPE)

    unresolved = fetch_publications(
        writer, {22222222: "panelapp:3604"}, {"10.1000/new": "panelapp:10260"}
    )

    assert (writer.added, writer.already_held, unresolved) == (2, 0, 0)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT doi, source_type, source_details FROM papers ORDER BY doi"
        ).fetchall()
    assert rows == [
        ("10.1000/new", "expansion", "panelapp:10260"),
        ("10.1000/pmid222", "expansion", "panelapp:3604"),
    ]


def _gene_entry(
    hgnc_id: int, publications: list[str], reviews: Sequence[list[str] | None] = ()
) -> dict[str, Any]:
    return {
        "gene_data": {"hgnc_id": f"HGNC:{hgnc_id}"},
        "publications": publications,
        "evaluations": [{"publications": review} for review in reviews],
    }


def _str_entry(hgnc_id: int, publications: list[str]) -> dict[str, Any]:
    return {"gene_data": {"hgnc_id": f"HGNC:{hgnc_id}"}, "publications": publications}


def test_gene_citations_unite_entries_and_reviews_on_every_panel() -> None:
    panels = {
        MENDELIOME: _snapshot_panel([_gene_entry(1, ["11111111"])]),
        # Cited for gene 1 only in a review on another panel.
        SKELETAL: _snapshot_panel([_gene_entry(1, [], reviews=[["10.1000/Review"], None])]),
        OTHER_PANEL: _snapshot_panel(
            [_gene_entry(2, ["44444444"]), _gene_entry(5, [])],
            strs=[_str_entry(1, ["33333333"])],
        ),
    }

    cited = gene_citations(panels, [1, 5, 6])

    assert cited == {
        1: PanelPublications(pmids={11111111, 33333333}, dois={"10.1000/Review"}),
        5: PanelPublications(pmids=set(), dois=set()),
    }


def _relevance(relevant: bool) -> str:
    return json.dumps({"relevant": relevant, "screen": {"relevant": relevant}})


def _make_run_db(tmp_path: Path) -> Path:
    """Run genes 1 and 3, from relevant initial papers. Two initial papers the screen
    rejected: one PanelApp cites for gene 1 (by PMID), one for gene 4, which is not a
    run gene."""
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.executemany(
            "INSERT INTO papers (doi, pmid, title, source, source_type, download_status, "
            "relevance_assessment_json) VALUES (?, ?, 'Held', 'pubmed', 'initial', ?, ?)",
            [
                ("10.1/relevant-a", None, "downloaded", _relevance(True)),
                ("10.1/relevant-b", None, "downloaded", _relevance(True)),
                ("10.1/rejected", 30000001, None, _relevance(False)),
                ("10.1/rejected-other", None, None, _relevance(False)),
            ],
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (?, 'G', ?, 'recent_evidence')",
            [(1, "10.1/relevant-a"), (3, "10.1/relevant-b")],
        )
    return db_path


def _make_seed_baseline(tmp_path: Path) -> Path:
    """A baseline holding one cited paper by DOI (in another letter case) and one by PMID."""
    db_path = tmp_path / "baseline.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.executemany(
            "INSERT INTO papers (doi, pmid, title, abstract, authors, journal, source, "
            "source_date, source_metadata, source_type, source_details) "
            "VALUES (?, ?, ?, 'Abstract', 'Doe, Jane', 'J Genet', ?, '2009-05-01', ?, ?, ?)",
            [
                (
                    "10.1000/in-baseline",
                    None,
                    "Baseline paper by DOI",
                    "crossref",
                    "{}",
                    CITATION_SOURCE_TYPE,
                    "panelapp-snapshot:2026-10-13",
                ),
                (
                    "10.1000/by-pmid",
                    40000001,
                    "Baseline paper by PMID",
                    "pubmed",
                    '{"pmcid": "PMC1"}',
                    "baseline_screening",
                    "pubmed25n0001.xml.gz",
                ),
            ],
        )
    return db_path


def _seed_panels(gene_1_cites_rejected: bool = True) -> dict[int, dict[str, Any]]:
    """Gene 1 cites the rejected paper on the Mendeliome, a baseline DOI in a review on
    another panel, and PMID 22222222 (in neither database) in a third panel's review;
    gene 3 cites PMID 22222222 and, as a repeat expansion, a baseline PMID; gene 4
    cites the other rejected paper."""
    return {
        MENDELIOME: _snapshot_panel(
            [_gene_entry(1, ["PMID: 30000001"] if gene_1_cites_rejected else [])]
        ),
        SKELETAL: _snapshot_panel([_gene_entry(1, [], reviews=[["10.1000/In-Baseline"]])]),
        OTHER_PANEL: _snapshot_panel(
            [
                _gene_entry(1, [], reviews=[["22222222"]]),
                _gene_entry(3, ["22222222"]),
                _gene_entry(4, ["10.1/rejected-other"]),
            ],
            strs=[_str_entry(3, ["40000001"])],
        ),
    }


def _seed(
    db_path: Path,
    baseline_path: Path,
    hgnc_resolver: HgncResolver,
    gene_1_cites_rejected: bool = True,
) -> SeedStats:
    return seed_panelapp_publications(
        db_path, _seed_panels(gene_1_cites_rejected), baseline_path, hgnc_resolver
    )


def _links(db_path: Path) -> list[tuple[int, str]]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT hgnc_id, paper_doi FROM panelapp_citations ORDER BY hgnc_id, paper_doi"
        ).fetchall()


EXPECTED_LINKS = [
    (1, "10.1/rejected"),
    (1, "10.1000/in-baseline"),
    (1, "10.1000/pmid222"),
    (3, "10.1000/by-pmid"),
    (3, "10.1000/pmid222"),
]


def test_seeding_copies_from_the_baseline_and_fetches_the_rest(
    tmp_path: Path, sources: _FakeSources, hgnc_resolver: HgncResolver
) -> None:
    db_path = _make_run_db(tmp_path)

    stats = _seed(db_path, _make_seed_baseline(tmp_path), hgnc_resolver)

    # Only the publication the baseline lacks is fetched; gene 4 is not a run gene.
    assert sources.pmid_requests == [22222222]
    assert sources.doi_requests == []
    assert (stats.genes_considered, stats.genes_on_panelapp) == (2, 2)
    assert (stats.copied_from_baseline, stats.fetched, stats.unresolved) == (2, 1, 0)
    with sqlite3.connect(db_path) as conn:
        added = conn.execute(
            "SELECT doi, pmid, title, abstract, source, source_metadata, source_type, "
            "source_details, download_status FROM papers WHERE source_type = 'expansion' "
            "ORDER BY doi"
        ).fetchall()
    # Baseline papers keep the baseline's DOI spelling and metadata; a paper cited for
    # several genes is attributed to the lowest HGNC ID.
    assert added == [
        (
            "10.1000/by-pmid",
            40000001,
            "Baseline paper by PMID",
            "Abstract",
            "pubmed",
            '{"pmcid": "PMC1"}',
            SEED_SOURCE_TYPE,
            "panelapp:3",
            "scheduled",
        ),
        (
            "10.1000/in-baseline",
            None,
            "Baseline paper by DOI",
            "Abstract",
            "crossref",
            "{}",
            SEED_SOURCE_TYPE,
            "panelapp:1",
            "scheduled",
        ),
        (
            "10.1000/pmid222",
            22222222,
            "Paper 10.1000/pmid222",
            "",
            "pubmed",
            "{}",
            SEED_SOURCE_TYPE,
            "panelapp:1",
            "scheduled",
        ),
    ]


def test_seeding_links_every_run_gene_to_the_held_papers_it_cites(
    tmp_path: Path, sources: _FakeSources, hgnc_resolver: HgncResolver
) -> None:
    db_path = _make_run_db(tmp_path)

    stats = _seed(db_path, _make_seed_baseline(tmp_path), hgnc_resolver)

    assert _links(db_path) == EXPECTED_LINKS
    assert (stats.links.links, stats.links.papers) == (5, 4)
    # The cited paper the screen rejected is scheduled for download, as are the papers
    # seeding added; the paper only a non-run gene cites is left alone.
    assert stats.links.scheduled == 1
    with sqlite3.connect(db_path) as conn:
        statuses = dict(
            conn.execute(
                "SELECT doi, download_status FROM papers WHERE source_type = 'initial'"
            ).fetchall()
        )
    assert statuses == {
        "10.1/relevant-a": "downloaded",
        "10.1/relevant-b": "downloaded",
        "10.1/rejected": "scheduled",
        "10.1/rejected-other": None,
    }


def test_seeding_is_idempotent_and_replaces_the_links(
    tmp_path: Path, sources: _FakeSources, hgnc_resolver: HgncResolver
) -> None:
    db_path = _make_run_db(tmp_path)
    baseline_path = _make_seed_baseline(tmp_path)
    _seed(db_path, baseline_path, hgnc_resolver)
    sources.pmid_requests.clear()

    rerun = _seed(db_path, baseline_path, hgnc_resolver)

    assert sources.pmid_requests == []
    assert (rerun.copied_from_baseline, rerun.fetched, rerun.already_present) == (0, 0, 5)
    assert rerun.links.scheduled == 0
    assert _links(db_path) == EXPECTED_LINKS

    # A snapshot that no longer cites the rejected paper for gene 1 drops its link.
    _seed(db_path, baseline_path, hgnc_resolver, gene_1_cites_rejected=False)
    assert _links(db_path) == [link for link in EXPECTED_LINKS if link != (1, "10.1/rejected")]


def test_held_papers_are_linked_and_scheduled_when_a_fetch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hgnc_resolver: HgncResolver
) -> None:
    def unreachable(pmids: Sequence[int], source_details: str) -> list[Paper]:
        raise PubmedFetchError("efetch failed")

    monkeypatch.setattr(panelapp_publications, "fetch_papers_by_pmids", unreachable)
    db_path = _make_run_db(tmp_path)

    with pytest.raises(PubmedFetchError):
        _seed(db_path, _make_seed_baseline(tmp_path), hgnc_resolver)

    # The papers held before fetching, the baseline's copies among them, are linked.
    assert _links(db_path) == [
        (1, "10.1/rejected"),
        (1, "10.1000/in-baseline"),
        (3, "10.1000/by-pmid"),
    ]
    with sqlite3.connect(db_path) as conn:
        (status,) = conn.execute(
            "SELECT download_status FROM papers WHERE doi = '10.1/rejected'"
        ).fetchone()
    assert status == "scheduled"
