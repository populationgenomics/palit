"""Tests for seeding PanelApp's cited publications into the run corpus."""

import sqlite3
from pathlib import Path
from typing import Any

from palit.panelapp_client import PanelGeneData
from palit.panelapp_publications import collect_missing_publications

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
            [(hgnc_id, "10.1/seed") for hgnc_id in genes],
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

    def get_gene_evaluations(self, panel_id: int, hgnc_id: int) -> list[dict[str, Any]]:
        publications = self._review_publications.get(hgnc_id)
        return [{"publications": publications}] if publications else []


def _panel_data(gene_panels: dict[int, set[int]], panel_ids: list[int]) -> PanelGeneData:
    return PanelGeneData(
        panel_ids=panel_ids,
        gene_confidence={},
        gene_panel_mapping=gene_panels,
        gene_moi={},
    )


def test_collects_publications_absent_from_the_corpus(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path, [("10.1/held", 111)], [3604])
    client = _FakeClient({3604: ["111", "222"]})
    panel_data = _panel_data({3604: {MENDELIOME}}, [MENDELIOME])

    missing = collect_missing_publications(db_path, client, panel_data, [3604])

    assert missing.pmid_owner == {222: 3604}
    assert missing.publications_cited == 2
    assert missing.already_present == 1
    assert missing.genes_on_target_panel == 1


def test_unions_entity_and_review_publications(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path, [], [3604])
    client = _FakeClient({3604: ["111"]}, review_publications={3604: ["222"]})
    panel_data = _panel_data({3604: {MENDELIOME}}, [MENDELIOME])

    missing = collect_missing_publications(db_path, client, panel_data, [3604])

    assert missing.pmid_owner == {111: 3604, 222: 3604}


def test_attributes_a_shared_publication_to_the_lowest_hgnc_id(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path, [], [3604, 10260])
    client = _FakeClient({3604: ["999"], 10260: ["999"]})
    panel_data = _panel_data({3604: {MENDELIOME}, 10260: {MENDELIOME}}, [MENDELIOME])

    missing = collect_missing_publications(db_path, client, panel_data, [3604, 10260])

    assert missing.pmid_owner == {999: 3604}


def test_skips_genes_absent_from_every_target_panel(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path, [], [3604])
    client = _FakeClient({3604: ["111"]})
    # Gene sits only on a panel outside the target set.
    panel_data = _panel_data({3604: {SKELETAL}}, [MENDELIOME])

    missing = collect_missing_publications(db_path, client, panel_data, [3604])

    assert missing.pmid_owner == {}
    assert missing.genes_on_target_panel == 0


def test_matches_held_dois_case_insensitively(tmp_path: Path) -> None:
    db_path = _make_db(tmp_path, [("10.1/mixedcase", None)], [3604])
    client = _FakeClient({3604: ["10.1/MixedCase", "10.1/other"]})
    panel_data = _panel_data({3604: {MENDELIOME}}, [MENDELIOME])

    missing = collect_missing_publications(db_path, client, panel_data, [3604])

    assert missing.doi_owner == {"10.1/other": 3604}
    assert missing.already_present == 1
