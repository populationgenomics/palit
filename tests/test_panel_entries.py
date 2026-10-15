"""Tests for each gene's PanelApp entries."""

import dataclasses
from typing import Any

import pytest

from palit.panel_entries import PanelEntry, ReviewState, entries_by_gene, reviewed_in_full
from palit.panelapp_integration import INCIDENTALOME_PANEL_ID, MENDELIOME_PANEL_ID

Snapshot = dict[int, dict[str, Any]]


def test_entries_include_repeat_expansions(
    panel_entries: dict[int, tuple[PanelEntry, ...]],
) -> None:
    assert [e.panel_name for e in panel_entries[2]] == ["Mendeliome"]


def test_entries_list_reference_panels_first(
    panel_entries: dict[int, tuple[PanelEntry, ...]],
) -> None:
    # Reference panels first, then the others by name.
    assert [(e.panel_name, e.reference) for e in panel_entries[3]] == [
        ("Incidentalome", True),
        ("Ataxia", False),
    ]


def test_entries_of_one_panel(panelapp_snapshot: Snapshot) -> None:
    entries = entries_by_gene([panelapp_snapshot[INCIDENTALOME_PANEL_ID]], [INCIDENTALOME_PANEL_ID])
    assert {
        hgnc_id: [e.panel_name for e in gene_entries] for hgnc_id, gene_entries in entries.items()
    } == {3: ["Incidentalome"]}


def test_reference_entity_without_hgnc_id_raises(panelapp_snapshot: Snapshot) -> None:
    other_panel = next(p for p in panelapp_snapshot.values() if p["name"] == "Ataxia")
    nameless = next(e for e in other_panel["genes"] if "hgnc_id" not in e["gene_data"])
    panelapp_snapshot[MENDELIOME_PANEL_ID]["genes"].append(nameless)
    with pytest.raises(ValueError, match="NOID"):
        entries_by_gene(list(panelapp_snapshot.values()), [MENDELIOME_PANEL_ID])


def test_only_gene_entries_without_reviews_are_unreviewed(panelapp_snapshot: Snapshot) -> None:
    ataxia = next(p for p in panelapp_snapshot.values() if p["name"] == "Ataxia")
    ataxia["genes"][0]["evaluations"] = []  # GENEB, reviewed on the Incidentalome
    panelapp_snapshot[MENDELIOME_PANEL_ID]["genes"][0]["evaluations"] = None  # not served
    entries = entries_by_gene(
        list(panelapp_snapshot.values()), [MENDELIOME_PANEL_ID, INCIDENTALOME_PANEL_ID]
    )
    assert [(e.panel_name, e.review_state, e.unreviewed) for e in entries[3]] == [
        ("Incidentalome", ReviewState.REVIEWED, False),
        ("Ataxia", ReviewState.UNREVIEWED, True),
    ]
    assert (entries[1][0].review_state, entries[1][0].unreviewed) == (ReviewState.UNKNOWN, False)
    # a repeat expansion, whose reviews the snapshot does not fetch
    assert (entries[2][0].review_state, entries[2][0].unreviewed) == (ReviewState.UNKNOWN, False)


def test_a_snapshot_without_reviews_raises(panelapp_snapshot: Snapshot) -> None:
    del panelapp_snapshot[MENDELIOME_PANEL_ID]["evaluations_fetched_at"]
    with pytest.raises(KeyError, match="add-reviews"):
        entries_by_gene([panelapp_snapshot[MENDELIOME_PANEL_ID]], [MENDELIOME_PANEL_ID])


def test_review_publications_come_from_gene_entries_reviews(panelapp_snapshot: Snapshot) -> None:
    mendeliome = panelapp_snapshot[MENDELIOME_PANEL_ID]
    mendeliome["genes"][0]["evaluations"] = [
        {"rating": "GREEN", "publications": ["PMID: 23456"]},
        {"rating": "AMBER", "publications": None},
        {"rating": "AMBER", "publications": ["10.1000/review"]},
    ]
    mendeliome["strs"][0]["evaluations"] = [{"rating": "GREEN", "publications": ["34567"]}]
    entries = entries_by_gene([mendeliome], [MENDELIOME_PANEL_ID])
    assert entries[1][0].publications == ("12345", "10.1000/Cited")
    assert entries[1][0].review_publications == ("PMID: 23456", "10.1000/review")
    assert entries[2][0].review_publications == ()  # the snapshot holds no repeat reviews


def _reviewed_in_full(
    snapshot: Snapshot, pmid_by_doi: dict[str, int | None], reference: list[int] | None = None
) -> bool:
    """Whether the target panels (the Mendeliome unless *reference*) have reviewed these
    papers of GENEA in full."""
    entries = entries_by_gene(list(snapshot.values()), reference or [MENDELIOME_PANEL_ID])
    return reviewed_in_full(entries[1], pmid_by_doi)


# GENEA's Mendeliome entry cites PMID 12345 and DOI 10.1000/Cited; its review cites
# 10.1000/Review (_cite_in_review). P1 is cited by its PMID, P2 by its DOI in another
# letter case, P4 by its review, by its DOI in another letter case; P3 is not cited.
P1: dict[str, int | None] = {"10.1/p1": 12345}
P2: dict[str, int | None] = {"10.1000/CITED": None}
P3: dict[str, int | None] = {"10.1/p3": 99999}
P4: dict[str, int | None] = {"10.1000/review": 77777}


def _cite_in_review(snapshot: Snapshot) -> None:
    snapshot[MENDELIOME_PANEL_ID]["genes"][0]["evaluations"] = [
        {"rating": "GREEN", "publications": ["10.1000/Review"]}
    ]


def test_papers_cited_and_reviewed_on_the_target_panels_are_reviewed_in_full(
    panelapp_snapshot: Snapshot,
) -> None:
    _cite_in_review(panelapp_snapshot)
    assert _reviewed_in_full(panelapp_snapshot, P1 | P2 | P4)


def test_an_uncited_paper_is_not_reviewed(panelapp_snapshot: Snapshot) -> None:
    assert not _reviewed_in_full(panelapp_snapshot, P1 | P2 | P3)


def test_a_doi_matches_only_its_own_paper(panelapp_snapshot: Snapshot) -> None:
    """A paper whose DOI is not cited is cited only through its own PMID: an uncited
    PMID, or none, leaves it uncited."""
    assert not _reviewed_in_full(panelapp_snapshot, {"10.1/other": 54321})
    assert not _reviewed_in_full(panelapp_snapshot, {"10.1/other": None})


def test_cited_papers_of_an_unreviewed_entry_are_not_reviewed(panelapp_snapshot: Snapshot) -> None:
    panelapp_snapshot[MENDELIOME_PANEL_ID]["genes"][0]["evaluations"] = []
    assert not _reviewed_in_full(panelapp_snapshot, P1 | P2)


def test_an_unreviewed_entry_on_another_target_panel_counts(panelapp_snapshot: Snapshot) -> None:
    """GENEA is reviewed on the Mendeliome, which cites the papers, but not on the
    Incidentalome."""
    incidentalome = panelapp_snapshot[INCIDENTALOME_PANEL_ID]
    incidentalome["genes"].append(dict(panelapp_snapshot[MENDELIOME_PANEL_ID]["genes"][0]))
    incidentalome["genes"][-1]["evaluations"] = []
    reference = [MENDELIOME_PANEL_ID, INCIDENTALOME_PANEL_ID]
    assert not _reviewed_in_full(panelapp_snapshot, P1 | P2, reference)
    incidentalome["genes"][-1]["evaluations"] = [{"rating": "GREEN"}]
    assert _reviewed_in_full(panelapp_snapshot, P1 | P2, reference)


def test_citations_on_other_panels_do_not_count(panelapp_snapshot: Snapshot) -> None:
    """GENEB's Ataxia entry, on no target panel, cites the paper; its Incidentalome
    entry does not."""
    ataxia = next(p for p in panelapp_snapshot.values() if p["name"] == "Ataxia")
    ataxia["genes"][0]["publications"] = ["12345"]
    entries = entries_by_gene(list(panelapp_snapshot.values()), [INCIDENTALOME_PANEL_ID])
    _, ataxia_entry = entries[3]
    assert not reviewed_in_full(entries[3], P1)
    assert reviewed_in_full([dataclasses.replace(ataxia_entry, reference=True)], P1)


def test_cited_papers_of_an_entry_with_unknown_reviews_are_not_reviewed(
    panelapp_snapshot: Snapshot,
) -> None:
    """PanelApp could not serve GENEA's Mendeliome reviews, so nobody may have reviewed it."""
    panelapp_snapshot[MENDELIOME_PANEL_ID]["genes"][0]["evaluations"] = None
    assert not _reviewed_in_full(panelapp_snapshot, P1 | P2)


def test_cited_papers_of_a_repeat_expansion_are_not_reviewed(panelapp_snapshot: Snapshot) -> None:
    """The snapshot holds no reviews of REPEAT1's repeat-expansion entry."""
    panelapp_snapshot[MENDELIOME_PANEL_ID]["strs"][0]["publications"] = ["12345"]
    entries = entries_by_gene([panelapp_snapshot[MENDELIOME_PANEL_ID]], [MENDELIOME_PANEL_ID])
    assert not reviewed_in_full(entries[2], P1)


def test_target_panels_cite_the_papers_between_them(panelapp_snapshot: Snapshot) -> None:
    """GENEA is reviewed on both target panels; the Mendeliome cites P1 only, the
    Incidentalome P2 only."""
    mendeliome_genea = panelapp_snapshot[MENDELIOME_PANEL_ID]["genes"][0]
    mendeliome_genea["publications"] = ["12345"]
    incidentalome_genea = {**mendeliome_genea, "publications": ["10.1000/Cited"]}
    panelapp_snapshot[INCIDENTALOME_PANEL_ID]["genes"].append(incidentalome_genea)
    reference = [MENDELIOME_PANEL_ID, INCIDENTALOME_PANEL_ID]
    assert _reviewed_in_full(panelapp_snapshot, P1 | P2, reference)
    assert not _reviewed_in_full(panelapp_snapshot, P1 | P2, [MENDELIOME_PANEL_ID])


def test_a_gene_on_no_target_panel_is_not_reviewed(panelapp_snapshot: Snapshot) -> None:
    assert not _reviewed_in_full(panelapp_snapshot, P1, [INCIDENTALOME_PANEL_ID])


def test_no_papers_raise(panelapp_snapshot: Snapshot) -> None:
    with pytest.raises(ValueError, match="No papers"):
        _reviewed_in_full(panelapp_snapshot, {})
