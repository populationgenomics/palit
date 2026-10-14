"""Tests for the PanelApp check's curated record and gene context."""

from pathlib import Path
from typing import Any

import pytest

from palit.hgnc import HgncResolver
from palit.panelapp_check import (
    ALL_PANELS_SCOPE,
    SCOPE_PANEL_SCOPE,
    CuratedRecord,
    gene_context,
    is_relevant,
    load_check_system,
    with_hgnc_ids,
    with_review_state,
)
from palit.panelapp_integration import INCIDENTALOME_PANEL_ID, MENDELIOME_PANEL_ID

Snapshot = dict[int, dict[str, Any]]

CHECK_PROMPT = Path(__file__).resolve().parents[1] / "prompts/relevance_panelapp_check_prompt.txt"


def test_record_includes_repeat_expansions(curated_record: CuratedRecord) -> None:
    record = curated_record
    assert record.genes[2][0].panel_name == "Mendeliome"
    assert record.reference_panel_names == ("Mendeliome", "Incidentalome")


def test_genes_on_reference_panels_come_with_all_their_panels(
    curated_record: CuratedRecord,
) -> None:
    entries = curated_record.genes[3]
    # Reference panels first, then the others by name.
    assert [(e.panel_name, e.reference) for e in entries] == [
        ("Incidentalome", True),
        ("Ataxia", False),
    ]


def test_genes_on_other_panels_only_are_not_in_the_record(curated_record: CuratedRecord) -> None:
    assert set(curated_record.genes) == {1, 2, 3}


def test_scoped_record_shows_the_scope_panel_only(panelapp_snapshot: Snapshot) -> None:
    record = CuratedRecord.build(
        panelapp_snapshot, "2026-09-01", [INCIDENTALOME_PANEL_ID], all_panels=False
    )
    assert [e.panel_name for e in record.genes[3]] == ["Incidentalome"]
    assert set(record.genes) == {3}


def test_reference_entity_without_hgnc_id_raises(panelapp_snapshot: Snapshot) -> None:
    other_panel = next(p for p in panelapp_snapshot.values() if p["name"] == "Ataxia")
    nameless = next(e for e in other_panel["genes"] if "hgnc_id" not in e["gene_data"])
    panelapp_snapshot[MENDELIOME_PANEL_ID]["genes"].append(nameless)
    with pytest.raises(ValueError, match="NOID"):
        CuratedRecord.build(panelapp_snapshot, "2026-09-01", [MENDELIOME_PANEL_ID], all_panels=True)


def test_gene_context_marks_citations_and_missing_genes(
    curated_record: CuratedRecord, hgnc_resolver: HgncResolver
) -> None:
    record, resolver = curated_record, hgnc_resolver
    cited = gene_context("GENEA", resolver, record, pmid=12345, doi="10.1/other")
    assert "Mendeliome (reference panel) | GREEN" in cited and "CITES THIS PAPER" in cited
    by_doi = gene_context("GENEA", resolver, record, pmid=None, doi="10.1000/cited")
    assert "CITES THIS PAPER" in by_doi
    assert "no entry on the reference panels" in gene_context("GENEC", resolver, record, None, "x")
    assert "could not be looked up" in gene_context("NOTAGENE", resolver, record, None, "x")


def test_gene_context_lists_entries_on_every_panel(
    curated_record: CuratedRecord, hgnc_resolver: HgncResolver
) -> None:
    context = gene_context("GENEB", hgnc_resolver, curated_record, pmid=None, doi="x")
    assert context == (
        "GENEB (HGNC:3, geneb protein): 2 entries\n"
        "- Incidentalome (reference panel) | GREEN | BIALLELIC, autosomal or pseudoautosomal"
        " | phenotypes: none listed\n"
        "- Ataxia | GREEN | BIALLELIC, autosomal or pseudoautosomal"
        " | phenotypes: Paediatric disease MIM#200000"
    )


def test_check_system_names_the_entries_shown(
    curated_record: CuratedRecord, panelapp_snapshot: Snapshot
) -> None:
    system = load_check_system(CHECK_PROMPT, curated_record)
    assert ALL_PANELS_SCOPE.format(reference_panels="Mendeliome, Incidentalome") in system
    assert "GenCC" not in system
    assert "{" not in system
    scoped = CuratedRecord.build(
        panelapp_snapshot, "2026-09-01", [INCIDENTALOME_PANEL_ID], all_panels=False
    )
    assert SCOPE_PANEL_SCOPE.format(reference_panels="Incidentalome") in load_check_system(
        CHECK_PROMPT, scoped
    )


def _checked(
    verdicts: list[tuple[str, str]], resolver: HgncResolver, record: CuratedRecord
) -> list[dict[str, Any]]:
    """The check's associations of (gene symbol, verdict) pairs, as store_checks stores them."""
    associations = [
        {"gene_symbol": symbol, "disease": "d", "verdict": verdict, "reason": "r"}
        for symbol, verdict in verdicts
    ]
    return with_review_state(with_hgnc_ids(associations, resolver), record)


def test_relevance_follows_verdicts(
    curated_record: CuratedRecord, hgnc_resolver: HgncResolver
) -> None:
    associations = _checked(
        [("GENEA", "already_curated"), ("NOTAGENE", "new_gene")], hgnc_resolver, curated_record
    )
    assert [(a["hgnc_id"], a["unreviewed"]) for a in associations] == [(1, False), (None, False)]
    assert is_relevant(associations)
    assert not is_relevant(associations[:1])


@pytest.mark.parametrize(
    ("verdict", "relevant"),
    [("already_curated", True), ("already_cited", True), ("out_of_scope", False)],
)
def test_an_unreviewed_reference_entry_keeps_curated_genes(
    panelapp_snapshot: Snapshot, hgnc_resolver: HgncResolver, verdict: str, relevant: bool
) -> None:
    panelapp_snapshot[MENDELIOME_PANEL_ID]["genes"][0]["evaluations"] = []
    record = CuratedRecord.build(
        panelapp_snapshot, "2026-09-01", [MENDELIOME_PANEL_ID], all_panels=True
    )
    associations = _checked([("GENEA", verdict)], hgnc_resolver, record)
    assert associations[0]["unreviewed"]
    assert is_relevant(associations) == relevant


def test_only_reference_entries_count_as_unreviewed(
    panelapp_snapshot: Snapshot, hgnc_resolver: HgncResolver
) -> None:
    ataxia = next(p for p in panelapp_snapshot.values() if p["name"] == "Ataxia")
    ataxia["genes"][0]["evaluations"] = []  # GENEB, reviewed on the Incidentalome
    panelapp_snapshot[MENDELIOME_PANEL_ID]["genes"][0]["evaluations"] = None  # not served
    record = CuratedRecord.build(
        panelapp_snapshot,
        "2026-09-01",
        [MENDELIOME_PANEL_ID, INCIDENTALOME_PANEL_ID],
        all_panels=True,
    )
    assert not record.unreviewed(3)
    assert not record.unreviewed(1)
    assert not record.unreviewed(2)  # a repeat expansion, whose reviews the snapshot lacks


def test_a_snapshot_without_reviews_raises(panelapp_snapshot: Snapshot) -> None:
    del panelapp_snapshot[MENDELIOME_PANEL_ID]["evaluations_fetched_at"]
    with pytest.raises(KeyError, match="add-reviews"):
        CuratedRecord.build(
            panelapp_snapshot, "2026-09-01", [MENDELIOME_PANEL_ID], all_panels=False
        )
