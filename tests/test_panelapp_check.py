"""Tests for the PanelApp check's curated record and gene context."""

from typing import Any

import pytest

from palit.hgnc import HgncResolver
from palit.panelapp_check import CuratedRecord, gene_context, is_relevant, with_hgnc_ids
from palit.panelapp_integration import INCIDENTALOME_PANEL_ID, MENDELIOME_PANEL_ID

Snapshot = dict[int, dict[str, Any]]


def test_record_includes_repeat_expansions(curated_record: CuratedRecord) -> None:
    record = curated_record
    assert record.genes[2].entries[0].panel_name == "Mendeliome"
    assert record.reference_panel_names == ("Mendeliome", "Incidentalome")


def test_incidentalome_only_gene_falls_back_to_all_panels(curated_record: CuratedRecord) -> None:
    gene = curated_record.genes[3]
    assert gene.all_panels
    assert {e.panel_name for e in gene.entries} == {"Incidentalome", "Ataxia"}


def test_without_fallback_incidentalome_entries_are_all_there_is(
    panelapp_snapshot: Snapshot,
) -> None:
    record = CuratedRecord.build(
        panelapp_snapshot,
        "2026-09-01",
        [MENDELIOME_PANEL_ID, INCIDENTALOME_PANEL_ID],
        incidentalome_fallback=False,
    )
    gene = record.genes[3]
    assert not gene.all_panels
    assert [e.panel_name for e in gene.entries] == ["Incidentalome"]


def test_reference_entity_without_hgnc_id_raises(panelapp_snapshot: Snapshot) -> None:
    other_panel = next(p for p in panelapp_snapshot.values() if p["name"] == "Ataxia")
    nameless = next(e for e in other_panel["genes"] if "hgnc_id" not in e["gene_data"])
    panelapp_snapshot[MENDELIOME_PANEL_ID]["genes"].append(nameless)
    with pytest.raises(ValueError, match="NOID"):
        CuratedRecord.build(
            panelapp_snapshot, "2026-09-01", [MENDELIOME_PANEL_ID], incidentalome_fallback=True
        )


def test_gene_context_marks_citations_and_missing_genes(
    curated_record: CuratedRecord, hgnc_resolver: HgncResolver
) -> None:
    record, resolver = curated_record, hgnc_resolver
    cited = gene_context("GENEA", resolver, record, pmid=12345, doi="10.1/other")
    assert "Mendeliome | GREEN" in cited and "CITES THIS PAPER" in cited
    by_doi = gene_context("GENEA", resolver, record, pmid=None, doi="10.1000/cited")
    assert "CITES THIS PAPER" in by_doi
    assert "no entry on the reference panels" in gene_context("GENEC", resolver, record, None, "x")
    assert "could not be looked up" in gene_context("NOTAGENE", resolver, record, None, "x")
    fallback = gene_context("GENEB", resolver, record, None, "x")
    assert "on all panels (Incidentalome-only gene)" in fallback
    assert "phenotypes: none listed" in fallback


def test_relevance_follows_verdicts(hgnc_resolver: HgncResolver) -> None:
    associations = with_hgnc_ids(
        [
            {"gene_symbol": "GENEA", "disease": "d", "verdict": "already_curated", "reason": "r"},
            {"gene_symbol": "NOTAGENE", "disease": "d", "verdict": "new_gene", "reason": "r"},
        ],
        hgnc_resolver,
    )
    assert [a["hgnc_id"] for a in associations] == [1, None]
    assert is_relevant(associations)
    assert not is_relevant(associations[:1])
