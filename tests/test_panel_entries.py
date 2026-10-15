"""Tests for each gene's PanelApp entries."""

from typing import Any

import pytest

from palit.panel_entries import PanelEntry, entries_by_gene
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
    assert [(e.panel_name, e.unreviewed) for e in entries[3]] == [
        ("Incidentalome", False),
        ("Ataxia", True),
    ]
    assert not entries[1][0].unreviewed
    assert not entries[2][0].unreviewed  # a repeat expansion, whose reviews the snapshot lacks


def test_a_snapshot_without_reviews_raises(panelapp_snapshot: Snapshot) -> None:
    del panelapp_snapshot[MENDELIOME_PANEL_ID]["evaluations_fetched_at"]
    with pytest.raises(KeyError, match="add-reviews"):
        entries_by_gene([panelapp_snapshot[MENDELIOME_PANEL_ID]], [MENDELIOME_PANEL_ID])
