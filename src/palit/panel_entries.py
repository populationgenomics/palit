"""Each gene's PanelApp Australia entries, gene and repeat-expansion entries alike.

Gene assessment shows the model a gene's entries on every panel, the target
panels marked; the report lists the target panels' entries nobody has reviewed.
"""

from dataclasses import dataclass
from typing import Any

from palit.panelapp_client import EVALUATIONS_FETCHED_AT

_RATINGS = {"3": "GREEN", "2": "AMBER", "1": "RED", "0": "RED"}


@dataclass(frozen=True)
class PanelEntry:
    """One gene or repeat-expansion entry on one panel."""

    panel_id: int
    panel_name: str
    reference: bool  # the panel is one of the reference (target) panels
    rating: str
    moi: str
    mode_of_pathogenicity: str  # empty when the entry has none
    phenotypes: tuple[str, ...]
    publications: tuple[str, ...]  # as PanelApp lists them
    # PanelApp lists no reviews of the entry. False for a repeat expansion, whose reviews
    # the snapshot does not hold, and for a gene whose reviews PanelApp could not serve.
    unreviewed: bool


def entries_by_gene(
    panels: list[dict[str, Any]], reference_panel_ids: list[int]
) -> dict[int, tuple[PanelEntry, ...]]:
    """Each gene's entries on *panels*, the reference panels first, then by panel name.

    An entity without an HGNC ID raises on a reference panel and is skipped elsewhere.
    """
    reference = set(reference_panel_ids)
    by_gene: dict[int, list[PanelEntry]] = {}
    for panel in panels:
        on_reference = panel["id"] in reference
        for hgnc_id, entry in _panel_entries(panel, on_reference):
            by_gene.setdefault(hgnc_id, []).append(entry)
    return {
        hgnc_id: tuple(sorted(entries, key=lambda e: (not e.reference, e.panel_name)))
        for hgnc_id, entries in by_gene.items()
    }


def _panel_entries(panel: dict[str, Any], reference: bool) -> list[tuple[int, PanelEntry]]:
    if EVALUATIONS_FETCHED_AT not in panel:
        raise KeyError(
            f"Panel {panel['id']} has no reviews in the snapshot; add them with "
            "`palit panelapp add-reviews`"
        )
    entities = [(gene, gene["evaluations"] == []) for gene in panel["genes"]]
    entities += [(str_entity, False) for str_entity in panel["strs"]]
    entries = []
    for entity, unreviewed in entities:
        hgnc_id = entity["gene_data"].get("hgnc_id")
        if hgnc_id is None:
            if reference:
                raise ValueError(
                    f"Entity '{entity['entity_name']}' in panel {panel['id']} has no hgnc_id"
                )
            continue
        entries.append(
            (
                int(hgnc_id.removeprefix("HGNC:")),
                PanelEntry(
                    panel_id=panel["id"],
                    panel_name=panel["name"],
                    reference=reference,
                    rating=_RATINGS[entity["confidence_level"]],
                    moi=entity.get("mode_of_inheritance") or "not specified",
                    mode_of_pathogenicity=entity.get("mode_of_pathogenicity") or "",
                    phenotypes=tuple(entity.get("phenotypes") or ()),
                    publications=tuple(entity.get("publications") or ()),
                    unreviewed=unreviewed,
                ),
            )
        )
    return entries
