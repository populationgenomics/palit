"""Each gene's PanelApp Australia entries, gene and repeat-expansion entries alike.

Gene assessment shows the model a gene's entries on every panel, the target
panels marked; the report lists the target panels' entries nobody has reviewed.
Evidence the target panels have already reviewed in full (``reviewed_in_full``) is
not aggregated, and an association resting on it is no finding.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from palit.panelapp_client import EVALUATIONS_FETCHED_AT, clean_panel_publications
from palit.panelapp_integration import MOI_NOT_SPECIFIED

_RATINGS = {"3": "GREEN", "2": "AMBER", "1": "RED", "0": "RED"}


class ReviewState(StrEnum):
    """What the snapshot knows about the reviews of one entry."""

    REVIEWED = "reviewed"  # PanelApp lists reviews of the entry
    UNREVIEWED = "unreviewed"  # PanelApp lists none
    # The snapshot does not hold them: a repeat expansion, whose reviews it does not
    # fetch, or a gene whose reviews PanelApp could not serve
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PanelEntry:
    """One gene or repeat-expansion entry on one panel."""

    panel_id: int
    panel_name: str
    reference: bool  # the panel is one of the reference (target) panels
    entity_type: Literal["gene", "str"]  # a gene or a repeat-expansion (STR) entry
    rating: str
    moi: str  # MOI_NOT_SPECIFIED when the entry has none
    mode_of_pathogenicity: str  # empty when the entry has none
    phenotypes: tuple[str, ...]
    publications: tuple[str, ...]  # as PanelApp lists them
    # The publications its reviews cite, as PanelApp lists them; none unless REVIEWED
    review_publications: tuple[str, ...]
    review_state: ReviewState

    @property
    def unreviewed(self) -> bool:
        """PanelApp lists no reviews of the entry."""
        return self.review_state is ReviewState.UNREVIEWED


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


def _review_state(reviews: list[dict[str, Any]] | None) -> ReviewState:
    """The state of an entry's reviews from the snapshot's ``evaluations``, None when it
    holds none."""
    if reviews is None:
        return ReviewState.UNKNOWN
    return ReviewState.REVIEWED if reviews else ReviewState.UNREVIEWED


def _panel_entries(panel: dict[str, Any], reference: bool) -> list[tuple[int, PanelEntry]]:
    if EVALUATIONS_FETCHED_AT not in panel:
        raise KeyError(
            f"Panel {panel['id']} has no reviews in the snapshot; add them with "
            "`palit panelapp add-reviews`"
        )
    entities: list[tuple[dict[str, Any], Literal["gene", "str"], list[dict[str, Any]] | None]]
    entities = [(gene, "gene", gene["evaluations"]) for gene in panel["genes"]]
    entities += [(str_entity, "str", None) for str_entity in panel["strs"]]
    entries = []
    for entity, entity_type, reviews in entities:
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
                    entity_type=entity_type,
                    rating=_RATINGS[entity["confidence_level"]],
                    moi=entity.get("mode_of_inheritance") or MOI_NOT_SPECIFIED,
                    mode_of_pathogenicity=entity.get("mode_of_pathogenicity") or "",
                    phenotypes=tuple(entity.get("phenotypes") or ()),
                    publications=tuple(entity.get("publications") or ()),
                    review_publications=tuple(
                        publication
                        for review in reviews or ()
                        for publication in review.get("publications") or ()
                    ),
                    review_state=_review_state(reviews),
                ),
            )
        )
    return entries


def reviewed_in_full(entries: Iterable[PanelEntry], pmid_by_doi: Mapping[str, int | None]) -> bool:
    """Whether the target panels have already reviewed these papers of the gene in full.

    *entries* are the gene's entries, of which those on the reference (target) panels
    count; *pmid_by_doi* holds the papers, each DOI with its PMID or None. They are
    reviewed in full when the target panels hold the gene, every one of its entries
    there is known to be reviewed, and between them the entries and their reviews cite
    every paper, by DOI in any letter case or by PMID. An entry whose reviews the
    snapshot does not hold (``ReviewState.UNKNOWN``) may be unreviewed, and its reviews'
    citations are unknown, so it rules the papers out. PanelApp records citations per
    entry, not per disease, so a paper cited for one of the gene's diseases counts for
    all.
    """
    if not pmid_by_doi:
        raise ValueError("No papers to compare with the target panels' citations")
    target = [e for e in entries if e.reference]
    if not target or any(e.review_state is not ReviewState.REVIEWED for e in target):
        return False
    cited = clean_panel_publications(
        [publication for e in target for publication in e.publications + e.review_publications]
    )
    cited_dois = {doi.lower() for doi in cited.dois}
    return all(
        doi.lower() in cited_dois or (pmid is not None and pmid in cited.pmids)
        for doi, pmid in pmid_by_doi.items()
    )
