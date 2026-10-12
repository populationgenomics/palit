"""The second relevance level: compare a paper's genes with what PanelApp has curated.

The scope screen finds papers with human genetic evidence for specific genes.
This check shows the model each gene's PanelApp Australia entries. A gene on a
reference panel is shown with its entries on every panel: the reference panels
rate the gene as a whole, and the panels that list only some of its diseases
show how PanelApp rates those diseases apart. A panel-scoped check shows the
scope panel's entry only. The model gives a verdict per gene-disease
association, and the paper stays relevant only if one of them is not yet
curated there.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from palit.hgnc import HgncResolver
from palit.panelapp_client import clean_panel_publications

RELEVANT_VERDICTS: frozenset[str] = frozenset(
    {"new_gene", "new_disease", "limited_evidence", "new_inheritance"}
)
_RATINGS = {"3": "GREEN", "2": "AMBER", "1": "RED", "0": "RED"}

# Which entries the check shows, in its system prompt, for a check over all panels and
# for a panel-scoped one.
ALL_PANELS_SCOPE = (
    "A gene with an entry on the reference panels ({reference_panels}) is shown with its"
    " entries on every panel; the reference panels are marked. A gene without an entry on"
    " the reference panels is new to them, whatever other panels hold it."
)
SCOPE_PANEL_SCOPE = (
    "The entries come from the reference panel ({reference_panels}); entries on other panels"
    " are not shown and do not count."
)


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
    pmids: frozenset[int]
    dois: frozenset[str]  # lowercased

    def cites(self, pmid: int | None, doi: str) -> bool:
        return (pmid is not None and pmid in self.pmids) or doi.lower() in self.dois


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


@dataclass(frozen=True)
class CuratedRecord:
    """The PanelApp entries the check compares against, for the genes on the reference panels."""

    panel_date: str
    reference_panel_ids: tuple[int, ...]
    reference_panel_names: tuple[str, ...]
    all_panels: bool  # genes are shown with their entries on every panel
    genes: dict[int, tuple[PanelEntry, ...]]

    @classmethod
    def build(
        cls,
        panel_data: dict[int, dict[str, Any]],
        panel_date: str,
        reference_panel_ids: list[int],
        *,
        all_panels: bool,
    ) -> "CuratedRecord":
        panels = (
            list(panel_data.values())
            if all_panels
            else [panel_data[pid] for pid in reference_panel_ids]
        )
        entries = entries_by_gene(panels, reference_panel_ids)
        return cls(
            panel_date=panel_date,
            reference_panel_ids=tuple(reference_panel_ids),
            reference_panel_names=tuple(panel_data[pid]["name"] for pid in reference_panel_ids),
            all_panels=all_panels,
            genes={
                hgnc_id: gene_entries
                for hgnc_id, gene_entries in entries.items()
                if gene_entries[0].reference
            },
        )


def _panel_entries(panel: dict[str, Any], reference: bool) -> list[tuple[int, PanelEntry]]:
    entries = []
    for entity in panel["genes"] + panel["strs"]:
        hgnc_id = entity["gene_data"].get("hgnc_id")
        if hgnc_id is None:
            if reference:
                raise ValueError(
                    f"Entity '{entity['entity_name']}' in panel {panel['id']} has no hgnc_id"
                )
            continue
        listed = entity.get("publications") or []
        publications = clean_panel_publications(listed)
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
                    publications=tuple(listed),
                    pmids=frozenset(publications.pmids),
                    dois=frozenset(doi.lower() for doi in publications.dois),
                ),
            )
        )
    return entries


def load_check_system(prompt_path: Path, record: CuratedRecord) -> str:
    reference_panels = ", ".join(record.reference_panel_names)
    scope = ALL_PANELS_SCOPE if record.all_panels else SCOPE_PANEL_SCOPE
    return prompt_path.read_text().format(
        entries_scope=scope.format(reference_panels=reference_panels)
    )


def gene_context(
    symbol: str,
    resolver: HgncResolver,
    record: CuratedRecord,
    pmid: int | None,
    doi: str,
) -> str:
    """The curated record of one gene as shown to the model."""
    hgnc = resolver.resolve(symbol)
    if hgnc is None:
        return f"{symbol}: symbol not recognised by HGNC, so it could not be looked up."
    header = f"{hgnc.symbol} (HGNC:{hgnc.hgnc_id}, {hgnc.name})"
    entries = record.genes.get(hgnc.hgnc_id)
    if entries is None:
        return f"{header}: no entry on the reference panels."
    lines = [f"{header}: {len(entries)} entries"]
    for entry in entries:
        phenotypes = "; ".join(entry.phenotypes) if entry.phenotypes else "none listed"
        panel = f"{entry.panel_name} (reference panel)" if entry.reference else entry.panel_name
        lines.append(
            f"- {panel} | {entry.rating} | {entry.moi} | phenotypes: {phenotypes}"
            + (" | CITES THIS PAPER" if entry.cites(pmid, doi) else "")
        )
    return "\n".join(lines)


def check_user_text(
    *,
    title: str,
    abstract: str,
    pmid: int | None,
    doi: str,
    screen_associations: list[dict[str, str]],
    resolver: HgncResolver,
    record: CuratedRecord,
) -> str:
    symbols = list(dict.fromkeys(a["gene_symbol"] for a in screen_associations))
    screening = "\n".join(f"- {a['gene_symbol']}: {a['disease']}" for a in screen_associations)
    contexts = "\n\n".join(gene_context(s, resolver, record, pmid, doi) for s in symbols)
    return (
        f"Title: {title}\nAbstract: {abstract}\n"
        f"PMID: {pmid if pmid is not None else 'none'}; DOI: {doi}\n\n"
        f"Associations found by the screening step:\n{screening}\n\n"
        f"PanelApp Australia entries:\n{contexts}"
    )


def with_hgnc_ids(
    associations: list[dict[str, str]], resolver: HgncResolver
) -> list[dict[str, Any]]:
    """The check's associations with each gene symbol resolved to an HGNC ID (None if unknown)."""
    resolved = []
    for association in associations:
        hgnc = resolver.resolve(association["gene_symbol"])
        resolved.append({**association, "hgnc_id": hgnc.hgnc_id if hgnc else None})
    return resolved


def is_relevant(associations: list[dict[str, Any]]) -> bool:
    return any(a["verdict"] in RELEVANT_VERDICTS for a in associations)
