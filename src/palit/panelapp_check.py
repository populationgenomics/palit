"""The second relevance level: compare a paper's genes with what PanelApp has curated.

The scope screen finds papers with human genetic evidence for specific genes.
This check shows the model each gene's PanelApp Australia entries on the
reference panels and asks for a verdict per gene-disease association; the
paper stays relevant only if one of them is not yet curated there.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from palit.hgnc import HgncResolver
from palit.panelapp_client import clean_panel_publications
from palit.panelapp_integration import INCIDENTALOME_PANEL_ID

RELEVANT_VERDICTS: frozenset[str] = frozenset(
    {"new_gene", "new_disease", "limited_evidence", "new_inheritance"}
)
_RATINGS = {"3": "GREEN", "2": "AMBER", "1": "RED", "0": "RED"}

FALLBACK_NOTE = (
    " The one exception is a gene whose only reference-panel entry is on the Incidentalome:"
    " that panel deliberately carries only a gene's adult-onset incidental-findings"
    " associations, so for such genes the entries from all panels are shown and count."
)


@dataclass(frozen=True)
class PanelEntry:
    """One gene or repeat-expansion entry on one panel."""

    panel_name: str
    rating: str
    moi: str
    phenotypes: tuple[str, ...]
    pmids: frozenset[int]
    dois: frozenset[str]  # lowercased

    def cites(self, pmid: int | None, doi: str) -> bool:
        return (pmid is not None and pmid in self.pmids) or doi.lower() in self.dois


@dataclass(frozen=True)
class GeneRecord:
    entries: tuple[PanelEntry, ...]
    all_panels: bool  # Incidentalome-only gene, shown with its entries from every panel


@dataclass(frozen=True)
class CuratedRecord:
    """The PanelApp entries the check compares against, keyed by HGNC ID."""

    panel_date: str
    reference_panel_ids: tuple[int, ...]
    reference_panel_names: tuple[str, ...]
    incidentalome_fallback: bool
    genes: dict[int, GeneRecord]

    @classmethod
    def build(
        cls,
        panel_data: dict[int, dict[str, Any]],
        panel_date: str,
        reference_panel_ids: list[int],
        *,
        incidentalome_fallback: bool,
    ) -> "CuratedRecord":
        reference: dict[int, list[PanelEntry]] = {}
        on_panels: dict[int, set[int]] = {}
        for panel_id in reference_panel_ids:
            for hgnc_id, entry in _panel_entries(panel_data[panel_id], require_hgnc_id=True):
                reference.setdefault(hgnc_id, []).append(entry)
                on_panels.setdefault(hgnc_id, set()).add(panel_id)
        everywhere: dict[int, list[PanelEntry]] = {}
        if incidentalome_fallback:
            for panel in panel_data.values():
                for hgnc_id, entry in _panel_entries(panel, require_hgnc_id=False):
                    everywhere.setdefault(hgnc_id, []).append(entry)
        genes = {
            hgnc_id: GeneRecord(tuple(everywhere[hgnc_id]), all_panels=True)
            if incidentalome_fallback and on_panels[hgnc_id] == {INCIDENTALOME_PANEL_ID}
            else GeneRecord(tuple(entries), all_panels=False)
            for hgnc_id, entries in reference.items()
        }
        return cls(
            panel_date=panel_date,
            reference_panel_ids=tuple(reference_panel_ids),
            reference_panel_names=tuple(panel_data[pid]["name"] for pid in reference_panel_ids),
            incidentalome_fallback=incidentalome_fallback,
            genes=genes,
        )


def _panel_entries(panel: dict[str, Any], *, require_hgnc_id: bool) -> list[tuple[int, PanelEntry]]:
    entries = []
    for entity in panel["genes"] + panel["strs"]:
        hgnc_id = entity["gene_data"].get("hgnc_id")
        if hgnc_id is None:
            if require_hgnc_id:
                raise ValueError(
                    f"Entity '{entity['entity_name']}' in panel {panel['id']} has no hgnc_id"
                )
            continue
        publications = clean_panel_publications(entity.get("publications") or [])
        entries.append(
            (
                int(hgnc_id.removeprefix("HGNC:")),
                PanelEntry(
                    panel_name=panel["name"],
                    rating=_RATINGS[entity["confidence_level"]],
                    moi=entity.get("mode_of_inheritance") or "not specified",
                    phenotypes=tuple(entity.get("phenotypes") or ()),
                    pmids=frozenset(publications.pmids),
                    dois=frozenset(doi.lower() for doi in publications.dois),
                ),
            )
        )
    return entries


def load_check_system(prompt_path: Path, record: CuratedRecord) -> str:
    return prompt_path.read_text().format(
        reference_panels=", ".join(record.reference_panel_names),
        fallback_note=FALLBACK_NOTE if record.incidentalome_fallback else "",
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
    gene = record.genes.get(hgnc.hgnc_id)
    if gene is None:
        return f"{header}: no entry on the reference panels."
    scope = (
        "on all panels (Incidentalome-only gene)" if gene.all_panels else "on the reference panels"
    )
    lines = [f"{header}: {len(gene.entries)} entries {scope}"]
    for entry in sorted(gene.entries, key=lambda e: (e.rating != "GREEN", e.rating != "AMBER")):
        phenotypes = "; ".join(entry.phenotypes) if entry.phenotypes else "none listed"
        lines.append(
            f"- {entry.panel_name} | {entry.rating} | {entry.moi} | phenotypes: {phenotypes}"
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
