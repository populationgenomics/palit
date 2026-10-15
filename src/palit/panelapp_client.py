#!/usr/bin/env python3
"""Reads the dated PanelApp snapshots that palit panelapp snapshot builds."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from palit.panelapp_integration import TARGET_PANEL_IDS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedGene:
    """A gene resolved from paper mention to HGNC ID."""

    hgnc_id: int  # HGNC ID (integer)
    paper_symbol: str  # Original symbol mentioned in paper


@dataclass
class PanelGeneData:
    """The genes and STRs of the target panels, keyed by HGNC ID."""

    panel_ids: list[int]  # Ordered list of panel IDs
    # hgnc_id → {panel_id → confidence level} for the panels holding the gene; a panel
    # holding it as both a gene and an STR gives the higher of the two
    gene_panel_confidence: dict[int, dict[int, int]]

    def highest_confidence(self, hgnc_id: int) -> int:
        """The gene's highest confidence level across the target panels holding it."""
        return max(self.gene_panel_confidence[hgnc_id].values())


def find_gene_panel(
    hgnc_id: int,
    target_panel_ids: list[int],
    panel_data: PanelGeneData,
) -> int | None:
    """Find first target panel containing this gene.

    Iterates through target panels in the given order and returns the first
    panel that contains the gene.

    Args:
        hgnc_id: HGNC ID of the gene to look up
        target_panel_ids: Ordered list of panel IDs to check
        panel_data: PanelApp gene data with panel mappings

    Returns:
        Panel ID if found, None if gene is novel (not in any target panel).
    """
    gene_panels = panel_data.gene_panel_confidence.get(hgnc_id, {})
    for panel_id in target_panel_ids:
        if panel_id in gene_panels:
            return panel_id
    return None


@dataclass
class AllPanelsData:
    """Gene data from ALL panels with panel information."""

    gene_to_panels: dict[int, set[int]]  # hgnc_id -> set of panel_ids
    panel_names: dict[int, str]  # panel_id -> panel name


@dataclass
class PanelPublications:
    """Publication identifiers extracted from PanelApp panels."""

    pmids: set[int]
    dois: set[str]


# A DOI anywhere in a publication string: bare, after a "doi:" label, or inside a
# doi.org or publisher URL. Whitespace or a comma ends it.
_DOI = re.compile(r"(?<!\d)10\.\d{4,9}/[^\s,]+")

# The whole token holding a DOI, such as "https://doi.org/10.1016/j.jaci.2022.10.022",
# blanked before PMIDs are read so that neither the DOI's digits nor its URL's words
# ("org") are taken for a PMID or for the word before one.
_DOI_TOKEN = re.compile(r"[^\s,]*" + _DOI.pattern)

# A PubMed URL ("https://pubmed.ncbi.nlm.nih.gov/16794080/"), rewritten to
# "PMID 16794080" before PMIDs are read.
_PUBMED_URL_TOKEN = re.compile(r"[^\s,]*pubmed\.ncbi\.nlm\.nih\.gov/(\d+)[^\s,]*")

# A preprint server's URL appends a version, and possibly a file suffix, to a DOI
# whose suffix is digits and dots: ".../10.1101/2021.04.27.441137v1.full.pdf".
_PREPRINT_VERSION = re.compile(r"(10\.\d{4,9}/[\d.]*\d)v\d+(?:\.[\w-]+)*")

_NUMBER = re.compile(r"\d+")
_WORD = re.compile(r"[A-Za-z]+")

# Words that label the number after them as a PMID, lowercased. "pubmed" also covers
# the path of an older PubMed URL (".../pubmed/16794080").
_PUBMED_LABELS = frozenset({"pmid", "pmids", "pubmed"})

# PMIDs read from PanelApp's publication fields have 5 to 8 digits and no leading
# zero. The highest PMID is around 43 million in 2026. PMIDs of 1 to 4 digits exist
# (papers from 1975), but in these fields shorter numbers are list markers, section
# and volume numbers, years, or PMIDs cut short: every number of 1 to 4 digits that
# is labelled as a PMID in the 2026-10-13 snapshot is a truncated one.
_PMID_DIGITS = range(5, 9)


def _clean_doi(match: str) -> str:
    """A DOI as matched in free text, without a URL's query or fragment, trailing
    punctuation, or a preprint URL's version suffix."""
    doi = re.split(r"[?#]", match, maxsplit=1)[0]
    while doi and (
        doi[-1] in ".;:'\""
        or (doi[-1] == ")" and doi.count(")") > doi.count("("))
        or (doi[-1] == "]" and doi.count("]") > doi.count("["))
    ):
        doi = doi[:-1]
    preprint = _PREPRINT_VERSION.fullmatch(doi)
    return preprint.group(1) if preprint is not None else doi


def _is_pmid(number: str, preceding_text: str) -> bool:
    """Whether a number in a publication string is a PMID.

    It is when it has a PMID's length and the nearest word before it labels it as one
    ("PMID: 12345678", "PubMed 12345678", "PubMed ID: 12345678",
    "PMID: 12345678, 23456789") or there is no word before it, as in a bare list of
    PMIDs. A number after any other word is a different identifier or part of prose:
    "OMIM 120130", "MIM# 601492", "bioRxiv 773028", "Corpus ID: 245670502",
    "NBK131811", "Ehlers-Danlos syndrome, ..., type, 1, 130070".
    """
    if number.startswith("0") or len(number) not in _PMID_DIGITS:
        return False
    words = [word.lower() for word in _WORD.findall(preceding_text)]
    if not words:
        return True
    if words[-1] == "id":
        return words[-2:-1] == ["pubmed"]
    return words[-1] in _PUBMED_LABELS


def clean_panel_publications(publications: list[str]) -> PanelPublications:
    """Extract the PMIDs and DOIs from PanelApp publication strings.

    PanelApp's publication fields are free text: mostly PMIDs, bare or labelled and
    often several to a string, but also DOIs, URLs, OMIM numbers, phenotype names
    and references in prose. DOIs are taken from anywhere in a string, then the
    tokens holding them are removed so their digits are not read as PMIDs; see
    `_is_pmid` for which numbers are PMIDs.
    """
    pmids: set[int] = set()
    dois: set[str] = set()
    for entry in publications:
        dois.update(_clean_doi(match) for match in _DOI.findall(entry))
        text = _PUBMED_URL_TOKEN.sub(r" PMID \1 ", _DOI_TOKEN.sub(" ", entry))
        for number in _NUMBER.finditer(text):
            if _is_pmid(number.group(), text[: number.start()]):
                pmids.add(int(number.group()))
    return PanelPublications(pmids=pmids, dois=dois)


def extract_panel_publications(panel_data: dict[str, Any]) -> PanelPublications:
    """Extract all PMIDs and DOIs from panel genes, strs, and regions publications."""
    all_pmids: set[int] = set()
    all_dois: set[str] = set()

    for entity_key in ("genes", "strs", "regions"):
        for entity in panel_data.get(entity_key, []):
            publications = entity.get("publications", [])
            if publications:
                result = clean_panel_publications(publications)
                all_pmids.update(result.pmids)
                all_dois.update(result.dois)

    return PanelPublications(pmids=all_pmids, dois=all_dois)


def extract_gene_publications(panel_data: dict[str, Any], hgnc_id: int) -> PanelPublications:
    """Extract PMIDs and DOIs from the entity matching hgnc_id within a panel.

    Searches the panel's genes, strs, and regions for an entity whose
    gene_data.hgnc_id matches HGNC:<hgnc_id>. Raises ValueError if not found —
    callers should only invoke this after confirming panel membership (e.g. via
    find_gene_panel), so missing-entity is an invariant violation.
    """
    target_id = f"HGNC:{hgnc_id}"
    for entity_key in ("genes", "strs", "regions"):
        for entity in panel_data.get(entity_key, []):
            gene_data = entity.get("gene_data") or {}
            if gene_data.get("hgnc_id") == target_id:
                return clean_panel_publications(entity.get("publications") or [])
    raise ValueError(f"Gene HGNC:{hgnc_id} not found in panel")


def collect_panelapp_gene_publications(
    panel_data: dict[str, Any],
    hgnc_id: int,
    existing_reviews: list[dict[str, Any]],
) -> PanelPublications:
    """Union of gene-entity publications and per-review publications for a gene."""
    entity_pubs = extract_gene_publications(panel_data, hgnc_id)
    pmids: set[int] = set(entity_pubs.pmids)
    dois: set[str] = set(entity_pubs.dois)
    for review in existing_reviews:
        review_pubs = clean_panel_publications(review.get("publications") or [])
        pmids.update(review_pubs.pmids)
        dois.update(review_pubs.dois)
    return PanelPublications(pmids=pmids, dois=dois)


def _parse_hgnc_id(hgnc_id_str: str) -> int:
    """Parse PanelApp HGNC ID string (e.g. 'HGNC:8607') to integer."""
    return int(hgnc_id_str.removeprefix("HGNC:"))


def format_panel_for_prompt(panel_id: int, panel_info: dict[str, Any]) -> str:
    """Format a single panel description for LLM prompt.

    Args:
        panel_id: Panel ID
        panel_info: Panel information dictionary with keys 'name',
                    'relevant_disorders', 'disease_group', 'disease_sub_group'
                    and 'description'

    Returns:
        Formatted panel description string
    """
    formatted = []
    formatted.append(f'<panel id="{panel_id}">')
    formatted.append(f"Name: {panel_info['name']}")
    if panel_info["relevant_disorders"]:
        # PanelApp splits the curator's text at every comma, which separates a term
        # from its ID when curators write "Name, ID; Name, ID". Joining as PanelApp's
        # panel page does restores the text as written.
        formatted.append(f"Relevant disorders: {', '.join(panel_info['relevant_disorders'])}")
    if panel_info["disease_group"]:
        formatted.append(f"Disease group: {panel_info['disease_group']}")
    if panel_info["disease_sub_group"]:
        formatted.append(f"Disease subgroup: {panel_info['disease_sub_group']}")
    if panel_info["description"]:
        formatted.append(f"Description: {panel_info['description']}")
    formatted.append("</panel>")
    return "\n".join(formatted)


DEFAULT_SNAPSHOT_DIR = Path("data/panelapp")
# Set on each panel of a snapshot: when its genes' reviews were fetched (ISO timestamp).
EVALUATIONS_FETCHED_AT = "evaluations_fetched_at"


def snapshot_path(snapshot_dir: Path, panel_date: str) -> Path:
    """The snapshot file of *panel_date* in *snapshot_dir*."""
    return snapshot_dir / f"{panel_date}.json"


class PanelAppClient:
    """Reads the PanelApp snapshot of one date, as built by ``palit panelapp snapshot``."""

    def __init__(self, panel_date: str, snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR):
        """The client of the snapshot of *panel_date* (YYYY-MM-DD), loaded on first use."""
        self.panel_date = panel_date
        self.snapshot_dir = snapshot_dir
        self._panel_data: dict[int, dict[str, Any]] | None = None
        self._gene_entries: dict[tuple[int, str], dict[str, Any]] | None = None

    def _ensure_cache_loaded(self) -> dict[int, dict[str, Any]]:
        """The snapshot's panels keyed by panel ID, loaded on first use."""
        if self._panel_data is None:
            path = snapshot_path(self.snapshot_dir, self.panel_date)
            if not path.exists():
                raise FileNotFoundError(
                    f"No PanelApp snapshot {path}; build it with "
                    f"`palit panelapp snapshot --panel-date {self.panel_date}`"
                )
            logger.info(f"Loading PanelApp snapshot {path}")
            with open(path) as f:
                self._panel_data = {int(k): v for k, v in json.load(f).items()}
        return self._panel_data

    def get_all_panel_data(self) -> dict[int, dict[str, Any]]:
        """Complete data of every panel in the snapshot, keyed by panel ID."""
        return self._ensure_cache_loaded()

    def get_panel_data(self, panel_id: int) -> dict[str, Any]:
        """Get panel data from cache.

        Args:
            panel_id: PanelApp panel ID

        Returns:
            Panel data with genes and metadata
        """
        panel_data_cache = self._ensure_cache_loaded()
        if panel_id not in panel_data_cache:
            raise ValueError(
                f"Panel {panel_id} not found in cached data for date {self.panel_date}"
            )
        return panel_data_cache[panel_id]

    def panel_publications(self, panel_ids: list[int]) -> PanelPublications:
        """The PMIDs and DOIs that the entries on *panel_ids* cite."""
        pmids: set[int] = set()
        dois: set[str] = set()
        for panel_id in panel_ids:
            publications = extract_panel_publications(self.get_panel_data(panel_id))
            pmids |= publications.pmids
            dois |= publications.dois
        return PanelPublications(pmids=pmids, dois=dois)

    def get_target_panels_genes(self, panel_ids: list[int] | None = None) -> PanelGeneData:
        """Get gene data from specified panels, keyed by HGNC ID.

        This includes both genes and STRs (Short Tandem Repeats) from the panels.

        Args:
            panel_ids: List of panel IDs to include. If None, uses TARGET_PANEL_IDS.

        Returns:
            PanelGeneData with confidence, panel mapping, and MoI keyed by HGNC ID.
        """
        if panel_ids is None:
            panel_ids = TARGET_PANEL_IDS

        panel_data_cache = self._ensure_cache_loaded()

        gene_panel_confidence: dict[int, dict[int, int]] = {}

        for panel_id in panel_ids:
            panel_data = panel_data_cache[panel_id]
            all_entities = panel_data.get("genes", []) + panel_data.get("strs", [])

            for entity in all_entities:
                gene_data = entity.get("gene_data", {})
                hgnc_id_str = gene_data.get("hgnc_id")
                if not hgnc_id_str:
                    entity_name = entity.get("entity_name", "unknown")
                    raise ValueError(f"Entity '{entity_name}' in panel {panel_id} has no hgnc_id")

                hgnc_id = _parse_hgnc_id(hgnc_id_str)

                confidence_level = entity.get("confidence_level")
                if confidence_level is None:
                    entity_name = entity.get("entity_name", "unknown")
                    raise ValueError(
                        f"Entity '{entity_name}' in panel {panel_id} has no confidence_level"
                    )

                panels = gene_panel_confidence.setdefault(hgnc_id, {})
                panels[panel_id] = max(panels.get(panel_id, 0), int(confidence_level))

        logger.info(
            f"Loaded {len(gene_panel_confidence)} genes from {len(panel_ids)} target panels"
        )

        return PanelGeneData(panel_ids=panel_ids, gene_panel_confidence=gene_panel_confidence)

    def get_all_panels_genes(self) -> AllPanelsData:
        """Get genes from ALL panels using cached data, keyed by HGNC ID.

        Returns:
            AllPanelsData containing gene-to-panels mapping and panel names
        """
        panel_data_cache = self._ensure_cache_loaded()

        gene_to_panels: dict[int, set[int]] = {}
        panel_names: dict[int, str] = {}

        for panel_id, panel_data in panel_data_cache.items():
            panel_names[panel_id] = panel_data.get("name", f"Panel {panel_id}")

            all_entities = panel_data.get("genes", []) + panel_data.get("strs", [])

            for entity in all_entities:
                gene_data = entity.get("gene_data", {})
                hgnc_id_str = gene_data.get("hgnc_id")
                if not hgnc_id_str:
                    continue  # Skip entities without HGNC ID in non-target panels
                hgnc_id = _parse_hgnc_id(hgnc_id_str)
                gene_to_panels.setdefault(hgnc_id, set()).add(panel_id)

        return AllPanelsData(
            gene_to_panels=gene_to_panels,
            panel_names=panel_names,
        )

    def get_all_panel_descriptions(self) -> dict[int, dict[str, Any]]:
        """Get descriptions of all cached panels, excluding super-panels.

        Super-panels are aggregations of other panels and should be excluded
        from panel matching to avoid redundancy.

        Returns:
            Dict mapping panel_id to panel information:
            {
                137: {
                    "name": "Mendeliome",
                    "description": "Panel description text",
                    "relevant_disorders": ["Rare genetic disorders"],
                    "disease_group": "Multiple systems",
                    "disease_sub_group": "General genetic disorders"
                },
                ...
            }
        """
        panel_data_cache = self._ensure_cache_loaded()

        panels = {}
        excluded_count = 0

        for panel_id, panel_data in panel_data_cache.items():
            # Skip super-panels (panels with non-empty child_panel_ids)
            child_panel_ids = panel_data.get("child_panel_ids", [])
            if child_panel_ids:
                excluded_count += 1
                logger.debug(f"Excluding super-panel {panel_id}: {panel_data.get('name')}")
                continue

            panels[panel_id] = {
                "name": panel_data["name"],
                "description": panel_data["description"],
                "relevant_disorders": panel_data["relevant_disorders"],
                "disease_group": panel_data["disease_group"],
                "disease_sub_group": panel_data["disease_sub_group"],
            }

        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} super-panels from panel descriptions")

        return panels

    def _gene_entry(self, panel_id: int, hgnc_id: int) -> dict[str, Any] | None:
        """The panel's gene entry for the gene; None if it holds the gene as no gene entry."""
        if self._gene_entries is None:
            self._gene_entries = {
                (pid, entry["gene_data"]["hgnc_id"]): entry
                for pid, panel in self._ensure_cache_loaded().items()
                for entry in panel["genes"]
            }
        return self._gene_entries.get((panel_id, f"HGNC:{hgnc_id}"))

    def holds_gene(self, panel_id: int, hgnc_id: int) -> bool:
        """Whether the panel holds the gene as a gene entry (not only as a repeat expansion)."""
        return self._gene_entry(panel_id, hgnc_id) is not None

    def gene_reviews(self, panel_id: int, hgnc_id: int) -> list[dict[str, Any]] | None:
        """The gene's reviews on the panel, with comments, most recent first; None when
        PanelApp could not serve them (see :mod:`palit.panelapp_snapshot`).

        They are the reviews as of the panel's ``reviews_fetched_at``, not of the
        panel date. Raises ValueError if the panel holds no gene entry for the gene,
        and KeyError if the snapshot was built without reviews.
        """
        entry = self._gene_entry(panel_id, hgnc_id)
        if entry is None:
            raise ValueError(f"Panel {panel_id} does not hold HGNC:{hgnc_id}")
        if "evaluations" not in entry:
            raise KeyError(
                f"The {self.panel_date} snapshot has no reviews; add them with "
                f"`palit panelapp add-reviews --panel-date {self.panel_date}`"
            )
        reviews: list[dict[str, Any]] | None = entry["evaluations"]
        return reviews

    def reviews_fetched_at(self, panel_id: int) -> str:
        """When the panel's reviews were fetched (ISO timestamp)."""
        fetched_at: str = self.get_panel_data(panel_id)[EVALUATIONS_FETCHED_AT]
        return fetched_at
