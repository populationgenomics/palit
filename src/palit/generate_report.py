#!/usr/bin/env python3
"""Generate gene-centric HTML assessment reports from aggregate assessments."""

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import markdown
import nh3
import typer
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from palit.docling import parse_bbox_mapping_from_json
from palit.hgnc import HgncResolver
from palit.panelapp_client import (
    PanelAppClient,
    PanelGeneData,
    get_current_panel_publications,
)
from palit.panelapp_integration import (
    MONDO_CATEGORIES,
    PANELAPP_CRITERIA,
    PANELAPP_MOI_TO_ENUM,
    calculate_gene_rating,
    count_families_by_moi,
    derive_aggregate_moi,
    panelapp_confidence_to_color,
    prepare_prefill_data,
)
from palit.papers import doi_to_path, is_preprint
from palit.relevance import compute_relevance_majority_vote

logger = logging.getLogger(__name__)

# Variant frequency thresholds for flagging based on inheritance mode
GNOMAD_HET_THRESHOLD = 30  # Monoallelic (dominant) - heterozygote count
GNOMAD_HOM_THRESHOLD = 15  # Biallelic (recessive) - homozygote count
GNOMAD_HEMI_THRESHOLD = 30  # X-linked - hemizygote count

# Minimum families required to highlight an MoI expansion
MIN_FAMILIES_FOR_MOI_EXPANSION = 2


def compare_moi(
    existing_moi: str | None,
    new_moi: str,
    moi_family_counts: dict[str, int] | None = None,
    min_families_for_expansion: int = MIN_FAMILIES_FOR_MOI_EXPANSION,
) -> dict[str, Any]:
    """Compare existing (PanelApp) and new (evidence-based) mode of inheritance.

    Args:
        existing_moi: Current MoI from PanelApp (mapped to enum)
        new_moi: New MoI from evidence (already in enum format)
        moi_family_counts: Family counts per inheritance mode (from count_families_by_moi)
        min_families_for_expansion: Minimum families required to highlight an expansion

    Returns:
        Dict with:
        - status: "same", "expansion", "contradiction"
        - highlighted: Whether to surface this prominently to curators
        - reason: Audit trail explaining the highlighting decision
        - message: Human-readable explanation
        - icon: Emoji for visual indicator
        - css_class: CSS class for styling
    """
    # Normalize for comparison
    existing = existing_moi.replace("_", " ").lower() if existing_moi else None
    new = new_moi.replace("_", " ").lower() if new_moi else None

    # Same mode
    if existing == new:
        return {
            "status": "same",
            "highlighted": False,
            "reason": "",
            "message": "",
            "icon": "",
            "css_class": "",
        }

    # Existing was unknown, now we have information - this is an expansion
    if existing in ["other", "nr", None] and new not in ["other", "nr"]:
        return {
            "status": "expansion",
            "highlighted": True,
            "reason": "New MoI information where none existed before",
            "message": "New evidence provides mode of inheritance information previously unknown",
            "icon": "➕",  # noqa: RUF001
            "css_class": "moi-expansion",
        }

    # Check for expansions (adding inheritance modes)
    if existing in ["monoallelic", "biallelic"] and new == "monoallelic and biallelic":
        # Determine which mode is being "added"
        added_mode = "Biallelic" if existing == "monoallelic" else "Monoallelic"
        added_count = (moi_family_counts or {}).get(added_mode, 0)

        if added_count >= min_families_for_expansion:
            return {
                "status": "expansion",
                "highlighted": True,
                "reason": f"{added_count} families with {added_mode.lower()} inheritance (threshold: {min_families_for_expansion})",
                "message": f"Evidence supports both modes ({added_count} families with {added_mode.lower()})",
                "icon": "➕",  # noqa: RUF001
                "css_class": "moi-expansion",
            }
        else:
            return {
                "status": "expansion",
                "highlighted": False,
                "reason": f"Only {added_count} family/families with {added_mode.lower()} inheritance (threshold: {min_families_for_expansion})",
                "message": f"Weak evidence for {added_mode.lower()} inheritance ({added_count} family/families)",
                "icon": "",
                "css_class": "",
            }

    # Everything else is a contradiction - always highlight
    return {
        "status": "contradiction",
        "highlighted": True,
        "reason": "MoI differs from existing classification",
        "message": "New evidence suggests a different mode of inheritance than currently recorded",
        "icon": "⚠️",
        "css_class": "moi-warning",
    }


@dataclass(frozen=True)
class CitationLink:
    """A resolved link from an assessment citation to an annotated PDF page."""

    paper_id: str  # AuthorYear short ID
    doi: str
    page: int


@dataclass
class VariantFrequency:
    """Variant frequency information from gnomAD."""

    variant_id: str  # gnomAD pseudo-VCF format
    doi: str
    box_id: int
    hgvs_c: str | None
    hgvs_p: str | None
    original_text: str
    gnomad_ac: int | None
    gnomad_an: int | None
    gnomad_hom: int | None  # Number of homozygotes
    gnomad_het: int | None  # Number of heterozygotes (computed)
    gnomad_hemi: int | None  # Number of hemizygotes
    gnomad_faf95_popmax: float | None  # FAF95 popmax value
    gnomad_faf95_popmax_population: str | None  # Population name for FAF95
    gnomad_link: str  # Direct link to gnomAD
    citation_page: int | None  # PDF page number for citation
    gnomad_not_found: bool  # True if variant not found in gnomAD
    gnomad_error: str | None  # Error message if gnomAD lookup failed


@dataclass
class DetailedPaper:
    """Complete paper information for detailed display."""

    doi: str
    title: str
    abstract: str | None
    authors: str | None
    journal: str | None
    source_date: str | None
    source_type: str | None
    source_details: str | None
    relevance_assessment: dict[str, Any] | None
    evidence_extraction: dict[str, Any] | None
    citation_pages: dict[int, int] | None  # box_id -> page number
    preprint: bool = False
    pmid: int | None = None  # For PubMed display links
    paper_id: str = ""  # AuthorYear short ID (computed per gene context)
    paper_gene_symbol: str | None = None
    variant_frequencies: list[VariantFrequency] = field(default_factory=list)
    filtered_reason: str | None = None  # Set when paper was excluded from assessment


@dataclass
class PanelMatch:
    """Represents a panel match for a gene."""

    panel_id: int
    panel_name: str
    rationale: str


@dataclass
class GeneAssessment:
    """Represents a gene's assessment (panel-agnostic)."""

    hgnc_id: int
    hgnc_symbol: str
    assessment_json: dict[str, Any]
    existing_rating: int | None  # Current confidence level in panel (for known genes)
    existing_moi: str | None  # Current mode of inheritance from panel (mapped to enum)
    new_moi: str  # Derived aggregate MoI from disease_entities
    new_moi_details: str  # Derived aggregate MoI details from disease_entities
    moi_comparison: dict[str, Any] | None  # Precomputed MoI comparison result
    new_rating: int  # Calculated confidence level: 1 (RED), 2 (AMBER), 3 (GREEN)
    contributing_papers: list[DetailedPaper]
    variant_frequencies: list[VariantFrequency]  # Variants with frequency data
    missing_panels: list[PanelMatch]  # Suggested panels gene is not in
    existing_panels: list[PanelMatch]  # Panels gene is already in
    prefill_json: str  # HTML-escaped JSON for data-prefill attribute


@dataclass
class GeneAssessmentResults:
    """Results from loading and sorting gene assessments."""

    novel_genes: list[GeneAssessment]
    known_genes: list[GeneAssessment]
    target_panel_data: PanelGeneData


@dataclass
class PanelValidationResult:
    """Panel publication validation results."""

    total_panel_papers: int
    panel_papers_in_db: int
    false_negatives: list[DetailedPaper]
    true_positives: list[DetailedPaper]
    sensitivity_pct: float  # TP / (TP + FN) - ability to identify relevant papers


@dataclass
class ComprehensiveStats:
    """All statistics for the report."""

    # Gene assessment stats
    total_genes_assessed: int
    novel_genes_count: int
    known_genes_upgraded: int
    total_contributing_papers: int

    # Panel validation stats
    total_panel_papers: int
    panel_papers_in_db: int
    validation_sensitivity_pct: float
    false_negatives_count: int
    true_positives_count: int

    # Source breakdown
    initial_papers: int
    expansion_papers: int

    # Panel suggestions
    total_panel_suggestions: int

    # MoI change stats
    moi_expansions_count: int
    moi_contradictions_count: int

    # Preprint stats
    preprints_relevant: int
    papers_filtered: int

    # Relevance assessment unanimity stats
    non_unanimous_pct: float


@dataclass
class NovelGeneCategories:
    """Categorized novel genes by potential rating."""

    green: list[GeneAssessment]
    amber: list[GeneAssessment]
    red: list[GeneAssessment]

    @property
    def total(self) -> int:
        """Total number of novel genes."""
        return len(self.green) + len(self.amber) + len(self.red)


@dataclass
class KnownGeneCategories:
    """Categorized known genes by upgrade type."""

    red_to_green: list[GeneAssessment]
    amber_to_green: list[GeneAssessment]
    red_to_amber: list[GeneAssessment]
    no_change_with_moi: list[GeneAssessment]
    no_change_without_moi: list[GeneAssessment]

    @property
    def total(self) -> int:
        """Total number of known genes."""
        return (
            len(self.red_to_green)
            + len(self.amber_to_green)
            + len(self.red_to_amber)
            + len(self.no_change_with_moi)
            + len(self.no_change_without_moi)
        )


app = typer.Typer(help="Generate gene-centric HTML reports from aggregate assessments")


def format_gene_with_aliases(gene_assessment: GeneAssessment) -> str:
    """Format gene symbol with alias information if aliases were used.

    Args:
        gene_assessment: GeneAssessment object with gene_symbol and contributing papers

    Returns:
        Formatted string like "PRKN (via alias: PARK2)" or just "PRKN"
    """
    hgnc_symbol: str = gene_assessment.hgnc_symbol

    # Collect all paper gene symbols from contributing papers
    paper_gene_symbols = set()
    for paper in gene_assessment.contributing_papers:
        if paper.paper_gene_symbol:
            paper_gene_symbols.add(paper.paper_gene_symbol)

    # Remove the HGNC symbol itself
    paper_gene_symbols.discard(hgnc_symbol)

    if paper_gene_symbols:
        aliases = sorted(paper_gene_symbols)
        alias_str = ", ".join(aliases)
        return f"{hgnc_symbol} <span class='gene-alias'>(via alias: {alias_str})</span>"
    else:
        return hgnc_symbol


POPULATION_NAMES = {
    "afr": "African/African American",
    "ami": "Amish",
    "amr": "Admixed American",
    "asj": "Ashkenazi Jewish",
    "eas": "East Asian",
    "fin": "Finnish",
    "mid": "Middle Eastern",
    "nfe": "European (non-Finnish)",
    "sas": "South Asian",
}


def _create_variant_frequency_from_db_row(
    variant_id: str,
    doi: str,
    box_id: int,
    normalization: dict[str, Any],
    gnomad: dict[str, Any],
    citation_page: int | None,
) -> VariantFrequency:
    """Create a VariantFrequency object from database row data."""
    gnomad_not_found = "variant_not_found" in gnomad
    gnomad_error = gnomad.get("error")
    gnomad_ac = None
    gnomad_an = None
    gnomad_hom = None
    gnomad_het = None
    gnomad_hemi = None
    gnomad_faf95_popmax = None
    gnomad_faf95_popmax_population = None

    if not gnomad_not_found and gnomad_error is None:
        variant_data = gnomad.get("variant")
        if variant_data and variant_data.get("joint"):
            joint_data = variant_data["joint"]
            gnomad_ac = joint_data.get("ac")
            gnomad_an = joint_data.get("an")
            gnomad_hom = joint_data.get("homozygote_count")
            gnomad_hemi = joint_data.get("hemizygote_count")

            # Calculate heterozygotes: het = ac - (2 * hom) - hemi
            if gnomad_ac is not None:
                gnomad_het = gnomad_ac
                if gnomad_hom is not None:
                    gnomad_het -= 2 * gnomad_hom
                if gnomad_hemi is not None:
                    gnomad_het -= gnomad_hemi

            # Extract FAF95 data
            faf95_data = joint_data.get("faf95")
            if faf95_data:
                gnomad_faf95_popmax = faf95_data.get("popmax")
                popmax_pop = faf95_data.get("popmax_population")
                gnomad_faf95_popmax_population = POPULATION_NAMES.get(popmax_pop, popmax_pop)

    return VariantFrequency(
        variant_id=variant_id,
        doi=doi,
        box_id=box_id,
        hgvs_c=normalization.get("hgvs_c"),
        hgvs_p=normalization.get("hgvs_p"),
        original_text=normalization.get("original_text", ""),
        gnomad_ac=gnomad_ac,
        gnomad_an=gnomad_an,
        gnomad_hom=gnomad_hom,
        gnomad_het=gnomad_het,
        gnomad_hemi=gnomad_hemi,
        gnomad_faf95_popmax=gnomad_faf95_popmax,
        gnomad_faf95_popmax_population=gnomad_faf95_popmax_population,
        gnomad_link=f"https://gnomad.broadinstitute.org/variant/{variant_id}?dataset=gnomad_r4",
        citation_page=citation_page,
        gnomad_not_found=gnomad_not_found,
        gnomad_error=gnomad_error,
    )


def load_variant_frequencies_for_gene(
    cursor: sqlite3.Cursor, hgnc_id: int, contributing_papers: list[DetailedPaper]
) -> list[VariantFrequency]:
    """Load variant frequency information for a specific gene.

    Args:
        cursor: Database cursor
        hgnc_id: HGNC ID of the gene to load variants for
        contributing_papers: Papers contributing to this gene assessment

    Returns:
        List of VariantFrequency objects with gnomAD data and citation info
    """
    # Create lookup for bbox mappings from contributing papers
    bbox_mappings = {}
    for paper in contributing_papers:
        if paper.citation_pages:
            bbox_mappings[paper.doi] = paper.citation_pages

    # Load variant frequencies from database
    cursor.execute(
        """
        SELECT
            vf.variant_id,
            vf.paper_doi,
            vf.box_id,
            vf.normalization,
            vf.gnomad
        FROM variant_frequencies vf
        WHERE vf.hgnc_id = ?
        ORDER BY vf.paper_doi DESC, vf.variant_id
    """,
        (hgnc_id,),
    )

    variant_frequencies = []
    for row in cursor.fetchall():
        variant_id = row["variant_id"]
        doi = row["paper_doi"]
        box_id = row["box_id"]

        # Parse JSON fields
        try:
            normalization = json.loads(row["normalization"])
            gnomad = json.loads(row["gnomad"])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for variant {variant_id}: {e}")
            continue

        # Get citation page from bbox mapping
        citation_page = None
        if doi in bbox_mappings and box_id in bbox_mappings[doi]:
            citation_page = bbox_mappings[doi][box_id]

        # Create variant frequency object using helper
        variant_freq = _create_variant_frequency_from_db_row(
            variant_id=variant_id,
            doi=doi,
            box_id=box_id,
            normalization=normalization,
            gnomad=gnomad,
            citation_page=citation_page,
        )
        variant_frequencies.append(variant_freq)

    return variant_frequencies


def load_variant_frequencies_for_paper(
    cursor: sqlite3.Cursor, doi: str, citation_pages: dict[int, int] | None
) -> list[VariantFrequency]:
    """Load variant frequency information for a specific paper.

    Args:
        cursor: Database cursor
        doi: Paper DOI to load variants for
        citation_pages: Bbox mapping for citation page lookups

    Returns:
        List of VariantFrequency objects for this paper
    """
    # Load variant frequencies from database for this paper
    cursor.execute(
        """
        SELECT
            vf.variant_id,
            vf.box_id,
            vf.normalization,
            vf.gnomad
        FROM variant_frequencies vf
        WHERE vf.paper_doi = ?
        ORDER BY vf.hgnc_id, vf.variant_id
    """,
        (doi,),
    )

    variant_frequencies = []
    for row in cursor.fetchall():
        variant_id = row["variant_id"]
        box_id = row["box_id"]

        # Parse JSON fields
        try:
            normalization = json.loads(row["normalization"])
            gnomad = json.loads(row["gnomad"])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for variant {variant_id} in paper {doi}: {e}")
            continue

        # Get citation page from bbox mapping
        citation_page = None
        if citation_pages and box_id in citation_pages:
            citation_page = citation_pages[box_id]

        # Create variant frequency object using helper
        variant_freq = _create_variant_frequency_from_db_row(
            variant_id=variant_id,
            doi=doi,
            box_id=box_id,
            normalization=normalization,
            gnomad=gnomad,
            citation_page=citation_page,
        )
        variant_frequencies.append(variant_freq)

    return variant_frequencies


def load_gene_assessments(
    db_path: Path,
    panel_date: str,
    hgnc_resolver: HgncResolver,
    target_panel_ids: list[int] | None = None,
) -> GeneAssessmentResults:
    """Load assessments from gene_assessments table for genes from initial papers only.

    Args:
        db_path: Path to the database
        panel_date: Date (YYYY-MM-DD) to check panel membership at
        target_panel_ids: List of panel IDs to use for novelty detection. If None, uses TARGET_PANEL_IDS.

    Returns:
        GeneAssessmentResults with sorted novel and known genes plus panel data
    """
    logger.info(f"Loading gene assessments from {db_path}...")

    # Create client for the specified date and fetch panel data
    panelapp_client = PanelAppClient(panel_date)
    target_panel_data = panelapp_client.get_target_panels_genes(target_panel_ids)
    all_panels_data = panelapp_client.get_all_panels_genes()
    logger.info(
        f"Loaded {len(target_panel_data.gene_confidence)} genes from target panels, {len(all_panels_data.gene_to_panels)} genes from all panels"
    )

    novel_genes = []
    known_genes = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all genes from gene_assessments (already filtered to working set during aggregate-assessment)
        cursor.execute("""
            SELECT
                hgnc_id,
                assessment_json,
                paper_id_mapping,
                matched_panels_json,
                filtered_papers_json
            FROM gene_assessments
            ORDER BY hgnc_id
        """)

        for row in cursor.fetchall():
            hgnc_id: int = row["hgnc_id"]
            assessment_json = json.loads(row["assessment_json"])
            paper_id_to_doi: dict[str, str] = json.loads(row["paper_id_mapping"])
            doi_to_paper_id = {doi: pid for pid, doi in paper_id_to_doi.items()}
            matched_panels = json.loads(row["matched_panels_json"] or "[]")
            filtered_doi_reasons: dict[str, str] = {
                fp["doi"]: fp["reason"] for fp in json.loads(row["filtered_papers_json"] or "[]")
            }

            # Calculate rating from assessment
            new_rating = calculate_gene_rating(assessment_json)

            # Get current panel membership from target panels only (for novel/known determination)
            # List preserves order from target_panel_ids
            gene_panels = target_panel_data.gene_panel_mapping.get(hgnc_id, set())
            target_panel_membership = [
                pid for pid in target_panel_data.panel_ids if pid in gene_panels
            ]

            # Separate matched panels into missing and existing panels
            missing_panels = []
            existing_panels = []

            for match in matched_panels:
                panel_id = match["panel_id"]
                rationale = match["rationale"]

                # Check if gene is currently in this matched panel (from all panels data)
                current_panels = all_panels_data.gene_to_panels.get(hgnc_id, set())
                panel_name = all_panels_data.panel_names.get(panel_id)
                if panel_name is None:
                    logger.warning(
                        f"Panel {panel_id} no longer exists in PanelApp, skipping match for HGNC:{hgnc_id}"
                    )
                    continue

                if panel_id in current_panels:
                    existing_panels.append(
                        PanelMatch(
                            panel_id=panel_id,
                            panel_name=panel_name,
                            rationale=rationale,
                        )
                    )
                else:
                    missing_panels.append(
                        PanelMatch(
                            panel_id=panel_id,
                            panel_name=panel_name,
                            rationale=rationale,
                        )
                    )

            # Sort alphabetically
            missing_panels.sort(key=lambda x: x.panel_name)
            existing_panels.sort(key=lambda x: x.panel_name)

            # Get contributing papers for this gene with full details (all sources)
            cursor.execute(
                """
                SELECT DISTINCT
                    p.doi,
                    p.title,
                    p.abstract,
                    p.authors,
                    p.journal,
                    p.source_date,
                    p.source_type,
                    p.source_details,
                    p.pmid,
                    p.relevance_assessment_json,
                    p.evidence_extraction_json,
                    p.bbox_mapping,
                    gm.paper_gene_symbol
                FROM papers p
                JOIN gene_mentions gm ON p.doi = gm.paper_doi
                WHERE gm.hgnc_id = ?
                AND p.evidence_extraction_json IS NOT NULL
                ORDER BY p.source_date DESC, p.doi DESC
            """,
                (hgnc_id,),
            )

            contributing_papers = []
            for paper_row in cursor.fetchall():
                # Parse JSON fields safely
                relevance_assessment = None
                if paper_row["relevance_assessment_json"]:
                    try:
                        relevance_assessment = compute_relevance_majority_vote(
                            json.loads(paper_row["relevance_assessment_json"])
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse relevance assessment for DOI {paper_row['doi']}"
                        )

                evidence_extraction = None
                if paper_row["evidence_extraction_json"]:
                    try:
                        evidence_extraction = json.loads(paper_row["evidence_extraction_json"])
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse evidence extraction for DOI {paper_row['doi']}"
                        )

                # Skip papers that don't have evidence extraction for this specific gene
                # (they may mention the gene but only have evidence for other genes)
                if evidence_extraction:
                    gene_evals = evidence_extraction.get("gene_evaluations", [])
                    has_gene_evidence = any(g.get("hgnc_id") == hgnc_id for g in gene_evals)
                    if not has_gene_evidence:
                        continue

                citation_pages = None
                if paper_row["bbox_mapping"]:
                    try:
                        bbox_data = parse_bbox_mapping_from_json(paper_row["bbox_mapping"])
                        citation_pages = {
                            box_id: info["page"] for box_id, info in bbox_data.items()
                        }
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse bbox mapping for DOI {paper_row['doi']}: {e}"
                        )

                # Load variant frequencies for this paper
                paper_variant_frequencies = load_variant_frequencies_for_paper(
                    cursor, paper_row["doi"], citation_pages
                )

                doi = paper_row["doi"]
                detailed_paper = DetailedPaper(
                    doi=doi,
                    title=paper_row["title"] or "Unknown Title",
                    abstract=paper_row["abstract"],
                    authors=paper_row["authors"],
                    journal=paper_row["journal"],
                    source_date=paper_row["source_date"],
                    source_type=paper_row["source_type"],
                    source_details=paper_row["source_details"],
                    relevance_assessment=relevance_assessment,
                    evidence_extraction=evidence_extraction,
                    citation_pages=citation_pages,
                    preprint=is_preprint(paper_row["journal"], paper_row["pmid"]),
                    pmid=paper_row["pmid"],
                    paper_gene_symbol=paper_row["paper_gene_symbol"],
                    variant_frequencies=paper_variant_frequencies,
                    filtered_reason=filtered_doi_reasons.get(doi),
                )
                contributing_papers.append(detailed_paper)

            # Assign AuthorYear paper IDs from stored mapping
            # Filtered papers aren't in the mapping (excluded from assessment)
            for paper in contributing_papers:
                paper_id = doi_to_paper_id.get(paper.doi)
                if paper_id:
                    paper.paper_id = paper_id

            # Load variant frequencies for this gene
            variant_frequencies = load_variant_frequencies_for_gene(
                cursor, hgnc_id, contributing_papers
            )

            # Gene is novel if not in any target panel
            is_novel = not target_panel_membership

            # Get existing rating and MoI for known genes
            existing_rating = None
            existing_moi = None
            if not is_novel:
                existing_rating = target_panel_data.gene_confidence[hgnc_id]
                # Map PanelApp MoI to our enum
                existing_moi = PANELAPP_MOI_TO_ENUM[target_panel_data.gene_moi[hgnc_id]]

            # Compute MoI comparison (precompute for sorting and display)
            disease_entities = assessment_json.get("disease_entities", [])
            new_moi, new_moi_details = derive_aggregate_moi(disease_entities)
            moi_family_counts = count_families_by_moi(disease_entities)

            moi_comparison = None
            if existing_moi and new_moi:
                comparison = compare_moi(existing_moi, new_moi, moi_family_counts=moi_family_counts)
                if comparison["status"] != "same":
                    moi_comparison = {
                        "existing": existing_moi,
                        "new": new_moi,
                        "new_details": new_moi_details,
                        "status": comparison["status"],
                        "highlighted": comparison["highlighted"],
                        "reason": comparison["reason"],
                        "message": comparison["message"],
                        "icon": comparison["icon"],
                        "css_class": comparison["css_class"],
                    }

            # Compute prefill data
            if is_novel:
                prefill_panel_id = target_panel_data.panel_ids[0]
                prefill_form_type = "add"
            else:
                prefill_panel_id = target_panel_membership[0]
                prefill_form_type = "review"

            prefill_data = prepare_prefill_data(
                hgnc_id=hgnc_id,
                assessment_json=assessment_json,
                form_type=prefill_form_type,
                panel_id=prefill_panel_id,
                cited_papers=[(p.doi, p.pmid) for p in contributing_papers],
            )
            prefill_json = json.dumps(asdict(prefill_data))

            # Create assessment
            assessment = GeneAssessment(
                hgnc_id=hgnc_id,
                hgnc_symbol=hgnc_resolver.get_symbol(hgnc_id),
                assessment_json=assessment_json,
                existing_rating=existing_rating,
                existing_moi=existing_moi,
                new_moi=new_moi,
                new_moi_details=new_moi_details,
                moi_comparison=moi_comparison,
                new_rating=new_rating,
                contributing_papers=contributing_papers,
                variant_frequencies=variant_frequencies,
                missing_panels=missing_panels,
                existing_panels=existing_panels,
                prefill_json=prefill_json,
            )

            # Categorize by panel membership
            if is_novel:
                novel_genes.append(assessment)
            else:
                known_genes.append(assessment)

    # Sort novel genes: by new rating (highest first), then highlighted MoI changes first, then gene name
    novel_genes.sort(
        key=lambda g: (
            -g.new_rating,  # Negative for descending: 3 (GREEN), 2 (AMBER), 1 (RED)
            0
            if (g.moi_comparison and g.moi_comparison.get("highlighted"))
            else 1,  # Highlighted MoI changes first
            g.hgnc_symbol,
        )
    )

    # Sort known genes: by existing rating (lowest first), then new rating (highest first), then highlighted MoI changes first, then gene name
    def known_sort_key(g: GeneAssessment) -> tuple:
        # Get confidence level from target panels
        target_confidence = target_panel_data.gene_confidence.get(g.hgnc_id)

        if target_confidence is None:
            raise ValueError(
                f"Known gene {g.hgnc_symbol} (HGNC:{g.hgnc_id}) has no confidence level in target panels"
            )

        # Sort by existing confidence (ascending), new rating (descending), highlighted MoI (first), gene symbol
        has_highlighted_moi = 0 if (g.moi_comparison and g.moi_comparison.get("highlighted")) else 1
        return (target_confidence, -g.new_rating, has_highlighted_moi, g.hgnc_symbol)

    known_genes.sort(key=known_sort_key)

    logger.info(f"Loaded {len(novel_genes)} novel genes, {len(known_genes)} known genes")

    return GeneAssessmentResults(
        novel_genes=novel_genes,
        known_genes=known_genes,
        target_panel_data=target_panel_data,
    )


def load_panel_publications_validation(
    db_path: Path, target_panel_ids: list[int] | None
) -> PanelValidationResult:
    """Load panel publication validation using source_type filtering.

    Compares LLM assessments against publications mentioned in PanelApp panels.
    Uses source_type = 'initial' to avoid date range issues with expansion.

    Args:
        db_path: Path to the database
        target_panel_ids: List of panel IDs to use for validation. If None, uses TARGET_PANEL_IDS.
    """
    logger.info("Loading panel publications validation...")

    # Get panel publications from PanelApp (using current/latest data)
    panel_pubs = get_current_panel_publications(target_panel_ids)
    total_panel_refs = len(panel_pubs.pmids) + len(panel_pubs.dois)
    logger.info(
        f"Found {len(panel_pubs.pmids)} PMIDs and {len(panel_pubs.dois)} DOIs in PanelApp panels"
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if not total_panel_refs:
            return PanelValidationResult(
                total_panel_papers=0,
                panel_papers_in_db=0,
                false_negatives=[],
                true_positives=[],
                sensitivity_pct=0.0,
            )

        # Match panel publications against our DB by both PMID and DOI
        chunk_size = 999  # SQLite parameter limit
        seen_dois: set[str] = set()
        all_panel_papers: list[sqlite3.Row] = []

        def _collect(rows: list[sqlite3.Row]) -> None:
            for row in rows:
                doi = row["doi"]
                if doi not in seen_dois:
                    seen_dois.add(doi)
                    all_panel_papers.append(row)

        _panel_cols = """p.doi, p.pmid, p.title, p.abstract, p.authors, p.journal,
            p.source_date, p.source_type, p.source_details,
            p.relevance_assessment_json, p.evidence_extraction_json, p.bbox_mapping"""

        # Match PMIDs
        pmids_list = list(panel_pubs.pmids)
        for i in range(0, len(pmids_list), chunk_size):
            pmid_chunk = pmids_list[i : i + chunk_size]
            placeholders = ",".join("?" * len(pmid_chunk))
            cursor.execute(
                f"""
                SELECT {_panel_cols} FROM papers p
                WHERE p.pmid IN ({placeholders})
                AND p.source_type = 'initial'
                AND p.relevance_assessment_json IS NOT NULL
            """,
                pmid_chunk,
            )
            _collect(cursor.fetchall())

        # Match DOIs
        dois_list = list(panel_pubs.dois)
        for i in range(0, len(dois_list), chunk_size):
            doi_chunk = dois_list[i : i + chunk_size]
            placeholders = ",".join("?" * len(doi_chunk))
            cursor.execute(
                f"""
                SELECT {_panel_cols} FROM papers p
                WHERE p.doi IN ({placeholders})
                AND p.source_type = 'initial'
                AND p.relevance_assessment_json IS NOT NULL
            """,
                doi_chunk,
            )
            _collect(cursor.fetchall())

        logger.info(f"Found {len(all_panel_papers)} panel papers in initial set with assessments")

        # Convert to DetailedPaper objects and categorize
        false_negatives = []
        true_positives = []

        for row in all_panel_papers:
            relevance_assessment = None
            try:
                relevance_assessment = compute_relevance_majority_vote(
                    json.loads(row["relevance_assessment_json"])
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse relevance assessment for panel DOI {row['doi']}")
                continue

            evidence_extraction = None
            if row["evidence_extraction_json"]:
                try:
                    evidence_extraction = json.loads(row["evidence_extraction_json"])
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse evidence extraction for panel DOI {row['doi']}"
                    )

            citation_pages = None
            if row["bbox_mapping"]:
                try:
                    bbox_data = parse_bbox_mapping_from_json(row["bbox_mapping"])
                    citation_pages = {box_id: info["page"] for box_id, info in bbox_data.items()}
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse bbox mapping for panel DOI {row['doi']}")

            detailed_paper = DetailedPaper(
                doi=row["doi"],
                title=row["title"] or "Unknown Title",
                abstract=row["abstract"],
                authors=row["authors"],
                journal=row["journal"],
                source_date=row["source_date"],
                source_type=row["source_type"],
                source_details=row["source_details"],
                relevance_assessment=relevance_assessment,
                evidence_extraction=evidence_extraction,
                citation_pages=citation_pages,
                preprint=is_preprint(row["journal"], row["pmid"]),
                pmid=row["pmid"],
            )

            # Categorize by LLM assessment
            if relevance_assessment["relevant"]:
                true_positives.append(detailed_paper)
            else:
                false_negatives.append(detailed_paper)

        # Calculate metrics
        total_assessed = len(false_negatives) + len(true_positives)
        tp_count = len(true_positives)
        fn_count = len(false_negatives)

        # Sensitivity = TP / (TP + FN) - ability to identify relevant papers
        sensitivity_pct = (tp_count / total_assessed * 100) if total_assessed > 0 else 0.0

        logger.info(f"Panel validation: {tp_count} true positives, {fn_count} false negatives")
        logger.info(f"Sensitivity: {sensitivity_pct:.1f}%")

        return PanelValidationResult(
            total_panel_papers=total_panel_refs,
            panel_papers_in_db=total_assessed,
            false_negatives=false_negatives,
            true_positives=true_positives,
            sensitivity_pct=sensitivity_pct,
        )


def load_low_confidence_irrelevant_papers(db_path: Path) -> list[DetailedPaper]:
    """Load papers marked as not relevant with low confidence for manual review."""
    logger.info("Loading low-confidence irrelevant papers...")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all papers with relevance assessments (filter in Python using majority vote)
        cursor.execute("""
            SELECT
                p.doi,
                p.pmid,
                p.title,
                p.abstract,
                p.authors,
                p.journal,
                p.source_date,
                p.source_type,
                p.source_details,
                p.relevance_assessment_json,
                p.evidence_extraction_json,
                p.bbox_mapping
            FROM papers p
            WHERE p.relevance_assessment_json IS NOT NULL
            ORDER BY p.source_date DESC, p.doi DESC
        """)

        low_confidence_papers = []
        for row in cursor.fetchall():
            relevance_assessment = None
            try:
                relevance_assessment = compute_relevance_majority_vote(
                    json.loads(row["relevance_assessment_json"])
                )
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse relevance assessment for DOI {row['doi']}")
                continue

            # Filter: only include if majority is NOT relevant AND LOW confidence
            if not relevance_assessment["relevant"] and relevance_assessment["confidence"] == "LOW":
                evidence_extraction = None
                if row["evidence_extraction_json"]:
                    try:
                        evidence_extraction = json.loads(row["evidence_extraction_json"])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse evidence extraction for DOI {row['doi']}")

                citation_pages = None
                if row["bbox_mapping"]:
                    try:
                        bbox_data = parse_bbox_mapping_from_json(row["bbox_mapping"])
                        citation_pages = {
                            box_id: info["page"] for box_id, info in bbox_data.items()
                        }
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse bbox mapping for DOI {row['doi']}")

                detailed_paper = DetailedPaper(
                    doi=row["doi"],
                    title=row["title"] or "Unknown Title",
                    abstract=row["abstract"],
                    authors=row["authors"],
                    journal=row["journal"],
                    source_date=row["source_date"],
                    source_type=row["source_type"],
                    source_details=row["source_details"],
                    relevance_assessment=relevance_assessment,
                    evidence_extraction=evidence_extraction,
                    citation_pages=citation_pages,
                    preprint=is_preprint(row["journal"], row["pmid"]),
                    pmid=row["pmid"],
                )
                low_confidence_papers.append(detailed_paper)

        logger.info(f"Found {len(low_confidence_papers)} low-confidence irrelevant papers")
        return low_confidence_papers


def load_manual_download_papers(db_path: Path) -> list[DetailedPaper]:
    """Load papers from recent literature that require manual download.

    These papers were not downloaded (missed or paywalled), so they only have
    basic metadata and relevance assessment from abstract.
    """
    logger.info("Loading papers requiring manual download...")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get papers from recent literature with manual_required download status
        cursor.execute("""
            SELECT
                p.doi,
                p.pmid,
                p.title,
                p.abstract,
                p.authors,
                p.journal,
                p.source_date,
                p.source_type,
                p.source_details,
                p.relevance_assessment_json
            FROM papers p
            WHERE p.source_type = 'initial'
            AND p.download_status = 'manual_required'
            ORDER BY p.source_date DESC, p.doi DESC
        """)

        manual_download_papers = []
        for row in cursor.fetchall():
            relevance_assessment = None
            if row["relevance_assessment_json"]:
                try:
                    relevance_assessment = compute_relevance_majority_vote(
                        json.loads(row["relevance_assessment_json"])
                    )
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse relevance assessment for DOI {row['doi']}")

            detailed_paper = DetailedPaper(
                doi=row["doi"],
                title=row["title"] or "Unknown Title",
                abstract=row["abstract"],
                authors=row["authors"],
                journal=row["journal"],
                source_date=row["source_date"],
                source_type=row["source_type"],
                source_details=row["source_details"],
                relevance_assessment=relevance_assessment,
                evidence_extraction=None,
                citation_pages=None,
                preprint=is_preprint(row["journal"], row["pmid"]),
                pmid=row["pmid"],
            )
            manual_download_papers.append(detailed_paper)

        logger.info(f"Found {len(manual_download_papers)} papers requiring manual download")
        return manual_download_papers


def calculate_unanimity_statistics(db_path: Path) -> tuple[int, int]:
    """Calculate unanimity statistics for relevance assessments.

    Returns:
        Tuple of (total_assessments, unanimous_assessments)
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT relevance_assessment_json
            FROM papers
            WHERE relevance_assessment_json IS NOT NULL
        """)

        total = 0
        unanimous = 0

        for row in cursor.fetchall():
            assessments = json.loads(row[0])
            assert len(assessments) == 3
            total += 1
            # Count how many are relevant=True; unanimous if 0 or 3
            relevant_count = sum(1 for a in assessments if a["relevant"])
            if relevant_count == 0 or relevant_count == 3:
                unanimous += 1

        return total, unanimous


def calculate_comprehensive_statistics(
    db_path: Path, results: GeneAssessmentResults, panel_validation: PanelValidationResult
) -> ComprehensiveStats:
    """Calculate enhanced statistics for report."""

    # Calculate unanimity statistics
    total_assessments, unanimous_assessments = calculate_unanimity_statistics(db_path)
    non_unanimous_pct = (
        ((total_assessments - unanimous_assessments) / total_assessments * 100)
        if total_assessments > 0
        else 0.0
    )

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Overall counts
        total_genes = len(results.novel_genes) + len(results.known_genes)
        novel_genes = len(results.novel_genes)

        # Count upgrades (known genes that could become GREEN)
        known_upgraded = 0
        for gene in results.known_genes:
            # Get current confidence from target panel data
            current_confidence = results.target_panel_data.gene_confidence.get(gene.hgnc_id)
            # Current is RED or AMBER (1, 2), new is GREEN (3)
            if current_confidence is not None and current_confidence < 3 and gene.new_rating == 3:
                known_upgraded += 1

        # Count total panel suggestions (no need for hardcoded panel IDs)
        total_panel_suggestions = 0
        all_genes = results.novel_genes + results.known_genes
        for gene in all_genes:
            total_panel_suggestions += len(gene.missing_panels)

        # Paper source breakdown - show ALL papers for statistics
        cursor.execute("""
            SELECT
                source_type,
                COUNT(DISTINCT doi) as count
            FROM papers
            WHERE evidence_extraction_json IS NOT NULL
            GROUP BY source_type
        """)
        source_counts = dict(cursor.fetchall())

        # Papers contributing to assessments
        all_dois = set()
        for gene in all_genes:
            for p in gene.contributing_papers:
                all_dois.add(p.doi)

        # Count MoI changes
        moi_expansions = 0
        moi_contradictions = 0
        for gene in all_genes:
            if gene.moi_comparison:
                if gene.moi_comparison["status"] == "expansion":
                    moi_expansions += 1
                elif gene.moi_comparison["status"] == "contradiction":
                    moi_contradictions += 1

        # Preprint stats (computed from already-loaded data)
        preprint_dois = {
            p.doi for gene in all_genes for p in gene.contributing_papers if p.preprint
        }
        papers_filtered = sum(
            1 for gene in all_genes for p in gene.contributing_papers if p.filtered_reason
        )

        return ComprehensiveStats(
            # Gene assessment stats
            total_genes_assessed=total_genes,
            novel_genes_count=novel_genes,
            known_genes_upgraded=known_upgraded,
            total_contributing_papers=len(all_dois),
            # Panel validation stats
            total_panel_papers=panel_validation.total_panel_papers,
            panel_papers_in_db=panel_validation.panel_papers_in_db,
            validation_sensitivity_pct=panel_validation.sensitivity_pct,
            false_negatives_count=len(panel_validation.false_negatives),
            true_positives_count=len(panel_validation.true_positives),
            # Source breakdown
            initial_papers=source_counts.get("initial", 0),
            expansion_papers=source_counts.get("expansion", 0),
            # Panel suggestions
            total_panel_suggestions=total_panel_suggestions,
            # MoI change stats
            moi_expansions_count=moi_expansions,
            moi_contradictions_count=moi_contradictions,
            # Preprint stats
            preprints_relevant=len(preprint_dois),
            papers_filtered=papers_filtered,
            # Relevance assessment unanimity stats
            non_unanimous_pct=non_unanimous_pct,
        )


def format_inheritance(mode: str, details: str | None = None) -> str:
    """Format inheritance mode and details for human-readable display.

    Args:
        mode: Inheritance mode enum value (may contain underscores)
        details: Optional inheritance details

    Returns:
        Formatted string with mode and details if provided
    """
    # Remove underscores and format the mode
    formatted_mode = mode.replace("_", " ") if mode else ""

    # Add details in parentheses if provided and not empty
    if details and details.strip():
        return f"{formatted_mode} ({details})"
    else:
        return formatted_mode


def get_variant_frequency_flag(variant: VariantFrequency, inheritance_mode: str) -> dict[str, Any]:
    """Determine if variant should be flagged based on inheritance mode.

    Args:
        variant: VariantFrequency object with gnomAD data
        inheritance_mode: One of the PanelApp inheritance mode enums

    Returns:
        Dict with:
        - should_flag: bool - whether to flag this variant
        - warning_message: str - mode-specific warning text
        - flagged_metric: str - what metric triggered the flag (e.g., "AC > 30")
    """
    should_flag = False
    warning_message = ""
    flagged_metric = ""

    # Skip if variant not found in gnomAD
    if variant.gnomad_not_found:
        return {
            "should_flag": False,
            "warning_message": "",
            "flagged_metric": "",
        }

    # Determine which threshold to apply based on inheritance mode
    if inheritance_mode == "Monoallelic":
        # Dominant: check heterozygote count
        if variant.gnomad_het is not None and variant.gnomad_het > GNOMAD_HET_THRESHOLD:
            should_flag = True
            flagged_metric = f"het > {GNOMAD_HET_THRESHOLD}"
            warning_message = f"High heterozygote count (het = {variant.gnomad_het} > {GNOMAD_HET_THRESHOLD}) - consider population frequency in assessment"

    elif inheritance_mode == "Biallelic":
        # Recessive: check homozygote count
        if variant.gnomad_hom is not None and variant.gnomad_hom > GNOMAD_HOM_THRESHOLD:
            should_flag = True
            flagged_metric = f"hom > {GNOMAD_HOM_THRESHOLD}"
            warning_message = f"High homozygote count (hom = {variant.gnomad_hom} > {GNOMAD_HOM_THRESHOLD}) - consider population frequency in assessment"

    elif inheritance_mode == "X-linked":
        # X-linked: check hemizygote count
        if variant.gnomad_hemi is not None and variant.gnomad_hemi > GNOMAD_HEMI_THRESHOLD:
            should_flag = True
            flagged_metric = f"hemi > {GNOMAD_HEMI_THRESHOLD}"
            warning_message = f"High hemizygote count (hemi = {variant.gnomad_hemi} > {GNOMAD_HEMI_THRESHOLD}) - consider population frequency in assessment"

    elif inheritance_mode == "Monoallelic_and_biallelic":
        # Both modes: flag if EITHER threshold exceeded
        het_exceeds = variant.gnomad_het is not None and variant.gnomad_het > GNOMAD_HET_THRESHOLD
        hom_exceeds = variant.gnomad_hom is not None and variant.gnomad_hom > GNOMAD_HOM_THRESHOLD

        if het_exceeds and hom_exceeds:
            should_flag = True
            flagged_metric = f"het > {GNOMAD_HET_THRESHOLD} and hom > {GNOMAD_HOM_THRESHOLD}"
            warning_message = f"High heterozygote count (het = {variant.gnomad_het}) and homozygote count (hom = {variant.gnomad_hom}) - consider population frequency in assessment"
        elif het_exceeds:
            should_flag = True
            flagged_metric = f"het > {GNOMAD_HET_THRESHOLD}"
            warning_message = f"High heterozygote count (het = {variant.gnomad_het} > {GNOMAD_HET_THRESHOLD}) - consider population frequency in assessment"
        elif hom_exceeds:
            should_flag = True
            flagged_metric = f"hom > {GNOMAD_HOM_THRESHOLD}"
            warning_message = f"High homozygote count (hom = {variant.gnomad_hom} > {GNOMAD_HOM_THRESHOLD}) - consider population frequency in assessment"

    else:
        # Default for Mitochondrial, Other, NR, or unknown: use het threshold
        if variant.gnomad_het is not None and variant.gnomad_het > GNOMAD_HET_THRESHOLD:
            should_flag = True
            flagged_metric = f"het > {GNOMAD_HET_THRESHOLD}"
            warning_message = f"High heterozygote count (het = {variant.gnomad_het} > {GNOMAD_HET_THRESHOLD}) - consider population frequency in assessment"

    return {
        "should_flag": should_flag,
        "warning_message": warning_message,
        "flagged_metric": flagged_metric,
    }


def prepare_aggregate_citation_links(
    citations: list[dict], contributing_papers: list[DetailedPaper]
) -> list[CitationLink]:
    """Prepare deduplicated citation links for aggregate assessment citations.

    Aggregate citations include a doi field specifying which paper the citation comes from.
    This function resolves box_ids to PDF page numbers via contributing papers' bbox mappings.

    Args:
        citations: List of citation dicts, each with doi and box_id fields
        contributing_papers: List of papers that might contain these citations

    Returns:
        Deduplicated CitationLink list, sorted by paper_id then page
    """
    links: set[CitationLink] = set()

    for citation in citations:
        box_id = citation.get("box_id")
        citation_doi = citation.get("doi")

        if box_id and citation_doi:
            for paper in contributing_papers:
                if (
                    paper.doi == citation_doi
                    and paper.citation_pages
                    and box_id in paper.citation_pages
                ):
                    page = paper.citation_pages[box_id]
                    links.add(CitationLink(paper_id=paper.paper_id, doi=paper.doi, page=page))
                    break

    return sorted(links, key=lambda link: (link.paper_id, link.page))


def build_report_config(
    report_id: str,
    target_panel_ids: list[int],
    novel_genes: list[GeneAssessment],
    known_genes: list[GeneAssessment],
) -> str:
    """Build the report-config JSON for PanelApp assignment integration.

    Args:
        report_id: Unique report identifier (e.g., "panel_arthrogryposis")
        target_panel_ids: List of panel IDs used for this report
        novel_genes: List of novel gene assessments
        known_genes: List of known gene assessments

    Returns:
        JSON string for embedding in HTML
    """
    genes = {}
    for gene in novel_genes + known_genes:
        genes[f"HGNC:{gene.hgnc_id}"] = {
            "hgnc_symbol": gene.hgnc_symbol,
            "suggested_rating": panelapp_confidence_to_color(gene.new_rating).upper(),
        }

    config = {
        "report_id": report_id,
        "target_panel_ids": target_panel_ids,
        "genes": genes,
    }
    return json.dumps(config)


def generate_html_report(
    novel_genes: list[GeneAssessment],
    known_genes: list[GeneAssessment],
    statistics: ComprehensiveStats,
    panel_validation: PanelValidationResult,
    low_confidence_papers: list[DetailedPaper],
    manual_download_papers: list[DetailedPaper],
    template_dir: Path,
    panel_date: str,
    pdf_base_dir: Path,
    output_dir: Path,
    report_id: str,
    target_panel_ids: list[int],
    *,
    panelapp_integration: bool,
) -> str:
    """Generate HTML report directly using Jinja2 templates."""

    # Calculate relative path from output directory to PDF base directory
    pdf_relative_path = os.path.relpath(pdf_base_dir, output_dir)

    # Build report config JSON for PanelApp integration
    report_config_json = (
        build_report_config(report_id, target_panel_ids, novel_genes, known_genes)
        if panelapp_integration
        else None
    )

    # Categorize novel genes by new rating
    novel_categories = NovelGeneCategories(
        green=[g for g in novel_genes if g.new_rating == 3],
        amber=[g for g in novel_genes if g.new_rating == 2],
        red=[g for g in novel_genes if g.new_rating == 1],
    )

    # Categorize known genes by upgrade type
    red_to_green = []
    amber_to_green = []
    red_to_amber = []
    no_change = []

    for gene in known_genes:
        if gene.existing_rating == 1 and gene.new_rating == 3:
            red_to_green.append(gene)
        elif gene.existing_rating == 2 and gene.new_rating == 3:
            amber_to_green.append(gene)
        elif gene.existing_rating == 1 and gene.new_rating == 2:
            red_to_amber.append(gene)
        else:
            no_change.append(gene)

    # Split no_change into highlighted MoI changes and others
    known_categories = KnownGeneCategories(
        red_to_green=red_to_green,
        amber_to_green=amber_to_green,
        red_to_amber=red_to_amber,
        no_change_with_moi=[
            g for g in no_change if g.moi_comparison and g.moi_comparison.get("highlighted")
        ],
        no_change_without_moi=[
            g for g in no_change if not (g.moi_comparison and g.moi_comparison.get("highlighted"))
        ],
    )

    # Set up Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Add custom filters
    env.filters["format_gene_with_aliases"] = format_gene_with_aliases
    env.filters["format_inheritance"] = format_inheritance
    env.filters["prepare_citation_links"] = prepare_aggregate_citation_links
    env.filters["get_variant_flag"] = get_variant_frequency_flag
    env.filters["confidence_to_color"] = panelapp_confidence_to_color
    env.filters["derive_moi"] = lambda pgs: derive_aggregate_moi(pgs)[0]
    env.filters["derive_moi_details"] = lambda pgs: derive_aggregate_moi(pgs)[1]

    # Add custom sort filter that handles None values
    def sort_by_family_count(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort disease entities by family_count, treating None as 0."""
        return sorted(entities, key=lambda e: e.get("family_count") or 0, reverse=True)

    env.filters["sort_by_family_count"] = sort_by_family_count

    # Add MONDO badge formatter
    def format_mondo_badge(entity: dict[str, str]) -> dict[str, str]:
        """Format disease entity into MONDO badge information."""
        mondo_id = entity["mondo_id"]
        category = MONDO_CATEGORIES.get(mondo_id)
        if category:
            return {
                "text": category["abbrev"],
                "label": category["label"],
                "css": category["abbrev"].lower(),
            }
        label = entity.get("mondo_label", mondo_id)
        return {"text": label, "label": label, "css": "specific"}

    env.filters["format_mondo_badge"] = format_mondo_badge
    # Render markdown then sanitize to a strict allowlist of tags.
    # This prevents LLM-generated summaries from injecting links, images, or scripts.
    _summary_allowed_tags = {"p", "strong", "em"}
    env.filters["markdown_to_html"] = lambda text: Markup(
        nh3.clean(markdown.markdown(text), tags=_summary_allowed_tags)
    )

    # Load custom CSS
    css_file = template_dir / "gene_report_styles.css"
    if not css_file.exists():
        raise FileNotFoundError(f"Custom CSS file not found at {css_file}")
    custom_css = Markup(css_file.read_text())
    logger.info(f"Loaded custom CSS from {css_file}")

    # Load main template
    template = env.get_template("gene_assessment_report.html")

    # Render HTML
    html = template.render(
        novel_categories=novel_categories,
        known_categories=known_categories,
        statistics=statistics,
        panel_validation=panel_validation,
        low_confidence_papers=low_confidence_papers,
        manual_download_papers=manual_download_papers,
        panel_date=panel_date,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        custom_css=custom_css,
        pdf_base_path=pdf_relative_path,
        panelapp_integration=panelapp_integration,
        report_config_json=report_config_json,
        gnomad_thresholds={
            "het": GNOMAD_HET_THRESHOLD,
            "hom": GNOMAD_HOM_THRESHOLD,
            "hemi": GNOMAD_HEMI_THRESHOLD,
        },
    )

    return html


@app.callback(invoke_without_command=True)
def main(
    report_id: str = typer.Option(
        ...,
        "--report-id",
        help="Unique report identifier (e.g., 'panel_arthrogryposis'). Used for output directory and PanelApp assignment tracking.",
    ),
    panel_date: str = typer.Option(..., "--panel-date", help="Panel state at date (YYYY-MM-DD)"),
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database with assessment results",
    ),
    target_panel_ids: list[int] | None = typer.Option(
        None,
        "--target-panel-ids",
        help="Panel IDs for novelty detection. Can be specified multiple times. If not specified, uses default TARGET_PANEL_IDS.",
    ),
    output_dir_prefix: Path = typer.Option(
        default=Path("reports"),
        help="Output directory prefix. Final path will be {prefix}/{report_id}/",
    ),
    annotated_dir: Path = typer.Option(
        Path("data/papers/annotated"),
        "--annotated-dir",
        help="Directory containing annotated PDFs",
    ),
    template_dir: Path = typer.Option(
        default=Path("templates"), help="Directory containing report templates"
    ),
    panelapp_integration: bool = typer.Option(
        True,
        "--panelapp-integration/--no-panelapp-integration",
        help="Enable PanelApp integration (prefill buttons, assignment support, CSRF token)",
    ),
) -> None:
    """Generate a self-contained directory with report and hierarchical annotated PDFs."""

    # Validate inputs
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if not annotated_dir.exists():
        logger.error(f"Annotated PDFs directory not found: {annotated_dir}")
        raise typer.Exit(1)

    # Construct output directory from prefix and report_id
    output_dir = output_dir_prefix / report_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating package from database: {db_path}")

    # Generate report
    hgnc_resolver = HgncResolver.from_file()
    results = load_gene_assessments(db_path, panel_date, hgnc_resolver, target_panel_ids)
    panel_validation = load_panel_publications_validation(db_path, target_panel_ids)
    low_confidence_papers = load_low_confidence_irrelevant_papers(db_path)
    manual_download_papers = load_manual_download_papers(db_path)
    statistics = calculate_comprehensive_statistics(db_path, results, panel_validation)
    # Use actual target_panel_ids from results (handles None -> defaults)
    actual_panel_ids = list(results.target_panel_data.panel_ids)

    html_content = generate_html_report(
        novel_genes=results.novel_genes,
        known_genes=results.known_genes,
        statistics=statistics,
        panel_validation=panel_validation,
        low_confidence_papers=low_confidence_papers,
        manual_download_papers=manual_download_papers,
        template_dir=template_dir,
        panel_date=panel_date,
        pdf_base_dir=Path("annotated"),  # Relative path within the package
        output_dir=Path("."),  # Root of the package
        report_id=report_id,
        target_panel_ids=actual_panel_ids,
        panelapp_integration=panelapp_integration,
    )

    # Save HTML as index.html
    index_file = output_dir / "index.html"
    index_file.write_text(html_content)

    # Symlink annotated PDFs directory into report (resolved at S3 upload time)
    annotated_link = output_dir / "annotated"
    os.symlink(annotated_dir.resolve(), annotated_link)

    # Validate that expected annotated PDFs exist
    missing = []
    found = 0
    all_genes = results.novel_genes + results.known_genes

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        for assessment in all_genes:
            for criterion_name in PANELAPP_CRITERIA:
                criterion = assessment.assessment_json[criterion_name]
                for citation in criterion.get("citations", []):
                    path = doi_to_path(
                        citation["doi"],
                        annotated_dir / str(assessment.hgnc_id),
                        ".pdf",
                    )
                    if path.exists():
                        found += 1
                    else:
                        missing.append(str(path))

            for disease_entity in assessment.assessment_json.get("disease_entities", []):
                for citation in disease_entity.get("citations", []):
                    path = doi_to_path(
                        citation["doi"],
                        annotated_dir / str(assessment.hgnc_id),
                        ".pdf",
                    )
                    if path.exists():
                        found += 1
                    else:
                        missing.append(str(path))

            cursor.execute(
                "SELECT DISTINCT paper_doi FROM variant_frequencies WHERE hgnc_id = ?",
                (assessment.hgnc_id,),
            )
            for row in cursor.fetchall():
                path = doi_to_path(
                    row["paper_doi"],
                    annotated_dir / str(assessment.hgnc_id),
                    ".pdf",
                )
                if path.exists():
                    found += 1
                else:
                    missing.append(str(path))

            for paper in assessment.contributing_papers:
                path = doi_to_path(paper.doi, annotated_dir / "individual", ".pdf")
                if path.exists():
                    found += 1
                else:
                    missing.append(str(path))

    if missing:
        logger.warning(f"Missing {len(missing)} annotated PDFs:")
        for missing_path in missing[:5]:
            logger.warning(f"  {missing_path}")
        if len(missing) > 5:
            logger.warning(f"  ... and {len(missing) - 5} more")

    logger.info("Package created successfully!")
    logger.info(f"   Output: {output_dir}")
    logger.info(f"   Contents: 1 HTML report + {found} annotated PDFs (symlinked)")


if __name__ == "__main__":
    app()
