#!/usr/bin/env python3
"""Generate gene-centric HTML assessment reports from aggregate assessments."""

import json
import logging
import shutil
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import markdown
import nh3
import typer
from anthropic.types import Message
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from palit.assess_genes import STAGE as ASSESS_GENES_STAGE
from palit.extract_evidence import STAGE as EXTRACTION_STAGE
from palit.gencc import MondoRef
from palit.hgnc import HgncResolver
from palit.llm import (
    FALLBACK_MODEL,
    FALLBACK_MODEL_NAME,
    MODEL_NAME,
    ResultStatus,
    parse_json_output,
)
from palit.map_mondo import STAGE as MAP_MONDO_STAGE
from palit.match_panels import STAGE as MATCH_PANELS_STAGE
from palit.panelapp_client import (
    AllPanelsData,
    PanelAppClient,
    PanelGeneData,
    get_current_panel_publications,
)
from palit.panelapp_integration import (
    MondoMatch,
    MondoTerm,
    PrefillAssociation,
    RelationStatus,
    calculate_association_rating,
    calculate_gene_rating,
    derive_aggregate_moi,
    panelapp_confidence_to_color,
    prepare_prefill_data,
)
from palit.papers import (
    build_display_ids,
    doi_to_key,
    doi_to_path,
    is_preprint,
    replace_paper_ids_for_display,
)

logger = logging.getLogger(__name__)

# Variant frequency thresholds for flagging based on inheritance mode
GNOMAD_HET_THRESHOLD = 30  # Monoallelic (dominant) - heterozygote count
GNOMAD_HOM_THRESHOLD = 15  # Biallelic (recessive) - homozygote count
GNOMAD_HEMI_THRESHOLD = 30  # X-linked - hemizygote count

# A new_moi association is highlighted only from this many independent families on
MIN_FAMILIES_FOR_MOI_EXPANSION = 2

# Hand-picked journals surfaced in the "Papers in Featured Journals" section.
# Strings are PubMed-style and must match papers.journal verbatim.
FAVORITE_JOURNALS: tuple[str, ...] = (
    "American journal of human genetics",
    "HGG advances",
    "European journal of human genetics : EJHG",
    "Genetics in medicine : official journal of the American College of Medical Genetics",
    "Genetics in medicine open",
    "Brain : a journal of neurology",
    "Annals of clinical and translational neurology",
    "Nature",
    "Nature genetics",
    "Science (New York, N.Y.)",
    "The New England journal of medicine",
    "Nature medicine",
    "Blood",
    "The Journal of clinical investigation",
    "Clinical genetics",
    "Neurology",
    "Journal of medical genetics",
    "Human mutation",
)


@dataclass(frozen=True)
class QuoteRef:
    """Where one of a paper's quotes sits: its index in the paper's citations file."""

    index: int
    page: int | None  # first highlighted page; None when the quote wasn't located


@dataclass(frozen=True)
class CitationLink:
    """A resolved link from a citation to its quote in the report's PDF viewer."""

    display_id: str  # "PMID {pmid}" for published papers, AuthorYear for preprints
    doi: str
    quote_index: int
    page: int | None


def aggregate_quotes(cursor: sqlite3.Cursor, doi: str) -> set[str]:
    """The quotes the aggregations of the paper's genes cite from the paper."""
    genes = "SELECT hgnc_id FROM gene_mentions WHERE paper_doi = ?"
    citations: list[dict[str, Any]] = []
    cursor.execute(f"SELECT assessment_json FROM associations WHERE hgnc_id IN ({genes})", (doi,))
    for (assessment_json,) in cursor.fetchall():
        entity = json.loads(assessment_json)
        citations += entity["citations"]
        for criterion in entity["evidence_assessments"]:
            citations += criterion["citations"]
    cursor.execute(
        f"SELECT quality_concerns_json FROM gene_aggregations WHERE hgnc_id IN ({genes})", (doi,)
    )
    for (concerns_json,) in cursor.fetchall():
        for concern in json.loads(concerns_json):
            citations += concern["citations"]
    return {citation["quote"] for citation in citations if citation["doi"] == doi}


def paper_quotes(cursor: sqlite3.Cursor, doi: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """A paper's quotes with their highlight boxes, in the order of its citations file.

    The extraction's quotes come first, then the quotes an aggregation cites
    that are not among them, which have no boxes. Both parts are sorted by
    text, so a quote's position doesn't depend on the aggregations.
    """
    cursor.execute(
        "SELECT quote, bboxes_json FROM citation_locations WHERE paper_doi = ? ORDER BY quote",
        (doi,),
    )
    quotes = [(quote, json.loads(bboxes_json)) for quote, bboxes_json in cursor.fetchall()]
    extracted = {quote for quote, _ in quotes}
    return quotes + [(quote, []) for quote in sorted(aggregate_quotes(cursor, doi) - extracted)]


def load_quote_refs(cursor: sqlite3.Cursor, doi: str) -> dict[str, QuoteRef]:
    """Index a paper's quotes, located or not, in the order of its citations file."""
    return {
        quote: QuoteRef(index=index, page=bboxes[0]["page"] if bboxes else None)
        for index, (quote, bboxes) in enumerate(paper_quotes(cursor, doi))
    }


def citation_link(paper: "DetailedPaper", quote: str) -> CitationLink | None:
    """The viewer link for one of *paper*'s quotes, or None if it isn't a known quote."""
    ref = paper.quote_refs.get(quote)
    if ref is None:
        return None
    return CitationLink(
        display_id=paper.display_id, doi=paper.doi, quote_index=ref.index, page=ref.page
    )


@dataclass
class VariantFrequency:
    """Variant frequency information from gnomAD."""

    variant_id: str  # gnomAD pseudo-VCF format
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
    gnomad_not_found: bool  # True if variant not found in gnomAD
    gnomad_error: str | None  # Error message if gnomAD lookup failed
    citations: list[CitationLink]  # Papers reporting this variant, sorted by display_id


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
    quote_refs: dict[str, QuoteRef]  # this paper's quotes -> position in its citations file
    preprint: bool = False
    pmid: int | None = None  # For PubMed display links
    display_id: str = ""  # "PMID {pmid}" for published papers, AuthorYear for preprints
    paper_gene_symbol: str | None = None
    filtered_reason: str | None = None  # Set when paper was excluded from assessment


@dataclass(frozen=True)
class FavoriteJournalGeneLink:
    """Hyperlink from a featured-journal paper to a gene's anchor in this report."""

    hgnc_id: int
    hgnc_symbol: str
    anchor: str  # "#novel-gene-{id}" or "#known-gene-{id}"


@dataclass
class FavoriteJournalPaper:
    """One paper from a hand-picked journal, surfaced at the corpus level."""

    doi: str
    pmid: int | None
    title: str
    journal: str
    source_date: str | None
    preprint: bool
    display_id: str  # "PMID {pmid}" for published papers, falls back to DOI
    gene_links: list[FavoriteJournalGeneLink]
    filtered_reason: str | None  # set when paper was filtered out of every gene assessment


@dataclass
class FavoriteJournalSections:
    """Featured-journal papers bucketed by the kind of contribution they made.

    Only initial-search papers are surfaced here; expansion papers (added via
    citation chasing) are excluded entirely. Transient pipeline states
    (downloaded but not yet extracted, etc.) are also dropped — they're not
    actionable and just clutter the section."""

    novel: list[FavoriteJournalPaper]
    rating_upgrade: list[FavoriteJournalPaper]
    moi_expansion: list[FavoriteJournalPaper]
    unreviewed: list[FavoriteJournalPaper]
    already_reviewed: list[FavoriteJournalPaper]
    filtered: list[FavoriteJournalPaper]
    manual_download: list[FavoriteJournalPaper]
    refused: list[FavoriteJournalPaper]  # both models refused its relevance assessment
    not_relevant: list[FavoriteJournalPaper]
    off_panel: list[FavoriteJournalPaper]

    @property
    def total(self) -> int:
        return (
            len(self.novel)
            + len(self.rating_upgrade)
            + len(self.moi_expansion)
            + len(self.unreviewed)
            + len(self.already_reviewed)
            + len(self.filtered)
            + len(self.manual_download)
            + len(self.refused)
            + len(self.not_relevant)
            + len(self.off_panel)
        )


@dataclass(frozen=True)
class PanelMatch:
    """A panel that match-panels matched one association to."""

    panel_id: int
    panel_name: str
    rationale: str
    gene_on_panel: bool  # the gene already has an entry on this panel


@dataclass(frozen=True)
class PanelSuggestionReason:
    """Why one association of the gene was matched to a panel."""

    disease_label: str
    inheritance_mode: str
    rationale: str


@dataclass
class PanelSuggestion:
    """A panel matched by at least one of the gene's associations."""

    panel_id: int
    panel_name: str
    reasons: list[PanelSuggestionReason]


@dataclass(frozen=True)
class ExistingPanelReviews:
    """PanelApp evaluations captured at assess time for one target panel that held the gene.

    An empty ``evaluations`` list means no expert has reviewed the gene on that
    panel, the signal that drives the "Unreviewed on Target Panel" section.
    """

    panel_id: int
    evaluations: list[dict[str, Any]]


@dataclass(frozen=True)
class DisputeRef:
    """A GenCC submission for the gene with the dispute status the association states.

    The model matches disputes to associations itself and names no submission, so
    every submission with that status is listed; ``same_term`` marks the ones on
    the association's own MONDO term.
    """

    submitter: str
    status: str  # "Disputed" or "Refuted"
    disease_title: str
    mondo_id: str
    moi_title: str
    date: str
    same_term: bool


class StageState(StrEnum):
    """Where map-mondo or match-panels stands for one association."""

    STORED = "stored"  # the association holds the stage's result
    NOT_RUN = "not_run"  # no request of the stage for this association
    REFUSED = "refused"  # FALLBACK_MODEL refused the latest request; the stage skips it from now on
    FAILED = "failed"  # requests ran but none gave a valid result; a rerun retries it


def stage_state(stored: bool, latest_status: str | None, latest_model: str | None) -> StageState:
    """The state of a stage for one association, from its stored result and latest request.

    A latest refusal by MODEL counts as failed: the next attempt sends the
    association to FALLBACK_MODEL.
    """
    if stored:
        return StageState.STORED
    if latest_status is None:
        return StageState.NOT_RUN
    if latest_status == ResultStatus.REFUSED and latest_model == FALLBACK_MODEL:
        return StageState.REFUSED
    return StageState.FAILED


@dataclass
class ReportAssociation:
    """One gene-disease-MoI association of a gene, prepared for display."""

    id: int
    position: int  # order in the model output
    assessment: dict[str, Any]  # the stored association JSON, paper IDs in display form
    disease_label: str  # the GenCC row's label when reused, else the proposed name
    mondo: MondoTerm | None  # None unless mondo_mapping is STORED
    mondo_mapping: StageState  # STORED for a reused GenCC row too
    rating: int  # computed from this association's criteria: 3 GREEN, 2 AMBER, 1 RED
    disputes: list[DisputeRef]
    matched_panels: list[PanelMatch] | None  # None unless panel_matching is STORED
    panel_matching: StageState

    @property
    def relation_status(self) -> RelationStatus:
        status: RelationStatus = self.assessment["panelapp_relation"]["status"]
        return status

    @property
    def independent_family_count(self) -> int | None:
        count: int | None = self.assessment["independent_family_count"]
        return count

    @property
    def new_moi_highlighted(self) -> bool:
        return is_highlighted_new_moi(self.assessment)


@dataclass(frozen=True)
class RefusedPaper:
    """A paper of the gene that both models refused in extract-evidence, so it was never read."""

    doi: str
    pmid: int | None
    title: str
    category: str | None  # the fallback model's refusal category


@dataclass
class GeneAssessment:
    """A gene's aggregation: its associations and the gene-level blocks."""

    hgnc_id: int
    hgnc_symbol: str
    associations: list[ReportAssociation]  # by corpus rating, then independent families
    unassessed_reports: list[dict[str, Any]]  # [{phenotype, inheritance_mode, dois, reason}]
    quality_concerns: list[dict[str, Any]]  # [{concern, dois, citations}]
    existing_rating: int | None  # the gene's rating on its first target panel; None if novel
    new_rating: int  # the top association rating: 3 (GREEN), 2 (AMBER), 1 (RED)
    aggregate_moi: str  # derive_aggregate_moi over the associations, for gnomAD flags
    contributing_papers: list[DetailedPaper]
    variant_frequencies: list[VariantFrequency]  # Variants with frequency data
    missing_panels: list[PanelSuggestion]  # matched panels the gene is not on
    existing_panels: list[PanelSuggestion]  # matched panels the gene is already on
    prefill_json: str  # HTML-escaped JSON for data-prefill attribute
    existing_panel_reviews: list[ExistingPanelReviews]  # per target panel; empty if novel
    refused_papers: list[RefusedPaper]

    @property
    def is_novel(self) -> bool:
        """The gene is on none of the target panels."""
        return self.existing_rating is None

    @property
    def has_highlighted_new_moi(self) -> bool:
        return any(a.new_moi_highlighted for a in self.associations)

    @property
    def unmatched_by_state(self) -> dict[StageState, int]:
        """The number of associations without panel matches per state, in StageState order."""
        counts = Counter(a.panel_matching for a in self.associations)
        return {
            state: counts[state]
            for state in StageState
            if state != StageState.STORED and counts[state]
        }


@dataclass(frozen=True)
class RefusedGene:
    """A gene that both models refused in assess-genes and that has no aggregation."""

    hgnc_id: int
    hgnc_symbol: str
    category: str | None


@dataclass
class GeneAssessmentResults:
    """Results from loading and sorting gene assessments."""

    novel_genes: list[GeneAssessment]
    known_genes: list[GeneAssessment]
    refused_genes: list[RefusedGene]
    target_panel_data: PanelGeneData
    target_panel_names: dict[int, str]  # panel_id → display name, for current target panels
    panels_matched: bool  # match-panels has run in this database; panel-scoped runs skip it


@dataclass
class PanelValidationResult:
    """Panel publication validation results."""

    total_panel_papers: int
    panel_papers_in_db: int  # with a relevance result, including the refused ones
    true_positives: list[DetailedPaper]
    screen_misses: list[DetailedPaper]  # rejected by the scope screen
    check_rejections: list[DetailedPaper]  # passed the screen, judged curated by the PanelApp check
    refused: list[DetailedPaper]  # both models refused a level; left out of the sensitivity
    sensitivity_pct: float  # share of the assessed (not refused) ones assessed relevant
    screen_sensitivity_pct: float  # share of the assessed (not refused) ones that passed the screen


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
    validation_screen_sensitivity_pct: float
    screen_misses_count: int
    check_rejections_count: int
    true_positives_count: int
    panel_refused_count: int  # relevance refused by both models, not in the sensitivity

    # Source breakdown
    initial_papers: int
    expansion_papers: int

    # Panel suggestions
    total_panel_suggestions: int

    # Associations, and new_moi associations highlighted for curators
    total_associations: int
    highlighted_new_moi_count: int

    # Papers of assessed genes that both models refused in extract-evidence
    refused_papers_count: int

    # Known genes on the target panel with zero expert reviews
    unreviewed_count: int

    # Preprint stats
    preprints_relevant: int
    papers_filtered: int


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
    no_change_new_moi: list[GeneAssessment]
    no_change_unreviewed: list[GeneAssessment]
    no_change_reviewed: list[GeneAssessment]

    @property
    def total(self) -> int:
        """Total number of known genes."""
        return (
            len(self.red_to_green)
            + len(self.amber_to_green)
            + len(self.red_to_amber)
            + len(self.no_change_new_moi)
            + len(self.no_change_unreviewed)
            + len(self.no_change_reviewed)
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

    # Paper symbols are uppercased; current symbols such as C9orf72 are not.
    paper_gene_symbols.discard(hgnc_symbol.upper())

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
    normalization: dict[str, Any],
    gnomad: dict[str, Any],
    citations: list[CitationLink],
) -> VariantFrequency:
    """Create a VariantFrequency object from database row data.

    The ``gnomad`` JSON is the flat shape written by
    extract-evidence's variant lookups: success rows carry ``ac`` / ``an`` /
    ``homozygote_count`` / ``heterozygote_count`` / ``hemizygote_count``
    / ``faf95_popmax`` / ``faf95_popmax_population`` directly; sentinel
    rows carry ``{"variant_not_found": true}`` or
    ``{"normalization_error": true}``.
    """
    gnomad_not_found = gnomad.get("variant_not_found", False)
    gnomad_error = "normalization_error" if gnomad.get("normalization_error") else None

    popmax_pop = gnomad.get("faf95_popmax_population")
    gnomad_faf95_popmax_population = (
        POPULATION_NAMES.get(popmax_pop, popmax_pop) if popmax_pop else None
    )

    return VariantFrequency(
        variant_id=variant_id,
        hgvs_c=normalization.get("hgvs_c"),
        hgvs_p=normalization.get("hgvs_p"),
        original_text=normalization.get("original_text", ""),
        gnomad_ac=gnomad.get("ac"),
        gnomad_an=gnomad.get("an"),
        gnomad_hom=gnomad.get("homozygote_count"),
        gnomad_het=gnomad.get("heterozygote_count"),
        gnomad_hemi=gnomad.get("hemizygote_count"),
        gnomad_faf95_popmax=gnomad.get("faf95_popmax"),
        gnomad_faf95_popmax_population=gnomad_faf95_popmax_population,
        gnomad_link=f"https://gnomad.broadinstitute.org/variant/{variant_id}?dataset=gnomad_r4",
        gnomad_not_found=gnomad_not_found,
        gnomad_error=gnomad_error,
        citations=citations,
    )


def load_variant_frequencies_for_gene(
    cursor: sqlite3.Cursor, hgnc_id: int, contributing_papers: list[DetailedPaper]
) -> list[VariantFrequency]:
    """Load variant frequency information for a specific gene.

    Variants are deduplicated by ``variant_id``: when the same allele is
    reported by multiple contributing papers, the gnomAD numbers (queried
    by genomic coordinate) are identical and would otherwise render as
    visually duplicate rows. Per-paper citation links are merged into the
    returned VariantFrequency's ``citations`` list.

    Args:
        cursor: Database cursor
        hgnc_id: HGNC ID of the gene to load variants for
        contributing_papers: Papers contributing to this gene assessment

    Returns:
        List of VariantFrequency objects with successful gnomAD lookups
        first, then ``variant_not_found``, then normalization errors;
        ``variant_id`` breaks ties within each bucket. Each entry carries
        a deduplicated, ordered list of citations.
    """
    papers_by_doi = {paper.doi: paper for paper in contributing_papers}
    contributing_dois = list(papers_by_doi)
    placeholders = ",".join("?" * len(contributing_dois))
    cursor.execute(
        f"""
        SELECT
            vf.variant_id,
            vf.paper_doi,
            vf.quote,
            vf.normalization,
            vf.gnomad
        FROM variant_frequencies vf
        WHERE vf.hgnc_id = ?
          AND vf.paper_doi IN ({placeholders})
    """,
        (hgnc_id, *contributing_dois),
    )

    rows_by_variant: dict[str, list[sqlite3.Row]] = {}
    for row in cursor.fetchall():
        rows_by_variant.setdefault(row["variant_id"], []).append(row)

    parsed: list[tuple[str, dict[str, Any], dict[str, Any], list[sqlite3.Row]]] = []
    for variant_id, rows in rows_by_variant.items():
        try:
            normalization = json.loads(rows[0]["normalization"])
            gnomad = json.loads(rows[0]["gnomad"])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for variant {variant_id}: {e}")
            continue
        parsed.append((variant_id, normalization, gnomad, rows))

    def sort_key(
        item: tuple[str, dict[str, Any], dict[str, Any], list[sqlite3.Row]],
    ) -> tuple[int, str]:
        variant_id, _, gnomad, _ = item
        if gnomad.get("normalization_error"):
            bucket = 2
        elif gnomad.get("variant_not_found"):
            bucket = 1
        else:
            bucket = 0
        return (bucket, variant_id)

    parsed.sort(key=sort_key)

    variant_frequencies: list[VariantFrequency] = []
    for variant_id, normalization, gnomad, rows in parsed:
        citation_set: set[CitationLink] = set()
        for row in rows:
            link = citation_link(papers_by_doi[row["paper_doi"]], row["quote"])
            if link is not None:
                citation_set.add(link)
        citations = sorted(citation_set, key=lambda c: (c.display_id, c.quote_index))

        variant_frequencies.append(
            _create_variant_frequency_from_db_row(
                variant_id=variant_id,
                normalization=normalization,
                gnomad=gnomad,
                citations=citations,
            )
        )

    return variant_frequencies


def is_highlighted_new_moi(assessment: dict[str, Any]) -> bool:
    """A new_moi association is highlighted once enough independent families support it."""
    return (
        assessment["panelapp_relation"]["status"] == "new_moi"
        and (assessment["independent_family_count"] or 0) >= MIN_FAMILIES_FOR_MOI_EXPANSION
    )


def association_sort_key(association: ReportAssociation) -> tuple[int, int, int]:
    """Highest corpus rating first, then most independent families, then model order."""
    return (
        -association.rating,
        -(association.independent_family_count or 0),
        association.position,
    )


def novel_gene_sort_key(gene: GeneAssessment) -> tuple[int, int, str]:
    """Highest rating first, then genes with a highlighted new MoI, then by symbol."""
    return (-gene.new_rating, 0 if gene.has_highlighted_new_moi else 1, gene.hgnc_symbol)


def known_gene_sort_key(gene: GeneAssessment) -> tuple[int, int, int, str]:
    """Lowest existing rating first, then highest corpus rating, then highlighted new MoI."""
    if gene.existing_rating is None:
        raise ValueError(
            f"Known gene {gene.hgnc_symbol} (HGNC:{gene.hgnc_id}) has no rating on the target panels"
        )
    return (
        gene.existing_rating,
        -gene.new_rating,
        0 if gene.has_highlighted_new_moi else 1,
        gene.hgnc_symbol,
    )


def load_contributing_papers(
    cursor: sqlite3.Cursor, hgnc_id: int, filtered_doi_reasons: dict[str, str]
) -> list[DetailedPaper]:
    """Papers with an extraction for the gene, initial and expansion alike, newest first.

    One row per paper, with the symbol from the extraction's gene mention (a
    relevance-stage mention may carry an older symbol of the same gene).
    """
    cursor.execute(
        """
        SELECT
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
            gm.paper_gene_symbol
        FROM papers p
        JOIN gene_mentions gm ON p.doi = gm.paper_doi
        WHERE gm.hgnc_id = ?
        AND gm.source IN ('recent_evidence', 'expansion_evidence')
        AND p.evidence_extraction_json IS NOT NULL
        ORDER BY p.source_date DESC, p.doi DESC
    """,
        (hgnc_id,),
    )

    contributing_papers = []
    for paper_row in cursor.fetchall():
        relevance_assessment = (
            json.loads(paper_row["relevance_assessment_json"])
            if paper_row["relevance_assessment_json"]
            else None
        )
        evidence_extraction = json.loads(paper_row["evidence_extraction_json"])
        # A paper may mention the gene but only have evidence for other genes.
        if not any(g.get("hgnc_id") == hgnc_id for g in evidence_extraction["gene_evaluations"]):
            continue

        doi = paper_row["doi"]
        contributing_papers.append(
            DetailedPaper(
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
                quote_refs=load_quote_refs(cursor, doi),
                preprint=is_preprint(paper_row["journal"], paper_row["pmid"]),
                pmid=paper_row["pmid"],
                paper_gene_symbol=paper_row["paper_gene_symbol"],
                filtered_reason=filtered_doi_reasons.get(doi),
            )
        )
    return contributing_papers


def load_refused_papers(cursor: sqlite3.Cursor, hgnc_id: int) -> list[RefusedPaper]:
    """The gene's papers without an extraction that extract-evidence refused for good.

    A paper is refused for good once FALLBACK_MODEL refused it; a paper only
    MODEL refused is still due its fallback request.

    A paper belongs to the gene when the relevance screen named the gene for it,
    when the gene's tournament selected it, or when it was added for the gene as a
    PanelApp-cited or referenced paper (``source_details``).
    """
    cursor.execute(
        """
        SELECT p.doi, p.pmid, p.title,
               (SELECT r2.refusal_category FROM llm_requests r2
                WHERE r2.stage = ? AND r2.subject = p.doi AND r2.status = 'refused'
                  AND r2.model = ?
                ORDER BY r2.completed_at DESC LIMIT 1) AS category
        FROM papers p
        WHERE p.evidence_extraction_json IS NULL
          AND EXISTS (
              SELECT 1 FROM llm_requests r
              WHERE r.stage = ? AND r.subject = p.doi AND r.status = 'refused' AND r.model = ?
          )
          AND (
              EXISTS (
                  SELECT 1 FROM gene_mentions gm
                  WHERE gm.paper_doi = p.doi AND gm.hgnc_id = ?
                    AND gm.source = 'relevance_assessment'
              )
              OR EXISTS (
                  SELECT 1 FROM tournament_results t, json_each(t.selected_dois_json) j
                  WHERE t.hgnc_id = ? AND j.value = p.doi
              )
              OR p.source_details IN (CAST(? AS TEXT), 'panelapp:' || ?)
              OR p.source_details LIKE 'referenced:' || ? || ':%'
          )
        ORDER BY p.source_date DESC, p.doi
        """,
        (
            EXTRACTION_STAGE,
            FALLBACK_MODEL,
            EXTRACTION_STAGE,
            FALLBACK_MODEL,
            hgnc_id,
            hgnc_id,
            hgnc_id,
            hgnc_id,
            hgnc_id,
        ),
    )
    return [
        RefusedPaper(doi=row["doi"], pmid=row["pmid"], title=row["title"], category=row["category"])
        for row in cursor.fetchall()
    ]


def load_refused_genes(cursor: sqlite3.Cursor, hgnc_resolver: HgncResolver) -> list[RefusedGene]:
    """Genes without an aggregation that assess-genes refused for good (FALLBACK_MODEL refused)."""
    cursor.execute(
        """
        SELECT CAST(r.subject AS INTEGER) AS hgnc_id, r.refusal_category
        FROM llm_requests r
        WHERE r.stage = ? AND r.status = 'refused' AND r.model = ?
          AND CAST(r.subject AS INTEGER) NOT IN (SELECT hgnc_id FROM gene_aggregations)
        GROUP BY r.subject
        ORDER BY hgnc_id
        """,
        (ASSESS_GENES_STAGE, FALLBACK_MODEL),
    )
    return [
        RefusedGene(
            hgnc_id=row["hgnc_id"],
            hgnc_symbol=hgnc_resolver.get_symbol(row["hgnc_id"]),
            category=row["refusal_category"],
        )
        for row in cursor.fetchall()
    ]


def _mondo_term(row: sqlite3.Row, gencc_rows: list[dict[str, Any]]) -> MondoTerm | None:
    """The association row's MONDO term, or None while it has none."""
    mondo_id: str | None = row["mondo_id"]
    if mondo_id is None:
        return None
    match: MondoMatch = row["mondo_match"]
    if match == "panelapp_gencc":
        gencc_row = next(r for r in gencc_rows if r["mondo_id"] == mondo_id)
        return MondoTerm(
            mondo_id=mondo_id,
            label=row["mondo_label"],
            match=match,
            rationale="",
            obsolete=gencc_row["obsolete"],
            replaced_by=tuple(MondoRef(**ref) for ref in gencc_row["replaced_by"]),
        )
    answer = parse_json_output(Message.model_validate_json(row["mondo_raw"]))
    return MondoTerm(
        mondo_id=mondo_id,
        label=row["mondo_label"],
        match=match,
        rationale=answer["rationale"],
        obsolete=False,
        replaced_by=(),
    )


def association_disputes(
    assessment: dict[str, Any], mondo_id: str | None, disputes: list[dict[str, Any]]
) -> list[DisputeRef]:
    """The gene's GenCC submissions whose status is the association's dispute status.

    Submissions on the association's own MONDO term come first.
    """
    status = assessment["dispute_status"]
    refs = [
        DisputeRef(
            submitter=d["submitter"],
            status=d["status"],
            disease_title=d["disease_title"],
            mondo_id=d["mondo_id"],
            moi_title=d["moi_title"],
            date=d["date"],
            same_term=d["mondo_id"] == mondo_id,
        )
        for d in disputes
        if d["status"] == status
    ]
    return sorted(refs, key=lambda d: not d.same_term)


def _panel_matches(
    matched_panels_json: str | None, hgnc_id: int, all_panels_data: AllPanelsData
) -> list[PanelMatch] | None:
    """An association's matched panels, or None while it has none stored."""
    if matched_panels_json is None:
        return None
    current_panels = all_panels_data.gene_to_panels.get(hgnc_id, set())
    matches = []
    for match in json.loads(matched_panels_json):
        panel_id = match["panel_id"]
        panel_name = all_panels_data.panel_names.get(panel_id)
        if panel_name is None:
            logger.warning(
                f"Panel {panel_id} no longer exists in PanelApp, skipping match for HGNC:{hgnc_id}"
            )
            continue
        matches.append(
            PanelMatch(
                panel_id=panel_id,
                panel_name=panel_name,
                rationale=match["rationale"],
                gene_on_panel=panel_id in current_panels,
            )
        )
    return sorted(matches, key=lambda m: m.panel_name)


def load_associations(
    cursor: sqlite3.Cursor,
    hgnc_id: int,
    panelapp_context: dict[str, Any],
    display_ids: dict[str, str],
    all_panels_data: AllPanelsData,
) -> list[ReportAssociation]:
    """The gene's associations, by corpus rating and then independent family count.

    Each association carries the state of map-mondo and match-panels for it, from
    its latest request of each stage. Both stages send immediate requests only,
    which are recorded once they have completed.
    """

    def latest(column: str) -> str:
        return f"""
            (SELECT r.{column} FROM llm_requests r
             WHERE r.stage = ? AND r.subject = CAST(a.id AS TEXT)
             ORDER BY r.completed_at DESC
             LIMIT 1)
        """

    cursor.execute(
        f"""
        SELECT a.id, a.position, a.assessment_json, a.mondo_id, a.mondo_label, a.mondo_match,
               a.mondo_raw, a.matched_panels_json,
               {latest("status")} AS mondo_status,
               {latest("model")} AS mondo_model,
               {latest("status")} AS matching_status,
               {latest("model")} AS matching_model
        FROM associations a
        WHERE a.hgnc_id = ?
        ORDER BY a.position
        """,
        (MAP_MONDO_STAGE, MAP_MONDO_STAGE, MATCH_PANELS_STAGE, MATCH_PANELS_STAGE, hgnc_id),
    )
    associations = []
    for row in cursor.fetchall():
        assessment = replace_paper_ids_for_display(json.loads(row["assessment_json"]), display_ids)
        mondo = _mondo_term(row, panelapp_context["gencc_rows"])
        associations.append(
            ReportAssociation(
                id=row["id"],
                position=row["position"],
                assessment=assessment,
                disease_label=(
                    mondo.label
                    if mondo is not None and mondo.match == "panelapp_gencc"
                    else assessment["proposed_disease_name"]
                ),
                mondo=mondo,
                mondo_mapping=stage_state(
                    mondo is not None, row["mondo_status"], row["mondo_model"]
                ),
                rating=calculate_association_rating(assessment),
                disputes=association_disputes(
                    assessment, row["mondo_id"], panelapp_context["disputes"]
                ),
                matched_panels=_panel_matches(row["matched_panels_json"], hgnc_id, all_panels_data),
                panel_matching=stage_state(
                    row["matched_panels_json"] is not None,
                    row["matching_status"],
                    row["matching_model"],
                ),
            )
        )
    return sorted(associations, key=association_sort_key)


def union_panel_suggestions(
    associations: list[ReportAssociation],
) -> tuple[list[PanelSuggestion], list[PanelSuggestion]]:
    """The panels matched by any association, split into (gene not on it, gene on it).

    Each panel lists the associations that matched it, with their rationales.
    Both lists are sorted by panel name.
    """
    by_panel: dict[int, tuple[PanelSuggestion, bool]] = {}
    for association in associations:
        for match in association.matched_panels or []:
            entry = by_panel.get(match.panel_id)
            if entry is None:
                entry = (PanelSuggestion(match.panel_id, match.panel_name, []), match.gene_on_panel)
                by_panel[match.panel_id] = entry
            entry[0].reasons.append(
                PanelSuggestionReason(
                    disease_label=association.disease_label,
                    inheritance_mode=association.assessment["inheritance_mode"],
                    rationale=match.rationale,
                )
            )
    suggestions = sorted(by_panel.values(), key=lambda e: e[0].panel_name)
    missing = [suggestion for suggestion, on_panel in suggestions if not on_panel]
    existing = [suggestion for suggestion, on_panel in suggestions if on_panel]
    return missing, existing


def load_gene(
    cursor: sqlite3.Cursor,
    row: sqlite3.Row,
    hgnc_resolver: HgncResolver,
    target_panel_data: PanelGeneData,
    all_panels_data: AllPanelsData,
) -> GeneAssessment:
    """One gene's aggregation row with its associations, papers and panel context."""
    hgnc_id: int = row["hgnc_id"]
    paper_id_to_doi: dict[str, str] = json.loads(row["paper_id_mapping"])
    filtered_doi_reasons: dict[str, str] = {
        fp["doi"]: fp["reason"] for fp in json.loads(row["filtered_papers_json"] or "[]")
    }
    existing_panel_reviews = [
        ExistingPanelReviews(panel_id=entry["panel_id"], evaluations=entry["evaluations"])
        for entry in json.loads(row["existing_panel_reviews_json"] or "[]")
    ]

    contributing_papers = load_contributing_papers(cursor, hgnc_id, filtered_doi_reasons)

    # Replace AuthorYear paper IDs with PMID display format
    doi_to_pmid = {p.doi: p.pmid for p in contributing_papers if p.pmid is not None}
    display_ids = build_display_ids(paper_id_to_doi, doi_to_pmid)
    doi_to_display_id = {paper_id_to_doi[pid]: did for pid, did in display_ids.items()}
    for paper in contributing_papers:
        if paper.doi in doi_to_display_id:
            paper.display_id = doi_to_display_id[paper.doi]

    associations = load_associations(
        cursor,
        hgnc_id,
        json.loads(row["panelapp_context_json"]),
        display_ids,
        all_panels_data,
    )
    gene_level = replace_paper_ids_for_display(
        {
            "unassessed_reports": json.loads(row["unassessed_reports_json"]),
            "quality_concerns": json.loads(row["quality_concerns_json"]),
        },
        display_ids,
    )
    missing_panels, existing_panels = union_panel_suggestions(associations)
    association_jsons = [a.assessment for a in associations]

    # Gene is novel if not in any target panel; the list keeps target-panel order
    gene_panels = target_panel_data.gene_panel_mapping.get(hgnc_id, set())
    target_panel_membership = [pid for pid in target_panel_data.panel_ids if pid in gene_panels]
    is_novel = not target_panel_membership

    if is_novel:
        prefill_panel_id = target_panel_data.panel_ids[0]
        prefill_form_type = "add"
    else:
        prefill_panel_id = target_panel_membership[0]
        prefill_form_type = "review"
    prefill_data = prepare_prefill_data(
        hgnc_id=hgnc_id,
        associations=[PrefillAssociation(a.assessment, a.mondo) for a in associations],
        form_type=prefill_form_type,
        panel_id=prefill_panel_id,
        doi_to_pmid={p.doi: p.pmid for p in contributing_papers},
    )

    return GeneAssessment(
        hgnc_id=hgnc_id,
        hgnc_symbol=hgnc_resolver.get_symbol(hgnc_id),
        associations=associations,
        unassessed_reports=gene_level["unassessed_reports"],
        quality_concerns=gene_level["quality_concerns"],
        existing_rating=None if is_novel else target_panel_data.gene_confidence[hgnc_id],
        new_rating=calculate_gene_rating(association_jsons),
        aggregate_moi=derive_aggregate_moi(association_jsons),
        contributing_papers=contributing_papers,
        variant_frequencies=load_variant_frequencies_for_gene(cursor, hgnc_id, contributing_papers),
        missing_panels=missing_panels,
        existing_panels=existing_panels,
        prefill_json=json.dumps(asdict(prefill_data)),
        existing_panel_reviews=existing_panel_reviews,
        refused_papers=load_refused_papers(cursor, hgnc_id),
    )


def build_gene_assessment_results(
    conn: sqlite3.Connection,
    hgnc_resolver: HgncResolver,
    target_panel_data: PanelGeneData,
    all_panels_data: AllPanelsData,
) -> GeneAssessmentResults:
    """Every aggregated gene with associations, split into novel and known genes and sorted.

    A gene whose reports all went to ``unassessed_reports`` has an aggregation but no
    associations. It has nothing to rate or submit, so the report leaves it out.

    *conn* must use ``sqlite3.Row`` as its row factory.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT COUNT(*) FROM gene_aggregations g
        WHERE NOT EXISTS (SELECT 1 FROM associations a WHERE a.hgnc_id = g.hgnc_id)
        """
    )
    without_associations: int = cursor.fetchone()[0]
    if without_associations:
        logger.info(f"Skipping {without_associations} aggregated genes without associations")
    cursor.execute(
        """
        SELECT hgnc_id, paper_id_mapping, filtered_papers_json, panelapp_context_json,
               existing_panel_reviews_json, unassessed_reports_json, quality_concerns_json
        FROM gene_aggregations g
        WHERE EXISTS (SELECT 1 FROM associations a WHERE a.hgnc_id = g.hgnc_id)
        ORDER BY hgnc_id
        """
    )
    novel_genes: list[GeneAssessment] = []
    known_genes: list[GeneAssessment] = []
    for row in cursor.fetchall():
        gene = load_gene(conn.cursor(), row, hgnc_resolver, target_panel_data, all_panels_data)
        (novel_genes if gene.is_novel else known_genes).append(gene)

    novel_genes.sort(key=novel_gene_sort_key)
    known_genes.sort(key=known_gene_sort_key)
    logger.info(f"Loaded {len(novel_genes)} novel genes, {len(known_genes)} known genes")

    target_panel_names = {
        pid: all_panels_data.panel_names[pid]
        for pid in target_panel_data.panel_ids
        if pid in all_panels_data.panel_names
    }
    return GeneAssessmentResults(
        novel_genes=novel_genes,
        known_genes=known_genes,
        refused_genes=load_refused_genes(cursor, hgnc_resolver),
        target_panel_data=target_panel_data,
        target_panel_names=target_panel_names,
        panels_matched=match_panels_has_run(cursor),
    )


def match_panels_has_run(cursor: sqlite3.Cursor) -> bool:
    """Whether match-panels has sent any request in this database."""
    cursor.execute(
        "SELECT EXISTS(SELECT 1 FROM llm_requests WHERE stage = ?)", (MATCH_PANELS_STAGE,)
    )
    has_run: bool = cursor.fetchone()[0] == 1
    return has_run


def load_gene_assessments(
    db_path: Path,
    panel_date: str,
    hgnc_resolver: HgncResolver,
    target_panel_ids: list[int] | None = None,
) -> GeneAssessmentResults:
    """Load every gene's aggregation and associations, with PanelApp membership at *panel_date*.

    Args:
        db_path: Path to the database
        panel_date: Date (YYYY-MM-DD) to check panel membership at
        target_panel_ids: List of panel IDs to use for novelty detection. If None, uses TARGET_PANEL_IDS.
    """
    logger.info(f"Loading gene aggregations from {db_path}...")

    panelapp_client = PanelAppClient(panel_date)
    target_panel_data = panelapp_client.get_target_panels_genes(target_panel_ids)
    all_panels_data = panelapp_client.get_all_panels_genes()
    logger.info(
        f"Loaded {len(target_panel_data.gene_confidence)} genes from target panels, {len(all_panels_data.gene_to_panels)} genes from all panels"
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return build_gene_assessment_results(
            conn, hgnc_resolver, target_panel_data, all_panels_data
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
                true_positives=[],
                screen_misses=[],
                check_rejections=[],
                refused=[],
                sensitivity_pct=0.0,
                screen_sensitivity_pct=0.0,
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
            p.relevance_assessment_json, p.evidence_extraction_json"""

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
        true_positives: list[DetailedPaper] = []
        screen_misses: list[DetailedPaper] = []
        check_rejections: list[DetailedPaper] = []
        refused: list[DetailedPaper] = []

        for row in all_panel_papers:
            relevance_assessment = None
            try:
                relevance_assessment = json.loads(row["relevance_assessment_json"])
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
                quote_refs=load_quote_refs(cursor, row["doi"]),
                preprint=is_preprint(row["journal"], row["pmid"]),
                pmid=row["pmid"],
            )

            if relevance_assessment.get("refused") is not None:
                refused.append(detailed_paper)
            elif relevance_assessment["relevant"]:
                true_positives.append(detailed_paper)
            elif relevance_assessment["screen"]["relevant"]:
                check_rejections.append(detailed_paper)
            else:
                screen_misses.append(detailed_paper)

        total_assessed = len(true_positives) + len(screen_misses) + len(check_rejections)

        def pct(count: int) -> float:
            return count / total_assessed * 100 if total_assessed > 0 else 0.0

        sensitivity_pct = pct(len(true_positives))
        screen_sensitivity_pct = pct(len(true_positives) + len(check_rejections))
        logger.info(
            f"Panel validation: {len(true_positives)} relevant, {len(screen_misses)} missed by "
            f"the screen, {len(check_rejections)} judged curated by the PanelApp check, "
            f"{len(refused)} refused by both models"
        )
        logger.info(f"Sensitivity: {sensitivity_pct:.1f}% (screen: {screen_sensitivity_pct:.1f}%)")

        return PanelValidationResult(
            total_panel_papers=total_panel_refs,
            panel_papers_in_db=total_assessed + len(refused),
            true_positives=true_positives,
            screen_misses=screen_misses,
            check_rejections=check_rejections,
            refused=refused,
            sensitivity_pct=sensitivity_pct,
            screen_sensitivity_pct=screen_sensitivity_pct,
        )


def load_low_confidence_irrelevant_papers(db_path: Path) -> list[DetailedPaper]:
    """Load papers marked as not relevant with low confidence for manual review."""
    logger.info("Loading low-confidence irrelevant papers...")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all papers with relevance assessments (filtered in Python below)
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
                p.evidence_extraction_json
            FROM papers p
            WHERE p.relevance_assessment_json IS NOT NULL
            ORDER BY p.source_date DESC, p.doi DESC
        """)

        low_confidence_papers = []
        for row in cursor.fetchall():
            relevance_assessment = None
            try:
                relevance_assessment = json.loads(row["relevance_assessment_json"])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse relevance assessment for DOI {row['doi']}")
                continue

            # Screen rejections the screen itself was unsure about. A paper both models
            # refused has no rejection to review.
            if relevance_assessment.get("refused") is not None:
                continue
            screen = relevance_assessment["screen"]
            if not screen["relevant"] and screen["confidence"] == "LOW":
                evidence_extraction = None
                if row["evidence_extraction_json"]:
                    try:
                        evidence_extraction = json.loads(row["evidence_extraction_json"])
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse evidence extraction for DOI {row['doi']}")

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
                    quote_refs=load_quote_refs(cursor, row["doi"]),
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
                    relevance_assessment = json.loads(row["relevance_assessment_json"])
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
                quote_refs={},
                preprint=is_preprint(row["journal"], row["pmid"]),
                pmid=row["pmid"],
            )
            manual_download_papers.append(detailed_paper)

        logger.info(f"Found {len(manual_download_papers)} papers requiring manual download")
        return manual_download_papers


def unreviewed_target_panels(gene: GeneAssessment, target_panel_ids: set[int]) -> list[int]:
    """The target panels holding the gene, when none of them has an expert review; else [].

    Only panels in the current target set count (assess and report may run with
    different ``--target-panel-ids``), so a gene whose stored panels are all outside
    it lands in "Already Reviewed" rather than a false positive.
    """
    panels = [r for r in gene.existing_panel_reviews if r.panel_id in target_panel_ids]
    if any(r.evaluations for r in panels):
        return []
    return [r.panel_id for r in panels]


def _gene_contribution_bucket(
    gene: GeneAssessment, is_novel: bool, target_panel_ids: set[int]
) -> str:
    """Classify a gene by the kind of contribution it represents.

    Returns one of: 'novel', 'rating_upgrade', 'moi_expansion', 'unreviewed',
    'already_reviewed'.
    """
    if is_novel:
        return "novel"
    if gene.existing_rating != gene.new_rating:
        return "rating_upgrade"
    if gene.has_highlighted_new_moi:
        return "moi_expansion"
    if unreviewed_target_panels(gene, target_panel_ids):
        return "unreviewed"
    return "already_reviewed"


_BUCKET_PRIORITY = (
    "novel",
    "rating_upgrade",
    "moi_expansion",
    "unreviewed",
    "already_reviewed",
)


def load_favorite_journal_papers(
    db_path: Path,
    novel_genes: list[GeneAssessment],
    known_genes: list[GeneAssessment],
    target_panel_ids: set[int],
) -> FavoriteJournalSections:
    """Load every initial-search paper from FAVORITE_JOURNALS, bucketed by the
    kind of contribution it made — mirroring the report's gene structure
    (novel / rating upgrade / MoI expansion / unreviewed / already reviewed)
    and the reasons a paper did not contribute (filtered out of aggregate /
    requiring manual download / relevance refused by both models / screened
    out by relevance / off-panel evidence). Expansion papers are excluded
    entirely."""
    gene_anchors: dict[int, FavoriteJournalGeneLink] = {}
    gene_buckets: dict[int, str] = {}
    for gene in novel_genes:
        gene_anchors[gene.hgnc_id] = FavoriteJournalGeneLink(
            hgnc_id=gene.hgnc_id,
            hgnc_symbol=gene.hgnc_symbol,
            anchor=f"#novel-gene-{gene.hgnc_id}",
        )
        gene_buckets[gene.hgnc_id] = _gene_contribution_bucket(
            gene, is_novel=True, target_panel_ids=target_panel_ids
        )
    for gene in known_genes:
        gene_anchors[gene.hgnc_id] = FavoriteJournalGeneLink(
            hgnc_id=gene.hgnc_id,
            hgnc_symbol=gene.hgnc_symbol,
            anchor=f"#known-gene-{gene.hgnc_id}",
        )
        gene_buckets[gene.hgnc_id] = _gene_contribution_bucket(
            gene, is_novel=False, target_panel_ids=target_panel_ids
        )

    # Build doi -> [(hgnc_id, filtered_reason)] from each gene's contributing papers.
    # A paper can appear under multiple genes; filtered_reason is per (gene, paper).
    paper_to_genes: dict[str, list[tuple[int, str | None]]] = {}
    for gene in (*novel_genes, *known_genes):
        for contrib in gene.contributing_papers:
            paper_to_genes.setdefault(contrib.doi, []).append(
                (gene.hgnc_id, contrib.filtered_reason)
            )

    placeholders = ",".join("?" * len(FAVORITE_JOURNALS))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT doi, pmid, title, journal, source_date, download_status,
                   relevance_assessment_json, evidence_extraction_json
            FROM papers
            WHERE journal IN ({placeholders})
              AND source_type = 'initial'
            ORDER BY source_date DESC, doi DESC
            """,
            FAVORITE_JOURNALS,
        )
        rows = cursor.fetchall()

    sections = FavoriteJournalSections(
        novel=[],
        rating_upgrade=[],
        moi_expansion=[],
        unreviewed=[],
        already_reviewed=[],
        filtered=[],
        manual_download=[],
        refused=[],
        not_relevant=[],
        off_panel=[],
    )

    for row in rows:
        doi = row["doi"]
        pmid = row["pmid"]
        contribs = paper_to_genes.get(doi, [])

        contributed_links = sorted(
            (
                gene_anchors[hgnc_id]
                for hgnc_id, reason in contribs
                if reason is None and hgnc_id in gene_anchors
            ),
            key=lambda link: link.hgnc_symbol,
        )
        filtered_reasons = sorted({reason for _, reason in contribs if reason is not None})
        filtered_reason = "; ".join(filtered_reasons) if filtered_reasons else None

        paper = FavoriteJournalPaper(
            doi=doi,
            pmid=pmid,
            title=row["title"],
            journal=row["journal"],
            source_date=row["source_date"],
            preprint=is_preprint(row["journal"], pmid),
            display_id=f"PMID {pmid}" if pmid else doi,
            gene_links=contributed_links,
            filtered_reason=filtered_reason if not contributed_links else None,
        )

        if contributed_links:
            buckets_hit = {gene_buckets[link.hgnc_id] for link in contributed_links}
            for bucket_name in _BUCKET_PRIORITY:
                if bucket_name in buckets_hit:
                    getattr(sections, bucket_name).append(paper)
                    break
            continue

        if paper.filtered_reason:
            sections.filtered.append(paper)
            continue

        # Initial papers must have a relevance assessment to classify further;
        # if missing the paper is in a transient pre-relevance state — drop it.
        if row["relevance_assessment_json"] is None:
            continue

        assessment = json.loads(row["relevance_assessment_json"])
        if assessment.get("refused") is not None:
            sections.refused.append(paper)
            continue
        if not assessment["relevant"]:
            sections.not_relevant.append(paper)
            continue

        if row["download_status"] == "manual_required":
            sections.manual_download.append(paper)
        elif row["evidence_extraction_json"] is not None:
            sections.off_panel.append(paper)
        # else: relevant + downloaded + extraction in flight — drop.

    logger.info(
        "Featured journals: %d total — novel=%d, rating_upgrade=%d, moi_expansion=%d, "
        "unreviewed=%d, already_reviewed=%d, filtered=%d, manual_download=%d, refused=%d, "
        "not_relevant=%d, off_panel=%d",
        sections.total,
        len(sections.novel),
        len(sections.rating_upgrade),
        len(sections.moi_expansion),
        len(sections.unreviewed),
        len(sections.already_reviewed),
        len(sections.filtered),
        len(sections.manual_download),
        len(sections.refused),
        len(sections.not_relevant),
        len(sections.off_panel),
    )
    return sections


def calculate_comprehensive_statistics(
    db_path: Path, results: GeneAssessmentResults, panel_validation: PanelValidationResult
) -> ComprehensiveStats:
    """Calculate enhanced statistics for report."""

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

        all_associations = [a for gene in all_genes for a in gene.associations]

        # Count unreviewed known genes (per current target panel set)
        target_panel_id_set = set(results.target_panel_data.panel_ids)
        unreviewed_count = sum(
            1 for gene in results.known_genes if unreviewed_target_panels(gene, target_panel_id_set)
        )

        # Preprint stats (computed from already-loaded data)
        preprint_dois = {
            p.doi for gene in all_genes for p in gene.contributing_papers if p.preprint
        }
        papers_filtered = len(
            {p.doi for gene in all_genes for p in gene.contributing_papers if p.filtered_reason}
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
            validation_screen_sensitivity_pct=panel_validation.screen_sensitivity_pct,
            screen_misses_count=len(panel_validation.screen_misses),
            check_rejections_count=len(panel_validation.check_rejections),
            true_positives_count=len(panel_validation.true_positives),
            panel_refused_count=len(panel_validation.refused),
            # Source breakdown
            initial_papers=source_counts.get("initial", 0),
            expansion_papers=source_counts.get("expansion", 0),
            # Panel suggestions
            total_panel_suggestions=total_panel_suggestions,
            total_associations=len(all_associations),
            highlighted_new_moi_count=sum(a.new_moi_highlighted for a in all_associations),
            refused_papers_count=len({p.doi for gene in all_genes for p in gene.refused_papers}),
            unreviewed_count=unreviewed_count,
            # Preprint stats
            preprints_relevant=len(preprint_dois),
            papers_filtered=papers_filtered,
        )


def format_inheritance(mode: str, details: str = "") -> str:
    """The mode for display, followed by *details* in parentheses when there are any."""
    formatted_mode = mode.replace("_", " ")
    if details and details.strip():
        return f"{formatted_mode} ({details})"
    return formatted_mode


def combine_inheritance_details(disease_entities: list[dict[str, Any]]) -> str:
    """The distinct non-empty inheritance_details of *disease_entities*, sorted, joined by "; "."""
    details = {
        detail
        for entity in disease_entities
        if (detail := entity.get("inheritance_details")) and detail.strip()
    }
    return "; ".join(sorted(details))


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
    citations: list[dict[str, Any]], contributing_papers: list[DetailedPaper]
) -> list[CitationLink]:
    """Deduplicated viewer links for aggregate citations, each a (doi, quote) pair."""
    papers_by_doi = {paper.doi: paper for paper in contributing_papers}
    links: set[CitationLink] = set()
    for citation in citations:
        paper = papers_by_doi.get(citation["doi"])
        if paper is None:
            continue
        link = citation_link(paper, citation["quote"])
        if link is not None:
            links.add(link)
    return sorted(links, key=lambda link: (link.display_id, link.quote_index))


def prepare_paper_citation_links(
    citations: list[dict[str, Any]], paper: DetailedPaper
) -> list[CitationLink]:
    """Viewer links for one paper's own extraction citations."""
    links = [citation_link(paper, citation["quote"]) for citation in citations]
    return [link for link in links if link is not None]


def papers_for_dois(dois: list[str], papers: list[DetailedPaper]) -> list[DetailedPaper]:
    """The papers among *papers* with these DOIs, in the order of *dois*."""
    by_doi = {paper.doi: paper for paper in papers}
    return [by_doi[doi] for doi in dois if doi in by_doi]


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
    favorite_journal_papers: FavoriteJournalSections,
    template_dir: Path,
    panel_date: str,
    report_id: str,
    target_panel_ids: list[int],
    target_panel_names: dict[int, str],
    refused_genes: list[RefusedGene],
    *,
    panels_matched: bool,
    panelapp_integration: bool,
) -> str:
    """Generate HTML report directly using Jinja2 templates."""

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

    # Split no_change into three: a highlighted new-MoI association, unreviewed on the
    # target panels, and the remainder (already reviewed by at least one expert).
    target_panel_id_set = set(target_panel_ids)
    no_change_new_moi: list[GeneAssessment] = []
    no_change_unreviewed: list[GeneAssessment] = []
    no_change_reviewed: list[GeneAssessment] = []
    for gene in no_change:
        if gene.has_highlighted_new_moi:
            no_change_new_moi.append(gene)
        elif unreviewed_target_panels(gene, target_panel_id_set):
            no_change_unreviewed.append(gene)
        else:
            no_change_reviewed.append(gene)

    known_categories = KnownGeneCategories(
        red_to_green=red_to_green,
        amber_to_green=amber_to_green,
        red_to_amber=red_to_amber,
        no_change_new_moi=no_change_new_moi,
        no_change_unreviewed=no_change_unreviewed,
        no_change_reviewed=no_change_reviewed,
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
    env.filters["paper_citation_links"] = prepare_paper_citation_links
    env.filters["get_variant_flag"] = get_variant_frequency_flag
    env.filters["confidence_to_color"] = panelapp_confidence_to_color
    # Double-encode: first quote produces the on-disk filename (e.g. 10.1038%2Fxyz),
    # second quote escapes the % for use in file:/// URLs so browsers don't
    # decode %2F back to / (e.g. 10.1038%252Fxyz → opens 10.1038%2Fxyz.pdf).
    env.filters["paper_key"] = doi_to_key
    env.filters["short_id"] = lambda display_id: display_id.removeprefix("PMID ")
    env.filters["derive_moi"] = derive_aggregate_moi
    env.filters["derive_moi_details"] = combine_inheritance_details
    env.filters["papers_for_dois"] = papers_for_dois
    env.filters["unreviewed_target_panels"] = lambda gene: unreviewed_target_panels(
        gene, target_panel_id_set
    )

    # Add custom sort filter that handles None values
    def sort_by_rating_count(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sort disease entities by the rating-relevant independent family count (None as 0)."""
        return sorted(entities, key=lambda e: e.get("independent_family_count") or 0, reverse=True)

    env.filters["sort_by_rating_count"] = sort_by_rating_count

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
        favorite_journal_papers=favorite_journal_papers,
        panel_date=panel_date,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        custom_css=custom_css,
        panelapp_integration=panelapp_integration,
        report_config_json=report_config_json,
        target_panel_names=target_panel_names,
        refused_genes=refused_genes,
        panels_matched=panels_matched,
        min_families_for_moi_expansion=MIN_FAMILIES_FOR_MOI_EXPANSION,
        model_name=MODEL_NAME,
        fallback_model_name=FALLBACK_MODEL_NAME,
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
    papers_dir: Path = typer.Option(
        Path("data/papers"),
        "--papers-dir",
        help="Directory containing the papers' PDFs",
    ),
    viewer_dir: Path = typer.Option(
        Path("viewer/dist"),
        "--viewer-dir",
        help="Built PDF viewer page (npm --prefix viewer ci && npm --prefix viewer run build)",
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
    """Generate a self-contained report directory: the HTML, its papers, and the PDF viewer."""

    # Validate inputs
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if not (viewer_dir / "index.html").exists():
        logger.error(
            f"Viewer build not found in {viewer_dir}; run "
            "`npm --prefix viewer ci && npm --prefix viewer run build`"
        )
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
    # Use actual target_panel_ids from results (handles None -> defaults)
    actual_panel_ids = list(results.target_panel_data.panel_ids)
    favorite_journal_papers = load_favorite_journal_papers(
        db_path, results.novel_genes, results.known_genes, set(actual_panel_ids)
    )
    statistics = calculate_comprehensive_statistics(db_path, results, panel_validation)

    html_content = generate_html_report(
        novel_genes=results.novel_genes,
        known_genes=results.known_genes,
        statistics=statistics,
        panel_validation=panel_validation,
        low_confidence_papers=low_confidence_papers,
        manual_download_papers=manual_download_papers,
        favorite_journal_papers=favorite_journal_papers,
        template_dir=template_dir,
        panel_date=panel_date,
        report_id=report_id,
        target_panel_ids=actual_panel_ids,
        target_panel_names=results.target_panel_names,
        refused_genes=results.refused_genes,
        panels_matched=results.panels_matched,
        panelapp_integration=panelapp_integration,
    )

    # Save HTML as index.html
    index_file = output_dir / "index.html"
    index_file.write_text(html_content)

    all_genes = results.novel_genes + results.known_genes
    dois = sorted({paper.doi for gene in all_genes for paper in gene.contributing_papers})
    write_viewer_package(output_dir, db_path, dois, papers_dir, viewer_dir)

    logger.info("Package created successfully!")
    logger.info(f"   Output: {output_dir}")


def write_viewer_package(
    output_dir: Path, db_path: Path, dois: list[str], papers_dir: Path, viewer_dir: Path
) -> None:
    """Copy the viewer page and write each paper's PDF link and citations file.

    PDFs are symlinked (``aws s3 sync`` uploads the target); each
    ``citations/<key>.json`` lists the paper's quotes as :func:`paper_quotes`
    orders them, which is how :func:`load_quote_refs` numbers them.
    """
    viewer_out = output_dir / "viewer"
    if viewer_out.exists():
        shutil.rmtree(viewer_out)
    shutil.copytree(viewer_dir, viewer_out)

    papers_out = output_dir / "papers"
    citations_out = output_dir / "citations"
    papers_out.mkdir(exist_ok=True)
    citations_out.mkdir(exist_ok=True)

    missing: list[str] = []
    with sqlite3.connect(db_path) as conn:
        for doi in dois:
            pdf = doi_to_path(doi, papers_dir, ".pdf")
            if not pdf.exists():
                missing.append(doi)
                continue
            link = doi_to_path(doi, papers_out, ".pdf")
            if link.is_symlink():
                link.unlink()
            link.symlink_to(pdf.resolve())
            (title,) = conn.execute("SELECT title FROM papers WHERE doi = ?", (doi,)).fetchone()
            quotes = [
                {"quote": quote, "bboxes": bboxes}
                for quote, bboxes in paper_quotes(conn.cursor(), doi)
            ]
            doi_to_path(doi, citations_out, ".json").write_text(
                json.dumps({"doi": doi, "title": title, "quotes": quotes})
            )
    logger.info(f"   {len(dois) - len(missing)} papers linked with citations and highlights")
    if missing:
        logger.warning(f"Missing PDFs for {len(missing)} papers, e.g. {missing[:5]}")


if __name__ == "__main__":
    app()
