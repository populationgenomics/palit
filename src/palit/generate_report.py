#!/usr/bin/env python3
"""Generate gene-centric HTML assessment reports from aggregate assessments."""

import json
import logging
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from enum import StrEnum
from itertools import groupby
from pathlib import Path
from typing import Any

import markdown
import nh3
import typer
from anthropic.types import Message
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from palit.assess_genes import STAGE as ASSESS_GENES_STAGE
from palit.assess_relevance import STAGE as RELEVANCE_STAGE
from palit.extract_evidence import STAGE as EXTRACTION_STAGE
from palit.extract_evidence import evaluations_of_gene
from palit.gencc import MondoRef
from palit.hgnc import HgncResolver
from palit.llm import (
    FALLBACK_MODEL,
    FALLBACKS,
    MODEL,
    MODEL_NAMES,
    ResultStatus,
    parse_json_output,
)
from palit.map_mondo import STAGE as MAP_MONDO_STAGE
from palit.match_panels import STAGE as MATCH_PANELS_STAGE
from palit.panel_entries import PanelEntry, entries_by_gene, reviewed_in_full
from palit.panelapp_client import (
    AllPanelsData,
    PanelAppClient,
    PanelGeneData,
    PanelPublications,
    ordered_panel_publications,
)
from palit.panelapp_integration import (
    RATING_CONFIDENCE,
    Finding,
    FindingKind,
    MondoMatch,
    MondoTerm,
    PrefillAssociation,
    PrefillData,
    PrefillEntry,
    RelationStatus,
    calculate_association_rating,
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
from palit.run_corpus import CITED_PAPER, CORPUS_PAPER, RELEVANT_PAPER
from palit.variants import GeneVariant, GnomadFrequency, GnomadMiss, load_gene_variants

logger = logging.getLogger(__name__)

# gnomAD counts above which a variant is flagged, by the MoI of its association
GNOMAD_HET_THRESHOLD = 30  # heterozygotes; monoallelic and any other MoI
GNOMAD_HOM_THRESHOLD = 15  # homozygotes; biallelic
GNOMAD_HEMI_THRESHOLD = 30  # hemizygotes; X-linked

# A new_moi association is highlighted only from this many independent families on
MIN_FAMILIES_FOR_MOI_EXPANSION = 2

# The heading of a report without a scope panel
MONTHLY_REPORT_TITLE = "PanelApp Literature Assessment Report"

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
        for criterion in entity["evidence_assessments"]:
            citations += criterion["citations"]
        for concern in entity["quality_concerns"]:
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


@dataclass(frozen=True)
class ReportVariant:
    """One of the gene's distinct variants, a row of the report's variant tables."""

    variant: GeneVariant
    citations: list[CitationLink]  # its reports' quotes, by display ID, then quote index

    @property
    def variant_id(self) -> str:
        return self.variant.variant_id

    @property
    def hgvs_forms(self) -> list[tuple[str, str]]:
        """The distinct (HGVS c., HGVS p.) forms of its reports; empty when not normalised.

        Papers may number the variant on different transcripts, so there can be several.
        """
        return list(
            dict.fromkeys(
                (r.normalization.hgvs_c, r.normalization.hgvs_p)
                for r in self.variant.reports
                if r.normalization is not None
            )
        )

    @property
    def original_text(self) -> str:
        """The variant as the papers wrote it."""
        return " / ".join(dict.fromkeys(r.original_text for r in self.variant.reports))

    @property
    def frequency(self) -> GnomadFrequency | None:
        """Its gnomAD figures; None when it is absent from gnomAD or its lookup failed."""
        gnomad = self.variant.gnomad
        return gnomad if isinstance(gnomad, GnomadFrequency) else None

    @property
    def not_in_gnomad(self) -> bool:
        return self.variant.gnomad == GnomadMiss.NOT_FOUND

    @property
    def lookup_failed(self) -> bool:
        return self.variant.gnomad == GnomadMiss.LOOKUP_FAILED

    @property
    def faf95_population(self) -> str | None:
        """The name of the FAF95 popmax population, where gnomAD gives one."""
        frequency = self.frequency
        if frequency is None or frequency.faf95_popmax_population is None:
            return None
        code = frequency.faf95_popmax_population
        return POPULATION_NAMES.get(code, code)

    @property
    def gnomad_link(self) -> str | None:
        """Its gnomAD page; None when its lookup failed, as its key is then the paper's text."""
        if self.lookup_failed:
            return None
        return f"https://gnomad.broadinstitute.org/variant/{self.variant_id}?dataset=gnomad_r4"


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
    # The paper's evaluations of the gene whose section shows its evidence (see
    # evaluations_of_gene); empty where the paper is shown without evidence.
    gene_evaluations: list[dict[str, Any]]
    quote_refs: dict[str, QuoteRef]  # this paper's quotes -> position in its citations file
    cited_for_run_gene: bool  # PanelApp cites it for one of the run's genes
    preprint: bool = False
    pmid: int | None = None  # For PubMed display links
    display_id: str = ""  # "PMID {pmid}" for published papers, AuthorYear for preprints
    paper_gene_symbol: str | None = None
    filtered_reason: str | None = None  # Set when paper was excluded from assessment

    @property
    def passed_screen(self) -> bool:
        """The scope screen passed the paper."""
        return self.relevance_assessment is not None and self.relevance_assessment["relevant"]

    @property
    def entered_through_citation(self) -> bool:
        """The run took the paper in because PanelApp cites it for one of the run's genes,
        not because the scope screen passed it. As in load_paper_sources, a paper the
        screen passed counts as relevant, and a cited expansion paper as cited."""
        return self.cited_for_run_gene and not self.passed_screen


@dataclass(frozen=True)
class ManualDownloadPaper:
    """A paper awaiting manual download, with the figure the section orders it by."""

    paper: DetailedPaper
    # The journal's OpenAlex two-year mean citedness (journal_metrics); None for a
    # preprint, or when the paper has no journal ISSN or OpenAlex gives no figure.
    journal_citedness: float | None


@dataclass(frozen=True)
class FavoriteJournalGeneLink:
    """Hyperlink from a featured-journal paper to a gene's anchor in this report."""

    hgnc_id: int
    hgnc_symbol: str
    anchor: str  # the gene article's id: "novel-gene-{id}" or "known-gene-{id}"


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

    Only initial papers are surfaced here; expansion papers (added via
    citation chasing) are excluded entirely. Transient pipeline states
    (downloaded but not yet extracted, etc.) are also dropped: they're not
    actionable and just clutter the section."""

    novel: list[FavoriteJournalPaper]
    findings: list[FavoriteJournalPaper]
    unreviewed: list[FavoriteJournalPaper]
    already_reviewed: list[FavoriteJournalPaper]
    filtered: list[FavoriteJournalPaper]
    manual_download: list[FavoriteJournalPaper]
    refused: list[FavoriteJournalPaper]  # its relevance assessment was refused for good
    # Rejected by the scope screen, not read and not cited for a run gene; always empty in
    # a panel-scoped run
    not_relevant: list[FavoriteJournalPaper]
    read_no_evidence: list[FavoriteJournalPaper]  # read in full, no evidence in the report

    @property
    def total(self) -> int:
        return (
            len(self.novel)
            + len(self.findings)
            + len(self.unreviewed)
            + len(self.already_reviewed)
            + len(self.filtered)
            + len(self.manual_download)
            + len(self.refused)
            + len(self.not_relevant)
            + len(self.read_no_evidence)
        )


@dataclass(frozen=True)
class PanelMatch:
    """A panel that match-panels matched one association to."""

    panel_id: int
    panel_name: str
    rationale: str
    gene_on_panel: bool  # the gene already has an entry on this panel


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
    # A model without a fallback refused the latest request; the stage skips it from now on
    REFUSED = "refused"
    FAILED = "failed"  # requests ran but none gave a valid result; a rerun retries it


def stage_state(stored: bool, latest_status: str | None, latest_model: str | None) -> StageState:
    """The state of a stage for one association, from its stored result and latest request.

    A latest refusal by a model with a fallback counts as failed: the next
    attempt sends the association to the fallback.
    """
    if stored:
        return StageState.STORED
    if latest_status is None:
        return StageState.NOT_RUN
    if latest_status == ResultStatus.REFUSED and latest_model not in FALLBACKS:
        return StageState.REFUSED
    return StageState.FAILED


@dataclass
class ReportAssociation:
    """One gene-disease-MoI association of a gene, prepared for display."""

    id: int
    position: int  # order in the model output
    assessment: dict[str, Any]  # the stored association JSON, paper IDs in display form
    disease_label: str  # the GenCC term's title when reused, else the proposed name
    mondo: MondoTerm | None  # None unless mondo_mapping is STORED
    mondo_mapping: StageState  # STORED for a reused GenCC term too
    rating: int  # computed from this association's criteria: 3 GREEN, 2 AMBER, 1 RED
    disputes: list[DisputeRef]
    variants: list[ReportVariant]  # the association's variants, in the gene's order
    # None unless panel_matching is STORED; panels the gene is not on first, each group by name
    matched_panels: list[PanelMatch] | None
    panel_matching: StageState
    # The names of the target panels holding the gene when they have reviewed the
    # association's papers in full (see reviewed_in_full); empty otherwise
    reviewed_on: tuple[str, ...]
    # Why the association fits the scope panel of a panel-scoped run; None otherwise
    scope_rationale: str | None

    @property
    def already_reviewed(self) -> bool:
        """The target panels cite all the association's papers and have reviewed the gene."""
        return bool(self.reviewed_on)

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
class FindingGroup:
    """A gene's findings with the same badge: one badge in the gene header, one ToC chip.

    Its associations are adjacent in the gene's box, so the badge links to the first.
    """

    kind: FindingKind
    rating: int
    from_rating: int | None  # upgrades only, as in Finding
    associations: list[ReportAssociation]  # in report order

    @property
    def association_id(self) -> int:
        """The association the badge links to: the group's first."""
        return self.associations[0].id

    @property
    def count(self) -> int:
        return len(self.associations)


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
    # The associations with findings in the order of their findings, then the others
    # by corpus rating and independent families
    associations: list[ReportAssociation]
    unassessed_reports: list[dict[str, Any]]  # [{phenotype, inheritance_mode, dois, reason}]
    current_rating: int | None  # the gene's highest rating on the target panels; None if novel
    findings: list[Finding]  # in badge order, see gene_findings
    contributing_papers: list[DetailedPaper]
    unassociated_variants: list[ReportVariant]  # the gene's variants that no association lists
    prefill: PrefillData
    target_entries: tuple[PanelEntry, ...]  # the gene's entries on the target panels
    refused_papers: list[RefusedPaper]

    @property
    def is_novel(self) -> bool:
        """The gene is on none of the target panels."""
        return self.current_rating is None

    @property
    def unreviewed_target_panels(self) -> list[int]:
        """The target panels on which nobody has reviewed an entry of the gene."""
        return list(dict.fromkeys(e.panel_id for e in self.target_entries if e.unreviewed))

    @property
    def anchor(self) -> str:
        """The id of the gene's article in the report."""
        return f"{'novel' if self.is_novel else 'known'}-gene-{self.hgnc_id}"

    @property
    def finding_groups(self) -> list[FindingGroup]:
        """The findings merged by badge, strongest first."""
        by_id = {a.id: a for a in self.associations}
        return [
            FindingGroup(kind, rating, from_rating, [by_id[f.association_id] for f in findings])
            for (kind, rating, from_rating), findings in groupby(self.findings, lambda f: f.badge)
        ]

    @property
    def finding_association_ids(self) -> set[int]:
        return {f.association_id for f in self.findings}

    @property
    def finding_associations(self) -> list[ReportAssociation]:
        """The associations with findings, in badge order."""
        finding_ids = self.finding_association_ids
        return [a for a in self.associations if a.id in finding_ids]

    @property
    def other_associations(self) -> list[ReportAssociation]:
        """The associations without findings, by corpus rating."""
        finding_ids = self.finding_association_ids
        return [a for a in self.associations if a.id not in finding_ids]

    @property
    def other_associations_curated(self) -> bool:
        """PanelApp Australia curates every association without a finding."""
        return all(a.relation_status == "existing" for a in self.other_associations)

    @property
    def top_rating(self) -> int:
        """The highest corpus rating of the gene's associations."""
        return max(a.rating for a in self.associations)

    @property
    def prefill_json(self) -> str:
        """The prefill as JSON, for the prefill button's data-prefill attribute."""
        return json.dumps(asdict(self.prefill))


@dataclass(frozen=True)
class ReadPaper:
    """A paper with an extraction that the run read for a gene."""

    doi: str
    pmid: int | None
    title: str
    # Whether its extraction holds a disease entity of the gene
    disease_extracted: bool


class PapersRead(StrEnum):
    """What the run read for a gene that has no association in the report."""

    NONE = "none"  # no paper
    NO_DISEASE = "no_disease"  # papers, none with a disease entity of the gene
    NO_ASSOCIATION = "no_association"  # papers with a disease entity, but no association


@dataclass(frozen=True)
class UnreviewedEntry:
    """A target panel's entry that nobody has reviewed, for a gene with no association
    in this report."""

    hgnc_id: int
    hgnc_symbol: str
    panel_id: int
    panel_name: str
    rating: str  # GREEN, AMBER or RED
    papers: tuple[ReadPaper, ...]  # the papers the run read for the gene, newest first

    @property
    def papers_read(self) -> PapersRead:
        if not self.papers:
            return PapersRead.NONE
        if any(paper.disease_extracted for paper in self.papers):
            return PapersRead.NO_ASSOCIATION
        return PapersRead.NO_DISEASE


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
    scope_panel_id: int | None  # the one target panel of a panel-scoped run; None otherwise
    # by HGNC symbol, then panel name
    unreviewed_without_evidence: list[UnreviewedEntry]


@dataclass
class PanelValidationResult:
    """Panel publication validation results."""

    total_panel_papers: int
    panel_papers_in_db: int  # with a relevance result, including the refused ones
    true_positives: list[DetailedPaper]
    screen_misses: list[DetailedPaper]  # rejected by the scope screen
    # Of the screen misses, those read in full because PanelApp cites them for a run gene
    screen_misses_read: int
    refused: list[DetailedPaper]  # refused for good; left out of the sensitivity
    sensitivity_pct: float  # share of the assessed (not refused) ones assessed relevant


@dataclass(frozen=True)
class PaperSources:
    """The papers with an extraction, by how they entered the corpus (see palit.run_corpus)."""

    relevant: int  # passed by the scope screen
    cited: int  # not passed by the screen, but PanelApp cites them for one of the run's genes
    expansion: int  # the other expansion papers: tournament picks and referenced papers
    # Initial papers neither passed by the screen nor cited for one of the run's genes
    outside_corpus: int


@dataclass
class ComprehensiveStats:
    """All statistics for the report."""

    # Gene assessment stats
    total_genes_assessed: int
    new_genes_count: int  # on none of the target panels
    known_genes_with_findings: int
    total_contributing_papers: int

    # Panel validation stats
    total_panel_papers: int
    panel_papers_in_db: int
    validation_sensitivity_pct: float
    screen_misses_count: int
    screen_misses_read_count: int  # read in full because PanelApp cites them for a run gene
    true_positives_count: int
    panel_refused_count: int  # relevance refused for good, not in the sensitivity

    paper_sources: PaperSources

    # Distinct (gene, panel) pairs an association matched where the gene is not on the panel
    total_panel_suggestions: int

    # Associations, and the findings among them by kind
    total_associations: int
    new_disease_findings: int
    new_moi_findings: int
    upgrade_findings: int
    # Associations that would be findings but for the target panels having reviewed all
    # their papers already
    already_reviewed_findings: int

    # Papers of assessed genes that both models refused in extract-evidence
    refused_papers_count: int

    # Known genes on the target panel with zero expert reviews
    unreviewed_count: int

    # Preprint stats
    preprints_relevant: int
    papers_filtered: int


@dataclass(frozen=True)
class GeneGroup:
    """Genes placed together: one ToC section, and one subsection of the report body."""

    key: str  # the ToC section's data-section-key, e.g. "finding-green"
    title: str  # a rated group's title is followed by its rating badge
    rating: int | None  # None for the groups of known genes without findings
    genes: list[GeneAssessment]
    collapsed: bool  # its gene list starts collapsed in the ToC
    preselected: bool  # checked by default in the bulk assignment panel

    @property
    def label(self) -> str:
        """The group's name in the bulk assignment panel."""
        if self.rating is None:
            return self.title
        return f"{self.title} {panelapp_confidence_to_color(self.rating).upper()}"


@dataclass(frozen=True)
class GeneGroups:
    """The report's genes, each in exactly one group; groups without genes are left out."""

    new_genes: list[GeneGroup]  # by top association rating: GREEN, AMBER, RED
    with_findings: list[GeneGroup]  # known genes by best finding rating: GREEN, AMBER, RED
    without_findings: list[GeneGroup]  # known genes unreviewed on the target panels
    already_reviewed: list[GeneGroup]  # known genes reviewed on the target panels


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


def load_report_variants(
    cursor: sqlite3.Cursor, hgnc_id: int, papers: list[DetailedPaper]
) -> list[ReportVariant]:
    """The gene's distinct variants reported by *papers*, with links to their quotes.

    *papers* are the papers the gene's aggregation read, so these are the variants
    of its prompt's variant table. Variants with gnomAD figures come first, then
    those absent from gnomAD, then those whose lookup failed, each by variant_id.
    """
    papers_by_doi = {paper.doi: paper for paper in papers}

    def sort_key(variant: GeneVariant) -> tuple[bool, bool, str]:
        return (
            variant.gnomad == GnomadMiss.LOOKUP_FAILED,
            variant.gnomad == GnomadMiss.NOT_FOUND,
            variant.variant_id,
        )

    report_variants = []
    for variant in sorted(load_gene_variants(cursor, hgnc_id, list(papers_by_doi)), key=sort_key):
        links = {citation_link(papers_by_doi[r.doi], r.quote) for r in variant.reports}
        report_variants.append(
            ReportVariant(
                variant=variant,
                citations=sorted(
                    (link for link in links if link is not None),
                    key=lambda link: (link.display_id, link.quote_index),
                ),
            )
        )
    return report_variants


def association_variants(
    assessment: dict[str, Any], gene_variants: list[ReportVariant]
) -> list[ReportVariant]:
    """The gene's variants that the association lists, in the gene's variant order.

    The association names its variants by ``variant_frequencies.variant_id``,
    each of which must be among the gene's variants.
    """
    listed = set(assessment["variants"])
    unknown = listed - {variant.variant_id for variant in gene_variants}
    if unknown:
        raise ValueError(
            f"Association {assessment['description']!r} lists variants that are not among "
            f"its gene's variant rows: {sorted(unknown)}"
        )
    return [variant for variant in gene_variants if variant.variant_id in listed]


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


def association_finding(
    association: ReportAssociation, current_rating: int | None, scoped: bool
) -> Finding | None:
    """The association's finding, or None when it adds nothing to the target panels.

    *current_rating* is the gene's highest rating on the target panels, None
    when it is on none of them. An association whose papers the target panels
    have all reviewed already is no finding, whatever its relation and rating.
    Otherwise, every association of a gene on none of the target panels is a new
    disease, whatever PanelApp Australia curates on other panels. On a gene on a
    target panel, a new disease is a finding, and a new MoI once highlighted. An
    existing association is an upgrade when its corpus rating is above the rating
    it starts from. In a panel-scoped run (*scoped*), that is the gene's rating
    on the scope panel: a panel named after a clinical feature rates how reliably
    the gene causes the feature, not each disease. Otherwise it is PanelApp's
    rating of the association, or the gene's current rating when that is unknown.
    """
    if association.already_reviewed:
        return None
    if current_rating is None:
        return Finding(FindingKind.NEW_DISEASE, association.rating, association.id, None)
    status = association.relation_status
    if status == "new_disease":
        return Finding(FindingKind.NEW_DISEASE, association.rating, association.id, None)
    if status == "new_moi":
        if not association.new_moi_highlighted:
            return None
        return Finding(FindingKind.NEW_MOI, association.rating, association.id, None)
    stated: str | None = association.assessment["panelapp_relation"]["existing_rating"]
    from_rating = current_rating if scoped or stated is None else RATING_CONFIDENCE[stated]
    if association.rating <= from_rating:
        return None
    return Finding(FindingKind.UPGRADE, association.rating, association.id, from_rating)


def gene_findings(
    associations: list[ReportAssociation], current_rating: int | None, scoped: bool
) -> list[Finding]:
    """The findings of a gene's associations in badge order: strongest first, those with
    the same badge together.

    *associations* are by association_sort_key. A badge is placed by its first
    finding among them, so the badges go by rating and then by their strongest
    association; within a badge, the findings keep the order of *associations*.
    """
    by_badge: dict[tuple[FindingKind, int, int | None], list[Finding]] = {}
    for association in associations:
        finding = association_finding(association, current_rating, scoped)
        if finding is not None:
            by_badge.setdefault(finding.badge, []).append(finding)
    return [finding for findings in by_badge.values() for finding in findings]


def report_order(
    associations: list[ReportAssociation], findings: list[Finding]
) -> list[ReportAssociation]:
    """The associations with findings in the order of *findings*, then the others in
    their order in *associations*."""
    by_id = {a.id: a for a in associations}
    finding_ids = [f.association_id for f in findings]
    return [by_id[i] for i in finding_ids] + [a for a in associations if a.id not in finding_ids]


def group_genes(
    novel_genes: list[GeneAssessment],
    known_genes: list[GeneAssessment],
) -> GeneGroups:
    """Place each gene in one group, each group's genes by HGNC symbol.

    New genes go by their top association rating, known genes with findings by
    the rating of their strongest finding, and known genes without findings by
    whether nobody has reviewed one of their entries on the target panels. The
    reviewed genes start collapsed in the ToC; the groups of known genes without
    findings are not preselected for bulk assignment.
    """
    novel = sorted(novel_genes, key=lambda g: g.hgnc_symbol)
    known = sorted(known_genes, key=lambda g: g.hgnc_symbol)

    def by_rating(
        key: str,
        title: str,
        genes: list[GeneAssessment],
        rating_of: Callable[[GeneAssessment], int],
    ) -> list[GeneGroup]:
        groups = [
            GeneGroup(
                key=f"{key}-{panelapp_confidence_to_color(rating).lower()}",
                title=title,
                rating=rating,
                genes=[g for g in genes if rating_of(g) == rating],
                collapsed=False,
                preselected=True,
            )
            for rating in (3, 2, 1)
        ]
        return [group for group in groups if group.genes]

    unreviewed: list[GeneAssessment] = []
    reviewed: list[GeneAssessment] = []
    for gene in known:
        if not gene.findings:
            (unreviewed if gene.unreviewed_target_panels else reviewed).append(gene)

    def single(
        key: str, title: str, genes: list[GeneAssessment], collapsed: bool
    ) -> list[GeneGroup]:
        if not genes:
            return []
        return [GeneGroup(key, title, None, genes, collapsed=collapsed, preselected=False)]

    return GeneGroups(
        new_genes=by_rating("new", "New gene", novel, lambda g: g.top_rating),
        with_findings=by_rating(
            "finding",
            "Best finding",
            [g for g in known if g.findings],
            lambda g: g.findings[0].rating,
        ),
        without_findings=single("unreviewed", "Unreviewed on target panel", unreviewed, False),
        already_reviewed=single("reviewed", "Already reviewed", reviewed, True),
    )


def load_contributing_papers(
    cursor: sqlite3.Cursor, hgnc_id: int, filtered_doi_reasons: dict[str, str]
) -> list[DetailedPaper]:
    """Papers with an extraction for the gene, initial and expansion alike, newest first.

    One row per paper, with the symbol from the extraction's gene mention (a
    relevance-stage mention may carry an older symbol of the same gene).
    """
    cursor.execute(
        f"""
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
            {CITED_PAPER} AS cited_for_run_gene,
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
        doi = paper_row["doi"]
        gene_evaluations = evaluations_of_gene(
            json.loads(paper_row["evidence_extraction_json"]), hgnc_id, doi
        )
        # A paper may mention the gene but only have evidence for other genes.
        if not gene_evaluations:
            continue

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
                gene_evaluations=gene_evaluations,
                quote_refs=load_quote_refs(cursor, doi),
                cited_for_run_gene=bool(paper_row["cited_for_run_gene"]),
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
    when the gene's tournament selected it, when PanelApp cites it for the gene
    (``panelapp_citations``), or when it was added for the gene as a referenced
    paper (``source_details``).
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
              OR EXISTS (
                  SELECT 1 FROM panelapp_citations pc
                  WHERE pc.paper_doi = p.doi AND pc.hgnc_id = ?
              )
              OR p.source_details = CAST(? AS TEXT)
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


def _mondo_term(row: sqlite3.Row, gencc_terms: list[dict[str, Any]]) -> MondoTerm | None:
    """The association row's MONDO term, or None while it has none."""
    mondo_id: str | None = row["mondo_id"]
    if mondo_id is None:
        return None
    match: MondoMatch = row["mondo_match"]
    if match == "gencc":
        term = next(t for t in gencc_terms if t["mondo_id"] == mondo_id)
        return MondoTerm(
            mondo_id=mondo_id,
            label=row["mondo_label"],
            match=match,
            rationale="",
            obsolete=term["obsolete"],
            replaced_by=tuple(MondoRef(**ref) for ref in term["replaced_by"]),
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
    matched_panels_json: str | None,
    hgnc_id: int,
    all_panels_data: AllPanelsData,
    display_ids: dict[str, str],
) -> list[PanelMatch] | None:
    """An association's matched panels, or None while it has none stored.

    The rationales name papers by their *display_ids*, as the summary does. The
    panels the gene is not on come first; within each group, by panel name.
    """
    if matched_panels_json is None:
        return None
    current_panels = all_panels_data.gene_to_panels.get(hgnc_id, set())
    stored = replace_paper_ids_for_display(
        {"matched_panels": json.loads(matched_panels_json)}, display_ids
    )["matched_panels"]
    matches = []
    for match in stored:
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
    return sorted(matches, key=lambda m: (m.gene_on_panel, m.panel_name))


def load_associations(
    cursor: sqlite3.Cursor,
    hgnc_id: int,
    panelapp_context: dict[str, Any],
    display_ids: dict[str, str],
    all_panels_data: AllPanelsData,
    gene_variants: list[ReportVariant],
    target_entries: tuple[PanelEntry, ...],
    pmid_by_doi: dict[str, int | None],
    scoped: bool,
) -> list[ReportAssociation]:
    """The gene's associations, by corpus rating and then independent family count.

    Each association carries its variants among *gene_variants* and the state of
    map-mondo and match-panels for it, from its latest request of each stage. Both
    stages send immediate requests only, which are recorded once they have completed.
    *target_entries* are the gene's entries on the target panels, against which each
    association's papers, with their PMIDs from *pmid_by_doi*, are checked for having
    been reviewed already. In a panel-scoped run (*scoped*), each association carries
    its ``scope_rationale``. Raises ValueError for an association aggregated without the
    scope panel in a panel-scoped run, or with one in a run without it.
    """
    target_panel_names = tuple(dict.fromkeys(e.panel_name for e in target_entries))

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
        mondo = _mondo_term(row, panelapp_context["gencc_terms"])
        reviewed = reviewed_in_full(
            target_entries, {doi: pmid_by_doi[doi] for doi in assessment["dois"]}
        )
        scope_rationale: str | None = assessment.get("scope_rationale")
        if (scope_rationale is None) == scoped:
            raise ValueError(
                f"HGNC:{hgnc_id} association {row['id']} has no scope_rationale; aggregate the "
                "gene again with assess-genes --scope-panel-id"
                if scoped
                else f"HGNC:{hgnc_id} association {row['id']} was aggregated with a scope "
                "panel; generate the report with the run's --scope-panel-id"
            )
        associations.append(
            ReportAssociation(
                id=row["id"],
                position=row["position"],
                assessment=assessment,
                disease_label=(
                    mondo.label
                    if mondo is not None and mondo.match == "gencc"
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
                variants=association_variants(assessment, gene_variants),
                matched_panels=_panel_matches(
                    row["matched_panels_json"], hgnc_id, all_panels_data, display_ids
                ),
                panel_matching=stage_state(
                    row["matched_panels_json"] is not None,
                    row["matching_status"],
                    row["matching_model"],
                ),
                reviewed_on=target_panel_names if reviewed else (),
                scope_rationale=scope_rationale,
            )
        )
    return sorted(associations, key=association_sort_key)


@dataclass(frozen=True)
class PrefillSelection:
    """The associations a gene's prefill describes, and whether they rate the form."""

    associations: list[ReportAssociation]  # in report order
    rated: bool  # see prepare_prefill_data


def prefill_associations(
    associations: list[ReportAssociation], findings: list[Finding]
) -> PrefillSelection:
    """The associations a gene's prefill describes, in report order.

    The prefill describes what the report adds to the target panels: the
    associations with findings, which for a gene on no target panel are all of
    them. A gene without findings describes the associations whose papers the
    target panels have not all reviewed yet, so its form isn't empty: a review is
    often the first for an unreviewed gene. When the target panels have reviewed
    the papers of every association, the form describes them all but keeps the
    gene's rating on the panel.
    """
    if findings:
        finding_ids = {f.association_id for f in findings}
        return PrefillSelection([a for a in associations if a.id in finding_ids], rated=True)
    unreviewed = [a for a in associations if not a.already_reviewed]
    if unreviewed:
        return PrefillSelection(unreviewed, rated=True)
    return PrefillSelection(associations, rated=False)


def prefill_entry(
    target_entries: tuple[PanelEntry, ...], panel_id: int, rating: int
) -> PrefillEntry:
    """The gene's entry on *panel_id*, which its review form extends; *rating* is the
    gene's confidence level there.

    *target_entries* are the gene's entries on the target panels. The review is of the
    gene entry; a gene that the panel holds only as repeat expansions gets its first
    repeat-expansion entry there. Raises ValueError when the panel holds neither.
    """
    on_panel = [e for e in target_entries if e.panel_id == panel_id]
    if not on_panel:
        raise ValueError(f"The gene has no entry on panel {panel_id} for its review form")
    gene_entries = [e for e in on_panel if e.entity_type == "gene"]
    entry = (gene_entries or on_panel)[0]
    return PrefillEntry(
        rating=rating,
        moi=entry.moi,
        phenotypes=entry.phenotypes,
        publications=tuple(ordered_panel_publications(list(entry.publications))),
    )


def load_gene(
    cursor: sqlite3.Cursor,
    row: sqlite3.Row,
    hgnc_resolver: HgncResolver,
    target_panel_data: PanelGeneData,
    all_panels_data: AllPanelsData,
    target_entries: tuple[PanelEntry, ...],
    scoped: bool,
) -> GeneAssessment:
    """One gene's aggregation row with its associations, papers and panel context.

    *target_entries* are the gene's entries on the target panels. *scoped* is true in a
    panel-scoped run, whose one target panel is the scope panel.
    """
    hgnc_id: int = row["hgnc_id"]
    paper_id_to_doi: dict[str, str] = json.loads(row["paper_id_mapping"])
    filtered_doi_reasons: dict[str, str] = {
        fp["doi"]: fp["reason"] for fp in json.loads(row["filtered_papers_json"] or "[]")
    }

    contributing_papers = load_contributing_papers(cursor, hgnc_id, filtered_doi_reasons)
    pmid_by_doi = {p.doi: p.pmid for p in contributing_papers}

    # Replace AuthorYear paper IDs with PMID display format
    doi_to_pmid = {p.doi: p.pmid for p in contributing_papers if p.pmid is not None}
    display_ids = build_display_ids(paper_id_to_doi, doi_to_pmid)
    doi_to_display_id = {paper_id_to_doi[pid]: did for pid, did in display_ids.items()}
    for paper in contributing_papers:
        if paper.doi in doi_to_display_id:
            paper.display_id = doi_to_display_id[paper.doi]

    # paper_id_mapping holds the papers the aggregation read (the contributing papers
    # less those the preprint gate filtered), so the variants are those of its prompt.
    papers_by_doi = {paper.doi: paper for paper in contributing_papers}
    gene_variants = load_report_variants(
        cursor, hgnc_id, [papers_by_doi[doi] for doi in paper_id_to_doi.values()]
    )

    # The gene is novel if it is on no target panel. A known gene's prefill reviews it
    # on the first target panel holding it; a novel gene's adds it to the first one.
    panel_confidence = target_panel_data.gene_panel_confidence.get(hgnc_id, {})
    holding = [pid for pid in target_panel_data.panel_ids if pid in panel_confidence]
    current_rating = target_panel_data.highest_confidence(hgnc_id) if holding else None
    prefill_panel_id = holding[0] if holding else target_panel_data.panel_ids[0]

    rated_associations = load_associations(
        cursor,
        hgnc_id,
        json.loads(row["panelapp_context_json"]),
        display_ids,
        all_panels_data,
        gene_variants,
        target_entries,
        pmid_by_doi,
        scoped,
    )
    findings = gene_findings(rated_associations, current_rating, scoped)
    associations = report_order(rated_associations, findings)
    associated_ids = {v.variant_id for a in associations for v in a.variants}
    unassessed_reports = replace_paper_ids_for_display(
        {"unassessed_reports": json.loads(row["unassessed_reports_json"])}, display_ids
    )["unassessed_reports"]

    described = prefill_associations(associations, findings)
    finding_by_association = {f.association_id: f for f in findings}
    prefill = prepare_prefill_data(
        hgnc_id=hgnc_id,
        associations=[
            PrefillAssociation(a.assessment, a.mondo, finding_by_association.get(a.id))
            for a in described.associations
        ],
        moi_associations=[a.assessment for a in associations],
        panel_id=prefill_panel_id,
        entry=(
            prefill_entry(target_entries, prefill_panel_id, panel_confidence[prefill_panel_id])
            if holding
            else None
        ),
        doi_to_pmid=pmid_by_doi,
        rated=described.rated,
    )

    return GeneAssessment(
        hgnc_id=hgnc_id,
        hgnc_symbol=hgnc_resolver.get_symbol(hgnc_id),
        associations=associations,
        unassessed_reports=unassessed_reports,
        current_rating=current_rating,
        findings=findings,
        contributing_papers=contributing_papers,
        unassociated_variants=[v for v in gene_variants if v.variant_id not in associated_ids],
        prefill=prefill,
        target_entries=target_entries,
        refused_papers=load_refused_papers(cursor, hgnc_id),
    )


def build_gene_assessment_results(
    conn: sqlite3.Connection,
    hgnc_resolver: HgncResolver,
    target_panel_data: PanelGeneData,
    all_panels_data: AllPanelsData,
    scope_panel_id: int | None,
    target_entries: dict[int, tuple[PanelEntry, ...]],
) -> GeneAssessmentResults:
    """Every aggregated gene with associations, split into novel and known genes.

    A gene whose reports all went to ``unassessed_reports`` has an aggregation but no
    associations. It has nothing to rate or submit, so the report leaves it out.
    *target_entries* are the target panels' entries, by gene. The entries nobody has
    reviewed of genes the report leaves out are listed on their own.

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
               unassessed_reports_json
        FROM gene_aggregations g
        WHERE EXISTS (SELECT 1 FROM associations a WHERE a.hgnc_id = g.hgnc_id)
        ORDER BY hgnc_id
        """
    )
    novel_genes: list[GeneAssessment] = []
    known_genes: list[GeneAssessment] = []
    for row in cursor.fetchall():
        gene = load_gene(
            conn.cursor(),
            row,
            hgnc_resolver,
            target_panel_data,
            all_panels_data,
            target_entries.get(row["hgnc_id"], ()),
            scoped=scope_panel_id is not None,
        )
        (novel_genes if gene.is_novel else known_genes).append(gene)

    logger.info(f"Loaded {len(novel_genes)} novel genes, {len(known_genes)} known genes")

    target_panel_names = {
        pid: all_panels_data.panel_names[pid]
        for pid in target_panel_data.panel_ids
        if pid in all_panels_data.panel_names
    }
    reported = {gene.hgnc_id for gene in novel_genes + known_genes}
    unreviewed = {
        hgnc_id: entries
        for hgnc_id, entries in unreviewed_target_entries(target_entries).items()
        if hgnc_id not in reported
    }
    return GeneAssessmentResults(
        novel_genes=novel_genes,
        known_genes=known_genes,
        refused_genes=load_refused_genes(cursor, hgnc_resolver),
        target_panel_data=target_panel_data,
        target_panel_names=target_panel_names,
        panels_matched=match_panels_has_run(cursor),
        scope_panel_id=scope_panel_id,
        unreviewed_without_evidence=unreviewed_without_evidence(
            unreviewed, load_read_papers(cursor, list(unreviewed)), hgnc_resolver
        ),
    )


def unreviewed_target_entries(
    entries: dict[int, tuple[PanelEntry, ...]],
) -> dict[int, list[PanelEntry]]:
    """The entries on the reference panels that nobody has reviewed, by gene."""
    by_gene = {
        hgnc_id: [e for e in gene_entries if e.reference and e.unreviewed]
        for hgnc_id, gene_entries in entries.items()
    }
    return {hgnc_id: entries for hgnc_id, entries in by_gene.items() if entries}


def load_read_papers(cursor: sqlite3.Cursor, hgnc_ids: list[int]) -> dict[int, list[ReadPaper]]:
    """The papers with an extraction that the run read for each of *hgnc_ids*, newest first.

    A paper was read for a gene when it has an extraction and the relevance screen
    named the gene for it, PanelApp cites it for the gene (``panelapp_citations``),
    or its extraction holds a disease entity of the gene. A gene without such papers
    is left out.
    """
    cursor.execute(
        """
        WITH read (hgnc_id, paper_doi) AS (
            SELECT hgnc_id, paper_doi FROM gene_mentions
            UNION
            SELECT hgnc_id, paper_doi FROM panelapp_citations
        )
        SELECT r.hgnc_id, p.doi, p.pmid, p.title,
               EXISTS (
                   SELECT 1 FROM gene_mentions gm
                   WHERE gm.paper_doi = p.doi AND gm.hgnc_id = r.hgnc_id
                     AND gm.source IN ('recent_evidence', 'expansion_evidence')
               ) AS disease_extracted
        FROM read r
        JOIN papers p ON p.doi = r.paper_doi
        WHERE p.evidence_extraction_json IS NOT NULL
          AND r.hgnc_id IN (SELECT value FROM json_each(?))
        ORDER BY r.hgnc_id, p.source_date DESC, p.doi
        """,
        (json.dumps(hgnc_ids),),
    )
    papers: dict[int, list[ReadPaper]] = {}
    for row in cursor.fetchall():
        papers.setdefault(row["hgnc_id"], []).append(
            ReadPaper(
                doi=row["doi"],
                pmid=row["pmid"],
                title=row["title"],
                disease_extracted=bool(row["disease_extracted"]),
            )
        )
    return papers


def unreviewed_without_evidence(
    unreviewed_entries: dict[int, list[PanelEntry]],
    read_papers: dict[int, list[ReadPaper]],
    hgnc_resolver: HgncResolver,
) -> list[UnreviewedEntry]:
    """The unreviewed entries with the papers read for their gene, by symbol and then panel
    name."""
    entries = [
        UnreviewedEntry(
            hgnc_id,
            hgnc_resolver.get_symbol(hgnc_id),
            e.panel_id,
            e.panel_name,
            e.rating,
            tuple(read_papers.get(hgnc_id, ())),
        )
        for hgnc_id, gene_entries in unreviewed_entries.items()
        for e in gene_entries
    ]
    return sorted(entries, key=lambda e: (e.hgnc_symbol, e.panel_name))


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
    scope_panel_id: int | None = None,
) -> GeneAssessmentResults:
    """Load every gene's aggregation and associations, with PanelApp membership at *panel_date*.

    Args:
        db_path: Path to the database
        panel_date: Date (YYYY-MM-DD) to check panel membership at
        target_panel_ids: List of panel IDs to use for novelty detection. If None, uses TARGET_PANEL_IDS.
        scope_panel_id: The panel of a panel-scoped run, which is then the only target panel;
            exclusive with *target_panel_ids*.
    """
    logger.info(f"Loading gene aggregations from {db_path}...")
    if scope_panel_id is not None:
        if target_panel_ids is not None:
            raise ValueError("scope_panel_id and target_panel_ids are mutually exclusive")
        target_panel_ids = [scope_panel_id]

    panelapp_client = PanelAppClient(panel_date)
    target_panel_data = panelapp_client.get_target_panels_genes(target_panel_ids)
    all_panels_data = panelapp_client.get_all_panels_genes()
    logger.info(
        f"Loaded {len(target_panel_data.gene_panel_confidence)} genes from target panels, {len(all_panels_data.gene_to_panels)} genes from all panels"
    )

    target_entries = entries_by_gene(
        [panelapp_client.get_panel_data(panel_id) for panel_id in target_panel_data.panel_ids],
        target_panel_data.panel_ids,
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return build_gene_assessment_results(
            conn,
            hgnc_resolver,
            target_panel_data,
            all_panels_data,
            scope_panel_id,
            target_entries,
        )


def load_panel_publications_validation(
    db_path: Path, panel_pubs: PanelPublications
) -> PanelValidationResult:
    """Load panel publication validation using source_type filtering.

    Compares LLM assessments against *panel_pubs*, the publications the target panels
    cite in the run's snapshot. Uses source_type = 'initial' to avoid date range issues
    with expansion. Of the screen misses, it counts those read in full all the same,
    because PanelApp cites them for one of the run's genes.
    """
    logger.info("Loading panel publications validation...")

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
                screen_misses_read=0,
                refused=[],
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

        _panel_cols = f"""p.doi, p.pmid, p.title, p.abstract, p.authors, p.journal,
            p.source_date, p.source_type, p.source_details,
            p.relevance_assessment_json,
            p.evidence_extraction_json IS NOT NULL AS extracted,
            {CITED_PAPER} AS cited_for_run_gene"""

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

        # Match DOIs in any letter case: PanelApp's spelling can differ from the stored one
        dois_list = sorted({doi.lower() for doi in panel_pubs.dois})
        for i in range(0, len(dois_list), chunk_size):
            doi_chunk = dois_list[i : i + chunk_size]
            placeholders = ",".join("?" * len(doi_chunk))
            cursor.execute(
                f"""
                SELECT {_panel_cols} FROM papers p
                WHERE lower(p.doi) IN ({placeholders})
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
        screen_misses_read = 0
        refused: list[DetailedPaper] = []

        for row in all_panel_papers:
            relevance_assessment = None
            try:
                relevance_assessment = json.loads(row["relevance_assessment_json"])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse relevance assessment for panel DOI {row['doi']}")
                continue

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
                gene_evaluations=[],
                quote_refs=load_quote_refs(cursor, row["doi"]),
                cited_for_run_gene=bool(row["cited_for_run_gene"]),
                preprint=is_preprint(row["journal"], row["pmid"]),
                pmid=row["pmid"],
            )

            if relevance_assessment.get("refused") is not None:
                refused.append(detailed_paper)
            elif relevance_assessment["relevant"]:
                true_positives.append(detailed_paper)
            else:
                screen_misses.append(detailed_paper)
                if detailed_paper.cited_for_run_gene and row["extracted"]:
                    screen_misses_read += 1

        total_assessed = len(true_positives) + len(screen_misses)
        sensitivity_pct = len(true_positives) / total_assessed * 100 if total_assessed else 0.0
        logger.info(
            f"Panel validation: {len(true_positives)} relevant, {len(screen_misses)} missed by "
            f"the screen ({screen_misses_read} of them read through PanelApp citations), "
            f"{len(refused)} refused for good"
        )
        logger.info(f"Sensitivity: {sensitivity_pct:.1f}%")

        return PanelValidationResult(
            total_panel_papers=total_panel_refs,
            panel_papers_in_db=total_assessed + len(refused),
            true_positives=true_positives,
            screen_misses=screen_misses,
            screen_misses_read=screen_misses_read,
            refused=refused,
            sensitivity_pct=sensitivity_pct,
        )


def load_low_confidence_irrelevant_papers(db_path: Path) -> list[DetailedPaper]:
    """Load papers marked as not relevant with low confidence for manual review."""
    logger.info("Loading low-confidence irrelevant papers...")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all papers with relevance assessments (filtered in Python below)
        cursor.execute(f"""
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
                {CITED_PAPER} AS cited_for_run_gene
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

            # Screen rejections the screen itself was unsure about. A paper refused for
            # good has no rejection to review.
            if relevance_assessment.get("refused") is not None:
                continue
            screen = relevance_assessment["screen"]
            if not screen["relevant"] and screen["confidence"] == "LOW":
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
                    gene_evaluations=[],
                    quote_refs=load_quote_refs(cursor, row["doi"]),
                    cited_for_run_gene=bool(row["cited_for_run_gene"]),
                    preprint=is_preprint(row["journal"], row["pmid"]),
                    pmid=row["pmid"],
                )
                low_confidence_papers.append(detailed_paper)

        logger.info(f"Found {len(low_confidence_papers)} low-confidence irrelevant papers")
        return low_confidence_papers


def load_relevance_refusers(db_path: Path) -> str:
    """Who refused the relevance assessment of the papers settled as refused, as the subject
    of "refused": "Opus 5.5 and its fallback Sonnet 5.5 both" after a run on a model with a
    fallback, the model alone (e.g. "Haiku 5.5") after a run on one without."""
    with sqlite3.connect(db_path) as conn:
        models = {
            model
            for (model,) in conn.execute(
                """
                SELECT DISTINCT model FROM llm_requests
                WHERE stage = ? AND status = 'refused'
                """,
                (RELEVANCE_STAGE,),
            )
        }
    phrases = []
    for final in sorted(models - FALLBACKS.keys()):
        refused_first = sorted(m for m in models if FALLBACKS.get(m) == final)
        phrases.append(
            f"{' and '.join(MODEL_NAMES[m] for m in refused_first)} and its fallback "
            f"{MODEL_NAMES[final]} both"
            if refused_first
            else MODEL_NAMES[final]
        )
    return " or ".join(phrases)


def load_manual_download_papers(db_path: Path) -> list[ManualDownloadPaper]:
    """Load papers from recent literature that require manual download.

    These papers were not downloaded (missed or paywalled), so they only have
    basic metadata and relevance assessment from abstract. They are ordered by
    their journal's OpenAlex citedness, highest first, then newest first. Preprints,
    which PubMed indexes too, have no figure, as a preprint server's says nothing of
    the paper; they come last with the papers whose journal has none. Raises
    ValueError when journal-metrics has not looked up the journal of one of them.
    """
    logger.info("Loading papers requiring manual download...")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get papers from recent literature with manual_required download status
        cursor.execute(f"""
            SELECT
                p.doi,
                p.pmid,
                p.title,
                p.abstract,
                p.authors,
                p.journal,
                p.journal_issn,
                m.issn IS NOT NULL AS journal_looked_up,
                m.two_year_mean_citedness,
                p.source_date,
                p.source_type,
                p.source_details,
                p.relevance_assessment_json,
                {CITED_PAPER} AS cited_for_run_gene
            FROM papers p
            LEFT JOIN journal_metrics m ON m.issn = p.journal_issn
            WHERE p.source_type = 'initial'
            AND p.download_status = 'manual_required'
            ORDER BY p.source_date DESC, p.doi DESC
        """)
        rows = cursor.fetchall()

        not_looked_up = sorted(
            {
                row["journal_issn"]
                for row in rows
                if row["journal_issn"] and not row["journal_looked_up"]
            }
        )
        if not_looked_up:
            raise ValueError(
                f"No journal metrics for ISSNs {not_looked_up} of papers awaiting manual "
                f"download; run `palit journal-metrics --db-path {db_path}` first"
            )

        manual_download_papers = []
        for row in rows:
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
                gene_evaluations=[],
                quote_refs={},
                cited_for_run_gene=bool(row["cited_for_run_gene"]),
                preprint=is_preprint(row["journal"], row["pmid"]),
                pmid=row["pmid"],
            )
            citedness = None if detailed_paper.preprint else row["two_year_mean_citedness"]
            manual_download_papers.append(ManualDownloadPaper(detailed_paper, citedness))

        logger.info(f"Found {len(manual_download_papers)} papers requiring manual download")
        # A stable sort, so papers with equal figures stay newest first.
        return sorted(
            manual_download_papers,
            key=lambda p: (p.journal_citedness is None, -(p.journal_citedness or 0.0)),
        )


def _gene_contribution_bucket(gene: GeneAssessment) -> str:
    """Classify a gene by its place in the report.

    Returns one of: 'novel', 'findings', 'unreviewed', 'already_reviewed'.
    """
    if gene.is_novel:
        return "novel"
    if gene.findings:
        return "findings"
    if gene.unreviewed_target_panels:
        return "unreviewed"
    return "already_reviewed"


_BUCKET_PRIORITY = ("novel", "findings", "unreviewed", "already_reviewed")

# A `papers` row `p` outside the run's corpus that the run did not read and whose relevance
# assessment was not refused: an initial paper the screen rejected and PanelApp does not
# cite for a run gene, or one without a relevance result
UNREAD_OUTSIDE_CORPUS = f"""(
    p.evidence_extraction_json IS NULL
    AND NOT {CORPUS_PAPER}
    AND json_extract(p.relevance_assessment_json, '$.refused') IS NULL
)"""


def load_favorite_journal_papers(
    db_path: Path,
    novel_genes: list[GeneAssessment],
    known_genes: list[GeneAssessment],
    scoped: bool,
) -> FavoriteJournalSections:
    """Load every initial paper from FAVORITE_JOURNALS, bucketed by the
    kind of contribution it made, mirroring the report's gene groups
    (new genes / known genes with findings / unreviewed / already reviewed)
    and the reasons a paper did not contribute (filtered out of aggregate /
    read in full without evidence for the report / requiring manual download /
    relevance refused for good / screened out by relevance). Expansion papers
    are excluded entirely. Papers are bucketed by corpus membership (see
    palit.run_corpus): a corpus paper the run has not read is due for manual
    download or still in progress, whatever the screen said, and only papers
    outside the corpus are refused or screened out. A panel-scoped run
    (*scoped*) screens the whole baseline, so it leaves out the unread papers
    outside the corpus: there is no "screened out" bucket."""
    gene_anchors: dict[int, FavoriteJournalGeneLink] = {}
    gene_buckets: dict[int, str] = {}
    for gene in (*novel_genes, *known_genes):
        gene_anchors[gene.hgnc_id] = FavoriteJournalGeneLink(
            hgnc_id=gene.hgnc_id, hgnc_symbol=gene.hgnc_symbol, anchor=gene.anchor
        )
        gene_buckets[gene.hgnc_id] = _gene_contribution_bucket(gene)

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
            SELECT p.doi, p.pmid, p.title, p.journal, p.source_date, p.download_status,
                   p.relevance_assessment_json, p.evidence_extraction_json,
                   {CORPUS_PAPER} AS in_corpus
            FROM papers p
            WHERE p.journal IN ({placeholders})
              AND p.source_type = 'initial'
              {f"AND NOT {UNREAD_OUTSIDE_CORPUS}" if scoped else ""}
            ORDER BY p.source_date DESC, p.doi DESC
            """,
            FAVORITE_JOURNALS,
        )
        rows = cursor.fetchall()

    sections = FavoriteJournalSections(
        novel=[],
        findings=[],
        unreviewed=[],
        already_reviewed=[],
        filtered=[],
        manual_download=[],
        refused=[],
        not_relevant=[],
        read_no_evidence=[],
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

        # Read in full, whatever the screen said: a paper PanelApp cites for one of the
        # run's genes is read even when the screen rejected it.
        if row["evidence_extraction_json"] is not None:
            sections.read_no_evidence.append(paper)
            continue

        # Due to be read: the screen passed it, or PanelApp cites it for a run gene.
        if row["in_corpus"]:
            if row["download_status"] == "manual_required":
                sections.manual_download.append(paper)
            # else: its download or extraction is still in progress: drop.
            continue

        assessment = json.loads(row["relevance_assessment_json"])
        if assessment.get("refused") is not None:
            sections.refused.append(paper)
        else:
            sections.not_relevant.append(paper)

    logger.info(
        "Featured journals: %d total; novel=%d, findings=%d, "
        "unreviewed=%d, already_reviewed=%d, filtered=%d, read_no_evidence=%d, "
        "manual_download=%d, refused=%d, not_relevant=%s",
        sections.total,
        len(sections.novel),
        len(sections.findings),
        len(sections.unreviewed),
        len(sections.already_reviewed),
        len(sections.filtered),
        len(sections.read_no_evidence),
        len(sections.manual_download),
        len(sections.refused),
        "left out (panel-scoped run)" if scoped else len(sections.not_relevant),
    )
    return sections


def load_paper_sources(cursor: sqlite3.Cursor) -> PaperSources:
    """Count the papers with an extraction by how they entered the corpus.

    A paper the scope screen passed counts as relevant, whether or not PanelApp
    cites it. Of the others, a paper PanelApp cites for one of the run's genes
    counts as cited, initial or expansion, and the remaining expansion papers as
    expansion.
    """
    cursor.execute(
        f"""
        SELECT COALESCE(SUM(relevant), 0),
               COALESCE(SUM(NOT relevant AND cited), 0),
               COALESCE(SUM(NOT relevant AND NOT cited AND expansion), 0),
               COALESCE(SUM(NOT relevant AND NOT cited AND NOT expansion), 0)
        FROM (
            SELECT {RELEVANT_PAPER} AS relevant, {CITED_PAPER} AS cited,
                   p.source_type IS 'expansion' AS expansion
            FROM papers p
            WHERE p.evidence_extraction_json IS NOT NULL
        )
        """
    )
    relevant, cited, expansion, outside_corpus = cursor.fetchone()
    return PaperSources(
        relevant=relevant, cited=cited, expansion=expansion, outside_corpus=outside_corpus
    )


def calculate_comprehensive_statistics(
    db_path: Path, results: GeneAssessmentResults, panel_validation: PanelValidationResult
) -> ComprehensiveStats:
    """Calculate enhanced statistics for report."""

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        all_genes = results.novel_genes + results.known_genes
        finding_kinds = [finding.kind for gene in results.known_genes for finding in gene.findings]
        total_panel_suggestions = sum(
            len(
                {
                    match.panel_id
                    for association in gene.associations
                    for match in association.matched_panels or []
                    if not match.gene_on_panel
                }
            )
            for gene in all_genes
        )

        # Papers contributing to assessments
        all_dois = set()
        for gene in all_genes:
            for p in gene.contributing_papers:
                all_dois.add(p.doi)

        all_associations = [a for gene in all_genes for a in gene.associations]
        scoped = results.scope_panel_id is not None
        already_reviewed_findings = sum(
            1
            for gene in all_genes
            for a in gene.associations
            if a.already_reviewed
            and association_finding(replace(a, reviewed_on=()), gene.current_rating, scoped)
            is not None
        )

        # Known genes with an entry on a target panel that nobody has reviewed
        unreviewed_count = sum(1 for gene in results.known_genes if gene.unreviewed_target_panels)

        # Preprint stats (computed from already-loaded data)
        preprint_dois = {
            p.doi for gene in all_genes for p in gene.contributing_papers if p.preprint
        }
        papers_filtered = len(
            {p.doi for gene in all_genes for p in gene.contributing_papers if p.filtered_reason}
        )

        return ComprehensiveStats(
            # Gene assessment stats
            total_genes_assessed=len(all_genes),
            new_genes_count=len(results.novel_genes),
            known_genes_with_findings=sum(1 for gene in results.known_genes if gene.findings),
            total_contributing_papers=len(all_dois),
            # Panel validation stats
            total_panel_papers=panel_validation.total_panel_papers,
            panel_papers_in_db=panel_validation.panel_papers_in_db,
            validation_sensitivity_pct=panel_validation.sensitivity_pct,
            screen_misses_count=len(panel_validation.screen_misses),
            screen_misses_read_count=panel_validation.screen_misses_read,
            true_positives_count=len(panel_validation.true_positives),
            panel_refused_count=len(panel_validation.refused),
            paper_sources=load_paper_sources(cursor),
            total_panel_suggestions=total_panel_suggestions,
            total_associations=len(all_associations),
            new_disease_findings=finding_kinds.count(FindingKind.NEW_DISEASE),
            new_moi_findings=finding_kinds.count(FindingKind.NEW_MOI),
            upgrade_findings=finding_kinds.count(FindingKind.UPGRADE),
            already_reviewed_findings=already_reviewed_findings,
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


class DerivedFamilies(StrEnum):
    """How many of an entity's qualifying families are evidently patients counted as one
    family each.

    Papers that report patients without grouping them into families (COHORTS WITHOUT
    FAMILY STRUCTURE in the extraction and aggregation prompts) add no reported
    families, but each of their qualifying patients counts as one qualifying family.
    This shows only where the qualifying count exceeds the reported one: families that
    fail the qualifying gate can hide derived ones.
    """

    NOT_EVIDENT = "not_evident"  # no more qualifying families than reported ones
    SOME = "some"  # fewer reported families than qualifying ones
    ALL = "all"  # no reported families, but qualifying ones


def derived_families(entity: dict[str, Any]) -> DerivedFamilies:
    """Whether some or all of the qualifying families of an association or a paper's
    disease entity are evidently patients counted as one family each.

    An entity without ``reported_family_count`` comes from an extraction that did not
    count reported families, so none are evident.
    """
    qualifying: int | None = entity["family_count"]
    if not qualifying or "reported_family_count" not in entity:
        return DerivedFamilies.NOT_EVIDENT
    reported: int | None = entity["reported_family_count"]
    if reported is None:
        return DerivedFamilies.ALL
    return DerivedFamilies.SOME if reported < qualifying else DerivedFamilies.NOT_EVIDENT


@dataclass(frozen=True)
class FrequencyFlag:
    """A variant's gnomAD count above the threshold of the MoI it is judged by."""

    metric: str  # "het", "hom" or "hemi"
    genotype: str  # "heterozygote", "homozygote" or "hemizygote"
    count: int
    threshold: int

    @property
    def warning(self) -> str:
        return (
            f"High {self.genotype} count ({self.metric} = {self.count} > {self.threshold}) "
            "- consider population frequency in assessment"
        )


def variant_frequency_flag(
    variant: ReportVariant, inheritance_mode: str | None
) -> FrequencyFlag | None:
    """The variant's gnomAD count when it is above the threshold of *inheritance_mode*.

    A biallelic association is judged by homozygotes, an X-linked one by
    hemizygotes, any other by heterozygotes. A variant without gnomAD figures, or
    in no association (*inheritance_mode* None), is never flagged.
    """
    frequency = variant.frequency
    if frequency is None or inheritance_mode is None:
        return None
    match inheritance_mode:
        case "Biallelic":
            flag = FrequencyFlag(
                "hom", "homozygote", frequency.homozygote_count, GNOMAD_HOM_THRESHOLD
            )
        case "X-linked":
            flag = FrequencyFlag(
                "hemi", "hemizygote", frequency.hemizygote_count, GNOMAD_HEMI_THRESHOLD
            )
        case _:
            flag = FrequencyFlag(
                "het", "heterozygote", frequency.heterozygote_count, GNOMAD_HET_THRESHOLD
            )
    return flag if flag.count > flag.threshold else None


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

    A gene's suggested rating is its prefill's rating, against which curator
    ratings count as concordant.

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
            "suggested_rating": gene.prefill.rating,
        }

    config = {
        "report_id": report_id,
        "target_panel_ids": target_panel_ids,
        "genes": genes,
    }
    return json.dumps(config)


def report_title(scope_panel_id: int | None, target_panel_names: dict[int, str]) -> str:
    """The report's title and heading: the scope panel's name and ID in a panel-scoped run."""
    if scope_panel_id is None:
        return MONTHLY_REPORT_TITLE
    return f"{target_panel_names[scope_panel_id]} ({scope_panel_id}): Literature Assessment Report"


def generate_html_report(
    novel_genes: list[GeneAssessment],
    known_genes: list[GeneAssessment],
    statistics: ComprehensiveStats,
    panel_validation: PanelValidationResult,
    low_confidence_papers: list[DetailedPaper],
    manual_download_papers: list[ManualDownloadPaper],
    favorite_journal_papers: FavoriteJournalSections,
    template_dir: Path,
    panel_date: str,
    report_id: str,
    target_panel_ids: list[int],
    target_panel_names: dict[int, str],
    refused_genes: list[RefusedGene],
    relevance_refusers: str,
    *,
    panels_matched: bool,
    panelapp_integration: bool,
    unreviewed_without_evidence: list[UnreviewedEntry],
    scope_panel_id: int | None,
) -> str:
    """Generate HTML report directly using Jinja2 templates.

    *scope_panel_id* is the panel of a panel-scoped run, which is then the only target panel.
    """

    # Build report config JSON for PanelApp integration
    report_config_json = (
        build_report_config(report_id, target_panel_ids, novel_genes, known_genes)
        if panelapp_integration
        else None
    )

    groups = group_genes(novel_genes, known_genes)

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
    env.filters["frequency_flag"] = variant_frequency_flag
    env.filters["derived_families"] = derived_families
    env.filters["confidence_to_color"] = panelapp_confidence_to_color
    # Double-encode: first quote produces the on-disk filename (e.g. 10.1038%2Fxyz),
    # second quote escapes the % for use in file:/// URLs so browsers don't
    # decode %2F back to / (e.g. 10.1038%252Fxyz → opens 10.1038%2Fxyz.pdf).
    env.filters["paper_key"] = doi_to_key
    env.filters["short_id"] = lambda display_id: display_id.removeprefix("PMID ")
    env.filters["papers_for_dois"] = papers_for_dois

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
        title=report_title(scope_panel_id, target_panel_names),
        groups=groups,
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
        unreviewed_without_evidence=unreviewed_without_evidence,
        scoped=scope_panel_id is not None,
        min_families_for_moi_expansion=MIN_FAMILIES_FOR_MOI_EXPANSION,
        model_name=MODEL_NAMES[MODEL],
        relevance_refusers=relevance_refusers,
        fallback_model_name=MODEL_NAMES[FALLBACK_MODEL],
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
    scope_panel_id: int | None = typer.Option(
        None,
        "--scope-panel-id",
        help="Panel ID of a panel-scoped run: the only target panel, and the panel each "
        "upgrade starts from. Exclusive with --target-panel-ids.",
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
    if scope_panel_id is not None and target_panel_ids:
        logger.error("--scope-panel-id and --target-panel-ids are mutually exclusive")
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
    results = load_gene_assessments(
        db_path, panel_date, hgnc_resolver, target_panel_ids or None, scope_panel_id
    )
    # Use actual target_panel_ids from results (handles None -> defaults)
    actual_panel_ids = list(results.target_panel_data.panel_ids)
    panel_validation = load_panel_publications_validation(
        db_path, PanelAppClient(panel_date).panel_publications(actual_panel_ids)
    )
    low_confidence_papers = load_low_confidence_irrelevant_papers(db_path)
    manual_download_papers = load_manual_download_papers(db_path)
    favorite_journal_papers = load_favorite_journal_papers(
        db_path, results.novel_genes, results.known_genes, scoped=scope_panel_id is not None
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
        relevance_refusers=load_relevance_refusers(db_path),
        panels_matched=results.panels_matched,
        panelapp_integration=panelapp_integration,
        unreviewed_without_evidence=results.unreviewed_without_evidence,
        scope_panel_id=scope_panel_id,
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
