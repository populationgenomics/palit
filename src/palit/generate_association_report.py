#!/usr/bin/env python3
"""Generate an association-level HTML report from per-association assessments.

The unit of this report is one fixed gene-disease-inheritance association: each
carries its own rating, evidence and contributing papers, and associations are
grouped under a slim header for the gene they belong to. The curation source's
own classification for the association is shown next to ours for comparison and
was never part of any prompt, so the two verdicts are independent.
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import markdown
import nh3
import typer
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from palit.docling import parse_bbox_mapping_from_json
from palit.entities import MOI_DISPLAY, DiseaseEntity, load_entities
from palit.generate_report import (
    DetailedPaper,
    VariantFrequency,
    format_inheritance,
    get_variant_frequency_flag,
    load_variant_frequencies_for_gene,
    load_variant_frequencies_for_paper,
    prepare_aggregate_citation_links,
)
from palit.hgnc import HgncResolver
from palit.panelapp_integration import (
    GENCC_CLASSIFICATION_TO_CONFIDENCE,
    calculate_association_rating,
    decompose_moi,
    gencc_classification_to_confidence,
    panelapp_confidence_to_color,
)
from palit.papers import (
    build_display_ids,
    doi_to_path,
    is_preprint,
    replace_paper_ids_for_display,
)

app = typer.Typer(help="Generate an association-level HTML report from association assessments")
logger = logging.getLogger(__name__)

# Human-readable names for the evidence-weakening factors of the assessment schema.
# A factor outside this map is a schema change, so the KeyError is the signal.
WEAKENING_FACTOR_LABELS: dict[str, str] = {
    "founder_or_recurrent_variant": "Founder or recurrent variant",
    "inherited_no_segregation_with_affected": "Inherited variant without segregation in affected relatives",
    "missense_no_variant_specific_functional": "Missense variants without variant-specific functional data",
    "population_frequency_incompatible_with_moi": "Population frequency incompatible with the inheritance mode",
    "phenotype_non_mendelian": "Phenotype not clearly Mendelian",
    "functional_below_criterion_c_threshold": "Functional evidence below the Criterion C threshold",
}

# Inheritance modes that carry no claim about the mode, so comparing a paper's
# block against the association's fixed mode says nothing.
UNINFORMATIVE_INHERITANCE_MODES = frozenset({"NR", "Other"})

# Our own rating levels, strongest first — the rows of the concordance table.
RATING_LEVELS: tuple[int, ...] = (3, 2, 1)

# Column headings for the source confidence levels of the concordance table.
GENCC_COLUMN_LABELS: dict[int, str] = {
    3: "Strong-equiv",
    2: "Moderate",
    1: "Limited/Disputed",
    0: "No-list",
}


@dataclass(frozen=True)
class FilteredPaper:
    """A paper excluded from an association's prompt, with the reason it was dropped."""

    doi: str
    reason: str


@dataclass(frozen=True)
class UnattributedBlock:
    """An extracted evidence block the model could not assign to any fixed association."""

    doi: str
    display_label: str  # "PMID {pmid}" for published papers, the DOI otherwise
    block: dict[str, Any]


@dataclass
class AssociationAssessment:
    """One fixed gene-disease-inheritance association and everything shown for it."""

    entity: DiseaseEntity
    hgnc_symbol: str
    assessment_json: dict[str, Any]
    rating: int  # 1 (RED), 2 (AMBER), 3 (GREEN)
    gencc_confidence: int  # Source classification on the same 0-3 scale, display only
    contributing_papers: list[DetailedPaper]
    variant_frequencies: list[VariantFrequency]
    entity_blocks: dict[str, list[dict[str, Any]]]  # DOI -> blocks assigned to this association
    moi_mismatch_dois: list[str]  # Papers reporting a mode outside the association's fixed one
    assessed_moi_mismatch: bool  # The assessment's own mode is outside the fixed one
    filtered_papers: list[FilteredPaper]

    @property
    def independent_family_count(self) -> int:
        """Independent families behind the rating, with a missing count read as zero."""
        return self.assessment_json["independent_family_count"] or 0


@dataclass
class GeneSection:
    """One gene's associations, plus what the corpus could not be attributed to."""

    hgnc_id: int
    hgnc_symbol: str
    associations: list[AssociationAssessment]
    unassessed: list[DiseaseEntity]  # Associations with no evidence in this corpus
    unattributed: list[UnattributedBlock]
    paper_count: int  # Distinct contributing papers across the gene's associations

    @property
    def entity_count(self) -> int:
        """Fixed associations seeded for this gene, whether or not evidence reached them."""
        return len(self.associations) + len(self.unassessed)


@dataclass(frozen=True)
class GenccColumn:
    """One column of the concordance table: a source confidence level and its terms."""

    confidence: int
    label: str
    classifications: str  # The source classifications projected onto this level


@dataclass(frozen=True)
class TocEntry:
    """One table-of-contents line: an association and the article it links to."""

    entity: DiseaseEntity
    hgnc_symbol: str
    anchor: str | None  # None for associations with no article to link to


@dataclass(frozen=True)
class TocBucket:
    """One rating's table-of-contents panel: its badge, its color and its entries."""

    color: str  # "green" / "amber" / "red" / "grey" — badge and panel modifier
    badge: str  # Badge text; the badge stylesheet upper-cases it
    entries: list[TocEntry]


@dataclass(frozen=True)
class AssociationStats:
    """Corpus-level counts, headlined by the rating-versus-source concordance table."""

    total_associations: int
    assessed: int
    unassessed: int
    green: int
    amber: int
    red: int
    contributing_papers: int
    unattributed_blocks: int
    # The entity `source` values verbatim, so the report names the exact export
    # the fixed associations were seeded from.
    source_label: str
    rating_vs_gencc: dict[tuple[int, int], int]  # (our rating, source confidence) -> count


def gencc_columns() -> list[GenccColumn]:
    """Build the concordance table's columns from the source classification map."""
    columns: list[GenccColumn] = []
    for confidence, label in GENCC_COLUMN_LABELS.items():
        titles = sorted(
            title
            for title, level in GENCC_CLASSIFICATION_TO_CONFIDENCE.items()
            if level == confidence
        )
        columns.append(
            GenccColumn(confidence=confidence, label=label, classifications=", ".join(titles))
        )
    return columns


def build_toc(sections: list[GeneSection]) -> list[TocBucket]:
    """Bucket every association by the rating this corpus earned it.

    The gene sections arrive sorted by symbol and, within a gene, strongest
    association first, so each bucket inherits that order. Associations with no
    evidence in the corpus have no article to link to and land in a final bucket.
    """
    by_rating: dict[int, list[TocEntry]] = {rating: [] for rating in RATING_LEVELS}
    unassessed: list[TocEntry] = []

    for section in sections:
        for association in section.associations:
            by_rating[association.rating].append(
                TocEntry(
                    entity=association.entity,
                    hgnc_symbol=section.hgnc_symbol,
                    anchor=f"assoc-{association.entity.id}",
                )
            )
        unassessed.extend(
            TocEntry(entity=entity, hgnc_symbol=section.hgnc_symbol, anchor=None)
            for entity in section.unassessed
        )

    buckets = [
        TocBucket(
            color=panelapp_confidence_to_color(rating).lower(),
            badge=panelapp_confidence_to_color(rating),
            entries=by_rating[rating],
        )
        for rating in RATING_LEVELS
    ]
    buckets.append(TocBucket(color="grey", badge="No evidence", entries=unassessed))
    return [bucket for bucket in buckets if bucket.entries]


def entity_blocks_for(
    extraction: dict[str, Any], hgnc_id: int, entity_id: int
) -> list[dict[str, Any]]:
    """Select the extracted disease entity blocks assigned to one association."""
    blocks: list[dict[str, Any]] = []
    for gene_eval in extraction["gene_evaluations"]:
        if gene_eval.get("hgnc_id") != hgnc_id:
            continue
        blocks.extend(
            block for block in gene_eval["disease_entities"] if block.get("entity_id") == entity_id
        )
    return blocks


def display_label(doi: str, pmid: int | None) -> str:
    """Label a paper by PMID where it has one, otherwise by DOI."""
    return f"PMID {pmid}" if pmid is not None else doi


@dataclass(frozen=True)
class _PaperEvidence:
    """A contributing paper paired with the blocks it contributed to one association."""

    paper: DetailedPaper
    blocks: list[dict[str, Any]]


def _load_contributing_papers(
    cursor: sqlite3.Cursor, entity: DiseaseEntity
) -> list[_PaperEvidence]:
    """Load the papers whose extraction contributed evidence to one association."""
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
            p.evidence_extraction_json,
            p.bbox_mapping,
            (
                SELECT gm.paper_gene_symbol
                FROM gene_mentions gm
                WHERE gm.paper_doi = p.doi AND gm.hgnc_id = ?
                ORDER BY gm.id
                LIMIT 1
            ) AS paper_gene_symbol
        FROM papers p
        JOIN entity_mentions em ON p.doi = em.paper_doi
        WHERE em.entity_id = ?
        AND p.evidence_extraction_json IS NOT NULL
        ORDER BY p.source_date DESC, p.doi DESC
        """,
        (entity.hgnc_id, entity.id),
    )

    papers: list[_PaperEvidence] = []
    for row in cursor.fetchall():
        citation_pages: dict[int, int] | None = None
        if row["bbox_mapping"]:
            bbox_data = parse_bbox_mapping_from_json(row["bbox_mapping"])
            citation_pages = {box_id: info["page"] for box_id, info in bbox_data.items()}

        extraction: dict[str, Any] = json.loads(row["evidence_extraction_json"])
        papers.append(
            _PaperEvidence(
                paper=DetailedPaper(
                    doi=row["doi"],
                    title=row["title"] or "Unknown Title",
                    abstract=row["abstract"],
                    authors=row["authors"],
                    journal=row["journal"],
                    source_date=row["source_date"],
                    source_type=row["source_type"],
                    source_details=row["source_details"],
                    relevance_assessment=None,
                    evidence_extraction=extraction,
                    citation_pages=citation_pages,
                    preprint=is_preprint(row["journal"], row["pmid"]),
                    pmid=row["pmid"],
                    paper_gene_symbol=row["paper_gene_symbol"],
                ),
                blocks=entity_blocks_for(extraction, entity.hgnc_id, entity.id),
            )
        )
    return papers


def _load_unattributed_blocks(cursor: sqlite3.Cursor, hgnc_id: int) -> list[UnattributedBlock]:
    """Collect the gene's extracted blocks that name no fixed association."""
    cursor.execute(
        """
        SELECT DISTINCT p.doi, p.pmid, p.evidence_extraction_json
        FROM papers p
        JOIN gene_mentions gm ON p.doi = gm.paper_doi
        WHERE gm.hgnc_id = ?
        AND gm.source = 'recent_evidence'
        AND p.evidence_extraction_json IS NOT NULL
        ORDER BY p.doi
        """,
        (hgnc_id,),
    )

    unattributed: list[UnattributedBlock] = []
    for row in cursor.fetchall():
        extraction = json.loads(row["evidence_extraction_json"])
        for gene_eval in extraction["gene_evaluations"]:
            if gene_eval.get("hgnc_id") != hgnc_id:
                continue
            for block in gene_eval["disease_entities"]:
                if block.get("entity_id") is None:
                    unattributed.append(
                        UnattributedBlock(
                            doi=row["doi"],
                            display_label=display_label(row["doi"], row["pmid"]),
                            block=block,
                        )
                    )
    return unattributed


def _build_association(
    cursor: sqlite3.Cursor,
    entity: DiseaseEntity,
    hgnc_symbol: str,
    row: sqlite3.Row,
    *,
    with_variant_frequencies: bool,
) -> AssociationAssessment:
    """Assemble one association from its stored assessment and its contributing papers."""
    assessment_json: dict[str, Any] = json.loads(row["assessment_json"])
    paper_id_to_doi: dict[str, str] = json.loads(row["paper_id_mapping"])
    filtered_papers = [
        FilteredPaper(doi=entry["doi"], reason=entry["reason"])
        for entry in json.loads(row["filtered_papers_json"] or "[]")
    ]

    paper_evidence = _load_contributing_papers(cursor, entity)
    contributing_papers = [item.paper for item in paper_evidence]

    # The stored assessment cites papers by the {LastName}{Year} IDs the model saw;
    # readers want the PMID, so rewrite both the citations and the prose.
    doi_to_pmid = {p.doi: p.pmid for p in contributing_papers if p.pmid is not None}
    display_ids = build_display_ids(paper_id_to_doi, doi_to_pmid)
    assessment_json = replace_paper_ids_for_display(assessment_json, display_ids)
    doi_to_display_id = {paper_id_to_doi[pid]: did for pid, did in display_ids.items()}

    entity_blocks: dict[str, list[dict[str, Any]]] = {}
    moi_mismatch_dois: list[str] = []
    fixed_modes = decompose_moi(entity.moi)
    for item in paper_evidence:
        paper = item.paper
        # Papers the prompt dropped are absent from the mapping, so fall back to
        # the PMID rather than showing a bare DOI in citation labels.
        display = doi_to_display_id.get(paper.doi)
        paper.display_id = display if display is not None else display_label(paper.doi, paper.pmid)
        if with_variant_frequencies:
            paper.variant_frequencies = load_variant_frequencies_for_paper(
                cursor, paper.doi, paper.display_id, paper.citation_pages
            )

        entity_blocks[paper.doi] = item.blocks
        if any(
            block["inheritance_mode"] not in fixed_modes
            and block["inheritance_mode"] not in UNINFORMATIVE_INHERITANCE_MODES
            for block in item.blocks
        ):
            moi_mismatch_dois.append(paper.doi)

    variant_frequencies: list[VariantFrequency] = []
    if with_variant_frequencies and contributing_papers:
        variant_frequencies = load_variant_frequencies_for_gene(
            cursor, entity.hgnc_id, contributing_papers
        )

    return AssociationAssessment(
        entity=entity,
        hgnc_symbol=hgnc_symbol,
        assessment_json=assessment_json,
        rating=calculate_association_rating(assessment_json),
        gencc_confidence=gencc_classification_to_confidence(entity.gencc_classification),
        contributing_papers=contributing_papers,
        variant_frequencies=variant_frequencies,
        entity_blocks=entity_blocks,
        moi_mismatch_dois=moi_mismatch_dois,
        assessed_moi_mismatch=assessment_json["inheritance_mode"] not in fixed_modes,
        filtered_papers=filtered_papers,
    )


def _build_statistics(
    sections: list[GeneSection], entities: list[DiseaseEntity]
) -> AssociationStats:
    """Total the loaded sections against the full list of fixed associations."""
    associations = [a for section in sections for a in section.associations]

    rating_vs_gencc: dict[tuple[int, int], int] = {}
    for association in associations:
        key = (association.rating, association.gencc_confidence)
        rating_vs_gencc[key] = rating_vs_gencc.get(key, 0) + 1

    return AssociationStats(
        total_associations=len(entities),
        assessed=len(associations),
        unassessed=sum(len(section.unassessed) for section in sections),
        green=sum(1 for a in associations if a.rating == 3),
        amber=sum(1 for a in associations if a.rating == 2),
        red=sum(1 for a in associations if a.rating == 1),
        contributing_papers=len({p.doi for a in associations for p in a.contributing_papers}),
        unattributed_blocks=sum(len(section.unattributed) for section in sections),
        source_label=", ".join(sorted({entity.source for entity in entities})),
        rating_vs_gencc=rating_vs_gencc,
    )


def load_association_report_data(
    db_path: Path, hgnc_resolver: HgncResolver
) -> tuple[list[GeneSection], AssociationStats]:
    """Load every fixed association, its assessment where one exists, and the totals."""
    entities = load_entities(db_path)
    logger.info(f"Loaded {len(entities)} fixed associations from {db_path}")

    sections: list[GeneSection] = []

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        with_variant_frequencies = bool(
            cursor.execute("SELECT EXISTS(SELECT 1 FROM variant_frequencies)").fetchone()[0]
        )
        if not with_variant_frequencies:
            logger.info("No variant frequencies in this database — omitting gnomAD tables")

        assessment_rows = {
            row["entity_id"]: row
            for row in cursor.execute(
                """
                SELECT entity_id, assessment_json, paper_id_mapping, filtered_papers_json
                FROM gene_disease_assessments
                """
            ).fetchall()
        }

        by_gene: dict[int, list[DiseaseEntity]] = {}
        for entity in entities:
            by_gene.setdefault(entity.hgnc_id, []).append(entity)

        for hgnc_id, gene_entities in by_gene.items():
            hgnc_symbol = hgnc_resolver.get_symbol(hgnc_id)
            associations: list[AssociationAssessment] = []
            unassessed: list[DiseaseEntity] = []

            for entity in gene_entities:
                row = assessment_rows.get(entity.id)
                if row is None:
                    unassessed.append(entity)
                    continue
                associations.append(
                    _build_association(
                        cursor,
                        entity,
                        hgnc_symbol,
                        row,
                        with_variant_frequencies=with_variant_frequencies,
                    )
                )

            associations.sort(
                key=lambda a: (
                    -a.rating,
                    -a.independent_family_count,
                    a.entity.disease_title,
                    a.entity.moi,
                )
            )
            unassessed.sort(key=lambda e: (e.disease_title, e.moi))

            sections.append(
                GeneSection(
                    hgnc_id=hgnc_id,
                    hgnc_symbol=hgnc_symbol,
                    associations=associations,
                    unassessed=unassessed,
                    unattributed=_load_unattributed_blocks(cursor, hgnc_id),
                    paper_count=len({p.doi for a in associations for p in a.contributing_papers}),
                )
            )

    sections.sort(key=lambda s: s.hgnc_symbol)
    stats = _build_statistics(sections, entities)

    logger.info(
        f"Assessed {stats.assessed} of {stats.total_associations} associations "
        f"over {len(sections)} genes ({stats.green} green, {stats.amber} amber, {stats.red} red)"
    )
    return sections, stats


def generate_association_report(
    sections: list[GeneSection],
    statistics: AssociationStats,
    template_dir: Path,
    *,
    pdf_links: bool,
) -> str:
    """Render the association report to a single self-contained HTML document."""
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    env.filters["format_inheritance"] = format_inheritance
    env.filters["prepare_citation_links"] = prepare_aggregate_citation_links
    env.filters["get_variant_flag"] = get_variant_frequency_flag
    env.filters["confidence_to_color"] = panelapp_confidence_to_color
    env.filters["gencc_confidence"] = gencc_classification_to_confidence
    env.filters["moi_display"] = lambda moi: MOI_DISPLAY[moi]
    env.filters["factor_label"] = lambda factor: WEAKENING_FACTOR_LABELS[factor]
    # Double-encode: the first quote produces the on-disk filename (e.g. 10.1038%2Fxyz),
    # the second escapes the % for use in file:/// URLs so browsers don't decode
    # %2F back to / (e.g. 10.1038%252Fxyz → opens 10.1038%2Fxyz.pdf).
    env.filters["encode_doi"] = lambda doi: quote(quote(doi, safe=""), safe="")
    # Render markdown then sanitize to a strict allowlist of tags, so an
    # LLM-written summary cannot inject links, images or scripts.
    summary_allowed_tags = {"p", "strong", "em"}
    env.filters["markdown_to_html"] = lambda text: Markup(
        nh3.clean(markdown.markdown(text), tags=summary_allowed_tags)
    )

    template = env.get_template("association_report.html")
    return template.render(
        sections=sections,
        statistics=statistics,
        toc=build_toc(sections),
        rating_levels=RATING_LEVELS,
        gencc_columns=gencc_columns(),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        pdf_links=pdf_links,
    )


def count_missing_pdfs(sections: list[GeneSection], papers_dir: Path) -> list[str]:
    """List the PDF paths the report links to that are not on disk."""
    missing: list[str] = []
    seen: set[str] = set()
    for section in sections:
        for association in section.associations:
            for paper in association.contributing_papers:
                if paper.doi in seen:
                    continue
                seen.add(paper.doi)
                path = doi_to_path(paper.doi, papers_dir, ".pdf")
                if not path.exists():
                    missing.append(str(path))
    return missing


@app.callback(invoke_without_command=True)
def main(
    report_id: str = typer.Option(
        ...,
        "--report-id",
        help="Unique report identifier; used as the output directory name",
    ),
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database with association assessments",
    ),
    output_dir_prefix: Path = typer.Option(
        default=Path("reports"),
        help="Output directory prefix. Final path will be {prefix}/{report_id}/",
    ),
    papers_dir: Path = typer.Option(
        default=Path("data/papers"),
        help="Directory containing the paper PDFs the citations link into",
    ),
    template_dir: Path = typer.Option(
        default=Path("templates"), help="Directory containing report templates"
    ),
    no_pdf_links: bool = typer.Option(
        False,
        "--no-pdf-links",
        help="Render citations as plain text and skip the PDF symlink",
    ),
) -> None:
    """Generate an association-level report from the stored association assessments."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    output_dir = output_dir_prefix / report_id
    output_dir.mkdir(parents=True, exist_ok=True)

    hgnc_resolver = HgncResolver.from_file()
    sections, statistics = load_association_report_data(db_path, hgnc_resolver)

    html = generate_association_report(
        sections,
        statistics,
        template_dir,
        pdf_links=not no_pdf_links,
    )
    index_file = output_dir / "index.html"
    index_file.write_text(html)

    if no_pdf_links:
        logger.info("Citations rendered as plain text; no PDF directory linked")
    else:
        papers_link = output_dir / "papers"
        if papers_link.is_symlink():
            papers_link.unlink()
        os.symlink(papers_dir.resolve(), papers_link)

        missing = count_missing_pdfs(sections, papers_dir)
        if missing:
            logger.warning(f"Missing {len(missing)} paper PDFs:")
            for path in missing[:5]:
                logger.warning(f"  {path}")
            if len(missing) > 5:
                logger.warning(f"  ... and {len(missing) - 5} more")

    logger.info(f"Report written to {index_file}")


if __name__ == "__main__":
    app()
