#!/usr/bin/env python3
"""Aggregate each gene's evidence across papers into gene-disease-MoI associations.

One request per gene. The model groups the contributing papers' disease entities
into associations, anchored on what PanelApp Australia already curates for the
gene (its GenCC rows, its panel entries and its reviews), and assesses each
association on its own. An association that reuses a GenCC row takes that row's
MONDO term; the others get theirs from ``map-mondo``. A gene MODEL refused goes
to FALLBACK_MODEL in the next attempt (see :mod:`palit.llm`). A gene whose
answers failed for good is left out until ``--retry-failed``.
"""

import asyncio
import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import typer
from anthropic.types import Message
from anthropic.types.output_config_param import OutputConfigParam
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from palit.gencc import GeneGencc, fetch_gencc, fetch_mondo
from palit.hgnc import HgncResolver
from palit.llm import (
    ALL_TIME,
    FALLBACK_MODEL,
    MODEL,
    BatchTransport,
    Effort,
    ImmediateTransport,
    LlmRequest,
    LlmResult,
    ResultStatus,
    StageHistory,
    Transport,
    invalid_answer_reason,
    json_output_config,
    log_failed_for_good,
    log_refusal,
    make_client,
    parse_json_output,
    record_result,
    stage_history,
)
from palit.llm_usage import print_stage_summary
from palit.panelapp_check import INCIDENTALOME_SCOPE, CuratedRecord, PanelEntry
from palit.panelapp_client import (
    PanelAppClient,
    PanelGeneData,
    PanelPublications,
    collect_panelapp_gene_publications,
    format_panel_for_prompt,
)
from palit.panelapp_integration import (
    INHERITANCE_DETAILS_VOCABULARY,
    NEW_RELATION_STATUSES,
    calculate_association_rating,
    criteria_object_to_list,
    inheritance_details_items,
    panelapp_confidence_to_color,
    validate_entities_criteria_complete,
    validate_independent_family_counts,
)
from palit.papers import MIN_PREPRINT_FAMILIES, generate_paper_ids, is_preprint
from palit.quotes import is_placeholder
from palit.run_corpus import RUN_GENES
from palit.variants import (
    GeneVariant,
    GnomadFrequency,
    GnomadMiss,
    VariantReport,
    load_gene_variants,
)

app = typer.Typer(help="Aggregate evidence across papers into gene-disease-MoI associations")
logger = logging.getLogger(__name__)

STAGE = "assess_genes"
EFFORT: Effort = "medium"
# FALLBACK_MODEL rarely thinks at medium effort, and an aggregation needs a plan
# before its first association.
FALLBACK_EFFORT: Effort = "high"
EFFORT_BY_MODEL: dict[str, Effort] = {MODEL: EFFORT, FALLBACK_MODEL: FALLBACK_EFFORT}
MAX_TOKENS = 64000

DB_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class PanelReviews:
    """The reviews of a gene on one target panel that holds it, most recent first."""

    panel_id: int
    panel_name: str
    evaluations: list[dict[str, Any]]  # raw PanelApp evaluation dicts


def _resolve_paper_id(paper_id: str, paper_id_to_doi: dict[str, str]) -> str:
    """Look up DOI for a paper_id. Raises ValueError if unknown."""
    doi = paper_id_to_doi.get(paper_id)
    if doi is None:
        raise ValueError(f"Unknown paper_id: {paper_id}")
    return doi


def _replace_citation_ids(citations: list[dict[str, Any]], paper_id_to_doi: dict[str, str]) -> None:
    for citation in citations:
        citation["doi"] = _resolve_paper_id(citation.pop("paper_id"), paper_id_to_doi)


def _replace_id_list(entry: dict[str, Any], paper_id_to_doi: dict[str, str]) -> None:
    entry["dois"] = [_resolve_paper_id(pid, paper_id_to_doi) for pid in entry.pop("paper_ids")]


def replace_paper_ids_with_dois(
    parsed_json: dict[str, Any], paper_id_to_doi: dict[str, str]
) -> None:
    """Replace every paper ID in an aggregation with its DOI.

    Citations get ``doi`` for ``paper_id``; the ``paper_ids`` lists of associations,
    unassessed reports and quality concerns become ``dois``. Expects the criteria as
    a list (after ``criteria_object_to_list``). Mutates parsed_json in place and
    raises ValueError on the first unknown paper ID, a hallucination that is retried.
    """
    for entity in parsed_json["disease_entities"]:
        _replace_id_list(entity, paper_id_to_doi)
        for criterion in entity["evidence_assessments"]:
            _replace_citation_ids(criterion["citations"], paper_id_to_doi)
    for report in parsed_json["unassessed_reports"]:
        _replace_id_list(report, paper_id_to_doi)
    for concern in parsed_json["quality_concerns"]:
        _replace_id_list(concern, paper_id_to_doi)
        _replace_citation_ids(concern["citations"], paper_id_to_doi)


def replace_variant_ids(parsed_json: dict[str, Any], variant_id_to_key: dict[str, str]) -> None:
    """Replace each association's ``variant_ids`` with ``variants``, their variant keys.

    A variant ID is a row ID of the prompt's variant table (V1, V2, ...); its key
    is the variant's ``variant_frequencies.variant_id``. Mutates parsed_json in
    place and raises ValueError on the first unknown variant ID, a hallucination
    that is retried.
    """
    for entity in parsed_json["disease_entities"]:
        keys = []
        for variant_id in entity.pop("variant_ids"):
            key = variant_id_to_key.get(variant_id)
            if key is None:
                raise ValueError(f"Unknown variant ID: {variant_id}")
            keys.append(key)
        entity["variants"] = keys


def _max_family_count(evidence: dict[str, Any]) -> int | None:
    """Compute max family_count across all disease entities in an evidence entry.

    Returns None if all family_count values are None (not reported).
    """
    max_fc: int | None = None
    for gene_eval in evidence.get("gene_evaluations", []):
        for entity in gene_eval.get("disease_entities", []):
            fc = entity.get("family_count")
            if fc is not None:
                max_fc = max(max_fc, fc) if max_fc is not None else fc
    return max_fc


def filter_preprint_evidence(
    evidence_list: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split evidence into (kept, filtered) based on preprint family count gate.

    Preprints with max family_count < MIN_PREPRINT_FAMILIES (or all null) are
    filtered out. Published papers always pass.

    Returns:
        Tuple of (kept evidence, filtered evidence with doi+reason dicts)
    """
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for evidence in evidence_list:
        if not is_preprint(evidence.get("journal"), evidence.get("pmid")):
            kept.append(evidence)
            continue
        max_fc = _max_family_count(evidence)
        if max_fc is not None and max_fc >= MIN_PREPRINT_FAMILIES:
            kept.append(evidence)
        elif max_fc is None:
            filtered.append(
                {"doi": evidence["doi"], "reason": "Preprint: family count not reported"}
            )
        else:
            filtered.append(
                {
                    "doi": evidence["doi"],
                    "reason": f"Preprint: {max_fc} families (min {MIN_PREPRINT_FAMILIES} required)",
                }
            )
    return kept, filtered


def evidence_already_in_panelapp(
    evidence_list: list[dict[str, Any]],
    panelapp_publications: PanelPublications,
) -> bool:
    """True iff every evidence paper matches a PanelApp pub by DOI or PMID.

    DOI matching is case-insensitive (PanelApp publication strings are free-text
    and not case-normalized). PMID matching is exact. Caller is responsible for
    guarding against empty evidence_list (which would vacuously return True).
    """
    panelapp_dois_lower = {d.lower() for d in panelapp_publications.dois}
    for evidence in evidence_list:
        doi = evidence["doi"]
        pmid = evidence["pmid"]
        in_dois = doi.lower() in panelapp_dois_lower
        in_pmids = pmid is not None and pmid in panelapp_publications.pmids
        if not (in_dois or in_pmids):
            return False
    return True


def _citation_lists(parsed_json: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Every citation list of an assessment: the associations' criteria and quality concerns."""
    lists: list[list[dict[str, Any]]] = []
    for entity in parsed_json["disease_entities"]:
        lists += [criterion["citations"] for criterion in entity["evidence_assessments"]]
    lists += [concern["citations"] for concern in parsed_json["quality_concerns"]]
    return lists


def drop_placeholder_citations(parsed_json: dict[str, Any]) -> int:
    """Remove the citations whose quote is a placeholder; returns how many."""
    dropped = 0
    for citations in _citation_lists(parsed_json):
        kept = [citation for citation in citations if not is_placeholder(citation["quote"])]
        dropped += len(citations) - len(kept)
        citations[:] = kept
    return dropped


def uncopied_citations(
    parsed_json: dict[str, Any], quotes_by_doi: dict[str, set[str]]
) -> tuple[int, list[tuple[str, str]]]:
    """Citations whose quote isn't one of that paper's extraction quotes.

    Checked after paper_id→doi replacement. Aggregate citations should copy an
    extraction quote exactly, because the report finds a quote's highlight by
    exact lookup in ``citation_locations``; a citation that doesn't is kept and
    shown as not located. Returns the total number of citations and the
    uncopied (doi, quote) pairs.
    """
    total = 0
    uncopied: list[tuple[str, str]] = []
    for citations in _citation_lists(parsed_json):
        total += len(citations)
        uncopied += [
            (c["doi"], c["quote"])
            for c in citations
            if c["quote"] not in quotes_by_doi.get(c["doi"], set())
        ]
    return total, uncopied


def fetch_quotes_by_doi(db_path: Path, dois: set[str]) -> dict[str, set[str]]:
    """Every extraction quote of each paper, located in its PDF or not."""
    quotes: dict[str, set[str]] = {doi: set() for doi in dois}
    with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
        for doi, quote in conn.execute(
            f"SELECT paper_doi, quote FROM citation_locations WHERE paper_doi IN ({','.join('?' * len(dois))})",
            sorted(dois),
        ):
            quotes[doi].add(quote)
    return quotes


class PaperBatchProcessor:
    """Handle database operations for aggregate gene assessment."""

    def __init__(self, db_path: Path):
        """Initialize with database path."""
        self.db_path = db_path

    def get_evidence_for_gene(self, hgnc_id: int) -> list[dict[str, Any]]:
        """Get all evidence extractions for a specific gene.

        One entry per paper. ``paper_gene_symbol`` comes from the extraction's own
        gene mention, so a relevance-stage mention under an older symbol of the same
        gene neither duplicates the paper nor adds a stale alias.

        Args:
            hgnc_id: HGNC ID of the gene to search for

        Returns:
            List of dicts with doi, date, title, authors, paper_gene_symbol, gene_evaluations
        """
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT p.doi, p.pmid, p.journal, p.source_date, p.title, p.authors,
                       p.evidence_extraction_json, gm.paper_gene_symbol
                FROM papers p
                JOIN gene_mentions gm ON p.doi = gm.paper_doi
                WHERE gm.hgnc_id = ?
                AND gm.source IN ('recent_evidence', 'expansion_evidence')
                AND p.evidence_extraction_json IS NOT NULL
                ORDER BY p.doi
            """,
                (hgnc_id,),
            )

            evidence_list = []
            for row in cursor.fetchall():
                try:
                    evidence_json = json.loads(row["evidence_extraction_json"])
                    # Filter to only include evaluations for this specific gene (by hgnc_id)
                    filtered_evaluations = []
                    for gene_eval in evidence_json.get("gene_evaluations", []):
                        if gene_eval.get("hgnc_id") == hgnc_id:
                            del gene_eval["variants"]  # Drop detailed variants from prompt.
                            filtered_evaluations.append(gene_eval)

                    if filtered_evaluations:
                        evidence_list.append(
                            {
                                "doi": row["doi"],
                                "pmid": row["pmid"],
                                "journal": row["journal"],
                                "date": row["source_date"],
                                "title": row["title"],
                                "authors": row["authors"],
                                "paper_gene_symbol": row["paper_gene_symbol"],
                                "gene_evaluations": filtered_evaluations,
                            }
                        )

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing evidence for DOI {row['doi']}: {e}")
                    continue

            logger.debug(f"Found {len(evidence_list)} papers with evidence for HGNC:{hgnc_id}")
            return evidence_list

    def get_variants_for_gene(self, hgnc_id: int, dois: list[str]) -> list[GeneVariant]:
        """The gene's distinct variants reported by these papers."""
        with closing(sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS)) as conn:
            return load_gene_variants(conn.cursor(), hgnc_id, dois)

    def count_remaining(self) -> int:
        """The run's genes that have no aggregation yet."""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            (remaining,) = conn.execute(
                f"""
                SELECT COUNT(DISTINCT hgnc_id) FROM ({RUN_GENES})
                WHERE hgnc_id NOT IN (SELECT hgnc_id FROM gene_aggregations)
                """
            ).fetchone()
        return int(remaining)


def _review_lines(reviews: list[dict[str, Any]]) -> list[str]:
    """One panel's PanelApp evaluations as prompt lines, numbered in the given order."""
    lines: list[str] = []
    for i, review in enumerate(reviews, 1):
        lines.append(f"Review {i}:")

        rating = review.get("rating")
        if rating:
            lines.append(f"  Rating: {rating}")

        moi = review.get("moi")
        if moi:
            lines.append(f"  Mode of Inheritance: {moi}")

        phenotypes = review.get("phenotypes")
        if phenotypes:
            lines.append(f"  Phenotypes: {', '.join(phenotypes)}")

        publications = review.get("publications")
        if publications:
            lines.append(f"  Publications: {'; '.join(publications)}")

        comments = review.get("comments", [])
        if comments:
            lines.append("  Comments:")
            for comment in comments:
                user = comment.get("user_name", "Unknown")
                date = comment.get("created", "")
                text = comment.get("comment", "")
                lines.append(f"    [{date}] {user}: {text}")

        lines.append("")
    return lines


def format_panel_reviews(panel_reviews: list[PanelReviews]) -> str:
    """Each target panel's reviews of the gene as one XML-tagged block labelled with the panel.

    Panels that hold the gene without any review are left out; empty when no panel has one.
    """
    blocks = [
        "\n".join(
            [
                f'<previous_reviews panel="{p.panel_name}" panel_id="{p.panel_id}">',
                *_review_lines(p.evaluations),
                "</previous_reviews>",
            ]
        )
        for p in panel_reviews
        if p.evaluations
    ]
    return "\n\n".join(blocks)


def dispute_blocks(
    gencc: GeneGencc, evidence_list: list[dict[str, Any]], doi_to_paper_id: dict[str, str]
) -> list[dict[str, Any]]:
    """One block per Disputed/Refuted submission, with the PMID overlap pre-computed.

    The overlap tells the model which contributing papers the submitter already
    evaluated, so it does not have to match PMIDs itself when deciding an overrule.
    """
    contributing = [
        {"paper_id": doi_to_paper_id[e["doi"]], "pmid": e["pmid"], "date": e["date"] or ""}
        for e in evidence_list
    ]
    by_pmid = {info["pmid"]: info for info in contributing if info["pmid"] is not None}
    blocks = []
    for d in gencc.disputes:
        evaluated = set(d.pmids)
        blocks.append(
            {
                **asdict(d),
                "evaluated_pmids": list(d.pmids),
                "overlapping_contributing_papers": [
                    by_pmid[pmid] for pmid in d.pmids if pmid in by_pmid
                ],
                "independent_contributing_papers": [
                    info
                    for info in contributing
                    if info["pmid"] is None or info["pmid"] not in evaluated
                ],
            }
        )
    return blocks


@dataclass(frozen=True)
class PanelAppContext:
    """What PanelApp Australia curates for one gene, as shown to the model."""

    gencc: GeneGencc
    panel_entries: tuple[PanelEntry, ...]
    all_panels: bool  # Incidentalome-only gene, shown with its entries from every panel
    panel_reviews: list[PanelReviews]

    def to_json(self) -> dict[str, Any]:
        """The ``panelapp_context_json`` of ``gene_aggregations``."""
        return {
            "gencc_rows": [asdict(a) for a in self.gencc.paa_associations],
            "disputes": [asdict(d) for d in self.gencc.disputes],
            "panel_entries": [
                {
                    "panel_id": e.panel_id,
                    "panel_name": e.panel_name,
                    "rating": e.rating,
                    "moi": e.moi,
                    "mode_of_pathogenicity": e.mode_of_pathogenicity,
                    "phenotypes": list(e.phenotypes),
                    "publications": list(e.publications),
                }
                for e in self.panel_entries
            ],
            "all_panels": self.all_panels,
        }

    @property
    def has_curation(self) -> bool:
        """Whether the model is shown any PanelApp Australia GenCC row, panel entry or review."""
        return bool(
            self.gencc.paa_associations
            or self.panel_entries
            or any(p.evaluations for p in self.panel_reviews)
        )

    def reviews_json(self) -> list[dict[str, Any]] | None:
        """The ``existing_panel_reviews_json`` of ``gene_aggregations``."""
        if not self.panel_reviews:
            return None
        return [{"panel_id": p.panel_id, "evaluations": p.evaluations} for p in self.panel_reviews]


@dataclass(frozen=True)
class VariantRow:
    """One row of the prompt's variant table, its cells formatted."""

    id: str  # V1, V2, ...
    hgvs: str
    paper_ids: str
    gnomad: str


def _scientific(value: float) -> str:
    return "0" if value == 0 else f"{value:.2e}"


def _gnomad_cell(gnomad: GnomadFrequency | GnomadMiss) -> str:
    match gnomad:
        case GnomadMiss.NOT_FOUND:
            return "not in gnomAD"
        case GnomadMiss.LOOKUP_FAILED:
            return "lookup failed"
        case GnomadFrequency():
            cell = (
                f"AF {_scientific(gnomad.allele_frequency)} (AC/AN {gnomad.ac}/{gnomad.an}); "
                f"het {gnomad.heterozygote_count}, hom {gnomad.homozygote_count}, "
                f"hemi {gnomad.hemizygote_count}"
            )
            if gnomad.faf95_popmax is None:
                return cell
            return (
                f"{cell}; FAF95 popmax {_scientific(gnomad.faf95_popmax)} "
                f"({gnomad.faf95_popmax_population})"
            )


def _hgvs_cell(reports: list[VariantReport]) -> str:
    """The variant's HGVS forms across *reports*, or the papers' text when it was not normalised.

    Papers may number the variant on different transcripts, so each distinct form is listed.
    """
    normalizations = [r.normalization for r in reports if r.normalization is not None]
    if not normalizations:
        return " / ".join(dict.fromkeys(r.original_text for r in reports))
    forms = dict.fromkeys(f"{n.hgvs_c} ({n.hgvs_p.split(':')[-1]})" for n in normalizations)
    cell = "; ".join(forms)
    if all(n.candidate_count > 1 for n in normalizations):
        cell += (
            f" [written {reports[0].original_text}: {normalizations[0].candidate_count} "
            "possible DNA changes, the one most frequent in gnomAD shown]"
        )
    return cell


def _genomic_position(variant: GeneVariant) -> int:
    """The position of a normalised variant's pseudo-VCF (chr-pos-ref-alt); 0 otherwise."""
    if variant.gnomad == GnomadMiss.LOOKUP_FAILED:
        return 0
    return int(variant.variant_id.split("-")[1])


def variant_table(
    variants: list[GeneVariant], doi_to_paper_id: dict[str, str]
) -> tuple[list[VariantRow], dict[str, GeneVariant]]:
    """The prompt's variant table and its variant ID → variant mapping.

    Rows go by their first reporting paper ID, then normalised variants by genomic
    position before those whose lookup failed, by text.
    """

    def sort_key(variant: GeneVariant) -> tuple[str, bool, int, str]:
        return (
            min(doi_to_paper_id[doi] for doi in variant.dois),
            variant.gnomad == GnomadMiss.LOOKUP_FAILED,
            _genomic_position(variant),
            variant.variant_id,
        )

    rows: list[VariantRow] = []
    by_id: dict[str, GeneVariant] = {}
    for i, variant in enumerate(sorted(variants, key=sort_key), 1):
        variant_id = f"V{i}"
        reports = sorted(variant.reports, key=lambda r: doi_to_paper_id[r.doi])
        rows.append(
            VariantRow(
                id=variant_id,
                hgvs=_hgvs_cell(reports),
                paper_ids=", ".join(sorted({doi_to_paper_id[r.doi] for r in reports})),
                gnomad=_gnomad_cell(variant.gnomad),
            )
        )
        by_id[variant_id] = variant
    return rows, by_id


@dataclass(frozen=True)
class RenderedPrompt:
    """One gene's aggregation prompt and the mappings of the IDs it introduces."""

    text: str
    paper_id_to_doi: dict[str, str]
    variants: dict[str, GeneVariant]  # by variant ID (V1, V2, ...)


def render_prompt(
    template_path: Path,
    hgnc_symbol: str,
    evidence_list: list[dict[str, Any]],
    variants: list[GeneVariant],
    context: PanelAppContext,
    panel_date: str,
    panel_formatted: str,
) -> RenderedPrompt:
    """The aggregation prompt for one gene, with its paper and variant ID mappings.

    Generates {LastName}{Year} paper IDs for LLM-friendly citation, and V1, V2, ...
    IDs for the variant table. The LLM refers to papers and variants by these IDs;
    the caller maps them back to DOIs and variant keys after receiving the response.
    ``panel_formatted`` is the scope panel's description in panel-scoped runs and
    empty otherwise.
    """
    paper_id_to_doi, doi_to_paper_id = generate_paper_ids(evidence_list)
    variant_rows, variants_by_id = variant_table(variants, doi_to_paper_id)
    prompt_evidence = [
        {
            "paper_id": doi_to_paper_id[e["doi"]],
            "date": e["date"],
            "title": e["title"],
            "gene_evaluations": e["gene_evaluations"],
        }
        for e in evidence_list
    ]

    # Paper symbols are uppercased; current symbols such as C9orf72 are not.
    aliases = {e["paper_gene_symbol"] for e in evidence_list} - {hgnc_symbol.upper()}
    gene_symbol = (
        f"{hgnc_symbol} (also referred to as: {', '.join(sorted(aliases))} in the papers)"
        if aliases
        else hgnc_symbol
    )

    env = Environment(
        loader=FileSystemLoader(template_path.parent),
        autoescape=False,
        undefined=StrictUndefined,
    )
    rendered = env.get_template(template_path.name).render(
        gene_symbol=gene_symbol,
        hgnc_symbol=hgnc_symbol,
        evidence_extractions=json.dumps(prompt_evidence, indent=2),
        variant_rows=variant_rows,
        paa_associations=context.gencc.paa_associations,
        panel_entries=context.panel_entries,
        all_panels=context.all_panels,
        incidentalome_scope=INCIDENTALOME_SCOPE,
        reviews_section=format_panel_reviews(context.panel_reviews),
        dispute_blocks=dispute_blocks(context.gencc, evidence_list, doi_to_paper_id),
        panel_date=panel_date,
        panel_formatted=panel_formatted,
    )
    return RenderedPrompt(text=rendered, paper_id_to_doi=paper_id_to_doi, variants=variants_by_id)


@dataclass
class _GeneBatchItem:
    """Everything needed to send, validate and store one gene's aggregation."""

    hgnc_id: int
    hgnc_symbol: str
    prompt: str
    paper_id_to_doi: dict[str, str]
    variants: dict[str, GeneVariant]  # by variant ID (V1, V2, ...)
    evidence_list: list[dict[str, Any]]
    filtered_papers: list[dict[str, Any]] | None
    context: PanelAppContext


@dataclass(frozen=True)
class _GenePreparation:
    """Shared inputs for preparing gene prompts."""

    db_processor: PaperBatchProcessor
    hgnc_resolver: HgncResolver
    panelapp_client: PanelAppClient
    panel_data: PanelGeneData
    record: CuratedRecord
    prompt_path: Path
    panel_formatted: str


def target_panels_holding(hgnc_id: int, panel_data: PanelGeneData) -> list[int]:
    """The target panels with an entry for the gene, in target-panel order."""
    held = panel_data.gene_panel_confidence.get(hgnc_id, {})
    return [panel_id for panel_id in panel_data.panel_ids if panel_id in held]


def fetch_panel_reviews(
    hgnc_id: int, client: PanelAppClient, panel_data: PanelGeneData
) -> list[PanelReviews]:
    """The gene's reviews on every target panel that holds it."""
    return [
        PanelReviews(
            panel_id=panel_id,
            panel_name=client.get_panel_data(panel_id)["name"],
            evaluations=client.get_gene_evaluations(panel_id, hgnc_id),
        )
        for panel_id in target_panels_holding(hgnc_id, panel_data)
    ]


def cited_on_panelapp(
    hgnc_id: int, client: PanelAppClient, panel_reviews: list[PanelReviews]
) -> PanelPublications:
    """Publications cited for the gene by its entries and reviews on these target panels."""
    pmids: set[int] = set()
    dois: set[str] = set()
    for panel in panel_reviews:
        cited = collect_panelapp_gene_publications(
            client.get_panel_data(panel.panel_id), hgnc_id, panel.evaluations
        )
        pmids |= cited.pmids
        dois |= cited.dois
    return PanelPublications(pmids=pmids, dois=dois)


def prepare_gene(hgnc_id: int, prep: _GenePreparation) -> _GeneBatchItem | None:
    """The prompt and validation context for one gene, or None when it is skipped."""
    hgnc_symbol = prep.hgnc_resolver.get_symbol(hgnc_id)
    evidence_list = prep.db_processor.get_evidence_for_gene(hgnc_id)
    if not evidence_list:
        logger.warning(f"No evidence found for {hgnc_symbol}")
        return None

    evidence_list, filtered_papers = filter_preprint_evidence(evidence_list)
    if filtered_papers:
        filtered_dois = [fp["doi"] for fp in filtered_papers]
        logger.info(
            f"  Filtered {len(filtered_papers)} preprint(s) for {hgnc_symbol}: {filtered_dois}"
        )
    if not evidence_list:
        logger.info(f"Skipping {hgnc_symbol} — all papers filtered by preprint family gate")
        return None

    panel_reviews = fetch_panel_reviews(hgnc_id, prep.panelapp_client, prep.panel_data)
    if panel_reviews and evidence_already_in_panelapp(
        evidence_list, cited_on_panelapp(hgnc_id, prep.panelapp_client, panel_reviews)
    ):
        logger.info(
            f"Skipping {hgnc_symbol} — all {len(evidence_list)} paper(s) already cited on "
            f"target panels {[p.panel_id for p in panel_reviews]}: "
            f"{[e['doi'] for e in evidence_list]}"
        )
        return None

    gene_record = prep.record.genes.get(hgnc_id)
    context = PanelAppContext(
        gencc=prep.record.gencc.for_gene(hgnc_id),
        panel_entries=gene_record.entries if gene_record is not None else (),
        all_panels=gene_record is not None and gene_record.all_panels,
        panel_reviews=panel_reviews,
    )
    prompt = render_prompt(
        prep.prompt_path,
        hgnc_symbol,
        evidence_list,
        prep.db_processor.get_variants_for_gene(hgnc_id, [e["doi"] for e in evidence_list]),
        context,
        prep.record.panel_date,
        prep.panel_formatted,
    )
    return _GeneBatchItem(
        hgnc_id=hgnc_id,
        hgnc_symbol=hgnc_symbol,
        prompt=prompt.text,
        paper_id_to_doi=prompt.paper_id_to_doi,
        variants=prompt.variants,
        evidence_list=evidence_list,
        filtered_papers=filtered_papers or None,
        context=context,
    )


def genes_to_assess(db_path: Path, only: list[int] | None, history: StageHistory) -> list[int]:
    """The run's genes without an aggregation, except ones skipped for good and in flight.

    The run's genes have recent evidence from a relevant paper (see
    :mod:`palit.run_corpus`). ``only`` restricts the result to these HGNC IDs.
    """
    with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT g.hgnc_id FROM ({RUN_GENES}) g
            WHERE g.hgnc_id NOT IN (SELECT hgnc_id FROM gene_aggregations)
              AND NOT EXISTS (
                  SELECT 1 FROM llm_requests r
                  WHERE r.stage = ? AND r.subject = CAST(g.hgnc_id AS TEXT)
                    AND r.status = 'pending'
              )
            ORDER BY g.hgnc_id
            """,
            (STAGE,),
        ).fetchall()
    hgnc_ids = [row[0] for row in rows if not history.skipped(str(row[0]))]
    if only is None:
        return hgnc_ids
    wanted = set(only)
    return [h for h in hgnc_ids if h in wanted]


def output_configs(schema: dict[str, Any]) -> dict[str, OutputConfigParam]:
    """The structured-output config of each model, with that model's effort."""
    return {model: json_output_config(schema, effort) for model, effort in EFFORT_BY_MODEL.items()}


def build_request(
    item: _GeneBatchItem, configs: dict[str, OutputConfigParam], model: str
) -> LlmRequest:
    return LlmRequest(
        subject=str(item.hgnc_id),
        params={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": item.prompt}],
            "output_config": configs[model],
        },
    )


def dois_with_disease_entities(evidence_list: list[dict[str, Any]]) -> set[str]:
    """Contributing papers whose extraction reports at least one disease entity for the gene."""
    return {
        e["doi"]
        for e in evidence_list
        if any(gene_eval["disease_entities"] for gene_eval in e["gene_evaluations"])
    }


def uncovered_dois(assessment: dict[str, Any], required_dois: set[str]) -> list[str]:
    """Papers of *required_dois* in no association's and no unassessed report's dois."""
    covered = {
        doi
        for section in ("disease_entities", "unassessed_reports")
        for entry in assessment[section]
        for doi in entry["dois"]
    }
    return sorted(required_dois - covered)


def association_problems(assessment: dict[str, Any], item: _GeneBatchItem) -> list[str]:
    """Checks of the association fields that send a gene back for another attempt."""
    paa_mondo_ids = {a.mondo_id for a in item.context.gencc.paa_associations}
    problems = []
    for entity in assessment["disease_entities"]:
        mondo_id = entity["existing_association_mondo_id"]
        proposed = entity["proposed_disease_name"]
        if (mondo_id is None) == (proposed is None):
            problems.append(
                f"{entity['description']!r}: set exactly one of existing_association_mondo_id "
                f"({mondo_id!r}) and proposed_disease_name ({proposed!r})"
            )
        if mondo_id is not None and mondo_id not in paa_mondo_ids:
            problems.append(
                f"{entity['description']!r}: {mondo_id} is not a PanelApp Australia GenCC row"
            )
        independent = entity["independent_family_count"]
        status = entity["panelapp_relation"]["status"]
        if status == "existing" and not item.context.has_curation:
            problems.append(
                f"{entity['description']!r}: status existing, but PanelApp Australia has no "
                "GenCC row, panel entry or review for this gene, so it curates no association "
                "of it; use new_disease, or unassessed_reports without a qualifying family"
            )
        if status in NEW_RELATION_STATUSES and (independent is None or independent < 1):
            problems.append(
                f"{entity['description']!r}: {status} association with independent_family_count "
                f"{independent!r} belongs in unassessed_reports"
            )
    return problems


def log_association_warnings(assessment: dict[str, Any], item: _GeneBatchItem) -> None:
    """Inconsistencies worth reviewing but not worth another attempt."""
    gencc = item.context.gencc
    variants = {v.variant_id: v for v in item.variants.values()}
    variant_ids = {v.variant_id: variant_id for variant_id, v in item.variants.items()}
    anchor_mois: dict[str, set[str | None]] = {}
    for a in gencc.paa_associations:
        anchor_mois.setdefault(a.mondo_id, set()).add(a.moi)
    for entity in assessment["disease_entities"]:
        relation = entity["panelapp_relation"]
        status = relation["status"]
        label = f"{item.hgnc_symbol} {entity['description']!r} ({entity['inheritance_mode']})"
        if status in NEW_RELATION_STATUSES and relation["existing_rating"] is not None:
            logger.warning("%s: %s association with existing_rating %s", label, status, relation)
        details = entity["inheritance_details"]
        unknown_details = [
            item
            for item in inheritance_details_items(details)
            if item not in INHERITANCE_DETAILS_VOCABULARY
        ]
        if unknown_details:
            logger.warning(
                "%s: inheritance_details %r has items outside the vocabulary: %s",
                label,
                details,
                unknown_details,
            )
        mondo_id = entity["existing_association_mondo_id"]
        if mondo_id is not None and status in NEW_RELATION_STATUSES:
            if entity["inheritance_mode"] in anchor_mois[mondo_id]:
                logger.warning(
                    "%s: reuses GenCC row %s with its MoI but says %s", label, mondo_id, status
                )
            elif status == "new_disease":
                logger.warning(
                    "%s: reuses the disease of GenCC row %s but says new_disease", label, mondo_id
                )
        if entity["dispute_status"] != "None" and not gencc.disputes:
            logger.warning(
                "%s: dispute_status %s without any dispute submission",
                label,
                entity["dispute_status"],
            )
        cited = {
            c["doi"] for criterion in entity["evidence_assessments"] for c in criterion["citations"]
        }
        unlisted = cited - set(entity["dois"])
        if unlisted:
            logger.warning(
                "%s: criteria cite papers missing from its paper_ids: %s", label, sorted(unlisted)
            )
        unreported = [
            f"{variant_ids[key]} ({key})"
            for key in entity["variants"]
            if not variants[key].dois & set(entity["dois"])
        ]
        if unreported:
            logger.warning(
                "%s: lists variants that none of its paper_ids reports: %s",
                label,
                ", ".join(unreported),
            )
    # The report lists every paper with an extraction for the gene, so a paper the
    # aggregation leaves out stays visible there; the gene isn't worth losing over it.
    missing = uncovered_dois(assessment, dois_with_disease_entities(item.evidence_list))
    if missing:
        doi_to_paper_id = {doi: paper_id for paper_id, doi in item.paper_id_to_doi.items()}
        logger.warning(
            "%s: %d contributing paper(s) in no association or unassessed report: %s",
            item.hgnc_symbol,
            len(missing),
            ", ".join(f"{doi_to_paper_id[doi]} ({doi})" for doi in missing),
        )


def drop_placeholders_and_log_quotes(
    assessment: dict[str, Any], item: _GeneBatchItem, db_path: Path
) -> None:
    """Drop the placeholder citations and log them with the uncopied ones.

    Quotes never send a gene back: a citation whose quote isn't copied from the
    paper's extraction is kept, and the report shows it as not located.
    """
    placeholders = drop_placeholder_citations(assessment)
    quotes_by_doi = fetch_quotes_by_doi(db_path, {e["doi"] for e in item.evidence_list})
    total, uncopied = uncopied_citations(assessment, quotes_by_doi)
    logger.info(
        "%s: %d citations; %d placeholder citations dropped; %d quotes not copied from "
        "the extractions%s",
        item.hgnc_symbol,
        total,
        placeholders,
        len(uncopied),
        f": {'; '.join(repr(quote[:60]) for _, quote in uncopied[:3])}" if uncopied else "",
    )


def assessment_problems(assessment: dict[str, Any], item: _GeneBatchItem) -> list[str]:
    """Problems that send a gene back for another attempt.

    Maps paper IDs to DOIs and variant IDs to variant keys in place.

    Expects an answer that validates against the full schema, with its criteria
    already converted to the stored list.
    """
    try:
        replace_paper_ids_with_dois(assessment, item.paper_id_to_doi)
    except ValueError as e:
        return [f"hallucinated paper ID: {e}"]
    try:
        replace_variant_ids(assessment, {vid: v.variant_id for vid, v in item.variants.items()})
    except ValueError as e:
        return [f"hallucinated variant ID: {e}"]
    problems = []
    if not validate_entities_criteria_complete(assessment["disease_entities"]):
        problems.append("incomplete per-association criteria")
    if not validate_independent_family_counts(assessment["disease_entities"]):
        problems.append("inconsistent independent_family_count")
    problems += association_problems(assessment, item)
    return problems


def store_gene_aggregation(
    conn: sqlite3.Connection,
    item: _GeneBatchItem,
    message: Message,
    assessment: dict[str, Any],
) -> None:
    """Replace the gene's aggregation and its associations with this one.

    The old association rows go, and with them any MONDO mapping or panel matches
    made for them. An association that reuses a PanelApp Australia GenCC row gets
    that row's MONDO term now; the others stay unmapped until ``map-mondo``.
    """
    gencc_titles = {a.mondo_id: a.disease_title for a in item.context.gencc.paa_associations}
    # Explicit, because the connection does not enable foreign keys, so the
    # schema's ON DELETE CASCADE does not fire; this also removes rows left
    # behind by a gene_aggregations row deleted by hand.
    conn.execute("DELETE FROM associations WHERE hgnc_id = ?", (item.hgnc_id,))
    conn.execute(
        """
        INSERT OR REPLACE INTO gene_aggregations
            (hgnc_id, assessment_raw, paper_id_mapping, filtered_papers_json,
             panelapp_context_json, existing_panel_reviews_json, unassessed_reports_json,
             quality_concerns_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.hgnc_id,
            message.to_json(),
            json.dumps(item.paper_id_to_doi),
            json.dumps(item.filtered_papers) if item.filtered_papers else None,
            json.dumps(item.context.to_json()),
            json.dumps(reviews) if (reviews := item.context.reviews_json()) is not None else None,
            json.dumps(assessment["unassessed_reports"]),
            json.dumps(assessment["quality_concerns"]),
        ),
    )
    rows = []
    for position, entity in enumerate(assessment["disease_entities"]):
        mondo_id = entity["existing_association_mondo_id"]
        rows.append(
            (
                item.hgnc_id,
                position,
                json.dumps(entity),
                mondo_id,
                gencc_titles[mondo_id] if mondo_id is not None else None,
                "panelapp_gencc" if mondo_id is not None else None,
            )
        )
    conn.executemany(
        """
        INSERT INTO associations
            (hgnc_id, position, assessment_json, mondo_id, mondo_label, mondo_match)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _association_overview(assessment: dict[str, Any]) -> str:
    return ", ".join(
        f"{panelapp_confidence_to_color(calculate_association_rating(e)).upper()} "
        f"{e['panelapp_relation']['status']}"
        for e in assessment["disease_entities"]
    )


@dataclass
class _Outcome:
    stored: int = 0
    refused: int = 0
    to_fallback: int = 0  # the refusals by MODEL: these genes go to FALLBACK_MODEL next
    failed: int = 0


def handle_results(
    results: list[LlmResult],
    items: dict[str, _GeneBatchItem],
    db_path: Path,
    validator: jsonschema.protocols.Validator,
) -> _Outcome:
    outcome = _Outcome()
    invalid_requests: list[str] = []
    for result in results:
        item = items.get(result.subject)
        message = result.message
        stored = False
        rejection: str | None = None  # why an answer was rejected
        if result.status == ResultStatus.REFUSED:
            outcome.refused += 1
            outcome.to_fallback += result.goes_to_fallback
            log_refusal(result, f"HGNC:{result.subject}")
        elif result.status != ResultStatus.SUCCEEDED or message is None:
            outcome.failed += 1
            if result.error_type == "invalid_request_error":
                invalid_requests.append(result.subject)
        elif item is None:
            logger.info(
                "HGNC:%s no longer needs an assessment; discarding its result", result.subject
            )
        else:
            try:
                assessment = parse_json_output(message)
                validator.validate(assessment)
                criteria_object_to_list(assessment["disease_entities"])
                problems = assessment_problems(assessment, item)
            except (ValueError, jsonschema.ValidationError) as e:
                problems = [invalid_answer_reason(e)]
            if problems:
                rejection = "; ".join(problems[:5])
                logger.warning("Rejected assessment for %s: %s", item.hgnc_symbol, rejection)
                outcome.failed += 1
            else:
                drop_placeholders_and_log_quotes(assessment, item, db_path)
                log_association_warnings(assessment, item)
                with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
                    record_result(conn, result)
                    store_gene_aggregation(conn, item, message, assessment)
                logger.info(
                    "Stored %s: %d associations (%s), %d unassessed reports",
                    item.hgnc_symbol,
                    len(assessment["disease_entities"]),
                    _association_overview(assessment),
                    len(assessment["unassessed_reports"]),
                )
                stored = True
                outcome.stored += 1
        if not stored:
            with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
                record_result(conn, result, rejection=rejection)
    if invalid_requests:
        raise RuntimeError(
            f"{len(invalid_requests)} assess-genes requests were invalid "
            f"(first: HGNC:{invalid_requests[0]}); see llm_requests.error_type"
        )
    return outcome


async def _process_assessments(
    *,
    transport: Transport,
    db_path: Path,
    prep: _GenePreparation,
    schema: dict[str, Any],
    only: list[int] | None,
    limit: int | None,
    max_retries: int,
    failures_since: datetime,
) -> None:
    """Assess every gene due an assessment, in attempts.

    An attempt makes progress when it stores an assessment or sends a gene on to
    FALLBACK_MODEL, so a round of refusals by MODEL is followed by the attempt
    that sends those genes to FALLBACK_MODEL. Failed answers count from
    *failures_since*.
    """

    def load_history() -> StageHistory:
        with closing(sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS)) as conn:
            return stage_history(conn, STAGE, failures_since=failures_since)

    validator = jsonschema.Draft202012Validator(schema)
    configs = output_configs(schema)
    items: dict[str, _GeneBatchItem] = {}
    skipped: set[int] = set()

    def prepared(hgnc_ids: list[int]) -> list[_GeneBatchItem]:
        ready = []
        for hgnc_id in hgnc_ids:
            if hgnc_id in skipped:
                continue
            item = items.get(str(hgnc_id))
            if item is None:
                item = prepare_gene(hgnc_id, prep)
                if item is None:
                    skipped.add(hgnc_id)
                    continue
                items[str(hgnc_id)] = item
            ready.append(item)
        return ready

    resumed = await transport.resume(STAGE)
    if resumed:
        prepared(sorted({int(r.subject) for r in resumed}))
        outcome = handle_results(resumed, items, db_path, validator)
        logger.info("Collected %d results from earlier batches: %s", len(resumed), outcome)

    log_failed_for_good(load_history().failures, STAGE)
    for attempt in range(1, max_retries + 1):
        history = load_history()
        hgnc_ids = [g for g in genes_to_assess(db_path, only, history) if g not in skipped]
        if limit is not None:
            hgnc_ids = hgnc_ids[:limit]
        batch = prepared(hgnc_ids)
        if not batch:
            logger.info("No genes left to assess")
            return
        logger.info("Attempt %d: assessing %d genes", attempt, len(batch))
        requests = [
            build_request(item, configs, history.model_for(str(item.hgnc_id))) for item in batch
        ]
        outcome = handle_results(await transport.run(STAGE, 1, requests), items, db_path, validator)
        logger.info(
            "Attempt %d: stored %d, refused %d (%d go to the fallback model), failed %d",
            attempt,
            outcome.stored,
            outcome.refused,
            outcome.to_fallback,
            outcome.failed,
        )
        if outcome.stored + outcome.to_fallback == 0:
            logger.error("No progress in attempt %d - stopping", attempt)
            return
        if limit is not None:
            return


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database",
    ),
    prompt_path: Path = typer.Option(
        Path("prompts/aggregate_assessment_prompt.j2"),
        "--prompt-path",
        "-p",
        help="Path to Jinja2 prompt template file",
    ),
    schema_path: Path = typer.Option(
        Path("prompts/aggregate_assessment_schema.json"),
        "--schema-path",
        "-s",
        help="Path to response schema file",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of attempts for genes whose assessment failed, was rejected or "
        "was refused; the fallback-model request after a refusal is one of them",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Send genes whose answers failed for good in earlier invocations (rejected or "
        "cut off at max_tokens, MAX_FAILED_ANSWERS times) again, until they fail that often "
        "in this invocation",
    ),
    panel_date: str = typer.Option(
        ...,
        "--panel-date",
        help="Panel state date (YYYY-MM-DD) for checking gene panel membership",
    ),
    target_panel_ids: list[int] | None = typer.Option(
        None,
        "--target-panel-ids",
        help="Panel IDs to check for existing genes. Can be specified multiple times. Defaults to TARGET_PANEL_IDS.",
    ),
    scope_panel_id: int | None = typer.Option(
        None,
        "--scope-panel-id",
        help="Panel ID for panel-scoped assessment. When set, each association's summary must explain how it relates to this panel's scope.",
    ),
    immediate: bool = typer.Option(
        False,
        "--immediate",
        help="Send requests immediately instead of as Message Batches (for prompt development)",
    ),
    hgnc_ids: list[int] | None = typer.Option(
        None,
        "--hgnc-id",
        help="Assess only this gene (HGNC ID without prefix; repeatable; for prompt development)",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Assess at most this many genes, in one attempt (for prompt development)",
    ),
) -> None:
    """Aggregate each gene's evidence from multiple papers into assessed associations."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)
    # With --retry-failed, only failed answers during this invocation count.
    failures_since = datetime.now(UTC) if retry_failed else ALL_TIME

    schema: dict[str, Any] = json.loads(schema_path.read_text())
    hgnc_resolver = HgncResolver.from_file()

    logger.info(f"Fetching PanelApp gene data for {panel_date}...")
    panelapp_client = PanelAppClient(panel_date)
    panel_data = panelapp_client.get_target_panels_genes(target_panel_ids)
    logger.info(
        f"  Loaded {len(panel_data.gene_panel_confidence)} genes from {len(panel_data.panel_ids)} target panels"
    )
    # Unscoped runs show an Incidentalome-only gene's entries from all panels, as the
    # PanelApp relevance check does.
    record = CuratedRecord.build(
        panelapp_client.get_all_panel_data(),
        panel_date,
        panel_data.panel_ids,
        fetch_gencc(db_path.parent, fetch_mondo(db_path.parent)),
        incidentalome_fallback=scope_panel_id is None,
    )

    panel_formatted = ""
    if scope_panel_id is not None:
        try:
            panel_info = panelapp_client.get_panel_data(scope_panel_id)
        except ValueError as e:
            logger.error(f"Panel {scope_panel_id} not found in PanelApp data for {panel_date}")
            raise typer.Exit(1) from e
        panel_formatted = format_panel_for_prompt(scope_panel_id, panel_info)
        logger.info(f"  Panel-scoped mode: {panel_info.get('name', 'Unknown')}")

    db_processor = PaperBatchProcessor(db_path)
    logger.info(f"{db_processor.count_remaining():,} genes of this run without an assessment")
    prep = _GenePreparation(
        db_processor=db_processor,
        hgnc_resolver=hgnc_resolver,
        panelapp_client=panelapp_client,
        panel_data=panel_data,
        record=record,
        prompt_path=prompt_path,
        panel_formatted=panel_formatted,
    )

    async def run() -> None:
        client = make_client()
        transport: Transport = (
            ImmediateTransport(client) if immediate else BatchTransport(client, db_path)
        )
        await _process_assessments(
            transport=transport,
            db_path=db_path,
            prep=prep,
            schema=schema,
            only=hgnc_ids,
            limit=limit,
            max_retries=max_retries,
            failures_since=failures_since,
        )

    asyncio.run(run())
    logger.info(f"{db_processor.count_remaining():,} genes still without an assessment")
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
