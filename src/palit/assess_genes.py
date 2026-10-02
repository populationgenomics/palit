#!/usr/bin/env python3
"""Aggregate evidence assessment across papers for each gene."""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import typer
from anthropic.types import Message
from anthropic.types.output_config_param import OutputConfigParam
from jinja2 import Environment, FileSystemLoader

from palit.hgnc import HgncResolver
from palit.llm import (
    MODEL,
    AnthropicSettings,
    BatchTransport,
    Effort,
    ImmediateTransport,
    LlmRequest,
    LlmResult,
    ResultStatus,
    Transport,
    json_output_config,
    make_client,
    parse_json_output,
    record_result,
)
from palit.llm_usage import print_stage_summary
from palit.mondo_lookup import DisputeRecord, DisputeStatus, MondoCandidate, MondoLookup
from palit.panelapp_client import (
    PanelAppClient,
    PanelGeneData,
    PanelPublications,
    collect_panelapp_gene_publications,
    find_gene_panel,
    format_panel_for_prompt,
)
from palit.panelapp_integration import (
    MONDO_CATEGORIES,
    criteria_object_to_list,
    validate_entities_criteria_complete,
    validate_independent_family_counts,
)
from palit.papers import MIN_PREPRINT_FAMILIES, generate_paper_ids, is_preprint

app = typer.Typer(help="Aggregate evidence assessment across papers for each gene")
logger = logging.getLogger(__name__)

STAGE = "assess_genes"
EFFORT: Effort = "medium"
MAX_TOKENS = 64000
# Above this share of citations not copied from the extractions, the assessment
# is redone; below it, those citations are dropped.
MAX_INVALID_CITATION_SHARE = 0.25

DB_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class MondoResolution:
    """Canonical resolution for an LLM-emitted mondo_disease_name."""

    mondo_id: str
    mondo_label: str
    dispute_status: DisputeStatus
    dispute_records: tuple[DisputeRecord, ...]


# Case-insensitive fallback name→resolution lookup (static, built once).
# Fallback categories carry no GenCC submissions, so dispute_status is "None".
_FALLBACK_NAME_LOOKUP: dict[str, MondoResolution] = {
    info["label"].lower(): MondoResolution(
        mondo_id=mondo_id,
        mondo_label=info["label"],
        dispute_status="None",
        dispute_records=(),
    )
    for mondo_id, info in MONDO_CATEGORIES.items()
}


def build_mondo_name_lookup(
    candidates: list[MondoCandidate],
) -> dict[str, MondoResolution]:
    """Build case-insensitive name→resolution lookup from candidates + fallbacks.

    The candidate's GenCC dispute_status overrides the fallback "None" if a
    fallback category name happens to match a candidate title.

    Args:
        candidates: Gene-specific MONDO candidates from GenCC

    Returns:
        Dict mapping lowercased disease name to MondoResolution
    """
    lookup = dict(_FALLBACK_NAME_LOOKUP)  # copy fallbacks
    for c in candidates:
        lookup[c.title.lower()] = MondoResolution(
            mondo_id=c.mondo_id,
            mondo_label=c.title,
            dispute_status=c.dispute_status,
            dispute_records=c.dispute_records,
        )
    return lookup


def _resolve_paper_id(paper_id: str, paper_id_to_doi: dict[str, str]) -> str:
    """Look up DOI for a paper_id. Raises ValueError if unknown."""
    doi = paper_id_to_doi.get(paper_id)
    if doi is None:
        raise ValueError(f"Unknown paper_id: {paper_id}")
    return doi


def replace_paper_ids_with_dois(
    parsed_json: dict[str, Any], paper_id_to_doi: dict[str, str]
) -> None:
    """Replace all paper_id fields with doi fields in parsed LLM output.

    Mutates parsed_json in place. Raises ValueError on the first unknown paper_id
    (LLM hallucination — caller should retry).
    """
    # disease_entities[].citations[] AND disease_entities[].evidence_assessments[].citations[]
    for entity in parsed_json.get("disease_entities", []):
        for citation in entity.get("citations", []):
            citation["doi"] = _resolve_paper_id(citation.pop("paper_id"), paper_id_to_doi)
        for criterion in entity.get("evidence_assessments", []):
            for citation in criterion.get("citations", []):
                citation["doi"] = _resolve_paper_id(citation.pop("paper_id"), paper_id_to_doi)

    # quality_concerns[].paper_ids list + citations[]
    for concern in parsed_json.get("quality_concerns", []):
        paper_ids_list = concern.pop("paper_ids", None)
        if paper_ids_list is not None:
            concern["dois"] = [_resolve_paper_id(pid, paper_id_to_doi) for pid in paper_ids_list]
        for citation in concern.get("citations", []):
            citation["doi"] = _resolve_paper_id(citation.pop("paper_id"), paper_id_to_doi)


def _allowed_dispute_statuses(canonical: DisputeStatus) -> set[str]:
    """Allowed LLM-emitted dispute_status values given the GenCC canonical.

    The LLM may uphold the GenCC marker (emit canonical) or overrule a
    flagged candidate by emitting "None" — its rationale must then do the
    overturning work. Escalation (Disputed → Refuted), demotion (Refuted →
    Disputed), or fabrication on an unflagged candidate are warned about
    but not enforced.
    """
    if canonical == "None":
        return {"None"}
    return {canonical, "None"}


def resolve_mondo_names(
    parsed_json: dict[str, Any],
    name_lookup: dict[str, MondoResolution],
    gene_symbol: str,
) -> list[str]:
    """Resolve mondo_disease_name → mondo_id + mondo_label in each disease entity.

    Preserves the LLM's emitted dispute_status (it reflects the LLM's
    decision: uphold = canonical, overrule = "None"). Logs a warning for
    invalid transitions (escalation, demotion, or fabricated dispute) so
    we can monitor compliance without forcing a retry.

    Mutates parsed_json in place. Returns list of unresolved names
    (empty = success).

    Args:
        parsed_json: Parsed LLM output
        name_lookup: Case-insensitive name→MondoResolution lookup
        gene_symbol: Current HGNC symbol for diagnostic logging

    Returns:
        List of disease names that could not be resolved
    """
    unresolved: list[str] = []
    for entity in parsed_json.get("disease_entities", []):
        disease_name = entity.get("mondo_disease_name", "")
        resolution = name_lookup.get(disease_name.lower())
        if resolution is None:
            unresolved.append(disease_name)
            continue

        entity["mondo_id"] = resolution.mondo_id
        entity["mondo_label"] = resolution.mondo_label
        del entity["mondo_disease_name"]

        canonical = resolution.dispute_status
        emitted = entity.get("dispute_status", "None")
        if emitted not in _allowed_dispute_statuses(canonical):
            logger.warning(
                f"{gene_symbol}: invalid dispute_status transition for "
                f"{resolution.mondo_id} ({resolution.mondo_label!r}): "
                f"canonical={canonical!r}, emitted={emitted!r}; preserving emitted"
            )

        if resolution.dispute_records:
            entity["dispute_panels"] = [
                {"submitter": r.submitter, "date": r.date} for r in resolution.dispute_records
            ]
    return unresolved


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
    """Every citation list of an assessment: entities, criteria, quality concerns."""
    lists: list[list[dict[str, Any]]] = []
    for entity in parsed_json["disease_entities"]:
        lists.append(entity["citations"])
        lists += [criterion["citations"] for criterion in entity["evidence_assessments"]]
    lists += [concern["citations"] for concern in parsed_json["quality_concerns"]]
    return lists


def prune_invalid_citations(
    parsed_json: dict[str, Any], quotes_by_doi: dict[str, set[str]]
) -> tuple[int, list[tuple[str, str]]]:
    """Remove citations whose quote isn't one of that paper's extraction quotes.

    Checked after paper_id→doi replacement. Aggregate citations must copy an
    extraction quote exactly, because the report locates quotes by exact lookup
    in ``citation_locations``. Returns the total number of citations and the
    removed (doi, quote) pairs.
    """
    total = 0
    removed: list[tuple[str, str]] = []
    for citations in _citation_lists(parsed_json):
        total += len(citations)
        kept = []
        for citation in citations:
            if citation["quote"] in quotes_by_doi.get(citation["doi"], set()):
                kept.append(citation)
            else:
                removed.append((citation["doi"], citation["quote"]))
        citations[:] = kept
    return total, removed


def fetch_quotes_by_doi(db_path: Path, dois: set[str]) -> dict[str, set[str]]:
    """Every located extraction quote of each paper."""
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

    def count_remaining(self) -> int:
        """Genes with recent evidence that have no assessment yet."""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            (remaining,) = conn.execute(
                """
                SELECT COUNT(DISTINCT hgnc_id) FROM gene_mentions
                WHERE source = 'recent_evidence'
                  AND hgnc_id NOT IN (SELECT hgnc_id FROM gene_assessments)
                """
            ).fetchone()
        return int(remaining)


def store_gene_assessment(
    conn: sqlite3.Connection,
    hgnc_id: int,
    message: Message,
    assessment: dict[str, Any],
    paper_id_to_doi: dict[str, str],
    filtered_papers: list[dict[str, Any]] | None,
    existing_panel_reviews: dict[str, Any] | None,
) -> None:
    """Store one aggregate assessment.

    ``existing_panel_reviews`` holds the PanelApp evaluations for the single target
    panel returned by ``find_gene_panel`` at assess time, shaped
    ``{"panel_id": <int>, "evaluations": [<raw evaluation dicts>]}``; None when the
    gene was not on any target panel.
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO gene_assessments
            (hgnc_id, assessment_raw, assessment_json, paper_id_mapping,
             filtered_papers_json, existing_panel_reviews_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            hgnc_id,
            message.to_json(),
            json.dumps(assessment),
            json.dumps(paper_id_to_doi),
            json.dumps(filtered_papers) if filtered_papers else None,
            json.dumps(existing_panel_reviews) if existing_panel_reviews is not None else None,
        ),
    )


def format_previous_reviews_data(existing_reviews: list[dict[str, Any]]) -> str:
    """Format existing PanelApp reviews as XML-tagged data for the prompt.

    Args:
        existing_reviews: List of evaluation dicts from PanelApp API, ordered most recent first

    Returns:
        Empty string if no reviews, otherwise XML-tagged formatted review data.
    """
    if not existing_reviews:
        return ""

    lines = ["<previous_reviews>"]

    for i, review in enumerate(existing_reviews, 1):
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

    lines.append("</previous_reviews>")
    return "\n".join(lines)


def prepare_aggregate_assessment_prompt(
    hgnc_symbol: str,
    evidence_list: list[dict[str, Any]],
    template_path: Path,
    existing_reviews: list[dict[str, Any]],
    panel_formatted: str,
    mondo_candidates: list[MondoCandidate],
) -> tuple[str, dict[str, str]]:
    """Prepare aggregate assessment prompt using Jinja2 template.

    Generates {LastName}{Year} paper IDs for LLM-friendly citation. The LLM cites
    by paper_id; caller maps back to DOI after receiving the response.

    Args:
        hgnc_symbol: Current HGNC symbol for the gene being assessed
        evidence_list: List of evidence extractions from multiple papers
        template_path: Path to Jinja2 template file
        existing_reviews: List of existing PanelApp reviews (empty for novel genes)
        panel_formatted: Formatted panel description for panel-scoped mode (empty string if not scoping)
        mondo_candidates: MONDO disease candidates for this gene from GenCC

    Returns:
        Tuple of (rendered prompt string, paper_id → DOI mapping)
    """
    # Generate paper IDs and build mappings
    paper_id_to_doi, doi_to_paper_id = generate_paper_ids(evidence_list)

    # Build prompt evidence: replace doi with paper_id, drop fields only needed for ID generation
    prompt_evidence = []
    for evidence in evidence_list:
        prompt_evidence.append(
            {
                "paper_id": doi_to_paper_id[evidence["doi"]],
                "date": evidence["date"],
                "title": evidence["title"],
                "gene_evaluations": evidence["gene_evaluations"],
            }
        )

    # Paper symbols are uppercased; current symbols such as C9orf72 are not.
    aliases = {evidence["paper_gene_symbol"] for evidence in evidence_list} - {hgnc_symbol.upper()}
    if aliases:
        gene_symbol_with_aliases = (
            f"{hgnc_symbol} (also referred to as: {', '.join(sorted(aliases))} in the papers)"
        )
    else:
        gene_symbol_with_aliases = hgnc_symbol

    # Create structured JSON with prompt evidence (paper_id, not doi)
    evidence_extractions = json.dumps(prompt_evidence, indent=2)

    # Format previous reviews data (empty string for novel genes)
    previous_reviews_section = format_previous_reviews_data(existing_reviews)

    # Build dispute blocks: one block per (flagged candidate, disputing panel)
    # pair, with the contributing-paper PMID intersection pre-computed so the
    # LLM does not have to perform the matching itself.
    all_contributing: list[dict[str, Any]] = [
        {
            "paper_id": doi_to_paper_id[e["doi"]],
            "pmid": e.get("pmid"),
            "date": e["date"] or "",
        }
        for e in evidence_list
    ]
    pmid_to_info: dict[int, dict[str, Any]] = {
        info["pmid"]: info for info in all_contributing if info["pmid"] is not None
    }
    dispute_blocks: list[dict[str, Any]] = []
    for c in mondo_candidates:
        if c.dispute_status == "None":
            continue
        for r in c.dispute_records:
            panel_pmid_set = set(r.pmids)
            overlapping = [pmid_to_info[pmid] for pmid in r.pmids if pmid in pmid_to_info]
            independent = [
                info
                for info in all_contributing
                if info["pmid"] is None or info["pmid"] not in panel_pmid_set
            ]
            dispute_blocks.append(
                {
                    "title": c.title,
                    "mondo_id": c.mondo_id,
                    "status": c.dispute_status,
                    "submitter": r.submitter,
                    "date": r.date,
                    "evaluated_pmids": list(r.pmids),
                    "overlapping_contributing_papers": overlapping,
                    "independent_contributing_papers": independent,
                    "rationale": r.rationale,
                }
            )

    # Load and render Jinja2 template
    env = Environment(loader=FileSystemLoader(template_path.parent), autoescape=False)
    template = env.get_template(template_path.name)

    rendered = template.render(
        gene_symbol=gene_symbol_with_aliases,
        evidence_extractions=evidence_extractions,
        has_previous_reviews=bool(existing_reviews),
        previous_reviews_section=previous_reviews_section,
        panel_formatted=panel_formatted,
        mondo_candidates=mondo_candidates,
        dispute_blocks=dispute_blocks,
    )

    return rendered, paper_id_to_doi


@dataclass
class _GeneBatchItem:
    """Per-gene metadata needed for post-LLM validation."""

    hgnc_id: int
    hgnc_symbol: str
    prompt: str
    paper_id_to_doi: dict[str, str]
    evidence_list: list[dict[str, Any]]
    mondo_name_lookup: dict[str, MondoResolution]
    filtered_papers: list[dict[str, Any]] | None
    existing_panel_id: int | None
    existing_reviews: list[dict[str, Any]]


@dataclass(frozen=True)
class _GenePreparation:
    """Shared inputs for preparing gene prompts."""

    db_processor: PaperBatchProcessor
    hgnc_resolver: HgncResolver
    panelapp_client: PanelAppClient
    panel_data: PanelGeneData
    mondo_lookup: MondoLookup
    prompt_path: Path
    panel_formatted: str


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

    existing_panel_id = find_gene_panel(hgnc_id, prep.panel_data.panel_ids, prep.panel_data)
    existing_reviews: list[dict[str, Any]] = []
    if existing_panel_id is not None:
        existing_reviews = prep.panelapp_client.get_gene_evaluations(existing_panel_id, hgnc_id)
        panelapp_pubs = collect_panelapp_gene_publications(
            prep.panelapp_client.get_panel_data(existing_panel_id),
            hgnc_id,
            existing_reviews,
        )
        if evidence_already_in_panelapp(evidence_list, panelapp_pubs):
            evidence_dois = [e["doi"] for e in evidence_list]
            logger.info(
                f"Skipping {hgnc_symbol} — all {len(evidence_list)} paper(s) already reviewed "
                f"in PanelApp panel {existing_panel_id}: {evidence_dois}"
            )
            return None

    mondo_candidates = prep.mondo_lookup.get_candidates(hgnc_symbol)
    prompt, paper_id_to_doi = prepare_aggregate_assessment_prompt(
        hgnc_symbol,
        evidence_list,
        prep.prompt_path,
        existing_reviews,
        prep.panel_formatted,
        mondo_candidates,
    )
    return _GeneBatchItem(
        hgnc_id=hgnc_id,
        hgnc_symbol=hgnc_symbol,
        prompt=prompt,
        paper_id_to_doi=paper_id_to_doi,
        evidence_list=evidence_list,
        mondo_name_lookup=build_mondo_name_lookup(mondo_candidates),
        filtered_papers=filtered_papers or None,
        existing_panel_id=existing_panel_id,
        existing_reviews=existing_reviews,
    )


def genes_to_assess(db_path: Path) -> list[int]:
    """Genes with recent evidence and no assessment, excluding refused ones and ones in flight."""
    with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT gm.hgnc_id FROM gene_mentions gm
            WHERE gm.source = 'recent_evidence'
              AND gm.hgnc_id NOT IN (SELECT hgnc_id FROM gene_assessments)
              AND NOT EXISTS (
                  SELECT 1 FROM llm_requests r
                  WHERE r.stage = ? AND r.subject = CAST(gm.hgnc_id AS TEXT)
                    AND r.status IN ('refused', 'pending')
              )
            ORDER BY gm.hgnc_id
            """,
            (STAGE,),
        ).fetchall()
    return [row[0] for row in rows]


def build_request(item: _GeneBatchItem, output_config: OutputConfigParam) -> LlmRequest:
    return LlmRequest(
        subject=str(item.hgnc_id),
        params={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": item.prompt}],
            "output_config": output_config,
        },
    )


def assessment_problems(
    assessment: dict[str, Any], item: _GeneBatchItem, db_path: Path
) -> list[str]:
    """Problems that send a gene back for another attempt. Maps paper IDs to DOIs in place."""
    try:
        replace_paper_ids_with_dois(assessment, item.paper_id_to_doi)
    except ValueError as e:
        return [f"hallucinated paper ID: {e}"]
    problems: list[str] = []
    quotes_by_doi = fetch_quotes_by_doi(db_path, {e["doi"] for e in item.evidence_list})
    total, removed = prune_invalid_citations(assessment, quotes_by_doi)
    if total and len(removed) > MAX_INVALID_CITATION_SHARE * total:
        problems.append(
            f"{len(removed)} of {total} citation quotes not copied from the extractions"
        )
    elif removed:
        logger.info(
            "%s: dropped %d citations whose quote isn't in the paper's extraction: %s",
            item.hgnc_symbol,
            len(removed),
            "; ".join(repr(quote[:60]) for _, quote in removed[:3]),
        )
    if unresolved := resolve_mondo_names(assessment, item.mondo_name_lookup, item.hgnc_symbol):
        problems.append(f"unresolved MONDO disease names: {unresolved}")
    if not validate_entities_criteria_complete(assessment["disease_entities"]):
        problems.append("incomplete per-entity criteria")
    if not validate_independent_family_counts(assessment["disease_entities"]):
        problems.append("inconsistent independent_family_count")
    return problems


@dataclass
class _Outcome:
    stored: int = 0
    refused: int = 0
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
        if result.status == ResultStatus.REFUSED:
            assert message is not None
            outcome.refused += 1
            logger.warning(
                "Refused: HGNC:%s (%s)",
                result.subject,
                message.stop_details.category if message.stop_details else "no category",
            )
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
                problems = assessment_problems(assessment, item, db_path)
            except (ValueError, jsonschema.ValidationError) as e:
                problems = [str(e)]
            if problems:
                logger.warning(
                    "Rejected assessment for %s: %s", item.hgnc_symbol, "; ".join(problems[:5])
                )
                outcome.failed += 1
            else:
                existing_panel_reviews = (
                    {"panel_id": item.existing_panel_id, "evaluations": item.existing_reviews}
                    if item.existing_panel_id is not None
                    else None
                )
                with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
                    record_result(conn, result)
                    store_gene_assessment(
                        conn,
                        item.hgnc_id,
                        message,
                        assessment,
                        item.paper_id_to_doi,
                        item.filtered_papers,
                        existing_panel_reviews,
                    )
                stored = True
                outcome.stored += 1
        if not stored:
            with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
                record_result(conn, result)
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
    limit: int | None,
    max_retries: int,
) -> None:
    validator = jsonschema.Draft202012Validator(schema)
    output_config = json_output_config(schema, EFFORT)
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

    for attempt in range(1, max_retries + 1):
        hgnc_ids = [g for g in genes_to_assess(db_path) if g not in skipped]
        if limit is not None:
            hgnc_ids = hgnc_ids[:limit]
        batch = prepared(hgnc_ids)
        if not batch:
            logger.info("No genes left to assess")
            return
        logger.info("Attempt %d: assessing %d genes", attempt, len(batch))
        results = await transport.run(
            STAGE, 1, [build_request(item, output_config) for item in batch]
        )
        outcome = handle_results(results, items, db_path, validator)
        logger.info(
            "Attempt %d: stored %d, refused %d, failed %d",
            attempt,
            outcome.stored,
            outcome.refused,
            outcome.failed,
        )
        if outcome.stored == 0:
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
        help="Maximum number of attempts for genes whose assessment failed or was rejected",
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
        help="Panel ID for panel-scoped assessment. When set, the summary must explain why the gene is relevant to this panel's scope.",
    ),
    immediate: bool = typer.Option(
        False,
        "--immediate",
        help="Send requests immediately instead of as Message Batches (for prompt development)",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Assess at most this many genes, in one attempt (for prompt development)",
    ),
) -> None:
    """Perform aggregate assessment of genes using evidence from multiple papers."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    schema: dict[str, Any] = json.loads(schema_path.read_text())
    mondo_lookup = MondoLookup(cache_dir=db_path.parent)
    hgnc_resolver = HgncResolver.from_file()

    logger.info(f"Fetching PanelApp gene data for {panel_date}...")
    panelapp_client = PanelAppClient(panel_date)
    panel_data = panelapp_client.get_target_panels_genes(target_panel_ids)
    logger.info(
        f"  Loaded {len(panel_data.gene_confidence)} genes from {len(panel_data.panel_ids)} target panels"
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
    logger.info(f"{db_processor.count_remaining():,} genes with recent evidence and no assessment")
    prep = _GenePreparation(
        db_processor=db_processor,
        hgnc_resolver=hgnc_resolver,
        panelapp_client=panelapp_client,
        panel_data=panel_data,
        mondo_lookup=mondo_lookup,
        prompt_path=prompt_path,
        panel_formatted=panel_formatted,
    )

    async def run() -> None:
        client = make_client(AnthropicSettings())
        transport: Transport = (
            ImmediateTransport(client) if immediate else BatchTransport(client, db_path)
        )
        await _process_assessments(
            transport=transport,
            db_path=db_path,
            prep=prep,
            schema=schema,
            limit=limit,
            max_retries=max_retries,
        )

    asyncio.run(run())
    logger.info(f"{db_processor.count_remaining():,} genes still without an assessment")
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
