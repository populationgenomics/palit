#!/usr/bin/env python3
"""Aggregate evidence assessment across papers for each gene."""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from palit.hgnc import HgncResolver
from palit.llm import LLMProcessor, create_llm_processor
from palit.mondo_lookup import MondoCandidate, MondoLookup
from palit.panelapp_client import PanelAppClient, PanelGeneData, format_panel_for_prompt
from palit.panelapp_integration import MONDO_CATEGORIES, validate_criteria_complete
from palit.papers import MIN_PREPRINT_FAMILIES, generate_paper_ids, is_preprint

app = typer.Typer(help="Aggregate evidence assessment across papers for each gene")
logger = logging.getLogger(__name__)

# Case-insensitive fallback name→ID lookup (static, built once)
_FALLBACK_NAME_TO_ID = {
    info["label"].lower(): mondo_id for mondo_id, info in MONDO_CATEGORIES.items()
}


def build_mondo_name_to_id(candidates: list[MondoCandidate]) -> dict[str, str]:
    """Build case-insensitive name→MONDO ID lookup from candidates + fallback categories.

    Args:
        candidates: Gene-specific MONDO candidates from GenCC

    Returns:
        Dict mapping lowercased disease name to MONDO ID
    """
    name_to_id = dict(_FALLBACK_NAME_TO_ID)  # copy fallbacks
    for c in candidates:
        name_to_id[c.title.lower()] = c.mondo_id
    return name_to_id


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
    # disease_entities[].citations[]
    for entity in parsed_json.get("disease_entities", []):
        for citation in entity.get("citations", []):
            citation["doi"] = _resolve_paper_id(citation.pop("paper_id"), paper_id_to_doi)

    # evidence_assessments[].citations
    for criterion in parsed_json.get("evidence_assessments", []):
        for citation in criterion.get("citations", []):
            citation["doi"] = _resolve_paper_id(citation.pop("paper_id"), paper_id_to_doi)

    # quality_concerns[].paper_ids list + citations[]
    for concern in parsed_json.get("quality_concerns", []):
        paper_ids_list = concern.pop("paper_ids", None)
        if paper_ids_list is not None:
            concern["dois"] = [_resolve_paper_id(pid, paper_id_to_doi) for pid in paper_ids_list]
        for citation in concern.get("citations", []):
            citation["doi"] = _resolve_paper_id(citation.pop("paper_id"), paper_id_to_doi)


def resolve_mondo_names(parsed_json: dict[str, Any], name_to_id: dict[str, str]) -> list[str]:
    """Resolve mondo_disease_name → mondo_id + mondo_label in each disease entity.

    Mutates parsed_json in place. Returns list of unresolved names (empty = success).

    Args:
        parsed_json: Parsed LLM output
        name_to_id: Case-insensitive name→MONDO ID lookup

    Returns:
        List of disease names that could not be resolved (empty if all resolved)
    """
    unresolved: list[str] = []
    for entity in parsed_json.get("disease_entities", []):
        disease_name = entity.get("mondo_disease_name", "")
        mondo_id = name_to_id.get(disease_name.lower())
        if mondo_id is None:
            unresolved.append(disease_name)
        else:
            entity["mondo_id"] = mondo_id
            entity["mondo_label"] = disease_name
            del entity["mondo_disease_name"]
    return unresolved


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
    gene_panels = panel_data.gene_panel_mapping.get(hgnc_id, set())
    for panel_id in target_panel_ids:
        if panel_id in gene_panels:
            return panel_id
    return None


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


def validate_box_ids_with_doi(
    parsed_json: dict[str, Any], valid_box_ids_by_doi: dict[str, set[int]]
) -> bool:
    """Check all (doi, box_id) pairs in citation structures are valid.

    Args:
        parsed_json: Parsed LLM output (after paper_id→doi replacement)
        valid_box_ids_by_doi: Map from DOI to set of valid box IDs for that paper

    Returns:
        True if all pairs are valid, False otherwise
    """

    def check_citation(citation: dict[str, Any]) -> bool:
        doi = citation.get("doi")
        box_id = citation.get("box_id")
        if isinstance(doi, str) and isinstance(box_id, int):
            valid_box_ids = valid_box_ids_by_doi.get(doi)
            if valid_box_ids is None or box_id not in valid_box_ids:
                return False
        return True

    # disease_entities[].citations[]
    for entity in parsed_json.get("disease_entities", []):
        for citation in entity.get("citations", []):
            if not check_citation(citation):
                return False

    # evidence_assessments[].citations
    for criterion in parsed_json.get("evidence_assessments", []):
        for citation in criterion.get("citations", []):
            if not check_citation(citation):
                return False

    # quality_concerns[].citations[]
    for concern in parsed_json.get("quality_concerns", []):
        for citation in concern.get("citations", []):
            if not check_citation(citation):
                return False

    return True


def fetch_valid_box_ids_by_doi(
    db_path: Path, evidence_list: list[dict[str, Any]]
) -> dict[str, set[int]]:
    """Query database to get valid box IDs for each paper in evidence_list.

    Args:
        db_path: Path to SQLite database
        evidence_list: List of evidence dicts containing DOIs

    Returns:
        Map from DOI to set of valid box IDs for that paper
    """
    dois = {evidence["doi"] for evidence in evidence_list}

    if not dois:
        return {}

    valid_box_ids_by_doi: dict[str, set[int]] = {}

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        for doi in dois:
            cursor.execute("SELECT bbox_mapping FROM papers WHERE doi = ?", (doi,))
            row = cursor.fetchone()

            if row and row[0]:
                try:
                    bbox_mapping = json.loads(row[0])
                    valid_box_ids_by_doi[doi] = {int(box_id) for box_id in bbox_mapping.keys()}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Error parsing bbox_mapping for DOI {doi}: {e}")
                    continue

    return valid_box_ids_by_doi


class PaperBatchProcessor:
    """Handle database operations for aggregate gene assessment."""

    def __init__(self, db_path: Path):
        """Initialize with database path."""
        self.db_path = db_path

    def get_evidence_for_gene(self, hgnc_id: int) -> list[dict[str, Any]]:
        """Get all evidence extractions for a specific gene.

        Args:
            hgnc_id: HGNC ID of the gene to search for

        Returns:
            List of dicts with doi, date, title, authors, paper_gene_symbol, gene_evaluations
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT DISTINCT p.doi, p.pmid, p.journal, p.source_date, p.title, p.authors,
                       p.evidence_extraction_json, gm.paper_gene_symbol
                FROM papers p
                JOIN gene_mentions gm ON p.doi = gm.paper_doi
                WHERE gm.hgnc_id = ?
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

    def update_gene_assessment(
        self,
        hgnc_id: int,
        assessment_data: tuple[str, dict[str, Any]],
        paper_id_to_doi: dict[str, str],
        filtered_papers: list[dict[str, str]] | None = None,
    ) -> None:
        """Store aggregate assessment result in gene_assessments table.

        Args:
            hgnc_id: HGNC ID of the gene
            assessment_data: Tuple of (raw_response, parsed_json)
            paper_id_to_doi: Mapping of AuthorYear paper IDs to DOIs used for this assessment
            filtered_papers: Papers excluded from assessment [{doi, reason}]
        """
        raw_response, json_data = assessment_data
        filtered_json = json.dumps(filtered_papers) if filtered_papers else None

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO gene_assessments
                    (hgnc_id, assessment_raw, assessment_json, paper_id_mapping,
                     filtered_papers_json)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        hgnc_id,
                        raw_response,
                        json.dumps(json_data),
                        json.dumps(paper_id_to_doi),
                        filtered_json,
                    ),
                )

                conn.commit()
                logger.info(f"Stored aggregate assessment for HGNC:{hgnc_id}")

            except sqlite3.Error as e:
                logger.error(f"Error storing aggregate assessment for HGNC:{hgnc_id}: {e}")

    def get_aggregate_assessment_statistics(self) -> dict[str, int]:
        """Get statistics about aggregate assessment progress."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total unique genes in the working set (recent_evidence)
            cursor.execute(
                """
                SELECT COUNT(DISTINCT hgnc_id)
                FROM gene_mentions
                WHERE source = 'recent_evidence'
            """
            )
            genes_with_evidence = cursor.fetchone()[0]

            # Already assessed genes
            cursor.execute("SELECT COUNT(*) FROM gene_assessments")
            assessed_genes = cursor.fetchone()[0]

            return {
                "genes_with_evidence": genes_with_evidence,
                "assessed_genes": assessed_genes,
                "remaining_to_assess": genes_with_evidence - assessed_genes,
            }


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

    # Extract unique paper gene symbols (aliases) from evidence
    paper_symbols = set()
    for evidence in evidence_list:
        if "paper_gene_symbol" in evidence:
            paper_symbols.add(evidence["paper_gene_symbol"])

    # Format gene symbol with aliases if they differ from current HGNC symbol
    aliases = paper_symbols - {hgnc_symbol}
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
    mondo_name_to_id: dict[str, str]
    filtered_papers: list[dict[str, Any]] | None


async def _process_assessments(
    *,
    llm_processor: LLMProcessor,
    db_processor: PaperBatchProcessor,
    db_path: Path,
    hgnc_resolver: HgncResolver,
    panelapp_client: PanelAppClient,
    panel_data: PanelGeneData,
    mondo_lookup: MondoLookup,
    schema: dict[str, Any],
    prompt_path: Path,
    panel_formatted: str,
    batch_size: int,
    max_retries: int,
    initial_remaining: int,
) -> None:
    """Run the aggregate assessment retry loop."""
    total_processed = 0
    genes_without_evidence: set[int] = set()
    consecutive_failures = 0
    retry_attempt = 0

    with tqdm(total=initial_remaining, desc="Processing genes") as pbar:
        while retry_attempt < max_retries:
            stats = db_processor.get_aggregate_assessment_statistics()
            if stats["remaining_to_assess"] == 0:
                logger.info("All genes successfully assessed!")
                break

            if retry_attempt > 0:
                logger.info(
                    f"Retry attempt {retry_attempt} - {stats['remaining_to_assess']} genes remaining"
                )

            logger.info("Fetching genes with evidence...")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT DISTINCT hgnc_id
                    FROM gene_mentions
                    WHERE source = 'recent_evidence'
                    ORDER BY hgnc_id
                """
                )
                gene_hgnc_ids = [row[0] for row in cursor.fetchall()]

                cursor.execute(
                    """
                    SELECT hgnc_id
                    FROM gene_assessments
                """
                )
                already_assessed = {row[0] for row in cursor.fetchall()}

            hgnc_ids_to_assess = [g for g in gene_hgnc_ids if g not in already_assessed]
            logger.info(f"Found {len(hgnc_ids_to_assess)} genes to assess")

            if not hgnc_ids_to_assess:
                logger.info("No genes need assessment!")
                break

            pass_processed = 0
            failed_genes: list[int] = []

            # Prepare all gene prompts, then process in batches
            batch_items: list[_GeneBatchItem] = []
            for hgnc_id in hgnc_ids_to_assess:
                hgnc_symbol = hgnc_resolver.get_symbol(hgnc_id)
                logger.info(f"Preparing {hgnc_symbol} (HGNC:{hgnc_id})")

                evidence_list = db_processor.get_evidence_for_gene(hgnc_id)

                if not evidence_list:
                    logger.warning(f"No evidence found for {hgnc_symbol}")
                    genes_without_evidence.add(hgnc_id)
                    continue

                evidence_list, filtered_papers = filter_preprint_evidence(evidence_list)
                if filtered_papers:
                    filtered_dois = [fp["doi"] for fp in filtered_papers]
                    logger.info(
                        f"  Filtered {len(filtered_papers)} preprint(s) for {hgnc_symbol}: "
                        f"{filtered_dois}"
                    )
                if not evidence_list:
                    logger.info(
                        f"Skipping {hgnc_symbol} — all papers filtered by preprint family gate"
                    )
                    genes_without_evidence.add(hgnc_id)
                    continue

                existing_panel_id = find_gene_panel(hgnc_id, panel_data.panel_ids, panel_data)

                existing_reviews: list[dict[str, Any]] = []
                if existing_panel_id is not None:
                    existing_reviews = panelapp_client.get_gene_evaluations(
                        existing_panel_id, hgnc_id
                    )
                    logger.info(
                        f"  Found {len(existing_reviews)} existing reviews in panel {existing_panel_id}"
                    )

                mondo_candidates = mondo_lookup.get_candidates(hgnc_symbol)
                mondo_name_to_id = build_mondo_name_to_id(mondo_candidates)
                if mondo_candidates:
                    logger.info(f"  {len(mondo_candidates)} MONDO candidates for {hgnc_symbol}")

                prompt, paper_id_to_doi = prepare_aggregate_assessment_prompt(
                    hgnc_symbol,
                    evidence_list,
                    prompt_path,
                    existing_reviews,
                    panel_formatted,
                    mondo_candidates,
                )

                batch_items.append(
                    _GeneBatchItem(
                        hgnc_id=hgnc_id,
                        hgnc_symbol=hgnc_symbol,
                        prompt=prompt,
                        paper_id_to_doi=paper_id_to_doi,
                        evidence_list=evidence_list,
                        mondo_name_to_id=mondo_name_to_id,
                        filtered_papers=filtered_papers or None,
                    )
                )

            # Process prepared genes in batches
            for i in range(0, len(batch_items), batch_size):
                batch = batch_items[i : i + batch_size]
                logger.info(
                    f"Processing batch of {len(batch)} genes "
                    f"({i + 1}-{i + len(batch)}/{len(batch_items)})"
                )

                prompts = [item.prompt for item in batch]
                results = await llm_processor.process_batch(prompts, schema)

                for item, result in zip(batch, results, strict=True):
                    if result is None:
                        logger.warning(
                            f"Failed to process aggregate assessment for {item.hgnc_symbol}"
                        )
                        failed_genes.append(item.hgnc_id)
                        continue

                    try:
                        replace_paper_ids_with_dois(result.parsed_json, item.paper_id_to_doi)
                    except ValueError:
                        logger.warning(
                            f"LLM hallucinated paper ID for {item.hgnc_symbol}, retrying"
                        )
                        failed_genes.append(item.hgnc_id)
                        continue
                    if not validate_box_ids_with_doi(
                        result.parsed_json,
                        fetch_valid_box_ids_by_doi(db_path, item.evidence_list),
                    ):
                        logger.warning(f"Invalid (doi, box_id) pairs for {item.hgnc_symbol}")
                        failed_genes.append(item.hgnc_id)
                        continue
                    if unresolved := resolve_mondo_names(result.parsed_json, item.mondo_name_to_id):
                        logger.warning(
                            f"Unresolved MONDO disease names for {item.hgnc_symbol}: {unresolved}"
                        )
                        failed_genes.append(item.hgnc_id)
                        continue
                    if not validate_criteria_complete(
                        result.parsed_json.get("evidence_assessments", [])
                    ):
                        logger.warning(
                            f"Incomplete criteria for {item.hgnc_symbol} "
                            f"(expected criterion_A-E, got {[c.get('name') for c in result.parsed_json.get('evidence_assessments', [])]})"
                        )
                        failed_genes.append(item.hgnc_id)
                        continue
                    db_processor.update_gene_assessment(
                        item.hgnc_id,
                        (result.raw_response, result.parsed_json),
                        item.paper_id_to_doi,
                        filtered_papers=item.filtered_papers,
                    )
                    pass_processed += 1
                    pbar.update(1)

            total_processed += pass_processed

            if pass_processed == 0:
                consecutive_failures += 1
                logger.warning(f"No progress made in retry attempt {retry_attempt}")
                if consecutive_failures >= 2:
                    logger.error("Multiple consecutive attempts with no progress - stopping")
                    break
            else:
                consecutive_failures = 0

            retry_attempt += 1

    final_stats = db_processor.get_aggregate_assessment_statistics()
    logger.info("Aggregate assessment complete!")
    logger.info("Final statistics:")
    logger.info(f"  Successfully processed: {total_processed:,}")
    logger.info(f"  Genes without evidence: {len(genes_without_evidence):,}")
    logger.info(f"  Still remaining: {final_stats['remaining_to_assess']:,}")

    if genes_without_evidence:
        no_evidence_list = list(genes_without_evidence)
        logger.info(
            f"  Genes without evidence: {no_evidence_list[:10]}..."
            if len(no_evidence_list) > 10
            else f"  Genes without evidence: {no_evidence_list}"
        )

    if final_stats["remaining_to_assess"] > 0:
        logger.warning(
            f"Failed to assess {final_stats['remaining_to_assess']} genes after {max_retries} attempts"
        )


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database",
    ),
    model: str = typer.Option(
        "openai/gpt-oss-120b",
        "--model",
        "-m",
        help="Model name for vLLM",
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        "-t",
        help="Sampling temperature",
    ),
    max_tokens: int = typer.Option(
        80000,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    tensor_parallel_size: int = typer.Option(
        1,
        "--tensor-parallel-size",
        help="Tensor parallelism size",
    ),
    max_model_len: int = typer.Option(
        131072,
        "--max-model-len",
        help="Maximum model context length",
    ),
    llm_config: str = typer.Option(
        "",
        "--llm-config",
        help="JSON dict of extra backend config (forwarded to LLM processor)",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
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
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Genes per LLM batch (increase for concurrent API backends like Bedrock)",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of retry attempts for failed genes",
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
) -> None:
    """Perform aggregate assessment of genes using evidence from multiple papers."""
    # Validate inputs
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if not prompt_path.exists():
        logger.error(f"Prompt template not found: {prompt_path}")
        raise typer.Exit(1)

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        raise typer.Exit(1)

    # Load schema (template is loaded via Jinja2 in prepare_aggregate_assessment_prompt)
    logger.info("Loading schema...")
    schema: dict[str, Any] = json.loads(schema_path.read_text())
    logger.info(f"  Loaded schema from {schema_path}")

    # Initialize MONDO lookup (downloads GenCC + MONDO if stale)
    logger.info("Initializing MONDO lookup...")
    mondo_lookup = MondoLookup(cache_dir=db_path.parent)

    # Load HGNC resolver for gene symbol lookup
    hgnc_resolver = HgncResolver.from_file()
    logger.info(f"  Loaded HgncResolver with {len(hgnc_resolver._by_symbol)} genes")

    # Initialize PanelApp client and fetch panel data
    logger.info(f"Fetching PanelApp gene data for {panel_date}...")
    panelapp_client = PanelAppClient(panel_date)
    panel_data = panelapp_client.get_target_panels_genes(target_panel_ids)
    logger.info(
        f"  Loaded {len(panel_data.gene_confidence)} genes from {len(panel_data.panel_ids)} target panels"
    )

    # Build panel_formatted for template (empty string if not scoping to a panel)
    if scope_panel_id is not None:
        logger.info(f"Fetching description for scope panel {scope_panel_id}...")
        try:
            panel_info = panelapp_client.get_panel_data(scope_panel_id)
        except ValueError as e:
            logger.error(f"Panel {scope_panel_id} not found in PanelApp data for {panel_date}")
            raise typer.Exit(1) from e

        panel_formatted = format_panel_for_prompt(scope_panel_id, panel_info)
        logger.info(f"  Panel-scoped mode: {panel_info.get('name', 'Unknown')}")
    else:
        panel_formatted = ""

    # Initialize components
    logger.info("Initializing database processor...")
    db_processor = PaperBatchProcessor(db_path)
    logger.info(f"  Connected to database at {db_path}")

    logger.info("Initializing LLM processor...")
    llm_processor = create_llm_processor(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        **(json.loads(llm_config) if llm_config else {}),
    )

    # Get initial statistics
    logger.info("Fetching aggregate assessment statistics...")
    stats = db_processor.get_aggregate_assessment_statistics()
    logger.info("Aggregate assessment statistics:")
    logger.info(f"  Genes with evidence: {stats['genes_with_evidence']:,}")
    logger.info(f"  Already assessed: {stats['assessed_genes']:,}")
    logger.info(f"  Remaining to assess: {stats['remaining_to_assess']:,}")

    if stats["remaining_to_assess"] == 0:
        logger.info("No genes remaining to assess!")
        return

    asyncio.run(
        _process_assessments(
            llm_processor=llm_processor,
            db_processor=db_processor,
            db_path=db_path,
            hgnc_resolver=hgnc_resolver,
            panelapp_client=panelapp_client,
            panel_data=panel_data,
            mondo_lookup=mondo_lookup,
            schema=schema,
            prompt_path=prompt_path,
            panel_formatted=panel_formatted,
            batch_size=batch_size,
            max_retries=max_retries,
            initial_remaining=stats["remaining_to_assess"],
        )
    )


if __name__ == "__main__":
    app()
