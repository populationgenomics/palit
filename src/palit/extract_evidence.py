#!/usr/bin/env python3
"""Extract evidence from full-text papers using vLLM inference."""

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

from palit.docling import serialize_with_bbox_ids
from palit.entities import (
    DiseaseEntity,
    entities_by_gene,
    entity_ref,
    format_entity_block,
    load_entities,
    load_entities_by_doi,
)
from palit.hgnc import HgncEntry, HgncResolver
from palit.llm import LLMProcessor, PromptResult, create_llm_processor
from palit.panelapp_client import (
    PanelAppClient,
    format_panel_for_prompt,
)
from palit.panelapp_integration import (
    decompose_moi,
    validate_entities_criteria_complete,
    validate_independent_family_counts,
)
from palit.papers import doi_to_path

app = typer.Typer(help="Extract structured evidence from full-text papers using vLLM")
logger = logging.getLogger(__name__)

# SQLite busy timeout in seconds. When sharding, multiple processes write to
# the same database; the default 5 s can be too short for large batch commits.
DB_TIMEOUT_SECONDS = 60

# Disaggregated runs only process papers whose relevance assessment named at
# least one gene that has fixed disease associations. The work query and the
# statistics query the retry loop exits on must carry this predicate identically:
# a paper counted as "remaining" but never fetched makes the loop abort with
# "no progress".
DISAGGREGATED_PAPER_FILTER = """
                AND EXISTS (
                    SELECT 1 FROM gene_mentions gm
                    JOIN gene_disease_entities e ON e.hgnc_id = gm.hgnc_id
                    WHERE gm.paper_doi = papers.doi
                    AND gm.source = 'relevance_assessment'
                )"""


@dataclass
class PaperPrompt:
    """A paper prepared for processing with its metadata."""

    prompt: str
    doi: str
    bbox_mapping: dict[int, dict[str, Any]]


@dataclass
class DeepAnalysisPreparation:
    """Result from preparing deep analysis prompts."""

    paper_prompts: list[PaperPrompt]
    missing_dois: list[str]


@dataclass(frozen=True)
class DisaggregatedContext:
    """The fixed disease associations a disaggregated extraction run assigns to.

    `entities_by_doi` holds, per paper, the associations of the genes that
    paper's relevance assessment named; `prompt_blocks` holds the rendered
    FIXED DISEASE ASSOCIATIONS block for those same genes. Both are keyed by
    DOI so a paper's prompt only ever lists its own genes.
    """

    entities_by_doi: dict[str, dict[int, list[DiseaseEntity]]]
    prompt_blocks: dict[str, str]


@dataclass
class EntityAssignmentResult:
    """Outcome of checking one paper's entity assignments.

    `errors` are model mistakes the paper is retried for; `off_target_symbols`
    name gene evaluations for genes this paper was not mapped to, which are
    dropped from the extraction instead.
    """

    errors: list[str]
    off_target_symbols: list[str]


def schema_expects_entity_refs(schema: dict[str, Any]) -> bool:
    """Whether a response schema is the disaggregated one, i.e. requires entity_ref."""
    disease_entities = schema["properties"]["gene_evaluations"]["items"]["properties"][
        "disease_entities"
    ]
    return "entity_ref" in disease_entities["items"]["required"]


def build_disaggregated_context(db_path: Path, resolver: HgncResolver) -> DisaggregatedContext:
    """Load the fixed associations and render one prompt block per paper."""
    entities = load_entities(db_path)
    by_gene = entities_by_gene(entities)
    logger.info(f"  Loaded {len(entities)} fixed disease associations over {len(by_gene)} genes")

    entities_by_doi = load_entities_by_doi(db_path)
    prompt_blocks = {
        doi: format_entity_block(paper_genes, resolver)
        for doi, paper_genes in entities_by_doi.items()
    }
    logger.info(f"  {len(entities_by_doi)} papers mapped to at least one association")
    return DisaggregatedContext(entities_by_doi=entities_by_doi, prompt_blocks=prompt_blocks)


def validate_entity_assignments(
    parsed_json: dict[str, Any],
    entities_for_paper: dict[int, list[DiseaseEntity]],
    resolver: HgncResolver,
) -> EntityAssignmentResult:
    """Check each disease entity block against its gene's fixed associations.

    A block references an association by `entity_ref`, or carries null when the
    model found no fitting association. A reference must belong to the gene it
    was emitted under and may be used at most once per gene evaluation. An
    observed inheritance mode that contradicts the association's own mode is a
    signal worth surfacing, not an error: the paper is reporting something the
    curation source does not list.
    """
    errors: list[str] = []
    off_target_symbols: list[str] = []

    for gene_eval in parsed_json["gene_evaluations"]:
        symbol: str = gene_eval["gene_symbol"]
        entry = resolver.resolve(symbol)
        if entry is None or entry.hgnc_id not in entities_for_paper:
            off_target_symbols.append(symbol)
            continue

        by_ref = {entity_ref(entity): entity for entity in entities_for_paper[entry.hgnc_id]}
        seen: set[str] = set()
        for block in gene_eval["disease_entities"]:
            ref: str | None = block["entity_ref"]
            if ref is None:
                continue

            entity = by_ref.get(ref)
            if entity is None:
                errors.append(
                    f"{symbol}: entity_ref '{ref}' is not one of this gene's fixed "
                    f"associations {sorted(by_ref)}"
                )
                continue
            if ref in seen:
                errors.append(f"{symbol}: entity_ref '{ref}' used by more than one disease entity")
                continue
            seen.add(ref)

            inheritance_mode: str = block["inheritance_mode"]
            if inheritance_mode not in ("NR", "Other") and inheritance_mode not in decompose_moi(
                entity.moi
            ):
                logger.warning(
                    f"{symbol}: observed inheritance '{inheritance_mode}' differs from the "
                    f"'{entity.moi}' inheritance of association '{ref}'"
                )

    return EntityAssignmentResult(errors=errors, off_target_symbols=off_target_symbols)


def drop_off_target_evaluations(parsed_json: dict[str, Any], off_target_symbols: list[str]) -> None:
    """Remove gene evaluations for genes outside this paper's fixed associations."""
    off_target = set(off_target_symbols)
    parsed_json["gene_evaluations"] = [
        gene_eval
        for gene_eval in parsed_json["gene_evaluations"]
        if gene_eval["gene_symbol"] not in off_target
    ]


def annotate_entity_ids(
    parsed_json: dict[str, Any],
    entities_for_paper: dict[int, list[DiseaseEntity]],
    resolver: HgncResolver,
) -> None:
    """Attach the fixed-association row id each disease entity block was assigned to.

    Resolution happens on the references as validated, before
    `normalize_extraction_genes` rewrites gene symbols across the serialized
    JSON. The ids it writes are integers, so that rewrite cannot corrupt them.
    """
    for gene_eval in parsed_json["gene_evaluations"]:
        symbol: str = gene_eval["gene_symbol"]
        entry = resolver.resolve(symbol)
        if entry is None or entry.hgnc_id not in entities_for_paper:
            raise ValueError(
                f"{symbol}: off-target gene evaluations must be dropped before entity ids "
                f"are resolved"
            )

        ids_by_ref = {entity_ref(e): e.id for e in entities_for_paper[entry.hgnc_id]}
        for block in gene_eval["disease_entities"]:
            ref: str | None = block["entity_ref"]
            block["entity_id"] = None if ref is None else ids_by_ref[ref]


def validate_box_ids(data: Any, valid_box_ids: set[int]) -> bool:
    """
    Recursively check all 'box_id' fields in data structure are valid.
    Returns False immediately on first invalid box_id (fail-fast).

    Args:
        data: JSON data structure (dict, list, or primitive)
        valid_box_ids: Set of valid box IDs for this paper

    Returns:
        True if all box_ids are valid, False otherwise
    """

    def recurse(obj: Any) -> bool:
        if isinstance(obj, dict):
            # Check if this dict has a 'box_id' field
            if "box_id" in obj:
                box_id = obj["box_id"]
                if isinstance(box_id, int) and box_id not in valid_box_ids:
                    return False
            # Recurse into all values
            for value in obj.values():
                if not recurse(value):
                    return False
        elif isinstance(obj, list):
            # Recurse into all list items
            for item in obj:
                if not recurse(item):
                    return False
        # Primitives: return True
        return True

    return recurse(data)


def normalize_extraction_genes(
    parsed_json: dict[str, Any],
    hgnc_resolver: HgncResolver,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize gene symbols in extraction JSON to current HGNC.

    For each unique gene_symbol in gene_evaluations:
    - Resolve via HgncResolver
    - If resolved and symbol changed: replace old symbol with current symbol
      throughout the entire JSON (summary, variant descriptions, etc.)
    - Resolved: add hgnc_id
    - Always: rename gene_symbol → paper_gene_symbol (uppercased)

    Returns:
        (normalized_json, unresolved_symbols)
    """
    # Resolve unique gene symbols
    resolved: dict[str, HgncEntry] = {}  # uppercased raw symbol → entry
    unresolved: set[str] = set()
    for gene_eval in parsed_json.get("gene_evaluations", []):
        upper: str = gene_eval["gene_symbol"].upper()
        if upper not in resolved and upper not in unresolved:
            entry = hgnc_resolver.resolve(upper)
            if entry is not None:
                resolved[upper] = entry
            else:
                unresolved.add(upper)

    # Full string replacement across serialized JSON for changed symbols
    replacements = {old: entry.symbol for old, entry in resolved.items() if old != entry.symbol}
    if replacements:
        json_str = json.dumps(parsed_json)
        for old in sorted(replacements, key=len, reverse=True):
            json_str = json_str.replace(old, replacements[old])
        parsed_json = json.loads(json_str)

    # Rewrite gene_evaluations: add hgnc_id if resolved, always rename gene_symbol.
    # Gene evaluations without hgnc_id are ignored by all downstream pipeline stages
    # (aggregation, annotation, reporting) — they exist only for diagnostic inspection.
    by_current_symbol: dict[str, int] = {entry.symbol: entry.hgnc_id for entry in resolved.values()}
    for gene_eval in parsed_json.get("gene_evaluations", []):
        gene_symbol: str = gene_eval.pop("gene_symbol")
        gene_eval["paper_gene_symbol"] = gene_symbol.upper()
        hgnc_id = by_current_symbol.get(gene_symbol)
        if hgnc_id is not None:
            gene_eval["hgnc_id"] = hgnc_id

    return parsed_json, sorted(unresolved)


class PaperBatchProcessor:
    """Handle database operations for evidence extraction."""

    def __init__(self, db_path: Path):
        """Initialize with database path."""
        self.db_path = db_path

    def get_papers_for_deep_analysis(
        self, shard_index: int, num_shards: int, disaggregated: bool
    ) -> list[dict[str, Any]]:
        """
        Get papers that have been downloaded and haven't been processed for evidence extraction.

        Args:
            shard_index: Shard index (0-based) for parallel processing
            num_shards: Total number of shards
            disaggregated: Restrict to papers mapped to at least one fixed association

        Returns list of papers with doi, source_date, title, abstract.
        """
        entity_filter = DISAGGREGATED_PAPER_FILTER if disaggregated else ""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT doi, source_date, title, abstract
                FROM papers
                WHERE download_status = 'downloaded'
                AND evidence_extraction_json IS NULL
                AND rowid % ? = ?{entity_filter}
                ORDER BY doi
                """,
                (num_shards, shard_index),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_paper_evidence_extraction(
        self,
        paper_prompts: list[PaperPrompt],
        results: list[PromptResult | None],
        hgnc_resolver: HgncResolver,
        disaggregated: bool,
    ) -> None:
        """
        Update evidence extraction and automatically sync gene_mentions with source.

        Args:
            paper_prompts: List of PaperPrompt objects with DOIs and bbox mappings
            results: List of PromptResult objects or None, same order as paper_prompts
            hgnc_resolver: HGNC resolver for gene symbol normalization
            disaggregated: Also sync entity_mentions from the assigned entity ids
        """
        if not paper_prompts or not results:
            return

        successful_updates = 0
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            for paper_prompt, result in zip(paper_prompts, results, strict=True):
                if result is None:
                    continue

                doi = paper_prompt.doi
                bbox_mapping_str = json.dumps(paper_prompt.bbox_mapping)

                # Get source_type to determine gene_mentions source
                cursor.execute("SELECT source_type FROM papers WHERE doi = ?", (doi,))
                row = cursor.fetchone()
                if not row:
                    raise ValueError(f"Paper DOI {doi} not found in database")

                source_type = row["source_type"]
                if not source_type:
                    raise ValueError(f"Paper DOI {doi} has NULL source_type")

                if source_type == "initial":
                    gene_source = "recent_evidence"
                elif source_type == "expansion":
                    gene_source = "expansion_evidence"
                else:
                    raise ValueError(
                        f"Unknown source_type '{source_type}' for DOI {doi}. "
                        f"Expected 'initial' or 'expansion'"
                    )

                # Normalize gene symbols to current HGNC (replaces across entire JSON)
                normalized_json, unresolved_symbols = normalize_extraction_genes(
                    result.parsed_json, hgnc_resolver
                )
                if unresolved_symbols:
                    logger.warning(f"DOI {doi}: unresolved gene symbols: {unresolved_symbols}")

                # Update papers table
                cursor.execute(
                    """
                    UPDATE papers
                    SET evidence_extraction_raw = ?,
                        evidence_extraction_json = ?,
                        bbox_mapping = ?
                    WHERE doi = ?
                    """,
                    (
                        result.raw_response,
                        json.dumps(normalized_json),
                        bbox_mapping_str,
                        doi,
                    ),
                )

                # Sync gene_mentions: delete existing, insert for resolved genes with patient data
                cursor.execute(
                    "DELETE FROM gene_mentions WHERE paper_doi = ? AND source = ?",
                    (doi, gene_source),
                )

                for gene_eval in normalized_json.get("gene_evaluations", []):
                    hgnc_id = gene_eval.get("hgnc_id")
                    if hgnc_id is None:
                        continue  # Unresolved gene
                    if not gene_eval.get("disease_entities"):
                        continue  # Mechanistic only, no patient data

                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO gene_mentions
                        (hgnc_id, paper_gene_symbol, paper_doi, source)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            hgnc_id,
                            gene_eval["paper_gene_symbol"],
                            doi,
                            gene_source,
                        ),
                    )

                if disaggregated:
                    # Sync entity_mentions from the associations this extraction assigned
                    # evidence to; unattributed blocks (null entity_id) contribute nothing.
                    cursor.execute("DELETE FROM entity_mentions WHERE paper_doi = ?", (doi,))
                    entity_ids = {
                        block["entity_id"]
                        for gene_eval in normalized_json.get("gene_evaluations", [])
                        for block in gene_eval["disease_entities"]
                        if block["entity_id"] is not None
                    }
                    cursor.executemany(
                        """
                        INSERT OR IGNORE INTO entity_mentions (entity_id, paper_doi)
                        VALUES (?, ?)
                        """,
                        [(entity_id, doi) for entity_id in sorted(entity_ids)],
                    )

                successful_updates += 1

            conn.commit()
            logger.info(
                f"Updated {successful_updates} papers with evidence extraction and synchronized gene_mentions"
            )

    def get_deep_analysis_statistics(
        self, shard_index: int, num_shards: int, disaggregated: bool
    ) -> dict[str, int]:
        """Get statistics about deep analysis processing progress for this shard.

        `disaggregated` applies the same paper filter as the work query, so
        `remaining_papers` counts exactly the papers the next pass will fetch.
        """
        logger.debug("Querying database for deep analysis statistics...")
        entity_filter = DISAGGREGATED_PAPER_FILTER if disaggregated else ""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()

            # Papers eligible for deep analysis (downloaded full text)
            logger.debug("  Counting papers with downloaded full text...")
            cursor.execute(
                f"""
                SELECT COUNT(*) FROM papers
                WHERE download_status = 'downloaded'
                AND rowid % ? = ?{entity_filter}
                """,
                (num_shards, shard_index),
            )
            eligible_papers = cursor.fetchone()[0]

            # Papers already processed for gene rating
            logger.debug("  Counting already processed papers...")
            cursor.execute(
                f"""
                SELECT COUNT(*) FROM papers
                WHERE evidence_extraction_json IS NOT NULL
                AND rowid % ? = ?{entity_filter}
                """,
                (num_shards, shard_index),
            )
            processed_papers = cursor.fetchone()[0]

            # Remaining papers to process
            logger.debug("  Counting remaining papers to process...")
            cursor.execute(
                f"""
                SELECT COUNT(*) FROM papers
                WHERE download_status = 'downloaded'
                AND evidence_extraction_json IS NULL
                AND rowid % ? = ?{entity_filter}
                """,
                (num_shards, shard_index),
            )
            remaining_papers = cursor.fetchone()[0]

            return {
                "eligible_papers": eligible_papers,
                "processed_papers": processed_papers,
                "remaining_papers": remaining_papers,
            }


CHARS_PER_TOKEN = 4  # conservative average for English text


def prepare_deep_analysis_prompts(
    papers: list[dict[str, Any]],
    template_path: Path,
    panel_formatted: str,
    papers_dir: Path,
    max_model_len: int,
    max_tokens: int,
    entity_blocks: dict[str, str],
) -> DeepAnalysisPreparation:
    """Prepare prompts for deep analysis with full text from Docling JSON files.

    Args:
        papers: List of paper dicts with doi, title, abstract, source_date
        template_path: Path to Jinja2 prompt template
        panel_formatted: Formatted panel description (empty string if not scoping)
        papers_dir: Directory containing Docling JSON files
        max_model_len: Maximum model context length
        max_tokens: Maximum tokens to generate
        entity_blocks: Rendered fixed-association block per DOI; empty in legacy mode

    Returns:
        DeepAnalysisPreparation with prepared prompts and missing DOIs
    """
    env = Environment(loader=FileSystemLoader(template_path.parent), autoescape=False)
    template = env.get_template(template_path.name)

    # Characters of context left once generation is reserved (conservative
    # token-to-char estimate); the full text gets whatever the rest of the
    # rendered prompt leaves over.
    context_chars = int((max_model_len - max_tokens) * CHARS_PER_TOKEN)

    paper_prompts = []
    missing_dois = []

    for paper in papers:
        doi = paper["doi"]
        json_file = doi_to_path(doi, papers_dir, ".json")

        if not json_file.exists():
            logger.warning(f"Missing JSON file for DOI {doi}: {json_file}")
            missing_dois.append(doi)
            continue

        try:
            full_text, bbox_mapping = serialize_with_bbox_ids(json_file)
        except Exception as e:
            logger.error(f"Failed to serialize JSON file for DOI {doi}: {e}")
            missing_dois.append(doi)
            continue

        # A non-empty mapping means disaggregated mode, where every fetched paper
        # has a precomputed block; a missing one is a data surprise, not a default.
        render_args: dict[str, Any] = {
            "title": paper["title"],
            "date": paper["source_date"],
            "abstract": paper["abstract"],
            "panel_formatted": panel_formatted,
            "entity_block": entity_blocks[doi] if entity_blocks else "",
        }

        # Measure the rendered prompt around this paper's full text rather than
        # guessing at it: title, abstract and the fixed-association block all vary.
        available_chars = context_chars - len(template.render(full_text="", **render_args))
        if available_chars <= 0:
            raise ValueError(
                f"Prompt boilerplate exceeds the {context_chars} char context budget for DOI {doi}"
                f" — raise --max-model-len or lower --max-tokens"
            )

        if len(full_text) > available_chars:
            logger.warning(f"Truncating DOI {doi}: {len(full_text)} -> {available_chars} chars")
            full_text = (
                full_text[:available_chars] + "\n\n[NOTE: Paper truncated to fit context window]"
            )

        paper_prompts.append(
            PaperPrompt(template.render(full_text=full_text, **render_args), doi, bbox_mapping)
        )

    return DeepAnalysisPreparation(paper_prompts, missing_dois)


async def _process_evidence(
    *,
    llm_processor: LLMProcessor,
    db_processor: PaperBatchProcessor,
    hgnc_resolver: HgncResolver,
    schema: dict[str, Any],
    prompt_path: Path,
    panel_formatted: str,
    papers_dir: Path,
    max_model_len: int,
    max_tokens: int,
    batch_size: int,
    max_retries: int,
    shard_index: int,
    num_shards: int,
    context: DisaggregatedContext | None,
) -> None:
    """Run the evidence extraction retry loop."""
    disaggregated = context is not None
    entity_blocks = context.prompt_blocks if context is not None else {}

    stats = db_processor.get_deep_analysis_statistics(shard_index, num_shards, disaggregated)
    initial_remaining = stats["remaining_papers"]
    total_processed = 0
    all_missing_dois: set[str] = set()
    consecutive_failures = 0
    retry_attempt = 0

    with tqdm(total=initial_remaining, desc="Processing papers") as pbar:
        while retry_attempt < max_retries:
            stats = db_processor.get_deep_analysis_statistics(
                shard_index, num_shards, disaggregated
            )
            if stats["remaining_papers"] == 0:
                logger.info("All papers successfully processed!")
                break

            if retry_attempt > 0:
                logger.info(
                    f"Retry attempt {retry_attempt} - {stats['remaining_papers']} papers remaining"
                )

            logger.info("Fetching papers for deep analysis...")
            papers = db_processor.get_papers_for_deep_analysis(
                shard_index, num_shards, disaggregated
            )
            logger.info(f"  Retrieved {len(papers)} papers for processing")

            if not papers:
                logger.info("No more papers to process")
                break

            logger.info("Preparing prompts with full text...")
            preparation = prepare_deep_analysis_prompts(
                papers,
                prompt_path,
                panel_formatted,
                papers_dir,
                max_model_len,
                max_tokens,
                entity_blocks,
            )
            logger.info(f"  Successfully prepared {len(preparation.paper_prompts)} prompts")
            logger.info(f"  Missing JSON files: {len(preparation.missing_dois)}")

            all_missing_dois.update(preparation.missing_dois)

            if not preparation.paper_prompts:
                logger.error("No papers could be processed - all JSON files are missing")
                break

            pass_processed = 0
            failed_papers: list[str] = []

            for i in range(0, len(preparation.paper_prompts), batch_size):
                batch = preparation.paper_prompts[i : i + batch_size]
                logger.info(
                    f"Processing batch of {len(batch)} papers "
                    f"({i + 1}-{i + len(batch)}/{len(preparation.paper_prompts)})"
                )

                prompts = [pp.prompt for pp in batch]
                results = await llm_processor.process_batch(prompts, schema)

                for paper_prompt, result in zip(batch, results, strict=True):
                    if result is None:
                        logger.warning(f"Failed to process DOI {paper_prompt.doi}")
                        failed_papers.append(paper_prompt.doi)
                        continue

                    valid_box_ids = set(paper_prompt.bbox_mapping.keys())
                    if not validate_box_ids(result.parsed_json, valid_box_ids):
                        logger.warning(f"Invalid box IDs for DOI {paper_prompt.doi}")
                        failed_papers.append(paper_prompt.doi)
                        continue

                    # Reject placeholder gene symbols (model gave up on real extraction)
                    placeholder_genes = [
                        ge.get("gene_symbol", "?")
                        for ge in result.parsed_json.get("gene_evaluations", [])
                        if any(
                            kw in ge.get("gene_symbol", "").upper()
                            for kw in ("PLACEHOLDER", "DELETE")
                        )
                    ]
                    if placeholder_genes:
                        logger.warning(
                            f"Placeholder gene symbols for DOI {paper_prompt.doi}: "
                            f"{placeholder_genes}"
                        )
                        failed_papers.append(paper_prompt.doi)
                        continue

                    # Validate per-entity evidence_assessments completeness for every gene evaluation
                    incomplete = [
                        ge.get("gene_symbol", "?")
                        for ge in result.parsed_json.get("gene_evaluations", [])
                        if not validate_entities_criteria_complete(ge.get("disease_entities", []))
                    ]
                    if incomplete:
                        logger.warning(
                            f"Incomplete per-entity evidence_assessments for DOI {paper_prompt.doi}, "
                            f"genes: {incomplete}"
                        )
                        failed_papers.append(paper_prompt.doi)
                        continue

                    inconsistent_counts = [
                        ge.get("gene_symbol", "?")
                        for ge in result.parsed_json.get("gene_evaluations", [])
                        if not validate_independent_family_counts(ge.get("disease_entities", []))
                    ]
                    if inconsistent_counts:
                        logger.warning(
                            f"Inconsistent independent_family_count for DOI {paper_prompt.doi}, "
                            f"genes: {inconsistent_counts}"
                        )
                        failed_papers.append(paper_prompt.doi)
                        continue

                    if context is not None:
                        entities_for_paper = context.entities_by_doi[paper_prompt.doi]
                        assignment = validate_entity_assignments(
                            result.parsed_json, entities_for_paper, hgnc_resolver
                        )
                        if assignment.errors:
                            logger.warning(
                                f"Invalid entity assignments for DOI {paper_prompt.doi}: "
                                f"{assignment.errors}"
                            )
                            failed_papers.append(paper_prompt.doi)
                            continue

                        if assignment.off_target_symbols:
                            logger.warning(
                                f"DOI {paper_prompt.doi}: dropping evaluations for genes without "
                                f"fixed associations for this paper: "
                                f"{assignment.off_target_symbols}"
                            )
                            drop_off_target_evaluations(
                                result.parsed_json, assignment.off_target_symbols
                            )

                        annotate_entity_ids(result.parsed_json, entities_for_paper, hgnc_resolver)

                    db_processor.update_paper_evidence_extraction(
                        [paper_prompt], [result], hgnc_resolver, disaggregated
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

    final_stats = db_processor.get_deep_analysis_statistics(shard_index, num_shards, disaggregated)
    logger.info("Deep analysis complete!")
    logger.info("Final statistics:")
    logger.info(f"  Successfully processed: {total_processed:,}")
    logger.info(f"  Missing JSON files: {len(all_missing_dois):,}")
    logger.info(f"  Still remaining: {final_stats['remaining_papers']:,}")

    if all_missing_dois:
        missing_list = sorted(all_missing_dois)
        logger.info(
            f"  DOIs with missing JSON files: {missing_list[:10]}..."
            if len(missing_list) > 10
            else f"  DOIs with missing JSON files: {missing_list}"
        )

    if final_stats["remaining_papers"] > 0:
        logger.warning(
            f"Failed to process {final_stats['remaining_papers']} papers after {max_retries} attempts"
        )


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database",
    ),
    papers_dir: Path = typer.Option(
        Path("data/papers"),
        "--papers-dir",
        help="Directory containing full text files",
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
        Path("prompts/evidence_extraction_prompt.j2"),
        "--prompt-path",
        "-p",
        help="Path to Jinja2 prompt template file",
    ),
    schema_path: Path = typer.Option(
        Path("prompts/evidence_extraction_schema.json"),
        "--schema-path",
        "-s",
        help="Path to response schema file",
    ),
    panel_date: str = typer.Option(
        ...,
        "--panel-date",
        help="Panel state date (YYYY-MM-DD) for gene alias resolution",
    ),
    scope_panel_id: int | None = typer.Option(
        None,
        "--scope-panel-id",
        help="Panel ID for panel-scoped evidence extraction (only extracts genes relevant to this panel)",
    ),
    disaggregated: bool = typer.Option(
        False,
        "--disaggregated/--no-disaggregated",
        help=(
            "Assign extracted evidence to the fixed gene-disease associations in "
            "gene_disease_entities; requires the disaggregated prompt and schema"
        ),
    ),
    shard_index: int = typer.Option(
        0,
        "--shard-index",
        help="Shard index (0-based) for parallel processing across multiple GPUs",
    ),
    num_shards: int = typer.Option(
        1,
        "--num-shards",
        help="Total number of shards for parallel processing (values > 1 require database WAL mode)",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Papers per LLM batch (increase for concurrent API backends like Bedrock)",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of retry attempts for failed papers",
    ),
) -> None:
    """Extract evidence from full text papers using vLLM inference."""
    # Validate inputs
    if disaggregated and scope_panel_id is not None:
        raise typer.BadParameter(
            "--disaggregated assigns evidence to the fixed gene-disease associations, "
            "which are not panel-scoped; drop --scope-panel-id"
        )

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if not papers_dir.exists():
        logger.error(f"Papers directory not found: {papers_dir}")
        raise typer.Exit(1)

    if not prompt_path.exists():
        logger.error(f"Prompt template not found: {prompt_path}")
        raise typer.Exit(1)

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        raise typer.Exit(1)

    # Load schema (template is loaded via Jinja2 in prepare_deep_analysis_prompts)
    logger.info("Loading schema...")
    schema: dict[str, Any] = json.loads(schema_path.read_text())
    logger.info(f"  Loaded schema from {schema_path}")

    if schema_expects_entity_refs(schema) != disaggregated:
        logger.error(
            f"Schema {schema_path} does not match the requested mode: --disaggregated needs the "
            f"schema whose disease entities require entity_ref "
            f"(prompts/evidence_extraction_disaggregated_schema.json, paired with "
            f"prompts/evidence_extraction_disaggregated_prompt.j2)"
        )
        raise typer.Exit(1)

    # Initialize components
    logger.info("Initializing database processor...")
    db_processor = PaperBatchProcessor(db_path)
    logger.info(f"  Connected to database at {db_path}")

    # Load HGNC resolver for gene symbol normalization
    hgnc_resolver = HgncResolver.from_file()
    logger.info(f"  Loaded HgncResolver with {len(hgnc_resolver._by_symbol)} genes")

    # Fail before the model is loaded if the fixed associations are missing
    context = build_disaggregated_context(db_path, hgnc_resolver) if disaggregated else None

    # Build panel_formatted for template (empty string if not scoping to a panel)
    client = PanelAppClient(panel_date)
    if scope_panel_id is not None:
        logger.info(f"Fetching description for scope panel {scope_panel_id}...")
        try:
            panel_info = client.get_panel_data(scope_panel_id)
        except ValueError as e:
            logger.error(f"Panel {scope_panel_id} not found in PanelApp data for {panel_date}")
            raise typer.Exit(1) from e

        panel_formatted = format_panel_for_prompt(scope_panel_id, panel_info)
        logger.info(f"  Panel-scoped mode: {panel_info.get('name', 'Unknown')}")
    else:
        panel_formatted = ""

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
    logger.info("Fetching database statistics...")
    stats = db_processor.get_deep_analysis_statistics(shard_index, num_shards, disaggregated)
    logger.info(f"Deep analysis statistics (shard {shard_index}/{num_shards}):")
    logger.info(f"  Papers with downloaded full text: {stats['eligible_papers']:,}")
    logger.info(f"  Already processed: {stats['processed_papers']:,}")
    logger.info(f"  Remaining to process: {stats['remaining_papers']:,}")

    if stats["remaining_papers"] == 0:
        logger.info("No papers remaining to process!")
        return

    asyncio.run(
        _process_evidence(
            llm_processor=llm_processor,
            db_processor=db_processor,
            hgnc_resolver=hgnc_resolver,
            schema=schema,
            prompt_path=prompt_path,
            panel_formatted=panel_formatted,
            papers_dir=papers_dir,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            batch_size=batch_size,
            max_retries=max_retries,
            shard_index=shard_index,
            num_shards=num_shards,
            context=context,
        )
    )


if __name__ == "__main__":
    app()
