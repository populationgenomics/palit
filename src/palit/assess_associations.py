#!/usr/bin/env python3
"""Aggregate evidence assessment for each fixed gene-disease association.

One LLM call per association (gene x disease x inheritance mode): the disease
entity is given, so the model only weighs how strongly the contributing papers
support it. Evidence is whatever per-paper extraction assigned to that entity, and
nothing else — no panel context, no prior reviews, no network access.
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from jinja2 import Environment, FileSystemLoader, Template

from palit.aggregation import (
    DB_TIMEOUT_SECONDS,
    fetch_valid_box_ids_by_doi,
    filter_preprint_evidence,
    rewrite_paper_ids,
    select_papers_within_budget,
    validate_citation_box_ids,
)
from palit.entities import MOI_PROMPT_GLOSS, DiseaseEntity, entities_by_gene, load_entities
from palit.hgnc import HgncResolver
from palit.llm import LLMProcessor, create_llm_processor
from palit.panelapp_integration import (
    decompose_moi,
    validate_entities_criteria_complete,
    validate_independent_family_count,
)
from palit.papers import generate_paper_ids
from palit.progress import LoggingProgress as Progress

app = typer.Typer(help="Aggregate evidence assessment for each fixed gene-disease association")
logger = logging.getLogger(__name__)

# Prompt sizes are budgeted in characters, converted with the usual English
# rule of thumb of four characters per token.
CHARS_PER_TOKEN = 4

# Context reserved on top of the measured template boilerplate and the generation
# budget, covering chat scaffolding and the imprecision of the character estimate.
CONTEXT_SAFETY_TOKENS = 2000

# Associations this stage assesses: entities of a gene in the recent-evidence
# working set that have at least one contributing paper with a completed
# extraction. The work query and the statistics query share this predicate
# verbatim — the retry loop exits on the remaining count, so any divergence would
# leave entities the loop can never assess and abort it as a no-progress pass.
_ENTITY_SCOPE_SQL = """
    FROM gene_disease_entities e
    WHERE e.hgnc_id IN (
        SELECT DISTINCT hgnc_id FROM gene_mentions WHERE source = 'recent_evidence'
    )
    AND e.hgnc_id % ? = ?
    AND e.id IN (
        SELECT em.entity_id
        FROM entity_mentions em
        JOIN papers p ON p.doi = em.paper_doi
        WHERE p.evidence_extraction_json IS NOT NULL
    )
"""

# Same gene/shard scope, but the entities that have no extracted evidence at all.
_ENTITY_WITHOUT_EVIDENCE_SQL = """
    FROM gene_disease_entities e
    WHERE e.hgnc_id IN (
        SELECT DISTINCT hgnc_id FROM gene_mentions WHERE source = 'recent_evidence'
    )
    AND e.hgnc_id % ? = ?
    AND e.id NOT IN (
        SELECT em.entity_id
        FROM entity_mentions em
        JOIN papers p ON p.doi = em.paper_doi
        WHERE p.evidence_extraction_json IS NOT NULL
    )
"""


@dataclass(frozen=True)
class AssessmentStatistics:
    """Progress of association assessment within one shard."""

    associations_with_evidence: int
    assessed: int
    remaining: int


@dataclass(frozen=True)
class AssociationPrompt:
    """A rendered prompt plus the paper IDs the model may cite in its response."""

    prompt: str
    paper_id_to_doi: dict[str, str]


@dataclass
class _AssociationBatchItem:
    """Per-association metadata needed for post-LLM validation and storage."""

    entity: DiseaseEntity
    gene_symbol: str
    prompt: str
    paper_id_to_doi: dict[str, str]
    evidence_list: list[dict[str, Any]]
    filtered_papers: list[dict[str, str]]


def get_evidence_for_entity(
    conn: sqlite3.Connection, entity: DiseaseEntity
) -> list[dict[str, Any]]:
    """Collect the extracted evidence assigned to one association.

    Each contributing paper is reduced to the gene evaluation for this entity's
    gene, and that evaluation's disease entity blocks are narrowed to the ones
    extraction attributed to this entity. Papers left with nothing are dropped.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT p.doi, p.pmid, p.journal, p.source_date, p.title, p.authors,
               p.evidence_extraction_json,
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
        ORDER BY p.doi
        """,
        (entity.hgnc_id, entity.id),
    )

    evidence_list: list[dict[str, Any]] = []
    for row in cursor.fetchall():
        extraction = json.loads(row["evidence_extraction_json"])

        gene_evaluations: list[dict[str, Any]] = []
        for gene_eval in extraction["gene_evaluations"]:
            if gene_eval.get("hgnc_id") != entity.hgnc_id:
                continue
            del gene_eval["variants"]  # Drop detailed variants from prompt.
            gene_eval["disease_entities"] = [
                block
                for block in gene_eval["disease_entities"]
                if block.get("entity_id") == entity.id
            ]
            if gene_eval["disease_entities"]:
                gene_evaluations.append(gene_eval)

        if not gene_evaluations:
            continue

        evidence_list.append(
            {
                "doi": row["doi"],
                "pmid": row["pmid"],
                "journal": row["journal"],
                "date": row["source_date"],
                "title": row["title"],
                "authors": row["authors"],
                "paper_gene_symbol": row["paper_gene_symbol"],
                "gene_evaluations": gene_evaluations,
            }
        )

    logger.debug(f"Found {len(evidence_list)} papers with evidence for entity {entity.id}")
    return evidence_list


class AssociationBatchProcessor:
    """Database operations for per-association aggregate assessment."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def get_evidence(self, entity: DiseaseEntity) -> list[dict[str, Any]]:
        """Read the extracted evidence assigned to one association."""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            conn.row_factory = sqlite3.Row
            return get_evidence_for_entity(conn, entity)

    def pending_entity_ids(self, shard_index: int, num_shards: int) -> list[int]:
        """Entity ids in this shard that have evidence but no stored assessment."""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            rows = conn.execute(
                f"""
                SELECT e.id
                {_ENTITY_SCOPE_SQL}
                AND e.id NOT IN (SELECT entity_id FROM gene_disease_assessments)
                ORDER BY e.id
                """,
                (num_shards, shard_index),
            ).fetchall()
        return [row[0] for row in rows]

    def get_statistics(self, shard_index: int, num_shards: int) -> AssessmentStatistics:
        """Count in-scope, assessed and remaining associations for this shard."""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            in_scope = conn.execute(
                f"SELECT COUNT(*) {_ENTITY_SCOPE_SQL}", (num_shards, shard_index)
            ).fetchone()[0]
            assessed = conn.execute(
                f"""
                SELECT COUNT(*)
                {_ENTITY_SCOPE_SQL}
                AND e.id IN (SELECT entity_id FROM gene_disease_assessments)
                """,
                (num_shards, shard_index),
            ).fetchone()[0]

        return AssessmentStatistics(
            associations_with_evidence=in_scope,
            assessed=assessed,
            remaining=in_scope - assessed,
        )

    def count_associations_without_evidence(self, shard_index: int, num_shards: int) -> int:
        """Count this shard's associations that no extracted paper contributes to."""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            return int(
                conn.execute(
                    f"SELECT COUNT(*) {_ENTITY_WITHOUT_EVIDENCE_SQL}", (num_shards, shard_index)
                ).fetchone()[0]
            )

    def store_assessment(
        self,
        entity: DiseaseEntity,
        raw_response: str,
        assessment_json: dict[str, Any],
        paper_id_to_doi: dict[str, str],
        filtered_papers: list[dict[str, str]],
    ) -> None:
        """Store one association's assessment in gene_disease_assessments."""
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gene_disease_assessments
                (entity_id, hgnc_id, assessment_raw, assessment_json, paper_id_mapping,
                 filtered_papers_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entity.id,
                    entity.hgnc_id,
                    raw_response,
                    json.dumps(assessment_json),
                    json.dumps(paper_id_to_doi),
                    json.dumps(filtered_papers) if filtered_papers else None,
                ),
            )
            conn.commit()
        logger.info(f"Stored assessment for entity {entity.id} (HGNC:{entity.hgnc_id})")


def format_gene_symbol_with_aliases(hgnc_symbol: str, evidence_list: list[dict[str, Any]]) -> str:
    """Append the symbols the contributing papers used, when they differ from HGNC."""
    paper_symbols = {
        evidence["paper_gene_symbol"]
        for evidence in evidence_list
        if evidence["paper_gene_symbol"] is not None
    }
    aliases = paper_symbols - {hgnc_symbol}
    if aliases:
        return f"{hgnc_symbol} (also referred to as: {', '.join(sorted(aliases))} in the papers)"
    return hgnc_symbol


def format_sibling_lines(entity: DiseaseEntity, gene_entities: list[DiseaseEntity]) -> str:
    """Render the gene's other associations, one line each, for the prompt."""
    return "\n".join(
        f"- {sibling.disease_title} — {MOI_PROMPT_GLOSS[sibling.moi]}"
        for sibling in gene_entities
        if sibling.id != entity.id
    )


def prepare_association_prompt(
    entity: DiseaseEntity,
    gene_symbol: str,
    sibling_lines: str,
    evidence_list: list[dict[str, Any]],
    template: Template,
) -> AssociationPrompt:
    """Render the association prompt and return the paper IDs it exposes.

    Papers are cited as {LastName}{Year} so the model never handles DOIs; the
    caller maps the citations back to DOIs once the response arrives.
    """
    paper_id_to_doi, doi_to_paper_id = generate_paper_ids(evidence_list)

    prompt_evidence = [
        {
            "paper_id": doi_to_paper_id[evidence["doi"]],
            "date": evidence["date"],
            "title": evidence["title"],
            "gene_evaluations": evidence["gene_evaluations"],
        }
        for evidence in evidence_list
    ]

    rendered = template.render(
        gene_symbol=gene_symbol,
        disease_title=entity.disease_title,
        mondo_id=entity.mondo_id,
        moi_gloss=MOI_PROMPT_GLOSS[entity.moi],
        sibling_lines=sibling_lines,
        evidence_extractions=json.dumps(prompt_evidence, indent=2),
    )

    return AssociationPrompt(prompt=rendered, paper_id_to_doi=paper_id_to_doi)


def measure_boilerplate_tokens(template: Template) -> int:
    """Estimate the prompt's fixed cost: everything but the evidence itself."""
    rendered = template.render(
        gene_symbol="",
        disease_title="",
        mondo_id="",
        moi_gloss="",
        sibling_lines="",
        evidence_extractions="",
    )
    return len(rendered) // CHARS_PER_TOKEN


def _prepare_batch_item(
    entity: DiseaseEntity,
    *,
    db_processor: AssociationBatchProcessor,
    hgnc_resolver: HgncResolver,
    gene_entities: list[DiseaseEntity],
    template: Template,
    budget_chars: int,
) -> _AssociationBatchItem | None:
    """Build one association's prompt, or None when it has no usable evidence."""
    gene_symbol = hgnc_resolver.get_symbol(entity.hgnc_id)
    label = f"{gene_symbol} {entity.mondo_id} [{entity.moi}] (entity {entity.id})"

    evidence_list = db_processor.get_evidence(entity)
    if not evidence_list:
        logger.warning(f"No evidence blocks attributed to {label}")
        return None

    evidence_list, filtered_papers = filter_preprint_evidence(evidence_list)
    if filtered_papers:
        logger.info(
            f"  Filtered {len(filtered_papers)} preprint(s) for {label}: "
            f"{[fp['doi'] for fp in filtered_papers]}"
        )
    if not evidence_list:
        logger.info(f"Skipping {label} — all papers filtered by preprint family gate")
        return None

    selection = select_papers_within_budget(evidence_list, budget_chars)
    for evidence, reason in selection.dropped:
        logger.info(f"  Dropped {evidence['doi']} from {label}: {reason}")
        filtered_papers.append({"doi": evidence["doi"], "reason": reason})
    evidence_list = selection.kept

    prepared = prepare_association_prompt(
        entity,
        format_gene_symbol_with_aliases(gene_symbol, evidence_list),
        format_sibling_lines(entity, gene_entities),
        evidence_list,
        template,
    )
    logger.info(
        f"Prepared {label}: {len(evidence_list)} paper(s), "
        f"~{len(prepared.prompt) // CHARS_PER_TOKEN:,} input tokens"
    )

    return _AssociationBatchItem(
        entity=entity,
        gene_symbol=gene_symbol,
        prompt=prepared.prompt,
        paper_id_to_doi=prepared.paper_id_to_doi,
        evidence_list=evidence_list,
        filtered_papers=filtered_papers,
    )


def validate_assessment(
    item: _AssociationBatchItem, parsed_json: dict[str, Any], db_path: Path
) -> bool:
    """Rewrite paper IDs to DOIs and check the response is usable.

    Mutates ``parsed_json`` (paper_id -> doi). Returns False when the association
    must be retried; a mismatch between the response's inheritance mode and the
    association's fixed one is logged but does not fail the response, since the
    prompt asks for exactly that discrepancy to be reported.
    """
    label = f"{item.gene_symbol} entity {item.entity.id}"

    try:
        rewrite_paper_ids(parsed_json, item.paper_id_to_doi)
    except ValueError:
        logger.warning(f"LLM hallucinated a paper ID for {label}, retrying")
        return False

    if not validate_citation_box_ids(
        parsed_json, fetch_valid_box_ids_by_doi(db_path, item.evidence_list)
    ):
        logger.warning(f"Invalid (doi, box_id) pairs for {label}")
        return False

    if not validate_entities_criteria_complete([parsed_json]):
        logger.warning(f"Incomplete criteria for {label} (need criterion_A through criterion_E)")
        return False

    if not validate_independent_family_count(parsed_json):
        logger.warning(
            f"Inconsistent independent_family_count for {label} "
            f"(must be null iff family_count is null, else 0 <= independent <= total)"
        )
        return False

    if parsed_json["inheritance_mode"] not in decompose_moi(item.entity.moi):
        logger.warning(
            f"{label}: assessed inheritance_mode {parsed_json['inheritance_mode']!r} "
            f"differs from the association's fixed mode {item.entity.moi!r}"
        )

    return True


async def _process_assessments(
    *,
    llm_processor: LLMProcessor,
    db_processor: AssociationBatchProcessor,
    db_path: Path,
    hgnc_resolver: HgncResolver,
    entities_by_id: dict[int, DiseaseEntity],
    by_gene: dict[int, list[DiseaseEntity]],
    schema: dict[str, Any],
    template: Template,
    budget_chars: int,
    batch_size: int,
    max_retries: int,
    initial_remaining: int,
    shard_index: int,
    num_shards: int,
) -> None:
    """Run the association assessment retry loop."""
    total_processed = 0
    skipped_entity_ids: set[int] = set()
    consecutive_failures = 0
    retry_attempt = 0

    with Progress() as progress:
        task = progress.add_task("Assessing associations", total=initial_remaining)

        while retry_attempt < max_retries:
            stats = db_processor.get_statistics(shard_index, num_shards)
            if stats.remaining == 0:
                logger.info("All associations successfully assessed!")
                break

            if retry_attempt > 0:
                logger.info(
                    f"Retry attempt {retry_attempt} - {stats.remaining} associations remaining"
                )

            pending = [
                entities_by_id[entity_id]
                for entity_id in db_processor.pending_entity_ids(shard_index, num_shards)
                if entity_id not in skipped_entity_ids
            ]
            logger.info(f"Found {len(pending)} associations to assess")

            if not pending:
                logger.info("No associations need assessment!")
                break

            batch_items: list[_AssociationBatchItem] = []
            for entity in pending:
                item = _prepare_batch_item(
                    entity,
                    db_processor=db_processor,
                    hgnc_resolver=hgnc_resolver,
                    gene_entities=by_gene[entity.hgnc_id],
                    template=template,
                    budget_chars=budget_chars,
                )
                if item is None:
                    skipped_entity_ids.add(entity.id)
                    continue
                batch_items.append(item)

            pass_processed = 0
            for i in range(0, len(batch_items), batch_size):
                batch = batch_items[i : i + batch_size]
                logger.info(
                    f"Processing batch of {len(batch)} associations "
                    f"({i + 1}-{i + len(batch)}/{len(batch_items)})"
                )

                results = await llm_processor.process_batch([item.prompt for item in batch], schema)

                for item, result in zip(batch, results, strict=True):
                    if result is None:
                        logger.warning(f"Failed to assess entity {item.entity.id}")
                        continue
                    if not validate_assessment(item, result.parsed_json, db_path):
                        continue

                    db_processor.store_assessment(
                        item.entity,
                        result.raw_response,
                        result.parsed_json,
                        item.paper_id_to_doi,
                        item.filtered_papers,
                    )
                    pass_processed += 1
                    progress.update(task, advance=1)

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

    final_stats = db_processor.get_statistics(shard_index, num_shards)
    logger.info("Association assessment complete!")
    logger.info("Final statistics:")
    logger.info(f"  Successfully processed: {total_processed:,}")
    logger.info(f"  Skipped for lack of usable evidence: {len(skipped_entity_ids):,}")
    logger.info(f"  Still remaining: {final_stats.remaining:,}")

    if final_stats.remaining > 0:
        logger.warning(
            f"Failed to assess {final_stats.remaining} associations after {max_retries} attempts"
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
        64000,
        "--max-tokens",
        help="Maximum tokens to generate; the remainder of the context window is the evidence budget",
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
    prompt_path: Path = typer.Option(
        Path("prompts/association_assessment_prompt.j2"),
        "--prompt-path",
        "-p",
        help="Path to Jinja2 prompt template file",
    ),
    schema_path: Path = typer.Option(
        Path("prompts/association_assessment_schema.json"),
        "--schema-path",
        "-s",
        help="Path to response schema file",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Associations per LLM batch (increase for concurrent API backends like Bedrock)",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of retry attempts for failed associations",
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
) -> None:
    """Assess each fixed gene-disease association from its contributing papers."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if not prompt_path.exists():
        logger.error(f"Prompt template not found: {prompt_path}")
        raise typer.Exit(1)

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        raise typer.Exit(1)

    logger.info("Loading schema...")
    schema: dict[str, Any] = json.loads(schema_path.read_text())
    logger.info(f"  Loaded schema from {schema_path}")

    env = Environment(loader=FileSystemLoader(prompt_path.parent), autoescape=False)
    template = env.get_template(prompt_path.name)
    boilerplate_tokens = measure_boilerplate_tokens(template)
    budget_chars = (
        max_model_len - max_tokens - boilerplate_tokens - CONTEXT_SAFETY_TOKENS
    ) * CHARS_PER_TOKEN
    logger.info(
        f"  Prompt boilerplate ~{boilerplate_tokens:,} tokens; "
        f"evidence budget {budget_chars:,} chars"
    )
    if budget_chars <= 0:
        logger.error(
            f"No context left for evidence: --max-model-len {max_model_len} cannot hold "
            f"--max-tokens {max_tokens} plus the prompt"
        )
        raise typer.Exit(1)

    hgnc_resolver = HgncResolver.from_file()

    entities = load_entities(db_path)
    entities_by_id = {entity.id: entity for entity in entities}
    by_gene = entities_by_gene(entities)
    logger.info(f"Loaded {len(entities)} entities over {len(by_gene)} genes")

    db_processor = AssociationBatchProcessor(db_path)

    logger.info("Initializing LLM processor...")
    llm_processor = create_llm_processor(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        **(json.loads(llm_config) if llm_config else {}),
    )

    stats = db_processor.get_statistics(shard_index, num_shards)
    without_evidence = db_processor.count_associations_without_evidence(shard_index, num_shards)
    logger.info(f"Association assessment statistics (shard {shard_index}/{num_shards}):")
    logger.info(f"  Associations with evidence: {stats.associations_with_evidence:,}")
    logger.info(f"  Without any extracted evidence (skipped): {without_evidence:,}")
    logger.info(f"  Already assessed: {stats.assessed:,}")
    logger.info(f"  Remaining to assess: {stats.remaining:,}")

    if stats.remaining == 0:
        logger.info("No associations remaining to assess!")
        return

    asyncio.run(
        _process_assessments(
            llm_processor=llm_processor,
            db_processor=db_processor,
            db_path=db_path,
            hgnc_resolver=hgnc_resolver,
            entities_by_id=entities_by_id,
            by_gene=by_gene,
            schema=schema,
            template=template,
            budget_chars=budget_chars,
            batch_size=batch_size,
            max_retries=max_retries,
            initial_remaining=stats.remaining,
            shard_index=shard_index,
            num_shards=num_shards,
        )
    )


if __name__ == "__main__":
    app()
