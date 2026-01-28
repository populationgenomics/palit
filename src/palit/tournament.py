#!/usr/bin/env python3
"""Shared tournament selection logic for literature filtering."""

import logging
from dataclasses import dataclass
from typing import Any

from palit.ingest_pubmed import Article

logger = logging.getLogger(__name__)


@dataclass
class TournamentPromptResult:
    """Result of a single tournament prompt."""

    selected_articles: list[Article]
    raw_response: str


@dataclass
class TournamentOutcome:
    """Final outcome of the tournament across all rounds."""

    selected_articles: list[Article]
    raw_responses_by_round: list[list[str]]


def format_papers_for_prompt(papers: list[Article]) -> str:
    """Format papers as XML-style list for LLM prompt using indices instead of PMIDs.

    Args:
        papers: List of Article objects

    Returns:
        Formatted string like:
        <paper id=0><date>2024-01-15</date><title>...</title><abstract>...</abstract></paper>
        <paper id=1><date>2024-02-20</date><title>...</title><abstract>...</abstract></paper>

    Note: Uses sequential indices (0, 1, 2...) to avoid LLM transcription errors with
    similar-looking 8-digit PMIDs. Caller maps indices back to PMIDs.
    """
    lines = []
    for idx, paper in enumerate(papers):
        title = paper.title
        abstract = paper.abstract or ""
        entrez_date = paper.entrez_date

        # Truncate abstract if too long
        if abstract and len(abstract) > 1000:
            abstract = abstract[:1000] + "..."

        lines.append(
            f"<paper id={idx}><date>{entrez_date}</date><title>{title}</title><abstract>{abstract}</abstract></paper>"
        )

    return "\n".join(lines)


def run_tournament_selection(
    gene_symbol: str,
    articles: list[Article],
    llm_processor: Any,
    prompt_template: str,
    schema: dict[str, Any],
    max_papers: int,
    papers_per_round: int,
    max_concurrent_batches: int,
    max_retries: int,
) -> TournamentOutcome:
    """Run hierarchical tournament selection for a gene.

    Args:
        gene_symbol: Gene symbol being processed
        articles: List of articles to run tournament on
        llm_processor: HarmonyBatchProcessor instance
        prompt_template: Prompt template string
        schema: JSON schema for LLM output
        max_papers: Maximum papers to select
        papers_per_round: Papers per batch
        max_concurrent_batches: Max concurrent batches
        max_retries: Max retries per batch

    Returns:
        TournamentOutcome with selected articles and raw responses
    """
    if len(articles) <= max_papers:
        logger.info(f"Gene has ≤{max_papers} papers, skipping tournament")
        return TournamentOutcome(selected_articles=articles, raw_responses_by_round=[])

    current_round_articles = articles
    round_num = 0
    raw_responses_by_round: list[list[str]] = []

    while True:
        round_num += 1
        logger.info(f"Round {round_num}: {len(current_round_articles)} papers")

        batches = [
            current_round_articles[i : i + papers_per_round]
            for i in range(0, len(current_round_articles), papers_per_round)
        ]
        logger.info(f"  Processing {len(batches)} batches...")

        prompt_results = _process_batches(
            gene_symbol=gene_symbol,
            llm_processor=llm_processor,
            prompt_template=prompt_template,
            schema=schema,
            batches=batches,
            max_papers=max_papers,
            max_concurrent_batches=max_concurrent_batches,
            max_retries=max_retries,
        )

        if prompt_results:
            raw_responses_by_round.append([result.raw_response for result in prompt_results])

        next_round_articles = []
        for result in prompt_results:
            next_round_articles.extend(result.selected_articles)

        current_round_articles = next_round_articles
        logger.info(f"  Round {round_num} result: {len(current_round_articles)} papers")

        if len(batches) <= 1:
            break

    logger.info(f"Tournament complete: {len(current_round_articles)} final papers")
    return TournamentOutcome(
        selected_articles=current_round_articles,
        raw_responses_by_round=raw_responses_by_round,
    )


def _process_batches(
    *,
    gene_symbol: str,
    llm_processor: Any,
    prompt_template: str,
    schema: dict[str, Any],
    batches: list[list[Article]],
    max_papers: int,
    max_concurrent_batches: int,
    max_retries: int,
) -> list[TournamentPromptResult]:
    prompt_results: list[TournamentPromptResult] = []

    for chunk_start in range(0, len(batches), max_concurrent_batches):
        chunk_end = min(chunk_start + max_concurrent_batches, len(batches))
        chunk_batches = batches[chunk_start:chunk_end]
        logger.info(
            f"  Processing batches {chunk_start + 1}-{chunk_end}/{len(batches)} "
            f"({len(chunk_batches)} prompts in parallel)..."
        )

        chunk_results = _process_chunk(
            gene_symbol=gene_symbol,
            llm_processor=llm_processor,
            prompt_template=prompt_template,
            schema=schema,
            chunk_batches=chunk_batches,
            chunk_offset=chunk_start,
            max_papers=max_papers,
            max_retries=max_retries,
        )
        prompt_results.extend(chunk_results)

    return prompt_results


def _process_chunk(
    *,
    gene_symbol: str,
    llm_processor: Any,
    prompt_template: str,
    schema: dict[str, Any],
    chunk_batches: list[list[Article]],
    chunk_offset: int,
    max_papers: int,
    max_retries: int,
) -> list[TournamentPromptResult]:
    pending_indices = list(range(len(chunk_batches)))
    results: dict[int, TournamentPromptResult] = {}
    failure_reasons: dict[int, str] = {}
    retry_counts: dict[int, int] = dict.fromkeys(pending_indices, 0)

    while pending_indices:
        prompts = [
            _build_prompt(
                gene_symbol=gene_symbol,
                prompt_template=prompt_template,
                max_papers=max_papers,
                batch=chunk_batches[idx],
            )
            for idx in pending_indices
        ]

        batch_results = llm_processor.process_batch(prompts, schema)

        next_pending: list[int] = []
        for local_idx, result in zip(pending_indices, batch_results, strict=True):
            global_batch_idx = chunk_offset + local_idx
            current_attempt = retry_counts[local_idx]

            if result is None:
                failure_reason = (
                    "LLM returned None (invalid/unparseable JSON or generation failure)"
                )
                failure_reasons[local_idx] = failure_reason
                retry_counts[local_idx] += 1

                if current_attempt + 1 < max_retries:
                    logger.warning(
                        "    Batch %s failed: %s, retrying (attempt %s/%s)",
                        global_batch_idx + 1,
                        failure_reason,
                        current_attempt + 2,
                        max_retries,
                    )
                    next_pending.append(local_idx)
                else:
                    logger.error(
                        "    Batch %s failed on final attempt: %s",
                        global_batch_idx + 1,
                        failure_reason,
                    )
                continue

            selected_indices = result.parsed_json.get("papers", [])[:max_papers]
            batch_articles = chunk_batches[local_idx]

            invalid_indices = [
                idx for idx in selected_indices if idx < 0 or idx >= len(batch_articles)
            ]
            if invalid_indices:
                failure_reason = (
                    f"Invalid indices: {invalid_indices} (batch has {len(batch_articles)} papers)"
                )
                failure_reasons[local_idx] = failure_reason

                logger.error(
                    "    Batch %s returned out-of-range indices: %s",
                    global_batch_idx + 1,
                    invalid_indices,
                )
                logger.error("      Valid range: 0-%d", len(batch_articles) - 1)
                logger.error("      All returned indices: %s", selected_indices)

                retry_counts[local_idx] += 1

                if current_attempt + 1 < max_retries:
                    logger.warning(
                        "    Batch %s failed: %s, retrying (attempt %s/%s)",
                        global_batch_idx + 1,
                        failure_reason,
                        current_attempt + 2,
                        max_retries,
                    )
                    next_pending.append(local_idx)
                else:
                    logger.error(
                        "    Batch %s failed on final attempt: %s",
                        global_batch_idx + 1,
                        failure_reason,
                    )
                continue

            selected_articles = [batch_articles[idx] for idx in selected_indices]

            logger.info(
                "    Batch %s: Selected %s papers",
                global_batch_idx + 1,
                len(selected_articles),
            )
            results[local_idx] = TournamentPromptResult(
                selected_articles=selected_articles,
                raw_response=result.raw_response,
            )

        pending_indices = next_pending

    if pending_indices:
        failed_batches_info = []
        for idx in pending_indices:
            batch_num = chunk_offset + idx + 1
            reason = failure_reasons.get(idx, "Unknown reason")
            attempts = retry_counts.get(idx, 0)
            failed_batches_info.append(f"Batch {batch_num} ({attempts} attempts): {reason}")

        failed_summary = "; ".join(failed_batches_info)
        raise RuntimeError(f"Failed batches: {failed_summary}")

    ordered_results: list[TournamentPromptResult] = []
    for local_idx in range(len(chunk_batches)):
        ordered_results.append(results[local_idx])

    return ordered_results


def _build_prompt(
    *, gene_symbol: str, prompt_template: str, max_papers: int, batch: list[Article]
) -> str:
    papers_list = format_papers_for_prompt(batch)
    return prompt_template.format(
        gene_symbol=gene_symbol,
        max_papers=max_papers,
        papers_list=papers_list,
    )
