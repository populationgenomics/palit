#!/usr/bin/env python3
"""Shared tournament selection logic for literature filtering.

Each gene's candidate papers are split into prompts of ``papers_per_round``
papers. Each prompt also shows the gene's PanelApp Australia GenCC rows, so that
selection can favour weakly curated or uncurated associations. The model keeps
up to ``max_papers`` from each prompt, and the survivors go into the next round
until a single prompt remains. All genes advance in lockstep, one Message Batch
per round across every gene, because each round depends on the previous one.

Tournament state lives only in memory: after an interruption the tournament
restarts, and results of batches still in flight are recorded for their usage
but not used. A prompt MODEL refuses is sent again on FALLBACK_MODEL within the
same round, before the round's selections are used.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
from anthropic.types.output_config_param import OutputConfigParam

from palit.gencc import PAA_ASSOCIATIONS_CAVEAT, GenccIndex, format_paa_associations
from palit.hgnc import HgncResolver
from palit.llm import (
    FALLBACK_MODEL,
    MODEL,
    Effort,
    LlmRequest,
    LlmResult,
    ResultStatus,
    Transport,
    json_output_config,
    parse_json_output,
    record_result,
)
from palit.papers import Paper

logger = logging.getLogger(__name__)

EFFORT: Effort = "medium"
MAX_TOKENS = 16000


@dataclass
class TournamentOutcome:
    """Final outcome of one gene's tournament across all rounds."""

    selected_papers: list[Paper]
    raw_responses_by_round: list[list[str]]


@dataclass
class TournamentEntry:
    """One gene's tournament input."""

    key: str  # HGNC ID as text
    gene_symbol: str
    gencc_context: str  # the gene's PanelApp Australia GenCC rows, as shown in the prompt
    papers: list[Paper]

    @classmethod
    def for_gene(
        cls, hgnc_id: int, papers: list[Paper], resolver: HgncResolver, gencc: GenccIndex
    ) -> "TournamentEntry":
        return cls(
            key=str(hgnc_id),
            gene_symbol=resolver.get_symbol(hgnc_id),
            gencc_context=format_paa_associations(gencc.for_gene(hgnc_id)),
            papers=papers,
        )


@dataclass
class _GeneState:
    entry: TournamentEntry
    current: list[Paper]
    responses_by_round: list[list[str]] = field(default_factory=list)


def format_papers_for_prompt(papers: list[Paper]) -> str:
    """Format papers as XML-style list for LLM prompt using sequential indices.

    Returns a string like:
        <paper id=0><date>2024-01-15</date><title>...</title><abstract>...</abstract></paper>
        <paper id=1><date>2024-02-20</date><title>...</title><abstract>...</abstract></paper>

    Uses sequential indices (0, 1, 2...) to avoid LLM transcription errors with
    long identifiers. Caller maps indices back to papers.
    """
    lines = []
    for idx, paper in enumerate(papers):
        abstract = paper.abstract or ""
        if len(abstract) > 1000:
            abstract = abstract[:1000] + "..."
        lines.append(
            f"<paper id={idx}><date>{paper.source_date}</date><title>{paper.title}</title>"
            f"<abstract>{abstract}</abstract></paper>"
        )
    return "\n".join(lines)


def tournament_prompt(
    template: str, entry: TournamentEntry, papers: list[Paper], max_papers: int
) -> str:
    """The user message selecting up to *max_papers* of *papers* for *entry*'s gene."""
    return template.format(
        gene_symbol=entry.gene_symbol,
        gencc_context=entry.gencc_context,
        gencc_caveat=PAA_ASSOCIATIONS_CAVEAT,
        max_papers=max_papers,
        papers_list=format_papers_for_prompt(papers),
    )


def record_abandoned(db_path: Path, results: list[LlmResult]) -> None:
    """Record results from an interrupted tournament for their usage only."""
    with sqlite3.connect(db_path) as conn:
        for result in results:
            record_result(conn, result)
    if results:
        logger.info("Recorded %d results from an interrupted tournament (not used)", len(results))


async def run_tournaments(
    entries: list[TournamentEntry],
    *,
    transport: Transport,
    db_path: Path,
    stage: str,
    prompt_template: str,
    schema: dict[str, Any],
    max_papers: int,
    papers_per_round: int,
    max_retries: int,
) -> dict[str, TournamentOutcome]:
    """Run every gene's tournament; returns outcomes keyed by entry key."""
    validator = jsonschema.Draft202012Validator(schema)
    output_config = json_output_config(schema, EFFORT)
    outcomes: dict[str, TournamentOutcome] = {}
    active: dict[str, _GeneState] = {}
    for entry in entries:
        if len(entry.papers) <= max_papers:
            outcomes[entry.key] = TournamentOutcome(entry.papers, raw_responses_by_round=[])
        else:
            active[entry.key] = _GeneState(entry=entry, current=entry.papers)

    round_no = 0
    while active:
        round_no += 1
        # (gene key, prompt index) -> papers in that prompt
        prompts: dict[tuple[str, int], list[Paper]] = {}
        for key, state in active.items():
            for index, start in enumerate(range(0, len(state.current), papers_per_round)):
                prompts[(key, index)] = state.current[start : start + papers_per_round]
        logger.info(
            "Tournament round %d: %d genes, %d prompts", round_no, len(active), len(prompts)
        )

        selections = await _run_prompts(
            prompts,
            active=active,
            transport=transport,
            db_path=db_path,
            stage=stage,
            round_no=round_no,
            prompt_template=prompt_template,
            output_config=output_config,
            validator=validator,
            max_papers=max_papers,
            max_retries=max_retries,
        )

        for key, state in list(active.items()):
            indices = sorted(index for (k, index) in prompts if k == key)
            state.current = [paper for index in indices for paper in selections[(key, index)][0]]
            state.responses_by_round.append([selections[(key, index)][1] for index in indices])
            if len(indices) <= 1:
                outcomes[key] = TournamentOutcome(state.current, state.responses_by_round)
                del active[key]
    return outcomes


async def _run_prompts(
    prompts: dict[tuple[str, int], list[Paper]],
    *,
    active: dict[str, _GeneState],
    transport: Transport,
    db_path: Path,
    stage: str,
    round_no: int,
    prompt_template: str,
    output_config: OutputConfigParam,
    validator: jsonschema.protocols.Validator,
    max_papers: int,
    max_retries: int,
) -> dict[tuple[str, int], tuple[list[Paper], str]]:
    """Selected papers and raw response per prompt, retrying failed prompts.

    Each try sends every prompt without a valid answer yet. A prompt MODEL refused
    goes to FALLBACK_MODEL from the next try on; that try counts toward
    *max_retries* like any other.
    """
    selections: dict[tuple[str, int], tuple[list[Paper], str]] = {}
    pending = list(prompts)
    failures: dict[tuple[str, int], str] = {}
    models = dict.fromkeys(prompts, MODEL)
    for _ in range(max_retries):
        if not pending:
            break
        subjects = {f"{key}:{round_no}:{index}": (key, index) for key, index in pending}
        requests = [
            LlmRequest(
                subject=subject,
                params={
                    "model": models[(key, index)],
                    "max_tokens": MAX_TOKENS,
                    "messages": [
                        {
                            "role": "user",
                            "content": tournament_prompt(
                                prompt_template,
                                active[key].entry,
                                prompts[(key, index)],
                                max_papers,
                            ),
                        }
                    ],
                    "output_config": output_config,
                },
            )
            for subject, (key, index) in subjects.items()
        ]
        results = await transport.run(stage, round_no, requests)
        with sqlite3.connect(db_path) as conn:
            for result in results:
                record_result(conn, result)
        pending = []
        for result in results:
            job = subjects[result.subject]
            papers = prompts[job]
            if result.status != ResultStatus.SUCCEEDED or result.message is None:
                if result.error_type == "invalid_request_error":
                    raise RuntimeError(f"invalid tournament request {result.subject}")
                if result.goes_to_fallback:
                    logger.warning(
                        "%s refused tournament prompt %s; sending it to %s",
                        result.model,
                        result.subject,
                        FALLBACK_MODEL,
                    )
                    models[job] = FALLBACK_MODEL
                failures[job] = f"{result.status.value} by {result.model}"
                pending.append(job)
                continue
            try:
                parsed = parse_json_output(result.message)
                validator.validate(parsed)
            except (ValueError, jsonschema.ValidationError) as e:
                failures[job] = str(e)
                pending.append(job)
                continue
            selected = parsed["papers"][:max_papers]
            out_of_range = [i for i in selected if not 0 <= i < len(papers)]
            if out_of_range:
                failures[job] = (
                    f"out-of-range indices {out_of_range} (prompt has {len(papers)} papers)"
                )
                pending.append(job)
                continue
            selections[job] = ([papers[i] for i in selected], json.dumps(parsed))
    if pending:
        summary = "; ".join(
            f"{key} prompt {index}: {failures[(key, index)]}" for key, index in pending
        )
        raise RuntimeError(f"Tournament prompts failed after {max_retries} attempts: {summary}")
    return selections
