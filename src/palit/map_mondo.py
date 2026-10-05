#!/usr/bin/env python3
"""Map associations that reuse no PanelApp Australia GenCC row onto MONDO terms.

Each association without a MONDO term goes to the model with its proposed disease
name, description, MoI and summary, and the gene's PanelApp Australia GenCC rows.
The model searches the local MONDO release with the tools in
:mod:`palit.mondo_tools` and answers with one term and a match type: ``exact``
when the term names the disease, ``broader`` when it is the most specific term
that includes it.

Requests go out immediately rather than as batches: inputs are small and a
mapping takes several tool rounds. Conversations are kept in memory, append-only,
because Opus 5.5 rejects replayed thinking after an edited history. An answer is
stored only when it passes the schema and names an eligible MONDO term; otherwise
the association stays unmapped and the next attempt starts a fresh conversation.
A conversation stays on the model it started on: when MODEL refuses an
association in any round, the next attempt starts its conversation afresh on
FALLBACK_MODEL, because thinking blocks cannot be replayed to another model.
"""

import asyncio
import json
import logging
import sqlite3
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import jsonschema
import typer
from anthropic.types import Message
from anthropic.types.output_config_param import OutputConfigParam

from palit.gencc import (
    GenccIndex,
    fetch_gencc,
    fetch_mondo,
    format_paa_associations,
)
from palit.hgnc import HgncResolver
from palit.llm import (
    ALL_TIME,
    Effort,
    ImmediateTransport,
    LlmRequest,
    LlmResult,
    ResultStatus,
    StageHistory,
    Transport,
    assistant_content,
    cached_system,
    invalid_answer_reason,
    json_output_config,
    log_failed_for_good,
    log_refusal,
    make_client,
    parse_json_output,
    record_result,
    stage_history,
    unfinished_answer_reason,
)
from palit.llm_usage import print_stage_summary
from palit.mondo_tools import TOOLS, MondoIndex, MondoToolRunner

app = typer.Typer(help="Map associations without a PanelApp Australia GenCC row onto MONDO")
logger = logging.getLogger(__name__)

STAGE = "map_mondo"
EFFORT: Effort = "medium"
MAX_TOKENS = 16000
# Rounds 1 to LAST_ROUND - 1 may call tools; the last round must answer.
LAST_ROUND = 8

PROMPT_PATH = Path("prompts/map_mondo_prompt.txt")
SCHEMA_PATH = Path("prompts/map_mondo_schema.json")

FINAL_ROUND_TEXT = (
    "These are the last tool results: further tool calls will not be answered. "
    "Give the final answer now."
)

MatchType = Literal["exact", "broader"]


@dataclass(frozen=True)
class Association:
    """The fields of one unmapped association that the model sees."""

    id: int
    hgnc_id: int
    proposed_disease_name: str
    description: str
    inheritance_mode: str
    summary: str


@dataclass(frozen=True)
class MondoMapping:
    mondo_id: str
    label: str  # the ontology's label
    match: MatchType
    rationale: str


@dataclass(frozen=True)
class Conversation:
    """The input messages of one round for one association, and the model it is on."""

    association_id: int
    round: int
    messages: list[dict[str, Any]]
    model: str


@dataclass
class MappingOutcome:
    stored: int = 0
    refused: int = 0
    to_fallback: int = 0  # the refusals by MODEL: these restart on FALLBACK_MODEL
    failed: int = 0


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def select_associations(
    db_path: Path, limit: int | None, history: StageHistory
) -> list[Association]:
    """Associations without a MONDO term, except the ones skipped for good.

    Only rows of a stored aggregation count: rows left behind by deleting a
    ``gene_aggregations`` row are skipped.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.hgnc_id, a.assessment_json
            FROM associations a
            JOIN gene_aggregations g ON g.hgnc_id = a.hgnc_id
            WHERE a.mondo_id IS NULL
            ORDER BY a.id
            """
        ).fetchall()
    due = [row for row in rows if not history.skipped(str(row[0]))]
    associations = []
    for association_id, hgnc_id, assessment_json in due if limit is None else due[:limit]:
        assessment = json.loads(assessment_json)
        associations.append(
            Association(
                id=association_id,
                hgnc_id=hgnc_id,
                proposed_disease_name=assessment["proposed_disease_name"],
                description=assessment["description"],
                inheritance_mode=assessment["inheritance_mode"],
                summary=assessment["summary"],
            )
        )
    return associations


def count_unmapped(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute(
            """
            SELECT COUNT(*) FROM associations a
            JOIN gene_aggregations g ON g.hgnc_id = a.hgnc_id
            WHERE a.mondo_id IS NULL
            """
        ).fetchone()
    return int(count)


def store_mapping(
    conn: sqlite3.Connection, association_id: int, mapping: MondoMapping, raw: str
) -> None:
    cursor = conn.execute(
        """
        UPDATE associations SET mondo_id = ?, mondo_label = ?, mondo_match = ?, mondo_raw = ?
        WHERE id = ? AND mondo_id IS NULL
        """,
        (mapping.mondo_id, mapping.label, mapping.match, raw, association_id),
    )
    if cursor.rowcount == 0:
        logger.warning(
            "Association %d no longer exists or was mapped meanwhile; mapping not stored",
            association_id,
        )


# ---------------------------------------------------------------------------
# Requests and answers
# ---------------------------------------------------------------------------


def association_text(association: Association, gene_symbol: str, gencc: GenccIndex) -> str:
    """The first user message for *association*."""
    return (
        f"GENE: {gene_symbol} (HGNC:{association.hgnc_id})\n\n"
        "ASSOCIATION TO MAP\n"
        f"Proposed disease name: {association.proposed_disease_name}\n"
        f"Mode of inheritance: {association.inheritance_mode}\n"
        f"Description: {association.description}\n"
        f"Summary: {association.summary}\n\n"
        "PANELAPP AUSTRALIA GENCC ROWS FOR THIS GENE (disease (MONDO id) | MoI | class)\n"
        f"{format_paa_associations(gencc.for_gene(association.hgnc_id))}"
    )


def build_request(
    conversation: Conversation, system: str, output_config: OutputConfigParam
) -> LlmRequest:
    return LlmRequest(
        subject=str(conversation.association_id),
        params={
            "model": conversation.model,
            "max_tokens": MAX_TOKENS,
            "system": cached_system(system),
            "tools": TOOLS,
            "messages": conversation.messages,  # type: ignore[typeddict-item]
            "output_config": output_config,
            # Each round's request extends the previous one, so caching the latest
            # message makes the conversation so far a cache read in the next round.
            "cache_control": {"type": "ephemeral"},
        },
    )


def parse_mapping(
    message: Message, validator: jsonschema.protocols.Validator, index: MondoIndex
) -> MondoMapping:
    """The mapping in a final message.

    Raises jsonschema.ValidationError when the answer breaks the schema, and
    ValueError when it is not JSON or names a term that is unknown, obsolete or
    not a human disease term.
    """
    answer = parse_json_output(message)
    validator.validate(answer)
    mondo_id: str = answer["mondo_id"]
    term = index.get(mondo_id)
    if term is None:
        raise ValueError(f"{mondo_id} is not in the loaded MONDO release")
    if term.obsolete:
        raise ValueError(f"{mondo_id} is obsolete (replaced by {', '.join(term.replaced_by)})")
    if not index.is_eligible(mondo_id):
        raise ValueError(f"{mondo_id} ({term.label}) is not a human disease term")
    if answer["label"].strip().lower() != term.label.lower():
        logger.warning(
            "Answer labels %s %r; MONDO's label is %r", mondo_id, answer["label"], term.label
        )
    return MondoMapping(
        mondo_id=mondo_id, label=term.label, match=answer["match"], rationale=answer["rationale"]
    )


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


class MappingRunner:
    def __init__(
        self,
        *,
        transport: Transport,
        db_path: Path,
        system: str,
        schema: dict[str, Any],
        index: MondoIndex,
    ) -> None:
        self._transport = transport
        self._db_path = db_path
        self._system = system
        self._output_config = json_output_config(schema, EFFORT)
        self._validator = jsonschema.Draft202012Validator(schema)
        self._index = index
        self._tools = MondoToolRunner(index)

    async def advance(self, conversations: list[Conversation]) -> MappingOutcome:
        """Run conversations round by round until each has ended."""
        total = MappingOutcome()
        while conversations:
            by_round: dict[int, list[Conversation]] = defaultdict(list)
            for conversation in conversations:
                by_round[conversation.round].append(conversation)
            conversations = []
            for round_no in sorted(by_round):
                current = {c.association_id: c for c in by_round[round_no]}
                requests = [
                    build_request(c, self._system, self._output_config) for c in current.values()
                ]
                results = await self._transport.run(STAGE, round_no, requests)
                conversations += self.handle(results, current, total)
        return total

    def handle(
        self,
        results: list[LlmResult],
        conversations: dict[int, Conversation],
        outcome: MappingOutcome,
    ) -> list[Conversation]:
        """Record every result, store valid answers, and return the next round's conversations."""
        next_round: list[Conversation] = []
        invalid_requests: list[str] = []
        for result in results:
            conversation = conversations[int(result.subject)]
            message = result.message
            if result.status == ResultStatus.REFUSED:
                outcome.refused += 1
                outcome.to_fallback += result.goes_to_fallback
                log_refusal(result, f"association {result.subject} in round {result.round}")
                self._record(result)
            elif result.status != ResultStatus.SUCCEEDED:
                outcome.failed += 1
                if result.error_type == "invalid_request_error":
                    invalid_requests.append(result.subject)
                self._record(result)
            elif message is None:
                raise AssertionError("succeeded result without a message")
            elif message.stop_reason == "tool_use" and result.round < LAST_ROUND:
                next_round.append(self._next_round(conversation, message))
                self._record(result)
            elif message.stop_reason == "end_turn":
                if self._store(result, message):
                    outcome.stored += 1
                else:
                    outcome.failed += 1
            else:
                rejection = unfinished_answer_reason(message.stop_reason)
                logger.warning(
                    "Rejected association %s round %d: %s", result.subject, result.round, rejection
                )
                outcome.failed += 1
                self._record(result, rejection=rejection)
        if invalid_requests:
            raise RuntimeError(
                f"{len(invalid_requests)} map-mondo requests were invalid "
                f"(first: association {invalid_requests[0]}); see llm_requests.error_type"
            )
        return next_round

    def _record(self, result: LlmResult, *, rejection: str | None = None) -> None:
        with sqlite3.connect(self._db_path) as conn:
            record_result(conn, result, rejection=rejection)

    def _next_round(self, conversation: Conversation, message: Message) -> Conversation:
        tool_results: list[dict[str, Any]] = []
        for block in message.content:
            if block.type != "tool_use":
                continue
            tool_outcome = self._tools.run(block.name, dict(block.input))
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_outcome.content,
                    "is_error": tool_outcome.is_error,
                }
            )
        next_round = conversation.round + 1
        user_content: list[dict[str, Any]] = tool_results
        if next_round == LAST_ROUND:
            user_content = [*tool_results, {"type": "text", "text": FINAL_ROUND_TEXT}]
        return Conversation(
            association_id=conversation.association_id,
            round=next_round,
            messages=[
                *conversation.messages,
                {"role": "assistant", "content": assistant_content(message)},
                {"role": "user", "content": user_content},
            ],
            model=conversation.model,
        )

    def _store(self, result: LlmResult, message: Message) -> bool:
        try:
            mapping = parse_mapping(message, self._validator, self._index)
        except (ValueError, jsonschema.ValidationError) as e:
            rejection = invalid_answer_reason(e)
            logger.warning("Rejected mapping for association %s: %s", result.subject, rejection)
            self._record(result, rejection=rejection)
            return False
        logger.info(
            "Association %s -> %s %s (%s): %s",
            result.subject,
            mapping.mondo_id,
            mapping.label,
            mapping.match,
            mapping.rationale,
        )
        with sqlite3.connect(self._db_path) as conn:
            record_result(conn, result)
            store_mapping(conn, int(result.subject), mapping, message.to_json())
        return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def map_associations(
    *,
    runner: MappingRunner,
    db_path: Path,
    hgnc_resolver: HgncResolver,
    gencc: GenccIndex,
    limit: int | None,
    max_retries: int,
    failures_since: datetime,
) -> None:
    """Map every unmapped association, retrying failed ones up to *max_retries* times.

    An attempt makes progress when it stores a mapping or sends an association on
    to FALLBACK_MODEL, so a round of refusals by MODEL is followed by the attempt
    that restarts those associations on FALLBACK_MODEL. Failed answers count from
    *failures_since*.
    """

    def load_history() -> StageHistory:
        with closing(sqlite3.connect(db_path)) as conn:
            return stage_history(conn, STAGE, failures_since=failures_since)

    log_failed_for_good(load_history().failures, STAGE)
    for attempt in range(1, max_retries + 1):
        history = load_history()
        associations = select_associations(db_path, limit, history)
        if not associations:
            logger.info("No associations left to map")
            return
        conversations = [
            Conversation(
                association_id=association.id,
                round=1,
                messages=[
                    {
                        "role": "user",
                        "content": association_text(
                            association, hgnc_resolver.get_symbol(association.hgnc_id), gencc
                        ),
                    }
                ],
                model=history.model_for(str(association.id)),
            )
            for association in associations
        ]
        logger.info("Attempt %d: mapping %d associations", attempt, len(conversations))
        outcome = await runner.advance(conversations)
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
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Map at most this many associations, in one attempt (for prompt development)",
    ),
    max_retries: int = typer.Option(
        3,
        "--max-retries",
        help="Maximum number of attempts for associations whose mapping failed, was rejected "
        "or was refused; the fallback-model conversation after a refusal is one of them",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Send associations whose answers failed for good in earlier invocations (rejected "
        "or cut off at max_tokens, MAX_FAILED_ANSWERS times) again, until they fail that often "
        "in this invocation",
    ),
) -> None:
    """Map every association without a MONDO term onto one, with the model and the MONDO tools."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)
    # With --retry-failed, only failed answers during this invocation count.
    failures_since = datetime.now(UTC) if retry_failed else ALL_TIME

    system = PROMPT_PATH.read_text().strip()
    schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())
    mondo = fetch_mondo(db_path.parent)
    index = MondoIndex.from_ontology(mondo)
    gencc = fetch_gencc(db_path.parent, mondo)
    hgnc_resolver = HgncResolver.from_file()
    logger.info(f"{count_unmapped(db_path):,} associations without a MONDO term")

    async def run() -> None:
        runner = MappingRunner(
            transport=ImmediateTransport(make_client()),
            db_path=db_path,
            system=system,
            schema=schema,
            index=index,
        )
        await map_associations(
            runner=runner,
            db_path=db_path,
            hgnc_resolver=hgnc_resolver,
            gencc=gencc,
            limit=limit,
            max_retries=max_retries,
            failures_since=failures_since,
        )

    asyncio.run(run())
    logger.info(f"{count_unmapped(db_path):,} associations still without a MONDO term")
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
