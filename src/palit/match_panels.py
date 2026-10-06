#!/usr/bin/env python3
"""Match each gene-disease-MoI association to diagnostic panels.

One request per association without panel matches: the system part lists every
panel and is the same for all requests, so it is cached; the user part carries
the association's disease, mode of inheritance and summary. Matches are stored
on the association row, in the transaction that records the request. An
association MODEL refused goes to FALLBACK_MODEL in the next attempt (see
:mod:`palit.llm`).

Requests go out immediately rather than as batches, so that every request but
the first per model reads the panel list from the cache. Concurrent requests
in a batch mostly write the cache instead of reading it. A cache entry belongs
to one model, so in each attempt the first request to each model goes out alone,
and the model's other requests follow once it has returned. Each result is
stored as soon as it arrives.
"""

import asyncio
import json
import logging
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import typer
from anthropic.types.output_config_param import OutputConfigParam

from palit.llm import (
    ALL_TIME,
    DEFAULT_IMMEDIATE_WORKERS,
    Effort,
    ImmediateTransport,
    LlmRequest,
    LlmResult,
    ResultStatus,
    StageHistory,
    Transport,
    cached_system,
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
from palit.panelapp_client import PanelAppClient, format_panel_for_prompt
from palit.progress import LoggingProgress as Progress

app = typer.Typer(help="Match gene-disease-MoI associations to diagnostic panels")
logger = logging.getLogger(__name__)

STAGE = "match_panels"
EFFORT: Effort = "low"
MAX_TOKENS = 16000
ASSOCIATION_HEADING = "ASSOCIATION:"

PROMPT_PATH = Path("prompts/panel_matching_prompt.txt")
SCHEMA_PATH = Path("prompts/panel_matching_schema.json")


@dataclass(frozen=True)
class Association:
    """The fields of one unmatched association that the model sees."""

    id: int
    description: str
    inheritance_mode: str
    inheritance_details: str
    summary: str


@dataclass(frozen=True)
class PanelMatch:
    panel_id: int
    rationale: str


@dataclass
class MatchOutcome:
    stored: int = 0
    to_fallback: int = 0  # the refusals by MODEL: these go to FALLBACK_MODEL in the next attempt


def format_all_panels_for_prompt(panels: dict[int, dict[str, Any]]) -> str:
    """All panel descriptions, ordered by panel id."""
    return "\n".join(
        format_panel_for_prompt(panel_id, info) for panel_id, info in sorted(panels.items())
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def select_associations(db_path: Path, history: StageHistory) -> list[Association]:
    """Associations without panel matches, except the ones skipped for good.

    Only rows of a stored aggregation count: rows left behind by deleting a
    ``gene_aggregations`` row are skipped.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.assessment_json
            FROM associations a
            JOIN gene_aggregations g ON g.hgnc_id = a.hgnc_id
            WHERE a.matched_panels_json IS NULL
            ORDER BY a.id
            """
        ).fetchall()
    associations = []
    for association_id, assessment_json in rows:
        if history.skipped(str(association_id)):
            continue
        assessment = json.loads(assessment_json)
        associations.append(
            Association(
                id=association_id,
                description=assessment["description"],
                inheritance_mode=assessment["inheritance_mode"],
                inheritance_details=assessment["inheritance_details"],
                summary=assessment["summary"],
            )
        )
    return associations


def count_unmatched(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute(
            """
            SELECT COUNT(*) FROM associations a
            JOIN gene_aggregations g ON g.hgnc_id = a.hgnc_id
            WHERE a.matched_panels_json IS NULL
            """
        ).fetchone()
    return int(count)


def store_matched_panels(
    conn: sqlite3.Connection, association_id: int, matches: list[PanelMatch], raw: str
) -> bool:
    """Store *matches* on the association; False when the row is gone or already matched."""
    cursor = conn.execute(
        """
        UPDATE associations SET matched_panels_json = ?, matched_panels_raw = ?
        WHERE id = ? AND matched_panels_json IS NULL
        """,
        (json.dumps([asdict(match) for match in matches]), raw, association_id),
    )
    if cursor.rowcount == 0:
        logger.warning(
            "Association %d no longer exists or was matched meanwhile; matches not stored",
            association_id,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Requests and answers
# ---------------------------------------------------------------------------


def split_prompt(template: str, panel_list: str) -> tuple[str, str]:
    """(system prompt with the panel list, per-association template)."""
    index = template.index(ASSOCIATION_HEADING)
    system = template[:index].format(panel_list=panel_list).rstrip()
    return system, template[index:]


def format_inheritance(association: Association) -> str:
    mode = "not reported" if association.inheritance_mode == "NR" else association.inheritance_mode
    if association.inheritance_details:
        return f"{mode} ({association.inheritance_details})"
    return mode


def build_request(
    association: Association,
    system: str,
    user_template: str,
    output_config: OutputConfigParam,
    model: str,
) -> LlmRequest:
    user_text = user_template.format(
        description=association.description,
        inheritance=format_inheritance(association),
        summary=association.summary,
    )
    return LlmRequest(
        subject=str(association.id),
        params={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": cached_system(system),
            "messages": [{"role": "user", "content": user_text}],
            "output_config": output_config,
        },
    )


def parse_matches(answer: dict[str, Any], name_to_id: dict[str, int]) -> list[PanelMatch]:
    """The answer's matches by panel id.

    Raises ValueError when the answer names a panel that is not in the list, or
    names one panel twice.
    """
    matches: list[PanelMatch] = []
    unknown: list[str] = []
    for match in answer["matched_panels"]:
        panel_id = name_to_id.get(match["panel_name"].lower())
        if panel_id is None:
            unknown.append(match["panel_name"])
        else:
            matches.append(PanelMatch(panel_id=panel_id, rationale=match["rationale"]))
    if unknown:
        raise ValueError(f"unknown panel names {unknown}")
    panel_ids = [match.panel_id for match in matches]
    if len(set(panel_ids)) != len(panel_ids):
        raise ValueError(f"repeated panels {panel_ids}")
    return matches


def handle_result(
    result: LlmResult,
    db_path: Path,
    validator: jsonschema.protocols.Validator,
    name_to_id: dict[str, int],
) -> bool:
    """Record *result* and store its matches if they are valid; True when they were stored.

    Raises RuntimeError, after recording it, when the request was invalid.
    """
    association_id = int(result.subject)
    message = result.message
    if result.status == ResultStatus.REFUSED:
        log_refusal(result, f"association {association_id}")
    if result.status != ResultStatus.SUCCEEDED or message is None:
        with sqlite3.connect(db_path) as conn:
            record_result(conn, result)
        if result.error_type == "invalid_request_error":
            raise RuntimeError(f"invalid match-panels request for association {association_id}")
        return False
    try:
        answer = parse_json_output(message)
        validator.validate(answer)
        matches = parse_matches(answer, name_to_id)
    except (ValueError, jsonschema.ValidationError) as e:
        rejection = invalid_answer_reason(e)
        logger.warning("Invalid panel matching for association %d: %s", association_id, rejection)
        with sqlite3.connect(db_path) as conn:
            record_result(conn, result, rejection=rejection)
        return False
    with sqlite3.connect(db_path) as conn:
        record_result(conn, result)
        return store_matched_panels(conn, association_id, matches, message.to_json())


async def send_primed(
    transport: Transport,
    requests: Sequence[LlmRequest],
    on_result: Callable[[LlmResult], None],
) -> None:
    """Send *requests* and pass each result to *on_result* as it arrives.

    The requests to one model share their cached prefix, and a cache entry belongs
    to one model. So the first request to each model goes out alone, writes the
    cache, and the model's other requests go out once it has returned, to read
    the cache. The models' requests run concurrently and share
    DEFAULT_IMMEDIATE_WORKERS request slots. An exception from *on_result*, such
    as for an invalid request, cancels the outstanding requests and propagates
    in an ExceptionGroup.
    """
    by_model: dict[str, list[LlmRequest]] = defaultdict(list)
    for request in requests:
        by_model[request.params["model"]].append(request)
    slots = asyncio.Semaphore(DEFAULT_IMMEDIATE_WORKERS)

    async def send(request: LlmRequest) -> None:
        async with slots:
            (result,) = await transport.run(STAGE, 1, [request])
        on_result(result)

    async with asyncio.TaskGroup() as tasks:

        async def send_model_group(first: LlmRequest, rest: list[LlmRequest]) -> None:
            await send(first)
            for request in rest:
                tasks.create_task(send(request))

        for first, *rest in by_model.values():
            tasks.create_task(send_model_group(first, rest))


async def run_attempt(
    transport: Transport,
    requests: Sequence[LlmRequest],
    db_path: Path,
    validator: jsonschema.protocols.Validator,
    name_to_id: dict[str, int],
) -> MatchOutcome:
    """Send *requests* with :func:`send_primed`, store each result as it arrives, count outcomes."""
    outcome = MatchOutcome()
    with Progress() as progress:
        task = progress.add_task("Matching associations", total=len(requests))

        def on_result(result: LlmResult) -> None:
            outcome.stored += handle_result(result, db_path, validator, name_to_id)
            outcome.to_fallback += result.goes_to_fallback
            progress.advance(task)

        await send_primed(transport, requests, on_result)
    return outcome


async def match_associations(
    *,
    transport: Transport,
    db_path: Path,
    schema: dict[str, Any],
    system: str,
    user_template: str,
    name_to_id: dict[str, int],
    max_retries: int,
    failures_since: datetime,
) -> None:
    """Match every association due matches, in attempts.

    An attempt makes progress when it stores matches or sends an association on
    to FALLBACK_MODEL, so a round of refusals by MODEL is followed by the attempt
    that sends those associations to FALLBACK_MODEL. Failed answers count from
    *failures_since*.
    """

    def load_history() -> StageHistory:
        with closing(sqlite3.connect(db_path)) as conn:
            return stage_history(conn, STAGE, failures_since=failures_since)

    validator = jsonschema.Draft202012Validator(schema)
    output_config = json_output_config(schema, EFFORT)
    log_failed_for_good(load_history().failures, STAGE)
    for attempt in range(1, max_retries + 1):
        history = load_history()
        associations = select_associations(db_path, history)
        if not associations:
            logger.info("No associations left to match")
            return
        logger.info("Attempt %d: matching %d associations", attempt, len(associations))
        requests = [
            build_request(
                association,
                system,
                user_template,
                output_config,
                history.model_for(str(association.id)),
            )
            for association in associations
        ]
        outcome = await run_attempt(transport, requests, db_path, validator, name_to_id)
        logger.info(
            "Attempt %d: stored %d of %d; %d go to the fallback model",
            attempt,
            outcome.stored,
            len(associations),
            outcome.to_fallback,
        )
        if outcome.stored + outcome.to_fallback == 0:
            logger.error("No progress in attempt %d - stopping", attempt)
            return


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database",
    ),
    panel_date: str = typer.Option(
        ...,
        "--panel-date",
        help="Panel state date (YYYY-MM-DD) for the panel descriptions",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of attempts for associations whose matching failed or was "
        "refused; the fallback-model request after a refusal is one of them",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Send associations whose answers failed for good in earlier invocations (rejected "
        "or cut off at max_tokens, MAX_FAILED_ANSWERS times) again, until they fail that often "
        "in this invocation",
    ),
) -> None:
    """Match gene-disease-MoI associations to diagnostic panels."""
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        raise typer.Exit(1)
    # With --retry-failed, only failed answers during this invocation count.
    failures_since = datetime.now(UTC) if retry_failed else ALL_TIME

    template = PROMPT_PATH.read_text()
    schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())

    logger.info("Fetching panel descriptions from PanelApp...")
    panels = PanelAppClient(panel_date).get_all_panel_descriptions()
    logger.info("Fetched %d panels", len(panels))
    name_to_id = {info["name"].lower(): panel_id for panel_id, info in panels.items()}
    system, user_template = split_prompt(template, format_all_panels_for_prompt(panels))

    logger.info("%s associations without panel matches", f"{count_unmatched(db_path):,}")

    async def run() -> None:
        await match_associations(
            transport=ImmediateTransport(make_client()),
            db_path=db_path,
            schema=schema,
            system=system,
            user_template=user_template,
            name_to_id=name_to_id,
            max_retries=max_retries,
            failures_since=failures_since,
        )

    asyncio.run(run())
    logger.info("%s associations still without panel matches", f"{count_unmatched(db_path):,}")
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
