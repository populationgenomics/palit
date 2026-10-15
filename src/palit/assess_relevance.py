#!/usr/bin/env python3
"""Assess paper relevance from title and abstract with Claude: the scope screen.

The screen (stage ``relevance``) runs on every paper with a title and abstract:
does the paper report new patients with pathogenic variants, or functional
tests of disease variants, for a specific gene within scope? Its decision is
the paper's relevance, which decides only whether the paper gets a full-text
read. It does not judge whether an association is already known or curated.
It lists the paper's gene-disease associations, and gene mentions record their
genes for every paper it passes. With ``--scope-panel-id`` the prompt carries
the panel's description, and the screen asks whether the paper might matter
for that panel.

The screen runs on the model ``--model`` picks (:data:`DEFAULT_MODEL` by
default). A paper refused by a model with a fallback goes to that fallback in
the next attempt (see :data:`palit.llm.FALLBACKS`). A refusal by a model
without a fallback is final.

A paper refused for good is settled as not relevant, so that the ledger does
not ingest it again. Its result records the refusal (see
:func:`refused_assessment`)::

    {"relevant": false, "screen": null,
     "refused": {"model": "claude-sonnet-5-5", "category": "bio"}}

Each invocation first settles the papers refused for good that still have no
result, as a run database's recorded refusals may lack one. ``--retry-refused``
instead reopens every settled refusal, and the papers refused for good again
are settled again.

A paper whose answers failed for good (see :mod:`palit.llm`) is left out and
stays without a result; ``--retry-failed`` sends it again.
"""

import asyncio
import json
import logging
import re
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import jsonschema
import typer
from anthropic.types.output_config_param import OutputConfigParam

from palit.hgnc import HgncResolver
from palit.llm import (
    ALL_TIME,
    FALLBACK_MODEL,
    FALLBACKS,
    LIGHT_MODEL,
    MODEL,
    BatchTransport,
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

app = typer.Typer(help="Assess paper relevance from title and abstract")
logger = logging.getLogger(__name__)

STAGE = "relevance"
EFFORT: Effort = "medium"
MAX_TOKENS = 8000

# Maximum abstract length in characters (~2500 tokens at 4 chars/token).
# Legitimate abstracts rarely exceed 5000 chars; this handles edge cases like
# taxonomic papers that dump species lists into the abstract.
MAX_ABSTRACT_CHARS = 10000

# The prompt files end with an input section ("** Input Paper **" or "**Input Paper**")
# holding the {title} and {abstract} placeholders. Everything before it is the
# shared, cacheable system prompt.
_INPUT_HEADING = re.compile(r"^\*\*\s*Input Paper\s*\*\*\s*$", re.MULTILINE)


class RelevanceModel(StrEnum):
    """The models the screen can run on. Only MODEL has a fallback (FALLBACK_MODEL); the
    others' refusals are final."""

    OPUS = MODEL
    SONNET = FALLBACK_MODEL
    HAIKU = LIGHT_MODEL


DEFAULT_MODEL = RelevanceModel.OPUS


@dataclass(frozen=True)
class Refusal:
    """The final refusal that settled a paper."""

    model: str
    category: str | None  # the safety classifier's category, if the refusal names one


def refused_assessment(refusal: Refusal) -> dict[str, Any]:
    """The settled result of a paper refused for good: not relevant, without a screen."""
    return {"relevant": False, "screen": None, "refused": asdict(refusal)}


@dataclass(frozen=True)
class RelevancePrompt:
    system: str
    user_template: str  # str.format template with {title} and {abstract}


def load_prompt(prompt_path: Path, panel_description: str | None) -> RelevancePrompt:
    """Split a relevance prompt file into its system prompt and per-paper template."""
    template = prompt_path.read_text()
    match = _INPUT_HEADING.search(template)
    if match is None:
        raise ValueError(f"{prompt_path} has no '** Input Paper **' heading")
    system_template = template[: match.start()]
    # format() also unescapes the prompt's literal {{ }} braces.
    if "{panel_description}" in system_template:
        if panel_description is None:
            raise ValueError(f"{prompt_path} needs a panel description (--scope-panel-id)")
        system = system_template.format(panel_description=panel_description)
    else:
        if panel_description is not None:
            raise ValueError(
                f"{prompt_path} has no {{panel_description}} placeholder for --scope-panel-id; "
                "use prompts/panel_relevance_assessment_prompt.txt"
            )
        system = system_template.format()
    return RelevancePrompt(system=system.rstrip(), user_template=template[match.start() :])


def _truncated_abstract(doi: str, abstract: str) -> str:
    if len(abstract) <= MAX_ABSTRACT_CHARS:
        return abstract
    logger.warning(
        f"Truncating abstract for DOI {doi} from {len(abstract)} to {MAX_ABSTRACT_CHARS} chars"
    )
    return abstract[:MAX_ABSTRACT_CHARS] + "... [truncated]"


def build_request(
    paper: dict[str, Any], prompt: RelevancePrompt, output_config: OutputConfigParam, model: str
) -> LlmRequest:
    """The scope-screen request for one paper."""
    user_text = prompt.user_template.format(
        title=paper["title"], abstract=_truncated_abstract(paper["doi"], paper["abstract"])
    )
    return LlmRequest(
        subject=paper["doi"],
        params={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": cached_system(prompt.system),
            "messages": [{"role": "user", "content": user_text}],
            "output_config": output_config,
        },
    )


def load_history(
    db_path: Path, refusals_since: datetime, failures_since: datetime, model: str
) -> StageHistory:
    with closing(sqlite3.connect(db_path)) as conn:
        return stage_history(conn, STAGE, refusals_since, failures_since, model)


def select_papers(db_path: Path, limit: int | None, history: StageHistory) -> list[dict[str, Any]]:
    """Papers without an assessment, except ones skipped for good and ones in flight."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT doi, title, abstract
            FROM papers p
            WHERE relevance_assessment_json IS NULL
              AND title IS NOT NULL
              AND abstract IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM llm_requests r
                  WHERE r.stage = ? AND r.subject = p.doi AND r.status = 'pending'
              )
            ORDER BY doi
            """,
            (STAGE,),
        ).fetchall()
    papers = [dict(row) for row in rows if not history.skipped(row["doi"])]
    return papers if limit is None else papers[:limit]


@dataclass
class StoreOutcome:
    stored: int = 0
    refused: int = 0
    to_fallback: int = 0  # refusals by a model with a fallback: these papers go to it next
    settled: int = 0  # final refusals: these papers are settled as not relevant
    failed: int = 0


def store_screens(
    db_path: Path,
    results: list[LlmResult],
    validator: jsonschema.protocols.Validator,
    resolver: HgncResolver,
) -> StoreOutcome:
    """Record every result, store each valid screen and settle each final refusal.

    A request that failed or produced invalid output leaves its paper unchanged,
    so the next attempt selects it again. Invalid requests are a bug and raise.
    """
    outcome = StoreOutcome()
    invalid_requests: list[str] = []
    with sqlite3.connect(db_path) as conn:
        for result in results:
            if result.status == ResultStatus.REFUSED:
                record_result(conn, result)
                outcome.refused += 1
                outcome.to_fallback += result.goes_to_fallback
                log_refusal(result, result.subject)
                if result.refused_for_good:
                    refusal = Refusal(result.model, result.refusal_category)
                    _store_refused(conn, result.subject, refusal)
                    outcome.settled += 1
                continue
            if result.status != ResultStatus.SUCCEEDED:
                record_result(conn, result)
                outcome.failed += 1
                if result.error_type == "invalid_request_error":
                    invalid_requests.append(result.subject)
                continue
            assert result.message is not None
            try:
                screen = parse_json_output(result.message)
                validator.validate(screen)
            except (ValueError, jsonschema.ValidationError) as e:
                rejection = invalid_answer_reason(e)
                logger.warning("Invalid screen for %s: %s", result.subject, rejection)
                record_result(conn, result, rejection=rejection)
                outcome.failed += 1
                continue
            record_result(conn, result)
            _store_screen(conn, result.subject, screen, result.message.to_json(), resolver)
            outcome.stored += 1
    if invalid_requests:
        raise RuntimeError(
            f"{len(invalid_requests)} {STAGE} requests were invalid "
            f"(first: {invalid_requests[0]}); see llm_requests.error_type"
        )
    return outcome


def _store_screen(
    conn: sqlite3.Connection,
    doi: str,
    screen: dict[str, Any],
    raw: str,
    resolver: HgncResolver,
) -> None:
    """Store a paper's screen as its relevance, and the gene mentions of a screen that
    passed it."""
    relevant: bool = screen["relevant"]
    conn.execute(
        """
        UPDATE papers
        SET relevance_assessment_raw = ?,
            relevance_assessment_json = ?,
            download_status = CASE
                WHEN ? = 1 AND download_status IS NULL THEN 'scheduled'
                ELSE download_status
            END
        WHERE doi = ?
        """,
        (raw, json.dumps({"relevant": relevant, "screen": screen}), 1 if relevant else 0, doi),
    )
    if not relevant:
        return
    for association in screen["associations"]:
        paper_gene_symbol: str = association["gene_symbol"]
        entry = resolver.resolve(paper_gene_symbol)
        if entry is None:
            logger.debug(f"Unresolved gene symbol '{paper_gene_symbol}' in DOI {doi}")
            continue
        conn.execute(
            """
            INSERT OR IGNORE INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source)
            VALUES (?, ?, ?, 'relevance_assessment')
            """,
            (entry.hgnc_id, paper_gene_symbol.upper(), doi),
        )


def _store_refused(conn: sqlite3.Connection, doi: str, refusal: Refusal) -> None:
    """Settle a paper refused for good as not relevant."""
    conn.execute(
        """
        UPDATE papers SET relevance_assessment_raw = NULL, relevance_assessment_json = ?
        WHERE doi = ?
        """,
        (json.dumps(refused_assessment(refusal)), doi),
    )


def settle_refused(db_path: Path) -> int:
    """Settle the papers refused for good that have no result yet, without API calls.

    These are the papers that selection skips because a model without a fallback
    refused them (see :func:`palit.llm.stage_refusals`), and that are not in
    flight. Each settled result carries the model and category of the paper's
    latest such refusal. Returns the number of papers settled.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT r.subject, r.model, r.refusal_category
            FROM llm_requests r
            JOIN papers p ON p.doi = r.subject
            WHERE r.stage = ? AND r.status = 'refused'
              AND r.model NOT IN ({", ".join("?" * len(FALLBACKS))})
              AND p.relevance_assessment_json IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM llm_requests q
                  WHERE q.stage = r.stage AND q.subject = r.subject AND q.status = 'pending'
              )
            ORDER BY r.completed_at
            """,
            (STAGE, *FALLBACKS),
        ).fetchall()
        # Later rows overwrite earlier ones, so each paper keeps its latest refusal.
        latest = {doi: Refusal(model, category) for doi, model, category in rows}
        for doi, refusal in latest.items():
            _store_refused(conn, doi, refusal)
    logger.info("Settled %d papers refused for good as not relevant", len(latest))
    return len(latest)


def reopen_refused(db_path: Path) -> int:
    """Remove the settled results of refused papers, so that selection includes them again."""
    with sqlite3.connect(db_path) as conn:
        reopened = conn.execute(
            """
            UPDATE papers
            SET relevance_assessment_json = NULL, relevance_assessment_raw = NULL
            WHERE json_extract(relevance_assessment_json, '$.refused') IS NOT NULL
            """
        ).rowcount
    logger.info("Reopened %d papers refused for good, to send them again", reopened)
    return reopened


def count_remaining(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        (remaining,) = conn.execute(
            """
            SELECT COUNT(*) FROM papers
            WHERE relevance_assessment_json IS NULL AND title IS NOT NULL AND abstract IS NOT NULL
            """
        ).fetchone()
    return int(remaining)


async def _process_relevance(
    *,
    transport: Transport,
    db_path: Path,
    prompt: RelevancePrompt,
    schema: dict[str, Any],
    resolver: HgncResolver,
    model: str,
    limit: int | None,
    max_retries: int,
    retry_refused: bool,
    retry_failed: bool,
) -> None:
    """Screen the papers without a result on *model*, in up to *max_retries* attempts.

    Each attempt selects the papers still without a result. An attempt makes
    progress when it stores or settles a result or sends a paper on to a
    fallback, so a round of refusals by a model with a fallback is followed by
    the attempt that sends those papers to the fallback.

    With *retry_refused*, only refusals during this invocation count, and the
    settled refusals are reopened first. The settle step at the end settles
    the reopened papers this invocation did not get to. With *retry_failed*,
    only failed answers during this invocation count.
    """
    started = datetime.now(UTC)
    refusals_since = started if retry_refused else ALL_TIME
    failures_since = started if retry_failed else ALL_TIME
    validator = jsonschema.Draft202012Validator(schema)
    output_config = json_output_config(schema, EFFORT)

    resumed = await transport.resume(STAGE)
    if resumed:
        outcome = store_screens(db_path, resumed, validator, resolver)
        logger.info("Collected %d earlier screens: %s", len(resumed), outcome)

    if retry_refused:
        reopen_refused(db_path)
    else:
        settle_refused(db_path)
    log_failed_for_good(
        load_history(db_path, refusals_since, failures_since, model).failures, STAGE
    )
    for attempt in range(1, max_retries + 1):
        history = load_history(db_path, refusals_since, failures_since, model)
        papers = select_papers(db_path, limit, history)
        if not papers:
            logger.info("No papers left to assess")
            break
        logger.info("Attempt %d: screening %d papers", attempt, len(papers))
        requests = [
            build_request(paper, prompt, output_config, history.model_for(paper["doi"]))
            for paper in papers
        ]
        outcome = store_screens(
            db_path, await transport.run(STAGE, 1, requests), validator, resolver
        )
        logger.info("Attempt %d: %s", attempt, outcome)
        if outcome.stored + outcome.to_fallback + outcome.settled == 0:
            logger.error("No progress in attempt %d - stopping", attempt)
            break
        if limit is not None:
            break
    settle_refused(db_path)


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database",
    ),
    scope_panel_id: int | None = typer.Option(
        None,
        "--scope-panel-id",
        help="Panel-scoped assessment: the screen gets this panel's description",
    ),
    panel_date: str | None = typer.Option(
        None,
        "--panel-date",
        help="PanelApp snapshot date (YYYY-MM-DD) of the scope panel; required with, and "
        "only used with, --scope-panel-id",
    ),
    prompt_path: Path = typer.Option(
        Path("prompts/relevance_assessment_prompt.txt"),
        "--prompt-path",
        "-p",
        help="Path to the scope-screen prompt template",
    ),
    schema_path: Path = typer.Option(
        Path("prompts/relevance_assessment_schema.json"),
        "--schema-path",
        "-s",
        help="Path to the scope-screen response schema",
    ),
    model: RelevanceModel = typer.Option(
        DEFAULT_MODEL,
        "--model",
        help="Model of the screen; only MODEL falls back (to FALLBACK_MODEL) on refusals, "
        "the others' refusals are final",
    ),
    immediate: bool = typer.Option(
        False,
        "--immediate",
        help="Send requests immediately instead of as Message Batches (for prompt development)",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Assess at most this many papers, in one attempt (for prompt development)",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of attempts for papers whose request failed or was refused; "
        "the fallback-model request after a refusal is one of them",
    ),
    retry_refused: bool = typer.Option(
        False,
        "--retry-refused",
        help="Send papers refused for good in earlier invocations again, once each, starting "
        "with --model and then its fallback, if it has one (refusals vary between calls); papers "
        "refused for good again stay settled as not relevant",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Send papers whose answers failed for good in earlier invocations (rejected or "
        "cut off at max_tokens, MAX_FAILED_ANSWERS times) again, until they fail that often "
        "in this invocation",
    ),
) -> None:
    """Assess the relevance of every paper that has no assessment yet."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)
    if (scope_panel_id is None) != (panel_date is None):
        logger.error("--scope-panel-id and --panel-date go together")
        raise typer.Exit(1)

    panel_description = None
    if scope_panel_id is not None:
        assert panel_date is not None
        try:
            panel_info = PanelAppClient(panel_date).get_panel_data(scope_panel_id)
        except ValueError as e:
            logger.error(f"Panel {scope_panel_id} not found in PanelApp data for {panel_date}")
            raise typer.Exit(1) from e
        panel_description = format_panel_for_prompt(scope_panel_id, panel_info)
        logger.info(f"Panel-scoped mode: {panel_info.get('name', 'Unknown')}")

    prompt = load_prompt(prompt_path, panel_description)
    schema: dict[str, Any] = json.loads(schema_path.read_text())
    resolver = HgncResolver.from_file()

    logger.info(f"{count_remaining(db_path):,} papers without a relevance assessment")
    logger.info(f"Assessing relevance with {model.value}")

    async def run() -> None:
        client = make_client()
        transport: Transport = (
            ImmediateTransport(client) if immediate else BatchTransport(client, db_path)
        )
        await _process_relevance(
            transport=transport,
            db_path=db_path,
            prompt=prompt,
            schema=schema,
            resolver=resolver,
            model=model.value,
            limit=limit,
            max_retries=max_retries,
            retry_refused=retry_refused,
            retry_failed=retry_failed,
        )

    asyncio.run(run())
    logger.info(f"{count_remaining(db_path):,} papers still without a relevance assessment")
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
