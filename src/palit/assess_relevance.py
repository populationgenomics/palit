#!/usr/bin/env python3
"""Assess paper relevance from title and abstract with Claude, in two levels.

1. Scope screen (stage ``relevance``), on every paper: does the paper report
   new patients with pathogenic variants, or functional tests of disease
   variants, for a specific gene within scope? It lists the genes, and does
   not judge novelty.
2. PanelApp check (stage ``relevance_panelapp``), on papers the screen passes
   with at least one gene: compared with the genes' PanelApp entries on the
   reference panels, could the evidence change the curation?

A paper the screen passes without naming any gene is relevant without the
check, and with ``--screen-only`` the screen decides alone (for building a
comprehensive repository, such as the retrospective baseline). The screen
result is stored as soon as it arrives, so an interrupted run resumes with
the check. Gene mentions record the screen's genes for every paper it passes.
"""

import asyncio
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import typer
from anthropic.types.output_config_param import OutputConfigParam

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
    cached_system,
    json_output_config,
    make_client,
    parse_json_output,
    record_result,
)
from palit.llm_usage import print_stage_summary
from palit.panelapp_check import (
    CuratedRecord,
    check_user_text,
    is_relevant,
    load_check_system,
    with_hgnc_ids,
)
from palit.panelapp_client import PanelAppClient, format_panel_for_prompt
from palit.panelapp_integration import TARGET_PANEL_IDS

app = typer.Typer(help="Assess paper relevance from title and abstract")
logger = logging.getLogger(__name__)

STAGE = "relevance"
EFFORT: Effort = "low"
MAX_TOKENS = 8000

CHECK_STAGE = "relevance_panelapp"
CHECK_EFFORT: Effort = "medium"
CHECK_MAX_TOKENS = 16000

# Maximum abstract length in characters (~2500 tokens at 4 chars/token).
# Legitimate abstracts rarely exceed 5000 chars; this handles edge cases like
# taxonomic papers that dump species lists into the abstract.
MAX_ABSTRACT_CHARS = 10000

# The prompt files end with an input section ("** Input Paper **" or "**Input Paper**")
# holding the {title} and {abstract} placeholders. Everything before it is the
# shared, cacheable system prompt.
_INPUT_HEADING = re.compile(r"^\*\*\s*Input Paper\s*\*\*\s*$", re.MULTILINE)


@dataclass(frozen=True)
class RelevancePrompt:
    system: str
    user_template: str  # str.format template with {title} and {abstract}


@dataclass(frozen=True)
class CheckSettings:
    """Everything the PanelApp check needs besides the paper."""

    system: str
    output_config: OutputConfigParam
    validator: jsonschema.protocols.Validator
    record: CuratedRecord
    resolver: HgncResolver


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
    paper: dict[str, Any], prompt: RelevancePrompt, output_config: OutputConfigParam
) -> LlmRequest:
    """The scope-screen request for one paper."""
    user_text = prompt.user_template.format(
        title=paper["title"], abstract=_truncated_abstract(paper["doi"], paper["abstract"])
    )
    return LlmRequest(
        subject=paper["doi"],
        params={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": cached_system(prompt.system),
            "messages": [{"role": "user", "content": user_text}],
            "output_config": output_config,
        },
    )


def build_check_request(paper: dict[str, Any], check: CheckSettings) -> LlmRequest:
    """The PanelApp-check request for one screened paper."""
    screen = json.loads(paper["relevance_screen_json"])
    user_text = check_user_text(
        title=paper["title"],
        abstract=_truncated_abstract(paper["doi"], paper["abstract"]),
        pmid=paper["pmid"],
        doi=paper["doi"],
        screen_associations=screen["associations"],
        resolver=check.resolver,
        record=check.record,
    )
    return LlmRequest(
        subject=paper["doi"],
        params={
            "model": MODEL,
            "max_tokens": CHECK_MAX_TOKENS,
            "system": cached_system(check.system),
            "messages": [{"role": "user", "content": user_text}],
            "output_config": check.output_config,
        },
    )


def select_papers(db_path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Papers without a screen or assessment, excluding refused ones and ones in flight."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT doi, title, abstract
            FROM papers p
            WHERE relevance_assessment_json IS NULL
              AND relevance_screen_json IS NULL
              AND title IS NOT NULL
              AND abstract IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM llm_requests r
                  WHERE r.stage = ? AND r.subject = p.doi AND r.status IN ('refused', 'pending')
              )
            ORDER BY doi
            LIMIT ?
            """,
            (STAGE, -1 if limit is None else limit),
        ).fetchall()
    return [dict(row) for row in rows]


def select_screened(db_path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Screened papers awaiting the PanelApp check, excluding refused ones and ones in flight."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT doi, pmid, title, abstract, relevance_screen_json
            FROM papers p
            WHERE relevance_assessment_json IS NULL
              AND relevance_screen_json IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM llm_requests r
                  WHERE r.stage = ? AND r.subject = p.doi AND r.status IN ('refused', 'pending')
              )
            ORDER BY doi
            LIMIT ?
            """,
            (CHECK_STAGE, -1 if limit is None else limit),
        ).fetchall()
    return [dict(row) for row in rows]


@dataclass
class StoreOutcome:
    stored: int = 0
    refused: int = 0
    failed: int = 0


def _parsed_outputs(
    conn: sqlite3.Connection,
    results: list[LlmResult],
    validator: jsonschema.protocols.Validator,
    outcome: StoreOutcome,
) -> list[tuple[LlmResult, dict[str, Any]]]:
    """Record every result; return the valid outputs and count the rest in *outcome*.

    A request that failed or produced invalid output leaves its paper unchanged,
    so the next attempt selects it again. Invalid requests are a bug and raise.
    """
    valid: list[tuple[LlmResult, dict[str, Any]]] = []
    invalid_requests: list[str] = []
    for result in results:
        record_result(conn, result)
        if result.status == ResultStatus.REFUSED:
            assert result.message is not None
            outcome.refused += 1
            logger.warning(
                "Refused: %s %s (%s)",
                result.stage,
                result.subject,
                result.message.stop_details.category
                if result.message.stop_details
                else "no category",
            )
            continue
        if result.status != ResultStatus.SUCCEEDED:
            outcome.failed += 1
            if result.error_type == "invalid_request_error":
                invalid_requests.append(result.subject)
            continue
        assert result.message is not None
        try:
            parsed = parse_json_output(result.message)
            validator.validate(parsed)
        except (ValueError, jsonschema.ValidationError) as e:
            logger.warning("Invalid %s output for %s: %s", result.stage, result.subject, e)
            outcome.failed += 1
            continue
        valid.append((result, parsed))
    if invalid_requests:
        raise RuntimeError(
            f"{len(invalid_requests)} {results[0].stage} requests were invalid "
            f"(first: {invalid_requests[0]}); see llm_requests.error_type"
        )
    return valid


def store_screens(
    db_path: Path,
    results: list[LlmResult],
    validator: jsonschema.protocols.Validator,
    resolver: HgncResolver,
    *,
    screen_only: bool,
) -> StoreOutcome:
    """Store each valid screen; finalise the papers that need no PanelApp check."""
    outcome = StoreOutcome()
    with sqlite3.connect(db_path) as conn:
        for result, screen in _parsed_outputs(conn, results, validator, outcome):
            assert result.message is not None
            doi = result.subject
            raw = result.message.to_json()
            if not screen_only and screen["relevant"] and screen["associations"]:
                conn.execute(
                    "UPDATE papers SET relevance_screen_json = ?, relevance_screen_raw = ? WHERE doi = ?",
                    (json.dumps(screen), raw, doi),
                )
            else:
                _store_final(
                    conn,
                    doi,
                    relevant=screen["relevant"],
                    screen=screen,
                    raw={"screen": json.loads(raw), "panelapp_check": None},
                    check=None,
                    resolver=resolver,
                )
            outcome.stored += 1
    return outcome


def store_checks(db_path: Path, results: list[LlmResult], check: CheckSettings) -> StoreOutcome:
    """Finalise each paper whose PanelApp check came back valid."""
    outcome = StoreOutcome()
    with sqlite3.connect(db_path) as conn:
        for result, parsed in _parsed_outputs(conn, results, check.validator, outcome):
            assert result.message is not None
            doi = result.subject
            if not parsed["associations"]:
                logger.warning("PanelApp check for %s returned no associations", doi)
                outcome.failed += 1
                continue
            screen_json, screen_raw = conn.execute(
                "SELECT relevance_screen_json, relevance_screen_raw FROM papers WHERE doi = ?",
                (doi,),
            ).fetchone()
            associations = with_hgnc_ids(parsed["associations"], check.resolver)
            panelapp_check = {
                "panel_date": check.record.panel_date,
                "reference_panel_ids": list(check.record.reference_panel_ids),
                "rationale": parsed["rationale"],
                "associations": associations,
            }
            _store_final(
                conn,
                doi,
                relevant=is_relevant(associations),
                screen=json.loads(screen_json),
                raw={
                    "screen": json.loads(screen_raw),
                    "panelapp_check": json.loads(result.message.to_json()),
                },
                check=panelapp_check,
                resolver=check.resolver,
            )
            outcome.stored += 1
    return outcome


def _store_final(
    conn: sqlite3.Connection,
    doi: str,
    *,
    relevant: bool,
    screen: dict[str, Any],
    raw: dict[str, Any],
    check: dict[str, Any] | None,
    resolver: HgncResolver,
) -> None:
    assessment = {"relevant": relevant, "screen": screen, "panelapp_check": check}
    conn.execute(
        """
        UPDATE papers
        SET relevance_assessment_raw = ?,
            relevance_assessment_json = ?,
            relevance_screen_raw = NULL,
            relevance_screen_json = NULL,
            download_status = CASE
                WHEN ? = 1 AND download_status IS NULL THEN 'scheduled'
                ELSE download_status
            END
        WHERE doi = ?
        """,
        (json.dumps(raw), json.dumps(assessment), 1 if relevant else 0, doi),
    )
    if not screen["relevant"]:
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
    check: CheckSettings | None,
    limit: int | None,
    max_retries: int,
) -> None:
    """Screen and check papers; *check* None means the screen decides alone."""
    validator = jsonschema.Draft202012Validator(schema)
    output_config = json_output_config(schema, EFFORT)
    screen_only = check is None

    def store(results: list[LlmResult]) -> StoreOutcome:
        return store_screens(db_path, results, validator, resolver, screen_only=screen_only)

    resumed = await transport.resume(STAGE)
    if resumed:
        logger.info("Collected %d earlier screens: %s", len(resumed), store(resumed))
    resumed = await transport.resume(CHECK_STAGE)
    if resumed:
        if check is None:
            raise RuntimeError(
                f"{len(resumed)} PanelApp checks from an earlier run are uncollected; "
                "collect them with a run without --screen-only"
            )
        logger.info(
            "Collected %d earlier checks: %s", len(resumed), store_checks(db_path, resumed, check)
        )

    for attempt in range(1, max_retries + 1):
        progress = 0
        papers = select_papers(db_path, limit)
        if papers:
            logger.info("Attempt %d: screening %d papers", attempt, len(papers))
            requests = [build_request(paper, prompt, output_config) for paper in papers]
            outcome = store(await transport.run(STAGE, 1, requests))
            logger.info("Attempt %d screen: %s", attempt, outcome)
            progress += outcome.stored
        screened = [] if check is None else select_screened(db_path, limit)
        if check is not None and screened:
            logger.info("Attempt %d: checking %d papers against PanelApp", attempt, len(screened))
            requests = [build_check_request(paper, check) for paper in screened]
            outcome = store_checks(db_path, await transport.run(CHECK_STAGE, 2, requests), check)
            logger.info("Attempt %d PanelApp check: %s", attempt, outcome)
            progress += outcome.stored
        if not papers and not screened:
            logger.info("No papers left to assess")
            return
        if progress == 0:
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
    panel_date: str | None = typer.Option(
        None,
        "--panel-date",
        help="PanelApp snapshot date (YYYY-MM-DD) for the PanelApp check and the scope panel; "
        "required unless --screen-only without --scope-panel-id",
    ),
    screen_only: bool = typer.Option(
        False,
        "--screen-only",
        help="Let the scope screen decide alone, without the PanelApp check "
        "(for a comprehensive repository such as the retrospective baseline)",
    ),
    target_panel_ids: list[int] | None = typer.Option(
        None,
        "--target-panel-id",
        help="Reference panels for the PanelApp check; repeatable. Defaults to TARGET_PANEL_IDS.",
    ),
    scope_panel_id: int | None = typer.Option(
        None,
        "--scope-panel-id",
        help="Panel-scoped assessment: the screen gets the panel description, and the "
        "PanelApp check compares against this panel only",
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
    check_prompt_path: Path = typer.Option(
        Path("prompts/relevance_panelapp_check_prompt.txt"),
        "--check-prompt-path",
        help="Path to the PanelApp-check system prompt",
    ),
    check_schema_path: Path = typer.Option(
        Path("prompts/relevance_panelapp_check_schema.json"),
        "--check-schema-path",
        help="Path to the PanelApp-check response schema",
    ),
    immediate: bool = typer.Option(
        False,
        "--immediate",
        help="Send requests immediately instead of as Message Batches (for prompt development)",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Assess at most this many papers per level, in one attempt (for prompt development)",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of attempts for papers whose request failed",
    ),
) -> None:
    """Assess the relevance of every paper that has no assessment yet."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)
    if scope_panel_id is not None and target_panel_ids:
        logger.error("--scope-panel-id and --target-panel-id are mutually exclusive")
        raise typer.Exit(1)
    if panel_date is None and (scope_panel_id is not None or not screen_only):
        logger.error("--panel-date is required for the PanelApp check and with --scope-panel-id")
        raise typer.Exit(1)

    panel_client = PanelAppClient(panel_date) if panel_date is not None else None
    panel_description = None
    if scope_panel_id is not None:
        assert panel_client is not None
        try:
            panel_info = panel_client.get_panel_data(scope_panel_id)
        except ValueError as e:
            logger.error(f"Panel {scope_panel_id} not found in PanelApp data for {panel_date}")
            raise typer.Exit(1) from e
        panel_description = format_panel_for_prompt(scope_panel_id, panel_info)
        logger.info(f"Panel-scoped mode: {panel_info.get('name', 'Unknown')}")
        reference_panel_ids = [scope_panel_id]
    else:
        reference_panel_ids = target_panel_ids or TARGET_PANEL_IDS

    prompt = load_prompt(prompt_path, panel_description)
    schema: dict[str, Any] = json.loads(schema_path.read_text())
    resolver = HgncResolver.from_file()
    check = None
    if not screen_only:
        assert panel_client is not None and panel_date is not None
        record = CuratedRecord.build(
            panel_client.get_all_panel_data(),
            panel_date,
            reference_panel_ids,
            incidentalome_fallback=scope_panel_id is None,
        )
        check_schema: dict[str, Any] = json.loads(check_schema_path.read_text())
        check = CheckSettings(
            system=load_check_system(check_prompt_path, record),
            output_config=json_output_config(check_schema, CHECK_EFFORT),
            validator=jsonschema.Draft202012Validator(check_schema),
            record=record,
            resolver=resolver,
        )

    logger.info(f"{count_remaining(db_path):,} papers without a relevance assessment")

    async def run() -> None:
        client = make_client(AnthropicSettings())
        transport: Transport = (
            ImmediateTransport(client) if immediate else BatchTransport(client, db_path)
        )
        await _process_relevance(
            transport=transport,
            db_path=db_path,
            prompt=prompt,
            schema=schema,
            resolver=resolver,
            check=check,
            limit=limit,
            max_retries=max_retries,
        )

    asyncio.run(run())
    logger.info(f"{count_remaining(db_path):,} papers still without a relevance assessment")
    print_stage_summary(db_path, STAGE)
    if check is not None:
        print_stage_summary(db_path, CHECK_STAGE)


if __name__ == "__main__":
    app()
