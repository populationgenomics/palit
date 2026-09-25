#!/usr/bin/env python3
"""Assess paper relevance from title and abstract with Claude."""

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
from palit.panelapp_client import PanelAppClient, format_panel_for_prompt

app = typer.Typer(help="Assess paper relevance from title and abstract")
logger = logging.getLogger(__name__)

STAGE = "relevance"
EFFORT: Effort = "low"
MAX_TOKENS = 8000

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


def build_request(
    paper: dict[str, Any], prompt: RelevancePrompt, output_config: OutputConfigParam
) -> LlmRequest:
    abstract: str = paper["abstract"]
    if len(abstract) > MAX_ABSTRACT_CHARS:
        logger.warning(
            f"Truncating abstract for DOI {paper['doi']} from {len(abstract)} to {MAX_ABSTRACT_CHARS} chars"
        )
        abstract = abstract[:MAX_ABSTRACT_CHARS] + "... [truncated]"
    user_text = prompt.user_template.format(title=paper["title"], abstract=abstract)
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


def select_papers(db_path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Papers without an assessment, excluding refused ones and ones in flight."""
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
                  WHERE r.stage = ? AND r.subject = p.doi AND r.status IN ('refused', 'pending')
              )
            ORDER BY doi
            LIMIT ?
            """,
            (STAGE, -1 if limit is None else limit),
        ).fetchall()
    return [dict(row) for row in rows]


@dataclass
class StoreOutcome:
    stored: int = 0
    refused: int = 0
    failed: int = 0


def store_results(
    db_path: Path,
    results: list[LlmResult],
    validator: jsonschema.protocols.Validator,
    hgnc_resolver: HgncResolver,
) -> StoreOutcome:
    """Record every result; store valid assessments and their gene mentions.

    A paper whose request failed or whose output is invalid keeps a NULL
    assessment, so the next attempt selects it again.
    """
    outcome = StoreOutcome()
    invalid_requests: list[str] = []
    with sqlite3.connect(db_path) as conn:
        for result in results:
            record_result(conn, result)
            if result.status == ResultStatus.REFUSED:
                assert result.message is not None
                outcome.refused += 1
                logger.warning(
                    "Refused: %s (%s)",
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
                assessment = parse_json_output(result.message)
                validator.validate(assessment)
            except (ValueError, jsonschema.ValidationError) as e:
                logger.warning("Invalid assessment for %s: %s", result.subject, e)
                outcome.failed += 1
                continue
            _store_assessment(conn, result, assessment, hgnc_resolver)
            outcome.stored += 1
    if invalid_requests:
        raise RuntimeError(
            f"{len(invalid_requests)} relevance requests were invalid (first: {invalid_requests[0]}); "
            "see llm_requests.error_type"
        )
    return outcome


def _store_assessment(
    conn: sqlite3.Connection,
    result: LlmResult,
    assessment: dict[str, Any],
    hgnc_resolver: HgncResolver,
) -> None:
    assert result.message is not None
    doi = result.subject
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
        (result.message.to_json(), json.dumps(assessment), 1 if assessment["relevant"] else 0, doi),
    )
    if not assessment["relevant"]:
        return
    for association in assessment["associations"]:
        paper_gene_symbol: str = association["gene_symbol"]
        entry = hgnc_resolver.resolve(paper_gene_symbol)
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
    hgnc_resolver: HgncResolver,
    limit: int | None,
    max_retries: int,
) -> None:
    validator = jsonschema.Draft202012Validator(schema)
    output_config = json_output_config(schema, EFFORT)

    resumed = await transport.resume(STAGE)
    if resumed:
        outcome = store_results(db_path, resumed, validator, hgnc_resolver)
        logger.info("Collected %d results from earlier batches: %s", len(resumed), outcome)

    for attempt in range(1, max_retries + 1):
        papers = select_papers(db_path, limit)
        if not papers:
            logger.info("No papers left to assess")
            return
        logger.info("Attempt %d: assessing %d papers", attempt, len(papers))
        requests = [build_request(paper, prompt, output_config) for paper in papers]
        results = await transport.run(STAGE, 1, requests)
        outcome = store_results(db_path, results, validator, hgnc_resolver)
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
    panel_date: str | None = typer.Option(
        None,
        "--panel-date",
        help="Panel state at date (YYYY-MM-DD); required with --scope-panel-id",
    ),
    scope_panel_id: int | None = typer.Option(
        None,
        "--scope-panel-id",
        help="Panel ID for panel-scoped relevance assessment (injects the panel description into the prompt)",
    ),
    prompt_path: Path = typer.Option(
        Path("prompts/relevance_assessment_prompt.txt"),
        "--prompt-path",
        "-p",
        help="Path to prompt template file",
    ),
    schema_path: Path = typer.Option(
        Path("prompts/relevance_assessment_schema.json"),
        "--schema-path",
        "-s",
        help="Path to response schema file",
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
        help="Maximum number of attempts for papers whose request failed",
    ),
) -> None:
    """Assess the relevance of every paper that has no assessment yet."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    panel_description = None
    if scope_panel_id is not None:
        if panel_date is None:
            logger.error("--scope-panel-id needs --panel-date")
            raise typer.Exit(1)
        panel_client = PanelAppClient(panel_date)
        try:
            panel_info = panel_client.get_panel_data(scope_panel_id)
        except ValueError as e:
            logger.error(f"Panel {scope_panel_id} not found in PanelApp data for {panel_date}")
            raise typer.Exit(1) from e
        panel_description = format_panel_for_prompt(scope_panel_id, panel_info)
        logger.info(f"Panel-scoped mode: {panel_info.get('name', 'Unknown')}")

    prompt = load_prompt(prompt_path, panel_description)
    schema: dict[str, Any] = json.loads(schema_path.read_text())
    hgnc_resolver = HgncResolver.from_file()

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
            hgnc_resolver=hgnc_resolver,
            limit=limit,
            max_retries=max_retries,
        )

    asyncio.run(run())
    logger.info(f"{count_remaining(db_path):,} papers still without a relevance assessment")
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
