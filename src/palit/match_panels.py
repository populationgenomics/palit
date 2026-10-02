#!/usr/bin/env python3
"""Match each gene-disease-MoI association to diagnostic panels.

One request per association without panel matches: the system part lists every
panel and is the same for all requests, so it is cached; the user part carries
the association's disease, mode of inheritance and summary. Matches are stored
on the association row, in the transaction that records the request.
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jsonschema
import typer
from anthropic.types.output_config_param import OutputConfigParam

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


def format_all_panels_for_prompt(panels: dict[int, dict[str, Any]]) -> str:
    """All panel descriptions, ordered by panel id."""
    return "\n".join(
        format_panel_for_prompt(panel_id, info) for panel_id, info in sorted(panels.items())
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def select_associations(db_path: Path) -> list[Association]:
    """Associations without panel matches, except refused ones and ones in flight."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.assessment_json
            FROM associations a
            WHERE a.matched_panels_json IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM llm_requests r
                  WHERE r.stage = ? AND r.subject = CAST(a.id AS TEXT)
                    AND r.status IN ('refused', 'pending')
              )
            ORDER BY a.id
            """,
            (STAGE,),
        ).fetchall()
    associations = []
    for association_id, assessment_json in rows:
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
            "SELECT COUNT(*) FROM associations WHERE matched_panels_json IS NULL"
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
    association: Association, system: str, user_template: str, output_config: OutputConfigParam
) -> LlmRequest:
    user_text = user_template.format(
        description=association.description,
        inheritance=format_inheritance(association),
        summary=association.summary,
    )
    return LlmRequest(
        subject=str(association.id),
        params={
            "model": MODEL,
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


def handle_results(
    results: list[LlmResult],
    db_path: Path,
    validator: jsonschema.protocols.Validator,
    name_to_id: dict[str, int],
) -> int:
    """Record every result and store valid matches; returns the number stored."""
    stored = 0
    with sqlite3.connect(db_path) as conn:
        for result in results:
            record_result(conn, result)
            association_id = int(result.subject)
            message = result.message
            if result.status == ResultStatus.REFUSED:
                logger.warning("Refused: association %d", association_id)
                continue
            if result.status != ResultStatus.SUCCEEDED or message is None:
                if result.error_type == "invalid_request_error":
                    raise RuntimeError(
                        f"invalid match-panels request for association {association_id}"
                    )
                continue
            try:
                answer = parse_json_output(message)
                validator.validate(answer)
                matches = parse_matches(answer, name_to_id)
            except (ValueError, jsonschema.ValidationError) as e:
                logger.warning("Invalid panel matching for association %d: %s", association_id, e)
                continue
            if store_matched_panels(conn, association_id, matches, message.to_json()):
                stored += 1
    return stored


async def _process_panel_matching(
    *,
    transport: Transport,
    db_path: Path,
    schema: dict[str, Any],
    system: str,
    user_template: str,
    name_to_id: dict[str, int],
    max_retries: int,
) -> None:
    validator = jsonschema.Draft202012Validator(schema)
    output_config = json_output_config(schema, EFFORT)
    resumed = await transport.resume(STAGE)
    if resumed:
        stored = handle_results(resumed, db_path, validator, name_to_id)
        logger.info("Collected %d results from earlier batches, stored %d", len(resumed), stored)
    for attempt in range(1, max_retries + 1):
        associations = select_associations(db_path)
        if not associations:
            logger.info("No associations left to match")
            return
        logger.info("Attempt %d: matching %d associations", attempt, len(associations))
        requests = [
            build_request(association, system, user_template, output_config)
            for association in associations
        ]
        results = await transport.run(STAGE, 1, requests)
        stored = handle_results(results, db_path, validator, name_to_id)
        logger.info("Attempt %d: stored %d of %d", attempt, stored, len(associations))
        if stored == 0:
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
        help="Maximum number of attempts for associations whose matching failed",
    ),
    immediate: bool = typer.Option(
        False,
        "--immediate",
        help="Send requests immediately instead of as Message Batches (for prompt development)",
    ),
) -> None:
    """Match gene-disease-MoI associations to diagnostic panels."""
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        raise typer.Exit(1)

    template = PROMPT_PATH.read_text()
    schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())

    logger.info("Fetching panel descriptions from PanelApp...")
    panels = PanelAppClient(panel_date).get_all_panel_descriptions()
    logger.info("Fetched %d panels", len(panels))
    name_to_id = {info["name"].lower(): panel_id for panel_id, info in panels.items()}
    system, user_template = split_prompt(template, format_all_panels_for_prompt(panels))

    logger.info("%s associations without panel matches", f"{count_unmatched(db_path):,}")

    async def run() -> None:
        client = make_client(AnthropicSettings())
        transport: Transport = (
            ImmediateTransport(client) if immediate else BatchTransport(client, db_path)
        )
        await _process_panel_matching(
            transport=transport,
            db_path=db_path,
            schema=schema,
            system=system,
            user_template=user_template,
            name_to_id=name_to_id,
            max_retries=max_retries,
        )

    asyncio.run(run())
    logger.info("%s associations still without panel matches", f"{count_unmatched(db_path):,}")
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
