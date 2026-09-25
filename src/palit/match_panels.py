#!/usr/bin/env python3
"""Match genes to diagnostic panels based on phenotype descriptions."""

import asyncio
import json
import logging
import sqlite3
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

app = typer.Typer(help="Match genes to diagnostic panels based on phenotype descriptions")
logger = logging.getLogger(__name__)

STAGE = "match_panels"
EFFORT: Effort = "low"
MAX_TOKENS = 16000
GENE_CONTEXT_HEADING = "GENE CONTEXT:"


def format_all_panels_for_prompt(panels: dict[int, dict[str, Any]]) -> str:
    """Format multiple panel descriptions for LLM prompt.

    Args:
        panels: Dictionary mapping panel IDs to panel information

    Returns:
        Formatted panel descriptions string with all panels
    """
    formatted = []
    for panel_id, info in sorted(panels.items()):
        formatted.append(format_panel_for_prompt(panel_id, info))
    return "\n".join(formatted)


class PaperBatchProcessor:
    """Handle database operations for panel matching."""

    def __init__(self, db_path: Path):
        """Initialize with database path."""
        self.db_path = db_path

    def get_genes_for_panel_matching(self) -> list[dict[str, Any]]:
        """Get all genes with assessments that need panel matching."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    hgnc_id,
                    assessment_json
                FROM gene_assessments g
                WHERE matched_panels_json IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM llm_requests r
                      WHERE r.stage = 'match_panels' AND r.subject = CAST(g.hgnc_id AS TEXT)
                        AND r.status IN ('refused', 'pending')
                  )
                ORDER BY hgnc_id
            """
            )

            genes = []
            for row in cursor.fetchall():
                try:
                    assessment = json.loads(row["assessment_json"])
                    summary = assessment["summary"]

                    # Extract phenotype from disease_entities
                    disease_entities = assessment.get("disease_entities", [])
                    if not disease_entities:
                        # Skip genes without phenotype information - can't match to panels
                        logger.warning(
                            f"Skipping {row['hgnc_id']}: no disease_entities in assessment"
                        )
                        continue

                    # Combine disease descriptions with their MoI for panel matching
                    disease_description_parts = []
                    for entity in disease_entities:
                        moi = entity.get("inheritance_mode")
                        if moi and moi != "NR":
                            disease_description_parts.append(f"{entity['description']} ({moi})")
                        else:
                            disease_description_parts.append(entity["description"])
                    disease_description = "; ".join(disease_description_parts)

                    genes.append(
                        {
                            "hgnc_id": row["hgnc_id"],
                            "summary": summary,
                            "disease_description": disease_description,
                        }
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing assessment for {row['hgnc_id']}: {e}")
                    continue

            return genes

    def count_remaining(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            (remaining,) = conn.execute(
                "SELECT COUNT(*) FROM gene_assessments WHERE matched_panels_json IS NULL"
            ).fetchone()
        return int(remaining)


def store_matched_panels(
    conn: sqlite3.Connection, hgnc_id: int, matched_panels: list[dict[str, Any]], raw: str
) -> None:
    conn.execute(
        "UPDATE gene_assessments SET matched_panels_json = ?, matched_panels_raw = ? WHERE hgnc_id = ?",
        (json.dumps(matched_panels), raw, hgnc_id),
    )


def split_prompt(template: str, panel_list: str) -> tuple[str, str]:
    """(system prompt with the panel list, per-gene template with {summary} and {disease_description})."""
    index = template.index(GENE_CONTEXT_HEADING)
    system = template[:index].format(panel_list=panel_list).rstrip()
    return system, template[index:]


def build_request(
    gene: dict[str, Any], system: str, user_template: str, output_config: OutputConfigParam
) -> LlmRequest:
    user_text = user_template.format(
        summary=gene["summary"], disease_description=gene["disease_description"]
    )
    return LlmRequest(
        subject=str(gene["hgnc_id"]),
        params={
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": cached_system(system),
            "messages": [{"role": "user", "content": user_text}],
            "output_config": output_config,
        },
    )


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
            message = result.message
            if result.status == ResultStatus.REFUSED:
                logger.warning("Refused: HGNC:%s", result.subject)
                continue
            if result.status != ResultStatus.SUCCEEDED or message is None:
                if result.error_type == "invalid_request_error":
                    raise RuntimeError(f"invalid match-panels request for HGNC:{result.subject}")
                continue
            try:
                parsed = parse_json_output(message)
                validator.validate(parsed)
            except (ValueError, jsonschema.ValidationError) as e:
                logger.warning("Invalid panel matching for HGNC:%s: %s", result.subject, e)
                continue
            matches = []
            invalid_names = []
            for match in parsed["matched_panels"]:
                panel_id = name_to_id.get(match["panel_name"].lower())
                if panel_id is None:
                    invalid_names.append(match["panel_name"])
                else:
                    matches.append({"panel_id": panel_id, "rationale": match["rationale"]})
            if invalid_names:
                logger.warning("Invalid panel names for HGNC:%s: %s", result.subject, invalid_names)
                continue
            store_matched_panels(conn, int(result.subject), matches, message.to_json())
            stored += 1
    return stored


async def _process_panel_matching(
    *,
    transport: Transport,
    db_processor: PaperBatchProcessor,
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
        stored = handle_results(resumed, db_processor.db_path, validator, name_to_id)
        logger.info("Collected %d results from earlier batches, stored %d", len(resumed), stored)
    for attempt in range(1, max_retries + 1):
        genes = db_processor.get_genes_for_panel_matching()
        if not genes:
            logger.info("No genes left to match")
            return
        logger.info("Attempt %d: matching %d genes", attempt, len(genes))
        requests = [build_request(gene, system, user_template, output_config) for gene in genes]
        results = await transport.run(STAGE, 1, requests)
        stored = handle_results(results, db_processor.db_path, validator, name_to_id)
        logger.info("Attempt %d: stored %d of %d", attempt, stored, len(genes))
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
        help="Maximum number of attempts for genes whose matching failed",
    ),
    immediate: bool = typer.Option(
        False,
        "--immediate",
        help="Send requests immediately instead of as Message Batches (for prompt development)",
    ),
) -> None:
    """Match genes to diagnostic panels based on phenotype descriptions."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    template = Path("prompts/panel_matching_prompt.txt").read_text()
    schema: dict[str, Any] = json.loads(Path("prompts/panel_matching_schema.json").read_text())

    logger.info("Fetching panel descriptions from PanelApp...")
    panels = PanelAppClient(panel_date).get_all_panel_descriptions()
    logger.info(f"Fetched {len(panels)} panels")
    name_to_id = {info["name"].lower(): panel_id for panel_id, info in panels.items()}
    system, user_template = split_prompt(template, format_all_panels_for_prompt(panels))

    db_processor = PaperBatchProcessor(db_path)
    logger.info(f"{db_processor.count_remaining():,} genes without panel matches")

    async def run() -> None:
        client = make_client(AnthropicSettings())
        transport: Transport = (
            ImmediateTransport(client) if immediate else BatchTransport(client, db_path)
        )
        await _process_panel_matching(
            transport=transport,
            db_processor=db_processor,
            schema=schema,
            system=system,
            user_template=user_template,
            name_to_id=name_to_id,
            max_retries=max_retries,
        )

    asyncio.run(run())
    logger.info(f"{db_processor.count_remaining():,} genes still without panel matches")
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
