#!/usr/bin/env python3
"""Match genes to diagnostic panels based on phenotype descriptions."""

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

from palit.llm import LLMProcessor, create_llm_processor
from palit.panelapp_client import PanelAppClient, format_panel_for_prompt

app = typer.Typer(help="Match genes to diagnostic panels based on phenotype descriptions")
logger = logging.getLogger(__name__)


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
                FROM gene_assessments
                WHERE matched_panels_json IS NULL
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

    def update_matched_panels(
        self,
        hgnc_id: int,
        matched_panels: list[dict[str, Any]],
        raw_response: str,
    ) -> None:
        """Update matched_panels_json and matched_panels_raw in gene_assessments table."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    UPDATE gene_assessments
                    SET matched_panels_json = ?, matched_panels_raw = ?
                    WHERE hgnc_id = ?
                    """,
                    (json.dumps(matched_panels), raw_response, hgnc_id),
                )

                conn.commit()
                logger.info(f"Updated matched panels for {hgnc_id}: {len(matched_panels)} panels")

            except sqlite3.Error as e:
                logger.error(f"Error updating matched panels for {hgnc_id}: {e}")


async def _process_panel_matching(
    *,
    llm_processor: LLMProcessor,
    db_processor: PaperBatchProcessor,
    schema: dict[str, Any],
    template: str,
    panel_list: str,
    name_to_id: dict[str, int],
    batch_size: int,
    max_retries: int,
    initial_count: int,
) -> None:
    """Run the panel matching retry loop."""
    total_processed = 0
    consecutive_failures = 0
    retry_attempt = 0

    with tqdm(total=initial_count, desc="Matching genes to panels") as pbar:
        while retry_attempt < max_retries:
            genes = db_processor.get_genes_for_panel_matching()

            if not genes:
                logger.info("All genes successfully matched to panels!")
                break

            if retry_attempt > 0:
                logger.info(f"Retry attempt {retry_attempt} - {len(genes)} genes remaining")

            pass_processed = 0
            for i in range(0, len(genes), batch_size):
                batch = genes[i : i + batch_size]

                prompts = []
                for gene_info in batch:
                    prompt = template.format(
                        panel_list=panel_list,
                        summary=gene_info["summary"],
                        disease_description=gene_info["disease_description"],
                    )
                    prompts.append(prompt)

                results = await llm_processor.process_batch(prompts, schema)

                for gene_info, result in zip(batch, results, strict=True):
                    hgnc_id = gene_info["hgnc_id"]

                    if result is not None and "matched_panels" in result.parsed_json:
                        llm_matches = result.parsed_json["matched_panels"]

                        all_valid = True
                        invalid_names = []
                        resolved_matches = []

                        for match in llm_matches:
                            panel_name = match.get("panel_name", "")
                            panel_id = name_to_id.get(panel_name.lower())
                            if panel_id is None:
                                all_valid = False
                                invalid_names.append(panel_name)
                            else:
                                resolved_matches.append(
                                    {"panel_id": panel_id, "rationale": match.get("rationale", "")}
                                )

                        if not all_valid:
                            logger.warning(
                                f"Invalid panel names for HGNC:{hgnc_id}: {invalid_names} - treating as failure"
                            )
                        else:
                            db_processor.update_matched_panels(
                                hgnc_id, resolved_matches, result.raw_response
                            )
                            pass_processed += 1
                            pbar.update(1)
                            if not resolved_matches:
                                logger.info(
                                    f"No panel matches for HGNC:{hgnc_id} (empty list is valid)"
                                )
                    else:
                        logger.warning(f"Failed to get valid JSON response for HGNC:{hgnc_id}")

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

    final_genes = db_processor.get_genes_for_panel_matching()
    final_remaining = len(final_genes)

    logger.info("Panel matching complete!")
    logger.info(f"Successfully matched {total_processed} genes to panels")

    if final_remaining > 0:
        logger.warning(
            f"Failed to match {final_remaining} genes to panels after {max_retries} attempts"
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
        20000,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    batch_size: int = typer.Option(
        100,
        "--batch-size",
        "-b",
        help="Number of genes per batch",
    ),
    tensor_parallel_size: int = typer.Option(
        1,
        "--tensor-parallel-size",
        help="Tensor parallelism size",
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
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
    panel_date: str = typer.Option(
        ...,
        "--panel-date",
        help="Panel state date (YYYY-MM-DD) for gene alias resolution",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of retry attempts for failed batches",
    ),
) -> None:
    """Match genes to diagnostic panels based on phenotype descriptions."""
    # Validate inputs
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    # Load prompts
    prompt_path = Path("prompts/panel_matching_prompt.txt")
    schema_path = Path("prompts/panel_matching_schema.json")

    if not prompt_path.exists():
        logger.error(f"Panel matching prompt not found: {prompt_path}")
        raise typer.Exit(1)

    if not schema_path.exists():
        logger.error(f"Panel matching schema not found: {schema_path}")
        raise typer.Exit(1)

    template = prompt_path.read_text()
    schema: dict[str, Any] = json.loads(schema_path.read_text())

    # Initialize components
    db_processor = PaperBatchProcessor(db_path)
    llm_processor = create_llm_processor(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        **(json.loads(llm_config) if llm_config else {}),
    )

    # Fetch all panel descriptions once
    logger.info("Fetching panel descriptions from PanelApp...")
    panelapp_client = PanelAppClient(panel_date)
    panels = panelapp_client.get_all_panel_descriptions()
    logger.info(f"Fetched {len(panels)} panels")

    # Create case-insensitive name→ID lookup for validation
    name_to_id = {info["name"].lower(): panel_id for panel_id, info in panels.items()}

    # Format panels for prompt (static part)
    panel_list = format_all_panels_for_prompt(panels)

    # Get initial count of genes needing panel matching
    initial_genes = db_processor.get_genes_for_panel_matching()
    initial_count = len(initial_genes)
    logger.info(f"Found {initial_count} genes to match to panels")

    if initial_count == 0:
        logger.info("No genes need panel matching")
        return

    asyncio.run(
        _process_panel_matching(
            llm_processor=llm_processor,
            db_processor=db_processor,
            schema=schema,
            template=template,
            panel_list=panel_list,
            name_to_id=name_to_id,
            batch_size=batch_size,
            max_retries=max_retries,
            initial_count=initial_count,
        )
    )


if __name__ == "__main__":
    app()
