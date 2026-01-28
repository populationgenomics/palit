#!/usr/bin/env python3
"""Match genes to diagnostic panels based on phenotype descriptions."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

from palit.llm import HarmonyBatchProcessor
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
                    panelapp_gene_symbol,
                    assessment_json
                FROM gene_assessments
                WHERE matched_panels_json IS NULL
                ORDER BY panelapp_gene_symbol
            """
            )

            genes = []
            for row in cursor.fetchall():
                try:
                    assessment = json.loads(row["assessment_json"])
                    summary = assessment["summary"]

                    # Extract phenotype from phenotype_groups
                    phenotype_groups = assessment.get("phenotype_groups", [])
                    if not phenotype_groups:
                        # Skip genes without phenotype information - can't match to panels
                        logger.warning(
                            f"Skipping {row['panelapp_gene_symbol']}: no phenotype_groups in assessment"
                        )
                        continue

                    # Combine phenotypes with their MoI for panel matching
                    phenotype_parts = []
                    for group in phenotype_groups:
                        moi = group.get("inheritance_mode")
                        if moi and moi != "NR":
                            phenotype_parts.append(f"{group['phenotype']} ({moi})")
                        else:
                            phenotype_parts.append(group["phenotype"])
                    phenotype = "; ".join(phenotype_parts)

                    genes.append(
                        {
                            "gene_symbol": row["panelapp_gene_symbol"],
                            "summary": summary,
                            "phenotype": phenotype,
                        }
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(
                        f"Error parsing assessment for {row['panelapp_gene_symbol']}: {e}"
                    )
                    continue

            return genes

    def update_matched_panels(
        self,
        panelapp_gene_symbol: str,
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
                    WHERE panelapp_gene_symbol = ?
                    """,
                    (json.dumps(matched_panels), raw_response, panelapp_gene_symbol),
                )

                conn.commit()
                logger.info(
                    f"Updated matched panels for {panelapp_gene_symbol}: {len(matched_panels)} panels"
                )

            except sqlite3.Error as e:
                logger.error(f"Error updating matched panels for {panelapp_gene_symbol}: {e}")


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
    inference_engine = HarmonyBatchProcessor(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
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

    # Retry loop
    total_processed = 0
    consecutive_failures = 0
    retry_attempt = 0

    with tqdm(total=initial_count, desc="Matching genes to panels") as pbar:
        while retry_attempt < max_retries:
            # Get genes needing panel matching (re-read from DB)
            genes = db_processor.get_genes_for_panel_matching()

            if not genes:
                logger.info("All genes successfully matched to panels!")
                break

            if retry_attempt > 0:
                logger.info(f"Retry attempt {retry_attempt} - {len(genes)} genes remaining")

            # Process in batches
            pass_processed = 0
            for i in range(0, len(genes), batch_size):
                batch = genes[i : i + batch_size]

                # Prepare prompts for batch
                prompts = []
                for gene_info in batch:
                    prompt = template.format(
                        panel_list=panel_list,
                        summary=gene_info["summary"],
                        phenotype_description=gene_info["phenotype"],
                    )
                    prompts.append(prompt)

                # Process batch
                results = inference_engine.process_batch(prompts, schema)

                # Validate and update database with matches
                for gene_info, result in zip(batch, results, strict=True):
                    gene_symbol = gene_info["gene_symbol"]

                    if result is not None and "matched_panels" in result.parsed_json:
                        llm_matches = result.parsed_json["matched_panels"]

                        # Validate panel names and convert to IDs
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
                            # Invalid panel names - treat as failure, do NOT update DB
                            logger.warning(
                                f"Invalid panel names for {gene_symbol}: {invalid_names} - treating as failure"
                            )
                        else:
                            # All panel names resolved - update DB with IDs
                            db_processor.update_matched_panels(
                                gene_symbol, resolved_matches, result.raw_response
                            )
                            pass_processed += 1
                            pbar.update(1)
                            if not resolved_matches:
                                logger.info(
                                    f"No panel matches for {gene_symbol} (empty list is valid)"
                                )
                    else:
                        logger.warning(f"Failed to get valid JSON response for {gene_symbol}")

            total_processed += pass_processed

            # Check if we made progress
            if pass_processed == 0:
                consecutive_failures += 1
                logger.warning(f"No progress made in retry attempt {retry_attempt}")
                if consecutive_failures >= 2:
                    logger.error("Multiple consecutive attempts with no progress - stopping")
                    break
            else:
                consecutive_failures = 0

            retry_attempt += 1

    # Final count
    final_genes = db_processor.get_genes_for_panel_matching()
    final_remaining = len(final_genes)

    logger.info("Panel matching complete!")
    logger.info(f"Successfully matched {total_processed} genes to panels")

    if final_remaining > 0:
        logger.warning(
            f"Failed to match {final_remaining} genes to panels after {max_retries} attempts"
        )


if __name__ == "__main__":
    app()
