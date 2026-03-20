#!/usr/bin/env python3
"""Standalone vLLM inference utility for assessing paper relevance."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

from palit.hgnc import HgncResolver
from palit.llm import HarmonyBatchProcessor, PromptResult
from palit.panelapp_client import (
    PanelAppClient,
    format_panel_for_prompt,
)
from palit.relevance import compute_relevance_majority_vote

app = typer.Typer(help="Assess paper relevance using vLLM inference with majority voting")
logger = logging.getLogger(__name__)

# SQLite busy timeout in seconds. When sharding, multiple processes write to
# the same database; the default 5 s can be too short for large batch commits.
DB_TIMEOUT_SECONDS = 60


class PaperBatchProcessor:
    """Handle database operations for batch processing papers."""

    def __init__(self, db_path: Path):
        """Initialize with database path."""
        self.db_path = db_path

    def get_batch_for_processing(
        self, batch_size: int, shard_index: int, num_shards: int
    ) -> list[dict[str, Any]]:
        """
        Get a batch of unprocessed papers for LLM inference.

        Args:
            batch_size: Number of papers to fetch
            shard_index: Shard index (0-based) for parallel processing
            num_shards: Total number of shards

        Returns list of papers with doi, title, abstract.
        """
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT doi, title, abstract
                FROM papers
                WHERE relevance_assessment_json IS NULL
                AND title IS NOT NULL
                AND abstract IS NOT NULL
                AND rowid % ? = ?
                ORDER BY doi
                LIMIT ?
            """,
                (num_shards, shard_index, batch_size),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_paper_relevance_assessments(
        self,
        papers: list[dict[str, Any]],
        all_results: list[tuple[PromptResult | None, PromptResult | None, PromptResult | None]],
        hgnc_resolver: HgncResolver,
    ) -> None:
        """
        Update paper relevance assessment responses and extracted JSON.
        Also populates gene_mentions with source='relevance_assessment'.

        Args:
            papers: List of paper dicts with DOIs
            all_results: List of 3-tuples of PromptResult objects or None, same order as papers
            hgnc_resolver: HGNC resolver for gene symbol normalization
        """
        if not papers or not all_results:
            return

        successful_updates = 0

        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()

            # Update each paper with its 3 results
            for paper, results_triple in zip(papers, all_results, strict=True):
                # Require all 3 results to succeed
                if not all(result is not None for result in results_triple):
                    continue

                doi = paper["doi"]

                # Store arrays of raw and parsed results
                raw_array = [result.raw_response for result in results_triple if result]
                json_array = [result.parsed_json for result in results_triple if result]

                # Compute majority vote for decision-making
                majority_result = compute_relevance_majority_vote(json_array)
                is_relevant = majority_result["relevant"]
                associations = majority_result["associations"]

                # Update papers table with arrays
                cursor.execute(
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
                    (
                        json.dumps(raw_array),
                        json.dumps(json_array),
                        1 if is_relevant else 0,
                        doi,
                    ),
                )

                # Extract and resolve gene symbols from associations (only for relevant papers)
                if is_relevant and associations:
                    for assoc in associations:
                        paper_gene_symbol: str = assoc["gene_symbol"]
                        entry = hgnc_resolver.resolve(paper_gene_symbol)
                        if entry is None:
                            logger.debug(
                                f"Unresolved gene symbol '{paper_gene_symbol}' in DOI {doi}"
                            )
                            continue
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO gene_mentions
                            (hgnc_id, paper_gene_symbol, paper_doi, source)
                            VALUES (?, ?, ?, 'relevance_assessment')
                            """,
                            (entry.hgnc_id, paper_gene_symbol.upper(), doi),
                        )

                successful_updates += 1

            conn.commit()
            logger.info(
                f"Updated {successful_updates} papers with relevance assessments and gene_mentions"
            )

    def get_processing_statistics(self, shard_index: int, num_shards: int) -> dict[str, int]:
        """Get statistics about processing progress for this shard."""
        logger.debug("Querying database for processing statistics...")
        with sqlite3.connect(self.db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
            cursor = conn.cursor()

            # Total papers
            logger.debug("  Counting total papers...")
            cursor.execute(
                "SELECT COUNT(*) FROM papers WHERE rowid % ? = ?", (num_shards, shard_index)
            )
            total_papers = cursor.fetchone()[0]

            # Papers with both title and abstract
            logger.debug("  Counting processable papers...")
            cursor.execute(
                """
                SELECT COUNT(*) FROM papers
                WHERE title IS NOT NULL AND abstract IS NOT NULL
                AND rowid % ? = ?
            """,
                (num_shards, shard_index),
            )
            processable_papers = cursor.fetchone()[0]

            # Processed papers
            logger.debug("  Counting already processed papers...")
            cursor.execute(
                """
                SELECT COUNT(*) FROM papers
                WHERE relevance_assessment_json IS NOT NULL
                AND rowid % ? = ?
            """,
                (num_shards, shard_index),
            )
            processed_papers = cursor.fetchone()[0]

            # Remaining papers
            logger.debug("  Counting remaining papers to process...")
            cursor.execute(
                """
                SELECT COUNT(*) FROM papers
                WHERE relevance_assessment_json IS NULL
                AND title IS NOT NULL
                AND abstract IS NOT NULL
                AND rowid % ? = ?
            """,
                (num_shards, shard_index),
            )
            remaining_papers = cursor.fetchone()[0]

            return {
                "total_papers": total_papers,
                "processable_papers": processable_papers,
                "processed_papers": processed_papers,
                "remaining_papers": remaining_papers,
            }


# Maximum abstract length in characters (~2500 tokens at 4 chars/token).
# Legitimate abstracts rarely exceed 5000 chars; this handles edge cases like
# taxonomic papers that dump species lists into the abstract.
MAX_ABSTRACT_CHARS = 10000


def prepare_prompts_for_papers(
    papers: list[dict[str, Any]],
    template: str,
    extra_template_vars: dict[str, Any] | None = None,
) -> list[str]:
    """
    Prepare prompts for papers.

    Args:
        papers: List of paper dicts with 'title' and 'abstract'
        template: Prompt template string with {title}, {abstract}, and optional other placeholders
        extra_template_vars: Optional dict of additional template variables (e.g., {"panel_description": "..."})

    Returns:
        List of prompts in same order as papers.
    """
    prompts = []
    for paper in papers:
        abstract = paper["abstract"]

        # Truncate excessively long abstracts
        if len(abstract) > MAX_ABSTRACT_CHARS:
            logger.warning(
                f"Truncating abstract for DOI {paper['doi']} from {len(abstract)} to {MAX_ABSTRACT_CHARS} chars"
            )
            abstract = abstract[:MAX_ABSTRACT_CHARS] + "... [truncated]"

        # Build template variables starting with title and abstract
        template_vars = {
            "title": paper["title"],
            "abstract": abstract,
        }

        # Add any extra template variables
        if extra_template_vars:
            template_vars.update(extra_template_vars)

        # Fill the template
        prompt = template.format(**template_vars)
        prompts.append(prompt)

    return prompts


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database",
    ),
    panel_date: str = typer.Option(
        ...,
        "--panel-date",
        help="Panel state at date (YYYY-MM-DD) for gene alias resolution",
    ),
    scope_panel_id: int | None = typer.Option(
        None,
        "--scope-panel-id",
        help="Panel ID for panel-scoped relevance assessment (injects panel description into prompt template)",
    ),
    shard_index: int = typer.Option(
        0,
        "--shard-index",
        help="Shard index (0-based) for parallel processing across multiple GPUs",
    ),
    num_shards: int = typer.Option(
        1,
        "--num-shards",
        help="Total number of shards for parallel processing (values > 1 require database WAL mode)",
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
        4096,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    batch_size: int = typer.Option(
        1000,
        "--batch-size",
        "-b",
        help="Number of papers per batch",
    ),
    tensor_parallel_size: int = typer.Option(
        1,
        "--tensor-parallel-size",
        help="Tensor parallelism size",
    ),
    max_model_len: int = typer.Option(
        8192,
        "--max-model-len",
        help="Maximum model context length",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
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
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of retry attempts for failed batches",
    ),
) -> None:
    """Assess paper relevance using vLLM inference on unprocessed papers in batches."""
    # Validate inputs
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if not prompt_path.exists():
        logger.error(f"Prompt template not found: {prompt_path}")
        raise typer.Exit(1)

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        raise typer.Exit(1)

    # Load prompt template and schema
    logger.info("Loading prompt template and schema...")
    template = prompt_path.read_text()
    logger.info(f"  Loaded prompt template from {prompt_path}")
    schema: dict[str, Any] = json.loads(schema_path.read_text())
    logger.info(f"  Loaded schema from {schema_path}")

    # Initialize components
    logger.info("Initializing database processor...")
    db_processor = PaperBatchProcessor(db_path)
    logger.info(f"  Connected to database at {db_path}")

    # Load HGNC resolver for gene symbol normalization
    hgnc_resolver = HgncResolver.from_file()
    logger.info(f"  Loaded HgncResolver with {len(hgnc_resolver._by_symbol)} genes")

    # Fetch panel description if scope_panel_id is provided
    client = PanelAppClient(panel_date)
    panel_description = None
    if scope_panel_id is not None:
        logger.info(f"Fetching description for scope panel {scope_panel_id}...")
        try:
            panel_info = client.get_panel_data(scope_panel_id)
        except ValueError as e:
            logger.error(f"Panel {scope_panel_id} not found in PanelApp data for {panel_date}")
            raise typer.Exit(1) from e

        panel_description = format_panel_for_prompt(scope_panel_id, panel_info)
        logger.info(f"  Panel-scoped mode: {panel_info.get('name', 'Unknown')}")

    logger.info("Initializing Harmony batch processor...")
    inference_engine = HarmonyBatchProcessor(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )

    # Get initial statistics
    logger.info("Fetching database statistics...")
    stats = db_processor.get_processing_statistics(shard_index, num_shards)
    logger.info(f"Database statistics (shard {shard_index}/{num_shards}):")
    logger.info(f"  Total papers: {stats['total_papers']:,}")
    logger.info(f"  Processable papers: {stats['processable_papers']:,}")
    logger.info(f"  Already processed: {stats['processed_papers']:,}")
    logger.info(f"  Remaining to process: {stats['remaining_papers']:,}")

    if stats["remaining_papers"] == 0:
        logger.info("No papers remaining to process!")
        return

    # Calculate initial estimate of batches needed
    initial_remaining = stats["remaining_papers"]
    estimated_batches = (initial_remaining + batch_size - 1) // batch_size
    logger.info(f"Estimated {estimated_batches} batches of {batch_size} papers each")

    # Retry loop
    total_processed = 0
    consecutive_failures = 0
    retry_attempt = 0

    with tqdm(total=initial_remaining, desc="Processing papers") as pbar:
        while retry_attempt < max_retries:
            # Check remaining work
            stats = db_processor.get_processing_statistics(shard_index, num_shards)
            if stats["remaining_papers"] == 0:
                logger.info("All papers successfully processed!")
                break

            if retry_attempt > 0:
                logger.info(
                    f"Retry attempt {retry_attempt} - {stats['remaining_papers']} papers remaining"
                )

            batch_num = 0
            batch_level_processed = 0

            while True:
                batch_num += 1
                logger.info(f"Starting batch {batch_num} (retry {retry_attempt})")

                # Get batch of papers
                logger.info(f"Fetching batch of {batch_size} papers from database...")
                papers = db_processor.get_batch_for_processing(batch_size, shard_index, num_shards)
                logger.info(f"  Retrieved {len(papers)} papers for processing")

                if not papers:
                    logger.info("No more papers in this pass")
                    break

                # Prepare prompts
                extra_vars = {"panel_description": panel_description} if panel_description else None
                prompts = prepare_prompts_for_papers(papers, template, extra_vars)

                # Process batch 3 times for majority voting
                logger.info("  Running inference 3 times per paper...")
                all_runs = []
                for run_num in range(3):
                    logger.info(f"    Run {run_num + 1}/3...")
                    results = inference_engine.process_batch(prompts, schema)
                    all_runs.append(results)

                # Transpose: convert from 3 lists of N results to N lists of 3 results
                all_results = list(zip(*all_runs, strict=True))

                # Update database immediately (only successful results)
                db_processor.update_paper_relevance_assessments(papers, all_results, hgnc_resolver)

                # Count successful results (all 3 must succeed)
                num_successful = sum(
                    1 for triple in all_results if all(result is not None for result in triple)
                )
                batch_level_processed += num_successful
                pbar.update(num_successful)

                logger.info(f"Completed batch {batch_num}")
                if num_successful < len(papers):
                    logger.warning(f"  {len(papers) - num_successful} papers failed in this batch")
                logger.info(f"Total processed in this pass: {batch_level_processed:,}")

            total_processed += batch_level_processed

            # Check if we made progress
            if batch_level_processed == 0:
                consecutive_failures += 1
                logger.warning(f"No progress made in retry attempt {retry_attempt}")
                if consecutive_failures >= 2:
                    logger.error("Multiple consecutive attempts with no progress - stopping")
                    break
            else:
                consecutive_failures = 0

            retry_attempt += 1

    # Final statistics
    final_stats = db_processor.get_processing_statistics(shard_index, num_shards)
    logger.info("Processing complete!")
    logger.info(f"Final statistics (shard {shard_index}/{num_shards}):")
    logger.info(f"  Total processed: {final_stats['processed_papers']:,}")
    logger.info(f"  Remaining: {final_stats['remaining_papers']:,}")

    if final_stats["remaining_papers"] > 0:
        logger.warning(
            f"Failed to process {final_stats['remaining_papers']} papers after {max_retries} attempts"
        )


if __name__ == "__main__":
    app()
