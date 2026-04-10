#!/usr/bin/env python3
"""Tournament-based literature reduction to minimize manual download burden."""

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

from palit.hgnc import HgncResolver
from palit.llm import LLMProcessor, create_llm_processor
from palit.papers import Paper, deserialize_source_metadata
from palit.tournament import TournamentOutcome, run_tournament_selection

app = typer.Typer(help="Reduce literature using tournament selection to minimize manual downloads")
logger = logging.getLogger(__name__)


def get_papers_for_gene(db_path: Path, hgnc_id: int, limit: int) -> list[Paper]:
    """Fetch all papers for a gene, regardless of download status.

    Args:
        db_path: Path to SQLite database
        hgnc_id: HGNC ID of the gene to look up
        limit: Maximum number of papers to return

    Returns:
        List of up to `limit` newest papers mentioning this gene
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT p.doi, p.pmid, p.title, p.abstract, p.authors, p.journal,
                   p.source, p.source_date, p.source_metadata
            FROM papers p
            JOIN gene_mentions gm ON p.doi = gm.paper_doi
            WHERE gm.hgnc_id = ?
            ORDER BY p.source_date DESC, p.doi DESC
            LIMIT ?
            """,
            (hgnc_id, limit),
        )

        papers = []
        for row in cursor.fetchall():
            papers.append(
                Paper(
                    doi=row["doi"],
                    pmid=row["pmid"],
                    title=row["title"],
                    abstract=row["abstract"],
                    authors=row["authors"],
                    journal=row["journal"],
                    source=row["source"],
                    source_date=row["source_date"],
                    source_metadata=deserialize_source_metadata(
                        row["source"], row["source_metadata"]
                    ),
                    source_type="initial",
                    source_details=str(hgnc_id),
                )
            )

        logger.info(f"Found {len(papers)} papers for HGNC:{hgnc_id}")
        return papers


def clear_unselected_papers(db_path: Path, all_selected_dois: set[str]) -> dict[str, int]:
    """Clear download_status for papers not selected by any gene.

    The aggregation step has a practical limit of ~30-40 papers per gene due to
    context window constraints (input evidence + output reasoning). For well-researched
    genes like MT-TL1 (200+ papers), we must reduce to a manageable subset before
    downloading.

    Correct workflow: assess-relevance -> reduce-literature -> download-papers

    Args:
        db_path: Path to SQLite database
        all_selected_dois: Union of all DOIs selected across all genes

    Returns:
        Dict with counts by previous download_status
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Get counts by status before clearing
        cursor.execute(
            """
            SELECT download_status, COUNT(*)
            FROM papers
            WHERE download_status IS NOT NULL
            GROUP BY download_status
            """
        )
        status_counts_before = dict(cursor.fetchall())

        if not all_selected_dois:
            # No papers selected - clear all
            cursor.execute(
                "UPDATE papers SET download_status = NULL WHERE download_status IS NOT NULL"
            )
            conn.commit()
            return {f"cleared_{k}": v for k, v in status_counts_before.items()}

        # Clear download_status for papers not in the selected set
        placeholders = ",".join("?" * len(all_selected_dois))
        cursor.execute(
            f"""
            UPDATE papers
            SET download_status = NULL
            WHERE download_status IS NOT NULL
            AND doi NOT IN ({placeholders})
            """,
            tuple(all_selected_dois),
        )
        cleared_count = cursor.rowcount

        # Get counts by status after clearing (for reporting)
        cursor.execute(
            """
            SELECT download_status, COUNT(*)
            FROM papers
            WHERE download_status IS NOT NULL
            GROUP BY download_status
            """
        )
        status_counts_after = dict(cursor.fetchall())

        conn.commit()

        # Calculate what was cleared per status
        result: dict[str, int] = {}
        for status, before_count in status_counts_before.items():
            after_count = status_counts_after.get(status, 0)
            result[f"cleared_{status}"] = before_count - after_count
            result[f"kept_{status}"] = after_count

        result["total_cleared"] = cleared_count
        return result


def _record_reduction_completion(db_path: Path, hgnc_id: int, outcome: TournamentOutcome) -> None:
    """Persist tournament results for resumability."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO tournament_results (
                hgnc_id,
                selected_dois_json,
                tournament_raw_responses_json
            )
            VALUES (?, ?, ?)
            ON CONFLICT(hgnc_id) DO UPDATE SET
                selected_dois_json = excluded.selected_dois_json,
                tournament_raw_responses_json = excluded.tournament_raw_responses_json
            """,
            (
                hgnc_id,
                json.dumps([p.doi for p in outcome.selected_papers]),
                json.dumps(outcome.raw_responses_by_round),
            ),
        )
        conn.commit()


async def _process_reduction(
    *,
    llm_processor: LLMProcessor,
    hgnc_resolver: HgncResolver,
    genes_with_counts: list[tuple[int, int]],
    db_path: Path,
    schema: dict[str, Any],
    template: str,
    paper_limit: int,
    max_papers: int,
    papers_per_round: int,
    max_concurrent_batches: int,
    max_retries: int,
) -> None:
    """Run the literature reduction loop."""
    # Phase 1a: Collect DOIs for genes that don't need reduction (≤max_papers)
    all_selected_dois: set[str] = set()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT gm.paper_doi
            FROM gene_mentions gm
            WHERE gm.hgnc_id IN (
                SELECT hgnc_id
                FROM gene_mentions
                GROUP BY hgnc_id
                HAVING COUNT(DISTINCT paper_doi) <= ?
            )
            """,
            (max_papers,),
        )
        small_gene_dois = {row[0] for row in cursor.fetchall()}
        all_selected_dois.update(small_gene_dois)
        logger.info(
            f"Auto-selected {len(small_gene_dois)} papers from genes with ≤{max_papers} papers"
        )

    # Phase 1b: Run tournament selection for genes with many papers
    for hgnc_id, paper_count in tqdm(genes_with_counts, desc="Reducing literature"):
        hgnc_symbol = hgnc_resolver.get_symbol(hgnc_id)
        papers = get_papers_for_gene(db_path, hgnc_id, paper_limit)

        if not papers:
            logger.warning(f"No papers found for {hgnc_symbol} (HGNC:{hgnc_id})")
            continue

        tournament_outcome = await run_tournament_selection(
            gene_symbol=hgnc_symbol,
            papers=papers,
            llm_processor=llm_processor,
            prompt_template=template,
            schema=schema,
            max_papers=max_papers,
            papers_per_round=papers_per_round,
            max_concurrent_batches=max_concurrent_batches,
            max_retries=max_retries,
        )

        selected_dois = {p.doi for p in tournament_outcome.selected_papers}
        all_selected_dois.update(selected_dois)

        logger.info(
            f"{hgnc_symbol} (HGNC:{hgnc_id}): {paper_count} -> {len(selected_dois)} selected"
        )

        _record_reduction_completion(db_path, hgnc_id, tournament_outcome)

    # Phase 2: Clear download_status for papers not selected by ANY gene
    logger.info("Clearing download_status for unselected papers...")
    clear_stats = clear_unselected_papers(db_path, all_selected_dois)

    logger.info("Literature reduction complete!")
    logger.info(f"Total genes processed: {len(genes_with_counts)}")
    logger.info(f"Total papers selected: {len(all_selected_dois)}")
    for key, value in clear_stats.items():
        logger.info(f"  {key}: {value}")


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        Path("data/db.sqlite"),
        "--db-path",
        help="Path to SQLite database",
    ),
    max_papers: int = typer.Option(
        20,
        "--max-papers",
        help="Maximum papers to keep per gene",
    ),
    papers_per_round: int = typer.Option(
        100,
        "--papers-per-round",
        help="Papers to show LLM in each tournament round",
    ),
    max_concurrent_batches: int = typer.Option(
        100,
        "--max-concurrent-batches",
        help="Maximum number of batches to process concurrently",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of retries for failed batches",
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
        6000,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    tensor_parallel_size: int = typer.Option(
        1,
        "--tensor-parallel-size",
        help="Tensor parallelism size",
    ),
    max_model_len: int = typer.Option(
        35000,
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
    paper_limit: int = typer.Option(
        10000,
        "--paper-limit",
        help="Maximum number of papers to consider per gene",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without making changes",
    ),
) -> None:
    """Reduce literature using tournament selection to minimize manual downloads.

    This command runs tournament selection on genes with many papers, keeping only
    the most informative ones for manual download. Papers that have already been
    downloaded (via PMC or manually) are preserved as "free bonus" for analysis.

    Typical workflow:
        1. Run `download-papers attempt-pmc` to auto-download what's easy
        2. Run `reduce-literature --max-papers 5` to select best papers per gene
        3. Run `download-papers open-browser` to manually download the reduced set
    """
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    # Load prompt and schema
    prompt_path = Path("prompts/tournament_selection_prompt.txt")
    schema_path = Path("prompts/tournament_selection_schema.json")

    if not prompt_path.exists():
        logger.error(f"Prompt not found: {prompt_path}")
        raise typer.Exit(1)

    if not schema_path.exists():
        logger.error(f"Schema not found: {schema_path}")
        raise typer.Exit(1)

    template = prompt_path.read_text()
    schema: dict[str, Any] = json.loads(schema_path.read_text())

    # Find genes that need reduction (have papers but no tournament results yet)
    logger.info("Finding genes to reduce...")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT gm.hgnc_id, COUNT(DISTINCT gm.paper_doi) as paper_count
            FROM gene_mentions gm
            LEFT JOIN tournament_results er
              ON gm.hgnc_id = er.hgnc_id
            WHERE er.hgnc_id IS NULL
            GROUP BY gm.hgnc_id
            HAVING paper_count > ?
            ORDER BY paper_count DESC
            """,
            (max_papers,),
        )
        genes_with_counts = cursor.fetchall()

    if not genes_with_counts:
        logger.info("No genes require reduction")
        return

    total_papers_before = sum(count for _, count in genes_with_counts)
    logger.info(
        f"Found {len(genes_with_counts)} genes with >{max_papers} papers "
        f"(total {total_papers_before} papers)"
    )

    # Initialize HGNC resolver for gene symbol display
    hgnc_resolver = HgncResolver.from_file()

    if dry_run:
        logger.info("Dry run - showing genes that would be processed:")
        for hgnc_id, count in genes_with_counts[:20]:
            symbol = hgnc_resolver.get_symbol(hgnc_id)
            logger.info(f"  {symbol} (HGNC:{hgnc_id}): {count} papers -> {max_papers}")
        if len(genes_with_counts) > 20:
            logger.info(f"  ... and {len(genes_with_counts) - 20} more genes")
        return

    # Initialize LLM processor
    logger.info("Initializing LLM processor...")
    llm_processor = create_llm_processor(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        reasoning_effort="medium",
        **(json.loads(llm_config) if llm_config else {}),
    )

    asyncio.run(
        _process_reduction(
            llm_processor=llm_processor,
            hgnc_resolver=hgnc_resolver,
            genes_with_counts=genes_with_counts,
            db_path=db_path,
            schema=schema,
            template=template,
            paper_limit=paper_limit,
            max_papers=max_papers,
            papers_per_round=papers_per_round,
            max_concurrent_batches=max_concurrent_batches,
            max_retries=max_retries,
        )
    )


if __name__ == "__main__":
    app()
