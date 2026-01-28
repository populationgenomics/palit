#!/usr/bin/env python3
"""Tournament-based literature reduction to minimize manual download burden."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

from palit.ingest_pubmed import Article
from palit.llm import HarmonyBatchProcessor
from palit.tournament import TournamentOutcome, run_tournament_selection

app = typer.Typer(help="Reduce literature using tournament selection to minimize manual downloads")
logger = logging.getLogger(__name__)


def get_articles_for_gene(db_path: Path, gene_symbol: str, limit: int) -> list[Article]:
    """Fetch all articles for a gene, regardless of download status.

    Args:
        db_path: Path to SQLite database
        gene_symbol: Gene symbol to look up
        limit: Maximum number of articles to return

    Returns:
        List of up to `limit` newest articles mentioning this gene
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT p.pmid, p.title, p.abstract, p.authors, p.journal, p.entrez_date
            FROM papers p
            JOIN gene_mentions gm ON p.pmid = gm.pmid
            WHERE gm.panelapp_gene_symbol = ?
            ORDER BY p.entrez_date DESC, p.pmid DESC
            LIMIT ?
            """,
            (gene_symbol, limit),
        )

        articles = []
        for row in cursor.fetchall():
            articles.append(
                Article(
                    pmid=row["pmid"],
                    title=row["title"],
                    abstract=row["abstract"],
                    authors=row["authors"],
                    journal=row["journal"],
                    entrez_date=row["entrez_date"],
                    source_type="initial",
                    source_details=gene_symbol,
                )
            )

        logger.info(f"Found {len(articles)} articles for gene {gene_symbol}")
        return articles


def clear_unselected_papers(db_path: Path, all_selected_pmids: set[int]) -> dict[str, int]:
    """Clear download_status for papers not selected by any gene.

    The aggregation step has a practical limit of ~30-40 papers per gene due to
    context window constraints (input evidence + output reasoning). For well-researched
    genes like MT-TL1 (200+ papers), we must reduce to a manageable subset before
    downloading.

    Correct workflow: assess-relevance -> reduce-literature -> download-papers

    Args:
        db_path: Path to SQLite database
        all_selected_pmids: Union of all PMIDs selected across all genes

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

        if not all_selected_pmids:
            # No papers selected - clear all
            cursor.execute(
                "UPDATE papers SET download_status = NULL WHERE download_status IS NOT NULL"
            )
            conn.commit()
            return {f"cleared_{k}": v for k, v in status_counts_before.items()}

        # Clear download_status for papers not in the selected set
        placeholders = ",".join("?" * len(all_selected_pmids))
        cursor.execute(
            f"""
            UPDATE papers
            SET download_status = NULL
            WHERE download_status IS NOT NULL
            AND pmid NOT IN ({placeholders})
            """,
            tuple(all_selected_pmids),
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


def _record_reduction_completion(
    db_path: Path, gene_symbol: str, outcome: TournamentOutcome
) -> None:
    """Persist tournament results for resumability."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO tournament_results (
                panelapp_gene_symbol,
                selected_pmids_json,
                tournament_raw_responses_json
            )
            VALUES (?, ?, ?)
            ON CONFLICT(panelapp_gene_symbol) DO UPDATE SET
                selected_pmids_json = excluded.selected_pmids_json,
                tournament_raw_responses_json = excluded.tournament_raw_responses_json
            """,
            (
                gene_symbol,
                json.dumps([a.pmid for a in outcome.selected_articles]),
                json.dumps(outcome.raw_responses_by_round),
            ),
        )
        conn.commit()


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
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level",
    ),
    pmid_limit: int = typer.Option(
        10000,
        "--pmid-limit",
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
            SELECT gm.panelapp_gene_symbol, COUNT(DISTINCT gm.pmid) as paper_count
            FROM gene_mentions gm
            LEFT JOIN tournament_results er
              ON gm.panelapp_gene_symbol = er.panelapp_gene_symbol
            WHERE er.panelapp_gene_symbol IS NULL
            GROUP BY gm.panelapp_gene_symbol
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

    if dry_run:
        logger.info("Dry run - showing genes that would be processed:")
        for gene, count in genes_with_counts[:20]:
            logger.info(f"  {gene}: {count} papers -> {max_papers}")
        if len(genes_with_counts) > 20:
            logger.info(f"  ... and {len(genes_with_counts) - 20} more genes")
        return

    # Initialize LLM processor
    logger.info("Initializing LLM processor...")
    inference_engine = HarmonyBatchProcessor(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        reasoning_effort="medium",
    )

    # Phase 1a: Collect PMIDs for genes that don't need reduction (≤max_papers)
    # These genes implicitly have all their papers selected
    all_selected_pmids: set[int] = set()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT gm.pmid
            FROM gene_mentions gm
            WHERE gm.panelapp_gene_symbol IN (
                SELECT panelapp_gene_symbol
                FROM gene_mentions
                GROUP BY panelapp_gene_symbol
                HAVING COUNT(DISTINCT pmid) <= ?
            )
            """,
            (max_papers,),
        )
        small_gene_pmids = {row[0] for row in cursor.fetchall()}
        all_selected_pmids.update(small_gene_pmids)
        logger.info(
            f"Auto-selected {len(small_gene_pmids)} papers from genes with ≤{max_papers} papers"
        )

    # Phase 1b: Run tournament selection for genes with many papers
    for gene_symbol, paper_count in tqdm(genes_with_counts, desc="Reducing literature"):
        articles = get_articles_for_gene(db_path, gene_symbol, pmid_limit)

        if not articles:
            logger.warning(f"No articles found for {gene_symbol}")
            continue

        tournament_outcome = run_tournament_selection(
            gene_symbol=gene_symbol,
            articles=articles,
            llm_processor=inference_engine,
            prompt_template=template,
            schema=schema,
            max_papers=max_papers,
            papers_per_round=papers_per_round,
            max_concurrent_batches=max_concurrent_batches,
            max_retries=max_retries,
        )

        selected_pmids = {a.pmid for a in tournament_outcome.selected_articles}
        all_selected_pmids.update(selected_pmids)

        logger.info(f"{gene_symbol}: {paper_count} -> {len(selected_pmids)} selected")

        _record_reduction_completion(db_path, gene_symbol, tournament_outcome)

    # Phase 2: Clear download_status for papers not selected by ANY gene
    logger.info("Clearing download_status for unselected papers...")
    clear_stats = clear_unselected_papers(db_path, all_selected_pmids)

    logger.info("Literature reduction complete!")
    logger.info(f"Total genes processed: {len(genes_with_counts)}")
    logger.info(f"Total papers selected: {len(all_selected_pmids)}")
    for key, value in clear_stats.items():
        logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    app()
