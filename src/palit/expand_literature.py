#!/usr/bin/env python3
"""Tournament-based literature expansion using hierarchical LLM filtering."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import typer
from tqdm import tqdm

from palit.hgnc import HgncResolver
from palit.ingest_pubmed import Article
from palit.llm import HarmonyBatchProcessor
from palit.tournament import TournamentOutcome, run_tournament_selection

app = typer.Typer(help="Tournament-based literature expansion using hierarchical LLM filtering")
logger = logging.getLogger(__name__)


def get_articles_for_gene(
    baseline_db_path: Path, hgnc_id: int, limit: int, cutoff_date: str
) -> list[Article]:
    """Fetch articles for a gene from baseline screening database.

    Args:
        baseline_db_path: Path to baseline screening SQLite database
        hgnc_id: HGNC ID of the gene to look up
        limit: Maximum number of articles to return
        cutoff_date: Only include papers with entrez_date <= this date (YYYY-MM-DD)

    Returns:
        List of up to `limit` newest articles mentioning this gene, using the "expansion" source type
    """
    with sqlite3.connect(baseline_db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT p.pmid, p.title, p.abstract, p.authors, p.journal, p.entrez_date
            FROM papers p
            JOIN gene_mentions gm ON p.pmid = gm.pmid
            WHERE gm.hgnc_id = ?
              AND p.entrez_date <= ?
            ORDER BY p.entrez_date DESC, p.pmid DESC
            LIMIT ?
            """,
            (hgnc_id, cutoff_date, limit),
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
                    source_type="expansion",
                    source_details=str(hgnc_id),
                )
            )

        logger.info(f"Found {len(articles)} articles for HGNC:{hgnc_id}")
        return articles


def store_expansion_papers(db_path: Path, articles: list[Article], gene_symbol: str) -> None:
    """Store expansion papers in database.

    Args:
        db_path: Path to SQLite database
        articles: List of Article objects to store
        gene_symbol: Gene symbol these papers were found for
    """
    logger.info(f"Storing {len(articles)} expansion papers for gene {gene_symbol}")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        new_papers = 0

        for article in articles:
            cursor.execute(
                """
                INSERT INTO papers
                (pmid, title, abstract, authors, journal, entrez_date, source_type, source_details, download_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'scheduled')
                ON CONFLICT(pmid) DO UPDATE SET
                    source_type = excluded.source_type,
                    source_details = excluded.source_details,
                    download_status = CASE
                        WHEN papers.download_status IS NULL THEN excluded.download_status
                        ELSE papers.download_status
                    END
                WHERE papers.source_type = 'expansion'
                  AND excluded.source_type = 'expansion'
                  AND excluded.source_details > papers.source_details
                """,
                (
                    article.pmid,
                    article.title,
                    article.abstract,
                    article.authors,
                    article.journal,
                    article.entrez_date,
                    article.source_type,
                    article.source_details,
                ),
            )

            if cursor.rowcount > 0:
                new_papers += 1

        conn.commit()
        logger.info(f"Added {new_papers} new expansion papers")


def _record_expansion_completion(db_path: Path, hgnc_id: int, outcome: TournamentOutcome) -> None:
    """Persist tournament results so resumable runs skip completed genes."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO tournament_results (
                hgnc_id,
                selected_pmids_json,
                tournament_raw_responses_json
            )
            VALUES (?, ?, ?)
            ON CONFLICT(hgnc_id) DO UPDATE SET
                selected_pmids_json = excluded.selected_pmids_json,
                tournament_raw_responses_json = excluded.tournament_raw_responses_json
            """,
            (
                hgnc_id,
                json.dumps([a.pmid for a in outcome.selected_articles]),
                json.dumps(outcome.raw_responses_by_round),
            ),
        )
        conn.commit()


@app.callback(invoke_without_command=True)
def main(
    cutoff_date: str = typer.Option(
        ...,
        "--cutoff-date",
        help="Only consider papers up to this date (YYYY-MM-DD)",
    ),
    db_path: Path = typer.Option(
        Path("data/db.sqlite"),
        "--db-path",
        help="Path to main SQLite database",
    ),
    baseline_db_path: Path = typer.Option(
        Path("data/pubmed_baseline_screening.sqlite"),
        "--baseline-db-path",
        help="Path to baseline screening database",
    ),
    max_papers: int = typer.Option(
        20,
        "--max-papers",
        help="Maximum papers to select per gene",
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
    force_gene: list[int] = typer.Option(
        [],
        "--force-gene",
        help="Re-run expansion for specific gene (HGNC ID)",
    ),
    force_all: bool = typer.Option(
        False,
        "--force-all",
        help="Re-run expansion for all genes",
    ),
    pmid_limit: int = typer.Option(
        10000,
        "--pmid-limit",
        help="Maximum number of papers to consider per gene",
    ),
) -> None:
    """Expand literature using tournament selection for all genes with evidence."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if not baseline_db_path.exists():
        logger.error(f"Baseline database not found: {baseline_db_path}")
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

    # Determine genes to process
    logger.info("Preparing expansion gene list...")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        if force_all:
            logger.info("force-all enabled: clearing expansion data")
            cursor.execute("DELETE FROM gene_mentions WHERE source = 'expansion_evidence'")
            cursor.execute("DELETE FROM papers WHERE source_type = 'expansion'")
            cursor.execute("DELETE FROM tournament_results")
            conn.commit()

        if force_gene:
            logger.info(f"Forcing reprocessing for {len(force_gene)} gene(s)")
            cursor.executemany(
                "DELETE FROM tournament_results WHERE hgnc_id = ?",
                [(hgnc_id,) for hgnc_id in force_gene],
            )
            conn.commit()

        cursor.execute(
            """
            SELECT DISTINCT gm.hgnc_id
            FROM gene_mentions gm
            LEFT JOIN tournament_results er
              ON gm.hgnc_id = er.hgnc_id
            WHERE gm.source = 'recent_evidence'
              AND er.hgnc_id IS NULL
            ORDER BY gm.hgnc_id
            """
        )
        genes = [row[0] for row in cursor.fetchall()]

    logger.info(f"Found {len(genes)} gene(s) to expand")

    if not genes:
        logger.info("No genes require expansion")
        return

    # Load HGNC resolver for gene symbol lookup
    hgnc_resolver = HgncResolver.from_file()

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

    total_papers_added = 0

    for hgnc_id in tqdm(genes, desc="Expanding literature for genes"):
        hgnc_symbol = hgnc_resolver.get_symbol(hgnc_id)
        articles = get_articles_for_gene(baseline_db_path, hgnc_id, pmid_limit, cutoff_date)

        if not articles:
            logger.warning(f"No articles found for {hgnc_symbol} (HGNC:{hgnc_id})")
            _record_expansion_completion(
                db_path,
                hgnc_id,
                TournamentOutcome(selected_articles=[], raw_responses_by_round=[]),
            )
            continue

        tournament_outcome = run_tournament_selection(
            gene_symbol=hgnc_symbol,
            articles=articles,
            llm_processor=inference_engine,
            prompt_template=template,
            schema=schema,
            max_papers=max_papers,
            papers_per_round=papers_per_round,
            max_concurrent_batches=max_concurrent_batches,
            max_retries=max_retries,
        )

        logger.info(
            f"Selected {len(tournament_outcome.selected_articles)} papers for {hgnc_symbol}"
        )

        store_expansion_papers(db_path, tournament_outcome.selected_articles, hgnc_symbol)
        total_papers_added += len(tournament_outcome.selected_articles)

        _record_expansion_completion(db_path, hgnc_id, tournament_outcome)

    logger.info("Literature expansion complete!")
    logger.info(f"Total genes processed: {len(genes)}")
    logger.info(f"Total papers added: {total_papers_added}")


if __name__ == "__main__":
    app()
