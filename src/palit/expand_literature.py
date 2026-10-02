#!/usr/bin/env python3
"""Tournament-based literature expansion using hierarchical LLM filtering."""

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import typer

from palit.gencc import GenccIndex, fetch_gencc, fetch_mondo
from palit.hgnc import HgncResolver
from palit.llm import BatchTransport, ImmediateTransport, Transport, make_client
from palit.llm_usage import print_stage_summary
from palit.panelapp_client import PanelAppClient
from palit.panelapp_publications import seed_panelapp_publications
from palit.papers import Paper, deserialize_source_metadata, serialize_source_metadata
from palit.tournament import TournamentEntry, TournamentOutcome, record_abandoned, run_tournaments

app = typer.Typer(help="Tournament-based literature expansion using hierarchical LLM filtering")
logger = logging.getLogger(__name__)


def get_papers_for_gene(
    baseline_db_path: Path, hgnc_id: int, limit: int, cutoff_date: str
) -> list[Paper]:
    """Fetch papers for a gene from baseline screening database.

    Args:
        baseline_db_path: Path to baseline screening SQLite database
        hgnc_id: HGNC ID of the gene to look up
        limit: Maximum number of papers to return
        cutoff_date: Only include papers with source_date <= this date (YYYY-MM-DD)

    Returns:
        List of up to `limit` newest papers mentioning this gene, using the "expansion" source type
    """
    with sqlite3.connect(baseline_db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT DISTINCT p.doi, p.pmid, p.title, p.abstract, p.authors, p.journal,
                   p.source, p.source_date, p.source_metadata
            FROM papers p
            JOIN gene_mentions gm ON p.doi = gm.paper_doi
            WHERE gm.hgnc_id = ?
              AND p.source_date <= ?
            ORDER BY p.source_date DESC, p.doi DESC
            LIMIT ?
            """,
            (hgnc_id, cutoff_date, limit),
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
                    source_type="expansion",
                    source_details=str(hgnc_id),
                )
            )

        logger.info(f"Found {len(papers)} papers for HGNC:{hgnc_id}")
        return papers


def store_expansion_papers(db_path: Path, papers: list[Paper], gene_symbol: str) -> None:
    """Store expansion papers in database.

    Args:
        db_path: Path to SQLite database
        papers: List of Paper objects to store
        gene_symbol: Gene symbol these papers were found for
    """
    logger.info(f"Storing {len(papers)} expansion papers for gene {gene_symbol}")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        new_papers = 0

        for paper in papers:
            cursor.execute(
                """
                INSERT INTO papers
                (doi, pmid, title, abstract, authors, journal, source, source_date, source_metadata,
                 source_type, source_details, download_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'scheduled')
                ON CONFLICT(doi) DO UPDATE SET
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
                    paper.doi,
                    paper.pmid,
                    paper.title,
                    paper.abstract,
                    paper.authors,
                    paper.journal,
                    paper.source,
                    paper.source_date,
                    serialize_source_metadata(paper.source_metadata),
                    paper.source_type,
                    paper.source_details,
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


STAGE = "expand_literature"


async def _process_expansion(
    *,
    transport: Transport,
    hgnc_resolver: HgncResolver,
    gencc: GenccIndex,
    genes: list[int],
    db_path: Path,
    baseline_db_path: Path,
    schema: dict[str, Any],
    template: str,
    paper_limit: int,
    cutoff_date: str,
    max_papers: int,
    papers_per_round: int,
    max_retries: int,
) -> None:
    """Run every gene's tournament, then store the selected papers."""
    record_abandoned(db_path, await transport.resume(STAGE))

    entries = []
    for hgnc_id in genes:
        papers = get_papers_for_gene(baseline_db_path, hgnc_id, paper_limit, cutoff_date)
        if not papers:
            logger.warning(
                f"No papers found for {hgnc_resolver.get_symbol(hgnc_id)} (HGNC:{hgnc_id})"
            )
            _record_expansion_completion(
                db_path, hgnc_id, TournamentOutcome(selected_papers=[], raw_responses_by_round=[])
            )
            continue
        entries.append(TournamentEntry.for_gene(hgnc_id, papers, hgnc_resolver, gencc))

    outcomes = await run_tournaments(
        entries,
        transport=transport,
        db_path=db_path,
        stage=STAGE,
        prompt_template=template,
        schema=schema,
        max_papers=max_papers,
        papers_per_round=papers_per_round,
        max_retries=max_retries,
    )

    total_papers_added = 0
    for entry in entries:
        outcome = outcomes[entry.key]
        logger.info(f"Selected {len(outcome.selected_papers)} papers for {entry.gene_symbol}")
        store_expansion_papers(db_path, outcome.selected_papers, entry.gene_symbol)
        total_papers_added += len(outcome.selected_papers)
        _record_expansion_completion(db_path, int(entry.key), outcome)

    logger.info("Literature expansion complete!")
    logger.info(f"Total genes processed: {len(genes)}")
    logger.info(f"Total papers added: {total_papers_added}")


@app.callback(invoke_without_command=True)
def main(
    cutoff_date: str = typer.Option(
        ...,
        "--cutoff-date",
        help="Only consider papers up to this date (YYYY-MM-DD)",
    ),
    panel_date: str = typer.Option(
        ...,
        "--panel-date",
        help="Panel state date (YYYY-MM-DD) whose snapshot supplies PanelApp's cited publications",
    ),
    target_panel_ids: list[int] | None = typer.Option(
        None,
        "--target-panel-ids",
        help="Panel IDs to source PanelApp publications from. Can be specified multiple times. Defaults to TARGET_PANEL_IDS.",
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
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of retries for failed batches",
    ),
    immediate: bool = typer.Option(
        False,
        "--immediate",
        help="Send requests immediately instead of as Message Batches (for prompt development)",
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
    paper_limit: int = typer.Option(
        10000,
        "--paper-limit",
        help="Maximum number of papers to consider per gene",
    ),
) -> None:
    """Expand literature for all genes with evidence, from two sources.

    Tournament selection picks a minimal, non-redundant set from the screened
    baseline. Seeding then adds, unconditionally, every publication PanelApp
    already cites for those genes — papers the tournament would discard as
    redundant, or that the baseline never contained.
    """
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

    # Load HGNC resolver for gene symbol lookup
    hgnc_resolver = HgncResolver.from_file()

    # Seed the publications PanelApp already cites. Runs after --force-all's
    # expansion wipe so seeded papers survive it, and before the tournament's
    # early return so a rerun still picks up publications added to a panel.
    panelapp_client = PanelAppClient(panel_date)
    panel_data = panelapp_client.get_target_panels_genes(target_panel_ids)
    seed_panelapp_publications(db_path, panelapp_client, panel_data, hgnc_resolver)

    logger.info(f"Found {len(genes)} gene(s) to expand")

    if not genes:
        logger.info("No genes require expansion")
        return

    gencc = fetch_gencc(db_path.parent, fetch_mondo(db_path.parent))

    async def run() -> None:
        client = make_client()
        transport: Transport = (
            ImmediateTransport(client) if immediate else BatchTransport(client, db_path)
        )
        await _process_expansion(
            transport=transport,
            hgnc_resolver=hgnc_resolver,
            gencc=gencc,
            genes=genes,
            db_path=db_path,
            baseline_db_path=baseline_db_path,
            schema=schema,
            template=template,
            paper_limit=paper_limit,
            cutoff_date=cutoff_date,
            max_papers=max_papers,
            papers_per_round=papers_per_round,
            max_retries=max_retries,
        )

    asyncio.run(run())
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
