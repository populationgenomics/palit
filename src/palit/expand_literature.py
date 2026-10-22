#!/usr/bin/env python3
"""Literature expansion for a run's genes: the publications PanelApp cites, and in
monthly runs tournament selection over the screened baseline using hierarchical LLM
filtering."""

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import typer

from palit.hgnc import HgncResolver
from palit.llm import BatchTransport, ImmediateTransport, Transport, make_client
from palit.llm_usage import print_stage_summary
from palit.panelapp_client import DEFAULT_SNAPSHOT_DIR
from palit.panelapp_publications import BASELINE_DB_PATH, seed_panelapp_publications
from palit.panelapp_snapshot import load_panels_with_reviews
from palit.papers import Paper, deserialize_source_metadata, serialize_source_metadata
from palit.run_corpus import RUN_GENES
from palit.tournament import TournamentEntry, TournamentOutcome, record_abandoned, run_tournaments

app = typer.Typer(
    help="Expand the run's literature with the publications PanelApp cites, and in monthly "
    "runs with tournament selection over the screened baseline"
)
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
                   p.journal_issn, p.source, p.source_date, p.source_metadata
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
                    journal_issn=row["journal_issn"],
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
                (doi, pmid, title, abstract, authors, journal, journal_issn, source, source_date,
                 source_metadata, source_type, source_details, download_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'scheduled')
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
                    paper.journal_issn,
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


def genes_to_expand(db_path: Path) -> list[int]:
    """The run's genes without a tournament result, in HGNC ID order.

    The run's genes have recent evidence from a relevant paper (see
    :mod:`palit.run_corpus`).
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT hgnc_id FROM ({RUN_GENES})
            WHERE hgnc_id NOT IN (SELECT hgnc_id FROM tournament_results)
            ORDER BY hgnc_id
            """
        ).fetchall()
    return [row[0] for row in rows]


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
        entries.append(TournamentEntry.for_gene(hgnc_id, papers, hgnc_resolver))

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
    panel_date: str = typer.Option(
        ...,
        "--panel-date",
        help="Panel state date (YYYY-MM-DD) whose snapshot supplies PanelApp's cited publications",
    ),
    scope_panel_id: int | None = typer.Option(
        None,
        "--scope-panel-id",
        help="Panel ID of a panel-scoped run, whose first corpus is already the whole screened "
        "baseline: expansion seeds PanelApp's citations and runs no tournament. Exclusive with "
        "--cutoff-date.",
    ),
    cutoff_date: str | None = typer.Option(
        None,
        "--cutoff-date",
        help="Tournament (monthly runs): only consider papers up to this date (YYYY-MM-DD). "
        "Required without --scope-panel-id.",
    ),
    db_path: Path = typer.Option(
        Path("data/db.sqlite"),
        "--db-path",
        help="Path to main SQLite database",
    ),
    baseline_db_path: Path = typer.Option(
        BASELINE_DB_PATH,
        "--baseline-db-path",
        help="Path to baseline screening database: the tournament's candidates, and the "
        "source of PanelApp-cited papers before PubMed and CrossRef",
    ),
    max_papers: int = typer.Option(
        20,
        "--max-papers",
        help="Maximum papers the tournament selects per gene",
    ),
    papers_per_round: int = typer.Option(
        100,
        "--papers-per-round",
        help="Papers to show LLM in each tournament round",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of retries for failed tournament batches",
    ),
    immediate: bool = typer.Option(
        False,
        "--immediate",
        help="Send the tournament's requests immediately instead of as Message Batches "
        "(for prompt development)",
    ),
    force_gene: list[int] = typer.Option(
        [],
        "--force-gene",
        help="Re-run the tournament for a specific gene (HGNC ID)",
    ),
    force_all: bool = typer.Option(
        False,
        "--force-all",
        help="Clear all expansion results (tournament picks, seeded papers and their citation "
        "links) and expand again",
    ),
    paper_limit: int = typer.Option(
        10000,
        "--paper-limit",
        help="Maximum number of papers the tournament considers per gene",
    ),
) -> None:
    """Expand the literature of the run's genes beyond the run's first corpus.

    Seeding adds, unconditionally, every publication PanelApp already cites for
    those genes on any panel, and records which genes cite each held paper (see
    :mod:`palit.panelapp_publications`). In a monthly run, tournament selection
    then picks a minimal, non-redundant set of each gene's papers up to
    --cutoff-date from the screened baseline. Seeding keeps the cited papers the
    tournament would discard as redundant, and those the baseline never contained.

    A panel-scoped run (--scope-panel-id) only seeds. Its first corpus is already
    the whole baseline, screened against the panel, so a tournament over the
    baseline has no candidates to add.
    """
    if scope_panel_id is None and cutoff_date is None:
        logger.error("--cutoff-date is required without --scope-panel-id")
        raise typer.Exit(1)

    if scope_panel_id is not None and cutoff_date is not None:
        logger.error(
            "--cutoff-date bounds the tournament's candidates, and a panel-scoped run "
            "(--scope-panel-id) runs no tournament"
        )
        raise typer.Exit(1)

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if not baseline_db_path.exists():
        logger.error(f"Baseline database not found: {baseline_db_path}")
        raise typer.Exit(1)

    panels = load_panels_with_reviews(panel_date, DEFAULT_SNAPSHOT_DIR)
    if scope_panel_id is not None and scope_panel_id not in panels:
        logger.error(f"Panel {scope_panel_id} is not in the {panel_date} snapshot")
        raise typer.Exit(1)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        if force_all:
            logger.info("force-all enabled: clearing expansion data")
            cursor.execute("DELETE FROM gene_mentions WHERE source = 'expansion_evidence'")
            cursor.execute("DELETE FROM panelapp_citations")
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

    hgnc_resolver = HgncResolver.from_file()

    # Seed the publications PanelApp already cites. Runs after --force-all's
    # expansion wipe so seeded papers survive it, and before the tournament's
    # early return so a rerun still picks up publications added to a panel.
    seed_panelapp_publications(db_path, panels, baseline_db_path, hgnc_resolver)

    if scope_panel_id is not None:
        logger.info(f"Panel-scoped run for panel {scope_panel_id}: seeding is the whole expansion")
        return
    assert cutoff_date is not None

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

    genes = genes_to_expand(db_path)
    logger.info(f"Found {len(genes)} gene(s) to expand")

    if not genes:
        logger.info("No genes require expansion")
        return

    async def run() -> None:
        client = make_client()
        transport: Transport = (
            ImmediateTransport(client) if immediate else BatchTransport(client, db_path)
        )
        await _process_expansion(
            transport=transport,
            hgnc_resolver=hgnc_resolver,
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
