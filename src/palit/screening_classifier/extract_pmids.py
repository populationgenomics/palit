#!/usr/bin/env python3
"""
Extract PMIDs with positive majority vote from main database.

Reads relevance_assessment_json (array of 3 assessments) from data/db.sqlite,
computes majority vote, and writes positive PMIDs to output file.
"""

import json
import logging
import sqlite3
from pathlib import Path

import typer

logger = logging.getLogger(__name__)
app = typer.Typer()


def compute_relevance_majority_vote(assessments: list[dict]) -> bool:
    """
    Compute majority vote from 3 relevance assessments.
    Returns True if majority says relevant, False otherwise.
    """
    if len(assessments) != 3:
        raise ValueError(f"Expected exactly 3 assessments, got {len(assessments)}")

    relevant_votes = sum(1 for a in assessments if a["relevant"])
    return relevant_votes >= 2


@app.command()
def extract(
    db_path: Path = typer.Option(Path("data/db.sqlite"), "--db-path", help="Path to main database"),
    output_file: Path = typer.Option(
        Path("data/relevant_pmids.txt").expanduser(),
        "--output",
        "-o",
        help="Output file for positive PMIDs",
    ),
) -> None:
    """Extract PMIDs with positive majority vote from database."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    logger.info(f"Reading from {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all papers with completed assessments
    cursor.execute("""
        SELECT pmid, relevance_assessment_json
        FROM papers
        WHERE relevance_assessment_json IS NOT NULL
    """)

    positive_pmids = []
    total_assessed = 0

    for pmid, json_str in cursor.fetchall():
        total_assessed += 1
        assessments = json.loads(json_str)

        # Skip if not exactly 3 assessments
        if len(assessments) != 3:
            logger.warning(f"PMID {pmid}: Expected 3 assessments, got {len(assessments)}, skipping")
            continue

        if compute_relevance_majority_vote(assessments):
            positive_pmids.append(pmid)

    conn.close()

    # Write to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for pmid in sorted(positive_pmids):
            f.write(f"{pmid}\n")

    logger.info(f"Total assessed papers: {total_assessed:,}")
    logger.info(f"Positive papers (majority vote): {len(positive_pmids):,}")
    logger.info(f"Positive rate: {len(positive_pmids) / total_assessed * 100:.2f}%")
    logger.info(f"Written to {output_file}")


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
