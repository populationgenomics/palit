#!/usr/bin/env python3
"""
Extract the DOIs the relevance scope screen passed, as classifier positives.

The classifier pre-filters papers ahead of the scope screen, so its positives
are the screen's decisions, not the final ones (which also depend on the
month's PanelApp state).
"""

import json
import logging
import sqlite3
from pathlib import Path

import typer

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def extract(
    db_path: Path = typer.Option(Path("data/db.sqlite"), "--db-path", help="Path to main database"),
    output_file: Path = typer.Option(
        Path("data/relevant_dois.txt").expanduser(),
        "--output",
        "-o",
        help="Output file for positive DOIs",
    ),
) -> None:
    """Extract DOIs the relevance scope screen passed."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    logger.info(f"Reading from {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all papers with completed assessments
    cursor.execute("""
        SELECT doi, relevance_assessment_json
        FROM papers
        WHERE relevance_assessment_json IS NOT NULL
    """)

    positive_dois = []
    total_assessed = 0

    for doi, json_str in cursor.fetchall():
        total_assessed += 1
        # A paper both models refused at the screen has no screen; one refused at the
        # PanelApp check passed the screen and counts as a positive.
        screen = json.loads(json_str)["screen"]
        if screen is not None and screen["relevant"]:
            positive_dois.append(doi)

    conn.close()

    # Write to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for doi in sorted(positive_dois):
            f.write(f"{doi}\n")

    logger.info(f"Total assessed papers: {total_assessed:,}")
    logger.info(f"Positive papers (scope screen): {len(positive_dois):,}")
    logger.info(f"Positive rate: {len(positive_dois) / total_assessed * 100:.2f}%")
    logger.info(f"Written to {output_file}")


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
