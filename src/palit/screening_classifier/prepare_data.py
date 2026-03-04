#!/usr/bin/env python3
"""
Prepare training data for relevance screening classifier.

Steps:
1. Read positive DOIs from text file
2. Update matching DOIs in DB to is_relevant=1
3. Fetch any missing positive DOIs using efetch (batch mode via PMID lookup)
4. Assign stratified train/val/test splits (70/20/10)
"""

import logging
import random
import sqlite3
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

console = Console()
logger = logging.getLogger(__name__)
app = typer.Typer()

BATCH_SIZE = 500  # Fetch papers in batches to avoid API limits


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    log_levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING}
    logging.basicConfig(
        level=log_levels.get(level.upper(), logging.INFO), format="%(levelname)s: %(message)s"
    )


def read_positive_dois(doi_file: Path) -> set[str]:
    """Read positive DOIs from text file."""
    console.print(f"[bold]Step 1: Reading positive DOIs from {doi_file}...[/bold]")

    if not doi_file.exists():
        logger.error(f"Positive DOI file not found: {doi_file}")
        raise typer.Exit(1)

    positive_dois = set()
    with open(doi_file) as f:
        for line in f:
            line = line.strip()
            if line:
                positive_dois.add(line)

    console.print(f"[green]✓[/green] Read {len(positive_dois):,} positive DOIs")
    return positive_dois


def update_existing_positives(positive_dois: set[str], conn: sqlite3.Connection) -> set[str]:
    """
    Update existing papers to is_relevant=1 for positive DOIs.

    Returns missing_dois.
    """
    console.print("[bold]Step 2: Labeling positive examples in database...[/bold]")

    cursor = conn.cursor()

    # Find which positive DOIs are already in DB
    placeholders = ",".join("?" * len(positive_dois))
    cursor.execute(f"SELECT doi FROM papers WHERE doi IN ({placeholders})", list(positive_dois))
    existing_dois = {row[0] for row in cursor.fetchall()}

    # Update them to is_relevant=1
    if existing_dois:
        placeholders = ",".join("?" * len(existing_dois))
        cursor.execute(
            f"UPDATE papers SET is_relevant = 1 WHERE doi IN ({placeholders})",
            list(existing_dois),
        )
        conn.commit()

    missing_dois = positive_dois - existing_dois

    console.print(f"[green]✓[/green] Labeled {len(existing_dois):,} positive papers")
    if missing_dois:
        console.print(
            f"[yellow]⚠[/yellow] {len(missing_dois):,} positive DOIs not in database (will fetch)"
        )

    return missing_dois


def fetch_papers_batch(dois: list[str], main_db_path: Path) -> dict[str, tuple[str, str | None]]:
    """
    Fetch title and abstract for a batch of DOIs using efetch.

    Looks up PMIDs from the main database's source_metadata, then fetches via efetch.
    Returns dict of {doi: (title, abstract)}.
    """
    if not dois:
        return {}

    # Look up PMIDs from main database
    main_conn = sqlite3.connect(main_db_path)
    main_cursor = main_conn.cursor()
    placeholders = ",".join("?" * len(dois))
    main_cursor.execute(
        f"SELECT doi, pmid FROM papers WHERE doi IN ({placeholders}) AND pmid IS NOT NULL",
        dois,
    )
    doi_to_pmid = dict(main_cursor.fetchall())
    main_conn.close()

    if not doi_to_pmid:
        logger.warning(f"No PMIDs found in main DB for {len(dois)} DOIs")
        return {}

    pmid_to_doi = {pmid: doi for doi, pmid in doi_to_pmid.items()}
    pmid_str = ",".join(str(p) for p in doi_to_pmid.values())

    try:
        result = subprocess.run(
            ["efetch", "-db", "pubmed", "-id", pmid_str, "-format", "xml"],
            capture_output=True,
            check=True,
            text=True,
            timeout=60,
        )

        root = ET.fromstring(result.stdout)

        results: dict[str, tuple[str, str | None]] = {}
        for paper_elem in root.findall(".//PubmedArticle"):
            pmid_elem = paper_elem.find(".//PMID")
            if pmid_elem is None or pmid_elem.text is None:
                continue
            pmid = int(pmid_elem.text)

            doi = pmid_to_doi.get(pmid)
            if doi is None:
                continue

            title_elem = paper_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None and title_elem.text else ""

            abstract_parts = []
            for abstract_text in paper_elem.findall(".//AbstractText"):
                if abstract_text.text:
                    abstract_parts.append(abstract_text.text)
            abstract = " ".join(abstract_parts) if abstract_parts else None

            results[doi] = (title, abstract)

        return results

    except (subprocess.CalledProcessError, ET.ParseError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to fetch batch of {len(dois)} papers: {e}")
        return {}


def fetch_missing_papers(
    missing_dois: set[str], conn: sqlite3.Connection, main_db_path: Path
) -> None:
    """Fetch missing papers in batches and add to database."""
    if not missing_dois:
        console.print("[green]✓[/green] No missing papers to fetch")
        return

    console.print(f"[bold]Step 3: Fetching {len(missing_dois):,} missing papers...[/bold]")

    missing_list = sorted(missing_dois)
    cursor = conn.cursor()

    fetched_count = 0
    failed_dois: list[str] = []

    with Progress(console=console) as progress:
        task = progress.add_task("Fetching papers...", total=len(missing_list))

        for i in range(0, len(missing_list), BATCH_SIZE):
            batch = missing_list[i : i + BATCH_SIZE]
            results = fetch_papers_batch(batch, main_db_path)

            for doi, (title, abstract) in results.items():
                cursor.execute(
                    "INSERT OR IGNORE INTO papers (doi, title, abstract, is_relevant) VALUES (?, ?, ?, 1)",
                    (doi, title, abstract),
                )
                fetched_count += 1

            failed_dois.extend(doi for doi in batch if doi not in results)

            conn.commit()
            progress.advance(task, advance=len(batch))

    if failed_dois:
        console.print(
            f"[yellow]⚠[/yellow] Failed to fetch {len(failed_dois):,} papers "
            f"(will be excluded from training)"
        )
        failed_file = Path("data/failed_dois.txt")
        with open(failed_file, "w") as f:
            for doi in sorted(failed_dois):
                f.write(f"{doi}\n")
        logger.info(f"Failed DOIs written to {failed_file}")

    console.print(f"[green]✓[/green] Fetched {fetched_count:,} missing papers")


def assign_splits(
    conn: sqlite3.Connection, train_ratio: float = 0.7, val_ratio: float = 0.2
) -> None:
    """
    Assign stratified train/val/test splits.

    Ensures similar positive:negative ratio in each split.
    """
    console.print("[bold]Step 4: Assigning train/val/test splits (70/20/10)...[/bold]")

    cursor = conn.cursor()

    # Get all DOIs, stratified by label
    cursor.execute("SELECT doi FROM papers WHERE is_relevant = 1")
    positive_dois = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT doi FROM papers WHERE is_relevant = 0")
    negative_dois = [row[0] for row in cursor.fetchall()]

    # Shuffle
    random.shuffle(positive_dois)
    random.shuffle(negative_dois)

    # Calculate split sizes
    n_pos = len(positive_dois)
    n_neg = len(negative_dois)

    pos_train_end = int(n_pos * train_ratio)
    pos_val_end = int(n_pos * (train_ratio + val_ratio))

    neg_train_end = int(n_neg * train_ratio)
    neg_val_end = int(n_neg * (train_ratio + val_ratio))

    # Assign splits
    splits = {
        "train": positive_dois[:pos_train_end] + negative_dois[:neg_train_end],
        "val": positive_dois[pos_train_end:pos_val_end] + negative_dois[neg_train_end:neg_val_end],
        "test": positive_dois[pos_val_end:] + negative_dois[neg_val_end:],
    }

    # Update database in batches to avoid SQL limits
    for split_name, dois in splits.items():
        chunk_size = 1000
        for i in range(0, len(dois), chunk_size):
            chunk = dois[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            cursor.execute(
                f"UPDATE papers SET split = ? WHERE doi IN ({placeholders})",
                [split_name, *chunk],
            )
        conn.commit()

    # Report statistics
    console.print("")
    for split_name in ["train", "val", "test"]:
        cursor.execute(
            "SELECT COUNT(*), SUM(is_relevant) FROM papers WHERE split = ?", (split_name,)
        )
        total, positives = cursor.fetchone()
        negatives = total - positives
        pos_rate = positives / total * 100 if total > 0 else 0
        console.print(
            f"  {split_name:5s}: {total:7,} papers "
            f"({positives:5,} pos / {negatives:7,} neg = {pos_rate:.2f}%)"
        )

    console.print("[green]✓[/green] Splits assigned")


@app.command()
def prepare(
    positive_dois_file: Path = typer.Option(
        Path("data/relevant_dois.txt").expanduser(),
        "--positive-dois",
        help="Text file with positive DOIs (one per line)",
    ),
    main_db_path: Path = typer.Option(
        Path("data/db.sqlite"),
        "--main-db-path",
        help="Main database path (for PMID lookups during fetch)",
    ),
    db_path: Path = typer.Option(
        Path("data/screening_classifier/training.sqlite"),
        "--db-path",
        help="Training database path",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for split assignment"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Prepare training data for relevance screening classifier."""
    setup_logging(log_level)
    random.seed(seed)

    # Validate inputs
    if not db_path.exists():
        console.print(f"[red]Database not found: {db_path}[/red]")
        console.print("[yellow]Did you run the database merge first?[/yellow]")
        raise typer.Exit(1)

    console.print(f"[cyan]Preparing training data in {db_path}[/cyan]\n")

    # Connect to database
    conn = sqlite3.connect(db_path)

    # Execute pipeline
    positive_dois = read_positive_dois(positive_dois_file)
    missing_dois = update_existing_positives(positive_dois, conn)
    fetch_missing_papers(missing_dois, conn, main_db_path)
    assign_splits(conn)

    # Final statistics
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(is_relevant) FROM papers")
    total, positives = cursor.fetchone()
    negatives = total - positives

    conn.close()

    console.print("\n[green]✓ Data preparation complete![/green]")
    console.print(
        f"[green]Total papers: {total:,} ({positives:,} positive, {negatives:,} negative)[/green]"
    )
    console.print(f"[green]Database: {db_path.absolute()}[/green]")


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
