#!/usr/bin/env python3
"""
Prepare training data for relevance screening classifier.

Steps:
1. Read positive PMIDs from text file
2. Update matching PMIDs in DB to is_relevant=1
3. Fetch any missing positive PMIDs using efetch (batch mode)
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

BATCH_SIZE = 500  # Fetch PMIDs in batches to avoid API limits


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    log_levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING}
    logging.basicConfig(
        level=log_levels.get(level.upper(), logging.INFO), format="%(levelname)s: %(message)s"
    )


def read_positive_pmids(pmid_file: Path) -> set[int]:
    """Read positive PMIDs from text file."""
    console.print(f"[bold]Step 1: Reading positive PMIDs from {pmid_file}...[/bold]")

    if not pmid_file.exists():
        logger.error(f"Positive PMID file not found: {pmid_file}")
        raise typer.Exit(1)

    positive_pmids = set()
    with open(pmid_file) as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                positive_pmids.add(int(line))

    console.print(f"[green]✓[/green] Read {len(positive_pmids):,} positive PMIDs")
    return positive_pmids


def update_existing_positives(positive_pmids: set[int], conn: sqlite3.Connection) -> set[int]:
    """
    Update existing papers to is_relevant=1 for positive PMIDs.

    Returns missing_pmids.
    """
    console.print("[bold]Step 2: Labeling positive examples in database...[/bold]")

    cursor = conn.cursor()

    # Find which positive PMIDs are already in DB
    placeholders = ",".join("?" * len(positive_pmids))
    cursor.execute(f"SELECT pmid FROM papers WHERE pmid IN ({placeholders})", list(positive_pmids))
    existing_pmids = {row[0] for row in cursor.fetchall()}

    # Update them to is_relevant=1
    if existing_pmids:
        placeholders = ",".join("?" * len(existing_pmids))
        cursor.execute(
            f"UPDATE papers SET is_relevant = 1 WHERE pmid IN ({placeholders})",
            list(existing_pmids),
        )
        conn.commit()

    missing_pmids = positive_pmids - existing_pmids

    console.print(f"[green]✓[/green] Labeled {len(existing_pmids):,} positive papers")
    if missing_pmids:
        console.print(
            f"[yellow]⚠[/yellow] {len(missing_pmids):,} positive PMIDs not in database (will fetch)"
        )

    return missing_pmids


def fetch_pmids_batch(pmids: list[int]) -> dict[int, tuple[str, str | None]]:
    """
    Fetch title and abstract for a batch of PMIDs using efetch.

    Returns dict of {pmid: (title, abstract)}.
    Skips PMIDs that fail to fetch.
    """
    if not pmids:
        return {}

    # Convert PMIDs to comma-separated string
    pmid_str = ",".join(str(p) for p in pmids)

    try:
        # Use efetch to get XML
        result = subprocess.run(
            ["efetch", "-db", "pubmed", "-id", pmid_str, "-format", "xml"],
            capture_output=True,
            check=True,
            text=True,
            timeout=60,
        )

        xml_data = result.stdout

        # Parse XML
        root = ET.fromstring(xml_data)

        results = {}
        for paper_elem in root.findall(".//PubmedArticle"):
            # Extract PMID
            pmid_elem = paper_elem.find(".//PMID")
            if pmid_elem is None or pmid_elem.text is None:
                continue
            pmid = int(pmid_elem.text)

            # Extract title
            title_elem = paper_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None and title_elem.text else ""

            # Extract abstract
            abstract_parts = []
            for abstract_text in paper_elem.findall(".//AbstractText"):
                if abstract_text.text:
                    abstract_parts.append(abstract_text.text)
            abstract = " ".join(abstract_parts) if abstract_parts else None

            results[pmid] = (title, abstract)

        return results

    except (subprocess.CalledProcessError, ET.ParseError, subprocess.TimeoutExpired) as e:
        logger.warning(f"Failed to fetch batch of {len(pmids)} PMIDs: {e}")
        return {}


def fetch_missing_pmids(missing_pmids: set[int], conn: sqlite3.Connection) -> None:
    """Fetch missing PMIDs in batches and add to database."""
    if not missing_pmids:
        console.print("[green]✓[/green] No missing PMIDs to fetch")
        return

    console.print(f"[bold]Step 3: Fetching {len(missing_pmids):,} missing PMIDs...[/bold]")

    missing_list = sorted(missing_pmids)
    cursor = conn.cursor()

    fetched_count = 0
    failed_pmids: list[int] = []

    with Progress(console=console) as progress:
        task = progress.add_task("Fetching PMIDs...", total=len(missing_list))

        for i in range(0, len(missing_list), BATCH_SIZE):
            batch = missing_list[i : i + BATCH_SIZE]
            results = fetch_pmids_batch(batch)

            # Insert fetched papers
            for pmid, (title, abstract) in results.items():
                cursor.execute(
                    "INSERT OR IGNORE INTO papers (pmid, title, abstract, is_relevant) VALUES (?, ?, ?, 1)",
                    (pmid, title, abstract),
                )
                fetched_count += 1

            # Track failed PMIDs
            failed_pmids.extend(pmid for pmid in batch if pmid not in results)

            conn.commit()
            progress.advance(task, advance=len(batch))

    if failed_pmids:
        console.print(
            f"[yellow]⚠[/yellow] Failed to fetch {len(failed_pmids):,} PMIDs "
            f"(will be excluded from training)"
        )
        # Write failed PMIDs to log file
        failed_file = Path("data/failed_pmids.txt")
        with open(failed_file, "w") as f:
            for pmid in sorted(failed_pmids):
                f.write(f"{pmid}\n")
        logger.info(f"Failed PMIDs written to {failed_file}")

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

    # Get all PMIDs, stratified by label
    cursor.execute("SELECT pmid FROM papers WHERE is_relevant = 1")
    positive_pmids = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT pmid FROM papers WHERE is_relevant = 0")
    negative_pmids = [row[0] for row in cursor.fetchall()]

    # Shuffle
    random.shuffle(positive_pmids)
    random.shuffle(negative_pmids)

    # Calculate split sizes
    n_pos = len(positive_pmids)
    n_neg = len(negative_pmids)

    pos_train_end = int(n_pos * train_ratio)
    pos_val_end = int(n_pos * (train_ratio + val_ratio))

    neg_train_end = int(n_neg * train_ratio)
    neg_val_end = int(n_neg * (train_ratio + val_ratio))

    # Assign splits
    splits = {
        "train": positive_pmids[:pos_train_end] + negative_pmids[:neg_train_end],
        "val": positive_pmids[pos_train_end:pos_val_end]
        + negative_pmids[neg_train_end:neg_val_end],
        "test": positive_pmids[pos_val_end:] + negative_pmids[neg_val_end:],
    }

    # Update database in batches to avoid SQL limits
    for split_name, pmids in splits.items():
        # Process in chunks of 1000 to avoid SQLITE_MAX_VARIABLE_NUMBER limit
        chunk_size = 1000
        for i in range(0, len(pmids), chunk_size):
            chunk = pmids[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            cursor.execute(
                f"UPDATE papers SET split = ? WHERE pmid IN ({placeholders})",
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
    positive_pmids_file: Path = typer.Option(
        Path("data/relevant_pmids.txt").expanduser(),
        "--positive-pmids",
        help="Text file with positive PMIDs (one per line)",
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
    positive_pmids = read_positive_pmids(positive_pmids_file)
    missing_pmids = update_existing_positives(positive_pmids, conn)
    fetch_missing_pmids(missing_pmids, conn)
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
