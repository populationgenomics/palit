#!/usr/bin/env python3
"""
Screen PubMed baseline snapshot through trained relevance classifier.

Processes all PubMed baseline XML files, classifies papers for relevance,
and stores flagged papers in a new database.
"""

import gzip
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from palit.ingest_pubmed import extract_papers_from_xml
from palit.papers import Paper, serialize_source_metadata
from palit.progress import LoggingProgress as Progress
from palit.screening_classifier.inference import (
    LabeledPaper,
    LoadedCheckpoint,
    PaperDataset,
    load_checkpoint,
)

console = Console()
logger = logging.getLogger(__name__)
app = typer.Typer(help="Screen PubMed baseline through trained relevance classifier")


@dataclass
class ScreeningStats:
    """Statistics for screening progress."""

    total_files: int
    files_completed: int
    total_papers: int
    papers_relevant: int
    start_time: float
    current_throughput: float = 0.0


def init_database(db_path: Path, schema_path: Path) -> None:
    """Initialize screening database from schema if it doesn't exist."""
    if db_path.exists():
        return

    console.print(f"[yellow]Creating new database: {db_path}[/yellow]")
    conn = sqlite3.connect(db_path)

    with open(schema_path) as f:
        schema_sql = f.read()

    conn.executescript(schema_sql)
    conn.commit()
    conn.close()
    console.print("[green]Database created successfully[/green]\n")


def load_progress(progress_path: Path) -> dict[str, dict[str, str | None]]:
    """Load progress from JSON file."""
    if not progress_path.exists():
        return {}

    with open(progress_path) as f:
        progress: dict[str, dict[str, str | None]] = json.load(f)
        return progress


def save_progress(progress_path: Path, progress: dict[str, dict[str, str | None]]) -> None:
    """Save progress to JSON file."""
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def get_processed_files(progress_path: Path) -> set[str]:
    """Get set of filenames that have been successfully processed."""
    progress = load_progress(progress_path)
    return {filename for filename, info in progress.items() if info["status"] == "completed"}


def mark_file_processing(
    progress_path: Path, filename: str, status: str, error_msg: str | None = None
) -> None:
    """Mark a file's processing status."""
    progress = load_progress(progress_path)
    progress[filename] = {"status": status, "error_message": error_msg}
    save_progress(progress_path, progress)


def insert_relevant_papers(
    db_path: Path,
    papers: list[Paper],
    probabilities: list[float],
    source_file: str,
) -> int:
    """
    Insert relevant papers into database.

    Returns number of papers inserted.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    rows = [
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
        )
        for paper in papers
    ]

    # Batch inserts to avoid SQLite's 999 variable limit
    batch_size = 100
    total_inserted = 0

    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        cursor.executemany(
            """
            INSERT INTO papers
            (doi, pmid, title, abstract, authors, journal, source, source_date,
             source_metadata, source_type, source_details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doi) DO NOTHING
            """,
            batch,
        )
        total_inserted += cursor.rowcount

    conn.commit()
    conn.close()

    return total_inserted


def process_file(
    xml_path: Path,
    ckpt: LoadedCheckpoint,
    device: torch.device,
    batch_size: int,
    db_path: Path,
    min_year: int,
) -> tuple[int, int]:
    """
    Process a single XML file.

    Returns (papers_processed, papers_relevant).
    """
    # Parse XML file
    with gzip.open(xml_path, "rb") as f:
        xml_content = f.read()

    papers, _stats = extract_papers_from_xml(
        xml_content,
        source_type="baseline_screening",
        source_details=xml_path.name,
        require_abstract=False,  # Process all papers
        min_year=min_year,
    )

    if not papers:
        return 0, 0

    # Convert to LabeledPaper format (with dummy labels for inference)
    labeled_papers = [
        LabeledPaper(
            doi=paper.doi,
            title=paper.title,
            abstract=paper.abstract,
            is_relevant=0,  # Dummy label for inference
        )
        for paper in papers
    ]

    # Create dataset and dataloader
    dataset = PaperDataset(labeled_papers, ckpt.tokenizer, max_length=1024)
    data_collator = DataCollatorWithPadding(tokenizer=ckpt.tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=1,
        pin_memory=True,
    )

    # Run inference
    all_probabilities = []

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = ckpt.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probabilities.extend(probs)

    # Filter to only relevant papers (those passing threshold)
    relevant_indices = [i for i, prob in enumerate(all_probabilities) if prob >= ckpt.threshold]
    relevant_papers = [papers[i] for i in relevant_indices]
    relevant_probs = [all_probabilities[i] for i in relevant_indices]

    # Insert only relevant papers into database
    if relevant_papers:
        insert_relevant_papers(db_path, relevant_papers, relevant_probs, xml_path.name)

    return len(papers), len(relevant_papers)


@app.callback(invoke_without_command=True)
def main(
    checkpoint: Path = typer.Option(
        ..., "--checkpoint", help="Path to trained model checkpoint (.pt file)"
    ),
    baseline_dir: Path = typer.Option(
        ..., "--baseline-dir", help="Directory containing baseline XML.gz files"
    ),
    output_db: Path = typer.Option(..., "--output-db", help="Output SQLite database path"),
    progress_file: Path = typer.Option(
        Path("data/screening_progress.json"),
        "--progress-file",
        help="JSON file to track processing progress",
    ),
    batch_size: int = typer.Option(1024, "--batch-size", help="Inference batch size"),
    device_name: str = typer.Option("cuda", "--device", help="Device to use"),
    compile_model: bool = typer.Option(
        True, "--compile/--no-compile", help="Use torch.compile for optimization"
    ),
    min_year: int = typer.Option(2000, "--min-year", help="Minimum publication year (source date)"),
) -> None:
    """Screen PubMed baseline snapshot through relevance classifier."""
    # Disable tokenizer parallelism to avoid fork warnings with DataLoader workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device(device_name)

    # Initialize database
    schema_path = Path(__file__).parent.parent.parent / "schema.sql"
    init_database(output_db, schema_path)

    # Load checkpoint
    console.print("\n[bold]Loading checkpoint...[/bold]")
    ckpt = load_checkpoint(checkpoint, device, compile_model=compile_model)
    console.print(f"[green]Model loaded. Threshold: {ckpt.threshold:.4f}[/green]\n")

    # Warmup for torch.compile if enabled
    if compile_model:
        console.print("[bold]Warming up compiled model...[/bold]")
        dummy_papers = [
            LabeledPaper(
                doi="10.0000/test", title="Test paper", abstract="Test abstract", is_relevant=0
            )
        ]
        dummy_dataset = PaperDataset(dummy_papers, ckpt.tokenizer, max_length=1024)
        dummy_collator = DataCollatorWithPadding(tokenizer=ckpt.tokenizer)
        dummy_loader = DataLoader(dummy_dataset, batch_size=1, collate_fn=dummy_collator)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            batch = next(iter(dummy_loader))
            with torch.no_grad():
                _ = ckpt.model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        console.print("[green]Warmup complete[/green]\n")

    # Get list of files to process
    xml_files = sorted(baseline_dir.glob("*.xml.gz"))
    if not xml_files:
        console.print(f"[red]No XML.gz files found in {baseline_dir}[/red]")
        raise typer.Exit(1)

    # Get already processed files
    processed = get_processed_files(progress_file)
    remaining_files = [f for f in xml_files if f.name not in processed]

    console.print(f"[cyan]Total baseline files: {len(xml_files)}[/cyan]")
    console.print(f"[cyan]Already processed: {len(processed)}[/cyan]")
    console.print(f"[cyan]Remaining to process: {len(remaining_files)}[/cyan]")
    console.print(f"[cyan]Filtering papers from year: {min_year}+[/cyan]\n")

    if not remaining_files:
        console.print("[green]All files already processed![/green]")
        return

    # Process files with progress tracking
    stats = ScreeningStats(
        total_files=len(xml_files),
        files_completed=len(processed),
        total_papers=0,
        papers_relevant=0,
        start_time=time.time(),
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        # Each task item takes about 2 minutes to complete, so increase the estimation period for stable ETAs.
        speed_estimate_period=300,
    ) as progress:
        file_task = progress.add_task("[cyan]Processing files...", total=len(remaining_files))

        for xml_file in remaining_files:
            try:
                file_start = time.time()
                papers_processed, papers_relevant = process_file(
                    xml_file, ckpt, device, batch_size, output_db, min_year
                )
                file_time = time.time() - file_start

                stats.total_papers += papers_processed
                stats.papers_relevant += papers_relevant
                stats.files_completed += 1
                stats.current_throughput = papers_processed / file_time if file_time > 0 else 0

                mark_file_processing(progress_file, xml_file.name, "completed")

                relevance_pct = (
                    (papers_relevant / papers_processed * 100) if papers_processed > 0 else 0
                )
                progress.update(
                    file_task,
                    advance=1,
                    description=f"[cyan]{xml_file.name} - {papers_processed:,} papers, {papers_relevant} relevant ({relevance_pct:.2f}%) - {stats.current_throughput:.0f} papers/s",
                )

            except Exception as e:
                logger.exception(f"Failed to process {xml_file.name}")
                mark_file_processing(progress_file, xml_file.name, "failed", str(e))
                progress.update(file_task, advance=1, description=f"[red]FAILED: {xml_file.name}")

    # Final summary
    elapsed = time.time() - stats.start_time
    avg_throughput = stats.total_papers / elapsed if elapsed > 0 else 0
    relevance_rate = (
        (stats.papers_relevant / stats.total_papers * 100) if stats.total_papers > 0 else 0
    )

    console.print("\n" + "=" * 70)
    console.print("[bold green]PubMed Baseline Screening Complete[/bold green]")
    console.print("=" * 70)
    console.print(
        f"\n[cyan]Files processed:[/cyan] {stats.files_completed:,} / {stats.total_files:,}"
    )
    console.print(f"[cyan]Total papers:[/cyan] {stats.total_papers:,}")
    console.print(
        f"[cyan]Relevant papers:[/cyan] {stats.papers_relevant:,} ({relevance_rate:.2f}%)"
    )
    console.print(f"[cyan]Total time:[/cyan] {elapsed / 3600:.2f} hours")
    console.print(f"[cyan]Average throughput:[/cyan] {avg_throughput:.0f} papers/s")
    console.print(f"\n[green]Database:[/green] {output_db}")


if __name__ == "__main__":
    app()
