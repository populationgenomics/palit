#!/usr/bin/env python3
"""Evaluate a checkpoint on test set."""

import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import TypedDict

import torch
import typer
import wandb
from rich.console import Console
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from palit.screening_classifier.inference import LabeledPaper, PaperDataset, load_checkpoint
from palit.screening_classifier.train import FocalLoss, compute_pr_auc
from palit.screening_classifier.train import evaluate as evaluate_model

console = Console()
logger = logging.getLogger(__name__)
app = typer.Typer()


class TestMetrics(TypedDict):
    """Test metrics structure."""

    checkpoint_epoch: int
    val_pr_auc: float
    loss: float
    pr_auc: float
    optimal_threshold: float
    confusion_matrix: dict[str, int]
    recall: float
    precision: float
    fpr: float
    reduction: float
    eval_time_seconds: float
    papers_per_second: float


@app.command()
def evaluate(
    checkpoint_path: Path = typer.Argument(..., help="Path to checkpoint to evaluate"),
    db_path: Path = typer.Option(
        Path("data/screening_classifier/training.sqlite"),
        "--db-path",
        help="Training database path",
    ),
    batch_size: int = typer.Option(
        1024, "--batch-size", help="Batch size (larger for inference-only)"
    ),
    device_name: str = typer.Option("cuda", "--device", help="Device to use"),
    log_wandb: bool = typer.Option(False, "--log-wandb", help="Log results to W&B"),
    wandb_project: str = typer.Option(
        "relevance-screening-classifier", "--wandb-project", help="W&B project name"
    ),
    compile_model: bool = typer.Option(
        True, "--compile/--no-compile", help="Use torch.compile for faster inference"
    ),
) -> None:
    """Evaluate a checkpoint on the test set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Disable tokenizer parallelism to avoid fork warnings with DataLoader workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device(device_name)

    console.print(f"[cyan]Evaluating checkpoint: {checkpoint_path}[/cyan]")
    console.print(f"[cyan]Database: {db_path}[/cyan]\n")

    # Load test data
    console.print("[bold]Loading test data...[/bold]")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT doi, title, abstract, is_relevant FROM papers WHERE split = 'test'")
    rows = cursor.fetchall()
    conn.close()

    test_papers = [
        LabeledPaper(
            doi=row["doi"],
            title=row["title"],
            abstract=row["abstract"],
            is_relevant=row["is_relevant"],
        )
        for row in rows
    ]

    test_pos = sum(1 for p in test_papers if p.is_relevant == 1)
    console.print(f"  Test: {len(test_papers):,} papers ({test_pos:,} positive)\n")

    # Load checkpoint
    console.print("[bold]Loading checkpoint...[/bold]")
    ckpt = load_checkpoint(checkpoint_path, device, compile_model=compile_model)

    console.print(f"  Checkpoint from epoch: {ckpt.epoch}")
    console.print(f"  Validation PR-AUC: {ckpt.val_pr_auc:.4f}")
    console.print(f"  Optimal threshold: {ckpt.threshold:.4f}\n")

    # Create dataset and dataloader
    test_dataset = PaperDataset(test_papers, ckpt.tokenizer, max_length=1024)
    data_collator = DataCollatorWithPadding(tokenizer=ckpt.tokenizer)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # Warmup for torch.compile if enabled
    if compile_model:
        console.print("[bold]Compiling model for optimized inference...[/bold]")
        console.print("  Running warmup to trigger compilation...")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            dummy_batch = next(iter(test_loader))
            with torch.no_grad():
                _ = ckpt.model(
                    dummy_batch["input_ids"].to(device),
                    dummy_batch["attention_mask"].to(device),
                )
        console.print("  Compilation complete\n")

    # Compute focal loss alpha from checkpoint config if available
    # Default to 0.5 if not found
    focal_alpha = 0.5
    train_conn = sqlite3.connect(db_path)
    train_cursor = train_conn.cursor()
    train_cursor.execute("SELECT COUNT(*) FROM papers WHERE split = 'train' AND is_relevant = 0")
    train_neg = train_cursor.fetchone()[0]
    train_cursor.execute("SELECT COUNT(*) FROM papers WHERE split = 'train'")
    train_total = train_cursor.fetchone()[0]
    train_conn.close()
    focal_alpha = train_neg / train_total if train_total > 0 else 0.5

    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)

    # Evaluate with BF16 (consistent with training)
    console.print("[bold]Evaluating on test set (BF16)...[/bold]")
    start_time = time.time()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        test_loss, test_labels, test_probs = evaluate_model(
            ckpt.model, test_loader, criterion, device
        )
    eval_time = time.time() - start_time
    papers_per_sec = len(test_papers) / eval_time

    console.print(f"  Evaluation time: {eval_time:.2f}s ({papers_per_sec:.0f} papers/s)\n")

    test_pr_auc = compute_pr_auc(test_labels, test_probs)

    # Compute metrics at optimal threshold
    test_preds = [1 if p >= ckpt.threshold else 0 for p in test_probs]
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()

    test_metrics: TestMetrics = {
        "checkpoint_epoch": ckpt.epoch,
        "val_pr_auc": ckpt.val_pr_auc,
        "loss": test_loss,
        "pr_auc": test_pr_auc,
        "optimal_threshold": ckpt.threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "recall": float(tp / (tp + fn)),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "fpr": float(fp / (fp + tn)),
        "reduction": float(tn / (tn + fp)),
        "eval_time_seconds": eval_time,
        "papers_per_second": papers_per_sec,
    }

    # Print results
    console.print("\n[bold green]Test Results:[/bold green]")
    console.print(f"[green]Checkpoint: Epoch {ckpt.epoch}[/green]")
    console.print(f"[green]Batch size: {batch_size}[/green]")
    console.print("[green]Precision: BF16[/green]")
    console.print(f"[green]torch.compile: {'enabled' if compile_model else 'disabled'}[/green]")
    console.print(f"[green]Test Loss: {test_loss:.4f}[/green]")
    console.print(f"[green]Test PR-AUC: {test_pr_auc:.4f}[/green]")
    console.print(f"[green]Threshold: {ckpt.threshold:.4f}[/green]")
    console.print(f"[green]Recall: {test_metrics['recall']:.1%}[/green]")
    console.print(f"[green]Precision: {test_metrics['precision']:.1%}[/green]")
    console.print(f"[green]FPR: {test_metrics['fpr']:.1%}[/green]")
    console.print(f"[green]Negative reduction: {test_metrics['reduction']:.1%}[/green]")
    console.print(f"[green]Throughput: {papers_per_sec:.0f} papers/s[/green]")
    console.print("\n[green]Confusion Matrix:[/green]")
    console.print(f"  TN: {tn:,}  FP: {fp:,}")
    console.print(f"  FN: {fn:,}  TP: {tp:,}")

    # Practical example
    console.print("\n[bold cyan]Practical Example (100k papers, 0.5% relevant):[/bold cyan]")
    example_total = 100_000
    example_prevalence = 0.005
    example_relevant = int(example_total * example_prevalence)  # 500 relevant papers

    # Calculate based on model's recall and precision
    example_tp = int(example_relevant * test_metrics["recall"])  # caught relevant
    # From precision = TP / (TP + FP), solve for FP
    if test_metrics["precision"] > 0:
        example_fp = int(example_tp / test_metrics["precision"] - example_tp)
    else:
        example_fp = 0

    papers_flagged = example_tp + example_fp
    papers_filtered = example_total - papers_flagged
    console.print(f"  Papers to review: {papers_flagged:,} ({papers_flagged / 1000:.1f}%)")
    console.print(f"  Papers filtered: {papers_filtered:,} ({papers_filtered / 1000:.1f}%)")

    # Save results
    results_path = checkpoint_path.parent / f"test_results_epoch_{ckpt.epoch}.json"
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    console.print(f"\n[cyan]Results saved to: {results_path}[/cyan]")

    # Log to W&B if requested
    if log_wandb:
        wandb.init(
            project=wandb_project,
            name=f"eval-epoch-{ckpt.epoch}",
            config={"checkpoint_epoch": ckpt.epoch},
        )
        wandb.log(
            {
                "test/loss": test_loss,
                "test/pr_auc": test_pr_auc,
                "test/recall": test_metrics["recall"],
                "test/precision": test_metrics["precision"],
                "test/fpr": test_metrics["fpr"],
                "test/reduction": test_metrics["reduction"],
            }
        )
        wandb.finish()


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
