#!/usr/bin/env python3
"""
Train BioClinical-ModernBERT classifier for relevance screening.

Single-file training script optimized for H100 GPU with BF16 mixed precision.
"""

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn
import typer
import wandb
from rich.console import Console
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from palit.screening_classifier.inference import LabeledPaper, PaperDataset, ScreeningClassifier

console = Console()
logger = logging.getLogger(__name__)
app = typer.Typer()


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification with class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float, gamma: float = 2.0):
        """
        Args:
            alpha: Weight for positive class (typically n_neg / n_total for imbalanced data)
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits from model (before sigmoid), shape (batch_size,)
            targets: Binary labels, shape (batch_size,)
        """
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        return torch.mean(loss)


# PaperDataset and ScreeningClassifier are now imported from .inference


def load_data(db_path: Path, split: str) -> list[LabeledPaper]:
    """Load papers from database for given split."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        "SELECT pmid, title, abstract, is_relevant FROM papers WHERE split = ?", (split,)
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        LabeledPaper(
            pmid=row["pmid"],
            title=row["title"],
            abstract=row["abstract"],
            is_relevant=row["is_relevant"],
        )
        for row in rows
    ]


def compute_pr_auc(labels: list[int], probs: list[float]) -> float:
    """Compute Precision-Recall AUC."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    return float(auc(recall, precision))


def find_optimal_threshold(
    labels: list[int], probs: list[float], recall_target: float
) -> tuple[float, dict]:
    """
    Find threshold that achieves target recall.

    Returns (threshold, metrics_at_threshold).
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs)

    # Find threshold for target recall
    valid_indices = recall >= recall_target
    if not valid_indices.any():
        logger.warning(f"Could not achieve {recall_target:.1%} recall, using lowest threshold")
        threshold = thresholds.min()
        idx = 0
    else:
        # Among thresholds achieving target recall, choose highest (best precision)
        valid_thresholds = thresholds[valid_indices[:-1]]  # Exclude last point
        threshold = valid_thresholds.max()
        idx = (thresholds == threshold).argmax()

    return float(threshold), {
        "threshold": float(threshold),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[float]]:
    """
    Evaluate model on dataset.

    Returns (loss, labels, probs).
    """
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())

    dataset = cast(PaperDataset, dataloader.dataset)
    avg_loss = total_loss / float(len(dataset))
    return avg_loss, all_labels, all_probs


def log_confusion_matrices(labels: list[int], probs: list[float], step: int) -> None:
    """Log confusion matrices at different thresholds to W&B."""
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    for threshold in thresholds:
        preds = [1 if p >= threshold else 0 for p in probs]

        # Log as W&B table
        wandb.log(
            {
                f"confusion_matrix/threshold_{threshold}": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=labels,
                    preds=preds,
                    class_names=["Negative", "Positive"],
                )
            },
            step=step,
        )


@app.command()
def train(
    db_path: Path = typer.Option(
        Path("data/screening_classifier/training.sqlite"),
        "--db-path",
        help="Training database path",
    ),
    output_dir: Path = typer.Option(
        Path("data/screening_classifier/models"), "--output-dir", help="Output directory"
    ),
    batch_size: int = typer.Option(128, "--batch-size", help="Batch size"),
    learning_rate: float = typer.Option(2e-5, "--lr", help="Learning rate"),
    max_epochs: int = typer.Option(5, "--epochs", help="Maximum epochs"),
    focal_gamma: float = typer.Option(2.0, "--focal-gamma", help="Focal loss gamma"),
    patience: int = typer.Option(2, "--patience", help="Early stopping patience"),
    recall_target: float = typer.Option(0.995, "--recall-target", help="Target recall"),
    wandb_project: str = typer.Option(
        "relevance-screening-classifier", "--wandb-project", help="W&B project name"
    ),
    wandb_run_name: str = typer.Option(None, "--wandb-run-name", help="W&B run name"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    device_name: str = typer.Option("cuda", "--device", help="Device to use (cuda, mps, or cpu)"),
    resume_from: Path | None = typer.Option(None, "--resume-from", help="Resume from checkpoint"),
) -> None:
    """Train relevance screening classifier."""
    # Setup
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.manual_seed(seed)
    device = torch.device(device_name)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    best_model_dir = output_dir / "best_model"
    best_model_dir.mkdir(exist_ok=True)

    console.print("[cyan]Training relevance screening classifier[/cyan]")
    console.print(f"[cyan]Database: {db_path}[/cyan]")
    console.print(f"[cyan]Output: {output_dir}[/cyan]\n")

    # Load data
    console.print("[bold]Loading data...[/bold]")
    train_papers = load_data(db_path, "train")
    val_papers = load_data(db_path, "val")
    test_papers = load_data(db_path, "test")

    train_pos = sum(1 for p in train_papers if p.is_relevant == 1)
    val_pos = sum(1 for p in val_papers if p.is_relevant == 1)
    test_pos = sum(1 for p in test_papers if p.is_relevant == 1)

    console.print(f"  Train: {len(train_papers):,} papers ({train_pos:,} positive)")
    console.print(f"  Val:   {len(val_papers):,} papers ({val_pos:,} positive)")
    console.print(f"  Test:  {len(test_papers):,} papers ({test_pos:,} positive)\n")

    # Compute focal loss alpha from training data
    focal_alpha = (len(train_papers) - train_pos) / len(train_papers)
    console.print(f"[bold]Focal loss: alpha={focal_alpha:.4f}, gamma={focal_gamma}[/bold]\n")

    # Initialize W&B
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "model": "thomas-sounack/BioClinical-ModernBERT-large",
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "max_epochs": max_epochs,
            "recall_target": recall_target,
            "train_size": len(train_papers),
            "val_size": len(val_papers),
            "test_size": len(test_papers),
        },
    )

    # Load tokenizer and create datasets
    console.print("[bold]Loading tokenizer and creating datasets...[/bold]")
    # Disable tokenizer parallelism to avoid fork warnings with DataLoader workers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("thomas-sounack/BioClinical-ModernBERT-large")

    # Max length of 1024 covers 99.87% of papers without truncation
    max_length = 1024
    train_dataset = PaperDataset(train_papers, tokenizer, max_length=max_length)
    val_dataset = PaperDataset(val_papers, tokenizer, max_length=max_length)
    test_dataset = PaperDataset(test_papers, tokenizer, max_length=max_length)

    # Dynamic padding: pad each batch to its longest sequence, not to global max_length
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=1,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=1,
        pin_memory=True,
    )

    # Initialize model
    console.print("[bold]Loading model...[/bold]")
    model = ScreeningClassifier("thomas-sounack/BioClinical-ModernBERT-large").to(device)

    # Enable gradient checkpointing to reduce memory usage during training
    # Trade-off: ~33% slower training for ~98% less activation memory
    model.encoder.gradient_checkpointing_enable()

    # Training setup
    criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    total_steps = len(train_loader) * max_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    console.print("[bold]Training setup complete[/bold]")
    console.print(f"  Total steps: {total_steps:,}")
    console.print(f"  Warmup steps: {warmup_steps:,}\n")

    # Resume from checkpoint if provided
    start_epoch = 0
    best_pr_auc = 0.0
    patience_counter = 0
    global_step = 0

    if resume_from:
        console.print(f"[bold]Resuming from checkpoint: {resume_from}[/bold]")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_pr_auc = checkpoint.get("val_pr_auc", 0.0)
        global_step = start_epoch * len(train_loader)
        console.print(f"  Resuming from epoch {start_epoch + 1}")
        console.print(f"  Best PR-AUC so far: {best_pr_auc:.4f}")
        console.print(f"  Global step: {global_step:,}\n")

    console.print("[bold]Starting training...[/bold]\n")

    for epoch in range(start_epoch, max_epochs):
        console.print(f"[bold cyan]Epoch {epoch + 1}/{max_epochs}[/bold cyan]")

        # Training
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass (BF16 - no gradient scaling needed)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)
            global_step += 1

            # Log every 100 steps
            if (batch_idx + 1) % 100 == 0:
                wandb.log(
                    {
                        "train/loss_step": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                    },
                    step=global_step,
                )

        avg_train_loss = train_loss / len(train_dataset)

        # Validation
        val_loss, val_labels, val_probs = evaluate(model, val_loader, criterion, device)
        val_pr_auc = compute_pr_auc(val_labels, val_probs)

        # Find threshold and compute metrics
        threshold, threshold_metrics = find_optimal_threshold(val_labels, val_probs, recall_target)

        # Log to W&B
        wandb.log(
            {
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "val/loss": val_loss,
                "val/pr_auc": val_pr_auc,
                "val/optimal_threshold": threshold,
                "val/precision_at_target_recall": threshold_metrics["precision"],
                "val/recall_at_threshold": threshold_metrics["recall"],
            },
            step=global_step,
        )

        # Log PR curve
        precision, recall, _ = precision_recall_curve(val_labels, val_probs)
        wandb.log(
            {
                "val/pr_curve": wandb.plot.line_series(
                    xs=recall.tolist(),
                    ys=[precision.tolist()],
                    keys=["Precision"],
                    title="Precision-Recall Curve",
                    xname="Recall",
                )
            },
            step=global_step,
        )

        # Log confusion matrices
        log_confusion_matrices(val_labels, val_probs, global_step)

        console.print(f"  Train Loss: {avg_train_loss:.4f}")
        console.print(f"  Val Loss:   {val_loss:.4f}")
        console.print(f"  Val PR-AUC: {val_pr_auc:.4f}")
        console.print(
            f"  Threshold:  {threshold:.4f} "
            f"(P={threshold_metrics['precision']:.3f}, R={threshold_metrics['recall']:.3f})"
        )

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_pr_auc": val_pr_auc,
                "threshold": threshold,
            },
            checkpoint_path,
        )

        # Check if best model
        if val_pr_auc > best_pr_auc:
            best_pr_auc = val_pr_auc
            patience_counter = 0

            # Save best model in HuggingFace format
            model.encoder.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            # Save classifier head separately
            torch.save(model.classifier.state_dict(), best_model_dir / "classifier_head.pt")

            # Save threshold
            with open(best_model_dir / "optimal_threshold.json", "w") as f:
                json.dump(threshold_metrics, f, indent=2)

            console.print(f"  [green]✓ New best model! PR-AUC: {val_pr_auc:.4f}[/green]")
        else:
            patience_counter += 1
            console.print(f"  Patience: {patience_counter}/{patience}")

        console.print()

        # Early stopping
        if patience_counter >= patience:
            console.print(f"[yellow]Early stopping triggered after {epoch + 1} epochs[/yellow]\n")
            break

    # Final evaluation on test set
    console.print("[bold]Evaluating on test set...[/bold]")

    # Load best model
    model.encoder = AutoModel.from_pretrained(best_model_dir).to(device)
    model.classifier.load_state_dict(torch.load(best_model_dir / "classifier_head.pt"))

    test_loss, test_labels, test_probs = evaluate(model, test_loader, criterion, device)
    test_pr_auc = compute_pr_auc(test_labels, test_probs)

    # Load optimal threshold
    with open(best_model_dir / "optimal_threshold.json") as f:
        threshold_info = json.load(f)
        optimal_threshold = threshold_info["threshold"]

    # Compute test metrics at optimal threshold
    test_preds = [1 if p >= optimal_threshold else 0 for p in test_probs]
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()

    test_metrics = {
        "loss": test_loss,
        "pr_auc": test_pr_auc,
        "optimal_threshold": optimal_threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "recall": float(tp / (tp + fn)),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "fpr": float(fp / (fp + tn)),
        "reduction": float(tn / (tn + fp)),  # Fraction of negatives correctly filtered
    }

    # Save test metrics
    with open(best_model_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Log to W&B
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

    console.print("\n[bold green]Training complete![/bold green]")
    console.print(f"[green]Best validation PR-AUC: {best_pr_auc:.4f}[/green]")
    console.print(f"[green]Test PR-AUC: {test_pr_auc:.4f}[/green]")
    console.print(f"[green]Test Recall: {test_metrics['recall']:.1%}[/green]")
    console.print(f"[green]Test Precision: {test_metrics['precision']:.1%}[/green]")
    console.print(f"[green]Negative reduction: {test_metrics['reduction']:.1%}[/green]")
    console.print(f"\n[green]Model saved to: {best_model_dir.absolute()}[/green]")

    wandb.finish()


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
