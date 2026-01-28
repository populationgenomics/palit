#!/usr/bin/env python3
"""
Shared inference components for the screening classifier.

This module provides the model architecture, dataset wrapper, and inference utilities
for the BioClinical-ModernBERT screening classifier. Used by both training/evaluation
scripts and the production baseline screening command.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class LabeledPaper:
    """Paper with relevance label for training/evaluation."""

    pmid: int
    title: str
    abstract: str
    is_relevant: int


class ScreeningClassifier(nn.Module):
    """BioClinical-ModernBERT with linear classification head for relevance screening."""

    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs, shape (batch_size, seq_len)
            attention_mask: Attention mask, shape (batch_size, seq_len)

        Returns:
            Logits, shape (batch_size,)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (first token)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits: torch.Tensor = self.classifier(cls_embedding).squeeze(-1)
        return logits


@dataclass
class LoadedCheckpoint:
    """Loaded checkpoint with model, tokenizer, and metadata."""

    model: ScreeningClassifier
    tokenizer: PreTrainedTokenizerBase
    threshold: float
    epoch: int
    val_pr_auc: float


class PaperDataset(Dataset):
    """Dataset for PubMed papers with tokenization."""

    def __init__(
        self,
        papers: list[LabeledPaper],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ):
        """
        Args:
            papers: List of LabeledPaper objects
            tokenizer: Tokenizer for encoding text
            max_length: Maximum sequence length
        """
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.papers)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        paper = self.papers[idx]

        # Concatenate title and abstract
        text = f"{paper.title} {paper.abstract or ''}"

        # Tokenize (no padding - will be done dynamically per batch)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(paper.is_relevant, dtype=torch.float32),
            "pmid": paper.pmid,
        }


def load_checkpoint(
    checkpoint_path: Path, device: torch.device, compile_model: bool = True
) -> LoadedCheckpoint:
    """
    Load trained screening classifier from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Device to load model on
        compile_model: Whether to use torch.compile for optimized inference

    Returns:
        LoadedCheckpoint with model, tokenizer, and metadata
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("thomas-sounack/BioClinical-ModernBERT-large")

    # Create and load model
    model = ScreeningClassifier("thomas-sounack/BioClinical-ModernBERT-large").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Compile for optimized inference if requested
    if compile_model:
        model = cast(ScreeningClassifier, torch.compile(model))

    return LoadedCheckpoint(
        model=model,
        tokenizer=tokenizer,
        threshold=checkpoint.get("threshold", 0.5),
        epoch=checkpoint["epoch"],
        val_pr_auc=checkpoint.get("val_pr_auc", 0.0),
    )


def load_model(
    checkpoint_path: Path, device: torch.device, compile_model: bool = True
) -> tuple[ScreeningClassifier, PreTrainedTokenizerBase, float]:
    """
    Load trained screening classifier model with optimal threshold.

    Args:
        checkpoint_path: Path to model directory (must contain model files and optimal_threshold.json)
        device: Device to load model on
        compile_model: Whether to use torch.compile for optimized inference

    Returns:
        (model, tokenizer, optimal_threshold)
    """
    # Load model
    model = ScreeningClassifier("thomas-sounack/BioClinical-ModernBERT-large")
    model.encoder = AutoModel.from_pretrained(checkpoint_path)
    model.classifier.load_state_dict(torch.load(checkpoint_path / "classifier_head.pt"))
    model = model.to(device)
    model.eval()

    # Compile for optimized inference if requested
    if compile_model:
        model = cast(ScreeningClassifier, torch.compile(model))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Load optimal threshold
    with open(checkpoint_path / "optimal_threshold.json") as f:
        threshold_info = json.load(f)
        optimal_threshold = threshold_info["threshold"]

    return model, tokenizer, optimal_threshold


def predict_batch(
    model: ScreeningClassifier,
    batch: dict[str, Any],
    device: torch.device,
    threshold: float,
) -> tuple[list[int], list[float]]:
    """
    Run inference on a batch of papers.

    Args:
        model: Screening classifier model
        batch: Batch dictionary with input_ids, attention_mask, pmid
        device: Device to run inference on
        threshold: Classification threshold

    Returns:
        (pmids, probabilities) for papers in batch
    """
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    pmids = batch["pmid"]

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().tolist()

    return pmids, probs
