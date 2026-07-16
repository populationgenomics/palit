#!/usr/bin/env python3
"""Relevance assessment logic and utilities."""

from typing import Any


def compute_relevance_majority_vote(assessments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute majority vote from 3 relevance assessments.

    Takes the first assessment from the majority side (by 'relevant' field).
    With n=3, there's always a clear majority (2-1 or 3-0).

    Its 'associations' are enriched with those named by the other assessments,
    keyed by gene symbol so the first occurrence of a gene wins. Associations
    decide which papers reach a gene's evidence extraction, and extraction is what
    establishes whether the gene is actually supported, so linkage favours recall:
    a gene named by any assessment keeps its paper rather than being dropped
    because the majority-side assessment happened not to name it.

    Args:
        assessments: List of 3 parsed relevance assessment dicts

    Returns:
        The first assessment dict from the majority side, with associations unioned
        across all 3 assessments
    """
    if len(assessments) != 3:
        raise ValueError(f"Expected exactly 3 assessments, got {len(assessments)}")

    # Count votes for relevant=True
    relevant_votes = sum(1 for a in assessments if a["relevant"])

    # Majority is True if 2 or 3 votes for relevant
    majority_is_relevant = relevant_votes >= 2

    # First assessment matching the majority, with every assessment's associations
    for assessment in assessments:
        if assessment["relevant"] == majority_is_relevant:
            return {**assessment, "associations": _union_associations(assessments)}

    # Should never reach here with valid data
    raise ValueError("No assessment matches majority vote")


def _union_associations(assessments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gene-disease associations named by any assessment, first mention per symbol."""
    merged: dict[str, dict[str, Any]] = {}
    for assessment in assessments:
        for association in assessment.get("associations") or []:
            gene_symbol = (association.get("gene_symbol") or "").strip()
            if gene_symbol:
                merged.setdefault(gene_symbol.upper(), association)
    return list(merged.values())
