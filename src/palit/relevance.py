#!/usr/bin/env python3
"""Relevance assessment logic and utilities."""

from typing import Any


def compute_relevance_majority_vote(assessments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute majority vote from 3 relevance assessments.

    Takes the first assessment from the majority side (by 'relevant' field).
    With n=3, there's always a clear majority (2-1 or 3-0).

    Args:
        assessments: List of 3 parsed relevance assessment dicts

    Returns:
        The first assessment dict from the majority side
    """
    if len(assessments) != 3:
        raise ValueError(f"Expected exactly 3 assessments, got {len(assessments)}")

    # Count votes for relevant=True
    relevant_votes = sum(1 for a in assessments if a["relevant"])

    # Majority is True if 2 or 3 votes for relevant
    majority_is_relevant = relevant_votes >= 2

    # Return first assessment that matches the majority
    for assessment in assessments:
        if assessment["relevant"] == majority_is_relevant:
            return assessment

    # Should never reach here with valid data
    raise ValueError("No assessment matches majority vote")
