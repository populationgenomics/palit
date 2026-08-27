#!/usr/bin/env python3
"""Helpers shared by the per-gene and per-association aggregation stages.

Both stages turn a set of per-paper evidence extractions into one LLM prompt and
then validate what comes back, so the paper gates (preprint family floor), the
citation plumbing (AuthorYear paper IDs back to DOIs, box-ID validation) and the
context-budget selection live here rather than in either command module.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from palit.papers import MIN_PREPRINT_FAMILIES, is_preprint

logger = logging.getLogger(__name__)

# SQLite busy timeout in seconds. When sharding, multiple processes write to
# the same database; the default 5 s can be too short for large batch commits.
DB_TIMEOUT_SECONDS = 60


def _max_family_count(evidence: dict[str, Any]) -> int | None:
    """Compute max family_count across all disease entities in an evidence entry.

    Returns None if all family_count values are None (not reported).
    """
    max_fc: int | None = None
    for gene_eval in evidence.get("gene_evaluations", []):
        for entity in gene_eval.get("disease_entities", []):
            fc = entity.get("family_count")
            if fc is not None:
                max_fc = max(max_fc, fc) if max_fc is not None else fc
    return max_fc


def filter_preprint_evidence(
    evidence_list: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split evidence into (kept, filtered) based on preprint family count gate.

    Preprints with max family_count < MIN_PREPRINT_FAMILIES (or all null) are
    filtered out. Published papers always pass.

    Returns:
        Tuple of (kept evidence, filtered evidence with doi+reason dicts)
    """
    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for evidence in evidence_list:
        if not is_preprint(evidence.get("journal"), evidence.get("pmid")):
            kept.append(evidence)
            continue
        max_fc = _max_family_count(evidence)
        if max_fc is not None and max_fc >= MIN_PREPRINT_FAMILIES:
            kept.append(evidence)
        elif max_fc is None:
            filtered.append(
                {"doi": evidence["doi"], "reason": "Preprint: family count not reported"}
            )
        else:
            filtered.append(
                {
                    "doi": evidence["doi"],
                    "reason": f"Preprint: {max_fc} families (min {MIN_PREPRINT_FAMILIES} required)",
                }
            )
    return kept, filtered


def fetch_valid_box_ids_by_doi(
    db_path: Path, evidence_list: list[dict[str, Any]]
) -> dict[str, set[int]]:
    """Query database to get valid box IDs for each paper in evidence_list.

    Args:
        db_path: Path to SQLite database
        evidence_list: List of evidence dicts containing DOIs

    Returns:
        Map from DOI to set of valid box IDs for that paper
    """
    dois = {evidence["doi"] for evidence in evidence_list}

    if not dois:
        return {}

    valid_box_ids_by_doi: dict[str, set[int]] = {}

    with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
        cursor = conn.cursor()

        for doi in dois:
            cursor.execute("SELECT bbox_mapping FROM papers WHERE doi = ?", (doi,))
            row = cursor.fetchone()

            if row and row[0]:
                try:
                    bbox_mapping = json.loads(row[0])
                    valid_box_ids_by_doi[doi] = {int(box_id) for box_id in bbox_mapping.keys()}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Error parsing bbox_mapping for DOI {doi}: {e}")
                    continue

    return valid_box_ids_by_doi


def _resolve_paper_id(paper_id: str, paper_id_to_doi: dict[str, str]) -> str:
    """Look up the DOI behind an AuthorYear paper ID. Raises ValueError if unknown."""
    doi = paper_id_to_doi.get(paper_id)
    if doi is None:
        raise ValueError(f"Unknown paper_id: {paper_id}")
    return doi


def rewrite_paper_ids(data: Any, paper_id_to_doi: dict[str, str]) -> None:
    """Replace every AuthorYear paper reference in an LLM response with its DOI.

    The walk is shape-independent: any dict carrying `paper_id` gains a `doi`, any
    dict carrying `paper_ids` gains `dois`, wherever they sit in the response. That
    keeps citation plumbing working when the response schema changes shape.

    Mutates `data` in place. Raises ValueError naming the first unknown paper ID
    (an LLM hallucination — the caller retries the association).
    """
    if isinstance(data, dict):
        paper_id = data.pop("paper_id", None)
        if paper_id is not None:
            data["doi"] = _resolve_paper_id(paper_id, paper_id_to_doi)

        paper_ids = data.pop("paper_ids", None)
        if paper_ids is not None:
            data["dois"] = [_resolve_paper_id(pid, paper_id_to_doi) for pid in paper_ids]

        for value in data.values():
            rewrite_paper_ids(value, paper_id_to_doi)
    elif isinstance(data, list):
        for item in data:
            rewrite_paper_ids(item, paper_id_to_doi)


def validate_citation_box_ids(data: Any, valid_box_ids_by_doi: dict[str, set[int]]) -> bool:
    """Check every (doi, box_id) citation pair against the paper's real box IDs.

    Run after `rewrite_paper_ids`. Only dicts carrying both keys are checked: a
    dict with `dois` and no `box_id` (a quality concern, which points at papers
    rather than at passages) needs no box.

    Returns False on the first violation, having logged it.
    """
    if isinstance(data, dict):
        doi = data.get("doi")
        box_id = data.get("box_id")
        if doi is not None and box_id is not None:
            valid_box_ids = valid_box_ids_by_doi.get(doi)
            if valid_box_ids is None or box_id not in valid_box_ids:
                logger.warning(f"Citation cites box_id {box_id}, which does not exist in {doi}")
                return False
        return all(
            validate_citation_box_ids(value, valid_box_ids_by_doi) for value in data.values()
        )
    if isinstance(data, list):
        return all(validate_citation_box_ids(item, valid_box_ids_by_doi) for item in data)
    return True


@dataclass
class BudgetSelection:
    """Papers that fit the prompt's character budget, and those that did not."""

    kept: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[tuple[dict[str, Any], str]] = field(default_factory=list)


def _max_independent_family_count(evidence: dict[str, Any]) -> int:
    """Strongest independent family count this paper contributes.

    -1 when the paper reports no count at all, so papers that quantify families
    always outrank papers that do not.
    """
    counts = [
        entity["independent_family_count"]
        for gene_eval in evidence["gene_evaluations"]
        for entity in gene_eval["disease_entities"]
        if entity["independent_family_count"] is not None
    ]
    return max(counts, default=-1)


def _by_priority(evidence_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order papers strongest-first: independent families, then recency, then DOI.

    Sorted least-significant key first and relying on sort stability, since the
    date key is descending while the DOI tiebreak is ascending.
    """
    ordered = sorted(evidence_list, key=lambda e: e["doi"])
    ordered.sort(key=lambda e: e["date"] or "", reverse=True)
    ordered.sort(key=_max_independent_family_count, reverse=True)
    return ordered


def select_papers_within_budget(
    evidence_list: list[dict[str, Any]], budget_chars: int
) -> BudgetSelection:
    """Pick the papers whose serialized evidence fits the prompt's character budget.

    Papers are considered strongest-first (see `_by_priority`) and each is kept if
    it still fits, so a small low-priority paper can occupy space a larger one
    could not. The strongest paper is always kept, even when it alone exceeds the
    budget — an association with no evidence at all is worse than an overlong
    prompt, which the model backend will surface as a context error.
    """
    selection = BudgetSelection()
    used_chars = 0

    for evidence in _by_priority(evidence_list):
        size = len(json.dumps(evidence, indent=2))
        if selection.kept and used_chars + size > budget_chars:
            over = used_chars + size - budget_chars
            selection.dropped.append(
                (evidence, f"Context budget exceeded: paper evidence dropped ({over} chars over)")
            )
            continue
        selection.kept.append(evidence)
        used_chars += size

    return selection
