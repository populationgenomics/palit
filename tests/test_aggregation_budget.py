"""Tests for the shared aggregation helpers.

Covers the deterministic context-budget selection (``select_papers_within_budget``)
and the two citation-plumbing walks used on association assessments:
``rewrite_paper_ids`` and ``validate_citation_box_ids``.
"""

import json
from typing import Any

import pytest

from palit.aggregation import (
    rewrite_paper_ids,
    select_papers_within_budget,
    validate_citation_box_ids,
)


def _paper(
    doi: str,
    date: str,
    independent_family_counts: list[int | None],
    padding: int = 0,
) -> dict[str, Any]:
    """Build an evidence entry in the shape ``get_evidence_for_entity`` returns.

    ``padding`` inflates the serialized size so budget behaviour can be exercised
    with predictable numbers.
    """
    return {
        "doi": doi,
        "date": date,
        "title": "T" * padding if padding else "title",
        "gene_evaluations": [
            {
                "hgnc_id": 1,
                "disease_entities": [
                    {"independent_family_count": count} for count in independent_family_counts
                ],
            }
        ],
    }


def _serialized_size(paper: dict[str, Any]) -> int:
    return len(json.dumps(paper, indent=2))


def test_all_papers_kept_when_under_budget() -> None:
    papers = [
        _paper("10.1/a", "2024-01-01", [3]),
        _paper("10.1/b", "2023-01-01", [1]),
    ]
    selection = select_papers_within_budget(papers, budget_chars=100_000)
    assert [p["doi"] for p in selection.kept] == ["10.1/a", "10.1/b"]
    assert selection.dropped == []


def test_weakest_independent_count_is_dropped_first() -> None:
    strong = _paper("10.1/strong", "2020-01-01", [9])
    weak = _paper("10.1/weak", "2024-01-01", [1])
    # Only the first paper fits, so priority — not input order — decides who stays.
    selection = select_papers_within_budget([weak, strong], _serialized_size(strong))
    assert [p["doi"] for p in selection.kept] == ["10.1/strong"]
    assert [p["doi"] for p, _ in selection.dropped] == ["10.1/weak"]


def test_papers_without_counts_rank_below_papers_with_counts() -> None:
    counted = _paper("10.1/counted", "2019-01-01", [0])
    uncounted = _paper("10.1/uncounted", "2025-01-01", [None])
    selection = select_papers_within_budget([uncounted, counted], _serialized_size(counted))
    assert [p["doi"] for p in selection.kept] == ["10.1/counted"]


def test_newer_paper_wins_at_equal_independent_count() -> None:
    older = _paper("10.1/older", "2021-05-01", [4])
    newer = _paper("10.1/newer", "2024-05-01", [4])
    selection = select_papers_within_budget([older, newer], _serialized_size(newer))
    assert [p["doi"] for p in selection.kept] == ["10.1/newer"]
    assert [p["doi"] for p, _ in selection.dropped] == ["10.1/older"]


def test_doi_breaks_ties_on_count_and_date() -> None:
    first = _paper("10.1/aaa", "2024-05-01", [4])
    second = _paper("10.1/bbb", "2024-05-01", [4])
    selection = select_papers_within_budget([second, first], _serialized_size(first))
    assert [p["doi"] for p in selection.kept] == ["10.1/aaa"]


def test_dropped_papers_carry_an_explicit_budget_reason() -> None:
    kept = _paper("10.1/kept", "2024-01-01", [5])
    dropped = _paper("10.1/dropped", "2024-01-01", [2])
    budget = _serialized_size(kept)
    selection = select_papers_within_budget([kept, dropped], budget)
    ((_, reason),) = selection.dropped
    expected_over = _serialized_size(kept) + _serialized_size(dropped) - budget
    assert reason == f"Context budget exceeded: paper evidence dropped ({expected_over} chars over)"


def test_highest_priority_paper_is_kept_even_when_it_alone_exceeds_budget() -> None:
    papers = [_paper("10.1/big", "2024-01-01", [7], padding=500)]
    selection = select_papers_within_budget(papers, budget_chars=10)
    assert [p["doi"] for p in selection.kept] == ["10.1/big"]
    assert selection.dropped == []


def test_smaller_lower_priority_paper_fills_leftover_space() -> None:
    strong = _paper("10.1/strong", "2024-01-01", [9], padding=400)
    medium = _paper("10.1/medium", "2024-01-01", [5], padding=400)
    small = _paper("10.1/small", "2024-01-01", [1])
    budget = _serialized_size(strong) + _serialized_size(small)
    selection = select_papers_within_budget([strong, medium, small], budget)
    assert [p["doi"] for p in selection.kept] == ["10.1/strong", "10.1/small"]
    assert [p["doi"] for p, _ in selection.dropped] == ["10.1/medium"]


def test_selection_is_deterministic_regardless_of_input_order() -> None:
    papers = [
        _paper("10.1/a", "2024-01-01", [3]),
        _paper("10.1/b", "2022-01-01", [3]),
        _paper("10.1/c", "2024-01-01", [8]),
    ]
    budget = sum(_serialized_size(p) for p in papers[:2])
    forward = select_papers_within_budget(papers, budget)
    backward = select_papers_within_budget(list(reversed(papers)), budget)
    assert [p["doi"] for p in forward.kept] == [p["doi"] for p in backward.kept]


MAPPING = {"Smith2024": "10.1/smith", "Jones2023": "10.1/jones"}


def test_rewrite_paper_ids_rewrites_nested_citations() -> None:
    data: dict[str, Any] = {
        "citations": [{"paper_id": "Smith2024", "box_id": 4, "commentary": "c"}],
        "evidence_assessments": [
            {
                "name": "criterion_A",
                "citations": [{"paper_id": "Jones2023", "box_id": 7, "commentary": "c"}],
            }
        ],
    }
    rewrite_paper_ids(data, MAPPING)
    assert data["citations"][0] == {"doi": "10.1/smith", "box_id": 4, "commentary": "c"}
    assert data["evidence_assessments"][0]["citations"][0]["doi"] == "10.1/jones"
    assert "paper_id" not in data["evidence_assessments"][0]["citations"][0]


def test_rewrite_paper_ids_turns_paper_ids_list_into_dois() -> None:
    data: dict[str, Any] = {
        "quality_concerns": [
            {
                "concern": "single lab",
                "paper_ids": ["Smith2024", "Jones2023"],
                "citations": [{"paper_id": "Smith2024", "box_id": 1, "commentary": "c"}],
            }
        ]
    }
    rewrite_paper_ids(data, MAPPING)
    concern = data["quality_concerns"][0]
    assert concern["dois"] == ["10.1/smith", "10.1/jones"]
    assert "paper_ids" not in concern
    assert concern["citations"][0]["doi"] == "10.1/smith"


def test_rewrite_paper_ids_rewrites_at_any_depth() -> None:
    # The walk is shape-independent, so a paper reference nested somewhere the
    # current schema does not put one is still rewritten.
    data: dict[str, Any] = {"a": [{"b": {"c": [{"paper_id": "Smith2024"}]}}]}
    rewrite_paper_ids(data, MAPPING)
    assert data["a"][0]["b"]["c"][0] == {"doi": "10.1/smith"}


def test_rewrite_paper_ids_raises_on_unknown_id() -> None:
    data: dict[str, Any] = {"citations": [{"paper_id": "Ghost2099", "box_id": 1}]}
    with pytest.raises(ValueError, match="Ghost2099"):
        rewrite_paper_ids(data, MAPPING)


def test_rewrite_paper_ids_leaves_data_without_paper_references_untouched() -> None:
    # Rewriting is not idempotent by design: it consumes `paper_id` keys and is
    # run exactly once per response. Data that carries none is simply unchanged.
    data: dict[str, Any] = {"summary": "prose", "family_count": 3, "citations": []}
    rewrite_paper_ids(data, MAPPING)
    assert data == {"summary": "prose", "family_count": 3, "citations": []}


VALID_BOX_IDS = {"10.1/smith": {1, 4, 9}, "10.1/jones": {7}}


def test_validate_citation_box_ids_accepts_valid_pairs() -> None:
    data: dict[str, Any] = {
        "citations": [{"doi": "10.1/smith", "box_id": 4, "commentary": "c"}],
        "evidence_assessments": [
            {"citations": [{"doi": "10.1/jones", "box_id": 7, "commentary": "c"}]}
        ],
    }
    assert validate_citation_box_ids(data, VALID_BOX_IDS)


def test_validate_citation_box_ids_rejects_box_from_another_paper() -> None:
    # Box 7 exists, but in Jones, not in Smith.
    data: dict[str, Any] = {"citations": [{"doi": "10.1/smith", "box_id": 7}]}
    assert not validate_citation_box_ids(data, VALID_BOX_IDS)


def test_validate_citation_box_ids_rejects_unknown_doi() -> None:
    data: dict[str, Any] = {"citations": [{"doi": "10.1/unknown", "box_id": 1}]}
    assert not validate_citation_box_ids(data, VALID_BOX_IDS)


def test_validate_citation_box_ids_accepts_paper_level_reference_without_box() -> None:
    # Quality concerns point at whole papers, so they carry `dois` and no box_id.
    data: dict[str, Any] = {
        "quality_concerns": [{"concern": "single lab", "dois": ["10.1/smith"], "citations": []}]
    }
    assert validate_citation_box_ids(data, VALID_BOX_IDS)


def test_validate_citation_box_ids_accepts_a_response_without_citations() -> None:
    assert validate_citation_box_ids({"summary": "prose", "family_count": None}, VALID_BOX_IDS)
