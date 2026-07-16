"""Tests for the 3-vote relevance majority vote.

Covers the ``relevant`` decision and the association union that determines which
papers reach a gene's evidence extraction. The union case is drawn from
DOI 10.1038/s41436-019-0680-1, a 17-gene panel paper whose GCN1 association was
named by one assessment only: taking the majority-side assessment's associations
verbatim dropped GCN1's only paper, leaving the gene with no evidence at all.
"""

import pytest

from palit.relevance import compute_relevance_majority_vote


def _assessment(relevant: bool, *gene_symbols: str) -> dict[str, object]:
    return {
        "relevant": relevant,
        "associations": [{"gene_symbol": s, "disease": f"{s}-opathy"} for s in gene_symbols],
    }


def _symbols(result: dict[str, object]) -> set[str]:
    associations: list[dict[str, str]] = result["associations"]  # type: ignore[assignment]
    return {a["gene_symbol"] for a in associations}


def test_majority_relevant_two_of_three() -> None:
    result = compute_relevance_majority_vote(
        [_assessment(True, "GCN1"), _assessment(False), _assessment(True, "GCN1")]
    )
    assert result["relevant"] is True


def test_majority_not_relevant_two_of_three() -> None:
    result = compute_relevance_majority_vote(
        [_assessment(False), _assessment(True, "GCN1"), _assessment(False)]
    )
    assert result["relevant"] is False


def test_associations_union_across_assessments() -> None:
    """A gene named by any assessment keeps its paper linkage."""
    result = compute_relevance_majority_vote(
        [
            _assessment(True, "RYR1"),
            _assessment(True, "RYR1", "GCN1"),
            _assessment(True, "IQSEC3"),
        ]
    )
    assert _symbols(result) == {"RYR1", "GCN1", "IQSEC3"}


def test_minority_assessment_associations_are_unioned() -> None:
    """The dissenting assessment's genes count too — only the paper's fate is voted on."""
    result = compute_relevance_majority_vote(
        [_assessment(True, "RYR1"), _assessment(True, "RYR1"), _assessment(False, "GCN1")]
    )
    assert result["relevant"] is True
    assert _symbols(result) == {"RYR1", "GCN1"}


def test_union_deduplicates_case_insensitively() -> None:
    result = compute_relevance_majority_vote(
        [_assessment(True, "Gcn1"), _assessment(True, "GCN1"), _assessment(True, "gcn1")]
    )
    assert _symbols(result) == {"Gcn1"}


def test_first_mention_of_a_symbol_wins() -> None:
    """Collisions resolve to the first occurrence; the rest are equivalent for linkage."""
    first = {"relevant": True, "associations": [{"gene_symbol": "GCN1", "disease": "first"}]}
    second = {"relevant": True, "associations": [{"gene_symbol": "GCN1", "disease": "second"}]}
    result = compute_relevance_majority_vote([first, second, _assessment(True)])
    assert result["associations"] == [{"gene_symbol": "GCN1", "disease": "first"}]


def test_majority_side_fields_are_preserved() -> None:
    majority_side = {"relevant": True, "reason": "case series", "associations": []}
    result = compute_relevance_majority_vote(
        [_assessment(False), majority_side, _assessment(True, "GCN1")]
    )
    assert result["reason"] == "case series"


def test_input_assessments_are_not_mutated() -> None:
    majority_side = _assessment(True, "RYR1")
    compute_relevance_majority_vote([majority_side, _assessment(True, "GCN1"), _assessment(False)])
    assert _symbols(majority_side) == {"RYR1"}


def test_blank_gene_symbols_are_skipped() -> None:
    result = compute_relevance_majority_vote(
        [
            {"relevant": True, "associations": [{"gene_symbol": "  ", "disease": "x"}]},
            _assessment(True, "GCN1"),
            _assessment(True),
        ]
    )
    assert _symbols(result) == {"GCN1"}


def test_missing_associations_key_is_tolerated() -> None:
    result = compute_relevance_majority_vote(
        [{"relevant": True}, _assessment(True, "GCN1"), {"relevant": True, "associations": None}]
    )
    assert _symbols(result) == {"GCN1"}


def test_requires_exactly_three_assessments() -> None:
    with pytest.raises(ValueError, match="Expected exactly 3 assessments"):
        compute_relevance_majority_vote([_assessment(True, "GCN1")])
