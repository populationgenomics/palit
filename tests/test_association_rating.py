"""Tests for per-association rating and independent-family-count validation.

Covers ``calculate_association_rating`` on the flat disaggregated assessment object
(one gene-disease association per LLM call) and the singular/plural pair
``validate_independent_family_count`` / ``validate_independent_family_counts``.
"""

from typing import Any

import pytest

from palit.panelapp_integration import (
    calculate_association_rating,
    validate_independent_family_count,
    validate_independent_family_counts,
)


def _criterion(name: str, result: bool) -> dict[str, Any]:
    """Build one evidence_assessments entry in the shape the extraction schema emits."""
    return {
        "name": name,
        "result": result,
        "rationale": f"rationale for {name}",
        "confidence": "HIGH",
        "citations": [{"doi": "10.1000/example", "page": 1}],
    }


def _assessment(
    results: dict[str, bool],
    independent_family_count: int | None,
) -> dict[str, Any]:
    """Build a flat association assessment from a criterion-name -> result map.

    Only the named criteria are emitted, so callers can exercise the
    missing-criterion behaviour of ``entity_meets_green``.
    """
    return {
        "evidence_assessments": [_criterion(name, result) for name, result in results.items()],
        "independent_family_count": independent_family_count,
    }


ALL_FALSE = {
    "criterion_A": False,
    "criterion_B": False,
    "criterion_C": False,
    "criterion_D": False,
    "criterion_E": False,
}


@pytest.mark.parametrize("green_criterion", ["criterion_A", "criterion_B", "criterion_C"])
def test_green_via_each_of_a_b_c(green_criterion: str) -> None:
    # (A OR B OR C) AND D AND E: any one of the first three suffices.
    results = ALL_FALSE | {green_criterion: True, "criterion_D": True, "criterion_E": True}
    assert calculate_association_rating(_assessment(results, 1)) == 3


def test_criterion_d_false_blocks_green() -> None:
    results = ALL_FALSE | {"criterion_A": True, "criterion_D": False, "criterion_E": True}
    assert calculate_association_rating(_assessment(results, 1)) == 1


def test_criterion_e_false_blocks_green() -> None:
    results = ALL_FALSE | {"criterion_A": True, "criterion_D": True, "criterion_E": False}
    assert calculate_association_rating(_assessment(results, 1)) == 1


def test_amber_at_two_independent_families_without_green() -> None:
    assert calculate_association_rating(_assessment(ALL_FALSE, 2)) == 2


@pytest.mark.parametrize("count", [1, 0, None])
def test_red_below_amber_threshold(count: int | None) -> None:
    # None (NR) is coerced to 0 by the `or 0`, exactly as in calculate_gene_rating.
    assert calculate_association_rating(_assessment(ALL_FALSE, count)) == 1


def test_green_when_unsatisfied_criteria_are_absent() -> None:
    # get_entity_criterion returns {} for a missing name, so B and C read as False;
    # A alone still carries the disjunction.
    results = {"criterion_A": True, "criterion_D": True, "criterion_E": True}
    assert calculate_association_rating(_assessment(results, 1)) == 3


def test_missing_criterion_d_reads_as_false() -> None:
    # D is absent rather than False; a missing conjunct blocks GREEN just like a False one.
    results = {"criterion_A": True, "criterion_E": True}
    assert calculate_association_rating(_assessment(results, 3)) == 2


def test_absent_evidence_assessments_reads_as_all_false() -> None:
    # entity_meets_green defaults the whole array to [], so nothing can satisfy the
    # conjunction and the rating falls through to the family-count rule.
    assert calculate_association_rating({"independent_family_count": 4}) == 2


def test_valid_independent_family_count() -> None:
    assert validate_independent_family_count({"family_count": 5, "independent_family_count": 3})


def test_independent_equal_to_total_is_valid() -> None:
    assert validate_independent_family_count({"family_count": 3, "independent_family_count": 3})


def test_zero_independent_family_count_is_valid() -> None:
    assert validate_independent_family_count({"family_count": 4, "independent_family_count": 0})


def test_both_counts_none_is_valid() -> None:
    # NR is recorded as null on both fields together.
    assert validate_independent_family_count(
        {"family_count": None, "independent_family_count": None}
    )


def test_both_counts_absent_is_valid() -> None:
    # Absent keys read as None via .get(), so they mirror the both-null case.
    assert validate_independent_family_count({})


def test_independent_greater_than_total_is_invalid() -> None:
    assert not validate_independent_family_count({"family_count": 2, "independent_family_count": 3})


def test_independent_set_while_total_none_is_invalid() -> None:
    assert not validate_independent_family_count(
        {"family_count": None, "independent_family_count": 2}
    )


def test_total_set_while_independent_none_is_invalid() -> None:
    assert not validate_independent_family_count(
        {"family_count": 2, "independent_family_count": None}
    )


def test_negative_independent_family_count_is_invalid() -> None:
    assert not validate_independent_family_count(
        {"family_count": 5, "independent_family_count": -1}
    )


def test_plural_requires_every_entity_to_be_valid() -> None:
    valid = {"family_count": 5, "independent_family_count": 3}
    both_none = {"family_count": None, "independent_family_count": None}
    invalid = {"family_count": 2, "independent_family_count": 3}
    assert validate_independent_family_counts([valid, both_none])
    assert not validate_independent_family_counts([valid, both_none, invalid])


def test_plural_on_empty_list_is_valid() -> None:
    assert validate_independent_family_counts([])
