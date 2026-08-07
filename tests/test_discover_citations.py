"""Tests for citation discovery helpers."""

from palit.discover_citations import strip_boolean_operators


def test_strips_operator_words_regardless_of_case() -> None:
    title = "Complete absence of Cockayne syndrome group B gene product gives rise to "
    title += "UV-sensitive syndrome but not Cockayne syndrome"
    assert strip_boolean_operators(title) == (
        "Complete absence of Cockayne syndrome group B gene product gives rise to "
        "UV-sensitive syndrome but Cockayne syndrome"
    )


def test_strips_uppercase_and_mixed_case_operators() -> None:
    assert strip_boolean_operators("Alpha AND beta Or gamma NoT delta") == (
        "Alpha beta gamma delta"
    )


def test_keeps_operators_embedded_in_longer_words() -> None:
    title = "NOTCH1 and ORAI1 variants in a proband with ANDersen syndrome"
    assert strip_boolean_operators(title) == (
        "NOTCH1 ORAI1 variants in a proband with ANDersen syndrome"
    )


def test_collapses_whitespace_left_behind() -> None:
    assert strip_boolean_operators("FLNC and ADD3 variants") == "FLNC ADD3 variants"


def test_leaves_titles_without_operators_untouched() -> None:
    title = "Novel STAG3 variant causes oligoasthenoteratozoospermia"
    assert strip_boolean_operators(title) == title
