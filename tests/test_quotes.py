"""Tests for locating quotes in a PDF built in memory, and for placeholder quotes."""

from collections.abc import Callable

import pytest

from palit.quotes import PaperQuotes, is_placeholder

TABLE_ROW = "c.1061C>T p.T354M 20 46 3"
FIGURE_LABEL = "KMT2D exon31:c.6752C>T"
FILLER = [
    f"Paragraph {i}: the cohort comprised unrelated probands referred for exome sequencing."
    for i in range(30)
]


def _paper(text_pdf: Callable[[list[str]], bytes]) -> PaperQuotes:
    return PaperQuotes(text_pdf([*FILLER[:15], TABLE_ROW, FIGURE_LABEL, *FILLER[15:]]))


def test_short_verbatim_quotes_are_located(text_pdf: Callable[[list[str]], bytes]) -> None:
    locations = _paper(text_pdf).locate([TABLE_ROW, FIGURE_LABEL])
    assert [len(boxes) for boxes in locations.values()] == [1, 1]
    assert all(box["page"] == 1 for boxes in locations.values() for box in boxes)


def test_a_quote_not_in_the_paper_has_no_boxes(text_pdf: Callable[[list[str]], bytes]) -> None:
    absent = "BRCA1 exon11:c.1234C>T"
    assert _paper(text_pdf).locate([absent]) == {absent: []}


@pytest.mark.parametrize(
    "quote", ["", " ", "x", "y", ":", "-", "...", "placeholder", "Pending", " TBD ", "n/a"]
)
def test_placeholders(quote: str) -> None:
    assert is_placeholder(quote)


@pytest.mark.parametrize("quote", ["p.R164X", "c.1132C>T", "Table 2", "No variants were found."])
def test_quotes(quote: str) -> None:
    assert not is_placeholder(quote)
