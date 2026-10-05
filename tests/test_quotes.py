"""Tests for quote grounding against a PDF built in memory."""

from collections.abc import Callable

import anchorite
import pytest

from palit.quotes import PaperQuotes

TABLE_ROW = "c.1061C>T p.T354M 20 46 3"
FIGURE_LABEL = "KMT2D exon31:c.6752C>T"
FILLER = [
    f"Paragraph {i}: the cohort comprised unrelated probands referred for exome sequencing."
    for i in range(30)
]


def _paper(text_pdf: Callable[[list[str]], bytes]) -> PaperQuotes:
    quotes = PaperQuotes(text_pdf([*FILLER[:15], TABLE_ROW, FIGURE_LABEL, *FILLER[15:]]))
    assert quotes.has_text_layer
    return quotes


def test_short_verbatim_quotes_are_grounded(text_pdf: Callable[[list[str]], bytes]) -> None:
    check = _paper(text_pdf).check([TABLE_ROW, FIGURE_LABEL])
    assert check.text_layer
    assert check.rejected == []


def test_short_quote_not_in_the_paper_is_rejected(
    text_pdf: Callable[[list[str]], bytes],
) -> None:
    absent = "BRCA1 exon11:c.1234C>T"
    assert _paper(text_pdf).check([TABLE_ROW, absent]).rejected == [absent]


def test_short_verbatim_quotes_are_located(text_pdf: Callable[[list[str]], bytes]) -> None:
    locations = _paper(text_pdf).locate([TABLE_ROW, FIGURE_LABEL])
    assert [len(boxes) for boxes in locations.values()] == [1, 1]
    assert all(box["page"] == 1 for boxes in locations.values() for box in boxes)


def test_without_text_layer_nothing_is_rejected(text_pdf: Callable[[list[str]], bytes]) -> None:
    quotes = PaperQuotes(text_pdf([TABLE_ROW]))
    check = quotes.check([TABLE_ROW, "BRCA1 exon11:c.1234C>T"])
    assert not check.text_layer
    assert check.rejected == []


def test_undecodable_text_layer_checks_quotes_without_highlights(
    text_pdf: Callable[[list[str]], bytes], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_index(pdf_bytes: bytes) -> None:
        raise UnicodeDecodeError("utf-16-le", b"\x00\xd8", 0, 2, "illegal UTF-16 surrogate")

    monkeypatch.setattr(anchorite, "PdfIndex", broken_index)
    quotes = _paper(text_pdf)

    assert not quotes.can_locate
    absent = "c.9999A>T p.Lys3333Ter 1 1 0"
    assert quotes.check([TABLE_ROW, absent]).rejected == [absent]
    assert quotes.locate([TABLE_ROW, FIGURE_LABEL]) == {TABLE_ROW: [], FIGURE_LABEL: []}
