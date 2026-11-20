"""Tests for palit.papers: paper IDs from author names."""

import pytest

from palit.papers import _extract_first_author_last_name, replace_paper_ids_for_display


@pytest.mark.parametrize(
    ("authors", "last_name"),
    [
        ("Smith, John A; Doe, Jane B", "Smith"),
        ("van der Berg, Anna; Doe, Jane", "VanDerBerg"),
        ("Traschütz, Andreas; Synofzik, Matthis", "Traschutz"),
        ("Klöbel, Tim", "Klobel"),
        ("Stanišić, Nina; Hämmerle, Michelle", "Stanisic"),
        ("O'Brien, Kate", "Obrien"),
        ("", "Unknown"),
    ],
)
def test_last_name_keeps_the_base_letter_of_letters_with_diacritics(
    authors: str, last_name: str
) -> None:
    assert _extract_first_author_last_name(authors) == last_name


def test_paper_ids_are_replaced_in_every_string_also_at_the_start_of_a_line() -> None:
    data = {
        "summary": "Smith2024 reports two families.\n\nSmith2024a reports one.\n- Jones2023: none",
        "paper_ids": ["Smith2024", "Jones2023"],
        "nested": [{"quote": "see Smith2024a", "count": 2, "missing": None}],
        "Smith2024": "a key is left alone",
    }
    display_ids = {"Smith2024": "PMID 1", "Smith2024a": "PMID 2", "Jones2023": "Jones2023"}
    assert replace_paper_ids_for_display(data, display_ids) == {
        "summary": "PMID 1 reports two families.\n\nPMID 2 reports one.\n- Jones2023: none",
        "paper_ids": ["PMID 1", "Jones2023"],
        "nested": [{"quote": "see PMID 2", "count": 2, "missing": None}],
        "Smith2024": "a key is left alone",
    }


def test_paper_ids_without_a_display_form_leave_the_data_unchanged() -> None:
    data = {"summary": "Jones2023 reports one family."}
    assert replace_paper_ids_for_display(data, {"Jones2023": "Jones2023"}) is data
