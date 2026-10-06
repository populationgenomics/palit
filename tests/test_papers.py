"""Tests for palit.papers: paper IDs from author names."""

import pytest

from palit.papers import _extract_first_author_last_name


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
