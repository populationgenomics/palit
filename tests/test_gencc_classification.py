"""Tests for the GenCC classification -> PanelApp confidence display mapping.

The mapping only drives report rendering; GenCC classifications are never fed to
the model. Unknown terms must raise so a GenCC vocabulary change is loud.
"""

import pytest

from palit.panelapp_integration import (
    GENCC_CLASSIFICATION_TO_CONFIDENCE,
    gencc_classification_to_confidence,
    panelapp_confidence_to_color,
)


@pytest.mark.parametrize(
    ("classification", "confidence"),
    [
        ("Definitive", 3),
        ("Strong", 3),
        ("Moderate", 2),
        ("Limited", 1),
        ("Disputed Evidence", 1),
        ("Refuted Evidence", 0),
        ("No Known Disease Relationship", 0),
        ("Animal Model Only", 0),
    ],
)
def test_known_classifications(classification: str, confidence: int) -> None:
    assert gencc_classification_to_confidence(classification) == confidence


def test_mapping_covers_exactly_the_parametrized_terms() -> None:
    # Guards against a term being added to the mapping without a test case.
    assert set(GENCC_CLASSIFICATION_TO_CONFIDENCE) == {
        "Definitive",
        "Strong",
        "Moderate",
        "Limited",
        "Disputed Evidence",
        "Refuted Evidence",
        "No Known Disease Relationship",
        "Animal Model Only",
    }


@pytest.mark.parametrize("unknown", ["definitive", "Supportive", "", "Unknown"])
def test_unknown_classification_raises(unknown: str) -> None:
    with pytest.raises(KeyError):
        gencc_classification_to_confidence(unknown)


@pytest.mark.parametrize("confidence", [0, 1, 2, 3])
def test_every_mapped_confidence_renders_a_color(confidence: int) -> None:
    # Round-trip guard: every value the mapping can produce is a value the report's
    # colour helper knows how to render.
    assert panelapp_confidence_to_color(confidence)


def test_all_classifications_render_a_color() -> None:
    for classification in GENCC_CLASSIFICATION_TO_CONFIDENCE:
        color = panelapp_confidence_to_color(gencc_classification_to_confidence(classification))
        assert color
