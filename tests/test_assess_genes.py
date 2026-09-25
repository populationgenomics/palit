"""Tests for aggregate citation handling."""

from palit.assess_genes import prune_invalid_citations


def test_prune_invalid_citations_keeps_extraction_quotes() -> None:
    assessment = {
        "disease_entities": [
            {
                "citations": [{"doi": "10.1/a", "quote": "exact quote", "commentary": "c"}],
                "evidence_assessments": [
                    {
                        "name": "criterion_A",
                        "citations": [{"doi": "10.1/a", "quote": "paraphrase", "commentary": "c"}],
                    }
                ],
            }
        ],
        "quality_concerns": [
            {
                "concern": "x",
                "citations": [{"doi": "10.1/b", "quote": "exact quote", "commentary": "c"}],
            }
        ],
    }
    total, removed = prune_invalid_citations(
        assessment, {"10.1/a": {"exact quote"}, "10.1/b": set()}
    )
    assert total == 3
    assert removed == [("10.1/a", "paraphrase"), ("10.1/b", "exact quote")]
    assert assessment["disease_entities"][0]["citations"][0]["quote"] == "exact quote"
    assert assessment["disease_entities"][0]["evidence_assessments"][0]["citations"] == []
    assert assessment["quality_concerns"][0]["citations"] == []
