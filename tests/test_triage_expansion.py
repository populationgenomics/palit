from typing import Any

import pytest

from palit.triage_expansion import _TriageItem, answer_problems, pursued, stored_associations


def _association(
    status: str, existing_rating: str | None, paper_ids: tuple[str, ...] = ("Smith2026",)
) -> dict[str, Any]:
    return {
        "description": "d",
        "inheritance_mode": "Biallelic",
        "paper_ids": list(paper_ids),
        "panelapp_relation": {"status": status, "existing_rating": existing_rating, "basis": "b"},
        "evidence_brief": "e",
    }


@pytest.mark.parametrize(
    ("status", "existing_rating", "gene_rating", "expected"),
    [
        # A gene on no target panel: every association is a new disease
        ("existing", "GREEN", None, True),
        ("new_disease", None, 3, True),
        ("new_moi", None, 3, True),
        # An existing association starts from its own rating when PanelApp gives one
        ("existing", "GREEN", 1, False),
        ("existing", "AMBER", 3, True),
        ("existing", "RED", 3, True),
        # and from the gene's highest target rating when it gives none
        ("existing", None, 3, False),
        ("existing", None, 2, True),
    ],
)
def test_pursued_follows_the_report_finding_rule(
    status: str, existing_rating: str | None, gene_rating: int | None, expected: bool
) -> None:
    assert pursued(_association(status, existing_rating), gene_rating) is expected


ITEM = _TriageItem(
    hgnc_id=1,
    hgnc_symbol="GENEA",
    prompt="p",
    paper_id_to_doi={"Smith2026": "10.1/s", "Lee2026": "10.1/l"},
    gene_rating=3,
)


def test_a_valid_answer_has_no_problems() -> None:
    answer = {
        "associations": [
            _association("existing", "AMBER", ("Smith2026",)),
            _association("new_disease", None, ("Lee2026", "Smith2026")),
        ]
    }
    assert answer_problems(answer, ITEM) == []
    stored = stored_associations(answer, ITEM.paper_id_to_doi, ITEM.gene_rating)
    assert [(a["dois"], a["pursued"]) for a in stored] == [
        (["10.1/s"], True),
        (["10.1/l", "10.1/s"], True),
    ]


def test_answer_problems_name_unknown_and_uncovered_papers_and_rated_new_associations() -> None:
    answer = {"associations": [_association("new_moi", "GREEN", ("Smith2026", "Jones2020"))]}
    assert answer_problems(answer, ITEM) == [
        "unknown paper IDs ['Jones2020']",
        "papers in no association ['Lee2026']",
        "'d' is new_moi with rating GREEN",
    ]
