"""Tests for the per-gene PanelApp prefill built from a gene's associations."""

from typing import Any

from palit.gencc import MondoRef
from palit.panelapp_integration import (
    ENUM_TO_PANELAPP_MOI,
    PANELAPP_CRITERIA,
    MondoMatch,
    MondoTerm,
    PrefillAssociation,
    prefill_phenotype,
    prepare_prefill_data,
)

HGNC_ID = 6772
PANEL_ID = 137


def _assessment(
    *,
    green: bool = False,
    independent: int = 1,
    inheritance_mode: str = "Monoallelic",
    dois: list[str] | None = None,
    proposed_disease_name: str | None = None,
    summary: str = "Summary.",
) -> dict[str, Any]:
    return {
        "inheritance_mode": inheritance_mode,
        "inheritance_details": "",
        "independent_family_count": independent,
        "evidence_assessments": [
            {"name": name, "result": green or name in ("criterion_D", "criterion_E")}
            for name in PANELAPP_CRITERIA
        ],
        "dois": dois or [],
        "proposed_disease_name": proposed_disease_name,
        "summary": summary,
    }


def _term(
    mondo_id: str,
    label: str,
    match: MondoMatch,
    *,
    obsolete: bool = False,
    replaced_by: tuple[MondoRef, ...] = (),
) -> MondoTerm:
    return MondoTerm(
        mondo_id=mondo_id,
        label=label,
        match=match,
        rationale="",
        obsolete=obsolete,
        replaced_by=replaced_by,
    )


def _prefill(associations: list[PrefillAssociation], doi_to_pmid: dict[str, int | None]) -> Any:
    return prepare_prefill_data(
        hgnc_id=HGNC_ID,
        associations=associations,
        form_type="review",
        panel_id=PANEL_ID,
        doi_to_pmid=doi_to_pmid,
    )


def test_rating_is_the_top_association_rating() -> None:
    prefill = _prefill(
        [
            PrefillAssociation(_assessment(independent=1, proposed_disease_name="a"), None),
            PrefillAssociation(_assessment(independent=3, proposed_disease_name="b"), None),
        ],
        {},
    )
    assert (prefill.rating, prefill.hgnc_id, prefill.panel_id) == ("AMBER", "HGNC:6772", 137)
    assert prefill.mode_of_pathogenicity is None


def test_moi_aggregates_over_all_associations() -> None:
    prefill = _prefill(
        [
            PrefillAssociation(
                _assessment(inheritance_mode="Monoallelic", proposed_disease_name="a"), None
            ),
            PrefillAssociation(
                _assessment(inheritance_mode="Biallelic", proposed_disease_name="b"), None
            ),
        ],
        {},
    )
    assert prefill.moi == ENUM_TO_PANELAPP_MOI["Monoallelic_and_biallelic"]


def test_phenotype_per_mondo_match() -> None:
    gencc = PrefillAssociation(
        _assessment(), _term("MONDO:0044315", "craniosynostosis 7", "panelapp_gencc")
    )
    exact = PrefillAssociation(
        _assessment(proposed_disease_name="SMAD6-related PAH"),
        _term("MONDO:0015924", "pulmonary arterial hypertension", "exact"),
    )
    broader = PrefillAssociation(
        _assessment(proposed_disease_name="SMAD6-related HHT-like disorder"),
        _term("MONDO:0019180", "hereditary hemorrhagic telangiectasia", "broader"),
    )
    unmapped = PrefillAssociation(_assessment(proposed_disease_name="SMAD6-related JIA"), None)
    assert [prefill_phenotype(a) for a in (gencc, exact, broader, unmapped)] == [
        "craniosynostosis 7, MONDO:0044315",
        "pulmonary arterial hypertension, MONDO:0015924",
        "SMAD6-related HHT-like disorder, MONDO:0019180",
        "SMAD6-related JIA",
    ]


def test_obsolete_gencc_term_uses_its_only_replacement() -> None:
    replacement = MondoRef("MONDO:0800448", "leukoencephalopathy with vanishing white matter")
    one = PrefillAssociation(
        _assessment(),
        _term(
            "MONDO:0011380",
            "VWM disease",
            "panelapp_gencc",
            obsolete=True,
            replaced_by=(replacement,),
        ),
    )
    none = PrefillAssociation(
        _assessment(), _term("MONDO:0000001", "disease A", "panelapp_gencc", obsolete=True)
    )
    several = PrefillAssociation(
        _assessment(),
        _term(
            "MONDO:0000002",
            "disease B",
            "panelapp_gencc",
            obsolete=True,
            replaced_by=(MondoRef("MONDO:0000003", "B1"), MondoRef("MONDO:0000004", "B2")),
        ),
    )
    assert [prefill_phenotype(a) for a in (one, none, several)] == [
        "leukoencephalopathy with vanishing white matter, MONDO:0800448",
        "disease A, MONDO:0000001",
        "disease B, MONDO:0000002",
    ]


def test_phenotypes_and_publications_are_unions_in_association_order() -> None:
    term = _term("MONDO:0019952", "congenital myopathy", "panelapp_gencc")
    prefill = _prefill(
        [
            PrefillAssociation(_assessment(dois=["10.1/b", "10.1/preprint"]), term),
            PrefillAssociation(
                _assessment(dois=["10.1/a", "10.1/b"], proposed_disease_name="z disease"), None
            ),
            PrefillAssociation(_assessment(inheritance_mode="Biallelic", dois=["10.1/a"]), term),
        ],
        {"10.1/a": 111, "10.1/b": 222, "10.1/preprint": None},
    )
    assert prefill.phenotypes == "congenital myopathy, MONDO:0019952;z disease"
    assert prefill.publications == "222;10.1/preprint;111"


def test_comment_sections_in_association_order_with_headings() -> None:
    prefill = _prefill(
        [
            PrefillAssociation(
                _assessment(green=True, summary="PMID 1 reports 18 families."),
                _term("MONDO:0044315", "craniosynostosis 7", "panelapp_gencc"),
            ),
            PrefillAssociation(
                _assessment(
                    independent=2,
                    inheritance_mode="Monoallelic_and_biallelic",
                    proposed_disease_name="SMAD6-related HHT-like disorder",
                    summary="PMID 2 reports 2 families.",
                ),
                _term("MONDO:0019180", "hereditary hemorrhagic telangiectasia", "broader"),
            ),
            PrefillAssociation(
                _assessment(proposed_disease_name="SMAD6-related JIA", summary="One family."),
                None,
            ),
        ],
        {},
    )
    assert prefill.comments.split("\n\n") == [
        "craniosynostosis 7, MONDO:0044315 | Monoallelic | GREEN in this corpus\n"
        "PMID 1 reports 18 families.",
        "SMAD6-related HHT-like disorder, MONDO:0019180"
        " (broader MONDO term: hereditary hemorrhagic telangiectasia)"
        " | Monoallelic and biallelic | AMBER in this corpus\n"
        "PMID 2 reports 2 families.",
        "SMAD6-related JIA | Monoallelic | RED in this corpus\nOne family.",
    ]
