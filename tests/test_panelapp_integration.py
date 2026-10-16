"""Tests for the per-gene PanelApp prefill built from a gene's associations."""

from pathlib import Path
from typing import Any

import pytest

from palit.gencc import MondoRef
from palit.panelapp_integration import (
    BOTH_BIALLELIC_MORE_SEVERE_PANELAPP_MOI,
    ENUM_TO_PANELAPP_MOI,
    IMPRINTED_STATUS_UNKNOWN_PANELAPP_MOI,
    IMPRINTING_TO_PANELAPP_MOI,
    INHERITANCE_DETAILS_VOCABULARY,
    MOI_NOT_SPECIFIED,
    PANELAPP_CRITERIA,
    X_LINKED_BIALLELIC_FEMALES_PANELAPP_MOI,
    Finding,
    FindingKind,
    MondoMatch,
    MondoTerm,
    PrefillAssociation,
    PrefillData,
    PrefillEntry,
    derive_review_moi,
    prefill_phenotype,
    prepare_prefill_data,
)

ROOT = Path(__file__).resolve().parents[1]
HGNC_ID = 6772
PANEL_ID = 137


def _assessment(
    *,
    green: bool = False,
    independent: int = 1,
    inheritance_mode: str = "Monoallelic",
    inheritance_details: str = "",
    dois: list[str] | None = None,
    proposed_disease_name: str | None = None,
    summary: str = "Summary.",
) -> dict[str, Any]:
    return {
        "inheritance_mode": inheritance_mode,
        "inheritance_details": inheritance_details,
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


def _described(
    assessment: dict[str, Any], mondo: MondoTerm | None = None, finding: Finding | None = None
) -> PrefillAssociation:
    return PrefillAssociation(assessment, mondo, finding)


def _entry(
    rating: int = 1,
    moi: str = "Unknown",
    phenotypes: tuple[str, ...] = (),
    publications: tuple[int | str, ...] = (),
) -> PrefillEntry:
    """The gene's entry on the panel; by default RED, with an MoI that names no mode."""
    return PrefillEntry(rating, moi, phenotypes, publications)


def _prefill(
    associations: list[PrefillAssociation],
    doi_to_pmid: dict[str, int | None],
    entry: PrefillEntry | None = None,
    moi_associations: list[dict[str, Any]] | None = None,
    *,
    add: bool = False,
    rated: bool = True,
) -> PrefillData:
    """A review form of *entry*, ``_entry()`` when None, or an add form when *add*."""
    review_entry = _entry() if entry is None else entry
    return prepare_prefill_data(
        hgnc_id=HGNC_ID,
        associations=associations,
        moi_associations=(
            [a.assessment for a in associations] if moi_associations is None else moi_associations
        ),
        panel_id=PANEL_ID,
        entry=None if add else review_entry,
        doi_to_pmid=doi_to_pmid,
        rated=rated,
    )


RED_AND_AMBER = [
    _described(_assessment(independent=1, proposed_disease_name="a")),
    _described(_assessment(independent=3, proposed_disease_name="b")),
]


def test_add_form_rating_is_the_top_association_rating() -> None:
    prefill = _prefill(RED_AND_AMBER, {}, add=True)
    assert (prefill.form_type, prefill.rating) == ("add", "AMBER")
    assert (prefill.hgnc_id, prefill.panel_id) == ("HGNC:6772", 137)
    assert prefill.mode_of_pathogenicity is None


@pytest.mark.parametrize(
    ("panel_rating", "rating"),
    [(1, "AMBER"), (2, "AMBER"), (3, "GREEN"), (0, "AMBER")],
)
def test_review_form_rating_is_the_higher_of_panel_and_top_association(
    panel_rating: int, rating: str
) -> None:
    """A GREEN gene with only RED and AMBER associations keeps GREEN."""
    prefill = _prefill(RED_AND_AMBER, {}, _entry(rating=panel_rating))
    assert (prefill.form_type, prefill.rating) == ("review", rating)


def test_moi_aggregates_over_all_associations() -> None:
    prefill = _prefill(
        [
            _described(_assessment(inheritance_mode="Monoallelic", proposed_disease_name="a")),
            _described(_assessment(inheritance_mode="Biallelic", proposed_disease_name="b")),
        ],
        {},
    )
    assert prefill.moi == ENUM_TO_PANELAPP_MOI["Monoallelic_and_biallelic"]


PATERNAL = "imprinted, paternal allele expressed"
MATERNAL = "imprinted, maternal allele expressed"


@pytest.mark.parametrize(
    ("details", "moi"),
    [
        (
            ["reduced penetrance; " + PATERNAL, PATERNAL],
            "MONOALLELIC, autosomal or pseudoautosomal, "
            "maternally imprinted (paternal allele expressed)",
        ),
        (
            [MATERNAL, ""],
            "MONOALLELIC, autosomal or pseudoautosomal, "
            "paternally imprinted (maternal allele expressed)",
        ),
        (
            [PATERNAL, "mosaic; " + MATERNAL],
            "MONOALLELIC, autosomal or pseudoautosomal, imprinted status unknown",
        ),
        (["reduced penetrance", ""], ENUM_TO_PANELAPP_MOI["Monoallelic"]),
    ],
)
def test_monoallelic_moi_carries_the_imprinting_direction(details: list[str], moi: str) -> None:
    associations = [
        _described(_assessment(inheritance_details=d, proposed_disease_name=f"disease {i}"))
        for i, d in enumerate(details)
    ]
    # An "Other" association does not change the Monoallelic mode or its imprinting status.
    other = _assessment(
        inheritance_mode="Other", inheritance_details=MATERNAL, proposed_disease_name="other"
    )
    associations.append(_described(other))
    assert _prefill(associations, {}).moi == moi
    assert _prefill(associations, {}, add=True).moi == moi


def test_imprinting_does_not_change_other_modes() -> None:
    prefill = _prefill(
        [
            _described(
                _assessment(
                    inheritance_mode="Monoallelic",
                    inheritance_details=PATERNAL,
                    proposed_disease_name="a",
                )
            ),
            _described(
                _assessment(
                    inheritance_mode="Biallelic",
                    inheritance_details=MATERNAL,
                    proposed_disease_name="b",
                )
            ),
        ],
        {},
    )
    assert prefill.moi == ENUM_TO_PANELAPP_MOI["Monoallelic_and_biallelic"]


def test_the_aggregation_prompt_and_schema_name_the_inheritance_details_vocabulary() -> None:
    prompt = (ROOT / "prompts" / "aggregate_assessment_prompt.j2").read_text()
    schema = (ROOT / "prompts" / "aggregate_assessment_schema.json").read_text()
    for phrase in INHERITANCE_DETAILS_VOCABULARY:
        assert f'"{phrase}"' in prompt
        assert f'\\"{phrase}\\"' in schema


MONOALLELIC = ENUM_TO_PANELAPP_MOI["Monoallelic"]
BIALLELIC = ENUM_TO_PANELAPP_MOI["Biallelic"]
BOTH = ENUM_TO_PANELAPP_MOI["Monoallelic_and_biallelic"]
X_LINKED = ENUM_TO_PANELAPP_MOI["X-linked"]
PATERNAL_EXPRESSED = IMPRINTING_TO_PANELAPP_MOI[PATERNAL]
MATERNAL_EXPRESSED = IMPRINTING_TO_PANELAPP_MOI[MATERNAL]


@pytest.mark.parametrize(
    ("entry_moi", "modes", "moi"),
    [
        (MONOALLELIC, [("Biallelic", "")], BOTH),
        (BIALLELIC, [("Monoallelic", "")], BOTH),
        (BOTH, [("Biallelic", "")], BOTH),
        # An MoI that names no mode leaves the associations' MoI.
        ("Unknown", [("Biallelic", "")], BIALLELIC),
        ("Other", [("Monoallelic", ""), ("Biallelic", "")], BOTH),
        (MOI_NOT_SPECIFIED, [("X-linked", "")], X_LINKED),
        # An entry's "Unknown" or "Other" stands when no association reports a mode.
        ("Unknown", [("NR", "")], "Unknown"),
        ("Other", [("NR", ""), ("NR", "")], "Other"),
        (MOI_NOT_SPECIFIED, [("NR", "")], ""),
        # The entry's own string stands when the associations leave its mode.
        (
            BOTH_BIALLELIC_MORE_SEVERE_PANELAPP_MOI,
            [("Monoallelic", "")],
            BOTH_BIALLELIC_MORE_SEVERE_PANELAPP_MOI,
        ),
        (BOTH_BIALLELIC_MORE_SEVERE_PANELAPP_MOI, [("X-linked", "")], "Other"),
        (
            X_LINKED_BIALLELIC_FEMALES_PANELAPP_MOI,
            [("X-linked", "")],
            X_LINKED_BIALLELIC_FEMALES_PANELAPP_MOI,
        ),
        (X_LINKED_BIALLELIC_FEMALES_PANELAPP_MOI, [], X_LINKED_BIALLELIC_FEMALES_PANELAPP_MOI),
        # X-linked mixed with another mode
        (X_LINKED, [("Monoallelic", "")], "Other"),
        (BIALLELIC, [("X-linked", "")], "Other"),
        (ENUM_TO_PANELAPP_MOI["Mitochondrial"], [("Mitochondrial", "")], "MITOCHONDRIAL"),
    ],
)
def test_review_moi_combines_the_entry_with_the_associations(
    entry_moi: str, modes: list[tuple[str, str]], moi: str
) -> None:
    associations = [
        _assessment(inheritance_mode=mode, inheritance_details=details) for mode, details in modes
    ]
    assert derive_review_moi(entry_moi, associations) == moi


@pytest.mark.parametrize(
    ("entry_moi", "details", "moi"),
    [
        # The entry's imprinting direction joins the associations' directions.
        (PATERNAL_EXPRESSED, [""], PATERNAL_EXPRESSED),
        (PATERNAL_EXPRESSED, [PATERNAL], PATERNAL_EXPRESSED),
        (PATERNAL_EXPRESSED, [MATERNAL], IMPRINTED_STATUS_UNKNOWN_PANELAPP_MOI),
        (MATERNAL_EXPRESSED, ["reduced penetrance"], MATERNAL_EXPRESSED),
        # "NOT imprinted" is PanelApp's plain monoallelic MoI: it states no direction.
        (MONOALLELIC, [PATERNAL], PATERNAL_EXPRESSED),
        (MONOALLELIC, [PATERNAL, MATERNAL], IMPRINTED_STATUS_UNKNOWN_PANELAPP_MOI),
        # "imprinted status unknown" stands until an association names a direction.
        (IMPRINTED_STATUS_UNKNOWN_PANELAPP_MOI, [""], IMPRINTED_STATUS_UNKNOWN_PANELAPP_MOI),
        (IMPRINTED_STATUS_UNKNOWN_PANELAPP_MOI, [MATERNAL], MATERNAL_EXPRESSED),
        # An entry that names no mode adds no direction.
        ("Unknown", [PATERNAL], PATERNAL_EXPRESSED),
    ],
)
def test_review_moi_combines_imprinting_directions(
    entry_moi: str, details: list[str], moi: str
) -> None:
    associations = [_assessment(inheritance_details=d) for d in details]
    assert derive_review_moi(entry_moi, associations) == moi


def test_imprinting_does_not_change_a_review_moi_of_both_modes() -> None:
    biallelic = _assessment(inheritance_mode="Biallelic", inheritance_details=MATERNAL)
    assert derive_review_moi(PATERNAL_EXPRESSED, [biallelic]) == BOTH


def test_review_moi_raises_for_an_unknown_panelapp_moi() -> None:
    with pytest.raises(KeyError):
        derive_review_moi("MONOALLELIC, somehow", [_assessment()])


def test_phenotype_per_mondo_match() -> None:
    gencc = _described(_assessment(), _term("MONDO:0044315", "craniosynostosis 7", "gencc"))
    exact = _described(
        _assessment(proposed_disease_name="SMAD6-related PAH"),
        _term("MONDO:0015924", "pulmonary arterial hypertension", "exact"),
    )
    broader = _described(
        _assessment(proposed_disease_name="SMAD6-related HHT-like disorder"),
        _term("MONDO:0019180", "hereditary hemorrhagic telangiectasia", "broader"),
    )
    unmapped = _described(_assessment(proposed_disease_name="SMAD6-related JIA"))
    assert [prefill_phenotype(a) for a in (gencc, exact, broader, unmapped)] == [
        "craniosynostosis 7, MONDO:0044315",
        "pulmonary arterial hypertension, MONDO:0015924",
        "SMAD6-related HHT-like disorder, MONDO:0019180",
        "SMAD6-related JIA",
    ]


def test_obsolete_gencc_term_uses_its_only_replacement() -> None:
    replacement = MondoRef("MONDO:0800448", "leukoencephalopathy with vanishing white matter")
    one = _described(
        _assessment(),
        _term(
            "MONDO:0011380",
            "VWM disease",
            "gencc",
            obsolete=True,
            replaced_by=(replacement,),
        ),
    )
    none = _described(_assessment(), _term("MONDO:0000001", "disease A", "gencc", obsolete=True))
    several = _described(
        _assessment(),
        _term(
            "MONDO:0000002",
            "disease B",
            "gencc",
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
    term = _term("MONDO:0019952", "congenital myopathy", "gencc")
    associations = [
        _described(_assessment(dois=["10.1/b", "10.1/preprint"]), term),
        _described(_assessment(dois=["10.1/a", "10.1/b"], proposed_disease_name="z disease")),
        _described(_assessment(inheritance_mode="Biallelic", dois=["10.1/a"]), term),
    ]
    for prefill in (
        _prefill(associations, {"10.1/a": 111, "10.1/b": 222, "10.1/preprint": None}),
        _prefill(associations, {"10.1/a": 111, "10.1/b": 222, "10.1/preprint": None}, add=True),
    ):
        assert prefill.phenotypes == "congenital myopathy, MONDO:0019952;z disease"
        assert prefill.publications == "222;10.1/preprint;111"


def test_review_phenotypes_extend_the_entry_s() -> None:
    """The entry's phenotypes come first, in their order; a phenotype the entry lists,
    up to surrounding whitespace, is not listed again."""
    entry = _entry(
        phenotypes=(
            "Craniosynostosis 7, OMIM:617439",
            " craniosynostosis 7, MONDO:0044315 ",
            "Craniosynostosis",
        )
    )
    prefill = _prefill(
        [
            _described(_assessment(), _term("MONDO:0044315", "craniosynostosis 7", "gencc")),
            _described(_assessment(proposed_disease_name="SMAD6-related JIA")),
            # fuzzy matches are listed: only the exact string is the same phenotype
            _described(_assessment(proposed_disease_name="craniosynostosis")),
        ],
        {},
        entry,
    )
    assert prefill.phenotypes.split(";") == [
        "Craniosynostosis 7, OMIM:617439",
        "craniosynostosis 7, MONDO:0044315",
        "Craniosynostosis",
        "SMAD6-related JIA",
        "craniosynostosis",
    ]


def test_review_publications_extend_the_entry_s() -> None:
    """The entry's PMIDs and DOIs come first, in its order. A paper of the associations
    that the entry lists is not listed again, whether the entry lists it by its PMID or
    by its DOI, in any letter case."""
    entry = _entry(publications=(333, "10.1/B", "10.1/C", "10.1/elsewhere"))
    prefill = _prefill(
        [
            _described(
                _assessment(
                    dois=[
                        "10.1/a",  # PMID 333, which the entry lists
                        "10.1/b",  # no PMID; the entry lists its DOI in another case
                        "10.1/c",  # PMID 444; the entry lists its DOI
                        "10.1/d",  # PMID 555, new
                        "10.1/E",  # no PMID, new
                    ],
                    proposed_disease_name="d",
                )
            ),
        ],
        {"10.1/a": 333, "10.1/b": None, "10.1/c": 444, "10.1/d": 555, "10.1/E": None},
        entry,
    )
    assert prefill.publications.split(";") == [
        "333",
        "10.1/B",
        "10.1/C",
        "10.1/elsewhere",
        "555",
        "10.1/E",
    ]


def test_add_form_fields_are_the_associations_alone() -> None:
    """An add form has no entry: its MoI, phenotypes and publications are the associations'."""
    prefill = _prefill(
        [_described(_assessment(inheritance_mode="Biallelic", proposed_disease_name="a"))],
        {},
        add=True,
    )
    assert (prefill.moi, prefill.phenotypes, prefill.publications) == (BIALLELIC, "a", "")


def _finding(kind: FindingKind, rating: int, from_rating: int | None = None) -> Finding:
    return Finding(kind, rating, 1, from_rating)


# Three associations: GREEN on a GenCC term, AMBER on a broader term, RED without one
CRANIOSYNOSTOSIS = _assessment(green=True, summary="PMID 1 reports 18 families.")
HHT_LIKE = _assessment(
    independent=2,
    inheritance_mode="Monoallelic_and_biallelic",
    proposed_disease_name="SMAD6-related HHT-like disorder",
    summary="PMID 2 reports 2 families.",
)
JIA = _assessment(
    inheritance_mode="X-linked", proposed_disease_name="SMAD6-related JIA", summary="One family."
)
CRANIOSYNOSTOSIS_TERM = _term("MONDO:0044315", "craniosynostosis 7", "gencc")
HHT_TERM = _term("MONDO:0019180", "hereditary hemorrhagic telangiectasia", "broader")


def test_review_comment_heads_each_association_by_its_finding() -> None:
    prefill = _prefill(
        [
            _described(
                CRANIOSYNOSTOSIS, CRANIOSYNOSTOSIS_TERM, _finding(FindingKind.UPGRADE, 3, 2)
            ),
            _described(HHT_LIKE, HHT_TERM, _finding(FindingKind.NEW_MOI, 2)),
            _described(JIA, None, _finding(FindingKind.NEW_DISEASE, 1)),
            _described(JIA | {"summary": "No finding."}),
        ],
        {},
    )
    assert prefill.comments.split("\n\n") == [
        "Upgrade from Amber: Craniosynostosis 7 (Monoallelic, Green)\nPMID 1 reports 18 families.",
        "New MoI: SMAD6-related HHT-like disorder (Monoallelic and biallelic, Amber)\n"
        "PMID 2 reports 2 families.",
        "New disease: SMAD6-related JIA (X-linked, Red)\nOne family.",
        "SMAD6-related JIA (X-linked, Red)\nNo finding.",
    ]


def test_comment_names_an_obsolete_term_by_its_replacement() -> None:
    replaced = _term(
        "MONDO:0011380",
        "VWM disease",
        "gencc",
        obsolete=True,
        replaced_by=(MondoRef("MONDO:0800448", "vanishing white matter"),),
    )
    prefill = _prefill([_described(_assessment(summary="S."), replaced)], {})
    assert prefill.comments == "Vanishing white matter (Monoallelic, Red)\nS."


def test_add_comment_of_one_association_is_its_summary() -> None:
    prefill = _prefill(
        [_described(CRANIOSYNOSTOSIS, CRANIOSYNOSTOSIS_TERM, _finding(FindingKind.NEW_DISEASE, 3))],
        {},
        add=True,
    )
    assert prefill.comments == "PMID 1 reports 18 families."


def test_add_comment_of_several_associations_heads_each_without_its_finding() -> None:
    """Every association of a gene new to the panel is a new disease, so the add form's
    headings don't say so."""
    prefill = _prefill(
        [
            _described(
                CRANIOSYNOSTOSIS, CRANIOSYNOSTOSIS_TERM, _finding(FindingKind.NEW_DISEASE, 3)
            ),
            _described(HHT_LIKE, HHT_TERM, _finding(FindingKind.NEW_DISEASE, 2)),
        ],
        {},
        add=True,
    )
    assert prefill.comments.split("\n\n") == [
        "Craniosynostosis 7 (Monoallelic, Green)\nPMID 1 reports 18 families.",
        "SMAD6-related HHT-like disorder (Monoallelic and biallelic, Amber)\n"
        "PMID 2 reports 2 families.",
    ]


def test_comment_names_no_mondo_id_and_not_the_report() -> None:
    """PanelApp shows the comment publicly, and the phenotype field holds the MONDO IDs."""
    associations = [
        _described(
            CRANIOSYNOSTOSIS | {"summary": "Eighteen families."},
            CRANIOSYNOSTOSIS_TERM,
            _finding(FindingKind.NEW_DISEASE, 3),
        ),
        _described(
            HHT_LIKE | {"summary": "Two families."}, HHT_TERM, _finding(FindingKind.NEW_DISEASE, 2)
        ),
    ]
    for prefill in (_prefill(associations, {}), _prefill(associations, {}, add=True)):
        comments = prefill.comments.lower()
        for text in ("mondo", "report", "papers reviewed", "broader", "hereditary"):
            assert text not in comments


def test_review_moi_covers_every_association_not_only_the_described_ones() -> None:
    curated = _assessment(inheritance_mode="Monoallelic", proposed_disease_name="curated")
    finding = _assessment(inheritance_mode="Biallelic", independent=3, proposed_disease_name="new")
    prefill = _prefill(
        [_described(finding)], {}, _entry(rating=3), moi_associations=[curated, finding]
    )
    assert prefill.phenotypes == "new"
    assert prefill.moi == "BOTH monoallelic and biallelic, autosomal or pseudoautosomal"


def test_review_moi_combines_the_entry_s_moi() -> None:
    prefill = _prefill(
        [_described(_assessment(inheritance_mode="Biallelic", proposed_disease_name="new"))],
        {},
        _entry(moi=MONOALLELIC),
    )
    assert prefill.moi == BOTH


def test_comment_headings_name_every_moi_in_words() -> None:
    """The heading capitalises the disease name; the phenotype field keeps it as it is."""
    associations = [
        _described(_assessment(inheritance_mode=mode, proposed_disease_name=f"{mode} disease"))
        for mode in ("Other", "NR", "Mitochondrial", "X-linked")
    ]
    prefill = _prefill(associations, {})
    assert [section.split("\n")[0] for section in prefill.comments.split("\n\n")] == [
        "Other disease (other MoI, Red)",
        "NR disease (MoI not reported, Red)",
        "Mitochondrial disease (Mitochondrial, Red)",
        "X-linked disease (X-linked, Red)",
    ]
    lowercase = _described(_assessment(proposed_disease_name="isolated Pierre-Robin syndrome"))
    prefill = _prefill([lowercase], {})
    assert prefill.comments.startswith("Isolated Pierre-Robin syndrome (Monoallelic, Red)\n")
    assert prefill.phenotypes == "isolated Pierre-Robin syndrome"


def test_an_upgrade_from_confidence_0_is_from_red() -> None:
    """PanelApp's review form has no Grey: an entry not rated (confidence 0) is Red."""
    prefill = _prefill(
        [
            _described(
                _assessment(independent=2, proposed_disease_name="d"),
                None,
                _finding(FindingKind.UPGRADE, 2, 0),
            )
        ],
        {},
        _entry(rating=0),
    )
    assert prefill.comments.startswith("Upgrade from Red: D (Monoallelic, Amber)\n")


def test_an_unrated_review_of_a_confidence_0_entry_is_red() -> None:
    prefill = _prefill(
        [_described(_assessment(proposed_disease_name="d"))], {}, _entry(rating=0), rated=False
    )
    assert prefill.rating == "RED"


def test_review_phenotypes_skip_a_mondo_id_the_entry_holds() -> None:
    entry = _entry(phenotypes=("Stickler syndrome, MONDO:0019354", "OMIM:108300"))
    prefill = _prefill(
        [
            _described(
                _assessment(proposed_disease_name="Stickler syndrome, LOXL3-related"),
                _term("MONDO:0019354", "Stickler syndrome", "broader"),
            ),
            _described(_assessment(), _term("MONDO:0007160", "Stickler syndrome type 1", "gencc")),
        ],
        {},
        entry,
    )
    assert prefill.phenotypes.split(";") == [
        "Stickler syndrome, MONDO:0019354",
        "OMIM:108300",
        "Stickler syndrome type 1, MONDO:0007160",
    ]


def test_a_review_of_associations_all_reviewed_has_no_comment() -> None:
    """Nothing in the comment would be new; the fields still cover the associations."""
    prefill = _prefill(
        [_described(_assessment(dois=["10.1/a"], proposed_disease_name="d"))],
        {"10.1/a": 111},
        _entry(rating=2, phenotypes=("entry disease",)),
        rated=False,
    )
    assert (prefill.rating, prefill.comments) == ("AMBER", "")
    assert (prefill.phenotypes, prefill.publications) == ("entry disease;d", "111")
