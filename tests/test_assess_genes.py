"""Tests for aggregate evidence loading and citation handling."""

import dataclasses
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from anthropic.types import Message

from palit.assess_genes import (
    EFFORT,
    FALLBACK_EFFORT,
    STAGE,
    PanelAppContext,
    PanelReviews,
    PaperBatchProcessor,
    _GeneBatchItem,
    assessment_problems,
    association_problems,
    build_request,
    drop_placeholder_citations,
    genes_to_assess,
    handle_results,
    log_association_warnings,
    output_configs,
    render_prompt,
    replace_paper_ids_with_dois,
    replace_variant_ids,
    store_gene_aggregation,
    target_panels_holding,
    uncopied_citations,
)
from palit.gencc import GenccIndex, GeneGencc, MondoRef
from palit.llm import (
    FALLBACK_MODEL,
    MAX_TOKENS_REJECTION,
    MODEL,
    LlmResult,
    ResultStatus,
    stage_history,
)
from palit.panelapp_check import INCIDENTALOME_SCOPE, CuratedRecord
from palit.panelapp_client import PanelGeneData
from palit.panelapp_integration import (
    INCIDENTALOME_PANEL_ID,
    MENDELIOME_PANEL_ID,
    calculate_association_rating,
    criteria_object_to_list,
)
from palit.variants import GeneVariant, GnomadMiss, VariantReport

ROOT = Path(__file__).resolve().parents[1]
AARS1 = 20


def _relevance(relevant: bool) -> str:
    return json.dumps({"relevant": relevant, "screen": None, "panelapp_check": None})


def _extraction(hgnc_id: int, paper_gene_symbol: str) -> str:
    return json.dumps(
        {
            "genome_build": "unknown",
            "gene_evaluations": [
                {
                    "hgnc_id": hgnc_id,
                    "paper_gene_symbol": paper_gene_symbol,
                    "summary": "s",
                    "variants": [],
                    "disease_entities": [{"description": "d"}],
                    "quality_concerns": [],
                }
            ],
        }
    )


def test_evidence_for_gene_lists_each_paper_once_with_extraction_symbol(tmp_path: Path) -> None:
    """A relevance mention under an old symbol neither duplicates the paper nor adds an alias."""
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.execute(
            "INSERT INTO papers (doi, title, source, source_type, evidence_extraction_json) "
            "VALUES ('10.1/a', 't', 'pubmed', 'initial', ?)",
            (_extraction(AARS1, "AARS1"),),
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (?, ?, '10.1/a', ?)",
            [(AARS1, "AARS", "relevance_assessment"), (AARS1, "AARS1", "recent_evidence")],
        )

    evidence = PaperBatchProcessor(db_path).get_evidence_for_gene(AARS1)

    assert [(e["doi"], e["paper_gene_symbol"]) for e in evidence] == [("10.1/a", "AARS1")]


def test_uncopied_citations_are_kept_and_placeholders_dropped() -> None:
    assessment: dict[str, Any] = {
        "disease_entities": [
            {
                "evidence_assessments": [
                    {
                        "name": "criterion_A",
                        "citations": [
                            {"doi": "10.1/a", "quote": "exact quote", "commentary": "c"},
                            {"doi": "10.1/a", "quote": "x", "commentary": "x"},
                        ],
                    },
                    {
                        "name": "criterion_B",
                        "citations": [{"doi": "10.1/a", "quote": "paraphrase", "commentary": "c"}],
                    },
                ],
            }
        ],
        "quality_concerns": [
            {
                "concern": "x",
                "citations": [{"doi": "10.1/b", "quote": "", "commentary": ""}],
            }
        ],
    }
    assert drop_placeholder_citations(assessment) == 2
    total, uncopied = uncopied_citations(assessment, {"10.1/a": {"exact quote"}, "10.1/b": set()})
    assert total == 2
    assert uncopied == [("10.1/a", "paraphrase")]
    criterion_a, criterion_b = assessment["disease_entities"][0]["evidence_assessments"]
    assert [c["quote"] for c in criterion_a["citations"]] == ["exact quote"]
    assert criterion_b["citations"] == [{"doi": "10.1/a", "quote": "paraphrase", "commentary": "c"}]
    assert assessment["quality_concerns"][0]["citations"] == []


# --- Aggregation into associations -------------------------------------------------

PROMPT_PATH = ROOT / "prompts/aggregate_assessment_prompt.j2"
SCHEMA = json.loads((ROOT / "prompts/aggregate_assessment_schema.json").read_text())
GENEA, GENEB = 1, 3
PAPER_IDS = {"Smith2024": "10.1/a", "Jones2023": "10.1/b"}


def _criteria() -> dict[str, Any]:
    return {
        name: {"result": True, "rationale": "r", "confidence": "HIGH", "citations": []}
        for name in ("criterion_A", "criterion_B", "criterion_C", "criterion_D", "criterion_E")
    }


def _association(**fields: Any) -> dict[str, Any]:
    """One association as the model returns it, reusing GENEA's Strong GenCC row."""
    association: dict[str, Any] = {
        "description": "disease A",
        "inheritance_mode": "Biallelic",
        "inheritance_details": "",
        "grouping_rationale": "g",
        "paper_ids": ["Smith2024"],
        "variant_ids": ["V1"],
        "existing_association_mondo_id": "MONDO:0000001",
        "proposed_disease_name": None,
        "panelapp_relation": {"status": "existing", "existing_rating": "GREEN", "basis": "b"},
        "dispute_status": "None",
        "patient_count": 4,
        "reported_family_count": 3,
        "family_count": 3,
        "independent_family_count": 3,
        "count_reduction_reasoning": "No reduction (3 families)",
        "disease_mechanism": "NR",
        "evidence_weakening_factors": [
            {"factor": "founder_or_recurrent_variant", "present": False, "details": "Not present"}
        ],
        "evidence_assessments": _criteria(),
        "segregation": "segregates in 2 families",
        "functional": "NR",
        "summary": "Smith2024 reports 4 individuals from 3 families.",
    }
    return association | fields


def _new_association(status: str, independent: int | None) -> dict[str, Any]:
    return _association(
        description="new disease",
        existing_association_mondo_id=None,
        proposed_disease_name="GENEA-related new disease",
        panelapp_relation={"status": status, "existing_rating": None, "basis": "none"},
        family_count=independent,
        independent_family_count=independent,
    )


def _answer(*associations: dict[str, Any]) -> dict[str, Any]:
    return {
        "disease_entities": list(associations),
        "unassessed_reports": [
            {
                "phenotype": "p",
                "inheritance_mode": "Monoallelic",
                "paper_ids": ["Jones2023"],
                "reason": "r",
            }
        ],
        "quality_concerns": [
            {
                "concern": "c",
                "paper_ids": ["Jones2023"],
                "citations": [{"paper_id": "Jones2023", "quote": "q2"}],
            }
        ],
    }


def _evidence(doi: str, author: str, year: str, entities: int = 1) -> dict[str, Any]:
    return {
        "doi": doi,
        "pmid": None,
        "journal": "J",
        "date": f"{year}-01-01",
        "title": "t",
        "authors": f"{author}, A",
        "paper_gene_symbol": "GENEA",
        "gene_evaluations": [
            {"hgnc_id": GENEA, "disease_entities": [{"description": "d"}] * entities}
        ],
    }


def _variant(variant_id: str, *dois: str) -> GeneVariant:
    """A variant absent from gnomAD, reported by *dois*."""
    return GeneVariant(
        variant_id=variant_id,
        gnomad=GnomadMiss.NOT_FOUND,
        reports=tuple(VariantReport(doi, "q", variant_id, None) for doi in dois),
    )


VARIANTS = {"V1": _variant("1-100-A-G", "10.1/a"), "V2": _variant("1-200-C-T", "10.1/b")}
VARIANT_KEYS = {variant_id: v.variant_id for variant_id, v in VARIANTS.items()}


def _item(gencc_index: GenccIndex, evidence: list[dict[str, Any]] | None = None) -> _GeneBatchItem:
    evidence_list = evidence or [
        _evidence("10.1/a", "Smith", "2024"),
        _evidence("10.1/b", "Jones", "2023"),
    ]
    return _GeneBatchItem(
        hgnc_id=GENEA,
        hgnc_symbol="GENEA",
        prompt="",
        paper_id_to_doi=dict(PAPER_IDS),
        variants=dict(VARIANTS),
        evidence_list=evidence_list,
        filtered_papers=None,
        context=PanelAppContext(
            gencc=gencc_index.for_gene(GENEA), panel_entries=(), all_panels=False, panel_reviews=[]
        ),
    )


def _stored_form(answer: dict[str, Any]) -> dict[str, Any]:
    criteria_object_to_list(answer["disease_entities"])
    replace_paper_ids_with_dois(answer, PAPER_IDS)
    replace_variant_ids(answer, VARIANT_KEYS)
    return answer


def test_schema_accepts_the_three_relation_statuses_and_enforces_min_length() -> None:
    validator = jsonschema.Draft202012Validator(SCHEMA)
    for status in ("existing", "new_disease", "new_moi"):
        validator.validate(_answer(_new_association(status, 1)))
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(_answer(_new_association("new", 1)))
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(_answer(_association(summary="")))


def test_paper_ids_become_dois_everywhere() -> None:
    answer = _stored_form(_answer(_association()))
    association = answer["disease_entities"][0]
    assert association["dois"] == ["10.1/a"] and "paper_ids" not in association
    assert answer["unassessed_reports"][0]["dois"] == ["10.1/b"]
    assert answer["quality_concerns"][0]["dois"] == ["10.1/b"]
    assert answer["quality_concerns"][0]["citations"] == [{"quote": "q2", "doi": "10.1/b"}]


def test_criterion_citations_become_dois() -> None:
    association = _association()
    association["evidence_assessments"]["criterion_A"]["citations"] = [
        {"paper_id": "Jones2023", "quote": "q", "commentary": "c"}
    ]
    answer = _stored_form(_answer(association))
    criterion_a = answer["disease_entities"][0]["evidence_assessments"][0]
    assert criterion_a["name"] == "criterion_A"
    assert criterion_a["citations"] == [{"quote": "q", "commentary": "c", "doi": "10.1/b"}]


def test_unknown_paper_id_is_rejected() -> None:
    answer = _answer(_association(paper_ids=["Smith2024", "Invented2020"]))
    criteria_object_to_list(answer["disease_entities"])
    with pytest.raises(ValueError, match="Invented2020"):
        replace_paper_ids_with_dois(answer, PAPER_IDS)


def test_schema_requires_variant_ids_segregation_and_functional() -> None:
    validator = jsonschema.Draft202012Validator(SCHEMA)
    validator.validate(_answer(_association(variant_ids=[], segregation="NR", functional="NR")))
    for field in ("variant_ids", "segregation", "functional"):
        association = _association()
        del association[field]
        with pytest.raises(jsonschema.ValidationError, match=f"'{field}' is a required property"):
            validator.validate(_answer(association))
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(_answer(_association(segregation="")))


def test_variant_ids_become_variant_keys() -> None:
    answer = _stored_form(_answer(_association(variant_ids=["V2", "V1", "V2"])))
    association = answer["disease_entities"][0]
    assert association["variants"] == ["1-200-C-T", "1-100-A-G", "1-200-C-T"]
    assert "variant_ids" not in association
    assert association["segregation"] == "segregates in 2 families"
    assert association["functional"] == "NR"


def test_unknown_variant_id_is_rejected(gencc_index: GenccIndex) -> None:
    answer = _answer(_association(variant_ids=["V1", "V9"]))
    criteria_object_to_list(answer["disease_entities"])
    assert assessment_problems(answer, _item(gencc_index)) == [
        "hallucinated variant ID: Unknown variant ID: V9"
    ]


def test_variants_none_of_whose_papers_the_association_uses_are_logged(
    gencc_index: GenccIndex, caplog: pytest.LogCaptureFixture
) -> None:
    item = _item(gencc_index)
    log_association_warnings(_stored_form(_answer(_association(variant_ids=["V1"]))), item)
    assert "none of its paper_ids reports" not in caplog.text

    answer = _answer(_association(variant_ids=["V1", "V2"]))
    criteria_object_to_list(answer["disease_entities"])
    assert assessment_problems(answer, item) == []
    log_association_warnings(answer, item)
    assert (
        "GENEA 'disease A' (Biallelic): lists variants that none of its paper_ids reports: "
        "V2 (1-200-C-T)"
    ) in caplog.text


def _frequency(ac: int, an: int, het: int, hom: int, faf: float | None, pop: str | None) -> str:
    return json.dumps(
        {
            "ac": ac,
            "an": an,
            "homozygote_count": hom,
            "heterozygote_count": het,
            "hemizygote_count": 0,
            "faf95_popmax": faf,
            "faf95_popmax_population": pop,
        }
    )


def _normalized(hgvs_c: str, hgvs_p: str, original_text: str, candidates: int = 1) -> str:
    return json.dumps(
        {
            "hgvs_c": hgvs_c,
            "hgvs_p": hgvs_p,
            "original_text": original_text,
            "total_normalizations": candidates,
            "selected_for_max_ac": None,
        }
    )


def test_prompt_lists_the_genes_variants_with_their_gnomad_figures(
    tmp_path: Path, curated_record: CuratedRecord
) -> None:
    """One row per distinct variant of the gene's papers, by first reporting paper ID."""
    db_path = tmp_path / "run.sqlite"
    failed = json.dumps(
        {
            "original_text": "c.2990C>T",
            "error_code": "NORMALIZATION_ESEQUENCEMISMATCH",
            "error_message": "m",
            "upstream": None,
        }
    )
    not_found = json.dumps({"variant_not_found": True})
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, authors, source_date, source, source_type, "
            "evidence_extraction_json) VALUES (?, 't', ?, ?, 'pubmed', 'initial', ?)",
            [
                ("10.1/a", "Smith, A", "2024-01-01", _extraction(GENEA, "GENEA")),
                ("10.1/b", "Jones, B", "2023-01-01", _extraction(GENEA, "GENEA")),
            ],
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (?, 'GENEA', ?, 'recent_evidence')",
            [(GENEA, "10.1/a"), (GENEA, "10.1/b")],
        )
        conn.executemany(
            "INSERT INTO variant_frequencies (variant_id, hgnc_id, paper_doi, quote, "
            "normalization, gnomad) VALUES (?, ?, ?, 'q', ?, ?)",
            [
                (
                    "1-200-C-T",
                    GENEA,
                    "10.1/a",
                    _normalized("NM_1.1:c.30C>T", "NP_1.1:p.Arg10Ter", "p.R10*", candidates=2),
                    _frequency(2, 1613934, 2, 0, None, None),
                ),
                (
                    "1-100-G-A",
                    GENEA,
                    "10.1/a",
                    _normalized("NM_1.1:c.20G>A", "NP_1.1:p.Arg7Gln", "p.R7Q"),
                    _frequency(200, 1600000, 198, 1, 2.3e-05, "nfe"),
                ),
                (
                    "1-100-G-A",
                    GENEA,
                    "10.1/b",
                    _normalized("NM_2.1:c.50G>A", "NP_2.1:p.Arg17Gln", "c.50G>A"),
                    _frequency(200, 1600000, 198, 1, 2.3e-05, "nfe"),
                ),
                (
                    "1-50-T-C",
                    GENEA,
                    "10.1/b",
                    _normalized("NM_2.1:c.10T>C", "NP_2.1:p.Leu4Pro", "c.10T>C"),
                    not_found,
                ),
                ("c.2990C>T", GENEA, "10.1/b", failed, json.dumps({"normalization_error": True})),
                # Not this gene's, and not from one of its papers:
                ("1-300-A-T", GENEB, "10.1/a", failed, not_found),
                ("1-400-A-T", GENEA, "10.1/z", failed, not_found),
            ],
        )
    processor = PaperBatchProcessor(db_path)
    evidence = processor.get_evidence_for_gene(GENEA)
    assert all("variants" not in g for e in evidence for g in e["gene_evaluations"])
    variants = processor.get_variants_for_gene(GENEA, [e["doi"] for e in evidence])

    prompt = render_prompt(
        PROMPT_PATH, "GENEA", evidence, variants, _context(curated_record, GENEA), "2026-09-01", ""
    )

    assert {vid: v.variant_id for vid, v in prompt.variants.items()} == {
        "V1": "1-50-T-C",
        "V2": "1-100-G-A",
        "V3": "c.2990C>T",
        "V4": "1-200-C-T",
    }
    assert (
        "ID | HGVS | reported by | gnomAD v4.1\n"
        "V1 | NM_2.1:c.10T>C (p.Leu4Pro) | Jones2023 | not in gnomAD\n"
        "V2 | NM_2.1:c.50G>A (p.Arg17Gln); NM_1.1:c.20G>A (p.Arg7Gln) | Jones2023, Smith2024 | "
        "AF 1.25e-04 (AC/AN 200/1600000); het 198, hom 1, hemi 0; FAF95 popmax 2.30e-05 (nfe)\n"
        "V3 | c.2990C>T | Jones2023 | lookup failed\n"
        "V4 | NM_1.1:c.30C>T (p.Arg10Ter) [written p.R10*: 2 possible DNA changes, the one most "
        "frequent in gnomAD shown] | Smith2024 | AF 1.24e-06 (AC/AN 2/1613934); het 2, hom 0, "
        "hemi 0\n\nEVIDENCE EXTRACTIONS TO AGGREGATE:"
    ) in prompt.text


def test_prompt_without_variants_says_so(curated_record: CuratedRecord) -> None:
    prompt = _render(_context(curated_record, GENEA), "GENEA")
    assert (
        "VARIANTS OF GENEA (gnomAD v4.1):\nNone: the contributing papers report no variant of "
        "GENEA, so every association's `variant_ids` is []."
    ) in prompt


def test_papers_left_out_are_logged_not_rejected(
    gencc_index: GenccIndex, caplog: pytest.LogCaptureFixture
) -> None:
    answer = _stored_form(_answer(_association()))
    no_entities = _evidence("10.1/c", "Lee", "2022", entities=0)
    item = _item(
        gencc_index,
        [_evidence("10.1/a", "Smith", "2024"), _evidence("10.1/b", "Jones", "2023"), no_entities],
    )
    log_association_warnings(answer, item)
    assert "no association or unassessed report" not in caplog.text

    answer["unassessed_reports"] = []
    log_association_warnings(answer, item)
    assert "1 contributing paper(s) in no association or unassessed report: Jones2023 (10.1/b)" in (
        caplog.text
    )
    unmapped = _answer(_association())
    criteria_object_to_list(unmapped["disease_entities"])
    unmapped["unassessed_reports"] = []
    assert assessment_problems(unmapped, item) == []


def test_inheritance_details_outside_the_vocabulary_are_logged_not_rejected(
    gencc_index: GenccIndex, caplog: pytest.LogCaptureFixture
) -> None:
    item = _item(gencc_index)
    answer = _stored_form(_answer(_association(inheritance_details="reduced penetrance; mosaic")))
    log_association_warnings(answer, item)
    assert "outside the vocabulary" not in caplog.text

    details = "reduced penetrance; consanguineous"
    unmapped = _answer(_association(inheritance_details=details))
    criteria_object_to_list(unmapped["disease_entities"])
    assert assessment_problems(unmapped, item) == []
    log_association_warnings(_stored_form(_answer(_association(inheritance_details=details))), item)
    assert (
        f"GENEA 'disease A' (Biallelic): inheritance_details {details!r} has items outside "
        "the vocabulary: ['consanguineous']"
    ) in caplog.text


def test_association_fields_are_checked(gencc_index: GenccIndex) -> None:
    item = _item(gencc_index)
    answer = _stored_form(
        _answer(
            _association(),
            _association(description="both", proposed_disease_name="GENEA-related both"),
            _association(
                description="neither",
                existing_association_mondo_id=None,
                proposed_disease_name=None,
            ),
            _association(description="foreign", existing_association_mondo_id="MONDO:0009999"),
        )
    )
    problems = association_problems(answer, item)
    assert len(problems) == 3
    assert "'both': set exactly one" in problems[0]
    assert "'neither': set exactly one" in problems[1]
    assert "MONDO:0009999 is not a PanelApp Australia GenCC row" in problems[2]


@pytest.mark.parametrize("status", ["new_disease", "new_moi"])
def test_new_associations_need_a_qualifying_family(gencc_index: GenccIndex, status: str) -> None:
    item = _item(gencc_index)
    assert association_problems(_stored_form(_answer(_new_association(status, 1))), item) == []
    for independent in (0, None):
        answer = _stored_form(_answer(_new_association(status, independent)))
        (problem,) = association_problems(answer, item)
        assert f"{status} association" in problem and "unassessed_reports" in problem


UNCURATED = PanelAppContext(gencc=GeneGencc(), panel_entries=(), all_panels=False, panel_reviews=[])


def _existing_without_gencc_row() -> dict[str, Any]:
    """An existing association anchored on a panel entry or review, as STEP 3 allows."""
    return _association(
        existing_association_mondo_id=None, proposed_disease_name="GENEA-related disease"
    )


@pytest.mark.parametrize(
    "panel_reviews",
    [[], [PanelReviews(MENDELIOME_PANEL_ID, "Mendeliome", [])]],
    ids=["no target panel", "target panel without reviews"],
)
def test_existing_is_rejected_for_a_gene_without_any_curation(
    gencc_index: GenccIndex, panel_reviews: list[PanelReviews]
) -> None:
    item = dataclasses.replace(
        _item(gencc_index), context=dataclasses.replace(UNCURATED, panel_reviews=panel_reviews)
    )
    answer = _stored_form(_answer(_existing_without_gencc_row()))
    (problem,) = association_problems(answer, item)
    assert "status existing" in problem and "use new_disease" in problem

    new = _stored_form(_answer(_new_association("new_disease", 1)))
    assert association_problems(new, item) == []


def test_existing_is_kept_when_a_panel_entry_or_review_curates_the_gene(
    gencc_index: GenccIndex, curated_record: CuratedRecord
) -> None:
    reviewed = [PanelReviews(MENDELIOME_PANEL_ID, "Mendeliome", [{"rating": "AMBER"}])]
    answer = _stored_form(_answer(_existing_without_gencc_row()))
    for context in (
        dataclasses.replace(UNCURATED, panel_entries=curated_record.genes[GENEA].entries),
        dataclasses.replace(UNCURATED, panel_reviews=reviewed),
    ):
        item = dataclasses.replace(_item(gencc_index), context=context)
        assert association_problems(answer, item) == []


def test_existing_association_without_families_is_kept(gencc_index: GenccIndex) -> None:
    answer = _stored_form(_answer(_association(family_count=None, independent_family_count=None)))
    assert association_problems(answer, _item(gencc_index)) == []


def _message(answer: dict[str, Any]) -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(answer)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def test_storing_a_gene_replaces_its_associations(tmp_path: Path, gencc_index: GenccIndex) -> None:
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
    item = _item(gencc_index)
    first = _stored_form(_answer(_association(), _new_association("new_disease", 2)))
    with sqlite3.connect(db_path) as conn:
        store_gene_aggregation(conn, item, _message(first), first)
        # map-mondo's result on the new association, which a re-aggregation must not keep.
        conn.execute(
            "UPDATE associations SET mondo_id = 'MONDO:0000777', mondo_label = 'x', "
            "mondo_match = 'exact' WHERE mondo_id IS NULL"
        )

    second = _stored_form(_answer(_new_association("new_moi", 1)))
    with sqlite3.connect(db_path) as conn:
        store_gene_aggregation(conn, item, _message(second), second)
        rows = conn.execute(
            "SELECT position, mondo_id, mondo_label, mondo_match, assessment_json "
            "FROM associations WHERE hgnc_id = ?",
            (GENEA,),
        ).fetchall()
        (gene_rows,) = conn.execute("SELECT COUNT(*) FROM gene_aggregations").fetchone()
    assert gene_rows == 1
    assert [(r[0], r[1], r[2], r[3]) for r in rows] == [(0, None, None, None)]
    assert json.loads(rows[0][4])["panelapp_relation"]["status"] == "new_moi"


def test_storing_a_gene_removes_rows_left_by_a_deleted_aggregation(
    tmp_path: Path, gencc_index: GenccIndex
) -> None:
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
    item = _item(gencc_index)
    answer = _stored_form(_answer(_association(), _new_association("new_disease", 2)))
    with sqlite3.connect(db_path) as conn:
        store_gene_aggregation(conn, item, _message(answer), answer)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM gene_aggregations WHERE hgnc_id = ?", (GENEA,))
    with sqlite3.connect(db_path) as conn:
        store_gene_aggregation(conn, item, _message(answer), answer)
        positions = conn.execute(
            "SELECT position FROM associations WHERE hgnc_id = ? ORDER BY position", (GENEA,)
        ).fetchall()
    assert positions == [(0,), (1,)]


def test_storage_takes_mondo_from_the_reused_gencc_row(
    tmp_path: Path, gencc_index: GenccIndex
) -> None:
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
    item = _item(gencc_index)
    answer = _stored_form(_answer(_association(), _new_association("new_disease", 2)))
    with sqlite3.connect(db_path) as conn:
        store_gene_aggregation(conn, item, _message(answer), answer)
        rows = conn.execute(
            "SELECT position, mondo_id, mondo_label, mondo_match FROM associations ORDER BY position"
        ).fetchall()
        unassessed, concerns, context = conn.execute(
            "SELECT unassessed_reports_json, quality_concerns_json, panelapp_context_json "
            "FROM gene_aggregations"
        ).fetchone()
    assert rows == [
        (0, "MONDO:0000001", "disease A", "panelapp_gencc"),
        (1, None, None, None),
    ]
    assert json.loads(unassessed)[0]["dois"] == ["10.1/b"]
    assert json.loads(concerns)[0]["dois"] == ["10.1/b"]
    assert [row["mondo_id"] for row in json.loads(context)["gencc_rows"]] == [
        "MONDO:0000001",
        "MONDO:0000002",
    ]


def test_association_rating_is_computed_per_association() -> None:
    green = _stored_form(_answer(_association()))["disease_entities"][0]
    amber = _stored_form(
        _answer(
            _association(
                evidence_assessments=_criteria()
                | {"criterion_D": {**_criteria()["criterion_D"], "result": False}}
            )
        )
    )["disease_entities"][0]
    red = _stored_form(_answer(_new_association("new_disease", 1)))["disease_entities"][0]
    red["evidence_assessments"][0]["result"] = False
    red["evidence_assessments"][1]["result"] = False
    red["evidence_assessments"][2]["result"] = False
    assert [calculate_association_rating(e) for e in (green, amber, red)] == [3, 2, 1]


# --- Prompt context ----------------------------------------------------------------


def _context(
    record: CuratedRecord, hgnc_id: int, reviews: list[PanelReviews] | None = None
) -> PanelAppContext:
    gene = record.genes[hgnc_id]
    return PanelAppContext(
        gencc=record.gencc.for_gene(hgnc_id),
        panel_entries=gene.entries,
        all_panels=gene.all_panels,
        panel_reviews=reviews or [],
    )


def _render(context: PanelAppContext, symbol: str, panel_formatted: str = "") -> str:
    evidence = [_evidence("10.1/a", "Smith", "2024")]
    prompt = render_prompt(
        PROMPT_PATH, symbol, evidence, [], context, "2026-09-01", panel_formatted
    )
    assert prompt.paper_id_to_doi == {"Smith2024": "10.1/a"}
    assert prompt.variants == {}
    return prompt.text


def test_incidentalome_only_gene_shows_entries_from_all_panels(
    curated_record: CuratedRecord,
) -> None:
    prompt = _render(_context(curated_record, GENEB), "GENEB")
    assert "PANEL ENTRIES ON ALL PANELS (current, lumped):" in prompt
    assert f"Incidentalome, which {INCIDENTALOME_SCOPE}." in prompt
    assert "- Ataxia (panel 9001): rating GREEN" in prompt
    assert f"- Incidentalome (panel {INCIDENTALOME_PANEL_ID}): rating GREEN" in prompt


def test_gene_on_the_mendeliome_shows_target_panel_entries_and_gencc_rows(
    curated_record: CuratedRecord,
) -> None:
    prompt = _render(_context(curated_record, GENEA), "GENEA")
    assert "PANEL ENTRIES ON THE TARGET PANELS (current, lumped):" in prompt
    assert "PANEL ENTRIES ON ALL PANELS" not in prompt
    assert f"- Mendeliome (panel {MENDELIOME_PANEL_ID}): rating GREEN" in prompt
    assert '- MONDO:0000001 "disease A" — MoI "Autosomal recessive" (= Biallelic)' in prompt
    assert "obsolete MONDO term" not in prompt


def test_obsolete_gencc_term_is_marked(curated_record: CuratedRecord) -> None:
    context = _context(curated_record, GENEA)
    strong, limited = context.gencc.paa_associations
    obsolete = dataclasses.replace(
        strong, obsolete=True, replaced_by=(MondoRef("MONDO:0000003", "disease C"),)
    )
    context = dataclasses.replace(context, gencc=GeneGencc(paa_associations=(obsolete, limited)))
    prompt = _render(context, "GENEA")
    assert '- MONDO:0000001 (obsolete MONDO term; reuse this id as given) "disease A"' in prompt
    assert "MONDO:0000003" not in prompt
    stored = json.loads(json.dumps(context.to_json()))
    assert stored["gencc_rows"][0]["replaced_by"] == [
        {"mondo_id": "MONDO:0000003", "label": "disease C"}
    ]


def test_reviews_from_several_panels_are_labelled(curated_record: CuratedRecord) -> None:
    reviews = [
        PanelReviews(
            MENDELIOME_PANEL_ID,
            "Mendeliome",
            [{"rating": "GREEN", "comments": [{"user_name": "A", "created": "d", "comment": "m"}]}],
        ),
        PanelReviews(INCIDENTALOME_PANEL_ID, "Incidentalome", [{"rating": "AMBER"}]),
        PanelReviews(9002, "Silent", []),
    ]
    context = _context(curated_record, GENEA, reviews)
    prompt = _render(context, "GENEA")
    assert f'<previous_reviews panel="Mendeliome" panel_id="{MENDELIOME_PANEL_ID}">' in prompt
    assert f'<previous_reviews panel="Incidentalome" panel_id="{INCIDENTALOME_PANEL_ID}">' in prompt
    assert "    [d] A: m" in prompt
    assert 'panel="Silent"' not in prompt
    assert context.reviews_json() == [
        {"panel_id": MENDELIOME_PANEL_ID, "evaluations": reviews[0].evaluations},
        {"panel_id": INCIDENTALOME_PANEL_ID, "evaluations": [{"rating": "AMBER"}]},
        {"panel_id": 9002, "evaluations": []},
    ]
    no_reviews = _render(_context(curated_record, GENEA), "GENEA")
    assert "REVIEWS (one block per target panel that holds the gene):\n\nNone." in no_reviews


def test_target_panels_holding_keeps_target_panel_order() -> None:
    panel_data = PanelGeneData(
        panel_ids=[MENDELIOME_PANEL_ID, INCIDENTALOME_PANEL_ID, 203],
        gene_panel_confidence={GENEA: {203: 3, MENDELIOME_PANEL_ID: 2}},
    )
    assert target_panels_holding(GENEA, panel_data) == [MENDELIOME_PANEL_ID, 203]
    assert target_panels_holding(GENEB, panel_data) == []


def test_scoped_run_asks_for_panel_relevance_per_association(
    curated_record: CuratedRecord,
) -> None:
    context = _context(curated_record, GENEA)
    panel = '<panel id="47">\nName: Arthrogryposis\n</panel>'
    scoped = _render(context, "GENEA", panel)
    assert "PANEL RELEVANCE (panel-scoped run)" in scoped
    assert panel in scoped
    assert "End each association's `summary` with one sentence" in scoped

    unscoped = _render(context, "GENEA")
    assert "PANEL RELEVANCE" not in unscoped and "Arthrogryposis" not in unscoped
    assert 'one country."\nIMPORTANT INSTRUCTIONS:' in unscoped


def test_genes_failed_for_good_stay_out_unless_their_failures_predate_the_cutoff(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.execute(
            "INSERT INTO papers (doi, title, source, source_type, relevance_assessment_json) "
            "VALUES ('10.1/a', 't', 'pubmed', 'initial', ?)",
            (_relevance(True),),
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (?, 'G', '10.1/a', 'recent_evidence')",
            [(1,), (2,), (3,)],
        )
        conn.executemany(
            "INSERT INTO llm_requests (custom_id, stage, subject, round, model, status, "
            "stop_reason, rejection, completed_at) VALUES (?, ?, ?, 1, ?, 'succeeded', ?, ?, ?)",
            [
                ("a", STAGE, "2", MODEL, "max_tokens", None, "2026-10-01T00:00:00+00:00"),
                (
                    "b",
                    STAGE,
                    "2",
                    MODEL,
                    "end_turn",
                    "no associations",
                    "2026-10-01T01:00:00+00:00",
                ),
                ("c", STAGE, "3", MODEL, "max_tokens", None, "2026-10-01T00:00:00+00:00"),
            ],
        )
        history = stage_history(conn, STAGE)
        retried = stage_history(conn, STAGE, failures_since=datetime(2026, 10, 2, tzinfo=UTC))
    assert genes_to_assess(db_path, None, history) == [1, 3]
    assert genes_to_assess(db_path, None, retried) == [1, 2, 3]


def test_an_assessment_cut_off_at_max_tokens_records_the_rejection(
    tmp_path: Path, gencc_index: GenccIndex
) -> None:
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
    result = _result("assess_genes-1-a", _answer(_association()))
    assert result.message is not None
    cut_off = dataclasses.replace(
        result, message=result.message.model_copy(update={"stop_reason": "max_tokens"})
    )

    outcome = handle_results(
        [cut_off],
        {str(GENEA): _item(gencc_index)},
        db_path,
        jsonschema.Draft202012Validator(SCHEMA),
    )

    assert outcome.failed == 1
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT rejection FROM llm_requests").fetchall() == [
            (MAX_TOKENS_REJECTION,)
        ]


def test_genes_refused_by_model_go_to_the_fallback_and_refused_for_good_ones_stay_out(
    tmp_path: Path, gencc_index: GenccIndex
) -> None:
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.execute(
            "INSERT INTO papers (doi, title, source, source_type, relevance_assessment_json) "
            "VALUES ('10.1/a', 't', 'pubmed', 'initial', ?)",
            (_relevance(True),),
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (?, 'G', '10.1/a', 'recent_evidence')",
            [(1,), (2,), (3,), (4,)],
        )
        conn.executemany(
            "INSERT INTO llm_requests (custom_id, stage, subject, round, model, status, "
            "completed_at) VALUES (?, ?, ?, 1, ?, ?, '2026-10-01T00:00:00+00:00')",
            [
                ("a", STAGE, "2", MODEL, "refused"),
                ("b", STAGE, "3", MODEL, "refused"),
                ("c", STAGE, "3", FALLBACK_MODEL, "refused"),
                ("d", STAGE, "4", MODEL, "pending"),
            ],
        )
        history = stage_history(conn, STAGE)
    assert genes_to_assess(db_path, None, history) == [1, 2]
    assert genes_to_assess(db_path, [2, 3], history) == [2]
    assert [history.model_for(str(g)) for g in (1, 2)] == [MODEL, FALLBACK_MODEL]

    item = _item(gencc_index)
    configs = output_configs({"type": "object"})
    primary = build_request(item, configs, MODEL)
    fallback = build_request(item, configs, FALLBACK_MODEL)
    assert fallback.params["model"] == FALLBACK_MODEL
    assert {**fallback.params, "model": MODEL, "output_config": configs[MODEL]} == primary.params


def test_a_fallback_model_request_carries_the_higher_effort(gencc_index: GenccIndex) -> None:
    item = _item(gencc_index)
    configs = output_configs(SCHEMA)
    primary = build_request(item, configs, MODEL)
    fallback = build_request(item, configs, FALLBACK_MODEL)
    assert primary.params["output_config"]["effort"] == EFFORT == "medium"
    assert fallback.params["output_config"]["effort"] == FALLBACK_EFFORT == "high"
    assert fallback.params["output_config"]["format"] == primary.params["output_config"]["format"]


def test_only_genes_with_recent_evidence_from_a_relevant_paper_are_assessed(
    tmp_path: Path,
) -> None:
    """A gene whose recent papers were all assessed not relevant is not one of the run's genes.

    A not-relevant paper with an extraction still counts as evidence for a gene of the run.
    """
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type, relevance_assessment_json, "
            "evidence_extraction_json) VALUES (?, 't', 'pubmed', ?, ?, ?)",
            [
                ("10.1/relevant", "initial", _relevance(True), _extraction(1, "G1")),
                ("10.1/not-relevant", "initial", _relevance(False), _extraction(1, "G1")),
                ("10.1/expansion", "expansion", None, _extraction(4, "G4")),
            ],
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (?, 'G', ?, ?)",
            [
                (1, "10.1/relevant", "recent_evidence"),
                (1, "10.1/not-relevant", "recent_evidence"),
                (2, "10.1/not-relevant", "recent_evidence"),
                (3, "10.1/not-relevant", "relevance_assessment"),
                (4, "10.1/expansion", "expansion_evidence"),
            ],
        )
        history = stage_history(conn, STAGE)

    assert genes_to_assess(db_path, None, history) == [1]
    processor = PaperBatchProcessor(db_path)
    assert processor.count_remaining() == 1
    assert [e["doi"] for e in processor.get_evidence_for_gene(1)] == [
        "10.1/not-relevant",
        "10.1/relevant",
    ]


def _result(custom_id: str, answer: dict[str, Any]) -> LlmResult:
    return LlmResult(
        custom_id=custom_id,
        batch_id=None,
        stage=STAGE,
        subject=str(GENEA),
        round=1,
        model=MODEL,
        status=ResultStatus.SUCCEEDED,
        message=_message(answer),
        error_type=None,
        error_message=None,
    )


def test_quotes_not_copied_from_the_extractions_never_reject_a_gene(
    tmp_path: Path, gencc_index: GenccIndex
) -> None:
    """Every quote is new to the extractions: the gene is stored with all but placeholders."""
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
    association = _association()
    association["evidence_assessments"]["criterion_A"]["citations"] = [
        {"paper_id": "Smith2024", "quote": "Three families were affected.", "commentary": "c"},
        {"paper_id": "Smith2024", "quote": "placeholder", "commentary": "placeholder"},
    ]
    answer = _answer(association)
    answer["quality_concerns"][0]["citations"] = [
        {"paper_id": "Jones2023", "quote": "Segregation was not tested."}
    ]
    validator = jsonschema.Draft202012Validator(SCHEMA)

    outcome = handle_results(
        [_result("assess_genes-1-a", answer)],
        {str(GENEA): _item(gencc_index)},
        db_path,
        validator,
    )

    assert outcome.stored == 1
    with sqlite3.connect(db_path) as conn:
        (stored,) = conn.execute("SELECT assessment_json FROM associations").fetchone()
        (concerns,) = conn.execute("SELECT quality_concerns_json FROM gene_aggregations").fetchone()
        (rejection,) = conn.execute("SELECT rejection FROM llm_requests").fetchone()
    criterion_a = json.loads(stored)["evidence_assessments"][0]
    assert [c["quote"] for c in criterion_a["citations"]] == ["Three families were affected."]
    assert [c["quote"] for c in json.loads(concerns)[0]["citations"]] == [
        "Segregation was not tested."
    ]
    assert rejection is None


def test_a_rejected_assessment_records_its_reason(tmp_path: Path, gencc_index: GenccIndex) -> None:
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
    validator = jsonschema.Draft202012Validator(SCHEMA)
    schema_violation = _answer(_association(summary=""))
    inconsistent = _answer(_association(independent_family_count=5))

    outcome = handle_results(
        [
            _result("assess_genes-1-a", schema_violation),
            _result("assess_genes-1-b", inconsistent),
        ],
        {str(GENEA): _item(gencc_index)},
        db_path,
        validator,
    )

    assert outcome.failed == 2
    with sqlite3.connect(db_path) as conn:
        rejections = dict(conn.execute("SELECT custom_id, rejection FROM llm_requests"))
    assert rejections["assess_genes-1-a"].startswith(
        "schema violation at $.disease_entities[0].summary: "
    )
    assert rejections["assess_genes-1-b"] == "inconsistent independent_family_count"
