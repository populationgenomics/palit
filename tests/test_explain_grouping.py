"""Tests for explain-grouping: one request per gene, one statement per association."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from anthropic.types import Message

from palit import explain_grouping
from palit.explain_grouping import (
    count_unexplained,
    gene_text,
    handle_result,
    parse_statements,
    select_genes,
)
from palit.llm import MODEL, LlmResult, ResultStatus, stage_history
from palit.read_mechanisms import prompt_environment

ROOT = Path(__file__).resolve().parents[1]
SCHEMA: dict[str, Any] = json.loads((ROOT / explain_grouping.SCHEMA_PATH).read_text())

GENE = 1
LUMPED, SINGLE = 10, 11
GENCC_TERMS = [
    {
        "mondo_id": "MONDO:0000010",
        "disease_title": "disease A",
        "moi_titles": ["Autosomal recessive"],
    },
    {
        "mondo_id": "MONDO:0000011",
        "disease_title": "disease B",
        "moi_titles": ["Autosomal recessive"],
    },
]


def _assessment(mondo_id: str | None, mode: str, rationale: str) -> str:
    return json.dumps(
        {
            "description": f"description of the {mode.lower()} disease",
            "inheritance_mode": mode,
            "inheritance_details": "",
            "independent_family_count": 1,
            "evidence_assessments": [],
            "existing_association_mondo_id": mondo_id,
            "proposed_disease_name": None if mondo_id else "GENEA-related dominant disease",
            "panelapp_relation": {
                "status": "existing",
                "existing_rating": "GREEN",
                "basis": "BASIS",
            },
            "grouping_rationale": rationale,
        }
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """GENEA with a biallelic association named by a GenCC term, with mechanisms, and a
    monoallelic one with a proposed name, without."""
    path = tmp_path / "run.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.execute(
            "INSERT INTO gene_aggregations (hgnc_id, assessment_raw, paper_id_mapping, "
            "panelapp_context_json, unassessed_reports_json) VALUES (?, '{}', ?, ?, '[]')",
            (
                GENE,
                json.dumps({"Smith2020": "10.1/smith"}),
                json.dumps(
                    {
                        "gencc_terms": GENCC_TERMS,
                        "panel_entries": [
                            {
                                "panel_name": "Mendeliome",
                                "rating": "GREEN",
                                "moi": "BIALLELIC, autosomal or pseudoautosomal",
                                "phenotypes": ["Disease A, MIM# 100001", "Disease B, MIM# 100002"],
                            }
                        ],
                    }
                ),
            ),
        )
        conn.executemany(
            "INSERT INTO associations (id, hgnc_id, position, assessment_json, mondo_id, "
            "mondo_label, mondo_match, mechanisms_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    LUMPED,
                    GENE,
                    0,
                    _assessment("MONDO:0000010", "Biallelic", "Smith2020 lumps A and B (STEP 2)."),
                    "MONDO:0000010",
                    "disease A",
                    "gencc",
                    json.dumps(
                        {
                            "mechanisms": [
                                {
                                    "mechanism": "loss_of_function",
                                    "confidence": "Established",
                                    "statement": "s",
                                    "dois": [],
                                }
                            ],
                            "unknown_note": "",
                        }
                    ),
                ),
                (
                    SINGLE,
                    GENE,
                    1,
                    _assessment(None, "Monoallelic", "Kept apart by MoI."),
                    None,
                    None,
                    None,
                    None,
                ),
            ],
        )
    return path


def _gene(db_path: Path) -> explain_grouping.GeneToExplain:
    with sqlite3.connect(db_path) as conn:
        history = stage_history(conn, explain_grouping.STAGE)
    (gene,) = select_genes(db_path, history)
    return gene


def test_a_gene_comes_with_its_associations_and_its_listed_terms(db_path: Path) -> None:
    gene = _gene(db_path)
    lumped, single = gene.associations
    assert (lumped.label, lumped.name, lumped.inheritance, lumped.rating) == (
        "A10",
        "disease A",
        "Biallelic",
        "Red",
    )
    assert lumped.mondo == "disease A (MONDO:0000010), the disease's own term"
    assert lumped.mechanisms == "Loss of function (established)"
    assert (single.name, single.mondo, single.mechanisms) == (
        "GENEA-related dominant disease",
        "none yet",
        "not assessed",
    )
    assert gene.paper_ids == {"Smith2020"}
    template = prompt_environment().get_template(explain_grouping.GENE_TEMPLATE_NAME)
    lines = gene_text(gene, template, "GENEA").splitlines()
    assert "A10: disease A" in lines
    assert "  How it was grouped: Smith2020 lumps A and B (STEP 2)." in lines
    assert "- disease B (MONDO:0000011) | Autosomal recessive" in lines
    assert (
        "- Mendeliome: GREEN; BIALLELIC, autosomal or pseudoautosomal; phenotypes: "
        "Disease A, MIM# 100001; Disease B, MIM# 100002" in lines
    )
    assert lines[-1] == "None."  # no ClinGen curations


def _answer(**statements: str) -> dict[str, Any]:
    return {
        "associations": [
            {"association": label, "statement": statement}
            for label, statement in statements.items()
        ]
    }


def test_statements_are_keyed_by_association(db_path: Path) -> None:
    answer = _answer(A11=" It is a single entity. ", A10="It lumps disease A and disease B.")
    jsonschema.Draft202012Validator(SCHEMA).validate(answer)
    assert parse_statements(answer, _gene(db_path)) == {
        SINGLE: "It is a single entity.",
        LUMPED: "It lumps disease A and disease B.",
    }


@pytest.mark.parametrize(
    ("answer", "problem"),
    [
        (_answer(A10="x"), "asked for"),
        (_answer(A10="x", A11="y", A12="z"), "asked for"),
        (_answer(A10="As SMITH2020 shows, lumped.", A11="y"), "cites the paper SMITH2020"),
        (_answer(A10="One.\nTwo.", A11="y"), "line break in the statement of A10"),
        (_answer(A10="word " * 111, A11="y"), "the statement of A10 has 111 words"),
    ],
)
def test_an_unusable_answer_is_rejected(
    db_path: Path, answer: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        parse_statements(answer, _gene(db_path))


def _result(answer: dict[str, Any], custom_id: str) -> LlmResult:
    message = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [{"type": "text", "text": json.dumps(answer)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    return LlmResult(
        custom_id=custom_id,
        batch_id=None,
        stage=explain_grouping.STAGE,
        subject=str(GENE),
        round=1,
        model=MODEL,
        status=ResultStatus.SUCCEEDED,
        message=message,
        error_type=None,
        error_message=None,
    )


def test_valid_statements_are_stored_on_the_associations(db_path: Path) -> None:
    validator = jsonschema.Draft202012Validator(SCHEMA)
    gene = _gene(db_path)
    assert not handle_result(_result(_answer(A10="x"), "first"), db_path, validator, gene)
    assert count_unexplained(db_path) == 2
    answer = _answer(A10="Lumps A and B.", A11="A single entity.")
    assert handle_result(_result(answer, "second"), db_path, validator, gene)
    with sqlite3.connect(db_path) as conn:
        stored = dict(conn.execute("SELECT id, grouping_statement FROM associations"))
        (raw,) = conn.execute("SELECT grouping_statements_raw FROM gene_aggregations").fetchone()
        history = stage_history(conn, explain_grouping.STAGE)
    assert stored == {LUMPED: "Lumps A and B.", SINGLE: "A single entity."}
    assert json.loads(raw)["id"] == "msg_1"
    assert select_genes(db_path, history) == []


def test_statements_for_associations_that_are_gone_are_not_stored(db_path: Path) -> None:
    gene = _gene(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE associations SET id = 12 WHERE id = ?", (SINGLE,))
    answer = _answer(A10="Lumps A and B.", A11="A single entity.")
    validator = jsonschema.Draft202012Validator(SCHEMA)
    assert not handle_result(_result(answer, "only"), db_path, validator, gene)
    assert count_unexplained(db_path) == 2
