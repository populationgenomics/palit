"""Tests for read-mechanisms and assess-mechanisms: readings per paper, mechanisms per
association, and the ClinGen dosage sensitivity table they are shown."""

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from anthropic.types import Message

from palit import assess_mechanisms, read_mechanisms
from palit.assess_mechanisms import (
    Association,
    association_text,
    parse_assessment,
    select_associations,
)
from palit.clingen_dosage import TABLE_COLUMNS, DosageCuration, load_dosage
from palit.extract_evidence import UploadedPdf
from palit.hgnc import HgncResolver
from palit.llm import MODEL, LlmResult, ResultStatus, json_output_config, stage_history
from palit.mechanisms import (
    AssertedMechanism,
    AssociationReading,
    EvidenceItem,
    MechanismAssessment,
    review_comment,
)
from palit.read_mechanisms import (
    CitingAssociation,
    PaperToRead,
    build_request,
    count_unread,
    parse_reading,
    prompt_environment,
    select_papers,
    user_text,
)

ROOT = Path(__file__).resolve().parents[1]
READING_SCHEMA: dict[str, Any] = json.loads((ROOT / read_mechanisms.SCHEMA_PATH).read_text())
ASSESSMENT_SCHEMA: dict[str, Any] = json.loads((ROOT / assess_mechanisms.SCHEMA_PATH).read_text())

GENE = 1
SMITH = "10.1/smith"  # cited by both associations
JONES = "10.1/jones"  # cited by the monoallelic association only
PAPER_IDS = {"Smith2020": SMITH, "Jones2018": JONES}
MONOALLELIC, BIALLELIC = 10, 11


def _assessment(description: str, dois: tuple[str, ...], **fields: Any) -> str:
    """A stored association with the fields the mechanism stages read; RED unless
    *independent* gives it more than one independent family (AMBER)."""
    return json.dumps(
        {
            "description": description,
            "inheritance_mode": fields.get("inheritance_mode", "Monoallelic"),
            "inheritance_details": "",
            "independent_family_count": fields.get("independent", 1),
            "summary": f"SUMMARY OF {description}",
            "functional": "FUNCTIONAL TEXT",
            "disease_mechanism": fields.get("mechanism_text", "NR"),
            "variants": [],
            "dois": list(dois),
            "evidence_assessments": [],
        }
    )


def _item(mechanism: str = "loss_of_function") -> dict[str, str]:
    return {
        "mechanism": mechanism,
        "evidence_type": "variant_functional",
        "finding": f"FINDING {mechanism}",
        "quote": "A verbatim\nquote.",
    }


def _reading(*entries: tuple[int, list[dict[str, str]]]) -> str:
    return json.dumps(
        {
            "associations": [
                {"association_id": association_id, "authors_conclusion": "NR", "evidence": items}
                for association_id, items in entries
            ]
        }
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """GENEA with a monoallelic association citing Smith and Jones (AMBER) and a
    biallelic one citing Smith (RED)."""
    path = tmp_path / "run.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, abstract, source, source_date) "
            "VALUES (?, ?, 'ABSTRACT', 'pubmed', '2020-01-01')",
            [(SMITH, "Smith title"), (JONES, "Jones title")],
        )
        conn.execute(
            "INSERT INTO gene_aggregations (hgnc_id, assessment_raw, paper_id_mapping, "
            "panelapp_context_json, unassessed_reports_json) VALUES (?, '{}', ?, ?, '[]')",
            (
                GENE,
                json.dumps(PAPER_IDS),
                json.dumps(
                    {
                        "clingen_curations": [
                            {
                                "mondo_id": "MONDO:0000010",
                                "disease_label": "GENEA-related disease",
                                "moi": "Autosomal dominant",
                                "classification": "Definitive",
                                "date": "2023-06-15",
                                "expert_panel": "Test Expert Panel",
                                "report_url": "https://example.org/report",
                                "evidence_summary": "The mechanism is\nloss of function.",
                            }
                        ]
                    }
                ),
            ),
        )
        conn.executemany(
            "INSERT INTO associations (id, hgnc_id, position, assessment_json) VALUES (?, ?, ?, ?)",
            [
                (
                    MONOALLELIC,
                    GENE,
                    0,
                    _assessment(
                        "dominant disease",
                        (SMITH, JONES),
                        independent=2,
                        mechanism_text="proposed gain of function",
                    ),
                ),
                (
                    BIALLELIC,
                    GENE,
                    1,
                    _assessment("recessive disease", (SMITH,), inheritance_mode="Biallelic"),
                ),
            ],
        )
    return path


def _history(db_path: Path, stage: str) -> Any:
    with sqlite3.connect(db_path) as conn:
        return stage_history(conn, stage)


def _store_reading(db_path: Path, doi: str, reading_json: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO mechanism_readings (doi, reading_json, reading_raw) "
            "VALUES (?, ?, '{}')",
            (doi, reading_json),
        )


# ---------------------------------------------------------------------------
# read-mechanisms
# ---------------------------------------------------------------------------


def test_each_cited_paper_is_read_once_for_all_associations_citing_it(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    papers = select_papers(db_path, hgnc_resolver, None, _history(db_path, read_mechanisms.STAGE))
    assert [(p.paper["doi"], [a.id for a in p.associations]) for p in papers] == [
        (JONES, [MONOALLELIC]),
        (SMITH, [MONOALLELIC, BIALLELIC]),
    ]
    smith = papers[1]
    assert smith.associations[1] == CitingAssociation(
        id=BIALLELIC, gene_symbol="GENEA", description="recessive disease", inheritance="Biallelic"
    )


def test_a_paper_is_read_again_when_its_reading_covers_other_associations(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    history = _history(db_path, read_mechanisms.STAGE)
    _store_reading(db_path, JONES, _reading((MONOALLELIC, [])))
    # Smith's reading is from an earlier aggregation, whose associations had other ids
    _store_reading(db_path, SMITH, _reading((MONOALLELIC, []), (99, [])))
    assert [p.paper["doi"] for p in select_papers(db_path, hgnc_resolver, None, history)] == [SMITH]
    assert count_unread(db_path, hgnc_resolver, None) == 1
    _store_reading(db_path, SMITH, _reading((MONOALLELIC, []), (BIALLELIC, [])))
    assert select_papers(db_path, hgnc_resolver, None, history) == []


def test_only_the_papers_of_the_requested_genes_are_read(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    history = _history(db_path, read_mechanisms.STAGE)
    assert select_papers(db_path, hgnc_resolver, [3], history) == []
    assert len(select_papers(db_path, hgnc_resolver, [GENE], history)) == 2


def _to_read() -> PaperToRead:
    return PaperToRead(
        paper={"doi": SMITH, "title": "Smith title", "source_date": "2020-01-01", "abstract": "A"},
        associations=(
            CitingAssociation(MONOALLELIC, "GENEA", "dominant disease", "Monoallelic"),
            CitingAssociation(BIALLELIC, "GENEA", "recessive disease", "Biallelic"),
        ),
    )


def test_the_request_carries_the_pdf_and_the_citing_associations() -> None:
    to_read = _to_read()
    pdf = UploadedPdf(file_id="file_1", pages=4, total_pages=9)
    text = user_text(to_read, pdf)
    assert "first 4 of the paper's 9 pages" in text
    assert (
        "- A11: gene GENEA; disease: recessive disease; mode of inheritance: Biallelic"
        in text.splitlines()
    )
    system = prompt_environment().get_template(read_mechanisms.PROMPT_NAME).render()
    request = build_request(
        to_read, pdf, system, json_output_config(READING_SCHEMA, read_mechanisms.EFFORT), MODEL
    )
    assert request.subject == SMITH
    messages: list[dict[str, Any]] = json.loads(json.dumps(list(request.params["messages"])))
    document, text_block = messages[0]["content"]
    assert document == {"type": "document", "source": {"type": "file", "file_id": "file_1"}}
    assert text_block == {"type": "text", "text": text}
    assert "not_loss_of_function (NOT LOF)" in system
    assert "{%" not in system


def test_a_reading_names_its_associations_by_their_ids() -> None:
    answer = {
        "associations": [
            {"association": "A11", "evidence": [], "authors_conclusion": "NR"},
            {"association": "A10", "evidence": [_item()], "authors_conclusion": "The authors..."},
        ]
    }
    jsonschema.Draft202012Validator(READING_SCHEMA).validate(answer)
    reading = parse_reading(answer, _to_read().associations)
    assert [entry["association_id"] for entry in reading["associations"]] == [11, 10]
    assert reading["associations"][1]["evidence"] == [_item()]


@pytest.mark.parametrize("labels", [["A10"], ["A10", "A10"], ["A10", "A11", "A12"]])
def test_a_reading_must_answer_for_every_citing_association_once(labels: list[str]) -> None:
    answer = {
        "associations": [
            {"association": label, "evidence": [], "authors_conclusion": "NR"} for label in labels
        ]
    }
    with pytest.raises(ValueError, match="asked for"):
        parse_reading(answer, _to_read().associations)


def _message(answer: dict[str, Any]) -> Message:
    return Message.model_validate(
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


def _result(stage: str, subject: str, answer: dict[str, Any]) -> LlmResult:
    return LlmResult(
        custom_id=f"{stage}-1-{subject}",
        batch_id=None,
        stage=stage,
        subject=subject,
        round=1,
        model=MODEL,
        status=ResultStatus.SUCCEEDED,
        message=_message(answer),
        error_type=None,
        error_message=None,
    )


def test_a_valid_reading_is_stored_and_an_invalid_one_rejected(db_path: Path) -> None:
    validator = jsonschema.Draft202012Validator(READING_SCHEMA)
    associations = _to_read().associations
    incomplete = {
        "associations": [{"association": "A10", "evidence": [], "authors_conclusion": "NR"}]
    }
    stage = read_mechanisms.STAGE
    assert not read_mechanisms.handle_result(
        _result(stage, SMITH, incomplete), db_path, validator, associations
    )
    complete = {
        "associations": [
            {"association": "A10", "evidence": [_item()], "authors_conclusion": "NR"},
            {"association": "A11", "evidence": [], "authors_conclusion": "NR"},
        ]
    }
    result = _result(stage, SMITH, complete)
    result = LlmResult(**{**result.__dict__, "custom_id": "second"})
    assert read_mechanisms.handle_result(result, db_path, validator, associations)
    with sqlite3.connect(db_path) as conn:
        rejections = conn.execute(
            "SELECT rejection IS NOT NULL FROM llm_requests ORDER BY completed_at"
        ).fetchall()
        (stored,) = conn.execute(
            "SELECT reading_json FROM mechanism_readings WHERE doi = ?", (SMITH,)
        ).fetchone()
    assert rejections == [(1,), (0,)]
    assert json.loads(stored)["associations"][0] == {
        "association_id": MONOALLELIC,
        "authors_conclusion": "NR",
        "evidence": [_item()],
    }


# ---------------------------------------------------------------------------
# assess-mechanisms
# ---------------------------------------------------------------------------


def _associations(db_path: Path) -> dict[int, Association]:
    history = _history(db_path, assess_mechanisms.STAGE)
    return {a.id: a for a in select_associations(db_path, history)}


def test_an_association_comes_with_its_papers_readings_and_its_siblings(db_path: Path) -> None:
    _store_reading(
        db_path, SMITH, _reading((MONOALLELIC, [_item("gain_of_function")]), (BIALLELIC, []))
    )
    associations = _associations(db_path)
    monoallelic = associations[MONOALLELIC]
    assert (monoallelic.rating, monoallelic.capped) == (2, False)
    assert [(p.paper_id, p.reading is not None) for p in monoallelic.papers] == [
        ("Jones2018", False),
        ("Smith2020", True),
    ]
    assert monoallelic.unread_papers == ["Jones2018"]
    assert monoallelic.papers[1].reading == AssociationReading(
        association_id=MONOALLELIC,
        authors_conclusion="NR",
        evidence=(
            EvidenceItem(
                "gain_of_function",
                "variant_functional",
                "FINDING gain_of_function",
                "A verbatim\nquote.",
            ),
        ),
    )
    assert [(o.description, o.inheritance) for o in monoallelic.others] == [
        ("recessive disease", "Biallelic")
    ]
    biallelic = associations[BIALLELIC]
    assert (biallelic.rating, biallelic.capped, biallelic.unread_papers) == (1, True, [])
    assert biallelic.paper_id_to_doi == {"Smith2020": SMITH}
    assert biallelic.others[0].mechanism_text == "proposed gain of function"


def test_an_association_with_mechanisms_is_not_assessed_again(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        assert assess_mechanisms.store_mechanisms(
            conn, BIALLELIC, MechanismAssessment((), "Nothing reported."), "{}"
        )
        assert not assess_mechanisms.store_mechanisms(
            conn, BIALLELIC, MechanismAssessment((), "Again."), "{}"
        )
    assert list(_associations(db_path)) == [MONOALLELIC]
    assert assess_mechanisms.count_unassessed(db_path) == 1


def test_the_request_shows_the_association_the_curations_and_the_evidence(db_path: Path) -> None:
    _store_reading(db_path, SMITH, _reading((MONOALLELIC, []), (BIALLELIC, [_item()])))
    biallelic = _associations(db_path)[BIALLELIC]
    template = prompt_environment().get_template(assess_mechanisms.ASSOCIATION_TEMPLATE_NAME)
    dosage = DosageCuration(
        "Sufficient Evidence for Haploinsufficiency",
        "No Evidence for Triplosensitivity",
        "https://example.org",
        "2025-09-24",
    )
    text = association_text(biallelic, template, "GENEA", dosage)
    lines = text.splitlines()
    assert "Disease: recessive disease" in lines
    assert "Rating from the papers: Red (every confidence term is capped at Suspected)" in lines
    assert (
        "- dominant disease; mode of inheritance: Monoallelic; mechanism, as the literature "
        "review summarised it: proposed gain of function" in lines
    )
    assert "Haploinsufficiency: Sufficient Evidence for Haploinsufficiency." in text
    assert "  Evidence summary: The mechanism is loss of function." in lines
    assert "Paper Smith2020:" in lines
    assert "  - [loss_of_function; variant_functional] FINDING loss_of_function" in lines
    assert '    Quote: "A verbatim quote."' in lines
    assert "None recorded." in lines  # no variants
    assert "{%" not in text

    monoallelic = _associations(db_path)[MONOALLELIC]
    text = association_text(monoallelic, template, "GENEA", None)
    assert "Rating from the papers: Amber" in text.splitlines()
    assert "Not curated." in text
    assert "  Not read for mechanism evidence." in text.splitlines()
    assert "  No evidence on the mechanism of this association." in text.splitlines()


def _answer(*mechanisms: tuple[str, str], unknown_note: str = "", **fields: Any) -> dict[str, Any]:
    return {
        "mechanisms": [
            {
                "mechanism": mechanism,
                "confidence": confidence,
                "statement": fields.get("statement", f"Statement on {mechanism} (Smith2020)."),
                "paper_ids": fields.get("paper_ids", ["Smith2020"]),
            }
            for mechanism, confidence in mechanisms
        ],
        "unknown_note": unknown_note,
    }


def test_asserted_mechanisms_are_stored_with_their_papers_dois(db_path: Path) -> None:
    monoallelic = _associations(db_path)[MONOALLELIC]
    answer = _answer(
        ("gain_of_function", "Established"),
        ("not_loss_of_function", "Likely"),
        paper_ids=["Smith2020", "Jones2018", "Smith2020"],
    )
    jsonschema.Draft202012Validator(ASSESSMENT_SCHEMA).validate(answer)
    assessment = parse_assessment(answer, monoallelic)
    assert assessment.mechanisms[0] == AssertedMechanism(
        mechanism="gain_of_function",
        confidence="Established",
        statement="Statement on gain_of_function (Smith2020).",
        dois=(SMITH, JONES),
    )
    assert not assessment.unknown
    assert assessment.confidence_of("not_loss_of_function") == "Likely"
    assert assessment.confidence_of("dominant_negative") is None
    assert assessment.statement_of("dominant_negative") == ""
    assert MechanismAssessment.from_json(assessment.to_json()) == assessment


def test_paper_ids_are_spelled_as_the_aggregation_gave_them(db_path: Path) -> None:
    monoallelic = _associations(db_path)[MONOALLELIC]
    answer = _answer(
        ("loss_of_function", "Likely"),
        statement="No effect in yeast (SMITH2020), unlike JONES2018 and Jonesy2018.",
        paper_ids=["smith2020"],
    )
    (mechanism,) = parse_assessment(answer, monoallelic).mechanisms
    assert mechanism.statement == "No effect in yeast (Smith2020), unlike Jones2018 and Jonesy2018."
    assert mechanism.dois == (SMITH,)


def test_the_review_comment_carries_the_mechanism_notes() -> None:
    asserted = MechanismAssessment(
        mechanisms=(
            AssertedMechanism("dominant_negative", "Established", "Mutant blocks wild type.", ()),
            AssertedMechanism("not_loss_of_function", "Suspected", "Carriers are healthy.", ()),
        ),
        unknown_note="",
    )
    assert review_comment("SUMMARY", asserted) == (
        "SUMMARY\n\nMechanism of disease:\n"
        "- Dominant negative (established): Mutant blocks wild type.\n"
        "- Not loss of function (suspected): Carriers are healthy."
    )
    unknown = MechanismAssessment(mechanisms=(), unknown_note="Only proposed.")
    assert unknown.notes() == "Mechanism of disease:\n- Unknown: Only proposed."
    assert review_comment("SUMMARY", None) == "SUMMARY"


def test_an_unknown_mechanism_needs_its_note(db_path: Path) -> None:
    biallelic = _associations(db_path)[BIALLELIC]
    assessment = parse_assessment(_answer(unknown_note=" Only proposed. "), biallelic)
    assert (assessment.unknown, assessment.unknown_note) == (True, "Only proposed.")
    with pytest.raises(ValueError, match="neither mechanisms nor unknown_note"):
        parse_assessment(_answer(), biallelic)


@pytest.mark.parametrize(
    ("association_id", "answer", "problem"),
    [
        (
            MONOALLELIC,
            _answer(("loss_of_function", "Likely"), ("loss_of_function", "Suspected")),
            "asserted more than once",
        ),
        (
            BIALLELIC,
            _answer(("dominant_negative", "Suspected")),
            "dominant_negative asserted for a biallelic association",
        ),
        (
            BIALLELIC,
            _answer(("loss_of_function", "Likely")),
            "above the cap of Suspected for limited evidence",
        ),
        (
            MONOALLELIC,
            _answer(("loss_of_function", "Likely"), unknown_note="unclear"),
            "unknown_note given although mechanisms are asserted",
        ),
        (
            BIALLELIC,
            _answer(("loss_of_function", "Suspected"), paper_ids=["Jones2018"]),
            "unknown paper IDs",
        ),
        (
            MONOALLELIC,
            _answer(("loss_of_function", "Likely"), statement="One.\n- Two."),
            "line break in the statement",
        ),
    ],
)
def test_an_inconsistent_answer_is_rejected(
    db_path: Path, association_id: int, answer: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        parse_assessment(answer, _associations(db_path)[association_id])


def test_valid_mechanisms_are_stored_on_the_association(db_path: Path) -> None:
    validator = jsonschema.Draft202012Validator(ASSESSMENT_SCHEMA)
    biallelic = _associations(db_path)[BIALLELIC]
    stage = assess_mechanisms.STAGE
    rejected = _result(stage, str(BIALLELIC), _answer(("loss_of_function", "Established")))
    assert not assess_mechanisms.handle_result(rejected, db_path, validator, biallelic)
    accepted = _result(stage, str(BIALLELIC), _answer(("loss_of_function", "Suspected")))
    accepted = LlmResult(**{**accepted.__dict__, "custom_id": "second"})
    assert assess_mechanisms.handle_result(accepted, db_path, validator, biallelic)
    with sqlite3.connect(db_path) as conn:
        (stored,) = conn.execute(
            "SELECT mechanisms_json FROM associations WHERE id = ?", (BIALLELIC,)
        ).fetchone()
        (rejection,) = conn.execute(
            "SELECT rejection FROM llm_requests WHERE rejection IS NOT NULL"
        ).fetchone()
    assert json.loads(stored) == {
        "mechanisms": [
            {
                "mechanism": "loss_of_function",
                "confidence": "Suspected",
                "statement": "Statement on loss_of_function (Smith2020).",
                "dois": [SMITH],
            }
        ],
        "unknown_note": "",
    }
    assert "above the cap" in rejection


# ---------------------------------------------------------------------------
# ClinGen dosage sensitivity
# ---------------------------------------------------------------------------


def _dosage_table(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["CLINGEN DOSAGE SENSITIVITY CURATIONS", "", "", "", "", ""])
        writer.writerow(["+" * len(column) for column in TABLE_COLUMNS])
        writer.writerow(TABLE_COLUMNS)
        writer.writerow(["+" * len(column) for column in TABLE_COLUMNS])
        writer.writerows(rows)
    return path


DOSAGE_ROW = [
    "GENEA",
    "HGNC:1",
    "Sufficient Evidence for Haploinsufficiency",
    "No Evidence for Triplosensitivity",
    "https://search.clinicalgenome.org/kb/gene-dosage/HGNC:1",
    "2025-09-24T12:30:07-04:00",
]


def test_dosage_curations_are_keyed_by_hgnc_id(tmp_path: Path) -> None:
    table = _dosage_table(tmp_path / "dosage.csv", [DOSAGE_ROW])
    assert load_dosage(table) == {
        1: DosageCuration(
            haploinsufficiency="Sufficient Evidence for Haploinsufficiency",
            triplosensitivity="No Evidence for Triplosensitivity",
            report_url="https://search.clinicalgenome.org/kb/gene-dosage/HGNC:1",
            date="2025-09-24",
        )
    }


def test_a_dosage_table_with_an_unexpected_gene_id_or_a_repeated_gene_raises(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="not an HGNC id"):
        load_dosage(_dosage_table(tmp_path / "a.csv", [["G", "NCBI:1", "", "", "", ""]]))
    with pytest.raises(ValueError, match="Two ClinGen dosage sensitivity rows"):
        load_dosage(_dosage_table(tmp_path / "b.csv", [DOSAGE_ROW, DOSAGE_ROW]))
