"""Tests for the two relevance levels' storage, without network."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from anthropic.types import Message

from palit.assess_relevance import (
    CHECK_STAGE,
    STAGE,
    CheckSettings,
    select_papers,
    select_screened,
    store_checks,
    store_screens,
)
from palit.hgnc import HgncResolver
from palit.llm import LlmResult, ResultStatus, json_output_config
from palit.panelapp_check import CuratedRecord

ROOT = Path(__file__).resolve().parents[1]
SCREEN_SCHEMA = json.loads((ROOT / "prompts/relevance_assessment_schema.json").read_text())
CHECK_SCHEMA = json.loads((ROOT / "prompts/relevance_panelapp_check_schema.json").read_text())


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "run.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        for doi in ("10.1/out", "10.1/nogene", "10.1/gene"):
            conn.execute(
                "INSERT INTO papers (doi, pmid, title, abstract, source) VALUES (?, 1, 't', 'a', 'pubmed')",
                (doi,),
            )
    return path


def _result(stage: str, subject: str, output: dict[str, Any]) -> LlmResult:
    message = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(output)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "stop_details": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    return LlmResult(
        custom_id=f"{stage}-{subject}",
        batch_id=None,
        stage=stage,
        subject=subject,
        round=1,
        model="claude-opus-5-5",
        status=ResultStatus.SUCCEEDED,
        message=message,
        error_type=None,
    )


def _screen(relevant: bool, genes: list[str]) -> dict[str, Any]:
    return {
        "relevant": relevant,
        "confidence": "MEDIUM",
        "rationale": "r",
        "associations": [{"gene_symbol": g, "disease": "d"} for g in genes],
    }


@pytest.fixture
def check(curated_record: CuratedRecord, hgnc_resolver: HgncResolver) -> CheckSettings:
    return CheckSettings(
        system="s",
        output_config=json_output_config(CHECK_SCHEMA, "medium"),
        validator=jsonschema.Draft202012Validator(CHECK_SCHEMA),
        record=curated_record,
        resolver=hgnc_resolver,
    )


def _row(db_path: Path, doi: str) -> sqlite3.Row:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row: sqlite3.Row = conn.execute("SELECT * FROM papers WHERE doi = ?", (doi,)).fetchone()
    return row


def test_screen_finalises_rejections_and_gene_less_papers(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    validator = jsonschema.Draft202012Validator(SCREEN_SCHEMA)
    outcome = store_screens(
        db_path,
        [
            _result(STAGE, "10.1/out", _screen(False, [])),
            _result(STAGE, "10.1/nogene", _screen(True, [])),
            _result(STAGE, "10.1/gene", _screen(True, ["GENEA", "GENEC"])),
        ],
        validator,
        hgnc_resolver,
        screen_only=False,
    )
    assert outcome.stored == 3

    out = json.loads(_row(db_path, "10.1/out")["relevance_assessment_json"])
    assert out == {"relevant": False, "screen": _screen(False, []), "panelapp_check": None}
    nogene = _row(db_path, "10.1/nogene")
    assert json.loads(nogene["relevance_assessment_json"])["relevant"] is True
    assert nogene["download_status"] == "scheduled"

    pending = _row(db_path, "10.1/gene")
    assert pending["relevance_assessment_json"] is None
    assert json.loads(pending["relevance_screen_json"])["associations"][0]["gene_symbol"] == "GENEA"
    assert select_papers(db_path, None) == []
    assert [p["doi"] for p in select_screened(db_path, None)] == ["10.1/gene"]


def test_check_finalises_with_verdicts_and_mentions(
    db_path: Path, check: CheckSettings, hgnc_resolver: HgncResolver
) -> None:
    store_screens(
        db_path,
        [_result(STAGE, "10.1/gene", _screen(True, ["GENEA", "GENEC"]))],
        jsonschema.Draft202012Validator(SCREEN_SCHEMA),
        hgnc_resolver,
        screen_only=False,
    )
    check_output = {
        "rationale": "GENEC is new",
        "associations": [
            {"gene_symbol": "GENEA", "disease": "d", "verdict": "already_curated", "reason": "r"},
            {"gene_symbol": "GENEC", "disease": "d", "verdict": "new_gene", "reason": "r"},
        ],
    }
    outcome = store_checks(db_path, [_result(CHECK_STAGE, "10.1/gene", check_output)], check)
    assert outcome.stored == 1

    row = _row(db_path, "10.1/gene")
    assessment = json.loads(row["relevance_assessment_json"])
    assert assessment["relevant"] is True
    assert assessment["panelapp_check"]["reference_panel_ids"] == [137, 126]
    assert [a["hgnc_id"] for a in assessment["panelapp_check"]["associations"]] == [1, 4]
    assert row["relevance_screen_json"] is None
    assert row["download_status"] == "scheduled"
    assert set(json.loads(row["relevance_assessment_raw"])) == {"screen", "panelapp_check"}
    with sqlite3.connect(db_path) as conn:
        mentions = conn.execute("SELECT hgnc_id, source FROM gene_mentions").fetchall()
    assert sorted(mentions) == [(1, "relevance_assessment"), (4, "relevance_assessment")]
    assert select_screened(db_path, None) == []


def test_check_with_only_curated_genes_is_not_relevant(
    db_path: Path, check: CheckSettings, hgnc_resolver: HgncResolver
) -> None:
    store_screens(
        db_path,
        [_result(STAGE, "10.1/gene", _screen(True, ["GENEA"]))],
        jsonschema.Draft202012Validator(SCREEN_SCHEMA),
        hgnc_resolver,
        screen_only=False,
    )
    check_output = {
        "rationale": "curated",
        "associations": [
            {"gene_symbol": "GENEA", "disease": "d", "verdict": "already_curated", "reason": "r"}
        ],
    }
    store_checks(db_path, [_result(CHECK_STAGE, "10.1/gene", check_output)], check)
    row = _row(db_path, "10.1/gene")
    assert json.loads(row["relevance_assessment_json"])["relevant"] is False
    assert row["download_status"] is None
    with sqlite3.connect(db_path) as conn:
        # the screen passed it, so its genes stay recorded for the baseline repository
        assert conn.execute("SELECT hgnc_id FROM gene_mentions").fetchall() == [(1,)]


def test_screen_only_decides_alone(db_path: Path, hgnc_resolver: HgncResolver) -> None:
    store_screens(
        db_path,
        [_result(STAGE, "10.1/gene", _screen(True, ["GENEA"]))],
        jsonschema.Draft202012Validator(SCREEN_SCHEMA),
        hgnc_resolver,
        screen_only=True,
    )
    row = _row(db_path, "10.1/gene")
    assessment = json.loads(row["relevance_assessment_json"])
    assert assessment["relevant"] is True and assessment["panelapp_check"] is None
    assert row["relevance_screen_json"] is None
    assert select_screened(db_path, None) == []


def test_empty_check_leaves_the_paper_for_another_attempt(
    db_path: Path, check: CheckSettings, hgnc_resolver: HgncResolver
) -> None:
    store_screens(
        db_path,
        [_result(STAGE, "10.1/gene", _screen(True, ["GENEA"]))],
        jsonschema.Draft202012Validator(SCREEN_SCHEMA),
        hgnc_resolver,
        screen_only=False,
    )
    outcome = store_checks(
        db_path,
        [_result(CHECK_STAGE, "10.1/gene", {"rationale": "r", "associations": []})],
        check,
    )
    assert (outcome.stored, outcome.failed) == (0, 1)
    assert _row(db_path, "10.1/gene")["relevance_assessment_json"] is None
