"""Tests for match-panels: one request per association, matches stored on its row."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from anthropic.types import Message

from palit.llm import (
    FALLBACK_MODEL,
    MODEL,
    LlmRequest,
    LlmResult,
    ResultStatus,
    StageHistory,
    json_output_config,
    stage_history,
)
from palit.match_panels import (
    EFFORT,
    PROMPT_PATH,
    SCHEMA_PATH,
    STAGE,
    Association,
    build_request,
    count_unmatched,
    handle_results,
    select_associations,
    split_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA: dict[str, Any] = json.loads((ROOT / SCHEMA_PATH).read_text())
PANEL_LIST = '<panel id="250">\nName: Hereditary Spastic Paraplegia\n</panel>'
NAME_TO_ID = {"hereditary spastic paraplegia": 250, "mendeliome": 137}


def _assessment(description: str, **fields: str) -> str:
    return json.dumps(
        {
            "description": description,
            "inheritance_mode": fields.get("inheritance_mode", "Biallelic"),
            "inheritance_details": fields.get("inheritance_details", ""),
            "summary": fields.get("summary", "Five families with spastic paraplegia."),
        }
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Gene 11273 with a matched association (1), unmatched ones (2, 4), one this stage
    refused for good (3) and one in flight (5). Association 4 was refused by MODEL here and
    by map-mondo."""
    path = tmp_path / "run.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.execute(
            """
            INSERT INTO gene_aggregations (hgnc_id, assessment_raw, paper_id_mapping,
                panelapp_context_json, unassessed_reports_json, quality_concerns_json)
            VALUES (11273, '{}', '{}', '{}', '[]', '[]')
            """
        )
        conn.executemany(
            "INSERT INTO associations (id, hgnc_id, position, assessment_json) VALUES (?, 11273, ?, ?)",
            [
                (1, 0, _assessment("matched disease")),
                (
                    2,
                    1,
                    _assessment(
                        "adult-onset hereditary spastic paraplegia",
                        inheritance_details="reduced penetrance",
                    ),
                ),
                (3, 2, _assessment("refused disease")),
                (4, 3, _assessment("unclear disease", inheritance_mode="NR")),
                (5, 4, _assessment("disease in flight")),
            ],
        )
        conn.execute(
            "UPDATE associations SET matched_panels_json = '[]', matched_panels_raw = '{}' WHERE id = 1"
        )
        conn.executemany(
            """
            INSERT INTO llm_requests (custom_id, stage, subject, round, model, status,
                                      completed_at)
            VALUES (?, ?, ?, 1, ?, ?, '2026-10-01T00:00:00+00:00')
            """,
            [
                ("match_panels-1-a", STAGE, "3", MODEL, "refused"),
                ("match_panels-1-a2", STAGE, "3", FALLBACK_MODEL, "refused"),
                ("match_panels-1-b", STAGE, "5", MODEL, "pending"),
                ("match_panels-1-c", STAGE, "2", MODEL, "errored"),
                ("match_panels-1-d", STAGE, "4", MODEL, "refused"),
                ("map_mondo-1-a", "map_mondo", "4", FALLBACK_MODEL, "refused"),
            ],
        )
    return path


def _prompt() -> tuple[str, str]:
    return split_prompt((ROOT / PROMPT_PATH).read_text(), PANEL_LIST)


def test_select_associations_skips_matched_refused_and_pending_rows(db_path: Path) -> None:
    assert select_associations(db_path, _history(db_path)) == [
        Association(
            id=2,
            description="adult-onset hereditary spastic paraplegia",
            inheritance_mode="Biallelic",
            inheritance_details="reduced penetrance",
            summary="Five families with spastic paraplegia.",
        ),
        Association(
            id=4,
            description="unclear disease",
            inheritance_mode="NR",
            inheritance_details="",
            summary="Five families with spastic paraplegia.",
        ),
    ]
    assert count_unmatched(db_path) == 4
    history = _history(db_path)
    assert (history.model_for("2"), history.model_for("4")) == (MODEL, FALLBACK_MODEL)


def test_rows_of_a_deleted_aggregation_are_neither_selected_nor_counted(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM gene_aggregations WHERE hgnc_id = 11273")
        (left_behind,) = conn.execute("SELECT COUNT(*) FROM associations").fetchone()
    assert left_behind == 5
    assert select_associations(db_path, _history(db_path)) == []
    assert count_unmatched(db_path) == 0


def test_split_prompt_keeps_the_panel_list_in_the_system_part() -> None:
    system, user_template = _prompt()
    assert PANEL_LIST in system
    assert "{panel_list}" not in system
    assert "ASSOCIATION:" not in system
    assert user_template.startswith("ASSOCIATION:")


def _history(db_path: Path) -> StageHistory:
    with sqlite3.connect(db_path) as conn:
        return stage_history(conn, STAGE)


def _sent_messages(request: LlmRequest) -> list[dict[str, Any]]:
    """The request's messages as JSON, the form the API receives."""
    messages: list[dict[str, Any]] = json.loads(json.dumps(list(request.params["messages"])))
    return messages


def test_build_request_describes_one_association(db_path: Path) -> None:
    system, user_template = _prompt()
    output_config = json_output_config(SCHEMA, EFFORT)
    first, second = (
        build_request(association, system, user_template, output_config, MODEL)
        for association in select_associations(db_path, _history(db_path))
    )
    assert first.subject == "2"
    assert first.params["system"] == second.params["system"]
    assert first.params["messages"] == [
        {
            "role": "user",
            "content": (
                "ASSOCIATION:\n"
                "Disease: adult-onset hereditary spastic paraplegia\n"
                "Mode of inheritance: Biallelic (reduced penetrance)\n"
                "Summary: Five families with spastic paraplegia.\n"
            ),
        }
    ]
    assert "Mode of inheritance: not reported\n" in _sent_messages(second)[0]["content"]


def _result(subject: str, answer: dict[str, Any], custom_id: str) -> LlmResult:
    message = Message.model_validate(
        {
            "id": f"msg_{custom_id}",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(answer)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    return LlmResult(
        custom_id=custom_id,
        batch_id=None,
        stage=STAGE,
        subject=subject,
        round=1,
        model="claude-opus-5-5",
        status=ResultStatus.SUCCEEDED,
        message=message,
        error_type=None,
        error_message=None,
    )


def _matched(db_path: Path) -> dict[int, list[dict[str, Any]] | None]:
    with sqlite3.connect(db_path) as conn:
        return {
            association_id: None if matched is None else json.loads(matched)
            for association_id, matched in conn.execute(
                "SELECT id, matched_panels_json FROM associations"
            )
        }


def test_handle_results_stores_valid_matches_on_the_association(db_path: Path) -> None:
    hsp = {"panel_name": "Hereditary Spastic Paraplegia", "rationale": "Spastic paraplegia."}
    results = [
        _result("2", {"matched_panels": [hsp]}, "match_panels-1-ok"),
        _result(
            "4",
            {"matched_panels": [{"panel_name": "Not a panel", "rationale": "Made up."}]},
            "match_panels-1-unknown",
        ),
        _result("5", {"matched_panels": [hsp, hsp]}, "match_panels-1-repeated"),
        _result("99", {"matched_panels": []}, "match_panels-1-gone"),
    ]
    validator = jsonschema.Draft202012Validator(SCHEMA)
    assert handle_results(results, db_path, validator, NAME_TO_ID) == 1

    matched = _matched(db_path)
    assert matched[2] == [{"panel_id": 250, "rationale": "Spastic paraplegia."}]
    assert matched[4] is None
    assert matched[5] is None
    with sqlite3.connect(db_path) as conn:
        recorded = {
            custom_id: (subject, status)
            for custom_id, subject, status in conn.execute(
                "SELECT custom_id, subject, status FROM llm_requests WHERE stage = ?", (STAGE,)
            )
        }
        (raw,) = conn.execute("SELECT matched_panels_raw FROM associations WHERE id = 2").fetchone()
    assert recorded["match_panels-1-ok"] == ("2", "succeeded")
    assert recorded["match_panels-1-unknown"] == ("4", "succeeded")
    assert recorded["match_panels-1-gone"] == ("99", "succeeded")
    assert json.loads(raw)["id"] == "msg_match_panels-1-ok"
