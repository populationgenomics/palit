"""Tests for match-panels: one request per association, matches stored on its row."""

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
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
    handle_result,
    match_associations,
    select_associations,
    send_primed,
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
    """Gene 11273 with a matched association (1), unmatched ones (2, 4, 5) and one this
    stage refused for good (3). Association 4 was refused by MODEL here and by map-mondo."""
    path = tmp_path / "run.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.execute(
            """
            INSERT INTO gene_aggregations (hgnc_id, assessment_raw, paper_id_mapping,
                panelapp_context_json, unassessed_reports_json)
            VALUES (11273, '{}', '{}', '{}', '[]')
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
                (5, 4, _assessment("hereditary spastic paraplegia")),
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
                ("match_panels-1-c", STAGE, "2", MODEL, "errored"),
                ("match_panels-1-d", STAGE, "4", MODEL, "refused"),
                ("map_mondo-1-a", "map_mondo", "4", FALLBACK_MODEL, "refused"),
            ],
        )
    return path


def _prompt() -> tuple[str, str]:
    return split_prompt((ROOT / PROMPT_PATH).read_text(), PANEL_LIST)


def test_select_associations_skips_matched_and_refused_rows(db_path: Path) -> None:
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
        Association(
            id=5,
            description="hereditary spastic paraplegia",
            inheritance_mode="Biallelic",
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
    first, second, _ = (
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


def _message(stop_reason: str, content: list[dict[str, Any]], message_id: str) -> Message:
    return Message.model_validate(
        {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


def _answer(answer: dict[str, Any], message_id: str = "msg_1") -> Message:
    return _message("end_turn", [{"type": "text", "text": json.dumps(answer)}], message_id)


def _refusal() -> Message:
    return _message("refusal", [], "msg_refused")


def _result(
    subject: str,
    message: Message | None,
    custom_id: str,
    *,
    model: str = MODEL,
    error_type: str | None = None,
) -> LlmResult:
    if message is None:
        status = ResultStatus.ERRORED
    elif message.stop_reason == "refusal":
        status = ResultStatus.REFUSED
    else:
        status = ResultStatus.SUCCEEDED
    return LlmResult(
        custom_id=custom_id,
        batch_id=None,
        stage=STAGE,
        subject=subject,
        round=1,
        model=model,
        status=status,
        message=message,
        error_type=error_type,
        error_message=None if error_type is None else "invalid",
    )


def _matched(db_path: Path) -> dict[int, list[dict[str, Any]] | None]:
    with sqlite3.connect(db_path) as conn:
        return {
            association_id: None if matched is None else json.loads(matched)
            for association_id, matched in conn.execute(
                "SELECT id, matched_panels_json FROM associations"
            )
        }


HSP = {"panel_name": "Hereditary Spastic Paraplegia", "rationale": "Spastic paraplegia."}


def test_handle_result_stores_valid_matches_on_the_association(db_path: Path) -> None:
    results = [
        _result("2", _answer({"matched_panels": [HSP]}, "msg_ok"), "match_panels-1-ok"),
        _result(
            "4",
            _answer({"matched_panels": [{"panel_name": "Not a panel", "rationale": "Made up."}]}),
            "match_panels-1-unknown",
        ),
        _result("5", _answer({"matched_panels": [HSP, HSP]}), "match_panels-1-repeated"),
        _result("99", _answer({"matched_panels": []}), "match_panels-1-gone"),
    ]
    validator = jsonschema.Draft202012Validator(SCHEMA)
    assert [handle_result(result, db_path, validator, NAME_TO_ID) for result in results] == [
        True,
        False,
        False,
        False,
    ]

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
    assert json.loads(raw)["id"] == "msg_ok"


def test_handle_result_records_an_invalid_request_before_raising(db_path: Path) -> None:
    validator = jsonschema.Draft202012Validator(SCHEMA)
    result = _result("2", None, "match_panels-1-invalid", error_type="invalid_request_error")
    with pytest.raises(RuntimeError, match="association 2"):
        handle_result(result, db_path, validator, NAME_TO_ID)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT subject, status, error_type FROM llm_requests WHERE custom_id = ?",
            ("match_panels-1-invalid",),
        ).fetchone() == ("2", "errored", "invalid_request_error")


def _request(subject: str, model: str) -> LlmRequest:
    system, user_template = _prompt()
    association = Association(
        id=int(subject),
        description="d",
        inheritance_mode="Biallelic",
        inheritance_details="",
        summary="s",
    )
    output_config = json_output_config(SCHEMA, EFFORT)
    return build_request(association, system, user_template, output_config, model)


def test_the_first_request_to_each_model_returns_before_its_others_are_sent() -> None:
    """Each model's first request writes the cache; the model's other requests read it."""
    events: list[tuple[str, str, str]] = []

    class RecordingTransport:
        async def run(
            self, stage: str, round_no: int, requests: Sequence[LlmRequest]
        ) -> list[LlmResult]:
            (request,) = requests
            model = request.params["model"]
            events.append(("sent", request.subject, model))
            for _ in range(3):
                await asyncio.sleep(0)
            events.append(("returned", request.subject, model))
            message = _answer({"matched_panels": []})
            return [_result(request.subject, message, f"c{request.subject}", model=model)]

        async def resume(self, stage: str) -> list[LlmResult]:
            return []

    requests = [
        _request(subject, model)
        for subject, model in [
            ("1", MODEL),
            ("2", FALLBACK_MODEL),
            ("3", MODEL),
            ("4", FALLBACK_MODEL),
            ("5", MODEL),
            ("6", FALLBACK_MODEL),
        ]
    ]
    handled: list[str] = []
    asyncio.run(
        send_primed(RecordingTransport(), requests, lambda result: handled.append(result.subject))
    )

    assert sorted(handled) == ["1", "2", "3", "4", "5", "6"]
    for model, first in [(MODEL, "1"), (FALLBACK_MODEL, "2")]:
        model_events = [(kind, subject) for kind, subject, m in events if m == model]
        assert model_events[:2] == [("sent", first), ("returned", first)]
    # The other requests overlap rather than going out one by one.
    assert events.index(("sent", "5", MODEL)) < events.index(("returned", "3", MODEL))


def test_a_refusal_by_the_model_goes_to_the_fallback_model_in_the_next_attempt(
    db_path: Path,
) -> None:
    """MODEL refuses association 2 in attempt 1, and FALLBACK_MODEL matches it in attempt 2;
    association 4, which MODEL refused earlier, goes to FALLBACK_MODEL in attempt 1."""

    class ScriptedTransport:
        async def run(
            self, stage: str, round_no: int, requests: Sequence[LlmRequest]
        ) -> list[LlmResult]:
            (request,) = requests
            model = request.params["model"]
            message = (
                _refusal()
                if (request.subject, model) == ("2", MODEL)
                else _answer({"matched_panels": [HSP]})
            )
            custom_id = f"run-{request.subject}-{model}"
            return [_result(request.subject, message, custom_id, model=model)]

        async def resume(self, stage: str) -> list[LlmResult]:
            return []

    system, user_template = _prompt()
    asyncio.run(
        match_associations(
            transport=ScriptedTransport(),
            db_path=db_path,
            schema=SCHEMA,
            system=system,
            user_template=user_template,
            name_to_id=NAME_TO_ID,
            max_retries=3,
            failures_since=datetime.min.replace(tzinfo=UTC),
        )
    )

    with sqlite3.connect(db_path) as conn:
        recorded = conn.execute(
            "SELECT subject, model, status FROM llm_requests WHERE custom_id LIKE 'run-%' "
            "ORDER BY rowid"
        ).fetchall()
    assert sorted(recorded[:3]) == [
        ("2", MODEL, "refused"),
        ("4", FALLBACK_MODEL, "succeeded"),
        ("5", MODEL, "succeeded"),
    ]
    assert recorded[3:] == [("2", FALLBACK_MODEL, "succeeded")]
    hsp_match = [{"panel_id": 250, "rationale": "Spastic paraplegia."}]
    matched = _matched(db_path)
    assert (matched[2], matched[4], matched[5]) == (hsp_match, hsp_match, hsp_match)
