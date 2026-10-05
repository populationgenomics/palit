"""Offline tests for palit.llm: batching limits, prices, bookkeeping, resume."""

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
import tenacity
from anthropic.types import Message
from anthropic.types.messages import MessageBatchIndividualResponse

from palit.llm import (
    FALLBACK_MODEL,
    MAX_FAILED_ANSWERS,
    MAX_TOKENS_REJECTION,
    MID_STREAM_ATTEMPTS,
    MIN_BATCH_REQUESTS,
    MODEL,
    BatchTransport,
    ImmediateTransport,
    LlmRequest,
    LlmResult,
    ResultStatus,
    chunk_for_batches,
    json_output_config,
    parse_json_output,
    record_result,
    request_cost,
    stage_failures,
    stage_history,
    stage_refusals,
)
from palit.llm_usage import (
    FailedSubject,
    SubjectOutcome,
    list_failed_for_good,
    list_refusals,
    outcome_counts,
    summarise_usage,
)

SCHEMA_SQL = Path(__file__).resolve().parents[1] / "schema.sql"


def _request(subject: str, text: str = "x", model: str = MODEL) -> LlmRequest:
    return LlmRequest(
        subject=subject,
        params={
            "model": model,
            "max_tokens": 16,
            "messages": [{"role": "user", "content": text}],
        },
    )


def _message(stop_reason: str, text: str = "{}") -> dict[str, Any]:
    stop_details = (
        {"type": "refusal", "category": "bio", "explanation": None}
        if stop_reason == "refusal"
        else None
    )
    return {
        "stop_details": stop_details,
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 1000,
            "cache_creation_input_tokens": 0,
            "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
            "service_tier": "batch",
        },
    }


class FakeBatches:
    """In-memory stand-in for ``client.messages.batches``."""

    def __init__(self, outcomes: dict[str, dict[str, Any]]) -> None:
        self._outcomes = outcomes  # subject text -> batch result payload
        self.created: list[list[Any]] = []
        self._responses: dict[str, list[MessageBatchIndividualResponse]] = {}

    async def create(self, requests: list[Any]) -> SimpleNamespace:
        batch_id = f"msgbatch_{len(self.created)}"
        self.created.append(requests)
        self._responses[batch_id] = [
            MessageBatchIndividualResponse.model_validate(
                {
                    "custom_id": r["custom_id"],
                    "result": self._outcomes[r["params"]["messages"][0]["content"]],
                }
            )
            for r in requests
        ]
        return SimpleNamespace(id=batch_id)

    async def retrieve(self, batch_id: str) -> SimpleNamespace:
        counts = SimpleNamespace(processing=0, succeeded=0, errored=0, expired=0, canceled=0)
        return SimpleNamespace(processing_status="ended", request_counts=counts)

    async def results(self, batch_id: str) -> AsyncIterator[MessageBatchIndividualResponse]:
        async def iterate() -> AsyncIterator[MessageBatchIndividualResponse]:
            for response in self._responses[batch_id]:
                yield response

        return iterate()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "run.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
    return path


def _transport(db_path: Path, batches: FakeBatches) -> BatchTransport:
    """A transport that batches every request, however few."""
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    return BatchTransport(client, db_path, poll_seconds=0, min_batch_requests=1)  # type: ignore[arg-type]


def test_chunking_respects_request_count() -> None:
    chunks = chunk_for_batches([_request(str(i)) for i in range(5)], max_requests=2)
    assert [len(c) for c in chunks] == [2, 2, 1]


def test_chunking_respects_byte_budget() -> None:
    requests = [_request(str(i), "y" * 1000) for i in range(3)]
    chunks = chunk_for_batches(requests, max_bytes=2500)
    assert [len(c) for c in chunks] == [2, 1]


def test_chunking_rejects_oversized_request() -> None:
    with pytest.raises(ValueError, match="above the batch size limit"):
        chunk_for_batches([_request("big", "z" * 5000)], max_bytes=1000)


def test_request_cost_batch_opus_55() -> None:
    # 1M uncached input at $2 + 1M 5m writes at $2.50 + 1M reads at $0.10 + 1M output at $10.
    cost = request_cost("claude-opus-5-5", "batch", 1_000_000, 1_000_000, 0, 1_000_000, 1_000_000)
    assert cost == pytest.approx(14.6)


def test_json_output_config_expands_nullable_types() -> None:
    config = json_output_config(
        {
            "type": "object",
            "properties": {"n": {"type": ["integer", "null"]}},
            "required": ["n"],
            "additionalProperties": False,
        },
        "low",
    )
    assert config["effort"] == "low"
    assert "anyOf" in config["format"]["schema"]["properties"]["n"]  # type: ignore[index]


def test_batch_round_trip_records_and_collects(db_path: Path) -> None:
    batches = FakeBatches(
        {
            "a": {"type": "succeeded", "message": _message("end_turn")},
            "b": {"type": "succeeded", "message": _message("refusal")},
            "c": {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "busy"},
                },
            },
        }
    )
    transport = _transport(db_path, batches)
    results = asyncio.run(
        transport.run(
            "relevance", 1, [_request("doi-a", "a"), _request("doi-b", "b"), _request("doi-c", "c")]
        )
    )
    by_subject = {r.subject: r for r in results}
    assert by_subject["doi-a"].status == ResultStatus.SUCCEEDED
    assert by_subject["doi-b"].status == ResultStatus.REFUSED
    assert by_subject["doi-c"].status == ResultStatus.ERRORED
    assert by_subject["doi-c"].error_type == "overloaded_error"
    assert by_subject["doi-c"].error_message == "busy"
    assert by_subject["doi-c"].too_long_prompt_tokens is None

    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM llm_requests WHERE status = 'pending'"
        ).fetchone() == (3,)
        for result in results[:2]:
            record_result(conn, result)
        assert conn.execute("SELECT collected_at FROM llm_batches").fetchone() == (None,)
        record_result(conn, results[2])
        assert conn.execute("SELECT collected_at IS NOT NULL FROM llm_batches").fetchone() == (1,)
        assert conn.execute(
            "SELECT input_tokens, cache_read_tokens, output_tokens, service_tier FROM llm_requests WHERE subject = 'doi-a'"
        ).fetchone() == (100, 1000, 50, "batch")


TOO_LONG_MESSAGE = "prompt is too long: 8328354 tokens > 1000000 maximum"


def _invalid_request(message: str) -> dict[str, Any]:
    return {
        "type": "errored",
        "error": {"type": "error", "error": {"type": "invalid_request_error", "message": message}},
    }


def test_batch_result_rejected_as_too_long_keeps_the_token_count(db_path: Path) -> None:
    batches = FakeBatches(
        {"long": _invalid_request(TOO_LONG_MESSAGE), "bad": _invalid_request("tools.0: bad")}
    )
    results = asyncio.run(
        _transport(db_path, batches).run(
            "extraction", 1, [_request("doi-long", "long"), _request("doi-bad", "bad")]
        )
    )
    by_subject = {r.subject: r for r in results}
    assert by_subject["doi-long"].error_type == "invalid_request_error"
    assert by_subject["doi-long"].too_long_prompt_tokens == 8_328_354
    assert by_subject["doi-bad"].too_long_prompt_tokens is None


def test_resume_returns_only_unrecorded_results(db_path: Path) -> None:
    batches = FakeBatches(
        {
            "a": {"type": "succeeded", "message": _message("end_turn")},
            "b": {"type": "succeeded", "message": _message("end_turn")},
        }
    )
    transport = _transport(db_path, batches)
    results = asyncio.run(
        transport.run("extraction", 1, [_request("doi-a", "a"), _request("doi-b", "b")])
    )
    recorded = next(r for r in results if r.subject == "doi-a")
    with sqlite3.connect(db_path) as conn:
        record_result(conn, recorded)

    resumed = asyncio.run(transport.resume("extraction"))
    assert [r.subject for r in resumed] == ["doi-b"]
    assert asyncio.run(transport.resume("relevance")) == []


def test_refusal_category_is_recorded_and_listed(db_path: Path) -> None:
    batches = FakeBatches({"b": {"type": "succeeded", "message": _message("refusal")}})
    results = asyncio.run(
        _transport(db_path, batches).run("relevance", 1, [_request("doi-b", "b")])
    )
    with sqlite3.connect(db_path) as conn:
        record_result(conn, results[0])
    [refusal] = list_refusals(db_path, "relevance")
    assert (refusal.subject, refusal.category) == ("doi-b", "bio")


def test_rejections_are_recorded_and_counted_per_stage(db_path: Path) -> None:
    batches = FakeBatches(
        {
            "a": {"type": "succeeded", "message": _message("end_turn")},
            "b": {"type": "succeeded", "message": _message("end_turn")},
        }
    )
    results = asyncio.run(
        _transport(db_path, batches).run(
            "extraction", 1, [_request("doi-a", "a"), _request("doi-b", "b")]
        )
    )
    reason = "structural: GENEA: inconsistent independent_family_count"
    with sqlite3.connect(db_path) as conn:
        record_result(conn, results[0], rejection=reason)
        record_result(conn, results[1])
        rejections = conn.execute("SELECT subject, rejection FROM llm_requests ORDER BY subject")
        assert rejections.fetchall() == [("doi-a", reason), ("doi-b", None)]
    [usage] = summarise_usage(db_path)
    assert (usage.stage, usage.requests, usage.rejected) == ("extraction", 2, 1)


def _stream_error(status: int, error_type: str) -> anthropic.APIStatusError:
    """What the SDK raises for an error: a failed response, or an event inside a stream (200)."""
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.anthropic.com"))
    body = {"type": "error", "error": {"type": error_type, "message": "x"}}
    return anthropic.APIStatusError(str(body), response=response, body=body)


class FakeStreams:
    """``client.messages.stream`` that raises the queued errors before answering."""

    def __init__(self, errors: list[Exception]) -> None:
        self._errors = errors
        self.calls = 0

    def stream(self, **params: Any) -> "FakeStreams":
        self.calls += 1
        return self

    async def __aenter__(self) -> "FakeStreams":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get_final_message(self) -> Message:
        if self._errors:
            raise self._errors.pop(0)
        return Message.model_validate(_message("end_turn"))


def _immediate(streams: FakeStreams) -> ImmediateTransport:
    client = SimpleNamespace(messages=streams)
    return ImmediateTransport(client, mid_stream_wait=tenacity.wait_none())  # type: ignore[arg-type]


def test_immediate_retries_error_events_inside_the_stream() -> None:
    streams = FakeStreams([_stream_error(200, "api_error"), _stream_error(200, "overloaded_error")])
    [result] = asyncio.run(_immediate(streams).run("test", 1, [_request("a")]))
    assert result.status == ResultStatus.SUCCEEDED
    assert streams.calls == 3


def test_immediate_reports_persistent_stream_errors_as_errored() -> None:
    streams = FakeStreams([_stream_error(200, "api_error") for _ in range(MID_STREAM_ATTEMPTS)])
    [result] = asyncio.run(_immediate(streams).run("test", 1, [_request("a")]))
    assert result.status == ResultStatus.ERRORED
    assert result.error_type == "APIStatusError"
    assert streams.calls == MID_STREAM_ATTEMPTS


def test_immediate_reports_overloaded_responses_as_errored() -> None:
    response = httpx2.Response(529, request=httpx2.Request("POST", "https://api.anthropic.com"))
    streams = FakeStreams([anthropic.OverloadedError("overloaded", response=response, body=None)])
    [result] = asyncio.run(_immediate(streams).run("test", 1, [_request("a")]))
    assert result.status == ResultStatus.ERRORED
    assert streams.calls == 1  # the SDK already retried failed responses


def test_immediate_raises_invalid_request_errors_inside_the_stream() -> None:
    streams = FakeStreams([_stream_error(200, "invalid_request_error")])
    with pytest.raises(anthropic.APIStatusError):
        asyncio.run(_immediate(streams).run("test", 1, [_request("a")]))


def test_immediate_returns_a_prompt_too_long_as_an_errored_result() -> None:
    """As a batch does, so that stages handle the rejection the same way in both modes."""
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://api.anthropic.com"))
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": TOO_LONG_MESSAGE},
    }
    error = anthropic.BadRequestError(str(body), response=response, body=body)
    [result] = asyncio.run(_immediate(FakeStreams([error])).run("test", 1, [_request("a")]))
    assert result.status == ResultStatus.ERRORED
    assert result.error_type == "invalid_request_error"
    assert result.too_long_prompt_tokens == 8_328_354


def test_immediate_raises_other_bad_requests() -> None:
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://api.anthropic.com"))
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "tools: bad"}}
    error = anthropic.BadRequestError(str(body), response=response, body=body)
    with pytest.raises(anthropic.BadRequestError):
        asyncio.run(_immediate(FakeStreams([error])).run("test", 1, [_request("a")]))


def test_request_cost_batch_sonnet_55() -> None:
    # 1M uncached input at $1 + 1M 5m writes at $1.25 + 1M 1h writes at $2
    # + 1M reads at $0.10 + 1M output at $5.
    cost = request_cost(FALLBACK_MODEL, "batch", *[1_000_000] * 5)
    assert cost == pytest.approx(9.35)
    assert request_cost(FALLBACK_MODEL, "standard", 0, 0, 0, 1_000_000, 0) == pytest.approx(0.2)


def _add_requests(db_path: Path, rows: list[tuple[str, str, str, str, str | None]]) -> None:
    """Requests as (stage, subject, model, status, completed_at)."""
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO llm_requests (custom_id, stage, subject, round, model, status,
                                      completed_at)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            """,
            [(f"r{i}", *row) for i, row in enumerate(rows)],
        )


def test_stage_refusals_route_by_the_model_that_refused(db_path: Path) -> None:
    _add_requests(
        db_path,
        [
            ("relevance", "by-model", MODEL, "refused", "2026-09-02T00:00:00+00:00"),
            ("relevance", "for-good", MODEL, "refused", "2026-09-02T00:00:00+00:00"),
            ("relevance", "for-good", FALLBACK_MODEL, "refused", "2026-09-02T01:00:00+00:00"),
            ("relevance", "old-model", "claude-opus-5", "refused", "2026-09-02T00:00:00+00:00"),
            ("relevance", "answered", MODEL, "succeeded", "2026-09-02T00:00:00+00:00"),
            ("extraction", "other-stage", MODEL, "refused", "2026-09-02T00:00:00+00:00"),
            ("relevance", "earlier", FALLBACK_MODEL, "refused", "2026-08-01T00:00:00+00:00"),
        ],
    )
    with sqlite3.connect(db_path) as conn:
        refusals = stage_refusals(conn, "relevance")
        since_september = stage_refusals(conn, "relevance", datetime(2026, 9, 1, tzinfo=UTC))
    assert refusals.by_model == {"by-model", "for-good"}
    assert refusals.by_fallback == {"for-good", "earlier"}
    assert refusals.model_for("by-model") == FALLBACK_MODEL
    for subject in ("old-model", "answered", "other-stage", "new"):
        assert refusals.model_for(subject) == MODEL
    assert refusals.refused_for_good("for-good") and not refusals.refused_for_good("by-model")
    with pytest.raises(ValueError, match="refused"):
        refusals.model_for("for-good")
    # Refusals before the cutoff no longer count: the subject starts on MODEL again.
    assert since_september.model_for("earlier") == MODEL


def test_result_goes_to_fallback_only_after_a_refusal_by_model() -> None:
    def result(model: str, status: ResultStatus) -> LlmResult:
        return LlmResult("c", None, "relevance", "s", 1, model, status, None, None, None)

    assert result(MODEL, ResultStatus.REFUSED).goes_to_fallback
    assert not result(FALLBACK_MODEL, ResultStatus.REFUSED).goes_to_fallback
    assert not result(MODEL, ResultStatus.ERRORED).goes_to_fallback


def test_batch_transport_submits_one_model_per_batch(db_path: Path) -> None:
    batches = FakeBatches(
        {t: {"type": "succeeded", "message": _message("end_turn")} for t in "abc"}
    )
    requests = [
        _request("doi-a", "a"),
        _request("doi-b", "b", FALLBACK_MODEL),
        _request("doi-c", "c"),
    ]
    results = asyncio.run(_transport(db_path, batches).run("relevance", 1, requests))
    assert sorted(len(created) for created in batches.created) == [1, 2]
    assert {r.subject: r.model for r in results} == {
        "doi-a": MODEL,
        "doi-b": FALLBACK_MODEL,
        "doi-c": MODEL,
    }
    with sqlite3.connect(db_path) as conn:
        assert sorted(conn.execute("SELECT model FROM llm_batches").fetchall()) == [
            (MODEL,),
            (FALLBACK_MODEL,),
        ]
        assert conn.execute(
            "SELECT model FROM llm_requests WHERE subject = 'doi-b'"
        ).fetchone() == (FALLBACK_MODEL,)


def test_refusals_list_the_subject_outcome(db_path: Path) -> None:
    _add_requests(
        db_path,
        [
            ("relevance", "recovered", MODEL, "refused", "2026-09-02T00"),
            ("relevance", "recovered", FALLBACK_MODEL, "succeeded", "2026-09-02T01"),
            ("relevance", "for-good", MODEL, "refused", "2026-09-02T00"),
            ("relevance", "for-good", FALLBACK_MODEL, "refused", "2026-09-02T01"),
            ("relevance", "waiting", MODEL, "refused", "2026-09-02T00"),
            ("relevance", "waiting", FALLBACK_MODEL, "pending", None),
        ],
    )
    refusals = list_refusals(db_path, "relevance")
    assert sorted((r.subject, r.model, r.outcome) for r in refusals) == [
        ("for-good", MODEL, SubjectOutcome.REFUSED_FOR_GOOD),
        ("for-good", FALLBACK_MODEL, SubjectOutcome.REFUSED_FOR_GOOD),
        ("recovered", MODEL, SubjectOutcome.RECOVERED),
        ("waiting", MODEL, SubjectOutcome.UNANSWERED),
    ]
    assert outcome_counts(refusals) == {
        SubjectOutcome.RECOVERED: 1,
        SubjectOutcome.REFUSED_FOR_GOOD: 1,
        SubjectOutcome.UNANSWERED: 1,
    }


def _mixed_transport(db_path: Path, batches: FakeBatches, streams: FakeStreams) -> BatchTransport:
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches, stream=streams.stream))
    return BatchTransport(client, db_path, poll_seconds=0)  # type: ignore[arg-type]


def _requests(prefix: str, count: int, model: str) -> list[LlmRequest]:
    return [_request(f"{prefix}-{i}", "x", model) for i in range(count)]


def test_batch_transport_sends_a_small_model_group_immediately(db_path: Path) -> None:
    batches = FakeBatches({"x": {"type": "succeeded", "message": _message("end_turn")}})
    streams = FakeStreams([])
    requests = _requests("big", 12, MODEL) + _requests("small", 3, FALLBACK_MODEL)
    results = asyncio.run(_mixed_transport(db_path, batches, streams).run("relevance", 1, requests))

    assert [len(created) for created in batches.created] == [12]
    assert streams.calls == 3
    assert sorted(r.subject for r in results) == sorted(r.subject for r in requests)
    assert all(r.status == ResultStatus.SUCCEEDED for r in results)
    assert {r.model for r in results if r.batch_id is None} == {FALLBACK_MODEL}
    assert {r.model for r in results if r.batch_id is not None} == {MODEL}
    with sqlite3.connect(db_path) as conn:
        # Only the batched requests are in flight; the immediate ones get a row when recorded.
        assert conn.execute(
            "SELECT model, COUNT(*) FROM llm_requests WHERE status = 'pending' GROUP BY model"
        ).fetchall() == [(MODEL, 12)]


@pytest.mark.parametrize(
    ("count", "batched"), [(MIN_BATCH_REQUESTS - 1, False), (MIN_BATCH_REQUESTS, True)]
)
def test_batch_transport_batches_from_the_minimum(db_path: Path, count: int, batched: bool) -> None:
    batches = FakeBatches({"x": {"type": "succeeded", "message": _message("end_turn")}})
    streams = FakeStreams([])
    results = asyncio.run(
        _mixed_transport(db_path, batches, streams).run(
            "relevance", 1, _requests("doi", count, MODEL)
        )
    )
    assert len(results) == count
    assert len(batches.created) == (1 if batched else 0)
    assert streams.calls == (0 if batched else count)


def _add_answers(
    db_path: Path, rows: list[tuple[str, str, str, str | None, str | None, str]]
) -> None:
    """Requests as (stage, subject, status, stop_reason, rejection, completed_at)."""
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO llm_requests (custom_id, stage, subject, round, model, status,
                                      stop_reason, rejection, completed_at)
            VALUES (?, ?, ?, 2, 'claude-opus-5-5', ?, ?, ?, ?)
            """,
            [(f"r{i}", *row) for i, row in enumerate(rows)],
        )


def test_stage_failures_count_rejected_and_cut_off_answers_since_the_latest_accepted_one(
    db_path: Path,
) -> None:
    assert MAX_FAILED_ANSWERS == 2
    _add_answers(
        db_path,
        [
            # Rejected twice: failed for good, with the latest reason.
            ("extraction", "rejected", "succeeded", "end_turn", "invalid JSON: x", "2026-09-02T01"),
            ("extraction", "rejected", "succeeded", "end_turn", "structural: y", "2026-09-02T02"),
            # Cut off twice before cut-offs had a rejection: only the stop reason says so.
            ("extraction", "cut-off", "succeeded", "max_tokens", None, "2026-09-02T01"),
            ("extraction", "cut-off", "succeeded", "max_tokens", None, "2026-09-02T02"),
            # One of each.
            (
                "extraction",
                "mixed",
                "succeeded",
                "max_tokens",
                MAX_TOKENS_REJECTION,
                "2026-09-02T01",
            ),
            ("extraction", "mixed", "succeeded", "end_turn", "structural: z", "2026-09-02T02"),
            ("extraction", "once", "succeeded", "end_turn", "structural: z", "2026-09-02T01"),
            # Refusals, errors, tool rounds and pending requests are no failed answers.
            ("extraction", "other", "refused", "refusal", None, "2026-09-02T01"),
            ("extraction", "other", "errored", None, None, "2026-09-02T02"),
            ("extraction", "other", "succeeded", "tool_use", None, "2026-09-02T03"),
            ("extraction", "other", "pending", None, None, "2026-09-02T04"),
            # An accepted answer starts the count afresh.
            ("extraction", "rerun", "succeeded", "max_tokens", None, "2026-09-02T01"),
            ("extraction", "rerun", "succeeded", "max_tokens", None, "2026-09-02T02"),
            ("extraction", "rerun", "succeeded", "end_turn", None, "2026-09-02T03"),
            ("extraction", "rerun", "succeeded", "end_turn", "structural: z", "2026-09-02T04"),
            # Another stage's failures count there only.
            ("relevance", "rejected", "succeeded", "end_turn", "invalid JSON: x", "2026-09-02T03"),
        ],
    )
    with sqlite3.connect(db_path) as conn:
        failures = stage_failures(conn, "extraction")
        since = stage_failures(conn, "extraction", datetime(2026, 9, 2, 2, tzinfo=UTC))
        history = stage_history(conn, "extraction")
    assert failures.counts == {"rejected": 2, "cut-off": 2, "mixed": 2, "once": 1, "rerun": 1}
    assert failures.failed_for_good_subjects() == ["cut-off", "mixed", "rejected"]
    assert failures.last_rejection["rejected"] == "structural: y"
    assert failures.last_rejection["cut-off"] == MAX_TOKENS_REJECTION
    assert not failures.failed_for_good("once") and not failures.failed_for_good("other")
    # --retry-failed: failures before the invocation started no longer count.
    assert since.failed_for_good_subjects() == []
    assert history.skipped("rejected") and not history.skipped("once")
    assert history.model_for("rejected") == MODEL


def test_an_answer_cut_off_at_max_tokens_is_rejected_as_such() -> None:
    with pytest.raises(ValueError) as cut_off:
        parse_json_output(Message.model_validate(_message("max_tokens")))
    assert str(cut_off.value) == MAX_TOKENS_REJECTION
    with pytest.raises(ValueError) as tool_call:
        parse_json_output(Message.model_validate(_message("tool_use")))
    assert str(tool_call.value) == "stopped with tool_use"


def test_subjects_failed_for_good_are_listed_with_their_last_rejection(db_path: Path) -> None:
    _add_answers(
        db_path,
        [
            ("extraction", "10.1/a", "succeeded", "max_tokens", None, "2026-09-02T01"),
            ("extraction", "10.1/a", "succeeded", "end_turn", "structural: y", "2026-09-02T02"),
            ("extraction", "10.1/b", "succeeded", "max_tokens", None, "2026-09-02T01"),
            ("assess_genes", "1100", "succeeded", "max_tokens", None, "2026-09-02T01"),
            ("assess_genes", "1100", "succeeded", "max_tokens", None, "2026-09-02T02"),
        ],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO papers (doi, title, source, source_type) "
            "VALUES ('10.1/a', 'A paper', 'pubmed', 'initial')"
        )
    assert list_failed_for_good(db_path) == [
        FailedSubject("assess_genes", "1100", 2, MAX_TOKENS_REJECTION, None),
        FailedSubject("extraction", "10.1/a", 2, "structural: y", "A paper"),
    ]
    assert [f.subject for f in list_failed_for_good(db_path, "extraction")] == ["10.1/a"]
