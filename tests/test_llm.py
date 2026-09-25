"""Offline tests for palit.llm: batching limits, prices, bookkeeping, resume."""

import asyncio
import sqlite3
from collections.abc import AsyncIterator
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
    MID_STREAM_ATTEMPTS,
    BatchTransport,
    ImmediateTransport,
    LlmRequest,
    ResultStatus,
    chunk_for_batches,
    json_output_config,
    record_result,
    request_cost,
)
from palit.llm_usage import list_refusals

SCHEMA_SQL = Path(__file__).resolve().parents[1] / "schema.sql"


def _request(subject: str, text: str = "x") -> LlmRequest:
    return LlmRequest(
        subject=subject,
        params={
            "model": "claude-opus-5-5",
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
    client = SimpleNamespace(messages=SimpleNamespace(batches=batches))
    return BatchTransport(client, db_path, poll_seconds=0)  # type: ignore[arg-type]


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
