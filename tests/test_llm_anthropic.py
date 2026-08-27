"""Tests for the Anthropic backends: prefix routing, schema sanitizing, result mapping.

Every test runs against a stubbed SDK client — nothing here touches the network.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from palit import llm_anthropic
from palit.llm import PromptResult, create_llm_processor
from palit.llm_anthropic import (
    AnthropicBatchProcessor,
    AnthropicProcessor,
    sanitize_schema,
)

# --- helpers ----------------------------------------------------------------

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mondo_id": {"type": "string", "pattern": "^MONDO:[0-9]{7}$"},
        "family_count": {"type": ["integer", "null"], "description": "families"},
        "moi": {"type": "string", "enum": ["Monoallelic", "Biallelic"]},
    },
    "required": ["mondo_id", "family_count", "moi"],
    "additionalProperties": False,
}

PAYLOAD: dict[str, Any] = {
    "mondo_id": "MONDO:0013212",
    "family_count": 4,
    "moi": "Monoallelic",
}


def _message(payload: dict[str, Any], stop_reason: str = "end_turn") -> Any:
    """A stand-in for anthropic.types.Message carrying a single text block."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=11, output_tokens=22),
        to_json=lambda: json.dumps({"stop_reason": stop_reason}),
    )


class _StubStream:
    def __init__(self, outcome: Any):
        self._outcome = outcome

    async def __aenter__(self) -> "_StubStream":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def get_final_message(self) -> Any:
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


class _StubMessages:
    """Replays one outcome per prompt and records the requests it saw."""

    def __init__(self, outcomes: list[Any]):
        self._outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _StubStream:
        self.requests.append(kwargs)
        return _StubStream(self._outcomes.pop(0))


def _processor(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> AnthropicProcessor:
    monkeypatch.setattr(llm_anthropic, "load_api_key", lambda: "test-key")
    processor = AnthropicProcessor(model_id="claude-sonnet-5", temperature=1.0, max_tokens=4096)
    messages = _StubMessages(outcomes)
    processor._client = SimpleNamespace(messages=messages)  # type: ignore[assignment]
    return processor


# --- schema sanitizing ------------------------------------------------------


def test_type_unions_become_any_of() -> None:
    grammar = sanitize_schema(SCHEMA).grammar
    assert grammar["properties"]["family_count"]["anyOf"] == [
        {"type": "integer", "description": "families"},
        {"type": "null", "description": "families"},
    ]


def test_pattern_is_reported_and_kept_in_the_description() -> None:
    sanitized = sanitize_schema(SCHEMA)
    assert sanitized.unenforced == ["/properties/mondo_id/pattern"]
    assert "^MONDO:[0-9]{7}$" in sanitized.grammar["properties"]["mondo_id"]["description"]


def test_enum_and_required_survive() -> None:
    grammar = sanitize_schema(SCHEMA).grammar
    assert grammar["properties"]["moi"]["enum"] == ["Monoallelic", "Biallelic"]
    assert grammar["required"] == ["mondo_id", "family_count", "moi"]
    assert grammar["additionalProperties"] is False


def test_the_canonical_schema_is_not_mutated() -> None:
    before = json.dumps(SCHEMA, sort_keys=True)
    sanitize_schema(SCHEMA)
    assert json.dumps(SCHEMA, sort_keys=True) == before


# --- prefix routing ---------------------------------------------------------


def test_anthropic_prefix_selects_the_concurrent_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_anthropic, "load_api_key", lambda: "test-key")
    processor = create_llm_processor(
        model="anthropic/claude-sonnet-5", temperature=1.0, max_tokens=4096
    )
    assert isinstance(processor, AnthropicProcessor)
    assert processor.model_id == "claude-sonnet-5"


def test_anthropic_batch_prefix_selects_the_batch_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_anthropic, "load_api_key", lambda: "test-key")
    processor = create_llm_processor(
        model="anthropic-batch/claude-sonnet-5", temperature=1.0, max_tokens=4096
    )
    assert isinstance(processor, AnthropicBatchProcessor)
    assert processor.model_id == "claude-sonnet-5"


def test_llm_config_extras_reach_the_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_anthropic, "load_api_key", lambda: "test-key")
    processor = create_llm_processor(
        model="anthropic/claude-sonnet-5", temperature=1.0, max_tokens=4096, effort="high"
    )
    assert isinstance(processor, AnthropicProcessor)
    assert processor.effort == "high"


def test_non_unit_temperature_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm_anthropic, "load_api_key", lambda: "test-key")
    with pytest.raises(ValueError, match="adaptive thinking"):
        create_llm_processor(model="anthropic/claude-sonnet-5", temperature=0.2, max_tokens=4096)


# --- concurrent processor ---------------------------------------------------


def test_empty_batch_makes_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor(monkeypatch, [])
    assert asyncio.run(processor.process_batch([], SCHEMA)) == []


def test_each_prompt_yields_a_validated_result(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor(monkeypatch, [_message(PAYLOAD), _message(PAYLOAD)])
    results = asyncio.run(processor.process_batch(["a", "b"], SCHEMA))
    assert [r.parsed_json for r in results if isinstance(r, PromptResult)] == [PAYLOAD, PAYLOAD]


def test_the_request_carries_the_sanitized_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor(monkeypatch, [_message(PAYLOAD)])
    asyncio.run(processor.process_batch(["a"], SCHEMA))
    request = processor._client.messages.requests[0]  # type: ignore[attr-defined]
    assert request["output_config"]["format"]["schema"] == sanitize_schema(SCHEMA).grammar
    assert request["thinking"] == {"type": "adaptive"}
    assert request["temperature"] == 1.0
    assert request["max_tokens"] == 4096


def test_an_api_error_fails_only_its_own_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor(
        monkeypatch,
        [_message(PAYLOAD), llm_anthropic.AnthropicError("boom"), _message(PAYLOAD)],
    )
    results = asyncio.run(processor.process_batch(["a", "b", "c"], SCHEMA))
    assert [r is None for r in results] == [False, True, False]


def test_a_schema_violation_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor(monkeypatch, [_message({**PAYLOAD, "moi": "Trialleleic"})])
    assert asyncio.run(processor.process_batch(["a"], SCHEMA)) == [None]


def test_a_truncated_response_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    processor = _processor(monkeypatch, [_message(PAYLOAD, stop_reason="max_tokens")])
    assert asyncio.run(processor.process_batch(["a"], SCHEMA)) == [None]


# --- batch processor --------------------------------------------------------


class _StubBatches:
    """Ends immediately and returns results in an order that is not the input order."""

    def __init__(self, entries: list[Any]):
        self._entries = entries
        self.submitted: list[Any] = []

    async def create(self, *, requests: list[Any]) -> Any:
        self.submitted = requests
        return SimpleNamespace(id="batch-1", processing_status="ended")

    async def results(self, batch_id: str) -> Any:
        async def iterator() -> Any:
            for entry in self._entries:
                yield entry

        return iterator()


def _batch_entry(custom_id: str, payload: dict[str, Any]) -> Any:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="succeeded", message=_message(payload)),
    )


def _batch_processor(
    monkeypatch: pytest.MonkeyPatch, entries: list[Any]
) -> tuple[AnthropicBatchProcessor, _StubBatches]:
    monkeypatch.setattr(llm_anthropic, "load_api_key", lambda: "test-key")
    processor = AnthropicBatchProcessor(
        model_id="claude-sonnet-5", temperature=1.0, max_tokens=4096
    )
    batches = _StubBatches(entries)
    processor._client = SimpleNamespace(messages=SimpleNamespace(batches=batches))  # type: ignore[assignment]
    return processor, batches


def test_batch_results_are_reordered_by_custom_id(monkeypatch: pytest.MonkeyPatch) -> None:
    first = {**PAYLOAD, "family_count": 1}
    second = {**PAYLOAD, "family_count": 2}
    third = {**PAYLOAD, "family_count": 3}
    processor, batches = _batch_processor(
        monkeypatch,
        [
            _batch_entry("prompt-2", third),
            _batch_entry("prompt-0", first),
            _batch_entry("prompt-1", second),
        ],
    )

    results = asyncio.run(processor.process_batch(["a", "b", "c"], SCHEMA))

    assert [r.parsed_json for r in results if isinstance(r, PromptResult)] == [first, second, third]
    assert [request["custom_id"] for request in batches.submitted] == [
        "prompt-0",
        "prompt-1",
        "prompt-2",
    ]


def test_batch_requests_carry_the_prompt_and_grammar(monkeypatch: pytest.MonkeyPatch) -> None:
    processor, batches = _batch_processor(monkeypatch, [_batch_entry("prompt-0", PAYLOAD)])
    asyncio.run(processor.process_batch(["the prompt"], SCHEMA))
    params = batches.submitted[0]["params"]
    assert params["messages"] == [{"role": "user", "content": "the prompt"}]
    assert params["output_config"]["format"]["schema"] == sanitize_schema(SCHEMA).grammar
    assert params["model"] == "claude-sonnet-5"


def test_a_failed_batch_request_yields_none_for_that_custom_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errored = SimpleNamespace(custom_id="prompt-0", result=SimpleNamespace(type="errored"))
    processor, _ = _batch_processor(monkeypatch, [errored, _batch_entry("prompt-1", PAYLOAD)])
    results = asyncio.run(processor.process_batch(["a", "b"], SCHEMA))
    assert results[0] is None
    assert results[1] is not None


def test_a_missing_batch_result_yields_none(monkeypatch: pytest.MonkeyPatch) -> None:
    processor, _ = _batch_processor(monkeypatch, [_batch_entry("prompt-0", PAYLOAD)])
    results = asyncio.run(processor.process_batch(["a", "b"], SCHEMA))
    assert results[1] is None
