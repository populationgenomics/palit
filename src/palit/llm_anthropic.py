#!/usr/bin/env python3
"""Anthropic API backend for LLM batch processing.

Two variants share schema handling and response parsing:

* :class:`AnthropicProcessor` (``anthropic/<model-id>``) — one streaming
  Messages call per prompt, fanned out with :func:`asyncio.gather` so a
  ``--batch-size`` of *n* means *n* concurrent requests.
* :class:`AnthropicBatchProcessor` (``anthropic-batch/<model-id>``) — one
  Message Batches submission per ``process_batch`` call, polled to completion.

Both read ``ANTHROPIC_API_KEY`` from the environment or ``.env``.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

import jsonschema
from anthropic import AnthropicError, AsyncAnthropic, transform_schema
from anthropic.types import Message
from anthropic.types.json_output_format_param import JSONOutputFormatParam
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.output_config_param import OutputConfigParam
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from palit.llm import PromptResult

logger = logging.getLogger(__name__)

# The SDK honours ``retry-after`` on 429s, so a generous ceiling is the whole
# rate-limit strategy for the concurrent path.
DEFAULT_MAX_RETRIES = 8

# Streaming keeps a long generation off the request-duration limit, but the
# client-level timeout still spans the whole stream.
REQUEST_TIMEOUT_SECONDS = 1800.0

Effort = Literal["low", "medium", "high", "max"]
DEFAULT_EFFORT: Effort = "medium"

BATCH_POLL_INTERVAL_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Settings — pydantic-settings, .env is read automatically.
# ---------------------------------------------------------------------------


class AnthropicSettings(BaseSettings):
    """Loaded from environment + .env when a processor is constructed."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str = Field(..., alias="ANTHROPIC_API_KEY")


def load_api_key() -> str:
    """Resolve the Anthropic API key, failing with an actionable message."""
    try:
        return AnthropicSettings().api_key  # type: ignore[call-arg]
    except ValidationError as e:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it or add it to .env "
            "before using an anthropic/ or anthropic-batch/ model."
        ) from e


# ---------------------------------------------------------------------------
# Schema handling.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SanitizedSchema:
    """A schema rewritten for the structured-output grammar.

    ``unenforced`` lists the JSON pointers whose constraints the grammar cannot
    express. They are not lost silently: the SDK folds them into the enclosing
    field description, and every response is still validated against the full
    schema with ``jsonschema``.
    """

    grammar: dict[str, Any]
    unenforced: list[str]


def _expand_type_unions(node: Any) -> Any:
    """Rewrite ``"type": ["integer", "null"]`` as an equivalent ``anyOf``.

    The structured-output grammar takes a single type per subschema; our
    schemas use the list form for nullable fields.
    """
    if isinstance(node, list):
        return [_expand_type_unions(item) for item in node]
    if not isinstance(node, dict):
        return node

    expanded = {key: _expand_type_unions(value) for key, value in node.items()}
    types = expanded.get("type")
    if not isinstance(types, list):
        return expanded

    siblings = {key: value for key, value in expanded.items() if key != "type"}
    return {"anyOf": [{"type": one, **siblings} for one in types]}


def _schema_paths(node: Any, path: str = "") -> set[str]:
    """Every JSON pointer in a schema, used to report what sanitizing removed."""
    paths = set()
    if isinstance(node, dict):
        for key, value in node.items():
            paths.add(f"{path}/{key}")
            paths |= _schema_paths(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            paths |= _schema_paths(value, f"{path}/{index}")
    return paths


def sanitize_schema(schema: dict[str, Any]) -> SanitizedSchema:
    """Rewrite *schema* into the keyword subset the API's grammar accepts.

    The canonical schema files keep the full grammar for the vLLM path; this
    narrowing happens at request time and only for the Anthropic backends.
    """
    expanded = _expand_type_unions(schema)
    grammar: dict[str, Any] = transform_schema(expanded)
    unenforced = sorted(_schema_paths(expanded) - _schema_paths(grammar))
    return SanitizedSchema(grammar=grammar, unenforced=unenforced)


class _SchemaCache:
    """Sanitizes each distinct schema once, and reports the loss once."""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}

    def grammar_for(self, schema: dict[str, Any]) -> dict[str, Any]:
        key = json.dumps(schema, sort_keys=True)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        sanitized = sanitize_schema(schema)
        if sanitized.unenforced:
            logger.info(
                "Sanitized schema for the structured-output grammar; %d constraint(s) "
                "are no longer enforced by the API (kept in the field description, and "
                "re-checked with jsonschema after the call): %s",
                len(sanitized.unenforced),
                ", ".join(sanitized.unenforced),
            )
        self._cache[key] = sanitized.grammar
        return sanitized.grammar


# ---------------------------------------------------------------------------
# Shared request construction and response parsing.
# ---------------------------------------------------------------------------


def _output_config(effort: Effort, grammar: dict[str, Any]) -> OutputConfigParam:
    return OutputConfigParam(
        effort=effort,
        format=JSONOutputFormatParam(type="json_schema", schema=grammar),
    )


def _require_sampling_temperature_one(temperature: float) -> None:
    """Adaptive thinking pins temperature to 1; reject other values up front."""
    if temperature != 1.0:
        raise ValueError(
            f"The Anthropic backend runs with adaptive thinking, which only accepts "
            f"--temperature 1.0 (got {temperature})."
        )


def _parse_message(message: Message, schema: dict[str, Any]) -> PromptResult | None:
    """Turn a finished message into a validated :class:`PromptResult`."""
    logger.info(
        "Token usage: input=%d output=%d",
        message.usage.input_tokens,
        message.usage.output_tokens,
    )

    if message.stop_reason != "end_turn":
        logger.warning(
            "Response stopped with %s (%s); discarding",
            message.stop_reason,
            message.stop_details,
        )
        return None

    # Adaptive thinking can emit several thinking+text pairs; the structured
    # output is the final text block.
    text_blocks = [block for block in message.content if block.type == "text"]
    if not text_blocks:
        logger.warning("No text block in response")
        return None

    try:
        parsed_json = json.loads(text_blocks[-1].text)
        jsonschema.validate(parsed_json, schema)
    except (json.JSONDecodeError, jsonschema.ValidationError):
        logger.exception("Invalid JSON output")
        return None

    return PromptResult(raw_response=message.to_json(), parsed_json=parsed_json)


class AnthropicProcessor:
    """Concurrent streaming Messages API calls, one per prompt."""

    def __init__(
        self,
        model_id: str,
        temperature: float,
        max_tokens: int,
        effort: Effort = DEFAULT_EFFORT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        _require_sampling_temperature_one(temperature)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.effort = effort
        self._schemas = _SchemaCache()
        self._client = AsyncAnthropic(
            api_key=load_api_key(),
            max_retries=max_retries,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    async def process_batch(
        self, prompts: list[str], schema: dict[str, Any]
    ) -> list[PromptResult | None]:
        if not prompts:
            return []

        grammar = self._schemas.grammar_for(schema)
        tasks = [self._process_single(prompt, schema, grammar) for prompt in prompts]
        return list(await asyncio.gather(*tasks))

    async def _process_single(
        self, prompt: str, schema: dict[str, Any], grammar: dict[str, Any]
    ) -> PromptResult | None:
        try:
            async with self._client.messages.stream(
                model=self.model_id,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
                thinking={"type": "adaptive"},
                output_config=_output_config(self.effort, grammar),
            ) as stream:
                message = await stream.get_final_message()
        except AnthropicError:
            logger.exception("Failed to process prompt")
            return None

        return _parse_message(message, schema)


class AnthropicBatchProcessor:
    """Message Batches API: one submission per ``process_batch`` call.

    Batched requests are half price but asynchronous, so give the stage a
    ``--batch-size`` that covers the whole run rather than the concurrency a
    real-time backend wants.
    """

    def __init__(
        self,
        model_id: str,
        temperature: float,
        max_tokens: int,
        effort: Effort = DEFAULT_EFFORT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        poll_interval: float = BATCH_POLL_INTERVAL_SECONDS,
    ):
        _require_sampling_temperature_one(temperature)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.effort = effort
        self.poll_interval = poll_interval
        self._schemas = _SchemaCache()
        self._client = AsyncAnthropic(
            api_key=load_api_key(),
            max_retries=max_retries,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    async def process_batch(
        self, prompts: list[str], schema: dict[str, Any]
    ) -> list[PromptResult | None]:
        if not prompts:
            return []

        grammar = self._schemas.grammar_for(schema)
        custom_ids = [f"prompt-{index}" for index in range(len(prompts))]
        requests = [
            Request(custom_id=custom_id, params=self._params(prompt, grammar))
            for custom_id, prompt in zip(custom_ids, prompts, strict=True)
        ]

        batch = await self._client.messages.batches.create(requests=requests)
        logger.info("Submitted batch %s with %d requests", batch.id, len(prompts))

        while batch.processing_status != "ended":
            await asyncio.sleep(self.poll_interval)
            batch = await self._client.messages.batches.retrieve(batch.id)
            counts = batch.request_counts
            logger.info(
                "Batch %s: %s (processing=%d succeeded=%d errored=%d canceled=%d expired=%d)",
                batch.id,
                batch.processing_status,
                counts.processing,
                counts.succeeded,
                counts.errored,
                counts.canceled,
                counts.expired,
            )

        results_by_id = await self._collect(batch.id, schema)
        missing = [custom_id for custom_id in custom_ids if custom_id not in results_by_id]
        if missing:
            logger.warning("Batch %s returned no result for %d request(s)", batch.id, len(missing))

        return [results_by_id.get(custom_id) for custom_id in custom_ids]

    def _params(self, prompt: str, grammar: dict[str, Any]) -> MessageCreateParamsNonStreaming:
        return MessageCreateParamsNonStreaming(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            thinking={"type": "adaptive"},
            output_config=_output_config(self.effort, grammar),
        )

    async def _collect(
        self, batch_id: str, schema: dict[str, Any]
    ) -> dict[str, PromptResult | None]:
        results: dict[str, PromptResult | None] = {}
        async for entry in await self._client.messages.batches.results(batch_id):
            if entry.result.type == "succeeded":
                results[entry.custom_id] = _parse_message(entry.result.message, schema)
            else:
                logger.warning(
                    "Request %s in batch %s: %s", entry.custom_id, batch_id, entry.result.type
                )
                results[entry.custom_id] = None
        return results
