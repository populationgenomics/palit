"""Claude API access for the pipeline stages.

Stages build their own ``MessageCreateParamsNonStreaming`` and hand a list of
:class:`LlmRequest` to a :class:`Transport`:

* :class:`BatchTransport` (production) submits Message Batches, persists every
  batch and request in ``llm_batches`` / ``llm_requests`` before polling, and
  re-attaches to uncollected batches on :meth:`BatchTransport.resume`.
* :class:`ImmediateTransport` (prompt development, single-item debugging) sends
  requests concurrently through a fixed pool of workers.

Both return :class:`LlmResult` objects. The stage writes its own output and calls
:func:`record_result` for every result in the same SQLite transaction, so a crash
never leaves a stage output without its request bookkeeping, or the reverse.
"""

import asyncio
import json
import logging
import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

import anthropic
from anthropic import AsyncAnthropic, transform_schema
from anthropic.types import Message, TextBlockParam
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatchIndividualResponse
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.output_config_param import OutputConfigParam
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# Message Batches API limits per batch.
MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 256_000_000
# Headroom for the JSON envelope around each request's params.
_BATCH_BYTES_BUDGET = int(MAX_BATCH_BYTES * 0.95)

DEFAULT_POLL_SECONDS = 60.0
DEFAULT_IMMEDIATE_WORKERS = 50


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AnthropicSettings(BaseSettings):
    """Loaded from ``PALIT_ANTHROPIC_*`` environment variables and ``.env``.

    The profile is palit's own setting rather than ``ANTHROPIC_PROFILE``: Claude
    Code sessions export ``ANTHROPIC_PROFILE`` for their own workspace, and an
    explicit ``profile=`` makes the SDK ignore every credential environment variable.
    """

    model_config = SettingsConfigDict(
        env_prefix="PALIT_ANTHROPIC_", env_file=".env", extra="ignore"
    )

    profile: str = "palit"
    max_retries: int = 8


def make_client(settings: AnthropicSettings) -> AsyncAnthropic:
    """The process's single client, authenticated by an ``ant`` OAuth profile."""
    return AsyncAnthropic(profile=settings.profile, max_retries=settings.max_retries)


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


def _expand_type_unions(node: Any) -> Any:
    """Rewrite ``"type": ["integer", "null"]`` as an equivalent ``anyOf``."""
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


def json_output_config(schema: dict[str, Any], effort: Effort) -> OutputConfigParam:
    """Structured-output config for *schema*, narrowed to the API's grammar subset.

    Constraints the grammar cannot express are moved into field descriptions by
    the SDK; callers still validate responses against the full schema.
    """
    grammar: dict[str, Any] = transform_schema(_expand_type_unions(schema))
    return {"effort": effort, "format": {"type": "json_schema", "schema": grammar}}


def cached_system(text: str) -> list[TextBlockParam]:
    """A system prompt marked as the end of the shared, cacheable prefix."""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def parse_json_output(message: Message) -> Any:
    """The structured output of a finished message."""
    if message.stop_reason != "end_turn":
        raise ValueError(f"message {message.id} stopped with {message.stop_reason}, not end_turn")
    text = next(block.text for block in message.content if block.type == "text")
    return json.loads(text)


# ---------------------------------------------------------------------------
# Requests and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlmRequest:
    """One Messages request about one subject (a DOI, or an HGNC ID as text)."""

    subject: str
    params: MessageCreateParamsNonStreaming


class ResultStatus(StrEnum):
    SUCCEEDED = "succeeded"
    REFUSED = "refused"
    ERRORED = "errored"
    EXPIRED = "expired"
    CANCELED = "canceled"


@dataclass(frozen=True)
class LlmResult:
    custom_id: str
    batch_id: str | None
    stage: str
    subject: str
    round: int
    model: str
    status: ResultStatus
    message: Message | None  # set for SUCCEEDED and REFUSED
    error_type: str | None  # set for ERRORED, e.g. "invalid_request_error"


def _new_custom_id(stage: str, round_no: int) -> str:
    return f"{stage}-{round_no}-{uuid.uuid4().hex[:24]}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _status_for(message: Message) -> ResultStatus:
    return ResultStatus.REFUSED if message.stop_reason == "refusal" else ResultStatus.SUCCEEDED


def record_result(conn: sqlite3.Connection, result: LlmResult) -> None:
    """Store *result*'s outcome and usage in ``llm_requests``.

    Call inside the transaction that writes the stage's output. Every result must
    be recorded, including the ones the stage rejects and retries, because the
    row is also the usage record. Marks the batch collected once no request in it
    is pending.
    """
    usage = result.message.usage if result.message is not None else None
    cache_creation = usage.cache_creation if usage is not None else None
    stop_details = result.message.stop_details if result.message is not None else None
    refusal_category = stop_details.category if stop_details is not None else None
    conn.execute(
        """
        INSERT INTO llm_requests (
            custom_id, batch_id, stage, subject, round, model, status, stop_reason,
            refusal_category, error_type, service_tier, input_tokens, cache_write_5m_tokens,
            cache_write_1h_tokens, cache_read_tokens, output_tokens, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (custom_id) DO UPDATE SET
            status = excluded.status,
            stop_reason = excluded.stop_reason,
            refusal_category = excluded.refusal_category,
            error_type = excluded.error_type,
            service_tier = excluded.service_tier,
            input_tokens = excluded.input_tokens,
            cache_write_5m_tokens = excluded.cache_write_5m_tokens,
            cache_write_1h_tokens = excluded.cache_write_1h_tokens,
            cache_read_tokens = excluded.cache_read_tokens,
            output_tokens = excluded.output_tokens,
            completed_at = excluded.completed_at
        """,
        (
            result.custom_id,
            result.batch_id,
            result.stage,
            result.subject,
            result.round,
            result.model,
            result.status.value,
            result.message.stop_reason if result.message is not None else None,
            refusal_category,
            result.error_type,
            usage.service_tier if usage is not None else None,
            usage.input_tokens if usage is not None else None,
            cache_creation.ephemeral_5m_input_tokens if cache_creation is not None else None,
            cache_creation.ephemeral_1h_input_tokens if cache_creation is not None else None,
            usage.cache_read_input_tokens if usage is not None else None,
            usage.output_tokens if usage is not None else None,
            _now(),
        ),
    )
    if result.batch_id is not None:
        conn.execute(
            """
            UPDATE llm_batches SET collected_at = ?
            WHERE batch_id = ? AND collected_at IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM llm_requests WHERE batch_id = ? AND status = 'pending'
              )
            """,
            (_now(), result.batch_id, result.batch_id),
        )


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class Transport(Protocol):
    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        """Send *requests* and return one result per request, in any order."""
        ...

    async def resume(self, stage: str) -> list[LlmResult]:
        """Results of *stage*'s requests that were sent but never recorded."""
        ...


def chunk_for_batches(
    requests: Sequence[LlmRequest],
    max_requests: int = MAX_BATCH_REQUESTS,
    max_bytes: int = _BATCH_BYTES_BUDGET,
) -> list[list[LlmRequest]]:
    """Split *requests* into batches within the per-batch count and size limits."""
    chunks: list[list[LlmRequest]] = []
    current: list[LlmRequest] = []
    current_bytes = 0
    for request in requests:
        size = len(json.dumps(request.params).encode())
        if size > max_bytes:
            raise ValueError(
                f"request for {request.subject} is {size} bytes, above the batch size limit"
            )
        if current and (len(current) == max_requests or current_bytes + size > max_bytes):
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(request)
        current_bytes += size
    if current:
        chunks.append(current)
    return chunks


class BatchTransport:
    """Message Batches with request bookkeeping in the stage's SQLite database."""

    def __init__(
        self, client: AsyncAnthropic, db_path: Path, poll_seconds: float = DEFAULT_POLL_SECONDS
    ) -> None:
        self._client = client
        self._db_path = db_path
        self._poll_seconds = poll_seconds

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        batch_ids = [
            await self._submit(stage, round_no, chunk) for chunk in chunk_for_batches(requests)
        ]
        results: list[LlmResult] = []
        for batch_id in batch_ids:
            results += await self._collect(batch_id)
        return results

    async def resume(self, stage: str) -> list[LlmResult]:
        with sqlite3.connect(self._db_path) as conn:
            batch_ids = [
                row[0]
                for row in conn.execute(
                    "SELECT batch_id FROM llm_batches WHERE stage = ? AND collected_at IS NULL ORDER BY submitted_at",
                    (stage,),
                )
            ]
        results: list[LlmResult] = []
        for batch_id in batch_ids:
            logger.info("%s: re-attaching to uncollected batch %s", stage, batch_id)
            results += await self._collect(batch_id)
        return results

    async def _submit(self, stage: str, round_no: int, chunk: Sequence[LlmRequest]) -> str:
        custom_ids = [_new_custom_id(stage, round_no) for _ in chunk]
        batch = await self._client.messages.batches.create(
            requests=[
                Request(custom_id=custom_id, params=request.params)
                for custom_id, request in zip(custom_ids, chunk, strict=True)
            ]
        )
        # A crash between create() and this commit leaves a billed batch that no
        # database knows about; the window is one local write.
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO llm_batches (batch_id, stage, round, model, submitted_at) VALUES (?, ?, ?, ?, ?)",
                (batch.id, stage, round_no, chunk[0].params["model"], _now()),
            )
            conn.executemany(
                """
                INSERT INTO llm_requests (custom_id, batch_id, stage, subject, round, model, status)
                VALUES (?, ?, ?, ?, ?, ?, 'pending')
                """,
                [
                    (custom_id, batch.id, stage, request.subject, round_no, request.params["model"])
                    for custom_id, request in zip(custom_ids, chunk, strict=True)
                ],
            )
        logger.info(
            "%s round %d: submitted batch %s with %d requests",
            stage,
            round_no,
            batch.id,
            len(chunk),
        )
        return batch.id

    async def _collect(self, batch_id: str) -> list[LlmResult]:
        while True:
            batch = await self._client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            logger.info(
                "batch %s: %s (processing=%d succeeded=%d errored=%d expired=%d canceled=%d)",
                batch_id,
                batch.processing_status,
                counts.processing,
                counts.succeeded,
                counts.errored,
                counts.expired,
                counts.canceled,
            )
            if batch.processing_status == "ended":
                break
            await asyncio.sleep(self._poll_seconds)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "UPDATE llm_batches SET ended_at = ? WHERE batch_id = ? AND ended_at IS NULL",
                (_now(), batch_id),
            )
            pending = {
                row[0]: (row[1], row[2], row[3], row[4])
                for row in conn.execute(
                    "SELECT custom_id, stage, subject, round, model FROM llm_requests WHERE batch_id = ? AND status = 'pending'",
                    (batch_id,),
                )
            }

        results: list[LlmResult] = []
        async for response in await self._client.messages.batches.results(batch_id):
            row = pending.get(response.custom_id)
            if row is None:
                continue  # recorded before a crash
            stage, subject, round_no, model = row
            results.append(_result_from_batch(response, batch_id, stage, subject, round_no, model))
        return results


def _result_from_batch(
    response: MessageBatchIndividualResponse,
    batch_id: str,
    stage: str,
    subject: str,
    round_no: int,
    model: str,
) -> LlmResult:
    outcome = response.result
    message: Message | None = None
    error_type: str | None = None
    match outcome.type:
        case "succeeded":
            message = outcome.message
            status = _status_for(message)
        case "errored":
            status = ResultStatus.ERRORED
            error_type = outcome.error.error.type
        case "expired":
            status = ResultStatus.EXPIRED
        case "canceled":
            status = ResultStatus.CANCELED
    return LlmResult(
        custom_id=response.custom_id,
        batch_id=batch_id,
        stage=stage,
        subject=subject,
        round=round_no,
        model=model,
        status=status,
        message=message,
        error_type=error_type,
    )


class ImmediateTransport:
    """Concurrent streaming Messages calls through a fixed worker pool.

    Nothing is persisted until the stage records a result, so there is nothing
    to resume. Invalid requests raise; retryable failures that outlast the SDK's
    own retries come back as ERRORED results.
    """

    def __init__(self, client: AsyncAnthropic, workers: int = DEFAULT_IMMEDIATE_WORKERS) -> None:
        self._client = client
        self._workers = workers

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        queue: asyncio.Queue[LlmRequest] = asyncio.Queue()
        for request in requests:
            queue.put_nowait(request)
        results: list[LlmResult] = []

        async def worker() -> None:
            while not queue.empty():
                request = queue.get_nowait()
                results.append(await self._send(stage, round_no, request))

        await asyncio.gather(*(worker() for _ in range(min(self._workers, len(requests)))))
        return results

    async def resume(self, stage: str) -> list[LlmResult]:
        return []

    async def _send(self, stage: str, round_no: int, request: LlmRequest) -> LlmResult:
        custom_id = _new_custom_id(stage, round_no)
        model = request.params["model"]
        try:
            # stream() takes the same keyword arguments minus ``stream`` itself, which
            # the non-streaming params type declares but stages never set.
            params: dict[str, Any] = dict(request.params)
            async with self._client.messages.stream(**params) as stream:
                message = await stream.get_final_message()
        except (
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            anthropic.APIConnectionError,
        ) as e:
            logger.warning(
                "%s: request for %s failed after retries: %s",
                stage,
                request.subject,
                type(e).__name__,
            )
            return LlmResult(
                custom_id=custom_id,
                batch_id=None,
                stage=stage,
                subject=request.subject,
                round=round_no,
                model=model,
                status=ResultStatus.ERRORED,
                message=None,
                error_type=type(e).__name__,
            )
        return LlmResult(
            custom_id=custom_id,
            batch_id=None,
            stage=stage,
            subject=request.subject,
            round=round_no,
            model=model,
            status=_status_for(message),
            message=message,
            error_type=None,
        )


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrices:
    """Standard-tier USD per million tokens."""

    input: float
    output: float
    cache_read_multiplier: float  # of the input price


PRICES: dict[str, ModelPrices] = {
    "claude-opus-5-5": ModelPrices(input=4.0, output=20.0, cache_read_multiplier=0.05),
    "claude-opus-5": ModelPrices(input=5.0, output=25.0, cache_read_multiplier=0.1),
}
CACHE_WRITE_5M_MULTIPLIER = 1.25
CACHE_WRITE_1H_MULTIPLIER = 2.0
BATCH_MULTIPLIER = 0.5


def request_cost(
    model: str,
    service_tier: str,
    input_tokens: int,
    cache_write_5m_tokens: int,
    cache_write_1h_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
) -> float:
    """USD cost of one request's usage."""
    prices = PRICES[model]
    tier = BATCH_MULTIPLIER if service_tier == "batch" else 1.0
    input_equivalent = (
        input_tokens
        + cache_write_5m_tokens * CACHE_WRITE_5M_MULTIPLIER
        + cache_write_1h_tokens * CACHE_WRITE_1H_MULTIPLIER
        + cache_read_tokens * prices.cache_read_multiplier
    )
    return tier * (input_equivalent * prices.input + output_tokens * prices.output) / 1e6
