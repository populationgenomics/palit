"""Claude API access for the pipeline stages.

Stages build their own ``MessageCreateParamsNonStreaming`` and hand a list of
:class:`LlmRequest` to a :class:`Transport`:

* :class:`BatchTransport` (production) submits Message Batches, persists every
  batch and request in ``llm_batches`` / ``llm_requests`` before polling, and
  re-attaches to uncollected batches on :meth:`BatchTransport.resume`. A model's
  requests go out immediately instead when there are fewer than
  :data:`MIN_BATCH_REQUESTS` of them.
* :class:`ImmediateTransport` (prompt development, single-item debugging) sends
  requests concurrently through a fixed pool of workers.

Both return :class:`LlmResult` objects. The stage writes its own output and calls
:func:`record_result` for every result in the same SQLite transaction, so a crash
never leaves a stage output without its request bookkeeping, or the reverse.

Refusals fall back from :data:`MODEL` to :data:`FALLBACK_MODEL`. Stages pick the
model for each request with :func:`stage_refusals`: a subject goes to MODEL
until MODEL refuses it in that stage, and then to FALLBACK_MODEL with otherwise
identical params. A subject is refused for good once FALLBACK_MODEL refuses it.

A stage also gives up on a subject whose answers keep failing. Once the stage
has rejected :data:`MAX_FAILED_ANSWERS` answers about it, or seen them cut off
at max_tokens, since its latest accepted answer, the subject is failed for good
(see :func:`stage_failures`). Both kinds of skipping count across invocations.
"""

import asyncio
import json
import logging
import re
import sqlite3
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

import anthropic
import jsonschema
import tenacity
from anthropic import AsyncAnthropic, transform_schema
from anthropic.types import Message, TextBlockParam
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatchIndividualResponse
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.output_config_param import OutputConfigParam
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
# Gets every request about a subject once MODEL has refused the subject in that stage.
FALLBACK_MODEL = "claude-sonnet-5-5"
# For the report's notes on results that came from FALLBACK_MODEL.
MODEL_NAME = "Opus 5.5"
FALLBACK_MODEL_NAME = "Sonnet 5.5"
# Input tokens plus max_tokens of a request must fit this window, per model.
CONTEXT_WINDOW: dict[str, int] = {MODEL: 1_000_000, FALLBACK_MODEL: 1_000_000}

Effort = Literal["low", "medium", "high", "xhigh", "max"]

# Message Batches API limits per batch.
MAX_BATCH_REQUESTS = 100_000
MAX_BATCH_BYTES = 256_000_000
# Headroom for the JSON envelope around each request's params.
_BATCH_BYTES_BUDGET = int(MAX_BATCH_BYTES * 0.95)
# Fewer requests than this for one model are sent immediately: a batch can take
# hours however few requests it holds, and halving the price of a few saves little.
MIN_BATCH_REQUESTS = 10

DEFAULT_POLL_SECONDS = 60.0
DEFAULT_IMMEDIATE_WORKERS = 50

# Transient failures the SDK has already retried max_retries times when it raises them.
_TRANSIENT_ERRORS = (
    anthropic.RateLimitError,
    anthropic.OverloadedError,
    anthropic.ServiceUnavailableError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)
# An error event inside a started stream carries the stream's 200 status, so
# the SDK raises a plain APIStatusError and does not retry it.
_MID_STREAM_TRANSIENT_TYPES = frozenset({"api_error", "overloaded_error"})
MID_STREAM_ATTEMPTS = 5
# A subject is failed for good once its stage recorded this many failed answers.
MAX_FAILED_ANSWERS = 2
# The rejection of an answer cut off at max_tokens.
MAX_TOKENS_REJECTION = "output cut off at max_tokens"
# The API's message for a request whose prompt exceeds the model's context window.
_PROMPT_TOO_LONG = re.compile(r"prompt is too long: (\d+) tokens > \d+ maximum")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AnthropicSettings(BaseSettings):
    """Loaded from ``PALIT_ANTHROPIC_*`` environment variables and ``.env``.

    The profile is palit's own setting rather than ``ANTHROPIC_PROFILE``: Claude
    Code sessions export ``ANTHROPIC_PROFILE`` for their own workspace, and an
    explicit ``profile=`` makes the SDK ignore every credential environment variable.
    It has no default, and its alias is the full variable name so that a missing
    variable's validation error names it.
    """

    model_config = SettingsConfigDict(
        env_prefix="PALIT_ANTHROPIC_", env_file=".env", extra="ignore"
    )

    profile: str = Field(alias="PALIT_ANTHROPIC_PROFILE")
    max_retries: int = 8


def make_client() -> AsyncAnthropic:
    """The process's single client, authenticated by an ``ant`` OAuth profile.

    Raises ``pydantic.ValidationError`` when ``PALIT_ANTHROPIC_PROFILE`` is unset.
    """
    settings = AnthropicSettings()  # type: ignore[call-arg]
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


def assistant_content(message: Message) -> list[dict[str, Any]]:
    """The assistant turn exactly as returned, for replay in the next round.

    Only top-level ``None`` fields are dropped; nested values (tool inputs,
    thinking signatures) are kept verbatim.
    """
    return [
        {key: value for key, value in block.model_dump(mode="json").items() if value is not None}
        for block in message.content
    ]


def unfinished_answer_reason(stop_reason: str | None) -> str:
    """The rejection of an answer that stopped before it ended its turn."""
    return MAX_TOKENS_REJECTION if stop_reason == "max_tokens" else f"stopped with {stop_reason}"


def parse_json_output(message: Message) -> Any:
    """The structured output of a finished message.

    Raises ValueError, with :func:`unfinished_answer_reason` as its message, for
    a message that did not end its turn.
    """
    if message.stop_reason != "end_turn":
        raise ValueError(unfinished_answer_reason(message.stop_reason))
    text = next(block.text for block in message.content if block.type == "text")
    return json.loads(text)


def invalid_answer_reason(error: ValueError | jsonschema.ValidationError) -> str:
    """A short reason for rejecting an answer that isn't JSON or violates the schema.

    Stages also raise ValueError for answers that break their own rules; its
    message is the reason.
    """
    if isinstance(error, jsonschema.ValidationError):
        return f"schema violation at {error.json_path}: {error.message[:200]}"
    if isinstance(error, json.JSONDecodeError):
        return f"invalid JSON: {error}"
    return str(error)


async def count_input_tokens(
    client: AsyncAnthropic, params: MessageCreateParamsNonStreaming
) -> int:
    """The input tokens of the request *params*, from the Messages count_tokens endpoint.

    The endpoint takes the request's params except ``max_tokens``. Counting is
    free and sends nothing to the model. It does not accept Files API sources.
    """
    counted: dict[str, Any] = dict(params)
    del counted["max_tokens"]
    return (await client.messages.count_tokens(**counted)).input_tokens


# ---------------------------------------------------------------------------
# Requests and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LlmRequest:
    """One Messages request about one subject (a DOI, an HGNC ID or an association id, as text)."""

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
    error_message: str | None  # set for ERRORED; not stored in llm_requests

    @property
    def goes_to_fallback(self) -> bool:
        """Whether MODEL refused it, so the subject's next request goes to FALLBACK_MODEL."""
        return self.status == ResultStatus.REFUSED and self.model == MODEL

    @property
    def refused_for_good(self) -> bool:
        """Whether FALLBACK_MODEL refused it, so the stage skips the subject from now on."""
        return self.status == ResultStatus.REFUSED and self.model == FALLBACK_MODEL

    @property
    def refusal_category(self) -> str | None:
        """The safety classifier's category of a refusal, if the message names one."""
        stop_details = self.message.stop_details if self.message is not None else None
        return stop_details.category if stop_details is not None else None

    @property
    def too_long_prompt_tokens(self) -> int | None:
        """The prompt's token count, if the API rejected the request as too long for the model."""
        if self.error_type != "invalid_request_error" or self.error_message is None:
            return None
        match = _PROMPT_TOO_LONG.match(self.error_message)
        return int(match.group(1)) if match is not None else None


def log_refusal(result: LlmResult, label: str) -> None:
    """Warn about a refused *result*; *label* names its subject, e.g. ``"HGNC:1100"``."""
    logger.warning(
        "%s: %s refused %s (%s); %s",
        result.stage,
        result.model,
        label,
        result.refusal_category or "no category",
        f"the next attempt goes to {FALLBACK_MODEL}"
        if result.goes_to_fallback
        else "refused for good",
    )


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


# Refusals and failed answers recorded at or after this instant count: by
# default, all of them.
ALL_TIME = datetime.min.replace(tzinfo=UTC)


@dataclass(frozen=True)
class StageRefusals:
    """The subjects of one stage that MODEL and FALLBACK_MODEL refused.

    Refusals by any other model, such as one an earlier run used, do not count.
    """

    by_model: frozenset[str]
    by_fallback: frozenset[str]

    def refused_for_good(self, subject: str) -> bool:
        """Whether FALLBACK_MODEL refused *subject*, so the stage skips it."""
        return subject in self.by_fallback

    def model_for(self, subject: str) -> str:
        """The model of the stage's next request about *subject*.

        FALLBACK_MODEL once MODEL refused the subject, MODEL before. Raises
        ValueError for a subject refused for good: stages leave those out.
        """
        if subject in self.by_fallback:
            raise ValueError(f"{subject} was refused by {FALLBACK_MODEL} too")
        return FALLBACK_MODEL if subject in self.by_model else MODEL


def stage_refusals(
    conn: sqlite3.Connection, stage: str, since: datetime = ALL_TIME
) -> StageRefusals:
    """The refusals *stage* recorded at or after *since*, by model."""
    refused: dict[str, set[str]] = {MODEL: set(), FALLBACK_MODEL: set()}
    for model, subject in conn.execute(
        """
        SELECT model, subject FROM llm_requests
        WHERE stage = ? AND status = 'refused' AND model IN (?, ?) AND completed_at >= ?
        """,
        (stage, MODEL, FALLBACK_MODEL, since.isoformat()),
    ):
        refused[model].add(subject)
    return StageRefusals(
        by_model=frozenset(refused[MODEL]), by_fallback=frozenset(refused[FALLBACK_MODEL])
    )


# ---------------------------------------------------------------------------
# Failed answers
# ---------------------------------------------------------------------------


# An llm_requests row of an answer the stage did not use: rejected, or cut off at
# max_tokens. Rows recorded before cut-offs had a rejection have only the stop reason.
# Unqualified, so that it applies to the innermost llm_requests of a query.
FAILED_ANSWER = "status = 'succeeded' AND (rejection IS NOT NULL OR stop_reason = 'max_tokens')"
# An llm_requests row of an answer the stage stored.
ACCEPTED_ANSWER = "status = 'succeeded' AND stop_reason = 'end_turn' AND rejection IS NULL"


@dataclass(frozen=True)
class StageFailures:
    """The failed answers of one stage's subjects (see :func:`stage_failures`)."""

    counts: dict[str, int]
    last_rejection: dict[str, str]

    def failed_for_good(self, subject: str) -> bool:
        """Whether *subject* has MAX_FAILED_ANSWERS failed answers, so the stage skips it."""
        return self.counts.get(subject, 0) >= MAX_FAILED_ANSWERS

    def failed_for_good_subjects(self) -> list[str]:
        return sorted(subject for subject in self.counts if self.failed_for_good(subject))


def stage_failures(
    conn: sqlite3.Connection, stage: str, since: datetime = ALL_TIME
) -> StageFailures:
    """The failed answers *stage* recorded at or after *since*, per subject.

    A failed answer was rejected by the stage or cut off at max_tokens. Only the
    failed answers after the subject's latest accepted answer count, so a
    subject whose output is removed for a rerun starts afresh.
    """
    counts: dict[str, int] = defaultdict(int)
    last_rejection: dict[str, str] = {}
    for subject, rejection in conn.execute(
        f"""
        SELECT subject, COALESCE(rejection, ?) FROM llm_requests r
        WHERE stage = ? AND {FAILED_ANSWER} AND completed_at >= ?
          AND NOT EXISTS (
              SELECT 1 FROM llm_requests a
              WHERE a.stage = r.stage AND a.subject = r.subject AND {ACCEPTED_ANSWER}
                AND a.completed_at > r.completed_at
          )
        ORDER BY completed_at
        """,
        (MAX_TOKENS_REJECTION, stage, since.isoformat()),
    ):
        counts[subject] += 1
        last_rejection[subject] = rejection
    return StageFailures(counts=dict(counts), last_rejection=last_rejection)


def log_failed_for_good(failures: StageFailures, stage: str) -> None:
    """Warn about each subject *stage* leaves out because its answers failed for good."""
    for subject in failures.failed_for_good_subjects():
        logger.warning(
            "%s: leaving out %s, failed for good after %d failed answers (last: %s); "
            "see --retry-failed",
            stage,
            subject,
            failures.counts[subject],
            failures.last_rejection[subject],
        )


@dataclass(frozen=True)
class StageHistory:
    """The refusals and failed answers of one stage, which decide whether and how to send a subject."""

    refusals: StageRefusals
    failures: StageFailures

    def skipped(self, subject: str) -> bool:
        """Whether the stage leaves *subject* out: refused or failed for good."""
        return self.refusals.refused_for_good(subject) or self.failures.failed_for_good(subject)

    def model_for(self, subject: str) -> str:
        """The model of the stage's next request about *subject* (see :meth:`StageRefusals.model_for`)."""
        return self.refusals.model_for(subject)


def stage_history(
    conn: sqlite3.Connection,
    stage: str,
    refusals_since: datetime = ALL_TIME,
    failures_since: datetime = ALL_TIME,
) -> StageHistory:
    """The refusals and failed answers *stage* recorded since the given instants."""
    return StageHistory(
        refusals=stage_refusals(conn, stage, refusals_since),
        failures=stage_failures(conn, stage, failures_since),
    )


def _new_custom_id(stage: str, round_no: int) -> str:
    return f"{stage}-{round_no}-{uuid.uuid4().hex[:24]}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _status_for(message: Message) -> ResultStatus:
    return ResultStatus.REFUSED if message.stop_reason == "refusal" else ResultStatus.SUCCEEDED


def record_result(
    conn: sqlite3.Connection, result: LlmResult, *, rejection: str | None = None
) -> None:
    """Store *result*'s outcome and usage in ``llm_requests``.

    Call inside the transaction that writes the stage's output. Every result must
    be recorded, including the ones the stage rejects and retries, because the
    row is also the usage record; *rejection* is the reason the stage rejected
    the answer. Marks the batch collected once no request in it is pending.
    """
    usage = result.message.usage if result.message is not None else None
    cache_creation = usage.cache_creation if usage is not None else None
    conn.execute(
        """
        INSERT INTO llm_requests (
            custom_id, batch_id, stage, subject, round, model, status, stop_reason,
            refusal_category, error_type, service_tier, input_tokens, cache_write_5m_tokens,
            cache_write_1h_tokens, cache_read_tokens, output_tokens, rejection, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            rejection = excluded.rejection,
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
            result.refusal_category,
            result.error_type,
            usage.service_tier if usage is not None else None,
            usage.input_tokens if usage is not None else None,
            cache_creation.ephemeral_5m_input_tokens if cache_creation is not None else None,
            cache_creation.ephemeral_1h_input_tokens if cache_creation is not None else None,
            usage.cache_read_input_tokens if usage is not None else None,
            usage.output_tokens if usage is not None else None,
            rejection,
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
    """Message Batches with request bookkeeping in the stage's SQLite database.

    A model's requests go through an :class:`ImmediateTransport` instead when
    there are fewer than *min_batch_requests* of them. Like any immediate
    request, they get no ``llm_requests`` row until the stage records the
    result, so an interrupted run leaves their subjects due again rather than
    pending, and :meth:`resume` never returns them.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        db_path: Path,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        min_batch_requests: int = MIN_BATCH_REQUESTS,
    ) -> None:
        self._client = client
        self._db_path = db_path
        self._poll_seconds = poll_seconds
        self._min_batch_requests = min_batch_requests
        self._immediate = ImmediateTransport(client)

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        """Send *requests* in batches of one model each, as ``llm_batches`` records it.

        A model with fewer than *min_batch_requests* requests has them sent
        immediately, while the batches run.
        """
        by_model: dict[str, list[LlmRequest]] = defaultdict(list)
        for request in requests:
            by_model[request.params["model"]].append(request)
        immediate: list[LlmRequest] = []
        batch_ids: list[str] = []
        for model, group in by_model.items():
            if len(group) < self._min_batch_requests:
                logger.info(
                    "%s round %d: sending %d %s requests immediately, below the batch minimum of %d",
                    stage,
                    round_no,
                    len(group),
                    model,
                    self._min_batch_requests,
                )
                immediate += group
                continue
            for chunk in chunk_for_batches(group):
                batch_ids.append(await self._submit(stage, round_no, chunk))
        # Every batch is recorded before the immediate requests start: an invalid
        # immediate request raises, and must not interrupt a batch submission.
        batched, sent = await asyncio.gather(
            self._collect_all(batch_ids), self._immediate.run(stage, round_no, immediate)
        )
        return batched + sent

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

    async def _collect_all(self, batch_ids: Sequence[str]) -> list[LlmResult]:
        results: list[LlmResult] = []
        for batch_id in batch_ids:
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
    error_message: str | None = None
    match outcome.type:
        case "succeeded":
            message = outcome.message
            status = _status_for(message)
        case "errored":
            status = ResultStatus.ERRORED
            error_type = outcome.error.error.type
            error_message = outcome.error.error.message
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
        error_message=error_message,
    )


def is_mid_stream_transient(error: BaseException) -> bool:
    """Whether *error* is a server-side error event inside a started stream."""
    if not isinstance(error, anthropic.APIStatusError) or error.status_code != 200:
        return False
    match error.body:
        case {"error": {"type": str(error_type)}}:
            return error_type in _MID_STREAM_TRANSIENT_TYPES
        case _:
            return False


def prompt_too_long_message(error: anthropic.APIStatusError) -> str | None:
    """The API's message, if *error* rejects a prompt as too long for the model."""
    match error.body:
        case {"error": {"type": "invalid_request_error", "message": str(message)}} if (
            _PROMPT_TOO_LONG.match(message) is not None
        ):
            return message
        case _:
            return None


class ImmediateTransport:
    """Concurrent streaming Messages calls through a fixed worker pool.

    Nothing is persisted until the stage records a result, so there is nothing
    to resume. Invalid requests raise, except a prompt too long for the model,
    which comes back as an ERRORED ``invalid_request_error`` result, as it does
    from a batch. Transient failures come back as ERRORED results once retries
    are spent: the SDK's own retries for failed responses, and
    MID_STREAM_ATTEMPTS attempts for error events inside a started stream.
    """

    def __init__(
        self,
        client: AsyncAnthropic,
        workers: int = DEFAULT_IMMEDIATE_WORKERS,
        mid_stream_wait: tenacity.wait.wait_base = tenacity.wait_exponential_jitter(
            initial=2, max=60
        ),
    ) -> None:
        self._client = client
        self._workers = workers
        self._mid_stream_wait = mid_stream_wait

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
            message = await self._stream(request)
        except anthropic.APIStatusError as e:
            too_long = prompt_too_long_message(e)
            if too_long is not None:
                return self._errored(
                    custom_id, stage, round_no, request, "invalid_request_error", too_long
                )
            if not (isinstance(e, _TRANSIENT_ERRORS) or is_mid_stream_transient(e)):
                raise
            return self._failed_after_retries(custom_id, stage, round_no, request, e)
        except anthropic.APIConnectionError as e:
            return self._failed_after_retries(custom_id, stage, round_no, request, e)
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
            error_message=None,
        )

    async def _stream(self, request: LlmRequest) -> Message:
        # stream() takes the same keyword arguments minus ``stream`` itself, which
        # the non-streaming params type declares but stages never set.
        params: dict[str, Any] = dict(request.params)
        async for attempt in tenacity.AsyncRetrying(
            retry=tenacity.retry_if_exception(is_mid_stream_transient),
            wait=self._mid_stream_wait,
            stop=tenacity.stop_after_attempt(MID_STREAM_ATTEMPTS),
            before_sleep=lambda state: logger.warning(
                "request for %s: error inside the stream, retrying (attempt %d)",
                request.subject,
                state.attempt_number,
            ),
            reraise=True,
        ):
            with attempt:
                async with self._client.messages.stream(**params) as stream:
                    return await stream.get_final_message()
        raise AssertionError("tenacity returns or re-raises")

    @classmethod
    def _failed_after_retries(
        cls, custom_id: str, stage: str, round_no: int, request: LlmRequest, error: Exception
    ) -> LlmResult:
        logger.warning(
            "%s: request for %s failed after retries: %s",
            stage,
            request.subject,
            type(error).__name__,
        )
        return cls._errored(custom_id, stage, round_no, request, type(error).__name__, str(error))

    @staticmethod
    def _errored(
        custom_id: str,
        stage: str,
        round_no: int,
        request: LlmRequest,
        error_type: str,
        error_message: str,
    ) -> LlmResult:
        return LlmResult(
            custom_id=custom_id,
            batch_id=None,
            stage=stage,
            subject=request.subject,
            round=round_no,
            model=request.params["model"],
            status=ResultStatus.ERRORED,
            message=None,
            error_type=error_type,
            error_message=error_message,
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
    "claude-sonnet-5-5": ModelPrices(input=2.0, output=10.0, cache_read_multiplier=0.1),
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
