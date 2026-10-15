"""Tests for the relevance screen's storage and attempts, without network."""

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from anthropic.types import Message
from typer.testing import CliRunner

from palit.assess_relevance import (
    STAGE,
    RelevancePrompt,
    _process_relevance,
    app,
    load_history,
    load_prompt,
    reopen_refused,
    select_papers,
    settle_refused,
    store_screens,
)
from palit.hgnc import HgncResolver
from palit.llm import (
    ALL_TIME,
    FALLBACK_MODEL,
    LIGHT_MODEL,
    MODEL,
    LlmRequest,
    LlmResult,
    ResultStatus,
    StageHistory,
    record_result,
)

ROOT = Path(__file__).resolve().parents[1]
SCREEN_SCHEMA = json.loads((ROOT / "prompts/relevance_assessment_schema.json").read_text())
VALIDATOR = jsonschema.Draft202012Validator(SCREEN_SCHEMA)


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
        error_message=None,
    )


def _screen(relevant: bool, genes: list[str]) -> dict[str, Any]:
    return {
        "relevant": relevant,
        "confidence": "MEDIUM",
        "rationale": "r",
        "associations": [{"gene_symbol": g, "disease": "d"} for g in genes],
    }


def _row(db_path: Path, doi: str) -> sqlite3.Row:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row: sqlite3.Row = conn.execute("SELECT * FROM papers WHERE doi = ?", (doi,)).fetchone()
    return row


def test_the_screen_decides_relevance(db_path: Path, hgnc_resolver: HgncResolver) -> None:
    outcome = store_screens(
        db_path,
        [
            _result(STAGE, "10.1/out", _screen(False, ["GENEB"])),
            _result(STAGE, "10.1/nogene", _screen(True, [])),
            _result(STAGE, "10.1/gene", _screen(True, ["GENEA", "GENEC"])),
        ],
        VALIDATOR,
        hgnc_resolver,
    )
    assert outcome.stored == 3

    out = _row(db_path, "10.1/out")
    assert json.loads(out["relevance_assessment_json"]) == {
        "relevant": False,
        "screen": _screen(False, ["GENEB"]),
    }
    assert out["download_status"] is None
    assert json.loads(_row(db_path, "10.1/nogene")["relevance_assessment_json"])["relevant"]
    gene = _row(db_path, "10.1/gene")
    assert json.loads(gene["relevance_assessment_json"]) == {
        "relevant": True,
        "screen": _screen(True, ["GENEA", "GENEC"]),
    }
    assert gene["download_status"] == "scheduled"
    assert set(json.loads(gene["relevance_assessment_raw"])) >= {"id", "content"}
    with sqlite3.connect(db_path) as conn:
        mentions = conn.execute("SELECT hgnc_id, source FROM gene_mentions").fetchall()
    # A rejected screen's genes are not mentions.
    assert sorted(mentions) == [(1, "relevance_assessment"), (4, "relevance_assessment")]
    assert select_papers(db_path, None, _history(db_path)) == []


PANEL_PROMPT = ROOT / "prompts/panel_relevance_assessment_prompt.txt"
DEFAULT_PROMPT = ROOT / "prompts/relevance_assessment_prompt.txt"


def test_the_panel_prompt_takes_the_panel_description() -> None:
    prompt = load_prompt(PANEL_PROMPT, "Name: Pierre Robin Sequence")
    assert "Name: Pierre Robin Sequence" in prompt.system
    with pytest.raises(ValueError, match="needs a panel description"):
        load_prompt(PANEL_PROMPT, None)


def test_a_prompt_without_the_placeholder_rejects_a_panel_description() -> None:
    assert load_prompt(DEFAULT_PROMPT, None).system
    with pytest.raises(ValueError, match="placeholder for --scope-panel-id"):
        load_prompt(DEFAULT_PROMPT, "Name: Pierre Robin Sequence")


@pytest.mark.parametrize("options", [["--scope-panel-id", "160"], ["--panel-date", "2026-10-13"]])
def test_scope_panel_and_panel_date_go_together(db_path: Path, options: list[str]) -> None:
    result = CliRunner().invoke(app, ["--db-path", str(db_path), *options])
    assert result.exit_code == 1


def test_an_invalid_screen_leaves_the_paper_for_another_attempt(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    outcome = store_screens(
        db_path, [_result(STAGE, "10.1/gene", {"relevant": True})], VALIDATOR, hgnc_resolver
    )
    assert (outcome.stored, outcome.failed) == (0, 1)
    assert _row(db_path, "10.1/gene")["relevance_assessment_json"] is None
    with sqlite3.connect(db_path) as conn:
        (rejection,) = conn.execute("SELECT rejection FROM llm_requests").fetchone()
    assert rejection is not None
    assert "10.1/gene" in [p["doi"] for p in select_papers(db_path, None, _history(db_path))]


def _refused(stage: str, subject: str, model: str) -> LlmResult:
    message = Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": "refusal",
            "stop_sequence": None,
            "stop_details": {"type": "refusal", "category": "bio", "explanation": None},
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }
    )
    return LlmResult(
        custom_id=f"{stage}-{subject}-{model}",
        batch_id=None,
        stage=stage,
        subject=subject,
        round=1,
        model=model,
        status=ResultStatus.REFUSED,
        message=message,
        error_type=None,
        error_message=None,
    )


def _history(db_path: Path) -> StageHistory:
    return load_history(db_path, ALL_TIME, ALL_TIME, MODEL)


def _later() -> datetime:
    """A start after every recorded request, as --retry-refused uses for its invocation."""
    return datetime.now(UTC) + timedelta(seconds=1)


def test_selection_routes_refused_papers_to_the_fallback_model(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    outcome = store_screens(
        db_path,
        [
            _refused(STAGE, "10.1/out", MODEL),
            _refused(STAGE, "10.1/nogene", MODEL),
            _refused(STAGE, "10.1/nogene", FALLBACK_MODEL),
        ],
        VALIDATOR,
        hgnc_resolver,
    )
    assert (outcome.refused, outcome.to_fallback) == (3, 2)
    refusals = _history(db_path)
    papers = select_papers(db_path, None, refusals)
    # 10.1/nogene was refused for good; 10.1/out goes to the fallback model next.
    assert {p["doi"]: refusals.model_for(p["doi"]) for p in papers} == {
        "10.1/gene": MODEL,
        "10.1/out": FALLBACK_MODEL,
    }
    assert [p["doi"] for p in select_papers(db_path, 1, refusals)] == ["10.1/gene"]
    # With --retry-refused, refusals before this invocation no longer count, and the
    # paper settled as refused comes back once reopened.
    later = load_history(db_path, _later(), ALL_TIME, MODEL)
    assert len(select_papers(db_path, None, later)) == 2
    assert reopen_refused(db_path) == 1
    assert len(select_papers(db_path, None, later)) == 3
    assert later.model_for("10.1/out") == MODEL


class RefusingTransport:
    """The stage's model refuses every screen; FALLBACK_MODEL answers with a rejection."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []  # (subject, model)

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        results = []
        for request in requests:
            model = request.params["model"]
            self.sent.append((request.subject, model))
            if model != FALLBACK_MODEL:
                results.append(_refused(stage, request.subject, model))
            else:
                answer = _result(stage, request.subject, _screen(False, []))
                results.append(replace(answer, custom_id=f"{answer.custom_id}-f", model=model))
        return results

    async def resume(self, stage: str) -> list[LlmResult]:
        return []


def _run_refused(db_path: Path, resolver: HgncResolver, model: str) -> RefusingTransport:
    transport = RefusingTransport()
    asyncio.run(
        _process_relevance(
            transport=transport,
            db_path=db_path,
            prompt=RelevancePrompt(system="s", user_template="{title} {abstract}"),
            schema=SCREEN_SCHEMA,
            resolver=resolver,
            model=model,
            limit=None,
            max_retries=5,
            retry_refused=False,
            retry_failed=False,
        )
    )
    return transport


def _recorded(db_path: Path) -> list[tuple[str, str, int]]:
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT model, status, COUNT(*) FROM llm_requests GROUP BY model, status ORDER BY model"
        ).fetchall()


DOIS = ["10.1/gene", "10.1/nogene", "10.1/out"]


def test_a_round_of_refusals_is_followed_by_the_fallback_attempt(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    transport = _run_refused(db_path, hgnc_resolver, MODEL)
    assert transport.sent == [(doi, MODEL) for doi in DOIS] + [
        (doi, FALLBACK_MODEL) for doi in DOIS
    ]
    for doi in DOIS:
        assert json.loads(_row(db_path, doi)["relevance_assessment_json"])["relevant"] is False
    assert _recorded(db_path) == [(MODEL, "refused", 3), (FALLBACK_MODEL, "succeeded", 3)]


def test_refusals_by_a_model_without_a_fallback_are_final(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """LIGHT_MODEL never falls back to a higher tier: its refusals settle the papers."""
    transport = _run_refused(db_path, hgnc_resolver, LIGHT_MODEL)
    assert transport.sent == [(doi, LIGHT_MODEL) for doi in DOIS]
    for doi in DOIS:
        assessment = json.loads(_row(db_path, doi)["relevance_assessment_json"])
        assert assessment["refused"]["model"] == LIGHT_MODEL
    assert _recorded(db_path) == [(LIGHT_MODEL, "refused", 3)]


def test_refusals_count_only_for_the_model_the_stage_runs_on(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A paper MODEL refused in an earlier run goes to LIGHT_MODEL first in a run on it."""
    store_screens(db_path, [_refused(STAGE, "10.1/out", MODEL)], VALIDATOR, hgnc_resolver)
    assert _history(db_path).model_for("10.1/out") == FALLBACK_MODEL
    light = load_history(db_path, ALL_TIME, ALL_TIME, LIGHT_MODEL)
    assert light.model_for("10.1/out") == LIGHT_MODEL


def test_a_run_on_the_fallback_model_has_no_fallback(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """FALLBACK_MODEL's refusals are final when the stage runs on it."""
    outcome = store_screens(
        db_path, [_refused(STAGE, "10.1/out", FALLBACK_MODEL)], VALIDATOR, hgnc_resolver
    )
    assert (outcome.refused, outcome.to_fallback, outcome.settled) == (1, 0, 1)
    history = load_history(db_path, ALL_TIME, ALL_TIME, FALLBACK_MODEL)
    assert history.skipped("10.1/out")
    assert {
        p["doi"]: history.model_for(p["doi"]) for p in select_papers(db_path, None, history)
    } == {
        "10.1/gene": FALLBACK_MODEL,
        "10.1/nogene": FALLBACK_MODEL,
    }


# --- papers both models refused -------------------------------------------------


def _refused_at(stage: str, subject: str, model: str, category: str, at: str) -> LlmResult:
    """A refusal with its own custom_id, recorded at *at* by the caller's timestamp."""
    result = _refused(stage, subject, model)
    assert result.message is not None
    stop_details = result.message.stop_details
    assert stop_details is not None
    message = result.message.model_copy(
        update={"stop_details": stop_details.model_copy(update={"category": category})}
    )
    return replace(result, custom_id=f"{stage}-{subject}-{model}-{at}", message=message)


def _record_at(db_path: Path, result: LlmResult, at: str) -> None:
    """Record *result* without storing an output, as a run before settling did."""
    with sqlite3.connect(db_path) as conn:
        record_result(conn, result)
        conn.execute(
            "UPDATE llm_requests SET completed_at = ? WHERE custom_id = ?", (at, result.custom_id)
        )


def _assessment(db_path: Path, doi: str) -> dict[str, Any]:
    assessment: dict[str, Any] = json.loads(_row(db_path, doi)["relevance_assessment_json"])
    return assessment


def test_a_screen_both_models_refused_is_settled_as_not_relevant(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    outcome = store_screens(
        db_path, [_refused(STAGE, "10.1/out", FALLBACK_MODEL)], VALIDATOR, hgnc_resolver
    )
    assert (outcome.refused, outcome.to_fallback, outcome.settled) == (1, 0, 1)
    row = _row(db_path, "10.1/out")
    assert json.loads(row["relevance_assessment_json"]) == {
        "relevant": False,
        "screen": None,
        "refused": {"model": FALLBACK_MODEL, "category": "bio"},
    }
    assert row["relevance_assessment_raw"] is None
    assert row["download_status"] is None
    # Not selected again, even when earlier refusals no longer count (--retry-refused).
    later = load_history(db_path, _later(), ALL_TIME, MODEL)
    assert "10.1/out" not in [p["doi"] for p in select_papers(db_path, None, later)]


def test_settle_step_settles_earlier_refusals_once(db_path: Path) -> None:
    """A database from before settling: refused for good, and still without a result."""
    _record_at(db_path, _refused_at(STAGE, "10.1/out", MODEL, "bio", "1"), "2026-09-01T01")
    _record_at(
        db_path, _refused_at(STAGE, "10.1/out", FALLBACK_MODEL, "cyber", "2"), "2026-09-01T02"
    )
    _record_at(db_path, _refused_at(STAGE, "10.1/out", FALLBACK_MODEL, "bio", "3"), "2026-09-01T03")
    # Refused by MODEL only: due its fallback request, so not settled.
    _record_at(db_path, _refused_at(STAGE, "10.1/nogene", MODEL, "bio", "4"), "2026-09-01T04")

    assert settle_refused(db_path) == 1
    assert _assessment(db_path, "10.1/out")["refused"] == {
        "model": FALLBACK_MODEL,
        "category": "bio",  # of the latest refusal by FALLBACK_MODEL
    }
    assert _row(db_path, "10.1/nogene")["relevance_assessment_json"] is None
    assert settle_refused(db_path) == 0
    assert [p["doi"] for p in select_papers(db_path, None, _history(db_path))] == [
        "10.1/gene",
        "10.1/nogene",
    ]


def test_settle_step_leaves_papers_in_flight(db_path: Path) -> None:
    _record_at(db_path, _refused_at(STAGE, "10.1/out", FALLBACK_MODEL, "bio", "1"), "2026-09-01T01")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO llm_requests (custom_id, stage, subject, round, model, status)"
            " VALUES ('pending-1', ?, '10.1/out', 1, ?, 'pending')",
            (STAGE, MODEL),
        )
    assert settle_refused(db_path) == 0


class AnsweringTransport:
    """Answers every screen as relevant, with GENEC."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []  # (subject, model)

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        results = []
        for request in requests:
            model = request.params["model"]
            self.sent.append((request.subject, model))
            answer = _result(stage, request.subject, _screen(True, ["GENEC"]))
            results.append(replace(answer, custom_id=f"{answer.custom_id}-{model}", model=model))
        return results

    async def resume(self, stage: str) -> list[LlmResult]:
        return []


def _run(
    db_path: Path,
    transport: AnsweringTransport,
    resolver: HgncResolver,
    *,
    retry_refused: bool,
    limit: int | None = None,
) -> None:
    asyncio.run(
        _process_relevance(
            transport=transport,
            db_path=db_path,
            prompt=RelevancePrompt(system="s", user_template="{title} {abstract}"),
            schema=SCREEN_SCHEMA,
            resolver=resolver,
            model=MODEL,
            limit=limit,
            max_retries=5,
            retry_refused=retry_refused,
            retry_failed=False,
        )
    )


def _settle(db_path: Path, resolver: HgncResolver) -> None:
    """10.1/out and 10.1/gene refused for good, 10.1/nogene screened out."""
    store_screens(
        db_path,
        [
            _refused(STAGE, "10.1/out", FALLBACK_MODEL),
            _refused(STAGE, "10.1/gene", FALLBACK_MODEL),
            _result(STAGE, "10.1/nogene", _screen(False, [])),
        ],
        VALIDATOR,
        resolver,
    )


def test_a_run_does_not_send_settled_papers(db_path: Path, hgnc_resolver: HgncResolver) -> None:
    _settle(db_path, hgnc_resolver)
    transport = AnsweringTransport()
    _run(db_path, transport, hgnc_resolver, retry_refused=False)
    assert transport.sent == []
    assert _assessment(db_path, "10.1/out")["refused"]["model"] == FALLBACK_MODEL


def test_retry_refused_replaces_settled_results_on_success(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    _settle(db_path, hgnc_resolver)
    transport = AnsweringTransport()
    _run(db_path, transport, hgnc_resolver, retry_refused=True)
    # Both refused papers are screened again, starting with MODEL.
    assert transport.sent == [("10.1/gene", MODEL), ("10.1/out", MODEL)]
    for doi in ("10.1/out", "10.1/gene"):
        assert _assessment(db_path, doi) == {"relevant": True, "screen": _screen(True, ["GENEC"])}
        assert _row(db_path, doi)["download_status"] == "scheduled"
    # A later run without --retry-refused leaves the recovered papers alone.
    _run(db_path, transport, hgnc_resolver, retry_refused=False)
    assert "refused" not in _assessment(db_path, "10.1/out")


def test_retry_refused_settles_again_the_papers_it_did_not_send(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    store_screens(
        db_path,
        [
            _refused(STAGE, "10.1/nogene", FALLBACK_MODEL),
            _refused(STAGE, "10.1/out", FALLBACK_MODEL),
            _result(STAGE, "10.1/gene", _screen(False, [])),
        ],
        VALIDATOR,
        hgnc_resolver,
    )
    transport = AnsweringTransport()
    _run(db_path, transport, hgnc_resolver, retry_refused=True, limit=1)
    assert transport.sent == [("10.1/nogene", MODEL)]
    assert _assessment(db_path, "10.1/nogene")["relevant"] is True
    # Reopened, but beyond the limit: settled again at the end of the invocation.
    assert _assessment(db_path, "10.1/out")["refused"]["model"] == FALLBACK_MODEL
