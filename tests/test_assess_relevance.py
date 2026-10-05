"""Tests for the two relevance levels' storage, without network."""

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

from palit.assess_relevance import (
    CHECK_STAGE,
    STAGE,
    CheckSettings,
    Refusal,
    RelevancePrompt,
    SettledRefusals,
    _process_relevance,
    load_history,
    refused_assessment,
    reopen_refused,
    select_papers,
    select_screened,
    settle_refused,
    store_checks,
    store_screens,
)
from palit.hgnc import HgncResolver
from palit.llm import (
    ALL_TIME,
    FALLBACK_MODEL,
    MODEL,
    LlmRequest,
    LlmResult,
    ResultStatus,
    StageHistory,
    json_output_config,
    record_result,
)
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
        error_message=None,
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
    assert select_papers(db_path, None, _history(db_path, STAGE)) == []
    assert [p["doi"] for p in select_screened(db_path, None, _history(db_path, CHECK_STAGE))] == [
        "10.1/gene"
    ]


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
    assert select_screened(db_path, None, _history(db_path, CHECK_STAGE)) == []


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
    assert select_screened(db_path, None, _history(db_path, CHECK_STAGE)) == []


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
    with sqlite3.connect(db_path) as conn:
        rejection = conn.execute(
            "SELECT rejection FROM llm_requests WHERE stage = ?", (CHECK_STAGE,)
        ).fetchone()
    assert rejection == ("no associations",)


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


def _history(db_path: Path, stage: str) -> StageHistory:
    return load_history(db_path, stage, ALL_TIME, ALL_TIME)


def test_selection_routes_refused_papers_to_the_fallback_model(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    validator = jsonschema.Draft202012Validator(SCREEN_SCHEMA)
    outcome = store_screens(
        db_path,
        [
            _refused(STAGE, "10.1/out", MODEL),
            _refused(STAGE, "10.1/nogene", MODEL),
            _refused(STAGE, "10.1/nogene", FALLBACK_MODEL),
        ],
        validator,
        hgnc_resolver,
        screen_only=False,
    )
    assert (outcome.refused, outcome.to_fallback) == (3, 2)
    refusals = _history(db_path, STAGE)
    papers = select_papers(db_path, None, refusals)
    # 10.1/nogene was refused for good; 10.1/out goes to the fallback model next.
    assert {p["doi"]: refusals.model_for(p["doi"]) for p in papers} == {
        "10.1/gene": MODEL,
        "10.1/out": FALLBACK_MODEL,
    }
    assert [p["doi"] for p in select_papers(db_path, 1, refusals)] == ["10.1/gene"]
    # With --retry-refused, refusals before this invocation no longer count, and the
    # paper settled as refused comes back once reopened.
    later = load_history(db_path, STAGE, datetime.now(UTC) + timedelta(seconds=1), ALL_TIME)
    assert len(select_papers(db_path, None, later)) == 2
    assert reopen_refused(db_path) == 1
    assert len(select_papers(db_path, None, later)) == 3
    assert later.model_for("10.1/out") == MODEL


def test_check_level_falls_back_on_its_own(
    db_path: Path, check: CheckSettings, hgnc_resolver: HgncResolver
) -> None:
    store_screens(
        db_path,
        [_result(STAGE, "10.1/gene", _screen(True, ["GENEA"]))],
        jsonschema.Draft202012Validator(SCREEN_SCHEMA),
        hgnc_resolver,
        screen_only=False,
    )
    store_checks(db_path, [_refused(CHECK_STAGE, "10.1/gene", MODEL)], check)
    refusals = _history(db_path, CHECK_STAGE)
    [paper] = select_screened(db_path, None, refusals)
    assert refusals.model_for(paper["doi"]) == FALLBACK_MODEL
    assert _history(db_path, STAGE).model_for("10.1/gene") == MODEL
    store_checks(db_path, [_refused(CHECK_STAGE, "10.1/gene", FALLBACK_MODEL)], check)
    assert select_screened(db_path, None, _history(db_path, CHECK_STAGE)) == []


class RefusingTransport:
    """MODEL refuses every screen; FALLBACK_MODEL answers with a rejection."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []  # (subject, model)

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        results = []
        for request in requests:
            model = request.params["model"]
            self.sent.append((request.subject, model))
            if model == MODEL:
                results.append(_refused(stage, request.subject, model))
            else:
                answer = _result(stage, request.subject, _screen(False, []))
                results.append(replace(answer, custom_id=f"{answer.custom_id}-f", model=model))
        return results

    async def resume(self, stage: str) -> list[LlmResult]:
        return []


def test_a_round_of_refusals_is_followed_by_the_fallback_attempt(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    transport = RefusingTransport()
    asyncio.run(
        _process_relevance(
            transport=transport,
            db_path=db_path,
            prompt=RelevancePrompt(system="s", user_template="{title} {abstract}"),
            schema=SCREEN_SCHEMA,
            resolver=hgnc_resolver,
            check=None,
            limit=None,
            max_retries=5,
            retry_refused=False,
            retry_failed=False,
        )
    )
    dois = ["10.1/gene", "10.1/nogene", "10.1/out"]
    assert transport.sent == [(doi, MODEL) for doi in dois] + [
        (doi, FALLBACK_MODEL) for doi in dois
    ]
    for doi in dois:
        assert json.loads(_row(db_path, doi)["relevance_assessment_json"])["relevant"] is False
    with sqlite3.connect(db_path) as conn:
        recorded = conn.execute(
            "SELECT model, status, COUNT(*) FROM llm_requests GROUP BY model, status ORDER BY model"
        ).fetchall()
    assert recorded == [(MODEL, "refused", 3), (FALLBACK_MODEL, "succeeded", 3)]


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
        db_path,
        [_refused(STAGE, "10.1/out", FALLBACK_MODEL)],
        jsonschema.Draft202012Validator(SCREEN_SCHEMA),
        hgnc_resolver,
        screen_only=False,
    )
    assert (outcome.refused, outcome.to_fallback, outcome.settled) == (1, 0, 1)
    row = _row(db_path, "10.1/out")
    assert json.loads(row["relevance_assessment_json"]) == {
        "relevant": False,
        "screen": None,
        "panelapp_check": None,
        "refused": {"level": "screen", "model": FALLBACK_MODEL, "category": "bio"},
    }
    assert json.loads(row["relevance_assessment_raw"]) == {"screen": None, "panelapp_check": None}
    assert row["download_status"] is None
    # Not selected again, even when earlier refusals no longer count (--retry-refused).
    later = load_history(db_path, STAGE, datetime.now(UTC) + timedelta(seconds=1), ALL_TIME)
    assert "10.1/out" not in [p["doi"] for p in select_papers(db_path, None, later)]


def test_a_check_both_models_refused_keeps_the_screen(
    db_path: Path, check: CheckSettings, hgnc_resolver: HgncResolver
) -> None:
    store_screens(
        db_path,
        [_result(STAGE, "10.1/gene", _screen(True, ["GENEA"]))],
        jsonschema.Draft202012Validator(SCREEN_SCHEMA),
        hgnc_resolver,
        screen_only=False,
    )
    outcome = store_checks(db_path, [_refused(CHECK_STAGE, "10.1/gene", FALLBACK_MODEL)], check)
    assert outcome.settled == 1
    row = _row(db_path, "10.1/gene")
    assert json.loads(row["relevance_assessment_json"]) == refused_assessment(
        Refusal("panelapp_check", FALLBACK_MODEL, "bio"), _screen(True, ["GENEA"])
    )
    assert set(json.loads(row["relevance_assessment_raw"])["screen"]) >= {"id", "content"}
    assert row["relevance_screen_json"] is None
    with sqlite3.connect(db_path) as conn:
        # the screen passed it, so its genes are recorded as for any paper it passes
        assert conn.execute("SELECT hgnc_id FROM gene_mentions").fetchall() == [(1,)]
    later = load_history(db_path, CHECK_STAGE, datetime.now(UTC) + timedelta(seconds=1), ALL_TIME)
    assert select_screened(db_path, None, later) == []


def test_settle_step_settles_earlier_refusals_once(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A database from before settling: refused for good, and still without a result."""
    _record_at(db_path, _refused_at(STAGE, "10.1/out", MODEL, "bio", "1"), "2026-09-01T01")
    _record_at(
        db_path, _refused_at(STAGE, "10.1/out", FALLBACK_MODEL, "cyber", "2"), "2026-09-01T02"
    )
    _record_at(db_path, _refused_at(STAGE, "10.1/out", FALLBACK_MODEL, "bio", "3"), "2026-09-01T03")
    store_screens(
        db_path,
        [_result(STAGE, "10.1/gene", _screen(True, ["GENEA"]))],
        jsonschema.Draft202012Validator(SCREEN_SCHEMA),
        hgnc_resolver,
        screen_only=False,
    )
    _record_at(
        db_path, _refused_at(CHECK_STAGE, "10.1/gene", FALLBACK_MODEL, "bio", "4"), "2026-09-01T04"
    )
    # Refused by MODEL only: due its fallback request, so not settled.
    _record_at(db_path, _refused_at(STAGE, "10.1/nogene", MODEL, "bio", "5"), "2026-09-01T05")

    assert settle_refused(db_path, hgnc_resolver) == SettledRefusals(screen=1, panelapp_check=1)
    assert _assessment(db_path, "10.1/out")["refused"] == {
        "level": "screen",
        "model": FALLBACK_MODEL,
        "category": "bio",  # of the latest refusal by FALLBACK_MODEL
    }
    gene = _assessment(db_path, "10.1/gene")
    assert gene["refused"]["level"] == "panelapp_check"
    assert gene["screen"] == _screen(True, ["GENEA"])
    assert _row(db_path, "10.1/nogene")["relevance_assessment_json"] is None
    assert settle_refused(db_path, hgnc_resolver) == SettledRefusals(screen=0, panelapp_check=0)
    assert [p["doi"] for p in select_papers(db_path, None, _history(db_path, STAGE))] == [
        "10.1/nogene"
    ]
    assert select_screened(db_path, None, _history(db_path, CHECK_STAGE)) == []


def test_settle_step_leaves_papers_in_flight(db_path: Path, hgnc_resolver: HgncResolver) -> None:
    _record_at(db_path, _refused_at(STAGE, "10.1/out", FALLBACK_MODEL, "bio", "1"), "2026-09-01T01")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO llm_requests (custom_id, stage, subject, round, model, status)"
            " VALUES ('pending-1', ?, '10.1/out', 1, ?, 'pending')",
            (STAGE, MODEL),
        )
    assert settle_refused(db_path, hgnc_resolver) == SettledRefusals(screen=0, panelapp_check=0)


class AnsweringTransport:
    """Answers every screen with GENEC and every check with a new-gene verdict."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []  # (stage, subject, model)

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        results = []
        for request in requests:
            model = request.params["model"]
            self.sent.append((stage, request.subject, model))
            output = (
                _screen(True, ["GENEC"])
                if stage == STAGE
                else {
                    "rationale": "new",
                    "associations": [
                        {
                            "gene_symbol": "GENEC",
                            "disease": "d",
                            "verdict": "new_gene",
                            "reason": "r",
                        }
                    ],
                }
            )
            answer = _result(stage, request.subject, output)
            results.append(replace(answer, custom_id=f"{answer.custom_id}-{model}", model=model))
        return results

    async def resume(self, stage: str) -> list[LlmResult]:
        return []


def _run(
    db_path: Path,
    transport: AnsweringTransport,
    resolver: HgncResolver,
    check: CheckSettings,
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
            check=check,
            limit=limit,
            max_retries=5,
            retry_refused=retry_refused,
            retry_failed=False,
        )
    )


def _settle_both_levels(db_path: Path, check: CheckSettings, resolver: HgncResolver) -> None:
    """10.1/out refused for good at the screen, 10.1/gene at the PanelApp check."""
    validator = jsonschema.Draft202012Validator(SCREEN_SCHEMA)
    store_screens(
        db_path,
        [
            _refused(STAGE, "10.1/out", FALLBACK_MODEL),
            _result(STAGE, "10.1/gene", _screen(True, ["GENEA"])),
            _result(STAGE, "10.1/nogene", _screen(False, [])),
        ],
        validator,
        resolver,
        screen_only=False,
    )
    store_checks(db_path, [_refused(CHECK_STAGE, "10.1/gene", FALLBACK_MODEL)], check)


def test_a_run_does_not_send_settled_papers(
    db_path: Path, check: CheckSettings, hgnc_resolver: HgncResolver
) -> None:
    _settle_both_levels(db_path, check, hgnc_resolver)
    transport = AnsweringTransport()
    _run(db_path, transport, hgnc_resolver, check, retry_refused=False)
    assert transport.sent == []
    assert _assessment(db_path, "10.1/out")["refused"]["level"] == "screen"


def test_retry_refused_replaces_settled_results_on_success(
    db_path: Path, check: CheckSettings, hgnc_resolver: HgncResolver
) -> None:
    _settle_both_levels(db_path, check, hgnc_resolver)
    transport = AnsweringTransport()
    _run(db_path, transport, hgnc_resolver, check, retry_refused=True)
    # The paper refused at the screen is screened and then checked; the one refused at
    # the check is checked against its stored screen. Both start with MODEL again.
    assert transport.sent == [
        (STAGE, "10.1/out", MODEL),
        (CHECK_STAGE, "10.1/gene", MODEL),
        (CHECK_STAGE, "10.1/out", MODEL),
    ]
    for doi in ("10.1/out", "10.1/gene"):
        assessment = _assessment(db_path, doi)
        assert "refused" not in assessment
        assert assessment["relevant"] is True
        assert assessment["panelapp_check"]["associations"][0]["verdict"] == "new_gene"
        assert _row(db_path, doi)["download_status"] == "scheduled"
    assert _assessment(db_path, "10.1/gene")["screen"] == _screen(True, ["GENEA"])
    # A later run without --retry-refused leaves the recovered papers alone.
    _run(db_path, transport, hgnc_resolver, check, retry_refused=False)
    assert "refused" not in _assessment(db_path, "10.1/out")


def test_retry_refused_settles_again_the_papers_it_did_not_send(
    db_path: Path, check: CheckSettings, hgnc_resolver: HgncResolver
) -> None:
    store_screens(
        db_path,
        [
            _refused(STAGE, "10.1/nogene", FALLBACK_MODEL),
            _refused(STAGE, "10.1/out", FALLBACK_MODEL),
            _result(STAGE, "10.1/gene", _screen(False, [])),
        ],
        jsonschema.Draft202012Validator(SCREEN_SCHEMA),
        hgnc_resolver,
        screen_only=False,
    )
    transport = AnsweringTransport()
    _run(db_path, transport, hgnc_resolver, check, retry_refused=True, limit=1)
    assert transport.sent == [(STAGE, "10.1/nogene", MODEL), (CHECK_STAGE, "10.1/nogene", MODEL)]
    assert _assessment(db_path, "10.1/nogene")["relevant"] is True
    # Reopened, but beyond the limit: settled again at the end of the invocation.
    assert _assessment(db_path, "10.1/out")["refused"]["level"] == "screen"
