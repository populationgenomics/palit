"""Tests for extraction helpers that need no network or PDFs."""

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
import tenacity
from anthropic.types import Message

from palit.extract_evidence import (
    MAX_CONCURRENT_PAPERS,
    STAGE,
    Conversation,
    ExtractionRunner,
    Remaining,
    RequestSettings,
    RoundOutcome,
    count_remaining,
    extraction_quotes,
    normalize_extraction_genes,
    prune_citations,
    select_papers,
    start_conversation,
    structural_problems,
    symbol_pattern,
    variant_lookup_results,
)
from palit.hgnc import HgncEntry, HgncResolver
from palit.llm import (
    EVERY_REFUSAL,
    FALLBACK_MODEL,
    MODEL,
    LlmRequest,
    LlmResult,
    ResultStatus,
    json_output_config,
    stage_refusals,
)
from palit.lookup_tools import (
    MAX_ATTEMPTS,
    LookupRunner,
    VariantLookupClient,
    VariantLookupSettings,
    frequency_row,
    summarise_variant_response,
)
from palit.panelapp_integration import criteria_object_to_list
from palit.papers import doi_to_path
from palit.quotes import QuoteCheck

CRITERIA = ["criterion_A", "criterion_B", "criterion_C", "criterion_D", "criterion_E"]
SCHEMA_SQL = Path(__file__).resolve().parents[1] / "schema.sql"


def _citation(quote: str) -> dict[str, str]:
    return {"quote": quote, "commentary": "c"}


def _entity(family_count: int | None, independent: int | None) -> dict[str, Any]:
    return {
        "family_count": family_count,
        "independent_family_count": independent,
        "citations": [_citation("entity quote")],
        "evidence_assessments": [
            {"name": name, "result": False, "rationale": "r", "confidence": "LOW", "citations": []}
            for name in CRITERIA
        ],
    }


def _gene(symbol: str, entities: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "gene_symbol": symbol,
        "summary": "s",
        "variants": [{"variant": "c.1A>G", "quote": "variant quote"}],
        "disease_entities": entities,
        "quality_concerns": [{"concern": "x", "citations": [_citation("concern quote")]}],
    }


def test_extraction_quotes_covers_every_citation_site() -> None:
    extraction = {"gene_evaluations": [_gene("ABC1", [_entity(2, 2)])]}
    extraction["gene_evaluations"][0]["disease_entities"][0]["evidence_assessments"][0][
        "citations"
    ] = [_citation("criterion quote")]
    assert extraction_quotes(extraction) == [
        "variant quote",
        "entity quote",
        "criterion quote",
        "concern quote",
    ]


def test_structural_problems() -> None:
    assert structural_problems({"gene_evaluations": [_gene("ABC1", [_entity(2, 2)])]}) == []
    problems = structural_problems({"gene_evaluations": [_gene("PLACEHOLDER", [_entity(2, 3)])]})
    assert any("placeholder" in p for p in problems)
    assert any("independent_family_count" in p for p in problems)


def test_variant_lookup_results_reads_only_variant_tool_results() -> None:
    variant_result = {"gene_symbol": "ABC1", "variant": "c.1A>G", "status": "ok", "candidates": []}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "paper"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t1", "name": "lookup_variants", "input": {}},
                {"type": "tool_use", "id": "t2", "name": "lookup_genes", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": json.dumps({"results": [variant_result]}),
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "t2",
                    "content": json.dumps({"results": [{"symbol": "ABC1"}]}),
                },
                {"type": "text", "text": "finalise"},
            ],
        },
    ]
    assert variant_lookup_results(messages) == [variant_result]


def test_frequency_row_picks_highest_ac_candidate() -> None:
    item = {"gene_symbol": "ABC1", "variant": "p.Arg1Cys", "genome_build": "unknown"}
    response = {
        "normalized": [
            {
                "hgvs_c": "c.1C>T",
                "hgvs_p": "p.Arg1Cys",
                "pseudo_vcf": "1-100-C-T",
                "frequency": {"ac": 3, "an": 10},
            },
            {
                "hgvs_c": "c.1C>A",
                "hgvs_p": "p.Arg1Cys",
                "pseudo_vcf": "1-100-C-A",
                "frequency": None,
            },
            {
                "hgvs_c": "c.1C>G",
                "hgvs_p": "p.Arg1Cys",
                "pseudo_vcf": "1-100-C-G",
                "frequency": {"ac": 7, "an": 10},
            },
        ]
    }
    row = frequency_row(summarise_variant_response(item, response))
    assert row.variant_id == "1-100-C-G"
    assert row.gnomad == {"ac": 7, "an": 10}
    assert row.normalization["total_normalizations"] == 3
    assert row.normalization["selected_for_max_ac"] == 7


def test_frequency_row_for_absent_and_failed_variants() -> None:
    item = {"gene_symbol": "ABC1", "variant": "c.1A>G", "genome_build": "GRCh38"}
    absent = frequency_row(
        summarise_variant_response(
            item,
            {
                "normalized": [
                    {"hgvs_c": "c.1A>G", "hgvs_p": None, "pseudo_vcf": "1-5-A-G", "frequency": None}
                ]
            },
        )
    )
    assert absent.gnomad == {"variant_not_found": True}
    assert absent.normalization["selected_for_max_ac"] is None

    failed = frequency_row(
        summarise_variant_response(
            item, {"error": {"code": "PARSE_ERROR", "message": "bad notation"}}
        )
    )
    assert failed.variant_id == "c.1A>G"
    assert failed.gnomad == {"normalization_error": True}
    assert failed.normalization["error_code"] == "PARSE_ERROR"


def _hgnc_entry(hgnc_id: int, symbol: str, prev_symbols: tuple[str, ...]) -> HgncEntry:
    return HgncEntry(
        hgnc_id=hgnc_id,
        symbol=symbol,
        name=symbol,
        prev_symbols=prev_symbols,
        alias_symbols=(),
        locus_group="protein-coding gene",
        location=None,
        chromosome=None,
    )


def _resolver(*entries: HgncEntry) -> HgncResolver:
    by_prev: dict[str, list[HgncEntry]] = {}
    for entry in entries:
        for prev in entry.prev_symbols:
            by_prev.setdefault(prev.upper(), []).append(entry)
    return HgncResolver(
        _by_symbol={entry.symbol.upper(): entry for entry in entries},
        _by_prev=by_prev,
        _by_alias={},
        _by_hgnc_id={entry.hgnc_id: entry for entry in entries},
    )


PRKN = _hgnc_entry(8607, "PRKN", ("PARK2",))
AARS1 = _hgnc_entry(20, "AARS1", ("AARS",))
C9ORF72 = _hgnc_entry(28337, "C9orf72", ())


def test_normalize_extraction_genes_keeps_paper_symbol() -> None:
    normalized, unresolved = normalize_extraction_genes(
        {"gene_evaluations": [{"gene_symbol": "Park2", "summary": "PARK2 variants"}]},
        _resolver(PRKN),
    )
    gene = normalized["gene_evaluations"][0]
    assert gene["hgnc_id"] == 8607
    assert gene["paper_gene_symbol"] == "PARK2"
    assert "gene_symbol" not in gene
    assert gene["summary"] == "PRKN variants"
    assert unresolved == []


def test_normalize_extraction_genes_resolves_mixed_case_current_symbol() -> None:
    normalized, _ = normalize_extraction_genes(
        {"gene_evaluations": [{"gene_symbol": "C9orf72", "summary": "C9orf72 repeat"}]},
        _resolver(C9ORF72),
    )
    gene = normalized["gene_evaluations"][0]
    assert gene["hgnc_id"] == 28337
    assert gene["paper_gene_symbol"] == "C9ORF72"
    assert gene["summary"] == "C9orf72 repeat"


def test_normalize_extraction_genes_leaves_current_symbol_alone() -> None:
    normalized, _ = normalize_extraction_genes(
        {
            "gene_evaluations": [
                {
                    "gene_symbol": "AARS",
                    "summary": "AARS (now AARS1) and AARS-1; AARS-related neuropathy.",
                }
            ]
        },
        _resolver(AARS1),
    )
    gene = normalized["gene_evaluations"][0]
    assert gene["paper_gene_symbol"] == "AARS"
    assert gene["summary"] == "AARS1 (now AARS1) and AARS-1; AARS1-related neuropathy."


def test_normalize_extraction_genes_keeps_verbatim_fields() -> None:
    gene = {
        "gene_symbol": "PARK2",
        "summary": "PARK2 loss",
        "variants": [{"variant": "PARK2 c.1A>G", "quote": "PARK2 c.1A>G in two families"}],
        "disease_entities": [
            {
                "description": "PARK2 parkinsonism",
                "citations": [{"quote": "mutations in PARK2", "commentary": "PARK2 cases"}],
                "previously_reported_sources": [
                    {"title": "PARK2 mutations in parkinsonism", "context": "PARK2 cohort"}
                ],
            }
        ],
    }
    normalized, _ = normalize_extraction_genes({"gene_evaluations": [gene]}, _resolver(PRKN))
    result = normalized["gene_evaluations"][0]
    assert result["summary"] == "PRKN loss"
    assert result["variants"] == [
        {"variant": "PARK2 c.1A>G", "quote": "PARK2 c.1A>G in two families"}
    ]
    entity = result["disease_entities"][0]
    assert entity["description"] == "PRKN parkinsonism"
    assert entity["citations"] == [{"quote": "mutations in PARK2", "commentary": "PRKN cases"}]
    assert entity["previously_reported_sources"] == [
        {"title": "PARK2 mutations in parkinsonism", "context": "PRKN cohort"}
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("AARS", "AARS1"),
        ("AARS.", "AARS1."),
        ("(AARS)", "(AARS1)"),
        ("PARS/AARS", "PARS/AARS1"),
        ("AARS's", "AARS1's"),
        ("AARS-related", "AARS1-related"),
        ("AARS1", "AARS1"),
        ("AARS2", "AARS2"),
        ("AARS-1", "AARS-1"),
        ("AARS_HUMAN", "AARS_HUMAN"),
        ("AARS@", "AARS@"),
        ("MT-AARS", "MT-AARS"),
        ("XAARS", "XAARS"),
        ("aars", "aars"),
    ],
)
def test_symbol_pattern_matches_whole_symbols(text: str, expected: str) -> None:
    assert symbol_pattern(["AARS"]).sub("AARS1", text) == expected


def test_criteria_object_to_list_orders_and_names() -> None:
    entity: dict[str, Any] = {
        "evidence_assessments": {
            name: {"result": False, "rationale": name, "confidence": "LOW", "citations": []}
            for name in reversed(CRITERIA)
        }
    }
    criteria_object_to_list([entity])
    assert [c["name"] for c in entity["evidence_assessments"]] == CRITERIA
    assert entity["evidence_assessments"][0]["rationale"] == "criterion_A"


def test_prune_citations_keeps_variant_quotes() -> None:
    extraction = {"gene_evaluations": [_gene("ABC1", [_entity(2, 2)])]}
    removed = prune_citations(extraction, {"entity quote", "concern quote", "variant quote"})
    gene = extraction["gene_evaluations"][0]
    assert removed == 2
    assert gene["disease_entities"][0]["citations"] == []
    assert gene["quality_concerns"][0]["citations"] == []
    assert gene["variants"][0]["quote"] == "variant quote"


def test_select_papers_skips_only_papers_refused_for_good(tmp_path: Path) -> None:
    """A refusal by MODEL leaves a paper due its fallback; one by FALLBACK_MODEL excludes it.

    Refusals before the --retry-refused cutoff no longer count; in-flight papers stay out.
    """
    db_path = tmp_path / "run.sqlite"
    before = datetime(2026, 9, 1, tzinfo=UTC)
    cutoff = datetime(2026, 9, 2, tzinfo=UTC)
    after = datetime(2026, 9, 3, tzinfo=UTC)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type, download_status, "
            "relevance_assessment_json, evidence_extraction_json) "
            "VALUES (?, 't', 'pubmed', 'initial', 'downloaded', '{\"relevant\": true}', ?)",
            [
                ("10.1/new", None),
                ("10.1/to-fallback", None),
                ("10.1/refused-before", None),
                ("10.1/refused-after", None),
                ("10.1/pending", None),
                ("10.1/extracted", "{}"),
            ],
        )
        conn.executemany(
            "INSERT INTO llm_requests (custom_id, stage, subject, round, model, status, "
            "completed_at) VALUES (?, ?, ?, 2, ?, ?, ?)",
            [
                ("a", STAGE, "10.1/to-fallback", MODEL, "refused", after.isoformat()),
                ("b", STAGE, "10.1/refused-before", FALLBACK_MODEL, "refused", before.isoformat()),
                ("c", STAGE, "10.1/refused-after", FALLBACK_MODEL, "refused", after.isoformat()),
                ("d", STAGE, "10.1/pending", MODEL, "pending", None),
                ("e", "relevance", "10.1/new", FALLBACK_MODEL, "refused", before.isoformat()),
            ],
        )

    def selected(since: datetime) -> dict[str, str]:
        with sqlite3.connect(db_path) as conn:
            refusals = stage_refusals(conn, STAGE, since)
        return {
            p["doi"]: refusals.model_for(p["doi"]) for p in select_papers(db_path, None, refusals)
        }

    assert selected(EVERY_REFUSAL) == {"10.1/new": MODEL, "10.1/to-fallback": FALLBACK_MODEL}
    assert selected(cutoff) == {
        "10.1/new": MODEL,
        "10.1/to-fallback": FALLBACK_MODEL,
        "10.1/refused-before": MODEL,
    }


def test_select_papers_skips_initial_papers_not_assessed_relevant(tmp_path: Path) -> None:
    """A downloaded initial paper is extracted only when assessed relevant.

    Expansion papers are never assessed for relevance and are always extracted.
    """
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type, download_status, "
            "relevance_assessment_json) VALUES (?, 't', 'pubmed', ?, 'downloaded', ?)",
            [
                ("10.1/relevant", "initial", json.dumps({"relevant": True})),
                ("10.1/not-relevant", "initial", json.dumps({"relevant": False})),
                ("10.1/unassessed", "initial", None),
                ("10.1/expansion", "expansion", None),
            ],
        )
        refusals = stage_refusals(conn, STAGE)

    assert [p["doi"] for p in select_papers(db_path, None, refusals)] == [
        "10.1/expansion",
        "10.1/relevant",
    ]
    assert count_remaining(db_path) == Remaining(due=2, outside_corpus=2)


def _message(stop_reason: str, content: list[dict[str, Any]], model: str) -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": (
                {"type": "refusal", "category": "bio", "explanation": None}
                if stop_reason == "refusal"
                else None
            ),
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )


class RoundTwoRefuser:
    """Round 1 asks for a gene lookup; round 2 is refused, by whichever model gets it."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, LlmRequest]] = []

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        results = []
        for request in requests:
            self.sent.append((round_no, request))
            model = request.params["model"]
            if round_no == 1:
                message = _message(
                    "tool_use",
                    [
                        {"type": "thinking", "thinking": "", "signature": f"sig-{model}"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup_genes",
                            "input": {"symbols": ["GENEA"]},
                        },
                    ],
                    model,
                )
                status = ResultStatus.SUCCEEDED
            else:
                message = _message("refusal", [], model)
                status = ResultStatus.REFUSED
            results.append(
                LlmResult(
                    custom_id=f"{stage}-{round_no}-{len(self.sent)}",
                    batch_id=None,
                    stage=stage,
                    subject=request.subject,
                    round=round_no,
                    model=model,
                    status=status,
                    message=message,
                    error_type=None,
                )
            )
        return results

    async def resume(self, stage: str) -> list[LlmResult]:
        return []


def test_a_refusal_in_round_two_restarts_the_conversation_on_the_fallback_model(
    tmp_path: Path, hgnc_resolver: HgncResolver
) -> None:
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
    schema = json.loads((SCHEMA_SQL.parent / "prompts/evidence_extraction_schema.json").read_text())
    transport = RoundTwoRefuser()
    first_message = {"role": "user", "content": "paper"}
    starts: list[list[tuple[int, str]]] = []  # llm_conversations right after each start

    def conversations() -> list[tuple[int, str]]:
        with sqlite3.connect(db_path) as conn:
            return [
                (round_no, json.loads(messages_json)[-1]["role"])
                for round_no, messages_json in conn.execute(
                    "SELECT round, messages_json FROM llm_conversations ORDER BY round"
                )
            ]

    async def attempt(runner: ExtractionRunner) -> RoundOutcome:
        with sqlite3.connect(db_path) as conn:
            refusals = stage_refusals(conn, STAGE)
            conversation = Conversation(
                doi="10.1/a",
                round=1,
                messages=[first_message],
                model=refusals.model_for("10.1/a"),
            )
            start_conversation(conn, conversation)
        starts.append(conversations())
        return await runner.advance([conversation])

    async def run() -> tuple[RoundOutcome, RoundOutcome]:
        variants = VariantLookupClient(
            VariantLookupSettings(
                VARIANT_LOOKUP_BASE_URL="http://unused.invalid", VARIANT_LOOKUP_API_KEY="x"
            )
        )
        try:
            runner = ExtractionRunner(
                transport=transport,
                db_path=db_path,
                papers_dir=tmp_path,
                settings=RequestSettings(
                    system="s",
                    output_config=json_output_config(schema, "medium"),
                    cache_pdf=False,
                ),
                schema=schema,
                hgnc_resolver=hgnc_resolver,
                lookups=LookupRunner(variants, hgnc_resolver),
            )
            first = await attempt(runner)
            second = await attempt(runner)
        finally:
            await variants.aclose()
        return first, second

    first, second = asyncio.run(run())
    assert (first.refused, first.to_fallback) == (1, 1)
    assert (second.refused, second.to_fallback) == (1, 0)
    sent = [(round_no, request.params["model"]) for round_no, request in transport.sent]
    assert sent == [(1, MODEL), (2, MODEL), (1, FALLBACK_MODEL), (2, FALLBACK_MODEL)]
    # The fallback conversation starts from the paper alone and replays only its own turn.
    fallback_round_two = json.loads(json.dumps(list(transport.sent[3][1].params["messages"])))
    assert fallback_round_two[0] == first_message
    assert fallback_round_two[1]["content"][0]["signature"] == f"sig-{FALLBACK_MODEL}"
    assert len(fallback_round_two) == 3
    # Starting the fallback conversation dropped the refused conversation's round 2.
    assert starts == [[(1, "user")], [(1, "user")]]
    with sqlite3.connect(db_path) as conn:
        refusals = stage_refusals(conn, STAGE)
    assert refusals.refused_for_good("10.1/a")


# ---------------------------------------------------------------------------
# Handling one round's results
# ---------------------------------------------------------------------------


class GatedVariantClient(VariantLookupClient):
    """Holds every lookup until *expected* lookups are in flight at once.

    Handling papers one at a time therefore times out. Released lookups finish
    in reverse order of arrival, so the first paper to ask is the last served.
    """

    def __init__(self, expected: int) -> None:
        self._expected = expected
        self._arrivals = 0
        self._all_in = asyncio.Event()

    async def lookup_one(self, body: dict[str, Any]) -> dict[str, Any]:
        self._arrivals += 1
        arrival = self._arrivals
        if arrival == self._expected:
            self._all_in.set()
        await asyncio.wait_for(self._all_in.wait(), timeout=2)
        await asyncio.sleep(0.01 * (self._expected - arrival))
        return {
            "normalized": [
                {
                    "hgvs_c": body["variant"],
                    "hgvs_p": None,
                    "pseudo_vcf": f"vcf:{body['variant']}",
                    "frequency": {"ac": 1, "an": 10},
                }
            ]
        }

    async def aclose(self) -> None:
        pass


class GroundedQuotes:
    """Stands in for PaperQuotes: every quote is in the PDF."""

    def __init__(self, pdf_bytes: bytes) -> None:
        pass

    def check(self, quotes: list[str]) -> QuoteCheck:
        return QuoteCheck(rejected=[], text_layer=True)

    def locate(self, quotes: list[str]) -> dict[str, list[dict[str, Any]]]:
        return {quote: [] for quote in quotes}


def _round_one_db(tmp_path: Path, dois: list[str]) -> Path:
    """A run database with each paper's round-1 conversation, and an empty PDF file each."""
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type, download_status) "
            "VALUES (?, 't', 'pubmed', 'initial', 'downloaded')",
            [(doi,) for doi in dois],
        )
        for doi in dois:
            start_conversation(
                conn,
                Conversation(
                    doi=doi, round=1, messages=[{"role": "user", "content": "paper"}], model=MODEL
                ),
            )
    for doi in dois:
        doi_to_path(doi, tmp_path, ".pdf").write_bytes(b"")
    return db_path


def _runner(
    tmp_path: Path, db_path: Path, hgnc_resolver: HgncResolver, variants: VariantLookupClient
) -> ExtractionRunner:
    schema = {"type": "object"}
    return ExtractionRunner(
        transport=RoundTwoRefuser(),
        db_path=db_path,
        papers_dir=tmp_path,
        settings=RequestSettings(
            system="s", output_config=json_output_config(schema, "medium"), cache_pdf=False
        ),
        schema=schema,
        hgnc_resolver=hgnc_resolver,
        lookups=LookupRunner(variants, hgnc_resolver),
    )


def _round_one_result(
    doi: str, status: ResultStatus, message: Message | None, error_type: str | None = None
) -> LlmResult:
    return LlmResult(
        custom_id=f"{STAGE}-1-{doi}",
        batch_id=None,
        stage=STAGE,
        subject=doi,
        round=1,
        model=MODEL,
        status=status,
        message=message,
        error_type=error_type,
    )


def _variant_lookup(variant: str) -> Message:
    return _message(
        "tool_use",
        [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "lookup_variants",
                "input": {
                    "variants": [
                        {"gene_symbol": "GENEA", "variant": variant, "genome_build": "GRCh38"}
                    ]
                },
            }
        ],
        MODEL,
    )


def test_a_round_handles_papers_concurrently_and_counts_every_outcome(
    tmp_path: Path, hgnc_resolver: HgncResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two papers look up a variant in round 1, and one more in its final answer.

    The gated client releases those three lookups only once all are in flight,
    so the round must be handled concurrently. The next round keeps result order.
    """
    monkeypatch.setattr("palit.extract_evidence.PaperQuotes", GroundedQuotes)
    entity = {
        **_entity(1, 1),
        "evidence_assessments": {
            name: {"result": False, "rationale": "r", "confidence": "LOW", "citations": []}
            for name in CRITERIA
        },
    }
    extraction = {"genome_build": "GRCh38", "gene_evaluations": [_gene("GENEA", [entity])]}
    results = [
        _round_one_result("10.1/a", ResultStatus.SUCCEEDED, _variant_lookup("c.1A>G")),
        _round_one_result("10.1/b", ResultStatus.SUCCEEDED, _variant_lookup("c.2A>G")),
        _round_one_result("10.1/c", ResultStatus.REFUSED, _message("refusal", [], MODEL)),
        _round_one_result("10.1/d", ResultStatus.ERRORED, None, "api_error"),
        _round_one_result(
            "10.1/e",
            ResultStatus.SUCCEEDED,
            _message("end_turn", [{"type": "text", "text": json.dumps(extraction)}], MODEL),
        ),
        _round_one_result("10.1/f", ResultStatus.SUCCEEDED, _message("max_tokens", [], MODEL)),
    ]
    assert MAX_CONCURRENT_PAPERS >= 3
    db_path = _round_one_db(tmp_path, [result.subject for result in results])

    async def run() -> RoundOutcome:
        runner = _runner(tmp_path, db_path, hgnc_resolver, GatedVariantClient(3))
        return await runner.handle(results)

    outcome = asyncio.run(run())
    assert (outcome.stored, outcome.refused, outcome.to_fallback, outcome.failed) == (1, 1, 1, 2)
    assert [(c.doi, c.round) for c in outcome.next_round] == [("10.1/a", 2), ("10.1/b", 2)]
    looked_up = [
        json.loads(c.messages[-1]["content"][0]["content"])["results"][0]["variant"]
        for c in outcome.next_round
    ]
    assert looked_up == ["c.1A>G", "c.2A>G"]
    with sqlite3.connect(db_path) as conn:
        recorded = conn.execute("SELECT subject, status FROM llm_requests ORDER BY subject")
        assert recorded.fetchall() == [
            ("10.1/a", "succeeded"),
            ("10.1/b", "succeeded"),
            ("10.1/c", "refused"),
            ("10.1/d", "errored"),
            ("10.1/e", "succeeded"),
            ("10.1/f", "succeeded"),
        ]
        rounds = conn.execute(
            "SELECT subject, round FROM llm_conversations WHERE round > 1 ORDER BY subject"
        )
        assert rounds.fetchall() == [("10.1/a", 2), ("10.1/b", 2)]
        extracted = conn.execute("SELECT doi FROM papers WHERE evidence_extraction_json NOT NULL")
        assert extracted.fetchall() == [("10.1/e",)]
        frequencies = conn.execute("SELECT paper_doi, variant_id FROM variant_frequencies")
        assert frequencies.fetchall() == [("10.1/e", "vcf:c.1A>G")]


def test_a_variant_lookup_failing_every_attempt_becomes_a_lookup_failed_result(
    tmp_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """The service answers 503 to every attempt; the paper still moves on to round 2."""
    attempts = 0

    def unavailable(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        return httpx2.Response(503)

    results = [_round_one_result("10.1/a", ResultStatus.SUCCEEDED, _variant_lookup("c.1A>G"))]
    db_path = _round_one_db(tmp_path, ["10.1/a"])

    async def run() -> RoundOutcome:
        variants = VariantLookupClient(
            VariantLookupSettings(
                VARIANT_LOOKUP_BASE_URL="http://unused.invalid", VARIANT_LOOKUP_API_KEY="x"
            ),
            retry_wait=tenacity.wait_none(),
            transport=httpx2.MockTransport(unavailable),
        )
        try:
            return await _runner(tmp_path, db_path, hgnc_resolver, variants).handle(results)
        finally:
            await variants.aclose()

    outcome = asyncio.run(run())
    assert attempts == MAX_ATTEMPTS
    (conversation,) = outcome.next_round
    tool_result = conversation.messages[-1]["content"][0]
    assert not tool_result["is_error"]
    (item,) = json.loads(tool_result["content"])["results"]
    assert (item["variant"], item["status"], item["error_code"]) == (
        "c.1A>G",
        "error",
        "LOOKUP_FAILED",
    )


def test_invalid_requests_raise_once_every_result_is_recorded(
    tmp_path: Path, hgnc_resolver: HgncResolver
) -> None:
    results = [
        _round_one_result("10.1/x", ResultStatus.ERRORED, None, "invalid_request_error"),
        _round_one_result("10.1/y", ResultStatus.REFUSED, _message("refusal", [], MODEL)),
    ]
    db_path = _round_one_db(tmp_path, [result.subject for result in results])

    async def run() -> RoundOutcome:
        runner = _runner(tmp_path, db_path, hgnc_resolver, GatedVariantClient(0))
        return await runner.handle(results)

    with pytest.raises(RuntimeError, match=r"1 extraction requests were invalid \(first: 10.1/x\)"):
        asyncio.run(run())
    with sqlite3.connect(db_path) as conn:
        recorded = conn.execute("SELECT subject, status FROM llm_requests ORDER BY subject")
        assert recorded.fetchall() == [("10.1/x", "errored"), ("10.1/y", "refused")]
