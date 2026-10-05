"""Tests for extraction helpers that need no network or PDFs."""

import asyncio
import base64
import hashlib
import io
import json
import logging
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import pypdf
import pypdfium2 as pdfium
import pytest
import tenacity
from anthropic import AsyncAnthropic
from anthropic.types import Message
from pypdf.generic import DecodedStreamObject

from palit.extract_evidence import (
    CONTEXT_MARGIN_TOKENS,
    MAX_CONCURRENT_PAPERS,
    MAX_COUNTED_PAGES,
    MAX_PDF_PAGES,
    MAX_TOKENS,
    OVERSIZED_PAGE_TOKENS,
    PAGE_LIMIT_RULES_VERSION,
    STAGE,
    Conversation,
    ExtractionRunner,
    PageLimit,
    Remaining,
    RequestSettings,
    RoundOutcome,
    UploadedPdf,
    _process_evidence,
    count_remaining,
    drop_placeholder_citations,
    extraction_quotes,
    first_user_message,
    fitting_pages,
    load_conversation,
    normalize_extraction_genes,
    page_copy,
    save_conversation,
    select_papers,
    select_unsent_conversations,
    start_conversation,
    structural_problems,
    symbol_pattern,
    unselected_reasons,
    upload_pdfs,
    variant_lookup_results,
)
from palit.hgnc import HgncEntry, HgncResolver
from palit.llm import (
    CONTEXT_WINDOW,
    EVERY_REFUSAL,
    FALLBACK_MODEL,
    MODEL,
    LlmRequest,
    LlmResult,
    ResultStatus,
    StageRefusals,
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
from palit.quotes import PaperQuotes

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
    assert extraction_quotes(extraction) == {
        "variant quote",
        "entity quote",
        "criterion quote",
        "concern quote",
    }


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


def test_placeholder_citations_are_dropped_and_variants_kept() -> None:
    entity = _entity(2, 2)
    entity["citations"] += [
        {"quote": "", "commentary": ""},
        _citation("x"),
        _citation("Placeholder"),
        _citation(" pending "),
        _citation(":"),
        _citation("N/A"),
    ]
    entity["evidence_assessments"][0]["citations"] = [_citation("tbd"), _citation("criterion")]
    gene = _gene("ABC1", [entity])
    gene["variants"].append({"variant": "c.2A>G", "quote": "x"})
    extraction = {"gene_evaluations": [gene]}

    assert drop_placeholder_citations(extraction) == 7
    assert entity["citations"] == [_citation("entity quote")]
    assert entity["evidence_assessments"][0]["citations"] == [_citation("criterion")]
    assert [v["quote"] for v in gene["variants"]] == ["variant quote", "x"]


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
            p["doi"]: refusals.model_for(p["doi"])
            for p in select_papers(db_path, None, None, refusals, set())
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

    assert [p["doi"] for p in select_papers(db_path, None, None, refusals, set())] == [
        "10.1/expansion",
        "10.1/relevant",
    ]
    assert count_remaining(db_path) == Remaining(due=2, outside_corpus=2)


def test_select_papers_restricts_to_requested_dois_and_explains_the_rest(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "run.sqlite"
    relevant = json.dumps({"relevant": True})
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type, download_status, "
            "relevance_assessment_json, evidence_extraction_json) "
            "VALUES (?, 't', 'pubmed', ?, ?, ?, ?)",
            [
                ("10.1/due", "initial", "downloaded", relevant, None),
                ("10.1/other", "expansion", "downloaded", None, None),
                ("10.1/scheduled", "initial", "scheduled", relevant, None),
                ("10.1/extracted", "initial", "downloaded", relevant, "{}"),
                ("10.1/not-relevant", "initial", "downloaded", None, None),
                ("10.1/pending", "initial", "downloaded", relevant, None),
                ("10.1/refused", "initial", "downloaded", relevant, None),
            ],
        )
        conn.executemany(
            "INSERT INTO llm_requests (custom_id, stage, subject, round, model, status, "
            "completed_at) VALUES (?, ?, ?, 1, ?, ?, ?)",
            [
                ("a", STAGE, "10.1/pending", MODEL, "pending", None),
                ("b", STAGE, "10.1/refused", FALLBACK_MODEL, "refused", "2026-09-01T00:00:00"),
            ],
        )
        refusals = stage_refusals(conn, STAGE)
    requested = [
        "10.1/due",
        "10.1/missing",
        "10.1/scheduled",
        "10.1/extracted",
        "10.1/not-relevant",
        "10.1/pending",
        "10.1/refused",
    ]

    assert [p["doi"] for p in select_papers(db_path, requested, None, refusals, set())] == [
        "10.1/due"
    ]
    assert unselected_reasons(db_path, requested, refusals, {}) == {
        "10.1/missing": "not in the database",
        "10.1/scheduled": "not downloaded (download status scheduled)",
        "10.1/extracted": "already extracted",
        "10.1/not-relevant": "an initial paper not assessed relevant",
        "10.1/pending": "a request for it is still pending",
        "10.1/refused": "refused by both models (see --retry-refused)",
    }


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
                    error_message=None,
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


def test_a_doi_run_takes_only_that_paper_through_its_attempts(
    tmp_path: Path, hgnc_resolver: HgncResolver, caplog: pytest.LogCaptureFixture
) -> None:
    """--doi sends only the requested paper, on to the fallback model; others wait.

    A requested paper that is not due an extraction is logged with the reason.
    """
    document = pdfium.PdfDocument.new()
    document.new_page(595, 842)
    buffer = io.BytesIO()
    document.save(buffer)
    pdf = buffer.getvalue()
    expires_at = (datetime.now(UTC) + timedelta(days=30)).isoformat()
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type, download_status, "
            "evidence_extraction_json) VALUES (?, 't', 'pubmed', 'expansion', 'downloaded', ?)",
            [("10.1/a", None), ("10.1/b", None), ("10.1/done", "{}")],
        )
        for doi in ("10.1/a", "10.1/b"):
            doi_to_path(doi, tmp_path, ".pdf").write_bytes(pdf)
            conn.execute(
                "INSERT INTO uploaded_files (doi, file_id, sha256, uploaded_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (doi, f"file-{doi}", hashlib.sha256(pdf).hexdigest(), expires_at, expires_at),
            )
    schema = {"type": "object"}
    settings = RequestSettings(
        system="s", output_config=json_output_config(schema, "medium"), cache_pdf=False
    )
    transport = RoundTwoRefuser()

    async def run() -> None:
        variants = VariantLookupClient(
            VariantLookupSettings(
                VARIANT_LOOKUP_BASE_URL="http://unused.invalid", VARIANT_LOOKUP_API_KEY="x"
            )
        )
        try:
            await _process_evidence(
                client=AsyncAnthropic(api_key="unused"),
                runner=ExtractionRunner(
                    transport=transport,
                    db_path=db_path,
                    papers_dir=tmp_path,
                    settings=settings,
                    schema=schema,
                    hgnc_resolver=hgnc_resolver,
                    lookups=LookupRunner(variants, hgnc_resolver),
                ),
                transport=transport,
                db_path=db_path,
                papers_dir=tmp_path,
                settings=settings,
                only=["10.1/a", "10.1/done"],
                limit=None,
                max_retries=5,
                refusals_since=EVERY_REFUSAL,
            )
        finally:
            await variants.aclose()

    with caplog.at_level(logging.WARNING, logger="palit.extract_evidence"):
        asyncio.run(run())

    sent = [
        (round_no, request.subject, request.params["model"]) for round_no, request in transport.sent
    ]
    assert sent == [
        (1, "10.1/a", MODEL),
        (2, "10.1/a", MODEL),
        (1, "10.1/a", FALLBACK_MODEL),
        (2, "10.1/a", FALLBACK_MODEL),
    ]
    assert "Not extracting 10.1/done: already extracted" in caplog.messages


def _pdf(pages: int) -> bytes:
    document = pdfium.PdfDocument.new()
    for _ in range(pages):
        document.new_page(595, 842)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _uploaded_papers_db(tmp_path: Path, pdfs: dict[str, bytes]) -> Path:
    """A run database with these papers due an extraction, their PDFs already uploaded."""
    expires_at = (datetime.now(UTC) + timedelta(days=30)).isoformat()
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
        for doi, pdf in pdfs.items():
            conn.execute(
                "INSERT INTO papers (doi, title, source, source_type, download_status) "
                "VALUES (?, 't', 'pubmed', 'expansion', 'downloaded')",
                (doi,),
            )
            doi_to_path(doi, tmp_path, ".pdf").write_bytes(pdf)
            conn.execute(
                "INSERT INTO uploaded_files (doi, file_id, sha256, uploaded_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (doi, f"file-{doi}", hashlib.sha256(pdf).hexdigest(), expires_at, expires_at),
            )
    return db_path


def _run_process_evidence(
    tmp_path: Path,
    db_path: Path,
    hgnc_resolver: HgncResolver,
    client: Any,
    transport: RoundTwoRefuser,
    only: list[str] | None,
) -> None:
    schema = {"type": "object"}
    settings = RequestSettings(
        system="s", output_config=json_output_config(schema, "medium"), cache_pdf=False
    )

    async def run() -> None:
        variants = VariantLookupClient(
            VariantLookupSettings(
                VARIANT_LOOKUP_BASE_URL="http://unused.invalid", VARIANT_LOOKUP_API_KEY="x"
            )
        )
        try:
            await _process_evidence(
                client=client,
                runner=ExtractionRunner(
                    transport=transport,
                    db_path=db_path,
                    papers_dir=tmp_path,
                    settings=settings,
                    schema=schema,
                    hgnc_resolver=hgnc_resolver,
                    lookups=LookupRunner(variants, hgnc_resolver),
                ),
                transport=transport,
                db_path=db_path,
                papers_dir=tmp_path,
                settings=settings,
                only=only,
                limit=None,
                max_retries=5,
                refusals_since=EVERY_REFUSAL,
            )
        finally:
            await variants.aclose()

    asyncio.run(run())


class TooLongRejecter(RoundTwoRefuser):
    """Rejects every request about *subject* as too long; others as RoundTwoRefuser does."""

    def __init__(self, subject: str) -> None:
        super().__init__()
        self._subject = subject

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        rejected = [request for request in requests if request.subject == self._subject]
        results = await super().run(
            stage, round_no, [request for request in requests if request not in rejected]
        )
        for request in rejected:
            self.sent.append((round_no, request))
            results.append(
                LlmResult(
                    custom_id=f"{stage}-{round_no}-{len(self.sent)}",
                    batch_id="msgbatch_1",
                    stage=stage,
                    subject=request.subject,
                    round=round_no,
                    model=request.params["model"],
                    status=ResultStatus.ERRORED,
                    message=None,
                    error_type="invalid_request_error",
                    error_message="prompt is too long: 8328354 tokens > 1000000 maximum",
                )
            )
        return results


def test_a_paper_rejected_as_too_long_is_not_sent_again_in_the_invocation(
    tmp_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """The other paper's refusal leads to a second attempt, which leaves the rejected paper out."""
    db_path = _uploaded_papers_db(tmp_path, {"10.1/a": _pdf(1), "10.1/long": _pdf(1)})
    transport = TooLongRejecter("10.1/long")

    _run_process_evidence(tmp_path, db_path, hgnc_resolver, SimpleNamespace(), transport, only=None)

    sent = [(round_no, request.subject) for round_no, request in transport.sent]
    assert sent.count((1, "10.1/long")) == 1
    assert sent.count((1, "10.1/a")) == 2  # on MODEL, then on FALLBACK_MODEL
    with sqlite3.connect(db_path) as conn:
        statuses = conn.execute(
            "SELECT status, error_type FROM llm_requests WHERE subject = '10.1/long'"
        ).fetchall()
    assert statuses == [("errored", "invalid_request_error")]


# ---------------------------------------------------------------------------
# Truncating long PDFs to the pages that fit
# ---------------------------------------------------------------------------

WITHOUT_PDF_TOKENS = 30_000


def _numbered_pages_pdf(pages: int) -> bytes:
    """Blank pages, each 100 points plus its index wide, so a copy's pages can be told apart."""
    with closing(pdfium.PdfDocument.new()) as document:
        for index in range(pages):
            document.new_page(100 + index, 842)
        buffer = io.BytesIO()
        document.save(buffer)
    return buffer.getvalue()


class PageTokenCounter:
    """``client.messages.count_tokens`` for round-1 requests with a base64 PDF.

    The request without the PDF costs WITHOUT_PDF_TOKENS, and each page of the
    PDF what *page_tokens* gives for its index in the original PDF.
    """

    def __init__(self, page_tokens: Callable[[int], int], max_bytes: int) -> None:
        self._page_tokens = page_tokens
        self._max_bytes = max_bytes
        self.calls = 0

    async def count_tokens(self, **params: Any) -> SimpleNamespace:
        self.calls += 1
        assert "max_tokens" not in params
        assert {"model", "system", "tools", "output_config"} <= set(params)
        tokens = WITHOUT_PDF_TOKENS
        block = params["messages"][0]["content"][0]
        if block["type"] == "document":
            pdf = base64.standard_b64decode(block["source"]["data"])
            assert len(pdf) <= self._max_bytes
            with closing(pdfium.PdfDocument(pdf)) as document:
                assert len(document) <= MAX_COUNTED_PAGES
                tokens += sum(
                    self._page_tokens(round(document[i].get_width()) - 100)
                    for i in range(len(document))
                )
        return SimpleNamespace(input_tokens=tokens)


def _settings() -> RequestSettings:
    return RequestSettings(
        system="s", output_config=json_output_config({"type": "object"}, "medium"), cache_pdf=False
    )


def _article_with_supplement(index: int) -> int:
    """Ten article pages, then pages of dense supplementary tables."""
    return 5_000 if index < 10 else 50_000


# The pages of the PDF that fit beside the request without it.
ROOM = CONTEXT_WINDOW[MODEL] - MAX_TOKENS - CONTEXT_MARGIN_TOKENS - WITHOUT_PDF_TOKENS
ARTICLE_PAGES_THAT_FIT = 10 + (ROOM - 10 * 5_000) // 50_000


@pytest.mark.parametrize("narrow_ranges", [False, True])
def test_a_long_pdf_is_truncated_to_the_leading_pages_that_fit(
    monkeypatch: pytest.MonkeyPatch, narrow_ranges: bool
) -> None:
    """The page count is the largest that fits, also when ranges are narrowed to fit the byte cap."""
    pdf = _numbered_pages_pdf(120)
    max_bytes = 20_000_000
    if narrow_ranges:
        max_bytes = len(page_copy(pdf, 0, 10))
        monkeypatch.setattr("palit.extract_evidence.MAX_COUNTED_PDF_BYTES", max_bytes)
    counter = PageTokenCounter(_article_with_supplement, max_bytes)
    paper = {"doi": "10.1/long", "title": "t", "source_date": "2026-09-01", "abstract": "a"}

    limit = asyncio.run(
        fitting_pages(SimpleNamespace(messages=counter), paper, pdf, _settings(), MODEL)  # type: ignore[arg-type]
    )

    expected_tokens = WITHOUT_PDF_TOKENS + 10 * 5_000 + (ARTICLE_PAGES_THAT_FIT - 10) * 50_000
    assert limit == PageLimit(120, ARTICLE_PAGES_THAT_FIT, expected_tokens)
    assert counter.calls <= 2 + 120 // 10 + 4


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        (60, PageLimit(60, 60, WITHOUT_PDF_TOKENS + 60 * 5_000)),
        (
            MAX_PDF_PAGES + 1,
            PageLimit(MAX_PDF_PAGES + 1, MAX_PDF_PAGES, WITHOUT_PDF_TOKENS + MAX_PDF_PAGES * 100),
        ),
    ],
)
def test_a_pdf_that_fits_keeps_its_pages_up_to_the_api_page_limit(
    pages: int, expected: PageLimit
) -> None:
    pdf = _numbered_pages_pdf(pages)
    counter = PageTokenCounter(lambda index: 5_000 if pages == 60 else 100, 20_000_000)
    paper = {"doi": "10.1/long", "title": "t", "source_date": "2026-09-01", "abstract": "a"}

    limit = asyncio.run(
        fitting_pages(SimpleNamespace(messages=counter), paper, pdf, _settings(), MODEL)  # type: ignore[arg-type]
    )

    assert limit == expected


OVERSIZED_TEST_BYTES = 100_000


def _pdf_with_oversized_page(pages: int, oversized: int) -> bytes:
    """_numbered_pages_pdf whose page *oversized* alone is over OVERSIZED_TEST_BYTES."""
    writer = pypdf.PdfWriter(clone_from=io.BytesIO(_numbered_pages_pdf(pages)))
    content = DecodedStreamObject()
    content.set_data(b"% " + b"x" * 2 * OVERSIZED_TEST_BYTES + b"\n")
    writer.pages[oversized].replace_contents(content)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("page_tokens", "expected_pages", "expected_pdf_tokens"),
    [
        # The estimate fits, and so do all other pages.
        (lambda index: 2_000, 77, 76 * 2_000 + OVERSIZED_PAGE_TOKENS),
        # The estimate fits, then dense table pages fill the window.
        (
            lambda index: 5_000 if index < 29 else 50_000,
            29 + 1 + (ROOM - 29 * 5_000 - OVERSIZED_PAGE_TOKENS) // 50_000,
            29 * 5_000
            + OVERSIZED_PAGE_TOKENS
            + (ROOM - 29 * 5_000 - OVERSIZED_PAGE_TOKENS) // 50_000 * 50_000,
        ),
        # The pages before it leave less room than the estimate.
        (lambda index: 31_000, 29, 29 * 31_000),
    ],
)
def test_a_page_too_large_to_count_is_estimated(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    page_tokens: Callable[[int], int],
    expected_pages: int,
    expected_pdf_tokens: int,
) -> None:
    """Counting goes on after it; the PDF is truncated only where the window fills up."""
    pdf = _pdf_with_oversized_page(77, 29)
    monkeypatch.setattr("palit.extract_evidence.MAX_COUNTED_PDF_BYTES", OVERSIZED_TEST_BYTES)
    counter = PageTokenCounter(page_tokens, OVERSIZED_TEST_BYTES)
    paper = {"doi": "10.1/huge-figure", "title": "t", "source_date": "2026-09-01", "abstract": "a"}

    with caplog.at_level(logging.INFO, logger="palit.extract_evidence"):
        limit = asyncio.run(
            fitting_pages(SimpleNamespace(messages=counter), paper, pdf, _settings(), MODEL)  # type: ignore[arg-type]
        )

    assert limit == PageLimit(77, expected_pages, WITHOUT_PDF_TOKENS + expected_pdf_tokens)
    assert (
        "10.1/huge-figure: page 30 alone is over 100,000 bytes, too large to count; "
        f"taking it to cost {OVERSIZED_PAGE_TOKENS:,} tokens"
    ) in caplog.messages


def test_the_round_one_message_notes_a_truncated_pdf() -> None:
    paper = {"doi": "10.1/a", "title": "t", "source_date": "2026-09-01", "abstract": "a"}

    def text(pdf: UploadedPdf) -> str:
        return str(first_user_message(paper, pdf, cache_pdf=False)["content"][1]["text"])

    assert "first 38 of the paper's 298 pages" in text(UploadedPdf("file-1", 38, 298))
    assert "NOTE" not in text(UploadedPdf("file-1", 298, 298))


class FakeFiles:
    """``client.files``: records each upload's bytes."""

    def __init__(self) -> None:
        self.uploaded: list[bytes] = []

    async def upload(
        self, file: tuple[str, bytes, str], expires_in_seconds: int
    ) -> SimpleNamespace:
        self.uploaded.append(file[1])
        return SimpleNamespace(id=f"file-{len(self.uploaded)}")


def _upload_run(
    tmp_path: Path, pdfs: dict[str, bytes]
) -> tuple[Path, list[dict[str, Any]], StageRefusals]:
    """A run database with the papers of *pdfs*, by DOI, and their PDFs in *tmp_path*."""
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type, download_status) "
            "VALUES (?, 't', 'pubmed', 'expansion', 'downloaded')",
            [(doi,) for doi in pdfs],
        )
        refusals = stage_refusals(conn, STAGE)
    for doi, pdf in pdfs.items():
        doi_to_path(doi, tmp_path, ".pdf").write_bytes(pdf)
    papers = [
        {"doi": doi, "title": "t", "source_date": "2026-09-01", "abstract": "a"} for doi in pdfs
    ]
    return db_path, papers, refusals


def test_a_truncated_upload_is_counted_and_uploaded_once_per_pdf_version(tmp_path: Path) -> None:
    """A later run reuses the page count and the upload: same pages, same bytes, same sha256."""
    long_pdf = _numbered_pages_pdf(120)
    short_pdf = _numbered_pages_pdf(3)
    db_path, papers, refusals = _upload_run(
        tmp_path, {"10.1/long": long_pdf, "10.1/short": short_pdf}
    )
    counter = PageTokenCounter(_article_with_supplement, 20_000_000)
    files = FakeFiles()
    client = SimpleNamespace(messages=counter, files=files)

    def run() -> dict[str, UploadedPdf]:
        return asyncio.run(upload_pdfs(client, db_path, papers, tmp_path, _settings(), refusals))  # type: ignore[arg-type]

    first = run()
    calls = counter.calls
    second = run()

    assert first == second
    long_upload = first["10.1/long"]
    assert (long_upload.pages, long_upload.total_pages) == (ARTICLE_PAGES_THAT_FIT, 120)
    assert first["10.1/short"].pages == first["10.1/short"].total_pages == 3
    assert counter.calls == calls  # the second run counts nothing
    assert len(files.uploaded) == 2  # and uploads nothing
    truncated = files.uploaded[int(long_upload.file_id.removeprefix("file-")) - 1]
    with closing(pdfium.PdfDocument(truncated)) as document:
        assert len(document) == ARTICLE_PAGES_THAT_FIT
    with sqlite3.connect(db_path) as conn:
        uploaded = dict(conn.execute("SELECT doi, sha256 FROM uploaded_files"))
        limits = conn.execute("SELECT doi, sha256, total_pages, pages FROM pdf_page_limits")
        assert limits.fetchall() == [
            ("10.1/long", hashlib.sha256(long_pdf).hexdigest(), 120, ARTICLE_PAGES_THAT_FIT)
        ]
    assert uploaded == {
        "10.1/long": hashlib.sha256(truncated).hexdigest(),
        "10.1/short": hashlib.sha256(short_pdf).hexdigest(),
    }


def test_a_page_limit_counted_under_other_rules_is_counted_again(tmp_path: Path) -> None:
    """A stored limit of the same PDF version but another PAGE_LIMIT_RULES_VERSION is replaced."""
    long_pdf = _numbered_pages_pdf(120)
    sha256 = hashlib.sha256(long_pdf).hexdigest()
    db_path, papers, refusals = _upload_run(tmp_path, {"10.1/long": long_pdf})
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO pdf_page_limits "
            "(doi, sha256, rules_version, total_pages, pages, input_tokens, counted_at) "
            "VALUES ('10.1/long', ?, ?, 120, 7, 65000, '2026-10-05T00:00:00+00:00')",
            (sha256, PAGE_LIMIT_RULES_VERSION - 1),
        )
    counter = PageTokenCounter(_article_with_supplement, 20_000_000)
    client = SimpleNamespace(messages=counter, files=FakeFiles())

    uploads = asyncio.run(upload_pdfs(client, db_path, papers, tmp_path, _settings(), refusals))  # type: ignore[arg-type]

    assert counter.calls > 0
    assert uploads["10.1/long"].pages == ARTICLE_PAGES_THAT_FIT
    with sqlite3.connect(db_path) as conn:
        limits = conn.execute("SELECT sha256, rules_version, pages FROM pdf_page_limits")
        assert limits.fetchall() == [(sha256, PAGE_LIMIT_RULES_VERSION, ARTICLE_PAGES_THAT_FIT)]


def test_quotes_from_the_kept_pages_locate_in_the_original_pdf(
    text_pdf: Callable[[list[str]], bytes],
) -> None:
    """Extraction locates quotes in the local PDF, whose leading pages the upload keeps."""
    filler = [f"Line {i} of the cohort description for unrelated probands." for i in range(25)]
    page_quotes = [
        "c.1061C>T p.T354M in proband 1",
        "c.6752C>T in exon 31",
        "Table S2 row 4012",
    ]
    with closing(pdfium.PdfDocument.new()) as original:
        for quote in page_quotes:
            with closing(pdfium.PdfDocument(text_pdf([*filler, quote]))) as page:
                original.import_pages(page)
        buffer = io.BytesIO()
        original.save(buffer)
        original_pdf = buffer.getvalue()
    kept = page_copy(original_pdf, 0, 2)

    with (
        closing(pdfium.PdfDocument(kept)) as truncated,
        closing(pdfium.PdfDocument(original_pdf)) as source,
    ):
        assert [truncated[i].get_textpage().get_text_range() for i in range(2)] == [
            source[i].get_textpage().get_text_range() for i in range(2)
        ]
    locations = PaperQuotes(original_pdf).locate(page_quotes[:2])
    assert [{box["page"] for box in locations[quote]} for quote in page_quotes[:2]] == [{1}, {2}]


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
    """Stands in for PaperQuotes: every quote not in *unlocated* is placed on page 1."""

    can_locate = True
    unlocated: frozenset[str] = frozenset()

    def __init__(self, pdf_bytes: bytes) -> None:
        pass

    def locate(self, quotes: list[str]) -> dict[str, list[dict[str, Any]]]:
        box = {"page": 1, "top": 0, "left": 0, "bottom": 10, "right": 10}
        return {quote: [] if quote in self.unlocated else [box] for quote in quotes}


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
    doi: str,
    status: ResultStatus,
    message: Message | None,
    error_type: str | None = None,
    error_message: str | None = None,
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
        error_message=error_message,
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
        _round_one_result("10.1/d", ResultStatus.ERRORED, None, "api_error", "Internal error"),
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


def _criteria_with_citations() -> dict[str, Any]:
    return {
        name: {
            "result": False,
            "rationale": "r",
            "confidence": "LOW",
            "citations": [_citation(f"{name} quote")],
        }
        for name in CRITERIA
    }


def test_unlocated_quotes_are_stored_and_only_placeholders_dropped(
    tmp_path: Path,
    hgnc_resolver: HgncResolver,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No quote the PDF can place: the extraction is stored with every real citation."""
    entity = {**_entity(1, 1), "evidence_assessments": _criteria_with_citations()}
    entity["citations"] += [{"quote": "", "commentary": ""}, _citation("x")]
    extraction = {"genome_build": "GRCh38", "gene_evaluations": [_gene("GENEA", [entity])]}
    quotes = {
        "variant quote",
        "entity quote",
        "concern quote",
        *(f"{name} quote" for name in CRITERIA),
    }
    monkeypatch.setattr(GroundedQuotes, "unlocated", frozenset(quotes))
    monkeypatch.setattr("palit.extract_evidence.PaperQuotes", GroundedQuotes)
    final = _message("end_turn", [{"type": "text", "text": json.dumps(extraction)}], MODEL)
    result = _round_one_result("10.1/e", ResultStatus.SUCCEEDED, final)
    db_path = _round_one_db(tmp_path, ["10.1/e"])
    runner = _runner(tmp_path, db_path, hgnc_resolver, RecordingVariantClient(set()))

    with caplog.at_level(logging.INFO, logger="palit.extract_evidence"):
        outcome = asyncio.run(runner.handle([result]))

    assert outcome.stored == 1
    with sqlite3.connect(db_path) as conn:
        (stored,) = conn.execute("SELECT evidence_extraction_json FROM papers").fetchone()
        locations = dict(conn.execute("SELECT quote, bboxes_json FROM citation_locations"))
        (rejection,) = conn.execute("SELECT rejection FROM llm_requests").fetchone()
    gene = json.loads(stored)["gene_evaluations"][0]
    assert gene["disease_entities"][0]["citations"] == [_citation("entity quote")]
    assert all(
        len(c["citations"]) == 1 for c in gene["disease_entities"][0]["evidence_assessments"]
    )
    assert gene["quality_concerns"][0]["citations"] == [_citation("concern quote")]
    assert locations == dict.fromkeys(quotes, "[]")
    assert rejection is None
    assert (
        "10.1/e: 8 quotes; 2 placeholder citations dropped; not located in the PDF: "
        "1 variant quotes, 7 citation quotes"
    ) in caplog.messages


def test_a_rejected_extraction_records_its_reason(
    tmp_path: Path, hgnc_resolver: HgncResolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("palit.extract_evidence.PaperQuotes", GroundedQuotes)
    inconsistent = {
        "genome_build": "GRCh38",
        "gene_evaluations": [
            _gene("GENEA", [{**_entity(1, 3), "evidence_assessments": _criteria_with_citations()}])
        ],
    }
    results = [
        _round_one_result(
            "10.1/a",
            ResultStatus.SUCCEEDED,
            _message("end_turn", [{"type": "text", "text": "{not json"}], MODEL),
        ),
        _round_one_result(
            "10.1/b",
            ResultStatus.SUCCEEDED,
            _message("end_turn", [{"type": "text", "text": json.dumps(inconsistent)}], MODEL),
        ),
    ]
    db_path = _round_one_db(tmp_path, ["10.1/a", "10.1/b"])
    runner = _runner(tmp_path, db_path, hgnc_resolver, RecordingVariantClient(set()))

    outcome = asyncio.run(runner.handle(results))

    assert outcome.failed == 2
    with sqlite3.connect(db_path) as conn:
        rejections = dict(conn.execute("SELECT subject, rejection FROM llm_requests"))
        (extracted,) = conn.execute(
            "SELECT COUNT(*) FROM papers WHERE evidence_extraction_json IS NOT NULL"
        ).fetchone()
    assert extracted == 0
    assert rejections["10.1/a"].startswith("invalid JSON: ")
    assert rejections["10.1/b"] == "structural: GENEA: inconsistent independent_family_count"


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


TOO_LONG_MESSAGE = "prompt is too long: 8328354 tokens > 1000000 maximum"


def test_invalid_requests_raise_once_every_result_is_recorded(
    tmp_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A request rejected as too long does not count as invalid; any other rejection does."""
    invalid = "invalid_request_error"
    results = [
        _round_one_result("10.1/long", ResultStatus.ERRORED, None, invalid, TOO_LONG_MESSAGE),
        _round_one_result("10.1/x", ResultStatus.ERRORED, None, invalid, "tools.0: bad schema"),
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
        assert recorded.fetchall() == [
            ("10.1/long", "errored"),
            ("10.1/x", "errored"),
            ("10.1/y", "refused"),
        ]


def test_a_request_rejected_as_too_long_is_recorded_without_raising(
    tmp_path: Path, hgnc_resolver: HgncResolver, caplog: pytest.LogCaptureFixture
) -> None:
    results = [
        _round_one_result(
            "10.1/long", ResultStatus.ERRORED, None, "invalid_request_error", TOO_LONG_MESSAGE
        )
    ]
    db_path = _round_one_db(tmp_path, ["10.1/long"])

    async def run() -> RoundOutcome:
        runner = _runner(tmp_path, db_path, hgnc_resolver, GatedVariantClient(0))
        return await runner.handle(results)

    with caplog.at_level(logging.WARNING, logger="palit.extract_evidence"):
        outcome = asyncio.run(run())
    assert outcome.too_long == {"10.1/long": 8_328_354}
    assert (outcome.stored, outcome.refused, outcome.failed) == (0, 0, 0)
    assert any("10.1/long" in m and "8,328,354 input tokens" in m for m in caplog.messages)
    with sqlite3.connect(db_path) as conn:
        recorded = conn.execute("SELECT subject, status, error_type FROM llm_requests")
        assert recorded.fetchall() == [("10.1/long", "errored", "invalid_request_error")]


# ---------------------------------------------------------------------------
# Continuing saved conversations after an interruption
# ---------------------------------------------------------------------------


def test_unsent_rounds_are_the_latest_saved_rounds_no_request_followed(tmp_path: Path) -> None:
    """A round counts as sent only by a request after the one that led to it.

    A saved round 1 alone starts afresh. Papers with an unsent round are continued
    on their conversation's model, not started from round 1.
    """
    db_path = tmp_path / "run.sqlite"
    conversations = {
        "10.1/unsent": 2,  # an earlier attempt's round 2 errored before this round 1
        "10.1/sent": 2,
        "10.1/in-flight": 2,
        "10.1/fresh": 1,
        "10.1/fallback": 2,
        "10.1/round3": 3,
    }
    requests = [
        ("10.1/unsent", 2, MODEL, "errored", None, "2026-10-01T00:00:01+00:00"),
        ("10.1/unsent", 1, MODEL, "succeeded", "tool_use", "2026-10-01T00:00:02+00:00"),
        ("10.1/sent", 1, MODEL, "succeeded", "tool_use", "2026-10-01T00:00:01+00:00"),
        ("10.1/sent", 2, MODEL, "succeeded", "max_tokens", "2026-10-01T00:00:02+00:00"),
        ("10.1/in-flight", 1, MODEL, "succeeded", "tool_use", "2026-10-01T00:00:01+00:00"),
        ("10.1/in-flight", 2, MODEL, "pending", None, None),
        ("10.1/fallback", 2, MODEL, "refused", "refusal", "2026-10-01T00:00:01+00:00"),
        ("10.1/fallback", 1, FALLBACK_MODEL, "succeeded", "tool_use", "2026-10-01T00:00:02+00:00"),
        ("10.1/round3", 1, MODEL, "succeeded", "tool_use", "2026-10-01T00:00:01+00:00"),
        ("10.1/round3", 2, MODEL, "succeeded", "tool_use", "2026-10-01T00:00:02+00:00"),
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type, download_status) "
            "VALUES (?, 't', 'pubmed', 'expansion', 'downloaded')",
            [(doi,) for doi in conversations],
        )
        conn.executemany(
            "INSERT INTO llm_conversations (stage, subject, round, messages_json) "
            "VALUES (?, ?, ?, ?)",
            [
                (
                    STAGE,
                    doi,
                    round_no,
                    json.dumps([{"role": "user", "content": f"round {round_no}"}]),
                )
                for doi, latest in conversations.items()
                for round_no in range(1, latest + 1)
            ],
        )
        conn.executemany(
            "INSERT INTO llm_requests (custom_id, stage, subject, round, model, status, "
            "stop_reason, completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(f"r{i}", STAGE, *request) for i, request in enumerate(requests)],
        )
        refusals = stage_refusals(conn, STAGE)

    assert [p["doi"] for p in select_papers(db_path, None, None, refusals, set())] == [
        "10.1/fresh",
        "10.1/sent",
    ]
    continued = select_unsent_conversations(db_path, None, None, refusals)
    assert [(c.doi, c.round, c.model, c.messages) for c in continued] == [
        ("10.1/fallback", 2, FALLBACK_MODEL, [{"role": "user", "content": "round 2"}]),
        ("10.1/round3", 3, MODEL, [{"role": "user", "content": "round 3"}]),
        ("10.1/unsent", 2, MODEL, [{"role": "user", "content": "round 2"}]),
    ]
    assert [c.doi for c in select_unsent_conversations(db_path, None, 1, refusals)] == [
        "10.1/fallback"
    ]
    only = ["10.1/unsent", "10.1/in-flight"]
    assert [c.doi for c in select_unsent_conversations(db_path, only, None, refusals)] == [
        "10.1/unsent"
    ]
    assert select_papers(db_path, only, None, refusals, set()) == []
    assert unselected_reasons(db_path, only, refusals, {}) == {
        "10.1/unsent": "its saved round-2 conversation waits to be continued",
        "10.1/in-flight": "a request for it is still pending",
    }

    # A saved round after a round that did not end in a tool call breaks the protocol.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO llm_conversations (stage, subject, round, messages_json) "
            "VALUES (?, '10.1/fresh', 2, '[]')",
            (STAGE,),
        )
        conn.execute(
            "INSERT INTO llm_requests (custom_id, stage, subject, round, model, status, "
            "stop_reason, completed_at) VALUES ('broken', ?, '10.1/fresh', 1, ?, 'refused', "
            "'refusal', '2026-10-01T00:00:01+00:00')",
            (STAGE, MODEL),
        )
    with pytest.raises(
        RuntimeError, match=r"1 saved extraction conversations \(first: 10.1/fresh\)"
    ):
        select_papers(db_path, None, None, refusals, set())


class ScriptedBatches:
    """Keeps pending requests in ``llm_requests`` as BatchTransport does, and answers them.

    Round 1 looks up the paper's variant; round 2 is the final answer.
    """

    def __init__(self, db_path: Path, variants: dict[str, str], final_answer: str) -> None:
        self._db_path = db_path
        self._variants = variants
        self._final_answer = final_answer
        self.sent: list[tuple[int, str, str]] = []  # (round, subject, model)
        self.requests: list[LlmRequest] = []

    async def run(
        self, stage: str, round_no: int, requests: Sequence[LlmRequest]
    ) -> list[LlmResult]:
        rows = [
            (f"{stage}-{round_no}-{request.subject}", request.subject, request.params["model"])
            for request in requests
        ]
        with sqlite3.connect(self._db_path) as conn:
            conn.executemany(
                "INSERT INTO llm_requests (custom_id, stage, subject, round, model, status) "
                "VALUES (?, ?, ?, ?, ?, 'pending')",
                [
                    (custom_id, stage, subject, round_no, model)
                    for custom_id, subject, model in rows
                ],
            )
        self.sent += [(round_no, subject, model) for _, subject, model in rows]
        self.requests += requests
        return [self._result(stage, round_no, *row) for row in rows]

    async def resume(self, stage: str) -> list[LlmResult]:
        with sqlite3.connect(self._db_path) as conn:
            pending = conn.execute(
                "SELECT round, custom_id, subject, model FROM llm_requests "
                "WHERE stage = ? AND status = 'pending' ORDER BY custom_id",
                (stage,),
            ).fetchall()
        return [self._result(stage, *row) for row in pending]

    def _result(
        self, stage: str, round_no: int, custom_id: str, subject: str, model: str
    ) -> LlmResult:
        message = (
            _variant_lookup(self._variants[subject])
            if round_no == 1
            else _message("end_turn", [{"type": "text", "text": self._final_answer}], model)
        )
        return LlmResult(
            custom_id=custom_id,
            batch_id=None,
            stage=stage,
            subject=subject,
            round=round_no,
            model=model,
            status=ResultStatus.SUCCEEDED,
            message=message,
            error_type=None,
            error_message=None,
        )


class HangingVariantClient(VariantLookupClient):
    """Answers every variant lookup, except that the ones for *hanging* never return."""

    def __init__(self, hanging: set[str]) -> None:
        self._hanging = hanging

    async def lookup_one(self, body: dict[str, Any]) -> dict[str, Any]:
        if body["variant"] in self._hanging:
            await asyncio.Event().wait()
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


def test_a_restart_continues_the_conversations_an_interrupted_run_saved(
    tmp_path: Path,
    hgnc_resolver: HgncResolver,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Handling round 1 stops while one paper's lookup hangs; a restart pays no round 1 again.

    The two papers handled before the stop continue from their saved round 2, and
    the third from the resumed batch, all in one round-2 batch.
    """
    monkeypatch.setattr("palit.extract_evidence.PaperQuotes", GroundedQuotes)
    document = pdfium.PdfDocument.new()
    document.new_page(595, 842)
    buffer = io.BytesIO()
    document.save(buffer)
    pdf = buffer.getvalue()
    expires_at = (datetime.now(UTC) + timedelta(days=30)).isoformat()
    variants = {"10.1/a": "c.1A>G", "10.1/b": "c.2A>G", "10.1/c": "c.3A>G"}
    db_path = tmp_path / "run.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL.read_text())
        for doi in variants:
            conn.execute(
                "INSERT INTO papers (doi, title, source, source_type, download_status) "
                "VALUES (?, 't', 'pubmed', 'expansion', 'downloaded')",
                (doi,),
            )
            doi_to_path(doi, tmp_path, ".pdf").write_bytes(pdf)
            conn.execute(
                "INSERT INTO uploaded_files (doi, file_id, sha256, uploaded_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (doi, f"file-{doi}", hashlib.sha256(pdf).hexdigest(), expires_at, expires_at),
            )
    entity = {
        **_entity(1, 1),
        "evidence_assessments": {
            name: {"result": False, "rationale": "r", "confidence": "LOW", "citations": []}
            for name in CRITERIA
        },
    }
    final_answer = json.dumps(
        {"genome_build": "GRCh38", "gene_evaluations": [_gene("GENEA", [entity])]}
    )
    schema = {"type": "object"}
    settings = RequestSettings(
        system="s", output_config=json_output_config(schema, "medium"), cache_pdf=False
    )

    async def invocation(transport: ScriptedBatches, lookups: VariantLookupClient) -> None:
        await _process_evidence(
            client=AsyncAnthropic(api_key="unused"),
            runner=ExtractionRunner(
                transport=transport,
                db_path=db_path,
                papers_dir=tmp_path,
                settings=settings,
                schema=schema,
                hgnc_resolver=hgnc_resolver,
                lookups=LookupRunner(lookups, hgnc_resolver),
            ),
            transport=transport,
            db_path=db_path,
            papers_dir=tmp_path,
            settings=settings,
            only=None,
            limit=None,
            max_retries=5,
            refusals_since=EVERY_REFUSAL,
        )

    interrupted = ScriptedBatches(db_path, variants, final_answer)

    async def interrupted_run() -> None:
        await asyncio.wait_for(
            invocation(interrupted, HangingVariantClient({"c.2A>G"})), timeout=0.5
        )

    with pytest.raises(TimeoutError):
        asyncio.run(interrupted_run())
    assert interrupted.sent == [(1, doi, MODEL) for doi in variants]
    with sqlite3.connect(db_path) as conn:
        statuses = conn.execute("SELECT subject, status FROM llm_requests ORDER BY subject")
        assert statuses.fetchall() == [
            ("10.1/a", "succeeded"),
            ("10.1/b", "pending"),
            ("10.1/c", "succeeded"),
        ]

    restarted = ScriptedBatches(db_path, variants, final_answer)
    with caplog.at_level(logging.INFO, logger="palit.extract_evidence"):
        asyncio.run(invocation(restarted, HangingVariantClient(set())))

    assert restarted.sent == [(2, doi, MODEL) for doi in variants]
    assert "Continuing 3 saved conversations from their unsent round" in caplog.messages
    with sqlite3.connect(db_path) as conn:
        extracted = conn.execute(
            "SELECT doi FROM papers WHERE evidence_extraction_json IS NOT NULL ORDER BY doi"
        )
        assert [doi for (doi,) in extracted] == list(variants)
        rounds = conn.execute(
            "SELECT round, status, COUNT(*) FROM llm_requests GROUP BY round, status"
        )
        assert rounds.fetchall() == [(1, "succeeded", 3), (2, "succeeded", 3)]
    # Each round-2 request carries the paper's own round-1 lookup.
    assert {
        request.subject: variant_lookup_results(
            json.loads(json.dumps(list(request.params["messages"])))
        )[0]["variant"]
        for request in restarted.requests
    } == variants


def test_a_retried_variant_lookup_is_logged_with_its_gene_and_variant(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts = 0

    def time_out_once(request: httpx2.Request) -> httpx2.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx2.ReadTimeout("timed out", request=request)
        return httpx2.Response(200, json={"normalized": []})

    async def run() -> dict[str, Any]:
        client = VariantLookupClient(
            VariantLookupSettings(
                VARIANT_LOOKUP_BASE_URL="http://unused.invalid", VARIANT_LOOKUP_API_KEY="x"
            ),
            retry_wait=tenacity.wait_none(),
            transport=httpx2.MockTransport(time_out_once),
        )
        try:
            return await client.lookup_one({"gene": "GATA2", "variant": "c.1061C>T"})
        finally:
            await client.aclose()

    with caplog.at_level(logging.WARNING, logger="palit.lookup_tools"):
        assert asyncio.run(run()) == {"normalized": []}
    assert caplog.messages == [
        f"Variant lookup GATA2 c.1061C>T: ReadTimeout; attempt 2/{MAX_ATTEMPTS} in 0.0 s"
    ]


class RecordingVariantClient(VariantLookupClient):
    """Answers every variant lookup, except that the ones for *timing_out* time out upstream."""

    def __init__(self, timing_out: set[str]) -> None:
        self._timing_out = timing_out
        self.bodies: list[dict[str, Any]] = []

    async def lookup_one(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        if body["variant"] in self._timing_out:
            return {"error": {"code": "VV_UPSTREAM_TIMEOUT", "message": "timed out"}}
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


def _lookup_result(
    variant: str, error_code: str | None, genome_build: str = "GRCh38"
) -> dict[str, Any]:
    """A summarised ``lookup_variants`` item: an error with *error_code*, or a success."""
    item = {"gene_symbol": "GENEA", "variant": variant, "genome_build": genome_build}
    if error_code is None:
        response: dict[str, Any] = {
            "normalized": [
                {
                    "hgvs_c": variant,
                    "hgvs_p": None,
                    "pseudo_vcf": f"old:{variant}",
                    "frequency": None,
                }
            ]
        }
    else:
        response = {"error": {"code": error_code, "message": "m"}}
    return summarise_variant_response(item, response)


def _lookup_round(tool_id: str, results: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """An assistant turn that looks up variants, and the user turn with their results."""
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "t", "signature": "sig"},
                {"type": "tool_use", "id": tool_id, "name": "lookup_variants", "input": {}},
                {"type": "tool_use", "id": f"{tool_id}-genes", "name": "lookup_genes", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps({"results": results}),
                    "is_error": False,
                },
                {
                    "type": "tool_result",
                    "tool_use_id": f"{tool_id}-genes",
                    "content": json.dumps({"results": [{"symbol": "GENEA", "status": "ok"}]}),
                    "is_error": False,
                },
                {"type": "text", "text": text},
            ],
        },
    ]


def test_transiently_failed_lookups_of_an_unsent_round_are_run_again_and_saved(
    tmp_path: Path, hgnc_resolver: HgncResolver, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the unsent round's transient failures are looked up again; one still fails.

    The round-3 conversation's transient failure is in round 2's results, which
    a request has carried, so it stays as it is.
    """
    db_path = _round_one_db(tmp_path, ["10.1/a", "10.1/sent"])
    # The model wrote genome_build before gene_symbol; the rerun keeps that order.
    timeout = summarise_variant_response(
        {"genome_build": "GRCh37", "gene_symbol": "GENEA", "variant": "c.1A>G"},
        {"error": {"code": "VV_UPSTREAM_TIMEOUT", "message": "timed out"}},
    )
    unsent_results = [
        timeout,
        _lookup_result("c.2A>G", None),
        _lookup_result("c.3A>G", "NORMALIZATION_ESEQUENCEMISMATCH"),
        _lookup_result("c.4A>G", "LOOKUP_FAILED"),
    ]
    unsent = Conversation(
        doi="10.1/a",
        round=2,
        messages=[
            {"role": "user", "content": "paper"},
            *_lookup_round("toolu_1", unsent_results, "finalise"),
        ],
        model=MODEL,
    )
    sent_round_two = _lookup_round(
        "toolu_2", [_lookup_result("c.5A>G", "VV_UPSTREAM_TIMEOUT")], "finalise"
    )
    budget_exhausted = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_3", "name": "lookup_variants", "input": {}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_3",
                    "content": "budget exhausted",
                    "is_error": True,
                },
                {"type": "text", "text": "budget exhausted"},
            ],
        },
    ]
    round_three = Conversation(
        doi="10.1/sent",
        round=3,
        messages=[{"role": "user", "content": "paper"}, *sent_round_two, *budget_exhausted],
        model=MODEL,
    )
    original = json.dumps([unsent.messages, round_three.messages])
    with sqlite3.connect(db_path) as conn:
        save_conversation(conn, unsent)
        save_conversation(conn, round_three)
    variants = RecordingVariantClient(timing_out={"c.4A>G"})

    async def run() -> list[Conversation]:
        runner = _runner(tmp_path, db_path, hgnc_resolver, variants)
        return await runner.rerun_transient_lookups([unsent, round_three])

    with caplog.at_level(logging.INFO, logger="palit.extract_evidence"):
        rerun, untouched = asyncio.run(run())

    assert json.dumps([unsent.messages, round_three.messages]) == original
    assert variants.bodies == [
        {"gene": "GENEA", "variant": "c.1A>G", "genome_build": "GRCh37"},
        {"gene": "GENEA", "variant": "c.4A>G", "genome_build": "GRCh38"},
    ]
    assert untouched is round_three
    assert rerun.messages[:-1] == unsent.messages[:-1]
    variant_block, genes_block, text_block = rerun.messages[-1]["content"]
    assert (genes_block, text_block) == tuple(unsent.messages[-1]["content"][1:])
    refreshed, ok, nomenclature, still_failing = json.loads(variant_block["content"])["results"]
    assert list(refreshed) == ["genome_build", "gene_symbol", "variant", "status", "candidates"]
    assert refreshed["candidates"][0]["vcf"] == "vcf:c.1A>G"
    assert (ok, nomenclature) == (unsent_results[1], unsent_results[2])
    assert (still_failing["status"], still_failing["error_code"]) == (
        "error",
        "VV_UPSTREAM_TIMEOUT",
    )
    assert variant_block == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": json.dumps({"results": [refreshed, ok, nomenclature, still_failing]}),
        "is_error": False,
    }
    with sqlite3.connect(db_path) as conn:
        assert load_conversation(conn, "10.1/a", 2) == rerun.messages
        assert load_conversation(conn, "10.1/sent", 3) == round_three.messages
    assert (
        "Re-ran 2 variant lookups that had failed for a reason that can pass, in 1 "
        "conversations: 1 still failed that way"
    ) in caplog.messages


def test_frequency_rows_look_up_transient_failures_again(
    tmp_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A transient failure is looked up again with the model's input; other results stay."""
    messages = [
        {"role": "user", "content": "paper"},
        *_lookup_round(
            "toolu_1",
            [
                _lookup_result("c.1A>G", None),
                _lookup_result("c.2A>G", "NO_GENOMIC_COORDS"),
                _lookup_result("c.3A>G", "VV_UPSTREAM_TIMEOUT", genome_build="GRCh37"),
            ],
            "finalise",
        ),
    ]
    gene = _gene("GENEA", [])
    gene["variants"] = [
        {"variant": variant, "quote": f"quote {variant}"}
        for variant in ("c.1A>G", "c.2A>G", "c.3A>G", "c.4A>G")
    ]
    extraction = {"genome_build": "GRCh38", "gene_evaluations": [gene]}
    variants = RecordingVariantClient(timing_out=set())

    async def run() -> list[tuple[str, str, str, dict[str, Any]]]:
        runner = _runner(tmp_path, tmp_path / "unused.sqlite", hgnc_resolver, variants)
        return await runner._frequency_rows(extraction, messages)

    rows = asyncio.run(run())
    assert variants.bodies == [
        {"gene": "GENEA", "variant": "c.3A>G", "genome_build": "GRCh37"},
        {"gene": "GENEA", "variant": "c.4A>G", "genome_build": "GRCh38"},
    ]
    by_variant = {variant: item for _, variant, _, item in rows}
    assert by_variant["c.1A>G"]["candidates"][0]["vcf"] == "old:c.1A>G"
    assert by_variant["c.2A>G"]["error_code"] == "NO_GENOMIC_COORDS"
    assert by_variant["c.3A>G"]["candidates"][0]["vcf"] == "vcf:c.3A>G"
    assert by_variant["c.4A>G"]["candidates"][0]["vcf"] == "vcf:c.4A>G"
    assert {quote for _, _, quote, _ in rows} == {f"quote c.{n}A>G" for n in range(1, 5)}
