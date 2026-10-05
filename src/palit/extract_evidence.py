#!/usr/bin/env python3
"""Extract evidence from full-text PDFs with Claude, in a fixed two-round protocol.

Round 1: the model reads the PDF (native document input, figures included) and
requests every lookup it needs in one turn: ``lookup_genes`` and
``lookup_variants`` (see :mod:`palit.lookup_tools`). Round 2: the tool results
plus an instruction to finalise. Round 3 exists only as a backstop: any further
tool call is answered with an error. Every round goes out as one batch, and the
conversation behind each round is stored byte-for-byte in ``llm_conversations``,
because Opus 5.5 rejects replayed thinking blocks after an edited history.

A final answer is stored only when it passes the schema, the structural checks,
and the quote check (every quote verbatim in the PDF). Otherwise the paper stays
unextracted and the next attempt starts a fresh conversation.

An interrupted run loses no paid round: a restart handles the results of its
uncollected batches, then continues every saved conversation whose latest round
was never sent, before it starts any paper from round 1.

A conversation stays on the model it started on. When MODEL refuses a paper in
any round, the next attempt starts the paper's conversation afresh from round 1
on FALLBACK_MODEL: thinking blocks cannot be replayed to another model.
"""

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jsonschema
import pypdfium2 as pdfium
import typer
from anthropic import AsyncAnthropic
from anthropic.types import Message
from anthropic.types.output_config_param import OutputConfigParam
from jinja2 import Environment, FileSystemLoader
from rich.progress import TaskID

from palit.hgnc import HgncEntry, HgncResolver
from palit.llm import (
    EVERY_REFUSAL,
    BatchTransport,
    Effort,
    ImmediateTransport,
    LlmRequest,
    LlmResult,
    ResultStatus,
    StageRefusals,
    Transport,
    assistant_content,
    cached_system,
    json_output_config,
    log_refusal,
    make_client,
    parse_json_output,
    record_result,
    stage_refusals,
)
from palit.llm_usage import print_stage_summary
from palit.lookup_tools import (
    LOOKUP_VARIANTS_TOOL,
    MAX_IN_FLIGHT,
    TOOLS,
    LookupRunner,
    VariantLookupClient,
    VariantLookupSettings,
    frequency_row,
)
from palit.panelapp_client import PanelAppClient, format_panel_for_prompt
from palit.panelapp_integration import (
    criteria_object_to_list,
    validate_entities_criteria_complete,
    validate_independent_family_counts,
)
from palit.papers import doi_to_path
from palit.progress import LoggingProgress as Progress
from palit.quotes import PaperQuotes
from palit.run_corpus import CORPUS_PAPER

app = typer.Typer(help="Extract structured evidence from full-text PDFs")
logger = logging.getLogger(__name__)

STAGE = "extraction"
EFFORT: Effort = "medium"
MAX_TOKENS = 64000
LAST_ROUND = 3

# Files API: re-upload a PDF whose stored copy expires within this margin.
FILE_TTL = timedelta(days=30)
FILE_EXPIRY_MARGIN = timedelta(days=3)
MAX_CONCURRENT_UPLOADS = 8
# Claude's PDF input limit.
MAX_PDF_PAGES = 600

# Papers whose results one round handles at once. VariantLookupClient caps the
# requests in flight to the variant-lookup service at the service's own limit;
# this cap only bounds the papers waiting on it. Matching the two keeps the
# service busy even when each paper has a single variant to look up. Beyond
# that, more papers would only hold more PDF quote indexes in memory and
# delay the moment each paper's result is stored.
MAX_CONCURRENT_PAPERS = MAX_IN_FLIGHT

FINALISE_TEXT = (
    "These are all the lookup results. Write the final JSON answer now. "
    "No further lookups are available; any further tool call will return an error."
)
# Above this share of quotes not found in the PDF, the extraction is redone
# rather than pruned: the model was quoting from memory, not from the paper.
MAX_UNGROUNDED_QUOTE_SHARE = 0.25

BUDGET_EXHAUSTED_TEXT = (
    "Lookup budget exhausted. Write the final JSON answer now with the results you already have."
)


# ---------------------------------------------------------------------------
# Gene symbols
# ---------------------------------------------------------------------------


# Extraction fields that hold the paper's own text, left untouched by symbol
# replacement: quotes are checked verbatim against the PDF and keyed into
# citation_locations, titles of earlier papers are searched on PubMed, and the
# gene symbol and variant notation are the paper's.
VERBATIM_FIELDS = frozenset({"gene_symbol", "quote", "title", "variant"})


def symbol_pattern(symbols: list[str]) -> re.Pattern[str]:
    """Match any of ``symbols`` as a whole gene symbol, case-sensitively.

    A symbol is a run of word characters, '@' and hyphens. A match may not be
    preceded by any of these, so "ND1" stays put inside "MT-ND1". It may not be
    followed by a word character or '@', nor by a hyphen and a digit, so "AARS"
    stays put inside "AARS1" and "AARS-1". A hyphen and a letter may follow, so
    "AARS-related" matches.
    """
    alternatives = "|".join(re.escape(s) for s in sorted(symbols, key=len, reverse=True))
    return re.compile(rf"(?<![\w@-])(?:{alternatives})(?![\w@]|-\d)")


def _replace_symbols(node: Any, pattern: re.Pattern[str], replacements: dict[str, str]) -> Any:
    """Replace symbols in every string of ``node`` outside ``VERBATIM_FIELDS``."""
    if isinstance(node, str):
        return pattern.sub(lambda match: replacements[match.group()], node)
    if isinstance(node, list):
        return [_replace_symbols(item, pattern, replacements) for item in node]
    if isinstance(node, dict):
        return {
            key: value if key in VERBATIM_FIELDS else _replace_symbols(value, pattern, replacements)
            for key, value in node.items()
        }
    return node


def normalize_extraction_genes(
    parsed_json: dict[str, Any],
    hgnc_resolver: HgncResolver,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve each gene evaluation's symbol to HGNC and bring the prose up to date.

    Every gene evaluation's ``gene_symbol`` becomes ``paper_gene_symbol``: the
    symbol as the paper (and the model) wrote it, uppercased. Resolved
    evaluations gain ``hgnc_id``. Where a resolved symbol is not the current
    one, it is replaced by the current symbol in every string outside
    ``VERBATIM_FIELDS``, matching whole symbols only (see ``symbol_pattern``).

    Returns:
        (normalized_json, unresolved_symbols)
    """
    resolved: dict[str, HgncEntry] = {}  # uppercased paper symbol → entry
    unresolved: set[str] = set()
    for gene_eval in parsed_json["gene_evaluations"]:
        upper: str = gene_eval["gene_symbol"].upper()
        if upper not in resolved and upper not in unresolved:
            entry = hgnc_resolver.resolve(upper)
            if entry is not None:
                resolved[upper] = entry
            else:
                unresolved.add(upper)

    replacements = {old: entry.symbol for old, entry in resolved.items() if old != entry.symbol}
    if replacements:
        pattern = symbol_pattern(list(replacements))
        parsed_json = _replace_symbols(parsed_json, pattern, replacements)

    # Gene evaluations without hgnc_id are ignored by all downstream pipeline stages
    # (aggregation, reporting); they exist only for diagnostic inspection.
    for gene_eval in parsed_json["gene_evaluations"]:
        paper_symbol: str = gene_eval.pop("gene_symbol").upper()
        gene_eval["paper_gene_symbol"] = paper_symbol
        resolved_entry = resolved.get(paper_symbol)
        if resolved_entry is not None:
            gene_eval["hgnc_id"] = resolved_entry.hgnc_id

    return parsed_json, sorted(unresolved)


# ---------------------------------------------------------------------------
# Quotes in an extraction
# ---------------------------------------------------------------------------


def extraction_quotes(extraction: dict[str, Any]) -> list[str]:
    """Every quote an extraction cites: variants, entities, criteria, concerns."""
    quotes: list[str] = []
    for gene in extraction["gene_evaluations"]:
        quotes += [variant["quote"] for variant in gene["variants"]]
        for entity in gene["disease_entities"]:
            quotes += [citation["quote"] for citation in entity["citations"]]
            for assessment in entity["evidence_assessments"]:
                quotes += [citation["quote"] for citation in assessment["citations"]]
        for concern in gene["quality_concerns"]:
            quotes += [citation["quote"] for citation in concern["citations"]]
    return quotes


def prune_citations(extraction: dict[str, Any], rejected: set[str]) -> int:
    """Remove entity, criterion, and concern citations whose quote is in *rejected*.

    Variant quotes stay: a variant's notation is real even when its quote can't
    be placed (e.g. a row of a table printed as an image). Returns the number of
    citations removed.
    """
    removed = 0

    def keep(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal removed
        kept = [c for c in citations if c["quote"] not in rejected]
        removed += len(citations) - len(kept)
        return kept

    for gene in extraction["gene_evaluations"]:
        for entity in gene["disease_entities"]:
            entity["citations"] = keep(entity["citations"])
            for assessment in entity["evidence_assessments"]:
                assessment["citations"] = keep(assessment["citations"])
        for concern in gene["quality_concerns"]:
            concern["citations"] = keep(concern["citations"])
    return removed


def structural_problems(extraction: dict[str, Any]) -> list[str]:
    """Problems that send a paper back for another attempt (besides quotes)."""
    problems = []
    for gene in extraction["gene_evaluations"]:
        symbol = gene["gene_symbol"]
        if any(keyword in symbol.upper() for keyword in ("PLACEHOLDER", "DELETE")):
            problems.append(f"placeholder gene symbol {symbol}")
        if not validate_entities_criteria_complete(gene["disease_entities"]):
            problems.append(f"{symbol}: incomplete per-entity evidence_assessments")
        if not validate_independent_family_counts(gene["disease_entities"]):
            problems.append(f"{symbol}: inconsistent independent_family_count")
    return problems


# ---------------------------------------------------------------------------
# Requests and conversations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Conversation:
    """The input messages of one round for one paper, and the model the conversation is on."""

    doi: str
    round: int
    messages: list[dict[str, Any]]
    model: str


@dataclass(frozen=True)
class RequestSettings:
    system: str
    output_config: OutputConfigParam
    cache_pdf: bool  # immediate calls only: round 2 follows within seconds


def first_user_message(paper: dict[str, Any], file_id: str, cache_pdf: bool) -> dict[str, Any]:
    document: dict[str, Any] = {"type": "document", "source": {"type": "file", "file_id": file_id}}
    if cache_pdf:
        document["cache_control"] = {"type": "ephemeral"}
    text = (
        "PAPER TO EVALUATE (attached PDF)\n\n"
        f"Title: {paper['title']}\n\nDate: {paper['source_date']}\n\nAbstract: {paper['abstract']}"
    )
    return {"role": "user", "content": [document, {"type": "text", "text": text}]}


def build_request(conversation: Conversation, settings: RequestSettings) -> LlmRequest:
    return LlmRequest(
        subject=conversation.doi,
        params={
            "model": conversation.model,
            "max_tokens": MAX_TOKENS,
            "system": cached_system(settings.system),
            "tools": TOOLS,
            "messages": conversation.messages,  # type: ignore[typeddict-item]
            "output_config": settings.output_config,
        },
    )


def save_conversation(conn: sqlite3.Connection, conversation: Conversation) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO llm_conversations (stage, subject, round, messages_json)
        VALUES (?, ?, ?, ?)
        """,
        (STAGE, conversation.doi, conversation.round, json.dumps(conversation.messages)),
    )


def start_conversation(conn: sqlite3.Connection, conversation: Conversation) -> None:
    """Store a round-1 conversation in place of every round of the paper's earlier one."""
    conn.execute(
        "DELETE FROM llm_conversations WHERE stage = ? AND subject = ?", (STAGE, conversation.doi)
    )
    save_conversation(conn, conversation)


def load_conversation(conn: sqlite3.Connection, doi: str, round_no: int) -> list[dict[str, Any]]:
    (messages_json,) = conn.execute(
        "SELECT messages_json FROM llm_conversations WHERE stage = ? AND subject = ? AND round = ?",
        (STAGE, doi, round_no),
    ).fetchone()
    return json.loads(messages_json)  # type: ignore[no-any-return]


def variant_lookup_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every ``lookup_variants`` result item in a conversation's tool results."""
    variant_tool_ids = {
        block["id"]
        for message in messages
        if message["role"] == "assistant"
        for block in message["content"]
        if block["type"] == "tool_use" and block["name"] == LOOKUP_VARIANTS_TOOL["name"]
    }
    results: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] != "user" or isinstance(message["content"], str):
            continue
        for block in message["content"]:
            if block["type"] == "tool_result" and block["tool_use_id"] in variant_tool_ids:
                if not block.get("is_error"):
                    results += json.loads(block["content"])["results"]
    return results


# ---------------------------------------------------------------------------
# Paper selection and PDF upload
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnsentRound:
    """A paper's latest saved conversation round, for which no request was sent."""

    round: int
    model: str  # of the request that led to the round, which the conversation stays on


def unsent_rounds(conn: sqlite3.Connection) -> dict[str, UnsentRound]:
    """Every paper whose latest saved conversation round was never sent, by DOI.

    Round N + 1 is saved in the transaction that records the round-N request
    that led to it, which is the paper's latest round-N request: a later attempt
    starts with :func:`start_conversation`, which deletes the saved rounds. A
    round-N + 1 request sent after the save is pending or completed after that
    round-N request; one from an earlier attempt completed before it.

    A saved round 1 without a request is left to the next attempt, which starts
    the paper afresh: no request has been paid for in that conversation.

    Raises RuntimeError when the request that led to an unsent round is missing
    or did not end with a tool call: the two are written together.
    """
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT subject, MAX(round) AS round FROM llm_conversations
            WHERE stage = :stage
            GROUP BY subject
        ),
        led_by AS (
            SELECT l.subject, l.round, (
                SELECT r.custom_id FROM llm_requests r
                WHERE r.stage = :stage AND r.subject = l.subject AND r.round = l.round - 1
                  AND r.completed_at IS NOT NULL
                ORDER BY r.completed_at DESC
                LIMIT 1
            ) AS custom_id
            FROM latest l
            WHERE l.round > 1
        )
        SELECT b.subject, b.round, r.model, r.status, r.stop_reason
        FROM led_by b
        LEFT JOIN llm_requests r ON r.custom_id = b.custom_id
        WHERE NOT EXISTS (
            SELECT 1 FROM llm_requests s
            WHERE s.stage = :stage AND s.subject = b.subject AND s.round = b.round
              AND (s.completed_at IS NULL OR r.completed_at IS NULL
                   OR s.completed_at > r.completed_at)
        )
        """,
        {"stage": STAGE},
    ).fetchall()
    inconsistent = [
        subject
        for subject, _, _, status, stop_reason in rows
        if (status, stop_reason) != (ResultStatus.SUCCEEDED.value, "tool_use")
    ]
    if inconsistent:
        raise RuntimeError(
            f"{len(inconsistent)} saved extraction conversations (first: {inconsistent[0]}) "
            "follow no round that ended with a tool call; see llm_conversations"
        )
    return {
        subject: UnsentRound(round=round_no, model=model) for subject, round_no, model, _, _ in rows
    }


def _due_papers(
    conn: sqlite3.Connection, only: list[str] | None, refusals: StageRefusals
) -> list[dict[str, Any]]:
    """Downloaded corpus papers without an extraction, except ones in flight and ones refused for good.

    An initial paper is in the corpus only when assessed relevant (see :mod:`palit.run_corpus`).
    ``only`` restricts the result to these DOIs.
    """
    cursor = conn.cursor()
    cursor.row_factory = sqlite3.Row
    rows = cursor.execute(
        f"""
        SELECT doi, source_date, title, abstract
        FROM papers p
        WHERE download_status = 'downloaded'
          AND evidence_extraction_json IS NULL
          AND {CORPUS_PAPER}
          AND NOT EXISTS (
              SELECT 1 FROM llm_requests r
              WHERE r.stage = ? AND r.subject = p.doi AND r.status = 'pending'
          )
        ORDER BY doi
        """,
        (STAGE,),
    ).fetchall()
    papers = [dict(row) for row in rows if not refusals.refused_for_good(row["doi"])]
    if only is not None:
        wanted = set(only)
        papers = [paper for paper in papers if paper["doi"] in wanted]
    return papers


def select_papers(
    db_path: Path, only: list[str] | None, limit: int | None, refusals: StageRefusals
) -> list[dict[str, Any]]:
    """The papers due an extraction that start from round 1.

    These are the papers of :func:`_due_papers` without an unsent round (see
    :func:`unsent_rounds`), which :func:`select_unsent_conversations` continues.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        unsent = unsent_rounds(conn)
        papers = [
            paper for paper in _due_papers(conn, only, refusals) if paper["doi"] not in unsent
        ]
    return papers if limit is None else papers[:limit]


def select_unsent_conversations(
    db_path: Path, only: list[str] | None, limit: int | None, refusals: StageRefusals
) -> list[Conversation]:
    """The saved conversations of the papers due an extraction that have an unsent round.

    Each continues from its unsent round on the model it is on. ``only`` and
    ``limit`` restrict the papers as in :func:`select_papers`.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        unsent = unsent_rounds(conn)
        dois = [
            paper["doi"] for paper in _due_papers(conn, only, refusals) if paper["doi"] in unsent
        ]
        if limit is not None:
            dois = dois[:limit]
        return [
            Conversation(
                doi=doi,
                round=unsent[doi].round,
                messages=load_conversation(conn, doi, unsent[doi].round),
                model=unsent[doi].model,
            )
            for doi in dois
        ]


def unselected_reasons(db_path: Path, dois: list[str], refusals: StageRefusals) -> dict[str, str]:
    """The reason :func:`select_papers` leaves out each of *dois* that it leaves out."""
    with closing(sqlite3.connect(db_path)) as conn:
        unsent = unsent_rounds(conn)
        rows = conn.execute(
            f"""
            SELECT doi, download_status, evidence_extraction_json IS NOT NULL, {CORPUS_PAPER},
                EXISTS (
                    SELECT 1 FROM llm_requests r
                    WHERE r.stage = ? AND r.subject = p.doi AND r.status = 'pending'
                )
            FROM papers p
            WHERE doi IN ({",".join("?" * len(dois))})
            """,
            (STAGE, *dois),
        ).fetchall()
    known = {row[0]: row[1:] for row in rows}
    reasons: dict[str, str] = {}
    for doi in dois:
        flags = known.get(doi)
        if flags is None:
            reasons[doi] = "not in the database"
            continue
        download_status, extracted, in_corpus, pending = flags
        if download_status != "downloaded":
            reasons[doi] = f"not downloaded (download status {download_status or 'not set'})"
        elif extracted:
            reasons[doi] = "already extracted"
        elif not in_corpus:
            reasons[doi] = "an initial paper not assessed relevant"
        elif pending:
            reasons[doi] = "a request for it is still pending"
        elif refusals.refused_for_good(doi):
            reasons[doi] = "refused by both models (see --retry-refused)"
        elif (unsent_round := unsent.get(doi)) is not None:
            reasons[doi] = (
                f"its saved round-{unsent_round.round} conversation waits to be continued"
            )
    return reasons


async def upload_pdfs(
    client: AsyncAnthropic, db_path: Path, papers: list[dict[str, Any]], papers_dir: Path
) -> dict[str, str]:
    """Files API IDs for the papers' PDFs, uploading only new, changed, or expiring ones.

    Papers whose PDF is missing or has more pages than Claude accepts are skipped.
    """
    now = datetime.now(UTC)
    with sqlite3.connect(db_path) as conn:
        known = {
            doi: (file_id, sha256, datetime.fromisoformat(expires_at))
            for doi, file_id, sha256, expires_at in conn.execute(
                "SELECT doi, file_id, sha256, expires_at FROM uploaded_files"
            )
        }

    file_ids: dict[str, str] = {}
    to_upload: list[tuple[str, Path, str]] = []
    for paper in papers:
        doi = paper["doi"]
        pdf_path = doi_to_path(doi, papers_dir, ".pdf")
        if not pdf_path.exists():
            logger.warning("Missing PDF for %s: %s", doi, pdf_path)
            continue
        pages = len(pdfium.PdfDocument(pdf_path))
        if pages > MAX_PDF_PAGES:
            logger.warning(
                "Skipping %s: %d pages exceeds the %d-page limit", doi, pages, MAX_PDF_PAGES
            )
            continue
        sha256 = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        stored = known.get(doi)
        if stored is not None and stored[1] == sha256 and stored[2] > now + FILE_EXPIRY_MARGIN:
            file_ids[doi] = stored[0]
        else:
            to_upload.append((doi, pdf_path, sha256))

    if not to_upload:
        return file_ids

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)
    expires_at = (now + FILE_TTL).isoformat()

    async def upload(
        doi: str, pdf_path: Path, sha256: str, progress: Progress, task: TaskID
    ) -> None:
        async with semaphore:
            meta = await client.files.upload(
                file=(pdf_path.name, pdf_path.read_bytes(), "application/pdf"),
                expires_in_seconds=int(FILE_TTL.total_seconds()),
            )
        # Recorded as soon as it completes, so an interrupted run keeps its uploads.
        with _transaction(db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO uploaded_files
                    (doi, file_id, sha256, uploaded_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doi, meta.id, sha256, now.isoformat(), expires_at),
            )
        file_ids[doi] = meta.id
        progress.advance(task)

    logger.info("Uploading %d PDFs to the Files API", len(to_upload))
    with Progress() as progress:
        task = progress.add_task("Uploading PDFs", total=len(to_upload))
        await asyncio.gather(*(upload(*item, progress, task) for item in to_upload))
    return file_ids


# ---------------------------------------------------------------------------
# Result handling
# ---------------------------------------------------------------------------


@contextmanager
def _transaction(db_path: Path) -> Iterator[sqlite3.Connection]:
    """A connection that commits on success, rolls back on error, and is closed on exit.

    A round's results are handled concurrently on the event loop, so no
    connection may outlive the synchronous code that uses it: every
    transaction ends before the handler reaches its next ``await``.
    """
    with closing(sqlite3.connect(db_path)) as conn, conn:
        yield conn


@dataclass
class RoundOutcome:
    next_round: list[Conversation] = field(default_factory=list)
    stored: int = 0
    refused: int = 0
    to_fallback: int = 0  # the refusals by MODEL: these papers restart on FALLBACK_MODEL
    failed: int = 0

    @classmethod
    def combine(cls, outcomes: Iterable["RoundOutcome"]) -> "RoundOutcome":
        """The counts summed and the next-round conversations concatenated, in order."""
        total = cls()
        for outcome in outcomes:
            total.next_round += outcome.next_round
            total.stored += outcome.stored
            total.refused += outcome.refused
            total.to_fallback += outcome.to_fallback
            total.failed += outcome.failed
        return total


class ExtractionRunner:
    def __init__(
        self,
        *,
        transport: Transport,
        db_path: Path,
        papers_dir: Path,
        settings: RequestSettings,
        schema: dict[str, Any],
        hgnc_resolver: HgncResolver,
        lookups: LookupRunner,
    ) -> None:
        self._transport = transport
        self._db_path = db_path
        self._papers_dir = papers_dir
        self._settings = settings
        self._validator = jsonschema.Draft202012Validator(schema)
        self._hgnc = hgnc_resolver
        self._lookups = lookups

    async def advance(self, conversations: list[Conversation]) -> RoundOutcome:
        """Run conversations round by round until each has ended."""
        total = RoundOutcome()
        while conversations:
            by_round: dict[int, list[Conversation]] = defaultdict(list)
            for conversation in conversations:
                by_round[conversation.round].append(conversation)
            conversations = []
            for round_no in sorted(by_round):
                requests = [build_request(c, self._settings) for c in by_round[round_no]]
                results = await self._transport.run(STAGE, round_no, requests)
                outcome = await self.handle(results)
                total.stored += outcome.stored
                total.refused += outcome.refused
                total.to_fallback += outcome.to_fallback
                total.failed += outcome.failed
                conversations += outcome.next_round
        return total

    async def handle(self, results: list[LlmResult]) -> RoundOutcome:
        """Record every result and act on it, for up to MAX_CONCURRENT_PAPERS papers at once.

        The next round's conversations follow the order of *results*. Invalid
        requests are a bug: they raise once every result is recorded.
        """
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAPERS)

        with Progress() as progress:
            task = progress.add_task("Handling extraction results", total=len(results))

            async def handle_one(result: LlmResult) -> RoundOutcome:
                async with semaphore:
                    outcome = await self._handle_result(result)
                progress.advance(task)
                return outcome

            outcome = RoundOutcome.combine(await asyncio.gather(*map(handle_one, results)))
        invalid_requests = [
            result.subject for result in results if result.error_type == "invalid_request_error"
        ]
        if invalid_requests:
            raise RuntimeError(
                f"{len(invalid_requests)} extraction requests were invalid "
                f"(first: {invalid_requests[0]}); see llm_requests.error_type"
            )
        return outcome

    async def _handle_result(self, result: LlmResult) -> RoundOutcome:
        """Record *result* with its paper's next conversation or extraction, if any."""
        message = result.message
        if result.status == ResultStatus.REFUSED:
            log_refusal(result, f"{result.subject} in round {result.round}")
            self._record(result)
            return RoundOutcome(refused=1, to_fallback=int(result.goes_to_fallback))
        if result.status != ResultStatus.SUCCEEDED:
            self._record(result)
            return RoundOutcome(failed=1)
        if message is None:
            raise AssertionError("succeeded result without a message")
        if message.stop_reason == "tool_use" and result.round < LAST_ROUND:
            next_conversation = await self._next_round(result, self._messages(result), message)
            with _transaction(self._db_path) as conn:
                record_result(conn, result)
                save_conversation(conn, next_conversation)
            return RoundOutcome(next_round=[next_conversation])
        if message.stop_reason == "end_turn":
            if await self._store_final(result, self._messages(result), message):
                return RoundOutcome(stored=1)
            return RoundOutcome(failed=1)
        logger.warning(
            "%s round %d stopped with %s", result.subject, result.round, message.stop_reason
        )
        self._record(result)
        return RoundOutcome(failed=1)

    def _messages(self, result: LlmResult) -> list[dict[str, Any]]:
        """The input messages of the round *result* answers."""
        with _transaction(self._db_path) as conn:
            return load_conversation(conn, result.subject, result.round)

    def _record(self, result: LlmResult) -> None:
        with _transaction(self._db_path) as conn:
            record_result(conn, result)

    async def _next_round(
        self, result: LlmResult, messages: list[dict[str, Any]], message: Message
    ) -> Conversation:
        tool_uses = [block for block in message.content if block.type == "tool_use"]
        tool_results: list[dict[str, Any]] = []
        if result.round == 1:
            outcomes = await asyncio.gather(
                *(self._lookups.run(block.name, dict(block.input)) for block in tool_uses)
            )
            for block, tool_outcome in zip(tool_uses, outcomes, strict=True):
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_outcome.content,
                        "is_error": tool_outcome.is_error,
                    }
                )
            follow_up = FINALISE_TEXT
        else:
            logger.info(
                "%s asked for lookups in round %d; answering with errors",
                result.subject,
                result.round,
            )
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": BUDGET_EXHAUSTED_TEXT,
                    "is_error": True,
                }
                for block in tool_uses
            ]
            follow_up = BUDGET_EXHAUSTED_TEXT
        return Conversation(
            doi=result.subject,
            round=result.round + 1,
            messages=[
                *messages,
                {"role": "assistant", "content": assistant_content(message)},
                {"role": "user", "content": [*tool_results, {"type": "text", "text": follow_up}]},
            ],
            model=result.model,
        )

    async def _store_final(
        self, result: LlmResult, messages: list[dict[str, Any]], message: Message
    ) -> bool:
        """Validate the final answer and store it with its locations and frequencies."""
        doi = result.subject
        try:
            extraction = parse_json_output(message)
            self._validator.validate(extraction)
        except (ValueError, jsonschema.ValidationError) as e:
            logger.warning("Invalid extraction for %s: %s", doi, e)
            self._record(result)
            return False
        for gene in extraction["gene_evaluations"]:
            criteria_object_to_list(gene["disease_entities"])
        problems = structural_problems(extraction)
        pdf_quotes = PaperQuotes(doi_to_path(doi, self._papers_dir, ".pdf").read_bytes())
        quotes = extraction_quotes(extraction)
        check = pdf_quotes.check(quotes)
        if not check.text_layer:
            logger.warning(
                "%s has no usable text layer; quotes cannot be checked or highlighted", doi
            )
        if quotes and len(check.rejected) > MAX_UNGROUNDED_QUOTE_SHARE * len(quotes):
            problems.append(f"{len(check.rejected)} of {len(quotes)} quotes not found in the PDF")
        if problems:
            logger.warning("Rejected extraction for %s: %s", doi, "; ".join(problems[:5]))
            self._record(result)
            return False
        if check.rejected:
            removed = prune_citations(extraction, set(check.rejected))
            logger.info(
                "%s: dropped %d citations whose quote isn't in the PDF: %s",
                doi,
                removed,
                "; ".join(repr(q[:60]) for q in check.rejected[:3]),
            )
            quotes = extraction_quotes(extraction)

        locations = pdf_quotes.locate(quotes)
        frequencies = await self._frequency_rows(extraction, messages)
        normalized, unresolved = normalize_extraction_genes(extraction, self._hgnc)
        if unresolved:
            logger.warning("%s: unresolved gene symbols: %s", doi, unresolved)

        with _transaction(self._db_path) as conn:
            record_result(conn, result)
            _store_extraction(conn, doi, message, normalized, locations, frequencies, self._hgnc)
        return True

    async def _frequency_rows(
        self, extraction: dict[str, Any], messages: list[dict[str, Any]]
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """(paper gene symbol, variant, quote, lookup result) for every extracted variant.

        Variants the model extracted without looking them up are looked up here,
        so the report's variant table is complete.
        """
        looked_up = {
            (item["gene_symbol"].upper(), item["variant"]): item
            for item in variant_lookup_results(messages)
        }
        genome_build = {
            "GRCh38": "GRCh38",
            "hg38": "GRCh38",
            "GRCh37": "GRCh37",
            "hg19": "GRCh37",
        }.get(extraction["genome_build"], "unknown")
        rows: list[tuple[str, str, str, dict[str, Any]]] = []
        missing: list[tuple[str, str, str]] = []
        for gene in extraction["gene_evaluations"]:
            for variant in gene["variants"]:
                key = (gene["gene_symbol"].upper(), variant["variant"])
                item = looked_up.get(key)
                if item is None:
                    missing.append((gene["gene_symbol"], variant["variant"], variant["quote"]))
                else:
                    rows.append((gene["gene_symbol"], variant["variant"], variant["quote"], item))
        late = await asyncio.gather(
            *(
                self._lookups.lookup_variant(
                    {"gene_symbol": symbol, "variant": text, "genome_build": genome_build}
                )
                for symbol, text, _ in missing
            )
        )
        rows += [
            (symbol, text, quote, item)
            for (symbol, text, quote), item in zip(missing, late, strict=True)
        ]
        return rows


def _store_extraction(
    conn: sqlite3.Connection,
    doi: str,
    message: Message,
    extraction: dict[str, Any],
    locations: dict[str, list[dict[str, Any]]],
    frequencies: list[tuple[str, str, str, dict[str, Any]]],
    hgnc_resolver: HgncResolver,
) -> None:
    (source_type,) = conn.execute("SELECT source_type FROM papers WHERE doi = ?", (doi,)).fetchone()
    gene_source = {"initial": "recent_evidence", "expansion": "expansion_evidence"}[source_type]

    conn.execute(
        "UPDATE papers SET evidence_extraction_raw = ?, evidence_extraction_json = ? WHERE doi = ?",
        (message.to_json(), json.dumps(extraction), doi),
    )

    conn.execute("DELETE FROM gene_mentions WHERE paper_doi = ? AND source = ?", (doi, gene_source))
    for gene_eval in extraction["gene_evaluations"]:
        hgnc_id = gene_eval.get("hgnc_id")
        if hgnc_id is None or not gene_eval["disease_entities"]:
            continue  # unresolved gene, or mechanistic only
        conn.execute(
            """
            INSERT OR IGNORE INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source)
            VALUES (?, ?, ?, ?)
            """,
            (hgnc_id, gene_eval["paper_gene_symbol"], doi, gene_source),
        )

    conn.execute("DELETE FROM citation_locations WHERE paper_doi = ?", (doi,))
    conn.executemany(
        "INSERT INTO citation_locations (paper_doi, quote, bboxes_json) VALUES (?, ?, ?)",
        [(doi, quote, json.dumps(boxes)) for quote, boxes in locations.items()],
    )

    conn.execute("DELETE FROM variant_frequencies WHERE paper_doi = ?", (doi,))
    for symbol, _, quote, item in frequencies:
        entry = hgnc_resolver.resolve(symbol)
        if entry is None:
            continue
        row = frequency_row(item)
        conn.execute(
            """
            INSERT OR IGNORE INTO variant_frequencies
                (variant_id, hgnc_id, paper_doi, quote, normalization, gnomad)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row.variant_id,
                entry.hgnc_id,
                doi,
                quote,
                json.dumps(row.normalization),
                json.dumps(row.gnomad),
            ),
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Remaining:
    """Downloaded papers without an extraction."""

    due: int  # corpus papers, which extract-evidence extracts
    outside_corpus: int  # initial papers not assessed relevant, which it leaves out


def count_remaining(db_path: Path) -> Remaining:
    with sqlite3.connect(db_path) as conn:
        due, outside_corpus = conn.execute(
            f"""
            SELECT COALESCE(SUM({CORPUS_PAPER}), 0), COALESCE(SUM(NOT {CORPUS_PAPER}), 0)
            FROM papers p
            WHERE download_status = 'downloaded' AND evidence_extraction_json IS NULL
            """
        ).fetchone()
    return Remaining(due=int(due), outside_corpus=int(outside_corpus))


def log_remaining(db_path: Path) -> None:
    remaining = count_remaining(db_path)
    logger.info(
        "%s downloaded papers without an extraction; %s more are initial papers "
        "not assessed relevant, which are not extracted",
        f"{remaining.due:,}",
        f"{remaining.outside_corpus:,}",
    )


async def _process_evidence(
    *,
    client: AsyncAnthropic,
    runner: ExtractionRunner,
    transport: Transport,
    db_path: Path,
    papers_dir: Path,
    settings: RequestSettings,
    only: list[str] | None,
    limit: int | None,
    max_retries: int,
    refusals_since: datetime,
) -> None:
    """Extract every paper due an extraction, in attempts of whole conversations.

    First, the results of earlier batches are handled, and every saved
    conversation whose latest round was never sent is continued: the ones those
    results lead on to, and the ones an interrupted run saved but did not send.
    Then come the attempts, which start papers from round 1.

    ``only`` restricts the run to these DOIs; requested papers that are not due
    an extraction are logged with the reason. ``limit`` caps the papers this run
    sends requests for, continued conversations first. An attempt makes progress
    when it stores an extraction or sends a paper on to FALLBACK_MODEL, so a
    round of refusals by MODEL is followed by the attempt that restarts those
    papers on FALLBACK_MODEL.
    """
    resumed = await transport.resume(STAGE)
    if resumed:
        outcome = await runner.handle(resumed)
        logger.info(
            "Collected %d results from earlier batches (stored %d, refused %d, failed %d)",
            len(resumed),
            outcome.stored,
            outcome.refused,
            outcome.failed,
        )

    with sqlite3.connect(db_path) as conn:
        refusals = stage_refusals(conn, STAGE, refusals_since)
    unsent = select_unsent_conversations(db_path, only, limit, refusals)
    if unsent:
        logger.info("Continuing %d saved conversations from their unsent round", len(unsent))
        continued = await runner.advance(unsent)
        logger.info(
            "Continued conversations: stored %d, refused %d (%d go to the fallback model), "
            "failed %d",
            continued.stored,
            continued.refused,
            continued.to_fallback,
            continued.failed,
        )

    if only is not None:
        with sqlite3.connect(db_path) as conn:
            refusals = stage_refusals(conn, STAGE, refusals_since)
        for doi, reason in unselected_reasons(db_path, only, refusals).items():
            logger.warning("Not extracting %s: %s", doi, reason)

    if limit is not None:
        limit -= len(unsent)
        if limit == 0:
            logger.info("The continued conversations reached --limit; no new papers")
            return

    for attempt in range(1, max_retries + 1):
        with sqlite3.connect(db_path) as conn:
            refusals = stage_refusals(conn, STAGE, refusals_since)
        papers = select_papers(db_path, only, limit, refusals)
        if not papers:
            logger.info("No papers left to extract")
            return
        file_ids = await upload_pdfs(client, db_path, papers, papers_dir)
        conversations = [
            Conversation(
                doi=paper["doi"],
                round=1,
                messages=[first_user_message(paper, file_ids[paper["doi"]], settings.cache_pdf)],
                model=refusals.model_for(paper["doi"]),
            )
            for paper in papers
            if paper["doi"] in file_ids
        ]
        with sqlite3.connect(db_path) as conn:
            for conversation in conversations:
                start_conversation(conn, conversation)
        logger.info("Attempt %d: extracting %d papers", attempt, len(conversations))
        outcome = await runner.advance(conversations)
        logger.info(
            "Attempt %d: stored %d, refused %d (%d go to the fallback model), failed %d",
            attempt,
            outcome.stored,
            outcome.refused,
            outcome.to_fallback,
            outcome.failed,
        )
        if outcome.stored + outcome.to_fallback == 0:
            logger.error("No progress in attempt %d - stopping", attempt)
            return
        if limit is not None:
            return


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database",
    ),
    papers_dir: Path = typer.Option(
        Path("data/papers"),
        "--papers-dir",
        help="Directory containing the papers' PDFs",
    ),
    panel_date: str | None = typer.Option(
        None,
        "--panel-date",
        help="Panel state date (YYYY-MM-DD); required with --scope-panel-id",
    ),
    scope_panel_id: int | None = typer.Option(
        None,
        "--scope-panel-id",
        help="Panel ID for panel-scoped evidence extraction (only extracts genes relevant to this panel)",
    ),
    prompt_path: Path = typer.Option(
        Path("prompts/evidence_extraction_prompt.j2"),
        "--prompt-path",
        "-p",
        help="Path to Jinja2 prompt template file",
    ),
    schema_path: Path = typer.Option(
        Path("prompts/evidence_extraction_schema.json"),
        "--schema-path",
        "-s",
        help="Path to response schema file",
    ),
    immediate: bool = typer.Option(
        False,
        "--immediate",
        help="Send requests immediately instead of as Message Batches (for prompt development)",
    ),
    dois: list[str] | None = typer.Option(
        None,
        "--doi",
        help="Extract only this paper, if it is due an extraction (repeatable; for prompt "
        "development)",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Extract at most this many papers: saved conversations to continue first, then "
        "new papers in one attempt (for prompt development)",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of attempts for papers whose extraction failed, was rejected or "
        "was refused; the fallback-model conversation after a refusal is one of them",
    ),
    retry_refused: bool = typer.Option(
        False,
        "--retry-refused",
        help="Send papers that both models refused in earlier invocations again, once each, "
        "starting with the primary model (refusals vary between calls)",
    ),
) -> None:
    """Extract evidence from every downloaded paper that has no extraction yet."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)
    # With --retry-refused, only refusals during this invocation count.
    refusals_since = datetime.now(UTC) if retry_refused else EVERY_REFUSAL

    panel_formatted = ""
    if scope_panel_id is not None:
        if panel_date is None:
            logger.error("--scope-panel-id needs --panel-date")
            raise typer.Exit(1)
        panel_info = PanelAppClient(panel_date).get_panel_data(scope_panel_id)
        panel_formatted = format_panel_for_prompt(scope_panel_id, panel_info)
        logger.info(f"Panel-scoped extraction for panel {scope_panel_id}")

    env = Environment(loader=FileSystemLoader(prompt_path.parent), autoescape=False)
    system = env.get_template(prompt_path.name).render(panel_formatted=panel_formatted).strip()
    schema: dict[str, Any] = json.loads(schema_path.read_text())
    hgnc_resolver = HgncResolver.from_file()
    settings = RequestSettings(
        system=system, output_config=json_output_config(schema, EFFORT), cache_pdf=immediate
    )

    log_remaining(db_path)

    async def run() -> None:
        client = make_client()
        transport: Transport = (
            ImmediateTransport(client) if immediate else BatchTransport(client, db_path)
        )
        variant_client = VariantLookupClient(VariantLookupSettings())  # type: ignore[call-arg]
        try:
            runner = ExtractionRunner(
                transport=transport,
                db_path=db_path,
                papers_dir=papers_dir,
                settings=settings,
                schema=schema,
                hgnc_resolver=hgnc_resolver,
                lookups=LookupRunner(variant_client, hgnc_resolver),
            )
            await _process_evidence(
                client=client,
                runner=runner,
                transport=transport,
                db_path=db_path,
                papers_dir=papers_dir,
                settings=settings,
                only=dois,
                limit=limit,
                max_retries=max_retries,
                refusals_since=refusals_since,
            )
        finally:
            await variant_client.aclose()

    asyncio.run(run())
    log_remaining(db_path)
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
