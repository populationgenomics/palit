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
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
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

from palit.hgnc import HgncEntry, HgncResolver
from palit.llm import (
    MODEL,
    AnthropicSettings,
    BatchTransport,
    Effort,
    ImmediateTransport,
    LlmRequest,
    LlmResult,
    ResultStatus,
    Transport,
    cached_system,
    json_output_config,
    make_client,
    parse_json_output,
    record_result,
)
from palit.llm_usage import print_stage_summary
from palit.lookup_tools import (
    LOOKUP_VARIANTS_TOOL,
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
from palit.quotes import PaperQuotes

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


def normalize_extraction_genes(
    parsed_json: dict[str, Any],
    hgnc_resolver: HgncResolver,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize gene symbols in extraction JSON to current HGNC.

    For each unique gene_symbol in gene_evaluations:
    - Resolve via HgncResolver
    - If resolved and symbol changed: replace old symbol with current symbol
      throughout the entire JSON (summary, variant descriptions, etc.)
    - Resolved: add hgnc_id
    - Always: rename gene_symbol → paper_gene_symbol (uppercased)

    Returns:
        (normalized_json, unresolved_symbols)
    """
    resolved: dict[str, HgncEntry] = {}  # uppercased raw symbol → entry
    unresolved: set[str] = set()
    for gene_eval in parsed_json.get("gene_evaluations", []):
        upper: str = gene_eval["gene_symbol"].upper()
        if upper not in resolved and upper not in unresolved:
            entry = hgnc_resolver.resolve(upper)
            if entry is not None:
                resolved[upper] = entry
            else:
                unresolved.add(upper)

    # Full string replacement across serialized JSON for changed symbols
    replacements = {old: entry.symbol for old, entry in resolved.items() if old != entry.symbol}
    if replacements:
        json_str = json.dumps(parsed_json)
        for old in sorted(replacements, key=len, reverse=True):
            json_str = json_str.replace(old, replacements[old])
        parsed_json = json.loads(json_str)

    # Rewrite gene_evaluations: add hgnc_id if resolved, always rename gene_symbol.
    # Gene evaluations without hgnc_id are ignored by all downstream pipeline stages
    # (aggregation, reporting) — they exist only for diagnostic inspection.
    by_current_symbol: dict[str, int] = {entry.symbol: entry.hgnc_id for entry in resolved.values()}
    for gene_eval in parsed_json.get("gene_evaluations", []):
        gene_symbol: str = gene_eval.pop("gene_symbol")
        gene_eval["paper_gene_symbol"] = gene_symbol.upper()
        hgnc_id = by_current_symbol.get(gene_symbol)
        if hgnc_id is not None:
            gene_eval["hgnc_id"] = hgnc_id

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

    Variant quotes stay: a variant's notation is real even when its quote (often
    a short table cell) can't be placed. Returns the number of citations removed.
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
    """The input messages of one round for one paper."""

    doi: str
    round: int
    messages: list[dict[str, Any]]


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
            "model": MODEL,
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


def load_conversation(conn: sqlite3.Connection, doi: str, round_no: int) -> list[dict[str, Any]]:
    (messages_json,) = conn.execute(
        "SELECT messages_json FROM llm_conversations WHERE stage = ? AND subject = ? AND round = ?",
        (STAGE, doi, round_no),
    ).fetchone()
    return json.loads(messages_json)  # type: ignore[no-any-return]


def assistant_content(message: Message) -> list[dict[str, Any]]:
    """The assistant turn exactly as returned, for replay in the next round.

    Only top-level ``None`` fields are dropped; nested values (tool inputs,
    thinking signatures) are kept verbatim.
    """
    return [
        {key: value for key, value in block.model_dump(mode="json").items() if value is not None}
        for block in message.content
    ]


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


def select_papers(db_path: Path, limit: int | None) -> list[dict[str, Any]]:
    """Downloaded papers without an extraction, excluding refused ones and ones in flight."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT doi, source_date, title, abstract
            FROM papers p
            WHERE download_status = 'downloaded'
              AND evidence_extraction_json IS NULL
              AND NOT EXISTS (
                  SELECT 1 FROM llm_requests r
                  WHERE r.stage = ? AND r.subject = p.doi AND r.status IN ('refused', 'pending')
              )
            ORDER BY doi
            LIMIT ?
            """,
            (STAGE, -1 if limit is None else limit),
        ).fetchall()
    return [dict(row) for row in rows]


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

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_UPLOADS)

    async def upload(doi: str, pdf_path: Path, sha256: str) -> tuple[str, str, str]:
        async with semaphore:
            meta = await client.files.upload(
                file=(pdf_path.name, pdf_path.read_bytes(), "application/pdf"),
                expires_in_seconds=int(FILE_TTL.total_seconds()),
            )
        return doi, meta.id, sha256

    if to_upload:
        logger.info("Uploading %d PDFs to the Files API", len(to_upload))
    uploaded = await asyncio.gather(*(upload(*item) for item in to_upload))
    expires_at = (now + FILE_TTL).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO uploaded_files (doi, file_id, sha256, uploaded_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (doi, file_id, sha256, now.isoformat(), expires_at)
                for doi, file_id, sha256 in uploaded
            ],
        )
    file_ids.update({doi: file_id for doi, file_id, _ in uploaded})
    return file_ids


# ---------------------------------------------------------------------------
# Result handling
# ---------------------------------------------------------------------------


@dataclass
class RoundOutcome:
    next_round: list[Conversation]
    stored: int = 0
    refused: int = 0
    failed: int = 0


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
        total = RoundOutcome(next_round=[])
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
                total.failed += outcome.failed
                conversations += outcome.next_round
        return total

    async def handle(self, results: list[LlmResult]) -> RoundOutcome:
        outcome = RoundOutcome(next_round=[])
        invalid_requests: list[str] = []
        for result in results:
            with sqlite3.connect(self._db_path) as conn:
                messages = load_conversation(conn, result.subject, result.round)
            message = result.message
            if result.status == ResultStatus.REFUSED:
                assert message is not None
                outcome.refused += 1
                logger.warning(
                    "Refused: %s (%s)",
                    result.subject,
                    message.stop_details.category if message.stop_details else "no category",
                )
                self._record(result)
            elif result.status != ResultStatus.SUCCEEDED:
                outcome.failed += 1
                if result.error_type == "invalid_request_error":
                    invalid_requests.append(result.subject)
                self._record(result)
            elif message is None:
                raise AssertionError("succeeded result without a message")
            elif message.stop_reason == "tool_use" and result.round < LAST_ROUND:
                next_conversation = await self._next_round(result, messages, message)
                with sqlite3.connect(self._db_path) as conn:
                    record_result(conn, result)
                    save_conversation(conn, next_conversation)
                outcome.next_round.append(next_conversation)
            elif message.stop_reason == "end_turn":
                if await self._store_final(result, messages, message):
                    outcome.stored += 1
                else:
                    outcome.failed += 1
            else:
                logger.warning(
                    "%s round %d stopped with %s", result.subject, result.round, message.stop_reason
                )
                outcome.failed += 1
                self._record(result)
        if invalid_requests:
            raise RuntimeError(
                f"{len(invalid_requests)} extraction requests were invalid "
                f"(first: {invalid_requests[0]}); see llm_requests.error_type"
            )
        return outcome

    def _record(self, result: LlmResult) -> None:
        with sqlite3.connect(self._db_path) as conn:
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

        with sqlite3.connect(self._db_path) as conn:
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


def count_remaining(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        (remaining,) = conn.execute(
            """
            SELECT COUNT(*) FROM papers
            WHERE download_status = 'downloaded' AND evidence_extraction_json IS NULL
            """
        ).fetchone()
    return int(remaining)


async def _process_evidence(
    *,
    client: AsyncAnthropic,
    runner: ExtractionRunner,
    transport: Transport,
    db_path: Path,
    papers_dir: Path,
    settings: RequestSettings,
    limit: int | None,
    max_retries: int,
) -> None:
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
        continued = await runner.advance(outcome.next_round)
        logger.info("Finished resumed conversations: stored %d", continued.stored)

    for attempt in range(1, max_retries + 1):
        papers = select_papers(db_path, limit)
        if not papers:
            logger.info("No papers left to extract")
            return
        file_ids = await upload_pdfs(client, db_path, papers, papers_dir)
        conversations = [
            Conversation(
                doi=paper["doi"],
                round=1,
                messages=[first_user_message(paper, file_ids[paper["doi"]], settings.cache_pdf)],
            )
            for paper in papers
            if paper["doi"] in file_ids
        ]
        with sqlite3.connect(db_path) as conn:
            for conversation in conversations:
                save_conversation(conn, conversation)
        logger.info("Attempt %d: extracting %d papers", attempt, len(conversations))
        outcome = await runner.advance(conversations)
        logger.info(
            "Attempt %d: stored %d, refused %d, failed %d",
            attempt,
            outcome.stored,
            outcome.refused,
            outcome.failed,
        )
        if outcome.stored == 0:
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
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Extract at most this many papers, in one attempt (for prompt development)",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of attempts for papers whose extraction failed or was rejected",
    ),
) -> None:
    """Extract evidence from every downloaded paper that has no extraction yet."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

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

    logger.info(f"{count_remaining(db_path):,} downloaded papers without an extraction")

    async def run() -> None:
        client = make_client(AnthropicSettings())
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
                limit=limit,
                max_retries=max_retries,
            )
        finally:
            await variant_client.aclose()

    asyncio.run(run())
    logger.info(f"{count_remaining(db_path):,} downloaded papers still without an extraction")
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
