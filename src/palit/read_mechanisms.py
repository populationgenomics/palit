#!/usr/bin/env python3
"""Read each paper's PDF for the evidence it holds on the mechanism of disease.

One request per paper that an association cites: the system part states the
mechanisms and the evidence to record, and is the same for all requests; the user
part carries the paper's PDF and the associations that cite it, each with its gene,
disease and mode of inheritance. The answer lists, per association, the paper's
evidence items and the authors' own conclusion. It is stored in
``mechanism_readings``, in the transaction that records the request. A paper MODEL
refused goes to FALLBACK_MODEL in the next attempt (see :mod:`palit.llm`).

A reading names the associations it covers by their ids. An aggregation run gives
associations new ids, so a paper is read again when the associations that cite it
are not the ones its reading covers.

assess-mechanisms judges each association's mechanisms from these readings (see
:mod:`palit.mechanisms`).
"""

import asyncio
import json
import logging
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import typer
from anthropic import AsyncAnthropic
from anthropic.types.output_config_param import OutputConfigParam
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from palit.extract_evidence import RequestSettings, UploadedPdf, paper_text, upload_pdfs
from palit.hgnc import HgncResolver
from palit.llm import (
    ALL_TIME,
    Effort,
    ImmediateTransport,
    LlmRequest,
    LlmResult,
    ResultStatus,
    StageHistory,
    Transport,
    cached_system,
    invalid_answer_reason,
    json_output_config,
    log_failed_for_good,
    log_refusal,
    make_client,
    parse_json_output,
    record_result,
    stage_history,
)
from palit.llm_usage import print_stage_summary
from palit.match_panels import format_inheritance
from palit.mechanisms import readings_from_json
from palit.progress import LoggingProgress as Progress

app = typer.Typer(help="Read papers for the mechanism-of-disease evidence they hold")
logger = logging.getLogger(__name__)

STAGE = "read_mechanisms"
EFFORT: Effort = "medium"
MAX_TOKENS = 32000
# Each request carries a whole PDF, so few run at a time
MAX_CONCURRENT_PAPERS = 12

PROMPTS_DIR = Path("prompts")
PROMPT_NAME = "mechanism_reading_prompt.j2"
SCHEMA_PATH = PROMPTS_DIR / "mechanism_reading_schema.json"

ASSOCIATIONS_HEADING = "GENE-DISEASE ASSOCIATIONS THIS PAPER IS CITED FOR:"


@dataclass(frozen=True)
class CitingAssociation:
    """An association that cites a paper, as the request names it."""

    id: int
    gene_symbol: str
    description: str
    inheritance: str  # mode of inheritance with its details, see format_inheritance

    @property
    def label(self) -> str:
        """The ID the request gives the association."""
        return f"A{self.id}"


@dataclass(frozen=True)
class PaperToRead:
    paper: dict[str, Any]  # doi, source_date, title, abstract
    associations: tuple[CitingAssociation, ...]  # the associations that cite it, by id


@dataclass
class ReadOutcome:
    stored: int = 0
    to_fallback: int = 0  # the refusals by MODEL: these go to FALLBACK_MODEL in the next attempt


def prompt_environment() -> Environment:
    """The Jinja environment of the mechanism prompts."""
    return Environment(
        loader=FileSystemLoader(PROMPTS_DIR),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def citing_associations(
    conn: sqlite3.Connection, hgnc_resolver: HgncResolver, hgnc_ids: Sequence[int] | None
) -> dict[str, tuple[CitingAssociation, ...]]:
    """The associations that cite each paper, by DOI, each paper's by association id.

    Only associations of a stored aggregation count, and only those of *hgnc_ids*
    when given.
    """
    citing: dict[str, list[CitingAssociation]] = defaultdict(list)
    for association_id, hgnc_id, assessment_json in conn.execute(
        """
        SELECT a.id, a.hgnc_id, a.assessment_json
        FROM associations a
        JOIN gene_aggregations g ON g.hgnc_id = a.hgnc_id
        ORDER BY a.id
        """
    ):
        if hgnc_ids is not None and hgnc_id not in hgnc_ids:
            continue
        assessment = json.loads(assessment_json)
        association = CitingAssociation(
            id=association_id,
            gene_symbol=hgnc_resolver.get_symbol(hgnc_id),
            description=assessment["description"],
            inheritance=format_inheritance(
                assessment["inheritance_mode"], assessment["inheritance_details"]
            ),
        )
        for doi in assessment["dois"]:
            citing[doi].append(association)
    return {doi: tuple(associations) for doi, associations in citing.items()}


def covered_associations(conn: sqlite3.Connection) -> dict[str, frozenset[int]]:
    """The association ids each stored reading covers, by DOI."""
    return {
        doi: frozenset(readings_from_json(reading_json))
        for doi, reading_json in conn.execute("SELECT doi, reading_json FROM mechanism_readings")
    }


def select_papers(
    db_path: Path,
    hgnc_resolver: HgncResolver,
    hgnc_ids: Sequence[int] | None,
    history: StageHistory,
) -> list[PaperToRead]:
    """The cited papers without a reading that covers the associations citing them now,
    except the ones skipped for good, by DOI."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        citing = citing_associations(conn, hgnc_resolver, hgnc_ids)
        covered = covered_associations(conn)
        due = sorted(
            doi
            for doi, associations in citing.items()
            if covered.get(doi) != frozenset(a.id for a in associations)
            and not history.skipped(doi)
        )
        papers = {
            row["doi"]: dict(row)
            for row in conn.execute(
                f"""
                SELECT doi, source_date, title, abstract FROM papers
                WHERE doi IN ({",".join("?" * len(due))})
                """,
                due,
            )
        }
    return [PaperToRead(paper=papers[doi], associations=citing[doi]) for doi in due]


def count_unread(db_path: Path, hgnc_resolver: HgncResolver, hgnc_ids: Sequence[int] | None) -> int:
    """The cited papers without a reading that covers the associations citing them now."""
    with closing(sqlite3.connect(db_path)) as conn:
        citing = citing_associations(conn, hgnc_resolver, hgnc_ids)
        covered = covered_associations(conn)
    return sum(
        covered.get(doi) != frozenset(a.id for a in associations)
        for doi, associations in citing.items()
    )


def store_reading(conn: sqlite3.Connection, doi: str, reading: dict[str, Any], raw: str) -> None:
    """Store *reading* as the paper's reading, in place of an earlier one."""
    conn.execute(
        "INSERT OR REPLACE INTO mechanism_readings (doi, reading_json, reading_raw) "
        "VALUES (?, ?, ?)",
        (doi, json.dumps(reading), raw),
    )


# ---------------------------------------------------------------------------
# Requests and answers
# ---------------------------------------------------------------------------


def user_text(to_read: PaperToRead, pdf: UploadedPdf) -> str:
    """The text block of a paper's request: the paper, then the associations that cite it."""
    lines = [paper_text(to_read.paper, pdf.pages, pdf.total_pages), "", ASSOCIATIONS_HEADING]
    lines += [
        f"- {a.label}: gene {a.gene_symbol}; disease: {a.description}; "
        f"mode of inheritance: {a.inheritance}"
        for a in to_read.associations
    ]
    return "\n".join(lines)


def build_request(
    to_read: PaperToRead,
    pdf: UploadedPdf,
    system: str,
    output_config: OutputConfigParam,
    model: str,
) -> LlmRequest:
    return LlmRequest(
        subject=to_read.paper["doi"],
        params={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": cached_system(system),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "document", "source": {"type": "file", "file_id": pdf.file_id}},
                        {"type": "text", "text": user_text(to_read, pdf)},
                    ],
                }
            ],
            "output_config": output_config,
        },
    )


def parse_reading(
    answer: dict[str, Any], associations: Sequence[CitingAssociation]
) -> dict[str, Any]:
    """The answer as ``mechanism_readings.reading_json``, its associations by their ids.

    Raises ValueError unless the answer holds each of *associations* exactly once.
    """
    by_label = {a.label: a for a in associations}
    labels = [entry["association"] for entry in answer["associations"]]
    if sorted(labels) != sorted(by_label):
        raise ValueError(f"answers for associations {labels}, asked for {sorted(by_label)}")
    return {
        "associations": [
            {
                "association_id": by_label[entry["association"]].id,
                "authors_conclusion": entry["authors_conclusion"],
                "evidence": entry["evidence"],
            }
            for entry in answer["associations"]
        ]
    }


def handle_result(
    result: LlmResult,
    db_path: Path,
    validator: jsonschema.protocols.Validator,
    associations: Sequence[CitingAssociation],
) -> bool:
    """Record *result* and store its reading if it is valid; True when it was stored.

    Raises RuntimeError, after recording it, when the request was invalid.
    """
    doi = result.subject
    message = result.message
    if result.status == ResultStatus.REFUSED:
        log_refusal(result, doi)
    if result.status != ResultStatus.SUCCEEDED or message is None:
        with sqlite3.connect(db_path) as conn:
            record_result(conn, result)
        if result.error_type == "invalid_request_error":
            raise RuntimeError(f"invalid read-mechanisms request for {doi}")
        return False
    try:
        answer = parse_json_output(message)
        validator.validate(answer)
        reading = parse_reading(answer, associations)
    except (ValueError, jsonschema.ValidationError) as e:
        rejection = invalid_answer_reason(e)
        logger.warning("Invalid mechanism reading of %s: %s", doi, rejection)
        with sqlite3.connect(db_path) as conn:
            record_result(conn, result, rejection=rejection)
        return False
    with sqlite3.connect(db_path) as conn:
        record_result(conn, result)
        store_reading(conn, doi, reading, message.to_json())
    return True


async def send_each(
    transport: Transport,
    stage: str,
    requests: Sequence[LlmRequest],
    on_result: Callable[[LlmResult], None],
    workers: int,
) -> None:
    """Send *requests*, at most *workers* at a time, and pass each result to *on_result*
    as it arrives.

    An exception from *on_result*, such as for an invalid request, cancels the
    outstanding requests and propagates in an ExceptionGroup.
    """
    slots = asyncio.Semaphore(workers)

    async def send(request: LlmRequest) -> None:
        async with slots:
            (result,) = await transport.run(stage, 1, [request])
        on_result(result)

    async with asyncio.TaskGroup() as tasks:
        for request in requests:
            tasks.create_task(send(request))


async def run_attempt(
    transport: Transport,
    requests: Sequence[LlmRequest],
    db_path: Path,
    validator: jsonschema.protocols.Validator,
    by_doi: dict[str, PaperToRead],
) -> ReadOutcome:
    """Send *requests*, store each result as it arrives, and count the outcomes."""
    outcome = ReadOutcome()
    with Progress() as progress:
        task = progress.add_task("Reading papers", total=len(requests))

        def on_result(result: LlmResult) -> None:
            outcome.stored += handle_result(
                result, db_path, validator, by_doi[result.subject].associations
            )
            outcome.to_fallback += result.goes_to_fallback
            progress.advance(task)

        await send_each(transport, STAGE, requests, on_result, MAX_CONCURRENT_PAPERS)
    return outcome


async def read_papers(
    *,
    client: AsyncAnthropic,
    transport: Transport,
    db_path: Path,
    papers_dir: Path,
    hgnc_resolver: HgncResolver,
    hgnc_ids: Sequence[int] | None,
    limit: int | None,
    schema: dict[str, Any],
    system: str,
    max_retries: int,
    failures_since: datetime,
) -> None:
    """Read every paper due a reading, in attempts.

    An attempt makes progress when it stores a reading or sends a paper on to
    FALLBACK_MODEL. *limit* caps the papers of each attempt. A paper without a PDF
    in *papers_dir*, or whose PDF does not fit the context window, is left out.
    Failed answers count from *failures_since*.
    """

    def load_history() -> StageHistory:
        with closing(sqlite3.connect(db_path)) as conn:
            return stage_history(conn, STAGE, failures_since=failures_since)

    validator = jsonschema.Draft202012Validator(schema)
    output_config = json_output_config(schema, EFFORT)
    settings = RequestSettings(system=system, output_config=output_config, cache_pdf=False)
    log_failed_for_good(load_history().failures, STAGE)
    for attempt in range(1, max_retries + 1):
        history = load_history()
        due = select_papers(db_path, hgnc_resolver, hgnc_ids, history)[:limit]
        if not due:
            logger.info("No papers left to read")
            return
        uploads = await upload_pdfs(
            client, db_path, [to_read.paper for to_read in due], papers_dir, settings, history
        )
        readable = [to_read for to_read in due if to_read.paper["doi"] in uploads]
        if not readable:
            logger.error("None of the %d papers due a reading has a usable PDF", len(due))
            return
        logger.info("Attempt %d: reading %d papers", attempt, len(readable))
        by_doi = {to_read.paper["doi"]: to_read for to_read in readable}
        requests = [
            build_request(
                to_read,
                uploads[to_read.paper["doi"]],
                system,
                output_config,
                history.model_for(to_read.paper["doi"]),
            )
            for to_read in readable
        ]
        outcome = await run_attempt(transport, requests, db_path, validator, by_doi)
        logger.info(
            "Attempt %d: stored %d of %d; %d go to the fallback model",
            attempt,
            outcome.stored,
            len(readable),
            outcome.to_fallback,
        )
        if outcome.stored + outcome.to_fallback == 0:
            logger.error("No progress in attempt %d - stopping", attempt)
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
    hgnc_ids: list[int] | None = typer.Option(
        None,
        "--hgnc-id",
        help="Read only the papers that this gene's associations cite (repeatable)",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="Read at most this many papers per attempt (for prompt development)"
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of attempts for papers whose reading failed or was refused; the "
        "fallback-model request after a refusal is one of them",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Send papers whose answers failed for good in earlier invocations (rejected or "
        "cut off at max_tokens, MAX_FAILED_ANSWERS times) again, until they fail that often "
        "in this invocation",
    ),
) -> None:
    """Read the papers that associations cite for their mechanism-of-disease evidence."""
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        raise typer.Exit(1)
    # With --retry-failed, only failed answers during this invocation count.
    failures_since = datetime.now(UTC) if retry_failed else ALL_TIME

    system = prompt_environment().get_template(PROMPT_NAME).render()
    schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())
    hgnc_resolver = HgncResolver.from_file()
    only = hgnc_ids or None
    logger.info(
        "%s cited papers without a current reading",
        f"{count_unread(db_path, hgnc_resolver, only):,}",
    )

    async def run() -> None:
        client = make_client()
        await read_papers(
            client=client,
            transport=ImmediateTransport(client),
            db_path=db_path,
            papers_dir=papers_dir,
            hgnc_resolver=hgnc_resolver,
            hgnc_ids=only,
            limit=limit,
            schema=schema,
            system=system,
            max_retries=max_retries,
            failures_since=failures_since,
        )

    asyncio.run(run())
    logger.info(
        "%s cited papers still without a current reading",
        f"{count_unread(db_path, hgnc_resolver, only):,}",
    )
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
