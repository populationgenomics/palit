#!/usr/bin/env python3
"""Judge each gene-disease-MoI association's mechanisms of disease.

One request per association without mechanisms: the system part states the
mechanisms, the evidence that counts and how to judge the confidence, and is the
same for all requests; the user part carries the association (disease, mode of
inheritance, rating, summary, variants), the gene's other associations, ClinGen's
dosage sensitivity and gene-disease validity curations of the gene, and what
read-mechanisms recorded from each of the association's papers. The answer asserts
each mechanism the evidence supports, with a confidence term and a statement, or
says why the mechanism is unknown. It is stored on the association row, in the
transaction that records the request. An association MODEL refused goes to
FALLBACK_MODEL in the next attempt (see :mod:`palit.llm`).

An association is assessed once read-mechanisms has read all its papers; see
``--with-unread-papers``.
"""

import asyncio
import json
import logging
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import typer
from anthropic.types.output_config_param import OutputConfigParam
from jinja2 import Template

from palit.assess_genes import VariantRow, variant_table
from palit.clingen_dosage import DosageCuration, fetch_dosage
from palit.hgnc import HgncResolver
from palit.llm import (
    ALL_TIME,
    DEFAULT_IMMEDIATE_WORKERS,
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
from palit.mechanisms import (
    LIMITED_EVIDENCE_CAP,
    AssertedMechanism,
    AssociationReading,
    MechanismAssessment,
    readings_from_json,
)
from palit.panelapp_integration import PREFILL_RATINGS, calculate_association_rating
from palit.progress import LoggingProgress as Progress
from palit.read_mechanisms import PROMPTS_DIR, prompt_environment, send_each
from palit.variants import load_gene_variants

app = typer.Typer(help="Judge the mechanisms of disease of gene-disease-MoI associations")
logger = logging.getLogger(__name__)

STAGE = "assess_mechanisms"
EFFORT: Effort = "high"
MAX_TOKENS = 32000

PROMPT_NAME = "mechanism_assessment_prompt.j2"
ASSOCIATION_TEMPLATE_NAME = "mechanism_assessment_association.j2"
SCHEMA_PATH = PROMPTS_DIR / "mechanism_assessment_schema.json"

# The association rating whose gene-disease evidence is limited
LIMITED_EVIDENCE_RATING = 1


@dataclass(frozen=True)
class OtherAssociation:
    """Another association of the same gene, which is assessed on its own."""

    description: str
    inheritance: str  # mode of inheritance with its details, see format_inheritance
    mechanism_text: str  # the aggregation's free-text mechanism, or "NR"


@dataclass(frozen=True)
class PaperReading:
    """One of an association's papers with what read-mechanisms recorded for the association."""

    paper_id: str  # the {LastName}{Year} ID the aggregation gave the paper
    reading: AssociationReading | None  # None for a paper without a current reading


@dataclass(frozen=True)
class Association:
    """One association without mechanisms, as the model sees it."""

    id: int
    hgnc_id: int
    description: str
    inheritance_mode: str
    inheritance_details: str
    rating: int  # computed from the association's criteria: 3 GREEN, 2 AMBER, 1 RED
    independent_family_count: int | None
    summary: str
    functional: str
    mechanism_text: str  # the aggregation's free-text mechanism, or "NR"
    variants: tuple[VariantRow, ...]
    others: tuple[OtherAssociation, ...]  # in the aggregation's order
    papers: tuple[PaperReading, ...]  # by paper ID
    paper_id_to_doi: dict[str, str]  # of this association's papers
    clingen_curations: tuple[dict[str, Any], ...]  # ClinGen's validity curations of the gene

    @property
    def capped(self) -> bool:
        """Whether limited gene-disease evidence caps the confidence terms."""
        return self.rating == LIMITED_EVIDENCE_RATING

    @property
    def unread_papers(self) -> list[str]:
        """The IDs of the papers without a current reading."""
        return [paper.paper_id for paper in self.papers if paper.reading is None]


@dataclass
class AssessOutcome:
    stored: int = 0
    to_fallback: int = 0  # the refusals by MODEL: these go to FALLBACK_MODEL in the next attempt


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def select_associations(db_path: Path, history: StageHistory) -> list[Association]:
    """Associations without mechanisms, except the ones skipped for good.

    Only rows of a stored aggregation count. Each comes with the gene's other
    associations, its variants, and its papers (``dois``) by the IDs its
    aggregation gave them, with the paper's reading for this association where
    read-mechanisms stored one.
    """
    associations: list[Association] = []
    with closing(sqlite3.connect(db_path)) as conn:
        gene_rows: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        contexts: dict[int, tuple[dict[str, str], dict[str, Any]]] = {}
        for (
            association_id,
            hgnc_id,
            assessment_json,
            paper_id_mapping,
            context_json,
        ) in conn.execute(
            """
            SELECT a.id, a.hgnc_id, a.assessment_json, g.paper_id_mapping,
                   g.panelapp_context_json
            FROM associations a
            JOIN gene_aggregations g ON g.hgnc_id = a.hgnc_id
            WHERE a.hgnc_id IN (SELECT hgnc_id FROM associations WHERE mechanisms_json IS NULL)
            ORDER BY a.hgnc_id, a.position
            """
        ):
            gene_rows[hgnc_id].append((association_id, json.loads(assessment_json)))
            contexts[hgnc_id] = (json.loads(paper_id_mapping), json.loads(context_json))
        due = {
            association_id
            for (association_id,) in conn.execute(
                "SELECT id FROM associations WHERE mechanisms_json IS NULL"
            )
        }
        readings: dict[str, dict[int, AssociationReading]] = {
            doi: readings_from_json(reading_json)
            for doi, reading_json in conn.execute(
                "SELECT doi, reading_json FROM mechanism_readings"
            )
        }
        for hgnc_id, rows in gene_rows.items():
            paper_id_to_doi, context = contexts[hgnc_id]
            doi_to_paper_id = {doi: paper_id for paper_id, doi in paper_id_to_doi.items()}
            gene_variants = load_gene_variants(conn.cursor(), hgnc_id, list(doi_to_paper_id))
            for association_id, assessment in rows:
                if association_id not in due or history.skipped(str(association_id)):
                    continue
                keys = set(assessment["variants"])
                variant_rows, _ = variant_table(
                    [v for v in gene_variants if v.variant_id in keys], doi_to_paper_id
                )
                papers = sorted(
                    (
                        PaperReading(
                            paper_id=doi_to_paper_id[doi],
                            reading=readings.get(doi, {}).get(association_id),
                        )
                        for doi in assessment["dois"]
                    ),
                    key=lambda paper: paper.paper_id,
                )
                associations.append(
                    Association(
                        id=association_id,
                        hgnc_id=hgnc_id,
                        description=assessment["description"],
                        inheritance_mode=assessment["inheritance_mode"],
                        inheritance_details=assessment["inheritance_details"],
                        rating=calculate_association_rating(assessment),
                        independent_family_count=assessment["independent_family_count"],
                        summary=assessment["summary"],
                        functional=assessment["functional"],
                        mechanism_text=assessment["disease_mechanism"],
                        variants=tuple(variant_rows),
                        others=tuple(
                            OtherAssociation(
                                description=other["description"],
                                inheritance=format_inheritance(
                                    other["inheritance_mode"], other["inheritance_details"]
                                ),
                                mechanism_text=other["disease_mechanism"],
                            )
                            for other_id, other in rows
                            if other_id != association_id
                        ),
                        papers=tuple(papers),
                        paper_id_to_doi={doi_to_paper_id[doi]: doi for doi in assessment["dois"]},
                        # An aggregation stored before assess-genes read ClinGen's
                        # curations has no such key.
                        clingen_curations=tuple(context.get("clingen_curations", [])),
                    )
                )
    return associations


def count_unassessed(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute(
            """
            SELECT COUNT(*) FROM associations a
            JOIN gene_aggregations g ON g.hgnc_id = a.hgnc_id
            WHERE a.mechanisms_json IS NULL
            """
        ).fetchone()
    return int(count)


def store_mechanisms(
    conn: sqlite3.Connection, association_id: int, assessment: MechanismAssessment, raw: str
) -> bool:
    """Store *assessment* on the association; False when the row is gone or already has one."""
    cursor = conn.execute(
        """
        UPDATE associations SET mechanisms_json = ?, mechanisms_raw = ?
        WHERE id = ? AND mechanisms_json IS NULL
        """,
        (json.dumps(assessment.to_json()), raw, association_id),
    )
    if cursor.rowcount == 0:
        logger.warning(
            "Association %d no longer exists or was assessed meanwhile; mechanisms not stored",
            association_id,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Requests and answers
# ---------------------------------------------------------------------------


def association_text(
    association: Association,
    template: Template,
    gene_symbol: str,
    dosage: DosageCuration | None,
) -> str:
    """The user part of an association's request."""
    return template.render(
        gene_symbol=gene_symbol,
        association=association,
        inheritance=format_inheritance(
            association.inheritance_mode, association.inheritance_details
        ),
        rating=PREFILL_RATINGS[association.rating],
        capped=association.capped,
        dosage=dosage,
        clingen_curations=association.clingen_curations,
    )


def build_request(
    association: Association,
    system: str,
    user_text: str,
    output_config: OutputConfigParam,
    model: str,
) -> LlmRequest:
    return LlmRequest(
        subject=str(association.id),
        params={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": cached_system(system),
            "messages": [{"role": "user", "content": user_text}],
            "output_config": output_config,
        },
    )


def with_paper_ids_as_given(text: str, paper_ids: Iterable[str]) -> str:
    """*text* with each of *paper_ids* spelled as given, whatever its letter case in *text*.

    The model sometimes restores an author's capitalisation ("McLaughlin2011" for the
    ID "Mclaughlin2011"), and the report replaces paper IDs by exact match.
    """
    for paper_id in paper_ids:
        text = re.sub(rf"\b{re.escape(paper_id)}\b", paper_id, text, flags=re.IGNORECASE)
    return text


def parse_assessment(answer: dict[str, Any], association: Association) -> MechanismAssessment:
    """The answer as the association's mechanisms, its papers by DOI.

    The association's paper IDs are spelled as the aggregation gave them, in the
    statements, the note and the lists of papers (see :func:`with_paper_ids_as_given`).

    Raises ValueError for a mechanism asserted twice, a dominant-negative mechanism
    of a biallelic association, a confidence term above the cap of an association
    with limited evidence, a paper that is not one of the association's, a
    statement with a line break, and an ``unknown_note`` that does not go with
    the mechanisms: it is required without mechanisms and must be empty with them.
    """
    problems: list[str] = []
    paper_ids = association.paper_id_to_doi
    entries = [
        {
            **entry,
            "statement": with_paper_ids_as_given(entry["statement"].strip(), paper_ids),
            "paper_ids": [with_paper_ids_as_given(pid, paper_ids) for pid in entry["paper_ids"]],
        }
        for entry in answer["mechanisms"]
    ]
    asserted = [entry["mechanism"] for entry in entries]
    if len(set(asserted)) != len(asserted):
        problems.append(f"mechanisms asserted more than once: {asserted}")
    if "dominant_negative" in asserted and association.inheritance_mode == "Biallelic":
        problems.append("dominant_negative asserted for a biallelic association")
    unknown_note = with_paper_ids_as_given(answer["unknown_note"].strip(), paper_ids)
    if bool(asserted) == bool(unknown_note):
        problems.append(
            "unknown_note given although mechanisms are asserted"
            if asserted
            else "neither mechanisms nor unknown_note"
        )
    for entry in entries:
        if association.capped and entry["confidence"] != LIMITED_EVIDENCE_CAP:
            problems.append(
                f"{entry['mechanism']} is {entry['confidence']}, above the cap of "
                f"{LIMITED_EVIDENCE_CAP} for limited evidence"
            )
        if "\n" in entry["statement"]:
            problems.append(f"line break in the statement of {entry['mechanism']}")
        unknown = [pid for pid in entry["paper_ids"] if pid not in paper_ids]
        if unknown:
            problems.append(f"unknown paper IDs {unknown} for {entry['mechanism']}")
    if problems:
        raise ValueError("; ".join(problems))
    return MechanismAssessment(
        mechanisms=tuple(
            AssertedMechanism(
                mechanism=entry["mechanism"],
                confidence=entry["confidence"],
                statement=entry["statement"],
                dois=tuple(dict.fromkeys(paper_ids[pid] for pid in entry["paper_ids"])),
            )
            for entry in entries
        ),
        unknown_note=unknown_note,
    )


def handle_result(
    result: LlmResult,
    db_path: Path,
    validator: jsonschema.protocols.Validator,
    association: Association,
) -> bool:
    """Record *result* and store its mechanisms if they are valid; True when they were stored.

    Raises RuntimeError, after recording it, when the request was invalid.
    """
    message = result.message
    if result.status == ResultStatus.REFUSED:
        log_refusal(result, f"association {association.id}")
    if result.status != ResultStatus.SUCCEEDED or message is None:
        with sqlite3.connect(db_path) as conn:
            record_result(conn, result)
        if result.error_type == "invalid_request_error":
            raise RuntimeError(
                f"invalid assess-mechanisms request for association {association.id}"
            )
        return False
    try:
        answer = parse_json_output(message)
        validator.validate(answer)
        assessment = parse_assessment(answer, association)
    except (ValueError, jsonschema.ValidationError) as e:
        rejection = invalid_answer_reason(e)
        logger.warning("Invalid mechanisms for association %d: %s", association.id, rejection)
        with sqlite3.connect(db_path) as conn:
            record_result(conn, result, rejection=rejection)
        return False
    with sqlite3.connect(db_path) as conn:
        record_result(conn, result)
        return store_mechanisms(conn, association.id, assessment, message.to_json())


async def run_attempt(
    transport: Transport,
    requests: Sequence[LlmRequest],
    db_path: Path,
    validator: jsonschema.protocols.Validator,
    by_id: dict[str, Association],
) -> AssessOutcome:
    """Send *requests*, store each result as it arrives, and count the outcomes."""
    outcome = AssessOutcome()
    with Progress() as progress:
        task = progress.add_task("Assessing mechanisms", total=len(requests))

        def on_result(result: LlmResult) -> None:
            outcome.stored += handle_result(result, db_path, validator, by_id[result.subject])
            outcome.to_fallback += result.goes_to_fallback
            progress.advance(task)

        await send_each(transport, STAGE, requests, on_result, DEFAULT_IMMEDIATE_WORKERS)
    return outcome


async def assess_associations(
    *,
    transport: Transport,
    db_path: Path,
    hgnc_resolver: HgncResolver,
    dosage: dict[int, DosageCuration],
    schema: dict[str, Any],
    system: str,
    association_template: Template,
    with_unread_papers: bool,
    max_retries: int,
    failures_since: datetime,
) -> None:
    """Assess every association due mechanisms, in attempts.

    An association with a paper that read-mechanisms has not read is left out,
    unless *with_unread_papers*; its request then says which papers were not read.
    An attempt makes progress when it stores mechanisms or sends an association on
    to FALLBACK_MODEL. Failed answers count from *failures_since*.
    """

    def load_history() -> StageHistory:
        with closing(sqlite3.connect(db_path)) as conn:
            return stage_history(conn, STAGE, failures_since=failures_since)

    validator = jsonschema.Draft202012Validator(schema)
    output_config = json_output_config(schema, EFFORT)
    log_failed_for_good(load_history().failures, STAGE)
    for attempt in range(1, max_retries + 1):
        history = load_history()
        associations = select_associations(db_path, history)
        if not with_unread_papers:
            waiting = [a for a in associations if a.unread_papers]
            if waiting:
                logger.warning(
                    "%d associations wait for read-mechanisms to read %d of their papers",
                    len(waiting),
                    sum(len(a.unread_papers) for a in waiting),
                )
            associations = [a for a in associations if not a.unread_papers]
        if not associations:
            logger.info("No associations left to assess")
            return
        logger.info("Attempt %d: assessing %d associations", attempt, len(associations))
        requests = [
            build_request(
                association,
                system,
                association_text(
                    association,
                    association_template,
                    hgnc_resolver.get_symbol(association.hgnc_id),
                    dosage.get(association.hgnc_id),
                ),
                output_config,
                history.model_for(str(association.id)),
            )
            for association in associations
        ]
        by_id = {str(association.id): association for association in associations}
        outcome = await run_attempt(transport, requests, db_path, validator, by_id)
        logger.info(
            "Attempt %d: stored %d of %d; %d go to the fallback model",
            attempt,
            outcome.stored,
            len(associations),
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
    with_unread_papers: bool = typer.Option(
        False,
        "--with-unread-papers",
        help="Also assess associations with papers that read-mechanisms has not read, on the "
        "readings of their other papers",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of attempts for associations whose assessment failed or was "
        "refused; the fallback-model request after a refusal is one of them",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Send associations whose answers failed for good in earlier invocations (rejected "
        "or cut off at max_tokens, MAX_FAILED_ANSWERS times) again, until they fail that often "
        "in this invocation",
    ),
) -> None:
    """Judge the mechanisms of disease of gene-disease-MoI associations."""
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        raise typer.Exit(1)
    # With --retry-failed, only failed answers during this invocation count.
    failures_since = datetime.now(UTC) if retry_failed else ALL_TIME

    environment = prompt_environment()
    system = environment.get_template(PROMPT_NAME).render()
    association_template = environment.get_template(ASSOCIATION_TEMPLATE_NAME)
    schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())
    dosage = fetch_dosage(db_path.parent)

    logger.info("%s associations without mechanisms", f"{count_unassessed(db_path):,}")

    async def run() -> None:
        await assess_associations(
            transport=ImmediateTransport(make_client()),
            db_path=db_path,
            hgnc_resolver=HgncResolver.from_file(),
            dosage=dosage,
            schema=schema,
            system=system,
            association_template=association_template,
            with_unread_papers=with_unread_papers,
            max_retries=max_retries,
            failures_since=failures_since,
        )

    asyncio.run(run())
    logger.info("%s associations still without mechanisms", f"{count_unassessed(db_path):,}")
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
