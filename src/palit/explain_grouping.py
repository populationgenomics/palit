#!/usr/bin/env python3
"""Write each association's lumping and splitting statement for its review.

assess-genes records how it grouped a gene's diseases in ``grouping_rationale``: a
working note that names papers and the steps of its prompt. A curator who reviews
an association needs the reason in the form ClinGen's expert panels give it: which
disease entities the association lumps, which other associations of the gene it is
kept apart from, and on which criterion.

One request per gene with an association that has no statement: the system part
states the task and the criteria and is the same for all requests; the user part
carries the gene's associations with their grouping rationales and mechanisms of
disease, the disease terms GenCC and PanelApp Australia list for the gene, and
ClinGen's curations. The answer holds one statement per association. The
statements are stored on the association rows and the answer on the gene's
aggregation, in the transaction that records the request. A gene MODEL refused
goes to FALLBACK_MODEL in the next attempt (see :mod:`palit.llm`).

The stage explains the grouping and does not change it. It reads the mechanisms
that assess-mechanisms stored, so it runs after that stage.
"""

import asyncio
import json
import logging
import re
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import typer
from anthropic.types.output_config_param import OutputConfigParam
from jinja2 import Template

from palit.gencc import NO_GENCC_TERMS
from palit.generate_report import association_name
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
from palit.mechanisms import MechanismAssessment
from palit.panelapp_integration import PREFILL_RATINGS, calculate_association_rating
from palit.progress import LoggingProgress as Progress
from palit.read_mechanisms import PROMPTS_DIR, prompt_environment, send_each

app = typer.Typer(help="Write the lumping and splitting statement of each association")
logger = logging.getLogger(__name__)

STAGE = "explain_grouping"
EFFORT: Effort = "high"
MAX_TOKENS = 32000
MAX_STATEMENT_WORDS = 110  # the prompt asks for at most 90; a little over is accepted

PROMPT_NAME = "grouping_statement_prompt.j2"
GENE_TEMPLATE_NAME = "grouping_statement_gene.j2"
SCHEMA_PATH = PROMPTS_DIR / "grouping_statement_schema.json"

NOT_ASSESSED = "not assessed"


@dataclass(frozen=True)
class GroupedAssociation:
    """One association of a gene, as the request describes it."""

    id: int
    name: str  # see generate_report.association_name
    inheritance: str  # mode of inheritance with its details, see format_inheritance
    rating: str  # Green, Amber or Red, from the association's criteria
    mondo: str  # "label (MONDO id), exact", or why it has no term yet
    description: str
    grouping_rationale: str  # the aggregation's note, with its paper IDs
    relation: str  # the association's status in PanelApp Australia and its basis
    mechanisms: str  # e.g. "Dominant negative (likely), Loss of function (suspected)"

    @property
    def label(self) -> str:
        """The ID the request gives the association."""
        return f"A{self.id}"


@dataclass(frozen=True)
class GeneToExplain:
    """A gene with its associations and the context its grouping was decided on."""

    hgnc_id: int
    associations: tuple[GroupedAssociation, ...]  # in the aggregation's order
    paper_ids: frozenset[str]  # the IDs the aggregation gave the gene's papers
    gencc_terms: str  # one line per term, see format_gencc_terms
    panel_entries: tuple[dict[str, Any], ...]  # entries on every panel holding the gene
    clingen_curations: tuple[dict[str, Any], ...]


@dataclass
class ExplainOutcome:
    stored: int = 0
    to_fallback: int = 0  # the refusals by MODEL: these go to FALLBACK_MODEL in the next attempt


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def mechanisms_line(mechanisms_json: str | None) -> str:
    """An association's mechanisms in one line, or that they are unknown or not assessed."""
    if mechanisms_json is None:
        return NOT_ASSESSED
    assessment = MechanismAssessment.from_json(json.loads(mechanisms_json))
    if assessment.unknown:
        return "unknown"
    return ", ".join(f"{m.label} ({m.confidence.lower()})" for m in assessment.mechanisms)


def mondo_line(mondo_id: str | None, mondo_label: str | None, mondo_match: str | None) -> str:
    if mondo_id is None:
        return "none yet"
    kind = "a broader term" if mondo_match == "broader" else "the disease's own term"
    return f"{mondo_label} ({mondo_id}), {kind}"


def select_genes(db_path: Path, history: StageHistory) -> list[GeneToExplain]:
    """The genes with an association without a statement, except the ones skipped for
    good, by HGNC ID. Only genes of a stored aggregation count."""
    genes: list[GeneToExplain] = []
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        due = [
            row["hgnc_id"]
            for row in conn.execute(
                """
                SELECT DISTINCT a.hgnc_id FROM associations a
                JOIN gene_aggregations g ON g.hgnc_id = a.hgnc_id
                WHERE a.grouping_statement IS NULL
                ORDER BY a.hgnc_id
                """
            )
        ]
        for hgnc_id in due:
            if history.skipped(str(hgnc_id)):
                continue
            aggregation = conn.execute(
                "SELECT paper_id_mapping, panelapp_context_json FROM gene_aggregations "
                "WHERE hgnc_id = ?",
                (hgnc_id,),
            ).fetchone()
            context: dict[str, Any] = json.loads(aggregation["panelapp_context_json"])
            associations: list[GroupedAssociation] = []
            for row in conn.execute(
                """
                SELECT id, assessment_json, mondo_id, mondo_label, mondo_match, mechanisms_json
                FROM associations WHERE hgnc_id = ? ORDER BY position
                """,
                (hgnc_id,),
            ):
                assessment = json.loads(row["assessment_json"])
                relation = assessment["panelapp_relation"]
                associations.append(
                    GroupedAssociation(
                        id=row["id"],
                        name=association_name(assessment, context["gencc_terms"]),
                        inheritance=format_inheritance(
                            assessment["inheritance_mode"], assessment["inheritance_details"]
                        ),
                        rating=PREFILL_RATINGS[calculate_association_rating(assessment)],
                        mondo=mondo_line(row["mondo_id"], row["mondo_label"], row["mondo_match"]),
                        description=assessment["description"],
                        grouping_rationale=assessment["grouping_rationale"],
                        relation=f"{relation['status']}: {relation['basis']}",
                        mechanisms=mechanisms_line(row["mechanisms_json"]),
                    )
                )
            genes.append(
                GeneToExplain(
                    hgnc_id=hgnc_id,
                    associations=tuple(associations),
                    paper_ids=frozenset(json.loads(aggregation["paper_id_mapping"])),
                    gencc_terms="\n".join(
                        f"- {t['disease_title']} ({t['mondo_id']}) | {', '.join(t['moi_titles'])}"
                        for t in context["gencc_terms"]
                    )
                    or NO_GENCC_TERMS,
                    panel_entries=tuple(context["panel_entries"]),
                    # An aggregation stored before assess-genes read ClinGen's curations
                    # has no such key.
                    clingen_curations=tuple(context.get("clingen_curations", [])),
                )
            )
    return genes


def count_unexplained(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute(
            """
            SELECT COUNT(*) FROM associations a
            JOIN gene_aggregations g ON g.hgnc_id = a.hgnc_id
            WHERE a.grouping_statement IS NULL
            """
        ).fetchone()
    return int(count)


def store_statements(
    conn: sqlite3.Connection, hgnc_id: int, statements: dict[int, str], raw: str
) -> bool:
    """Store the gene's *statements* by association id, and the answer on its aggregation.

    False, with nothing stored, when the gene's associations are no longer the
    ones the statements are for: the gene was aggregated again meanwhile.
    """
    current = {
        association_id
        for (association_id,) in conn.execute(
            "SELECT id FROM associations WHERE hgnc_id = ?", (hgnc_id,)
        )
    }
    if current != set(statements):
        logger.warning(
            "HGNC:%d was aggregated again meanwhile; grouping statements not stored", hgnc_id
        )
        return False
    conn.executemany(
        "UPDATE associations SET grouping_statement = ? WHERE id = ?",
        [(statement, association_id) for association_id, statement in statements.items()],
    )
    conn.execute(
        "UPDATE gene_aggregations SET grouping_statements_raw = ? WHERE hgnc_id = ?",
        (raw, hgnc_id),
    )
    return True


# ---------------------------------------------------------------------------
# Requests and answers
# ---------------------------------------------------------------------------


def gene_text(gene: GeneToExplain, template: Template, gene_symbol: str) -> str:
    """The user part of a gene's request."""
    return template.render(
        gene_symbol=gene_symbol,
        associations=gene.associations,
        gencc_terms=gene.gencc_terms,
        panel_entries=gene.panel_entries,
        clingen_curations=gene.clingen_curations,
    )


def build_request(
    gene: GeneToExplain,
    system: str,
    user_text: str,
    output_config: OutputConfigParam,
    model: str,
) -> LlmRequest:
    return LlmRequest(
        subject=str(gene.hgnc_id),
        params={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": cached_system(system),
            "messages": [{"role": "user", "content": user_text}],
            "output_config": output_config,
        },
    )


def parse_statements(answer: dict[str, Any], gene: GeneToExplain) -> dict[int, str]:
    """The answer's statements by association id.

    Raises ValueError unless the answer holds each of the gene's associations
    exactly once, and for a statement with a line break, one of the gene's paper
    IDs, or more than MAX_STATEMENT_WORDS words.
    """
    by_label = {a.label: a for a in gene.associations}
    labels = [entry["association"] for entry in answer["associations"]]
    if sorted(labels) != sorted(by_label):
        raise ValueError(f"statements for associations {labels}, asked for {sorted(by_label)}")
    paper_id = (
        re.compile("|".join(rf"\b{re.escape(pid)}\b" for pid in gene.paper_ids), re.IGNORECASE)
        if gene.paper_ids
        else None
    )
    problems: list[str] = []
    statements: dict[int, str] = {}
    for entry in answer["associations"]:
        label, statement = entry["association"], entry["statement"].strip()
        if "\n" in statement:
            problems.append(f"line break in the statement of {label}")
        words = len(statement.split())
        if words > MAX_STATEMENT_WORDS:
            problems.append(f"the statement of {label} has {words} words")
        if paper_id is not None and (cited := paper_id.search(statement)) is not None:
            problems.append(f"the statement of {label} cites the paper {cited.group()}")
        statements[by_label[label].id] = statement
    if problems:
        raise ValueError("; ".join(problems))
    return statements


def handle_result(
    result: LlmResult,
    db_path: Path,
    validator: jsonschema.protocols.Validator,
    gene: GeneToExplain,
) -> bool:
    """Record *result* and store its statements if they are valid; True when they were stored.

    Raises RuntimeError, after recording it, when the request was invalid.
    """
    message = result.message
    if result.status == ResultStatus.REFUSED:
        log_refusal(result, f"HGNC:{gene.hgnc_id}")
    if result.status != ResultStatus.SUCCEEDED or message is None:
        with sqlite3.connect(db_path) as conn:
            record_result(conn, result)
        if result.error_type == "invalid_request_error":
            raise RuntimeError(f"invalid explain-grouping request for HGNC:{gene.hgnc_id}")
        return False
    try:
        answer = parse_json_output(message)
        validator.validate(answer)
        statements = parse_statements(answer, gene)
    except (ValueError, jsonschema.ValidationError) as e:
        rejection = invalid_answer_reason(e)
        logger.warning("Invalid grouping statements for HGNC:%d: %s", gene.hgnc_id, rejection)
        with sqlite3.connect(db_path) as conn:
            record_result(conn, result, rejection=rejection)
        return False
    with sqlite3.connect(db_path) as conn:
        record_result(conn, result)
        return store_statements(conn, gene.hgnc_id, statements, message.to_json())


async def run_attempt(
    transport: Transport,
    requests: Sequence[LlmRequest],
    db_path: Path,
    validator: jsonschema.protocols.Validator,
    by_id: dict[str, GeneToExplain],
) -> ExplainOutcome:
    """Send *requests*, store each result as it arrives, and count the outcomes."""
    outcome = ExplainOutcome()
    with Progress() as progress:
        task = progress.add_task("Explaining groupings", total=len(requests))

        def on_result(result: LlmResult) -> None:
            outcome.stored += handle_result(result, db_path, validator, by_id[result.subject])
            outcome.to_fallback += result.goes_to_fallback
            progress.advance(task)

        await send_each(transport, STAGE, requests, on_result, DEFAULT_IMMEDIATE_WORKERS)
    return outcome


async def explain_genes(
    *,
    transport: Transport,
    db_path: Path,
    hgnc_resolver: HgncResolver,
    schema: dict[str, Any],
    system: str,
    gene_template: Template,
    max_retries: int,
    failures_since: datetime,
) -> None:
    """Write the statements of every gene due them, in attempts.

    An attempt makes progress when it stores a gene's statements or sends a gene
    on to FALLBACK_MODEL. Failed answers count from *failures_since*.
    """

    def load_history() -> StageHistory:
        with closing(sqlite3.connect(db_path)) as conn:
            return stage_history(conn, STAGE, failures_since=failures_since)

    validator = jsonschema.Draft202012Validator(schema)
    output_config = json_output_config(schema, EFFORT)
    log_failed_for_good(load_history().failures, STAGE)
    for attempt in range(1, max_retries + 1):
        history = load_history()
        genes = select_genes(db_path, history)
        if not genes:
            logger.info("No genes left to explain")
            return
        logger.info("Attempt %d: explaining the grouping of %d genes", attempt, len(genes))
        requests = [
            build_request(
                gene,
                system,
                gene_text(gene, gene_template, hgnc_resolver.get_symbol(gene.hgnc_id)),
                output_config,
                history.model_for(str(gene.hgnc_id)),
            )
            for gene in genes
        ]
        by_id = {str(gene.hgnc_id): gene for gene in genes}
        outcome = await run_attempt(transport, requests, db_path, validator, by_id)
        logger.info(
            "Attempt %d: stored %d of %d; %d go to the fallback model",
            attempt,
            outcome.stored,
            len(genes),
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
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of attempts for genes whose statements failed or were refused; "
        "the fallback-model request after a refusal is one of them",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Send genes whose answers failed for good in earlier invocations (rejected or "
        "cut off at max_tokens, MAX_FAILED_ANSWERS times) again, until they fail that often "
        "in this invocation",
    ),
) -> None:
    """Write the lumping and splitting statement of each association."""
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        raise typer.Exit(1)
    # With --retry-failed, only failed answers during this invocation count.
    failures_since = datetime.now(UTC) if retry_failed else ALL_TIME

    environment = prompt_environment()
    system = environment.get_template(PROMPT_NAME).render()
    gene_template = environment.get_template(GENE_TEMPLATE_NAME)
    schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())

    logger.info("%s associations without a grouping statement", f"{count_unexplained(db_path):,}")

    async def run() -> None:
        await explain_genes(
            transport=ImmediateTransport(make_client()),
            db_path=db_path,
            hgnc_resolver=HgncResolver.from_file(),
            schema=schema,
            system=system,
            gene_template=gene_template,
            max_retries=max_retries,
            failures_since=failures_since,
        )

    asyncio.run(run())
    logger.info(
        "%s associations still without a grouping statement",
        f"{count_unexplained(db_path):,}",
    )
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
