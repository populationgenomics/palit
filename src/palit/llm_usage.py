"""Per-stage Claude usage and cost from the ``llm_requests`` table."""

import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from palit.hgnc import HgncResolver
from palit.llm import FAILED_ANSWER, FALLBACK_MODEL, request_cost, stage_failures

app = typer.Typer(help="Claude API usage recorded in a run database")


@dataclass
class StageUsage:
    stage: str
    model: str
    service_tier: str
    requests: int = 0
    refused: int = 0
    rejected: int = 0  # answers the stage rejected or saw cut off, and asks for again
    errored: int = 0
    pending: int = 0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


def summarise_usage(db_path: Path) -> list[StageUsage]:
    rows: dict[tuple[str, str, str], StageUsage] = {}
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            f"""
            SELECT stage, model, COALESCE(service_tier, '-'), status, {FAILED_ANSWER},
                   COALESCE(input_tokens, 0), COALESCE(cache_write_5m_tokens, 0),
                   COALESCE(cache_write_1h_tokens, 0), COALESCE(cache_read_tokens, 0),
                   COALESCE(output_tokens, 0)
            FROM llm_requests
            """
        )
        for (
            stage,
            model,
            tier,
            status,
            failed_answer,
            uncached,
            write_5m,
            write_1h,
            read,
            output,
        ) in cursor:
            usage = rows.setdefault((stage, model, tier), StageUsage(stage, model, tier))
            usage.requests += 1
            usage.refused += status == "refused"
            usage.rejected += failed_answer
            usage.errored += status in ("errored", "expired", "canceled")
            usage.pending += status == "pending"
            usage.input_tokens += uncached
            usage.cache_write_tokens += write_5m + write_1h
            usage.cache_read_tokens += read
            usage.output_tokens += output
            usage.cost += request_cost(model, tier, uncached, write_5m, write_1h, read, output)
    return sorted(rows.values(), key=lambda u: (u.stage, u.model, u.service_tier))


class SubjectKind(StrEnum):
    """What a stage's ``llm_requests.subject`` identifies."""

    PAPER = "paper"  # a DOI
    GENE = "gene"  # an HGNC ID
    TOURNAMENT_PROMPT = "tournament prompt"  # "{hgnc_id}:{round}:{index}"
    ASSOCIATION = "association"  # an associations.id


# The stage modules import this module, so the stage names are literals here;
# tests/test_llm.py checks them against each module's STAGE.
STAGE_SUBJECT_KINDS: dict[str, SubjectKind] = {
    "relevance": SubjectKind.PAPER,
    "relevance_panelapp": SubjectKind.PAPER,
    "extraction": SubjectKind.PAPER,
    "assess_genes": SubjectKind.GENE,
    "scan_mechanisms": SubjectKind.GENE,
    "expand_literature": SubjectKind.TOURNAMENT_PROMPT,
    "reduce_literature": SubjectKind.TOURNAMENT_PROMPT,
    "map_mondo": SubjectKind.ASSOCIATION,
    "match_panels": SubjectKind.ASSOCIATION,
}


def subject_label(
    conn: sqlite3.Connection, hgnc_resolver: HgncResolver, stage: str, subject: str
) -> str | None:
    """A human-readable name for *subject* of *stage*; None when its row no longer exists.

    A paper is named by its title, a gene or tournament prompt by the gene symbol, and
    an association by "SYMBOL: description". Re-assessing a gene recreates its
    associations under new ids, so an older request's association may be gone.
    """
    match STAGE_SUBJECT_KINDS[stage]:
        case SubjectKind.PAPER:
            title = conn.execute("SELECT title FROM papers WHERE doi = ?", (subject,)).fetchone()
            return None if title is None else title[0]
        case SubjectKind.GENE:
            return hgnc_resolver.get_symbol(int(subject))
        case SubjectKind.TOURNAMENT_PROMPT:
            hgnc_id, _, _ = subject.partition(":")
            return hgnc_resolver.get_symbol(int(hgnc_id))
        case SubjectKind.ASSOCIATION:
            association = conn.execute(
                """
                SELECT hgnc_id, json_extract(assessment_json, '$.description')
                FROM associations WHERE id = ?
                """,
                (int(subject),),
            ).fetchone()
            if association is None:
                return None
            hgnc_id, description = association
            return f"{hgnc_resolver.get_symbol(hgnc_id)}: {description}"


class SubjectOutcome(StrEnum):
    """Where a refused subject stands in its stage, from the stage's latest request about it."""

    RECOVERED = "recovered"  # answered after the refusal, by either model
    REFUSED_FOR_GOOD = "refused for good"  # FALLBACK_MODEL refused it last
    UNANSWERED = "not answered yet"  # MODEL refused it last, or it failed; a rerun retries it


def subject_outcome(latest_status: str, latest_model: str) -> SubjectOutcome:
    if latest_status == "succeeded":
        return SubjectOutcome.RECOVERED
    if latest_status == "refused" and latest_model == FALLBACK_MODEL:
        return SubjectOutcome.REFUSED_FOR_GOOD
    return SubjectOutcome.UNANSWERED


@dataclass(frozen=True)
class Refusal:
    stage: str
    subject: str
    round: int
    model: str
    category: str | None
    label: str | None  # see subject_label
    outcome: SubjectOutcome  # of the subject, not of this request


def list_refusals(
    db_path: Path, hgnc_resolver: HgncResolver, stage: str | None = None
) -> list[Refusal]:
    """Refused requests, oldest first, with their subject's outcome; optionally for one stage.

    The outcome comes from the latest completed request of the stage about the
    subject. In a multi-round stage, an answered round after the refusal counts
    as recovered even while later rounds are still to come.
    """
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            """
            WITH ranked AS (
                SELECT stage, subject, status, model,
                       ROW_NUMBER() OVER (
                           PARTITION BY stage, subject ORDER BY completed_at DESC
                       ) AS recency
                FROM llm_requests
                WHERE status != 'pending' AND (? IS NULL OR stage = ?)
            )
            SELECT r.stage, r.subject, r.round, r.model, r.refusal_category,
                   latest.status, latest.model
            FROM llm_requests r
            JOIN ranked latest
              ON latest.stage = r.stage AND latest.subject = r.subject AND latest.recency = 1
            WHERE r.status = 'refused' AND (? IS NULL OR r.stage = ?)
            ORDER BY r.completed_at
            """,
            (stage, stage, stage, stage),
        ).fetchall()
        return [
            Refusal(
                stage=row_stage,
                subject=subject,
                round=round_no,
                model=model,
                category=category,
                label=subject_label(conn, hgnc_resolver, row_stage, subject),
                outcome=subject_outcome(latest_status, latest_model),
            )
            for (
                row_stage,
                subject,
                round_no,
                model,
                category,
                latest_status,
                latest_model,
            ) in rows
        ]


def outcome_counts(refusals: list[Refusal]) -> dict[SubjectOutcome, int]:
    """The number of refused subjects per outcome, in SubjectOutcome order."""
    counts = Counter({(r.stage, r.subject): r.outcome for r in refusals}.values())
    return {outcome: counts[outcome] for outcome in SubjectOutcome}


def _outcome_summary(refusals: list[Refusal]) -> str:
    return ", ".join(f"{count:,} {outcome}" for outcome, count in outcome_counts(refusals).items())


def _refusal_table(refusals: list[Refusal], title: str) -> Table:
    table = Table(title=title)
    for column in ("stage", "subject", "round", "model", "category", "subject outcome", "label"):
        table.add_column(column)
    for r in refusals:
        table.add_row(
            r.stage,
            r.subject,
            str(r.round),
            r.model,
            r.category or "-",
            r.outcome,
            (r.label or "")[:80],
        )
    return table


@dataclass(frozen=True)
class FailedSubject:
    """A subject whose answers failed for good in its stage (see :func:`palit.llm.stage_failures`)."""

    stage: str
    subject: str
    failures: int
    last_rejection: str
    label: str | None  # see subject_label


def list_failed_for_good(
    db_path: Path, hgnc_resolver: HgncResolver, stage: str | None = None
) -> list[FailedSubject]:
    """The subjects failed for good, by stage and subject; optionally of one stage."""
    found: list[FailedSubject] = []
    with closing(sqlite3.connect(db_path)) as conn:
        stages = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT stage FROM llm_requests WHERE ? IS NULL OR stage = ? ORDER BY stage",
                (stage, stage),
            )
        ]
        for one_stage in stages:
            failures = stage_failures(conn, one_stage)
            for subject in failures.failed_for_good_subjects():
                found.append(
                    FailedSubject(
                        stage=one_stage,
                        subject=subject,
                        failures=failures.counts[subject],
                        last_rejection=failures.last_rejection[subject],
                        label=subject_label(conn, hgnc_resolver, one_stage, subject),
                    )
                )
    return found


def _failed_table(failed: list[FailedSubject], title: str) -> Table:
    table = Table(title=title)
    for column in ("stage", "subject", "failed answers", "last rejection", "label"):
        table.add_column(column)
    for f in failed:
        table.add_row(
            f.stage, f.subject, str(f.failures), f.last_rejection[:100], (f.label or "")[:80]
        )
    return table


def print_stage_summary(db_path: Path, stage: str) -> None:
    """Outcome counts, cost, every refusal and every subject failed for good of *stage*."""
    usages = [u for u in summarise_usage(db_path) if u.stage == stage]
    requests = sum(u.requests for u in usages)
    refused = sum(u.refused for u in usages)
    rejected = sum(u.rejected for u in usages)
    errored = sum(u.errored for u in usages)
    pending = sum(u.pending for u in usages)
    cost = sum(u.cost for u in usages)
    console = Console()
    console.print(
        f"[bold]{stage}[/bold]: {requests:,} requests, {requests - refused - errored - pending:,} "
        f"answered ({rejected:,} of them rejected), [bold]{refused:,} refused[/bold], "
        f"{errored:,} errored, {pending:,} pending; "
        f"${cost:,.2f}"
    )
    hgnc_resolver = HgncResolver.from_file()
    refusals = list_refusals(db_path, hgnc_resolver, stage)
    if refusals:
        console.print(_refusal_table(refusals, f"{stage}: refused requests"))
        console.print(f"{stage}: refused subjects: {_outcome_summary(refusals)}")
    failed = list_failed_for_good(db_path, hgnc_resolver, stage)
    if failed:
        console.print(_failed_table(failed, f"{stage}: subjects failed for good"))


@app.command("refusals")
def refusals(
    db_path: Path = typer.Option(Path("data/db.sqlite"), "--db-path", help="Run database"),
    stage: str | None = typer.Option(None, "--stage", help="Only this stage"),
) -> None:
    """List refused requests with their safety-classifier category and subject outcome.

    Then list the subjects whose answers failed for good, with the last rejection.
    """
    hgnc_resolver = HgncResolver.from_file()
    found = list_refusals(db_path, hgnc_resolver, stage)
    by_category: dict[str, int] = {}
    for r in found:
        by_category[r.category or "-"] = by_category.get(r.category or "-", 0) + 1
    console = Console()
    console.print(_refusal_table(found, f"Refused requests in {db_path}"))
    console.print(
        f"{len(found)} refusals; by category: "
        + (", ".join(f"{k} {v}" for k, v in sorted(by_category.items())) or "none")
    )
    if found:
        console.print(f"Refused subjects: {_outcome_summary(found)}")
    failed = list_failed_for_good(db_path, hgnc_resolver, stage)
    if failed:
        console.print(_failed_table(failed, f"Subjects failed for good in {db_path}"))
    console.print(f"{len(failed)} subjects failed for good")


@app.command("costs")
def costs(
    db_path: Path = typer.Option(Path("data/db.sqlite"), "--db-path", help="Run database"),
) -> None:
    """Show requests, outcomes, tokens, and USD cost per stage, model, and service tier.

    Rejected counts the answers a stage rejected (schema violation, structural
    problem, invalid JSON, output cut off at max_tokens) and asks for again;
    ``llm_requests.rejection`` says why.
    """
    usages = summarise_usage(db_path)
    table = Table(title=f"Claude usage in {db_path}")
    for column in (
        "stage",
        "model",
        "tier",
        "requests",
        "refused",
        "rejected",
        "errored",
        "pending",
    ):
        table.add_column(
            column, justify="left" if column in ("stage", "model", "tier") else "right"
        )
    for column in ("input", "cache write", "cache read", "output", "USD"):
        table.add_column(column, justify="right")
    for u in usages:
        table.add_row(
            u.stage,
            u.model,
            u.service_tier,
            f"{u.requests:,}",
            f"{u.refused:,}",
            f"{u.rejected:,}",
            f"{u.errored:,}",
            f"{u.pending:,}",
            f"{u.input_tokens:,}",
            f"{u.cache_write_tokens:,}",
            f"{u.cache_read_tokens:,}",
            f"{u.output_tokens:,}",
            f"{u.cost:,.2f}",
        )
    table.add_section()
    table.add_row(
        "total", "", "", "", "", "", "", "", "", "", "", "", f"{sum(u.cost for u in usages):,.2f}"
    )
    Console().print(table)
