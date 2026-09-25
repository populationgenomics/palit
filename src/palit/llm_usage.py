"""Per-stage Claude usage and cost from the ``llm_requests`` table."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from palit.llm import request_cost

app = typer.Typer(help="Claude API usage recorded in a run database")


@dataclass
class StageUsage:
    stage: str
    model: str
    service_tier: str
    requests: int = 0
    refused: int = 0
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
            """
            SELECT stage, model, COALESCE(service_tier, '-'), status,
                   COALESCE(input_tokens, 0), COALESCE(cache_write_5m_tokens, 0),
                   COALESCE(cache_write_1h_tokens, 0), COALESCE(cache_read_tokens, 0),
                   COALESCE(output_tokens, 0)
            FROM llm_requests
            """
        )
        for stage, model, tier, status, uncached, write_5m, write_1h, read, output in cursor:
            usage = rows.setdefault((stage, model, tier), StageUsage(stage, model, tier))
            usage.requests += 1
            usage.refused += status == "refused"
            usage.errored += status in ("errored", "expired", "canceled")
            usage.pending += status == "pending"
            usage.input_tokens += uncached
            usage.cache_write_tokens += write_5m + write_1h
            usage.cache_read_tokens += read
            usage.output_tokens += output
            usage.cost += request_cost(model, tier, uncached, write_5m, write_1h, read, output)
    return sorted(rows.values(), key=lambda u: (u.stage, u.model, u.service_tier))


@app.command("costs")
def costs(
    db_path: Path = typer.Option(Path("data/db.sqlite"), "--db-path", help="Run database"),
) -> None:
    """Show requests, outcomes, tokens, and USD cost per stage, model, and service tier."""
    usages = summarise_usage(db_path)
    table = Table(title=f"Claude usage in {db_path}")
    for column in ("stage", "model", "tier", "requests", "refused", "errored", "pending"):
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
        "total", "", "", "", "", "", "", "", "", "", "", f"{sum(u.cost for u in usages):,.2f}"
    )
    Console().print(table)
