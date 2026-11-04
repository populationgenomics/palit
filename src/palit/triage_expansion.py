#!/usr/bin/env python3
"""Triage each run gene's new claims before literature expansion.

Expansion reads a gene's wider literature (the papers PanelApp Australia cites for
it, and tournament picks from the baseline) so that each association is rated on
all its evidence. That reading can only change a finding for some associations.
One request per gene groups the disease entities of its relevant initial papers
into gene-disease-MoI associations and relates each to PanelApp Australia's
curation, in the same terms as aggregation (``panelapp_relation``). Code then
decides, with the report's finding rule, which associations the wider literature
can turn into a finding (:func:`pursued`): every association of a gene on none of
the target panels, every new association, and an existing one whose upgrade
baseline is below GREEN. Monthly runs only.
"""

import asyncio
import json
import logging
import sqlite3
from collections import Counter
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jsonschema
import typer
from anthropic.types import Message
from anthropic.types.output_config_param import OutputConfigParam
from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from palit.assess_genes import (
    PanelAppContext,
    PaperBatchProcessor,
    filter_preprint_evidence,
    format_panel_reviews,
    panel_reviews,
)
from palit.gencc import GenccIndex, fetch_gencc, fetch_mondo
from palit.hgnc import HgncResolver
from palit.llm import (
    ALL_TIME,
    FALLBACK_MODEL,
    MODEL,
    BatchTransport,
    Effort,
    ImmediateTransport,
    LlmRequest,
    LlmResult,
    ResultStatus,
    StageHistory,
    Transport,
    invalid_answer_reason,
    json_output_config,
    log_failed_for_good,
    log_refusal,
    make_client,
    parse_json_output,
    raise_for_invalid_requests,
    record_result,
    stage_history,
)
from palit.llm_usage import print_stage_summary
from palit.panel_entries import PanelEntry, entries_by_gene
from palit.panelapp_client import PanelAppClient, PanelGeneData
from palit.panelapp_integration import (
    NEW_RELATION_STATUSES,
    RATING_CONFIDENCE,
    upgrade_baseline,
)
from palit.papers import generate_paper_ids
from palit.run_corpus import RUN_GENES

app = typer.Typer(help="Triage each run gene's new claims before literature expansion")
logger = logging.getLogger(__name__)

STAGE = "triage_expansion"
EFFORT_BY_MODEL: dict[str, Effort] = {MODEL: "high", FALLBACK_MODEL: "high"}
MAX_TOKENS = 32000
DB_TIMEOUT_SECONDS = 60

PROMPT_PATH = Path("prompts/expansion_triage_prompt.j2")
SCHEMA_PATH = Path("prompts/expansion_triage_schema.json")


def current_rating(panel_data: PanelGeneData, hgnc_id: int) -> int | None:
    """The gene's highest rating on the target panels, None when it is on none of them."""
    if hgnc_id not in panel_data.gene_panel_confidence:
        return None
    return panel_data.highest_confidence(hgnc_id)


def pursued(association: dict[str, Any], gene_rating: int | None) -> bool:
    """Whether the gene's wider literature can make this triaged association a finding.

    *gene_rating* is the gene's highest rating on the target panels (see
    :func:`current_rating`). Follows ``generate_report.association_finding`` for
    a monthly run: every association of a gene on none of the target panels is a
    new disease, a new association is a finding at any rating (a new MoI once
    enough families support it), and an existing association is an upgrade only
    when its corpus rating exceeds its baseline, which no rating can when the
    baseline is GREEN.
    """
    if gene_rating is None:
        return True
    relation = association["panelapp_relation"]
    if relation["status"] in NEW_RELATION_STATUSES:
        return True
    baseline = upgrade_baseline(relation["existing_rating"], gene_rating, scoped=False)
    return baseline < RATING_CONFIDENCE["GREEN"]


@dataclass(frozen=True)
class _TriageItem:
    """One gene's triage prompt and what its answer is checked against."""

    hgnc_id: int
    hgnc_symbol: str
    prompt: str
    paper_id_to_doi: dict[str, str]
    gene_rating: int | None  # the gene's highest rating on the target panels


@dataclass(frozen=True)
class _Preparation:
    """Shared inputs for preparing gene prompts."""

    db_processor: PaperBatchProcessor
    hgnc_resolver: HgncResolver
    panelapp_client: PanelAppClient
    panel_date: str
    entries: dict[int, tuple[PanelEntry, ...]]  # every panel's entries, by gene
    gencc: GenccIndex
    template: Template
    panel_data: PanelGeneData  # the target panels


def prepare_gene(hgnc_id: int, prep: _Preparation) -> _TriageItem | None:
    """The gene's triage prompt, or None when no claim survives the preprint gate."""
    hgnc_symbol = prep.hgnc_resolver.get_symbol(hgnc_id)
    evidence_list, filtered = filter_preprint_evidence(
        prep.db_processor.get_claim_evidence_for_gene(hgnc_id)
    )
    if filtered:
        logger.info(
            "Filtered %d preprint(s) for %s: %s",
            len(filtered),
            hgnc_symbol,
            [f["doi"] for f in filtered],
        )
    if not evidence_list:
        return None

    entries = prep.entries.get(hgnc_id, ())
    context = PanelAppContext(
        gencc=prep.gencc.for_gene(hgnc_id),
        panel_entries=entries,
        panel_reviews=panel_reviews(hgnc_id, prep.panelapp_client, entries),
    )
    paper_id_to_doi, doi_to_paper_id = generate_paper_ids(evidence_list)
    prompt_evidence = [
        {
            "paper_id": doi_to_paper_id[e["doi"]],
            "date": e["date"],
            "title": e["title"],
            "gene_evaluations": e["gene_evaluations"],
        }
        for e in evidence_list
    ]
    # Paper symbols are uppercased; current symbols such as C9orf72 are not.
    aliases = {e["paper_gene_symbol"] for e in evidence_list} - {hgnc_symbol.upper()}
    gene_symbol = (
        f"{hgnc_symbol} (also referred to as: {', '.join(sorted(aliases))} in the papers)"
        if aliases
        else hgnc_symbol
    )
    prompt = prep.template.render(
        gene_symbol=gene_symbol,
        hgnc_symbol=hgnc_symbol,
        evidence_extractions=json.dumps(prompt_evidence, indent=2),
        gencc_terms=context.gencc.terms,
        panel_entries=context.panel_entries,
        reviews_section=format_panel_reviews(context.panel_reviews),
        panel_date=prep.panel_date,
        copies_gencc_terms=False,
    )
    return _TriageItem(
        hgnc_id=hgnc_id,
        hgnc_symbol=hgnc_symbol,
        prompt=prompt,
        paper_id_to_doi=paper_id_to_doi,
        gene_rating=current_rating(prep.panel_data, hgnc_id),
    )


def genes_to_triage(db_path: Path, only: list[int] | None, history: StageHistory) -> list[int]:
    """The run's genes without a triage, except ones skipped for good and in flight."""
    with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT g.hgnc_id FROM ({RUN_GENES}) g
            WHERE g.hgnc_id NOT IN (SELECT hgnc_id FROM expansion_triage)
              AND NOT EXISTS (
                  SELECT 1 FROM llm_requests r
                  WHERE r.stage = ? AND r.subject = CAST(g.hgnc_id AS TEXT)
                    AND r.status = 'pending'
              )
            ORDER BY g.hgnc_id
            """,
            (STAGE,),
        ).fetchall()
    hgnc_ids = [row[0] for row in rows if not history.skipped(str(row[0]))]
    if only is None:
        return hgnc_ids
    wanted = set(only)
    return [h for h in hgnc_ids if h in wanted]


def answer_problems(answer: dict[str, Any], item: _TriageItem) -> list[str]:
    """Why the answer cannot be stored: unknown paper IDs, uncovered papers, or a rating
    on a new association."""
    problems = []
    cited = [pid for a in answer["associations"] for pid in a["paper_ids"]]
    unknown = sorted(set(cited) - item.paper_id_to_doi.keys())
    if unknown:
        problems.append(f"unknown paper IDs {unknown}")
    uncovered = sorted(item.paper_id_to_doi.keys() - set(cited))
    if uncovered:
        problems.append(f"papers in no association {uncovered}")
    for a in answer["associations"]:
        relation = a["panelapp_relation"]
        if relation["status"] in NEW_RELATION_STATUSES and relation["existing_rating"] is not None:
            problems.append(
                f"{a['description']!r} is {relation['status']} with rating "
                f"{relation['existing_rating']}"
            )
    return problems


def store_triage(
    conn: sqlite3.Connection,
    hgnc_id: int,
    associations: list[dict[str, Any]],
    message: Message | None,
) -> None:
    conn.execute(
        "INSERT INTO expansion_triage (hgnc_id, associations_json, triage_raw) VALUES (?, ?, ?)",
        (hgnc_id, json.dumps(associations), message.to_json() if message is not None else None),
    )


def stored_associations(
    answer: dict[str, Any], paper_id_to_doi: dict[str, str], gene_rating: int | None
) -> list[dict[str, Any]]:
    """The answer's associations as stored: paper IDs replaced by DOIs, each with
    whether it is pursued (see :func:`pursued`)."""
    return [
        {
            "description": a["description"],
            "inheritance_mode": a["inheritance_mode"],
            "dois": [paper_id_to_doi[pid] for pid in a["paper_ids"]],
            "panelapp_relation": a["panelapp_relation"],
            "evidence_brief": a["evidence_brief"],
            "pursued": pursued(a, gene_rating),
        }
        for a in answer["associations"]
    ]


# The HGNC IDs of the triaged genes with a pursued association, for use as a
# subquery: `hgnc_id IN ({PURSUED_GENES})`.
PURSUED_GENES = """
    SELECT t.hgnc_id FROM expansion_triage t
    WHERE EXISTS (
        SELECT 1 FROM json_each(t.associations_json) a
        WHERE json_extract(a.value, '$.pursued') IS 1
    )
"""


def pursued_associations(conn: sqlite3.Connection) -> dict[int, list[dict[str, Any]]]:
    """The pursued associations of each triaged gene that has any, by HGNC ID."""
    result = {}
    for hgnc_id, associations_json in conn.execute(
        "SELECT hgnc_id, associations_json FROM expansion_triage ORDER BY hgnc_id"
    ):
        chosen = [a for a in json.loads(associations_json) if a["pursued"]]
        if chosen:
            result[hgnc_id] = chosen
    return result


def untriaged_genes(conn: sqlite3.Connection) -> list[int]:
    """The run's genes without a triage, in HGNC ID order."""
    rows = conn.execute(
        f"""
        SELECT DISTINCT hgnc_id FROM ({RUN_GENES})
        WHERE hgnc_id NOT IN (SELECT hgnc_id FROM expansion_triage)
        ORDER BY hgnc_id
        """
    ).fetchall()
    return [row[0] for row in rows]


@dataclass
class _Outcome:
    stored: int = 0
    refused: int = 0
    to_fallback: int = 0  # the refusals by MODEL: these genes go to FALLBACK_MODEL next
    failed: int = 0


def handle_results(
    results: list[LlmResult],
    items: dict[str, _TriageItem],
    db_path: Path,
    validator: jsonschema.protocols.Validator,
) -> _Outcome:
    outcome = _Outcome()
    invalid_requests: list[LlmResult] = []
    for result in results:
        item = items.get(result.subject)
        message = result.message
        stored = False
        rejection: str | None = None
        if result.status == ResultStatus.REFUSED:
            outcome.refused += 1
            outcome.to_fallback += result.goes_to_fallback
            log_refusal(result, f"HGNC:{result.subject}")
        elif result.status != ResultStatus.SUCCEEDED or message is None:
            outcome.failed += 1
            if result.error_type == "invalid_request_error":
                invalid_requests.append(result)
        elif item is None:
            logger.info("HGNC:%s no longer needs a triage; discarding its result", result.subject)
        else:
            try:
                answer = parse_json_output(message)
                validator.validate(answer)
                problems = answer_problems(answer, item)
            except (ValueError, jsonschema.ValidationError) as e:
                problems = [invalid_answer_reason(e)]
            if problems:
                rejection = "; ".join(problems[:5])
                logger.warning("Rejected triage for %s: %s", item.hgnc_symbol, rejection)
                outcome.failed += 1
            else:
                associations = stored_associations(answer, item.paper_id_to_doi, item.gene_rating)
                with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
                    record_result(conn, result)
                    store_triage(conn, item.hgnc_id, associations, message)
                logger.info(
                    "Stored %s: %s",
                    item.hgnc_symbol,
                    ", ".join(
                        f"{a['panelapp_relation']['status']}"
                        f"/{a['panelapp_relation']['existing_rating']}"
                        for a in associations
                    ),
                )
                stored = True
                outcome.stored += 1
        if not stored:
            with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
                record_result(conn, result, rejection=rejection)
    raise_for_invalid_requests(invalid_requests, lambda result: f"HGNC:{result.subject}")
    return outcome


def build_request(
    item: _TriageItem, configs: dict[str, OutputConfigParam], model: str
) -> LlmRequest:
    return LlmRequest(
        subject=str(item.hgnc_id),
        params={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": item.prompt}],
            "output_config": configs[model],
        },
    )


async def _process_triage(
    *,
    transport: Transport,
    db_path: Path,
    prep: _Preparation,
    schema: dict[str, Any],
    only: list[int] | None,
    max_retries: int,
    failures_since: datetime,
) -> None:
    """Triage every gene due a triage, in attempts (see ``assess_genes._process_assessments``)."""

    def load_history() -> StageHistory:
        with closing(sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS)) as conn:
            return stage_history(conn, STAGE, failures_since=failures_since)

    validator = jsonschema.Draft202012Validator(schema)
    configs = {m: json_output_config(schema, e) for m, e in EFFORT_BY_MODEL.items()}
    items: dict[str, _TriageItem] = {}

    def prepared(hgnc_ids: list[int]) -> list[_TriageItem]:
        ready = []
        for hgnc_id in hgnc_ids:
            item = items.get(str(hgnc_id))
            if item is None:
                prepared_gene = prepare_gene(hgnc_id, prep)
                if prepared_gene is None:
                    logger.info(
                        "No claim of %s survives the preprint gate; nothing to pursue",
                        prep.hgnc_resolver.get_symbol(hgnc_id),
                    )
                    with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
                        store_triage(conn, hgnc_id, [], None)
                    continue
                item = items[str(hgnc_id)] = prepared_gene
            ready.append(item)
        return ready

    resumed = await transport.resume(STAGE)
    if resumed:
        prepared(sorted({int(r.subject) for r in resumed}))
        outcome = handle_results(resumed, items, db_path, validator)
        logger.info("Collected %d results from earlier batches: %s", len(resumed), outcome)

    log_failed_for_good(load_history().failures, STAGE)
    for attempt in range(1, max_retries + 1):
        history = load_history()
        batch = prepared(genes_to_triage(db_path, only, history))
        if not batch:
            logger.info("No genes left to triage")
            return
        logger.info("Attempt %d: triaging %d genes", attempt, len(batch))
        requests = [
            build_request(item, configs, history.model_for(str(item.hgnc_id))) for item in batch
        ]
        outcome = handle_results(await transport.run(STAGE, 1, requests), items, db_path, validator)
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


def log_pursued(db_path: Path) -> None:
    """Log how many triaged genes and associations the wider literature is pursued for."""
    with sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS) as conn:
        rows = conn.execute("SELECT associations_json FROM expansion_triage").fetchall()
    associations: Counter[bool] = Counter()
    genes = 0
    for (associations_json,) in rows:
        flags = [a["pursued"] for a in json.loads(associations_json)]
        associations.update(flags)
        genes += any(flags)
    logger.info(
        "Triaged %d genes: %d with an association to pursue; associations pursued %d, stopped %d",
        len(rows),
        genes,
        associations[True],
        associations[False],
    )


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(Path("data/db.sqlite"), "--db-path", help="Run database"),
    panel_date: str = typer.Option(
        ..., "--panel-date", help="Panel state date (YYYY-MM-DD) of the run's PanelApp snapshot"
    ),
    target_panel_ids: list[int] | None = typer.Option(
        None,
        "--target-panel-ids",
        help="Target panel IDs (repeatable). Defaults to TARGET_PANEL_IDS.",
    ),
    immediate: bool = typer.Option(
        False,
        "--immediate",
        help="Send requests immediately instead of as Message Batches (for evaluation)",
    ),
    hgnc_ids: list[int] | None = typer.Option(
        None, "--hgnc-id", help="Triage only this gene (HGNC ID; repeatable; for evaluation)"
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of attempts for genes whose triage failed, was rejected or was "
        "refused; the fallback-model request after a refusal is one of them",
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Send genes whose answers failed for good in earlier invocations again",
    ),
) -> None:
    """Group each run gene's new claims into associations and relate them to PanelApp
    Australia's curation, to decide which the wider literature is read for."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)
    failures_since = datetime.now(UTC) if retry_failed else ALL_TIME

    schema: dict[str, Any] = json.loads(SCHEMA_PATH.read_text())
    panelapp_client = PanelAppClient(panel_date)
    panel_data = panelapp_client.get_target_panels_genes(target_panel_ids)
    env = Environment(
        loader=FileSystemLoader(PROMPT_PATH.parent), autoescape=False, undefined=StrictUndefined
    )
    prep = _Preparation(
        db_processor=PaperBatchProcessor(db_path),
        hgnc_resolver=HgncResolver.from_file(),
        panelapp_client=panelapp_client,
        panel_date=panel_date,
        entries=entries_by_gene(
            list(panelapp_client.get_all_panel_data().values()), panel_data.panel_ids
        ),
        gencc=fetch_gencc(db_path.parent, fetch_mondo(db_path.parent)),
        template=env.get_template(PROMPT_PATH.name),
        panel_data=panel_data,
    )

    async def run() -> None:
        client = make_client()
        transport: Transport = (
            ImmediateTransport(client) if immediate else BatchTransport(client, db_path)
        )
        await _process_triage(
            transport=transport,
            db_path=db_path,
            prep=prep,
            schema=schema,
            only=hgnc_ids,
            max_retries=max_retries,
            failures_since=failures_since,
        )

    asyncio.run(run())
    log_pursued(db_path)
    print_stage_summary(db_path, STAGE)


if __name__ == "__main__":
    app()
