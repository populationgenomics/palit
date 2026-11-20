#!/usr/bin/env python3
"""Print a digest of the disaggregation review's aggregations, for checking them.

Per gene: how many of its seeded papers were read and used, ClinGen's curations,
the phenotypes PanelApp Australia lists, the associations of the earlier mock-up,
and the new associations with their grouping rationales.
"""

import json
import sqlite3
import textwrap
from contextlib import closing
from pathlib import Path
from typing import Any

import typer

from palit.hgnc import HgncResolver
from palit.panelapp_integration import calculate_association_rating, panelapp_confidence_to_color

WIDTH = 150


def wrapped(text: str, indent: int) -> str:
    return textwrap.indent(textwrap.fill(text, WIDTH - indent), " " * indent)


def paper_coverage(conn: sqlite3.Connection, hgnc_id: int, used: set[str]) -> None:
    """How the gene's seeded papers fared, and which extracted ones no association uses."""
    rows = conn.execute(
        """
        SELECT p.doi, p.pmid, p.download_status, p.evidence_extraction_json IS NOT NULL,
               substr(p.source_details, 1, instr(p.source_details, ':') - 1), p.title,
               EXISTS (SELECT 1 FROM gene_mentions e WHERE e.paper_doi = p.doi
                       AND e.hgnc_id = gm.hgnc_id AND e.source = 'recent_evidence'),
               (SELECT r.status FROM llm_requests r WHERE r.stage = 'extraction'
                AND r.subject = p.doi ORDER BY r.completed_at DESC LIMIT 1)
        FROM gene_mentions gm JOIN papers p ON p.doi = gm.paper_doi
        WHERE gm.hgnc_id = ? AND gm.source = 'relevance_assessment'
        """,
        (hgnc_id,),
    ).fetchall()
    seeded = len(rows)
    no_pdf = [r for r in rows if r[2] != "downloaded"]
    extracted = [r for r in rows if r[3]]
    with_entity = [r for r in extracted if r[6]]
    unused = [r for r in with_entity if r[0] not in used]
    not_extracted = [r for r in rows if r[2] == "downloaded" and not r[3]]
    print(
        f"  PAPERS seeded {seeded} | no PDF {len(no_pdf)} | extracted {len(extracted)} | "
        f"with evidence for the gene {len(with_entity)} | used by the aggregation "
        f"{len(with_entity) - len(unused)}"
    )
    by_source: dict[str, int] = {}
    for row in no_pdf:
        by_source[row[4]] = by_source.get(row[4], 0) + 1
    if by_source:
        print(f"    no PDF by source: {by_source}")
    for row in not_extracted:
        print(f"    NOT EXTRACTED ({row[7]}): PMID {row[1]} {row[5][:90]}")
    for row in unused:
        print(f"    EVIDENCE BUT UNUSED: PMID {row[1]} {row[5][:90]}")
    no_entity = [r for r in extracted if not r[6]]
    if no_entity:
        print(f"    extracted without a disease entity for the gene: {len(no_entity)}")
        for row in no_entity:
            print(f"      PMID {row[1]} [{row[4]}] {row[5][:90]}")


def mockup_associations(mockup_dir: Path, symbol: str) -> None:
    path = mockup_dir / f"{symbol}.json"
    spec = json.loads(path.read_text())
    print(f"  MOCK-UP v3 ({len(spec['associations'])} associations)")
    for a in spec["associations"]:
        print(f"    - {a['disease_label']} | {a['moi'].split(',')[0]} | {a['rating']}")


def gene_digest(
    conn: sqlite3.Connection, hgnc_id: int, symbol: str, mockup_dir: Path | None
) -> None:
    row = conn.execute(
        "SELECT panelapp_context_json, unassessed_reports_json, paper_id_mapping "
        "FROM gene_aggregations WHERE hgnc_id = ?",
        (hgnc_id,),
    ).fetchone()
    print("=" * WIDTH)
    if row is None:
        print(f"{symbol} (HGNC:{hgnc_id}): NO AGGREGATION")
        paper_coverage(conn, hgnc_id, set())
        return
    context: dict[str, Any] = json.loads(row[0])
    unassessed: list[dict[str, Any]] = json.loads(row[1])
    associations = [
        (json.loads(r[0]), r[1], r[2], r[3])
        for r in conn.execute(
            "SELECT assessment_json, mondo_id, mondo_label, mondo_match FROM associations "
            "WHERE hgnc_id = ? ORDER BY position",
            (hgnc_id,),
        )
    ]
    used = {doi for a, *_ in associations for doi in a["dois"]}
    used |= {doi for report in unassessed for doi in report["dois"]}
    print(
        f"{symbol} (HGNC:{hgnc_id}): {len(associations)} associations, "
        f"{len(unassessed)} unassessed reports, {len(json.loads(row[2]))} papers in the prompt"
    )
    paper_coverage(conn, hgnc_id, used)

    print(f"  CLINGEN ({len(context['clingen_curations'])})")
    for c in context["clingen_curations"]:
        print(
            f"    - {c['disease_label']} ({c['mondo_id']}) | {c['moi']} | {c['classification']} "
            f"| {c['date']} | {c['expert_panel']}"
        )
    print("  PANELAPP ENTRIES")
    for e in context["panel_entries"]:
        target = " [target]" if e["target"] else ""
        print(f"    - {e['panel_name']}{target}: {e['rating']} | {e['moi'].split(',')[0]}")
        print(wrapped("; ".join(e["phenotypes"]), 8))
    if mockup_dir is not None:
        mockup_associations(mockup_dir, symbol)

    print("  NEW ASSOCIATIONS")
    for a, mondo_id, mondo_label, mondo_match in associations:
        relation = a["panelapp_relation"]
        rating = panelapp_confidence_to_color(calculate_association_rating(a))
        print(
            f"    * [{a['inheritance_mode']}] {a['description'][:120]}\n"
            f"      {relation['status']} (PanelApp {relation['existing_rating']}) -> {rating} | "
            f"families reported/qualifying/independent {a['reported_family_count']}/"
            f"{a['family_count']}/{a['independent_family_count']} | {len(a['dois'])} papers | "
            f"dispute {a['dispute_status']} | MONDO {mondo_id} {mondo_label} ({mondo_match}) | "
            f"proposed {a['proposed_disease_name']}"
        )
        print(wrapped("basis: " + relation["basis"], 6))
        print(wrapped("grouping: " + a["grouping_rationale"], 6))
        print(wrapped("summary: " + a["summary"], 6))
    for report in unassessed:
        print(
            f"    ~ unassessed [{report['inheritance_mode']}] {report['phenotype'][:100]} "
            f"({len(report['dois'])} papers)"
        )
        print(wrapped("reason: " + report["reason"], 6))


def main(
    db_path: Path = typer.Option(Path("data/db_disaggregation_review.sqlite")),
    mockup_dir: Path | None = typer.Option(None, help="The mock-up's content directory"),
    hgnc_ids: list[int] | None = typer.Option(None, "--hgnc-id"),
) -> None:
    resolver = HgncResolver.from_file()
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        genes = hgnc_ids or [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT hgnc_id FROM gene_mentions "
                "WHERE source = 'relevance_assessment' ORDER BY hgnc_id"
            )
        ]
        for hgnc_id in sorted(genes, key=resolver.get_symbol):
            gene_digest(conn, hgnc_id, resolver.get_symbol(hgnc_id), mockup_dir)


if __name__ == "__main__":
    typer.run(main)
