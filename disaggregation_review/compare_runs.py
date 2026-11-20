#!/usr/bin/env python3
"""Compare the associations of the same genes across several run databases."""

import json
import sqlite3
import textwrap
from contextlib import closing
from pathlib import Path

import typer

from palit.hgnc import HgncResolver
from palit.panelapp_integration import calculate_association_rating, panelapp_confidence_to_color


def associations(db_path: Path, hgnc_id: int) -> list[str]:
    """One line per association and unassessed report of the gene in *db_path*."""
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT assessment_json FROM associations WHERE hgnc_id = ? ORDER BY position",
            (hgnc_id,),
        ).fetchall()
        aggregation = conn.execute(
            "SELECT unassessed_reports_json FROM gene_aggregations WHERE hgnc_id = ?", (hgnc_id,)
        ).fetchone()
    if aggregation is None:
        return ["NO AGGREGATION"]
    lines = []
    for (raw,) in rows:
        a = json.loads(raw)
        rating = panelapp_confidence_to_color(calculate_association_rating(a))
        relation = a["panelapp_relation"]
        lines.append(
            f"[{a['inheritance_mode'][:4]}] {a['description'][:78]:78} | {relation['status']:11} | "
            f"PanelApp {relation['existing_rating'] or '-':5} -> {rating:5} | "
            f"indep {a['independent_family_count']} | {len(a['dois'])} papers"
        )
    for report in json.loads(aggregation[0]):
        lines.append(f"   ~ unassessed: {report['phenotype'][:90]}")
    return lines


def rationales(db_path: Path, hgnc_id: int) -> list[str]:
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            "SELECT assessment_json FROM associations WHERE hgnc_id = ? ORDER BY position",
            (hgnc_id,),
        ).fetchall()
    return [
        f"{json.loads(raw)['description'][:60]}: {json.loads(raw)['grouping_rationale']}"
        for (raw,) in rows
    ]


def main(
    db_paths: list[Path] = typer.Argument(..., help="Run databases, in display order"),
    hgnc_ids: list[int] = typer.Option(..., "--hgnc-id"),
    show_rationales: bool = typer.Option(False, "--rationales"),
) -> None:
    resolver = HgncResolver.from_file()
    for hgnc_id in sorted(hgnc_ids, key=resolver.get_symbol):
        print("=" * 150)
        print(resolver.get_symbol(hgnc_id))
        for db_path in db_paths:
            lines = associations(db_path, hgnc_id)
            print(f"  {db_path.name} ({sum(not line.startswith('   ~') for line in lines)})")
            for line in lines:
                print(f"    {line}")
            if show_rationales:
                for rationale in rationales(db_path, hgnc_id):
                    print(textwrap.indent(textwrap.fill(rationale, 140), "        > "))


if __name__ == "__main__":
    typer.run(main)
