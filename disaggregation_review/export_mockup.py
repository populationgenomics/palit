#!/usr/bin/env python3
"""Export the review run's associations for the PanelApp UI mock-up.

One JSON file with, per gene, every association as the mock-up shows it: its
name, MoI as PanelApp writes it, rating from the papers, MONDO term, summary
with PMIDs in place of the run's paper IDs, the PMIDs of its papers, and its
mechanisms of disease.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import typer

from palit.generate_report import association_name
from palit.hgnc import HgncResolver
from palit.mechanisms import MechanismAssessment, review_comment
from palit.panelapp_integration import (
    ENUM_TO_PANELAPP_MOI,
    calculate_association_rating,
    panelapp_confidence_to_color,
)
from palit.papers import build_display_ids, replace_paper_ids_for_display

ROSTER_HGNC_IDS = [20, 5959, 4922, 6021, 29882, 6898, 9498, 6772, 11273, 12407, 16700]


def shown_mechanisms(
    mechanisms_raw: str | None, display_ids: dict[str, str]
) -> MechanismAssessment | None:
    """An association's mechanisms with PMIDs in place of the run's paper IDs; None
    before assess-mechanisms has run for it."""
    if mechanisms_raw is None:
        return None
    return MechanismAssessment.from_json(
        replace_paper_ids_for_display(json.loads(mechanisms_raw), display_ids)
    )


def mechanisms_export(assessment: MechanismAssessment | None) -> dict[str, Any] | None:
    """An association's mechanisms as the mock-up shows them: each with its label, its
    confidence term and the rating colour of the term."""
    if assessment is None:
        return None
    return {
        "mechanisms": [
            {
                "mechanism": m.mechanism,
                "label": m.label,
                "confidence": m.confidence,
                "colour": m.colour,
                "statement": m.statement,
            }
            for m in assessment.mechanisms
        ],
        "unknown_note": assessment.unknown_note,
    }


def gene_export(conn: sqlite3.Connection, hgnc_id: int, symbol: str) -> dict[str, Any]:
    mapping_raw, context_raw, unassessed_raw = conn.execute(
        "SELECT paper_id_mapping, panelapp_context_json, unassessed_reports_json "
        "FROM gene_aggregations WHERE hgnc_id = ?",
        (hgnc_id,),
    ).fetchone()
    paper_id_to_doi: dict[str, str] = json.loads(mapping_raw)
    context: dict[str, Any] = json.loads(context_raw)
    doi_to_pmid: dict[str, int] = dict(
        conn.execute("SELECT doi, pmid FROM papers WHERE pmid IS NOT NULL")
    )
    display_ids = build_display_ids(paper_id_to_doi, doi_to_pmid)
    panel_names = {e["panel_id"]: e["panel_name"] for e in context["panel_entries"]}

    associations = []
    for raw, mondo_id, mondo_label, mondo_match, panels_raw, mechanisms_raw in conn.execute(
        "SELECT assessment_json, mondo_id, mondo_label, mondo_match, matched_panels_json, "
        "mechanisms_json FROM associations WHERE hgnc_id = ? ORDER BY position",
        (hgnc_id,),
    ):
        stored: dict[str, Any] = json.loads(raw)
        shown = replace_paper_ids_for_display(stored, display_ids)
        mechanisms = shown_mechanisms(mechanisms_raw, display_ids)
        relation = stored["panelapp_relation"]
        associations.append(
            {
                "disease_label": association_name(stored, context["gencc_terms"]),
                "description": stored["description"],
                "inheritance_mode": stored["inheritance_mode"],
                "inheritance_details": stored["inheritance_details"],
                "panelapp_moi": ENUM_TO_PANELAPP_MOI[stored["inheritance_mode"]],
                "rating": panelapp_confidence_to_color(
                    calculate_association_rating(stored)
                ).lower(),
                "panelapp_status": relation["status"],
                "panelapp_existing_rating": relation["existing_rating"],
                "panelapp_basis": relation["basis"],
                "mondo_id": mondo_id,
                "mondo_label": mondo_label,
                "mondo_match": mondo_match,
                "summary": shown["summary"],
                "mechanism_assessment": mechanisms_export(mechanisms),
                "review_comment": review_comment(shown["summary"], mechanisms),
                "grouping_rationale": shown["grouping_rationale"],
                "pmids": sorted(doi_to_pmid[d] for d in stored["dois"] if d in doi_to_pmid),
                "papers_without_pmid": sorted(d for d in stored["dois"] if d not in doi_to_pmid),
                "independent_family_count": stored["independent_family_count"],
                "matched_panel_ids": [p["panel_id"] for p in json.loads(panels_raw)],
                "matched_panels_holding_gene": [
                    panel_names[p["panel_id"]]
                    for p in json.loads(panels_raw)
                    if p["panel_id"] in panel_names
                ],
            }
        )
    return {
        "gene": symbol,
        "hgnc_id": hgnc_id,
        "associations": associations,
        "unassessed_reports": replace_paper_ids_for_display(
            {"unassessed_reports": json.loads(unassessed_raw)}, display_ids
        )["unassessed_reports"],
        "clingen_curations": [
            {k: c[k] for k in ("disease_label", "mondo_id", "moi", "classification", "date")}
            for c in context["clingen_curations"]
        ],
    }


def main(
    db_path: Path = typer.Option(Path("data/db_disaggregation_review_e2e.sqlite")),
    output: Path = typer.Option(...),
) -> None:
    resolver = HgncResolver.from_file()
    with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as conn:
        genes = [gene_export(conn, h, resolver.get_symbol(h)) for h in ROSTER_HGNC_IDS]
    genes.sort(key=lambda g: g["gene"])
    output.write_text(json.dumps({"source": db_path.name, "genes": genes}, indent=1) + "\n")
    typer.echo(f"{sum(len(g['associations']) for g in genes)} associations of {len(genes)} genes")


if __name__ == "__main__":
    typer.run(main)
