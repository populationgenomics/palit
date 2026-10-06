"""Tests for loading and ordering a report's genes and associations."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message

from palit.assess_genes import STAGE as ASSESS_GENES_STAGE
from palit.assess_relevance import Refusal, refused_assessment
from palit.extract_evidence import STAGE as EXTRACTION_STAGE
from palit.gencc import MondoRef
from palit.generate_report import (
    FavoriteJournalSections,
    GeneAssessment,
    GeneAssessmentResults,
    PanelValidationResult,
    QuoteRef,
    ReportAssociation,
    StageState,
    build_gene_assessment_results,
    calculate_comprehensive_statistics,
    format_inheritance,
    generate_html_report,
    is_highlighted_new_moi,
    known_gene_sort_key,
    load_low_confidence_irrelevant_papers,
    load_panel_publications_validation,
    load_quote_refs,
    novel_gene_sort_key,
    write_viewer_package,
)
from palit.hgnc import HgncResolver
from palit.llm import FALLBACK_MODEL, MODEL
from palit.map_mondo import STAGE as MAP_MONDO_STAGE
from palit.match_panels import STAGE as MATCH_PANELS_STAGE
from palit.panelapp_client import AllPanelsData, PanelGeneData, PanelPublications
from palit.panelapp_integration import (
    ENUM_TO_PANELAPP_MOI,
    MENDELIOME_PANEL_ID,
    PANELAPP_CRITERIA,
)
from palit.papers import doi_to_path

ROOT = Path(__file__).resolve().parents[1]
ATAXIA_PANEL_ID = 9001
EPILEPSY_PANEL_ID = 9002


def _criteria(green: bool) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "result": green or name in ("criterion_D", "criterion_E"),
            "rationale": "r",
            "confidence": "HIGH",
            "citations": [],
        }
        for name in PANELAPP_CRITERIA
    ]


def _association(
    description: str,
    *,
    status: str = "existing",
    independent: int | None = 3,
    green: bool = False,
    mondo_id: str | None = None,
    dispute_status: str = "None",
    inheritance_mode: str = "Monoallelic",
    dois: tuple[str, ...] = ("10.1/a",),
) -> str:
    """A stored association: dois instead of paper IDs, criteria as a list."""
    return json.dumps(
        {
            "description": description,
            "inheritance_mode": inheritance_mode,
            "inheritance_details": "",
            "grouping_rationale": "g",
            "dois": list(dois),
            "existing_association_mondo_id": mondo_id,
            "proposed_disease_name": None if mondo_id else f"GENEA-related {description}",
            "panelapp_relation": {
                "status": status,
                "existing_rating": "GREEN" if status == "existing" else None,
                "basis": "b",
            },
            "dispute_status": dispute_status,
            "patient_count": 4,
            "reported_family_count": independent,
            "family_count": independent,
            "independent_family_count": independent,
            "count_reduction_reasoning": "none",
            "disease_mechanism": "NR",
            "citations": [],
            "evidence_weakening_factors": [],
            "evidence_assessments": _criteria(green),
            "summary": f"Smith2024 reports {description}.",
        }
    )


def _mondo_raw(rationale: str) -> str:
    answer = {"mondo_id": "MONDO:0000200", "label": "ataxia", "match": "broader"}
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(answer | {"rationale": rationale})}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "stop_details": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    ).to_json()


GENCC_ROW = {
    "mondo_id": "MONDO:0000001",
    "disease_title": "disease A",
    "moi_title": "Autosomal dominant",
    "moi": "Monoallelic",
    "classification": "Strong",
    "rating": "GREEN",
    "date": "2025-01-17",
    "definition": "",
    "obsolete": True,
    "replaced_by": [{"mondo_id": "MONDO:0000009", "label": "disease A, replacement"}],
}

DISPUTES = [
    {
        "submitter": "ClinGen",
        "mondo_id": "MONDO:0000005",
        "disease_title": "disease X",
        "moi_title": "Autosomal dominant",
        "status": "Disputed",
        "date": "2024-01-01",
        "pmids": [],
        "rationale": "",
    },
    {
        "submitter": "G2P",
        "mondo_id": "MONDO:0000001",
        "disease_title": "disease A",
        "moi_title": "Autosomal dominant",
        "status": "Disputed",
        "date": "2023-01-01",
        "pmids": [],
        "rationale": "",
    },
    {
        "submitter": "Ambry",
        "mondo_id": "MONDO:0000001",
        "disease_title": "disease A",
        "moi_title": "Autosomal dominant",
        "status": "Refuted",
        "date": "2022-01-01",
        "pmids": [],
        "rationale": "",
    },
]


def _context(gencc_rows: list[dict[str, Any]], disputes: list[dict[str, Any]]) -> str:
    return json.dumps(
        {"gencc_rows": gencc_rows, "disputes": disputes, "panel_entries": [], "all_panels": False}
    )


def _extraction(hgnc_id: int) -> str:
    return json.dumps({"gene_evaluations": [{"hgnc_id": hgnc_id, "disease_entities": []}]})


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """GENEA (HGNC:1), known, with four associations in mixed states and refused papers;
    GENEB (HGNC:3), novel, with one association; GENEC (HGNC:4) refused by assess-genes."""
    path = tmp_path / "run.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.executemany(
            """
            INSERT INTO papers (doi, pmid, title, source, source_type, source_details,
                                evidence_extraction_json)
            VALUES (?, ?, ?, 'pubmed', ?, ?, ?)
            """,
            [
                ("10.1/a", 111, "Paper A", "initial", "f.xml", _extraction(1)),
                ("10.1/b", 222, "Paper B", "initial", "f.xml", _extraction(3)),
                ("10.1/refused", 333, "Refused paper", "initial", "f.xml", None),
                ("10.1/seeded", None, "Seeded refused paper", "expansion", "panelapp:1", None),
                ("10.1/other", 444, "Other gene's refused paper", "initial", "f.xml", None),
                ("10.1/due", 555, "Paper due its fallback", "initial", "f.xml", None),
            ],
        )
        conn.execute(
            "UPDATE papers SET relevance_assessment_json = ? WHERE doi = '10.1/b'",
            (
                json.dumps(
                    {
                        "relevant": True,
                        "screen": {
                            "relevant": True,
                            "confidence": "HIGH",
                            "rationale": "New families.",
                            "associations": [],
                        },
                        "panelapp_check": None,
                    }
                ),
            ),
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) VALUES (?, ?, ?, ?)",
            [
                (1, "GENEA", "10.1/a", "recent_evidence"),
                (3, "GENEB", "10.1/b", "recent_evidence"),
                (1, "GENEA", "10.1/refused", "relevance_assessment"),
                (1, "GENEA", "10.1/due", "relevance_assessment"),
                (3, "GENEB", "10.1/other", "relevance_assessment"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO gene_aggregations (hgnc_id, assessment_raw, paper_id_mapping,
                panelapp_context_json, existing_panel_reviews_json, unassessed_reports_json,
                quality_concerns_json)
            VALUES (?, '{}', ?, ?, ?, ?, '[]')
            """,
            [
                (
                    1,
                    json.dumps({"Smith2024": "10.1/a"}),
                    _context([GENCC_ROW], DISPUTES),
                    json.dumps([{"panel_id": MENDELIOME_PANEL_ID, "evaluations": []}]),
                    json.dumps(
                        [
                            {
                                "phenotype": "p",
                                "inheritance_mode": "NR",
                                "dois": ["10.1/a"],
                                "reason": "Smith2024 reports one proband.",
                            }
                        ]
                    ),
                ),
                (3, json.dumps({"Jones2025": "10.1/b"}), _context([], []), None, "[]"),
            ],
        )
        conn.executemany(
            """
            INSERT INTO associations (id, hgnc_id, position, assessment_json, mondo_id,
                mondo_label, mondo_match, mondo_raw, matched_panels_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                # RED existing, reusing the obsolete GenCC row, disputed
                (
                    10,
                    1,
                    0,
                    _association(
                        "disease A",
                        independent=1,
                        mondo_id="MONDO:0000001",
                        dispute_status="Disputed",
                    ),
                    "MONDO:0000001",
                    "disease A",
                    "panelapp_gencc",
                    None,
                    json.dumps(
                        [
                            {"panel_id": ATAXIA_PANEL_ID, "rationale": "ataxia"},
                            {"panel_id": 4242, "rationale": "panel gone"},
                        ]
                    ),
                ),
                # AMBER new_moi with 2 families: highlighted
                (
                    11,
                    1,
                    1,
                    _association("biallelic disease", status="new_moi", independent=2),
                    None,
                    None,
                    None,
                    None,
                    json.dumps([{"panel_id": EPILEPSY_PANEL_ID, "rationale": "seizures"}]),
                ),
                # GREEN new_disease on a broader term
                (
                    12,
                    1,
                    2,
                    _association("ataxia variant", status="new_disease", green=True),
                    "MONDO:0000200",
                    "ataxia",
                    "broader",
                    _mondo_raw("no exact term"),
                    json.dumps([{"panel_id": ATAXIA_PANEL_ID, "rationale": "cerebellar"}]),
                ),
                # AMBER new_disease with more families than association 11: sorts before it
                (
                    13,
                    1,
                    3,
                    _association("other disease", status="new_disease", independent=5),
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
                (
                    20,
                    3,
                    0,
                    _association("B disease", status="new_disease", green=True, dois=("10.1/b",)),
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO llm_requests (custom_id, stage, subject, round, model, status,
                                      refusal_category, completed_at)
            VALUES (?, ?, ?, 1, ?, ?, ?, ?)
            """,
            [
                # refused by both models, the fallback's category shown
                (
                    "e0",
                    EXTRACTION_STAGE,
                    "10.1/refused",
                    MODEL,
                    "refused",
                    "cyber",
                    "2026-10-01T00",
                ),
                (
                    "e1",
                    EXTRACTION_STAGE,
                    "10.1/refused",
                    FALLBACK_MODEL,
                    "refused",
                    "bio",
                    "2026-10-01T01",
                ),
                (
                    "e2",
                    EXTRACTION_STAGE,
                    "10.1/seeded",
                    FALLBACK_MODEL,
                    "refused",
                    None,
                    "2026-10-01T02",
                ),
                (
                    "e3",
                    EXTRACTION_STAGE,
                    "10.1/other",
                    FALLBACK_MODEL,
                    "refused",
                    "bio",
                    "2026-10-01T03",
                ),
                # refused by MODEL only: due its fallback request, not a refused paper
                ("e5", EXTRACTION_STAGE, "10.1/due", MODEL, "refused", "bio", "2026-10-01T03"),
                # refused once, then extracted by the fallback model: not a refused paper
                ("e4", EXTRACTION_STAGE, "10.1/a", MODEL, "refused", "bio", "2026-10-01T04"),
                (
                    "e6",
                    EXTRACTION_STAGE,
                    "10.1/a",
                    FALLBACK_MODEL,
                    "succeeded",
                    None,
                    "2026-10-01T05",
                ),
                ("e7", EXTRACTION_STAGE, "10.1/b", MODEL, "succeeded", None, "2026-10-01T05"),
                ("g1", ASSESS_GENES_STAGE, "4", FALLBACK_MODEL, "refused", "bio", "2026-10-01T05"),
                # refused by MODEL only, without an aggregation: not a refused gene
                ("g3", ASSESS_GENES_STAGE, "2", MODEL, "refused", "bio", "2026-10-01T05"),
                # refused once, then aggregated by the fallback model
                ("g2", ASSESS_GENES_STAGE, "3", MODEL, "refused", "bio", "2026-10-01T06"),
                ("g4", ASSESS_GENES_STAGE, "3", FALLBACK_MODEL, "succeeded", None, "2026-10-01T07"),
                ("m1", MATCH_PANELS_STAGE, "10", MODEL, "succeeded", None, "2026-10-01T07"),
            ],
        )
    return path


def _load(db_path: Path, hgnc_resolver: HgncResolver) -> GeneAssessmentResults:
    target = PanelGeneData(
        panel_ids=[MENDELIOME_PANEL_ID],
        gene_confidence={1: 2},
        gene_panel_mapping={1: {MENDELIOME_PANEL_ID}},
        gene_moi={1: "MONOALLELIC, autosomal or pseudoautosomal, NOT imprinted"},
    )
    all_panels = AllPanelsData(
        gene_to_panels={1: {MENDELIOME_PANEL_ID, ATAXIA_PANEL_ID}},
        panel_names={
            MENDELIOME_PANEL_ID: "Mendeliome",
            ATAXIA_PANEL_ID: "Ataxia",
            EPILEPSY_PANEL_ID: "Epilepsy",
        },
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return build_gene_assessment_results(conn, hgnc_resolver, target, all_panels)


@pytest.fixture
def results(db_path: Path, hgnc_resolver: HgncResolver) -> GeneAssessmentResults:
    return _load(db_path, hgnc_resolver)


@pytest.mark.parametrize(
    ("status", "independent", "highlighted"),
    [
        ("new_moi", 2, True),
        ("new_moi", 5, True),
        ("new_moi", 1, False),
        ("new_moi", None, False),
        ("new_disease", 5, False),
        ("existing", 5, False),
    ],
)
def test_new_moi_highlight_needs_two_independent_families(
    status: str, independent: int | None, highlighted: bool
) -> None:
    assessment = json.loads(_association("d", status=status, independent=independent))
    assert is_highlighted_new_moi(assessment) is highlighted


def test_novel_and_known_split(results: GeneAssessmentResults) -> None:
    assert [g.hgnc_symbol for g in results.novel_genes] == ["GENEB"]
    assert [g.hgnc_symbol for g in results.known_genes] == ["GENEA"]
    genea = results.known_genes[0]
    assert genea.existing_rating == 2
    assert genea.new_rating == 3  # the top association rating
    assert genea.has_highlighted_new_moi
    assert results.novel_genes[0].existing_rating is None


def test_associations_by_rating_then_independent_families(results: GeneAssessmentResults) -> None:
    genea = results.known_genes[0]
    assert [(a.id, a.rating) for a in genea.associations] == [(12, 3), (13, 2), (11, 2), (10, 1)]


def test_disease_labels_and_mondo_states(results: GeneAssessmentResults) -> None:
    by_id = {a.id: a for a in results.known_genes[0].associations}

    reused = by_id[10]
    assert reused.disease_label == "disease A"
    assert reused.mondo is not None
    assert (reused.mondo.match, reused.mondo.obsolete, reused.mondo.replaced_by) == (
        "panelapp_gencc",
        True,
        (MondoRef("MONDO:0000009", "disease A, replacement"),),
    )

    broader = by_id[12]
    assert broader.disease_label == "GENEA-related ataxia variant"
    assert broader.mondo is not None
    assert (broader.mondo.match, broader.mondo.label, broader.mondo.rationale) == (
        "broader",
        "ataxia",
        "no exact term",
    )

    unmapped = by_id[11]
    assert unmapped.mondo is None
    assert unmapped.disease_label == "GENEA-related biallelic disease"


def test_disputes_with_the_association_status_same_term_first(
    results: GeneAssessmentResults,
) -> None:
    by_id = {a.id: a for a in results.known_genes[0].associations}
    assert [(d.submitter, d.same_term) for d in by_id[10].disputes] == [
        ("G2P", True),
        ("ClinGen", False),
    ]
    assert by_id[12].disputes == []


def test_matched_panels_per_association_and_union(results: GeneAssessmentResults) -> None:
    genea = results.known_genes[0]
    by_id = {a.id: a for a in genea.associations}
    reused_matches = by_id[10].matched_panels
    assert reused_matches is not None
    assert [(m.panel_name, m.gene_on_panel) for m in reused_matches] == [("Ataxia", True)]
    assert by_id[13].matched_panels is None
    assert {a.id: a.panel_matching for a in genea.associations} == {
        10: StageState.STORED,
        11: StageState.STORED,
        12: StageState.STORED,
        13: StageState.NOT_RUN,
    }
    assert genea.unmatched_by_state == {StageState.NOT_RUN: 1}

    assert [(s.panel_name, [r.disease_label for r in s.reasons]) for s in genea.missing_panels] == [
        ("Epilepsy", ["GENEA-related biallelic disease"])
    ]
    assert [
        (s.panel_name, [r.disease_label for r in s.reasons]) for s in genea.existing_panels
    ] == [("Ataxia", ["GENEA-related ataxia variant", "disease A"])]


def _add_requests(
    db_path: Path, stage: str, subject: str, requests: list[tuple[str, str | None, str]]
) -> None:
    """Requests of *stage* for *subject*, as (status, completed_at, model), in insertion order."""
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO llm_requests (custom_id, stage, subject, round, model, status, completed_at)
            VALUES (?, ?, ?, 1, ?, ?, ?)
            """,
            [
                (f"{stage}-{subject}-{i}", stage, subject, model, status, completed_at)
                for i, (status, completed_at, model) in enumerate(requests)
            ],
        )


STATE_CASES = [
    ([], StageState.NOT_RUN),
    ([("refused", "2026-10-01T10", FALLBACK_MODEL)], StageState.REFUSED),
    # refused by MODEL only: the next attempt goes to FALLBACK_MODEL
    ([("refused", "2026-10-01T10", MODEL)], StageState.FAILED),
    # the second answer succeeded but was rejected, so nothing was stored
    (
        [("errored", "2026-10-01T10", MODEL), ("succeeded", "2026-10-01T11", MODEL)],
        StageState.FAILED,
    ),
    ([("expired", "2026-10-01T10", MODEL)], StageState.FAILED),
    # the latest by completion time, not by insertion
    (
        [("refused", "2026-10-01T11", FALLBACK_MODEL), ("errored", "2026-10-01T10", MODEL)],
        StageState.REFUSED,
    ),
]


@pytest.mark.parametrize(("requests", "state"), STATE_CASES)
def test_panel_matching_state_of_an_unmatched_association(
    db_path: Path,
    hgnc_resolver: HgncResolver,
    requests: list[tuple[str, str | None, str]],
    state: StageState,
) -> None:
    _add_requests(db_path, MATCH_PANELS_STAGE, "13", requests)
    genea = _load(db_path, hgnc_resolver).known_genes[0]
    (association,) = [a for a in genea.associations if a.id == 13]
    assert association.panel_matching == state
    assert genea.unmatched_by_state == {state: 1}


@pytest.mark.parametrize(("requests", "state"), STATE_CASES)
def test_mondo_mapping_state_of_an_unmapped_association(
    db_path: Path,
    hgnc_resolver: HgncResolver,
    requests: list[tuple[str, str | None, str]],
    state: StageState,
) -> None:
    _add_requests(db_path, MAP_MONDO_STAGE, "13", requests)
    genea = _load(db_path, hgnc_resolver).known_genes[0]
    states = {a.id: a.mondo_mapping for a in genea.associations}
    assert states == {
        10: StageState.STORED,  # reuses a GenCC row
        11: StageState.NOT_RUN,
        12: StageState.STORED,
        13: state,
    }


def test_refused_papers_and_genes(results: GeneAssessmentResults) -> None:
    genea = results.known_genes[0]
    assert sorted((p.doi, p.category) for p in genea.refused_papers) == [
        ("10.1/refused", "bio"),
        ("10.1/seeded", None),
    ]
    assert results.novel_genes[0].refused_papers[0].doi == "10.1/other"
    assert [(g.hgnc_id, g.hgnc_symbol) for g in results.refused_genes] == [(4, "GENEC")]


def test_display_ids_and_gene_level_blocks(results: GeneAssessmentResults) -> None:
    genea = results.known_genes[0]
    assert genea.associations[0].assessment["summary"].startswith("PMID 111 reports")
    assert genea.unassessed_reports[0]["reason"] == "PMID 111 reports one proband."
    assert [p.doi for p in genea.contributing_papers] == ["10.1/a"]


def test_prefill_unions_the_associations_in_report_order(results: GeneAssessmentResults) -> None:
    prefill = json.loads(results.known_genes[0].prefill_json)
    assert (prefill["form_type"], prefill["panel_id"]) == ("review", MENDELIOME_PANEL_ID)
    assert prefill["rating"] == "GREEN"
    assert prefill["moi"] == ENUM_TO_PANELAPP_MOI["Monoallelic"]
    assert prefill["phenotypes"].split(";") == [
        "GENEA-related ataxia variant, MONDO:0000200",  # broader: the proposed name
        "GENEA-related other disease",
        "GENEA-related biallelic disease",
        "disease A, replacement, MONDO:0000009",  # the obsolete GenCC term's replacement
    ]
    assert prefill["publications"] == "111"
    assert [section.split("\n")[0] for section in prefill["comments"].split("\n\n")] == [
        "GENEA-related ataxia variant, MONDO:0000200 (broader MONDO term: ataxia)"
        " | Monoallelic | GREEN in this corpus",
        "GENEA-related other disease | Monoallelic | AMBER in this corpus",
        "GENEA-related biallelic disease | Monoallelic | AMBER in this corpus",
        "disease A, replacement, MONDO:0000009 | Monoallelic | RED in this corpus",
    ]
    assert prefill["comments"].split("\n")[1] == "PMID 111 reports ataxia variant."
    novel = json.loads(results.novel_genes[0].prefill_json)
    assert (novel["form_type"], novel["panel_id"]) == ("add", MENDELIOME_PANEL_ID)
    assert novel["publications"] == "222"


NO_PANEL_VALIDATION = PanelValidationResult(0, 0, [], [], [], [], 0.0, 0.0)


def _render(
    db_path: Path,
    results: GeneAssessmentResults,
    panel_validation: PanelValidationResult = NO_PANEL_VALIDATION,
) -> str:
    return generate_html_report(
        novel_genes=results.novel_genes,
        known_genes=results.known_genes,
        statistics=calculate_comprehensive_statistics(db_path, results, panel_validation),
        panel_validation=panel_validation,
        low_confidence_papers=[],
        manual_download_papers=[],
        favorite_journal_papers=FavoriteJournalSections([], [], [], [], [], [], [], [], [], []),
        template_dir=ROOT / "templates",
        panel_date="2026-10-01",
        report_id="test",
        target_panel_ids=[MENDELIOME_PANEL_ID],
        target_panel_names=results.target_panel_names,
        refused_genes=results.refused_genes,
        panels_matched=results.panels_matched,
        panelapp_integration=False,
    )


def test_panel_matching_shown_once_match_panels_has_run(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    assert results.panels_matched
    html = _render(db_path, results)
    assert "Suggested panels:" in html
    assert "Panel Suggestions:" in html
    assert "Associations without a matched panel" in html
    assert (
        "GENEA-related other disease (Monoallelic):</em> not matched yet: "
        "match-panels has not run for this association"
    ) in html
    assert "<em>GENEA-related biallelic disease (Monoallelic):</em> seizures" in html


def test_report_says_which_associations_were_refused_or_failed(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    refused = ("refused", "2026-10-01T10", FALLBACK_MODEL)
    _add_requests(db_path, MATCH_PANELS_STAGE, "13", [refused])
    _add_requests(db_path, MATCH_PANELS_STAGE, "20", [("errored", "2026-10-01T10", MODEL)])
    _add_requests(db_path, MAP_MONDO_STAGE, "13", [refused])
    _add_requests(db_path, MAP_MONDO_STAGE, "20", [("succeeded", "2026-10-01T10", MODEL)])
    html = _render(db_path, _load(db_path, hgnc_resolver))
    assert "Opus 5.5 and its fallback Sonnet 5.5 both refused match-panels" in html
    assert "match-panels gave no valid answer in any attempt so far" in html
    assert "(1 of 4 associations not matched: 1 refused)" in html
    assert "(1 of 1 associations not matched: 1 failed)" in html
    assert "No MONDO term: map-mondo refused" in html
    assert "No MONDO term: mapping failed" in html
    assert "No MONDO term yet" in html  # association 11


def test_panel_matching_omitted_when_match_panels_never_ran(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM llm_requests WHERE stage = ?", (MATCH_PANELS_STAGE,))
    results = _load(db_path, hgnc_resolver)
    assert not results.panels_matched
    html = _render(db_path, results)
    for text in ("Matched panels:", "not matched yet", "Suggested panels:", "Panel Suggestions:"):
        assert text not in html


def test_gene_without_associations_is_left_out(db_path: Path, hgnc_resolver: HgncResolver) -> None:
    """REPEAT1 (HGNC:2) is aggregated, but its only report went to unassessed_reports."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO papers (doi, pmid, title, source, source_type, source_details,
                                evidence_extraction_json)
            VALUES ('10.1/r', 555, 'Paper R', 'pubmed', 'initial', 'f.xml', ?)
            """,
            (_extraction(2),),
        )
        conn.execute(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source)"
            " VALUES (2, 'REPEAT1', '10.1/r', 'recent_evidence')"
        )
        conn.execute(
            """
            INSERT INTO gene_aggregations (hgnc_id, assessment_raw, paper_id_mapping,
                panelapp_context_json, unassessed_reports_json, quality_concerns_json)
            VALUES (2, '{}', ?, ?, ?, '[]')
            """,
            (
                json.dumps({"Brown2025": "10.1/r"}),
                _context([], []),
                json.dumps(
                    [
                        {
                            "phenotype": "r",
                            "inheritance_mode": "NR",
                            "dois": ["10.1/r"],
                            "reason": "Brown2025 reports one proband.",
                        }
                    ]
                ),
            ),
        )
    results = _load(db_path, hgnc_resolver)
    assert [g.hgnc_id for g in results.novel_genes] == [3]
    assert [g.hgnc_id for g in results.known_genes] == [1]
    statistics = calculate_comprehensive_statistics(db_path, results, NO_PANEL_VALIDATION)
    assert (statistics.total_genes_assessed, statistics.novel_genes_count) == (2, 1)
    html = _render(db_path, results)
    assert 'id="novel-gene-3"' in html
    assert "REPEAT1" not in html
    assert "gene-2" not in html


def _sortable(symbol: str, existing: int | None, new: int, highlighted: bool) -> GeneAssessment:
    """A gene with one association, a highlighted new-MoI one when *highlighted*."""
    association = ReportAssociation(
        id=1,
        position=0,
        assessment=json.loads(
            _association("d", status="new_moi" if highlighted else "existing", independent=2)
        ),
        disease_label="d",
        mondo=None,
        mondo_mapping=StageState.NOT_RUN,
        rating=new,
        disputes=[],
        matched_panels=None,
        panel_matching=StageState.NOT_RUN,
    )
    return GeneAssessment(
        hgnc_id=1,
        hgnc_symbol=symbol,
        associations=[association],
        unassessed_reports=[],
        quality_concerns=[],
        existing_rating=existing,
        new_rating=new,
        aggregate_moi="Monoallelic",
        contributing_papers=[],
        variant_frequencies=[],
        missing_panels=[],
        existing_panels=[],
        prefill_json="{}",
        existing_panel_reviews=[],
        refused_papers=[],
    )


def test_novel_sort_rating_then_highlighted_new_moi_then_symbol() -> None:
    genes = [
        _sortable("AAA", None, 2, False),
        _sortable("BBB", None, 3, False),
        _sortable("CCC", None, 2, True),
        _sortable("ABC", None, 2, False),
    ]
    assert [g.hgnc_symbol for g in sorted(genes, key=novel_gene_sort_key)] == [
        "BBB",
        "CCC",
        "AAA",
        "ABC",
    ]


def test_known_sort_existing_rating_then_corpus_rating_then_highlighted_new_moi() -> None:
    genes = [
        _sortable("GREEN1", 3, 3, False),
        _sortable("AMBER_PLAIN", 2, 3, False),
        _sortable("AMBER_NEW_MOI", 2, 3, True),
        _sortable("AMBER_LOW", 2, 1, True),
        _sortable("RED1", 1, 1, False),
    ]
    assert [g.hgnc_symbol for g in sorted(genes, key=known_gene_sort_key)] == [
        "RED1",
        "AMBER_NEW_MOI",
        "AMBER_PLAIN",
        "AMBER_LOW",
        "GREEN1",
    ]


def test_known_sort_rejects_a_gene_without_existing_rating() -> None:
    with pytest.raises(ValueError, match="no rating on the target panels"):
        known_gene_sort_key(_sortable("NOVEL", None, 3, False))


def _screen(relevant: bool, confidence: str) -> dict[str, Any]:
    return {
        "relevant": relevant,
        "confidence": confidence,
        "rationale": f"Screen rationale {confidence}.",
        "associations": [{"gene_symbol": "GENEA", "disease": "d"}] if relevant else [],
    }


def test_papers_both_models_refused_are_neither_passes_nor_rejections(
    db_path: Path, results: GeneAssessmentResults, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Panel publications: a screen miss, and one paper refused at each relevance level."""
    assessments = {
        "10.1/miss": {"relevant": False, "screen": _screen(False, "LOW"), "panelapp_check": None},
        "10.1/refused-screen": refused_assessment(Refusal("screen", FALLBACK_MODEL, "bio"), None),
        "10.1/refused-check": refused_assessment(
            Refusal("panelapp_check", FALLBACK_MODEL, None), _screen(True, "HIGH")
        ),
    }
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO papers (doi, pmid, title, source, source_type, relevance_assessment_json)
            VALUES (?, ?, ?, 'pubmed', 'initial', ?)
            """,
            [
                (doi, 900 + i, f"Title {doi}", json.dumps(assessment))
                for i, (doi, assessment) in enumerate(assessments.items())
            ],
        )
    monkeypatch.setattr(
        "palit.generate_report.get_current_panel_publications",
        lambda panel_ids: PanelPublications(pmids={900, 901, 902}, dois=set()),
    )

    validation = load_panel_publications_validation(db_path, [MENDELIOME_PANEL_ID])
    assert [p.doi for p in validation.screen_misses] == ["10.1/miss"]
    assert sorted(p.doi for p in validation.refused) == [
        "10.1/refused-check",
        "10.1/refused-screen",
    ]
    assert (validation.true_positives, validation.check_rejections) == ([], [])
    assert validation.panel_papers_in_db == 3
    assert validation.screen_sensitivity_pct == 0.0  # of the one assessed paper
    low_confidence = load_low_confidence_irrelevant_papers(db_path)
    assert [p.doi for p in low_confidence] == ["10.1/miss"]

    html = _render(db_path, results, validation)
    assert "Refused by Both Models (2)" in html
    assert "Refused by both models (left out of the sensitivity): 2" in html
    assert (
        "Opus 5.5 and its fallback Sonnet 5.5 both refused the scope screen of this paper "
        f"({FALLBACK_MODEL}, category: bio)"
    ) in html
    assert "both refused the PanelApp check of this paper" in html
    assert "category: none given" in html
    # The paper refused at the PanelApp check shows the screen it passed.
    assert "Screen rationale HIGH." in html


def test_quotes_an_aggregation_adds_follow_the_extraction_quotes_unlocated(
    tmp_path: Path,
) -> None:
    """A citation quote not copied from the extraction is listed, and opens as not located."""
    db_path = tmp_path / "run.sqlite"
    box = {"page": 3, "top": 1, "left": 1, "bottom": 2, "right": 2}
    association = json.loads(_association("disease A"))
    association["citations"] = [
        {"doi": "10.1/a", "quote": "Located quote.", "commentary": "c"},
        {"doi": "10.1/a", "quote": "Aggregate quote.", "commentary": "c"},
        {"doi": "10.1/b", "quote": "Other paper's quote.", "commentary": "c"},
    ]
    concerns = [
        {
            "concern": "c",
            "dois": ["10.1/a"],
            "citations": [{"doi": "10.1/a", "quote": "Concern quote."}],
        }
    ]
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.execute(
            "INSERT INTO papers (doi, title, source) VALUES ('10.1/a', 'Paper A', 'pubmed')"
        )
        conn.execute(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (1, 'GENEA', '10.1/a', 'recent_evidence')"
        )
        conn.execute(
            "INSERT INTO gene_aggregations (hgnc_id, assessment_raw, paper_id_mapping, "
            "panelapp_context_json, unassessed_reports_json, quality_concerns_json) "
            "VALUES (1, '{}', '{}', '{}', '[]', ?)",
            (json.dumps(concerns),),
        )
        conn.execute(
            "INSERT INTO associations (hgnc_id, position, assessment_json) VALUES (1, 0, ?)",
            (json.dumps(association),),
        )
        conn.executemany(
            "INSERT INTO citation_locations (paper_doi, quote, bboxes_json) VALUES ('10.1/a', ?, ?)",
            [("Located quote.", json.dumps([box])), ("Unlocated quote.", "[]")],
        )
        refs = load_quote_refs(conn.cursor(), "10.1/a")
    assert refs == {
        "Located quote.": QuoteRef(index=0, page=3),
        "Unlocated quote.": QuoteRef(index=1, page=None),
        "Aggregate quote.": QuoteRef(index=2, page=None),
        "Concern quote.": QuoteRef(index=3, page=None),
    }

    papers_dir, viewer_dir, output_dir = tmp_path / "papers", tmp_path / "viewer", tmp_path / "out"
    papers_dir.mkdir()
    viewer_dir.mkdir()
    output_dir.mkdir()
    doi_to_path("10.1/a", papers_dir, ".pdf").write_bytes(b"%PDF")
    write_viewer_package(output_dir, db_path, ["10.1/a"], papers_dir, viewer_dir)
    written = json.loads(doi_to_path("10.1/a", output_dir / "citations", ".json").read_text())
    assert [(q["quote"], q["bboxes"]) for q in written["quotes"]] == [
        ("Located quote.", [box]),
        ("Unlocated quote.", []),
        ("Aggregate quote.", []),
        ("Concern quote.", []),
    ]


def _article(html: str, anchor: str) -> str:
    """The HTML of the gene article with id *anchor*."""
    start = html.index(f'<article class="gene-assessment" id="{anchor}">')
    depth, position = 0, start
    for match in re.finditer(r"<article\b|</article>", html[start:]):
        depth += 1 if match.group() == "<article" else -1
        position = start + match.end()
        if depth == 0:
            break
    return html[start:position]


def test_association_headings_show_the_relation_then_the_corpus_rating(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    html = _render(db_path, results)
    known = _article(html, "known-gene-1")
    assert known.count(">new disease</span>") == 2
    assert known.count('data-title="PanelApp Australia: b">GREEN</span>') == 1
    assert known.count(">\N{HEAVY PLUS SIGN} new MoI</span>") == 1
    # The novel gene's header already says "Not in panel", so its new disease shows no badge.
    novel = _article(html, "novel-gene-3")
    assert "new disease" not in novel
    assert novel.count('class="rating-arrow"') == 1  # the gene header's
    for removed in ("Basis:", "As described:", "Grouping:", "Matched panels:", "by Sonnet 5.5"):
        assert removed not in html


@pytest.mark.parametrize(
    ("details", "shown"),
    [
        ("", "Monoallelic"),
        ("reduced penetrance", "Monoallelic (reduced penetrance)"),
        (
            "mosaic; imprinted, paternal allele expressed",
            "Monoallelic (mosaic; imprinted, paternal allele expressed)",
        ),
        ("de novo", "Monoallelic (de novo)"),
        (
            "reduced penetrance; consanguineous family",
            "Monoallelic (reduced penetrance; consanguineous family)",
        ),
    ],
)
def test_inheritance_details_show_verbatim(details: str, shown: str) -> None:
    assert format_inheritance("Monoallelic", details) == shown
