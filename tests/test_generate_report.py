"""Tests for loading and ordering a report's genes and associations."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message

from palit.assess_genes import STAGE as ASSESS_GENES_STAGE
from palit.extract_evidence import STAGE as EXTRACTION_STAGE
from palit.gencc import MondoRef
from palit.generate_report import (
    FavoriteJournalSections,
    GeneAssessment,
    GeneAssessmentResults,
    PanelValidationResult,
    ReportAssociation,
    build_gene_assessment_results,
    calculate_comprehensive_statistics,
    generate_html_report,
    is_highlighted_new_moi,
    known_gene_sort_key,
    novel_gene_sort_key,
)
from palit.hgnc import HgncResolver
from palit.match_panels import STAGE as MATCH_PANELS_STAGE
from palit.panelapp_client import AllPanelsData, PanelGeneData
from palit.panelapp_integration import (
    ENUM_TO_PANELAPP_MOI,
    MENDELIOME_PANEL_ID,
    PANELAPP_CRITERIA,
)

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
    """GENEA (HGNC:1), known, with four associations in mixed states and a refused paper;
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
            ],
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) VALUES (?, ?, ?, ?)",
            [
                (1, "GENEA", "10.1/a", "recent_evidence"),
                (3, "GENEB", "10.1/b", "recent_evidence"),
                (1, "GENEA", "10.1/refused", "relevance_assessment"),
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
            VALUES (?, ?, ?, 1, 'claude-opus-5-5', ?, ?, ?)
            """,
            [
                ("e1", EXTRACTION_STAGE, "10.1/refused", "refused", "bio", "2026-10-01T01"),
                ("e2", EXTRACTION_STAGE, "10.1/seeded", "refused", None, "2026-10-01T02"),
                ("e3", EXTRACTION_STAGE, "10.1/other", "refused", "bio", "2026-10-01T03"),
                # refused once, then extracted: not a refused paper
                ("e4", EXTRACTION_STAGE, "10.1/a", "refused", "bio", "2026-10-01T04"),
                ("g1", ASSESS_GENES_STAGE, "4", "refused", "bio", "2026-10-01T05"),
                # refused once, then aggregated
                ("g2", ASSESS_GENES_STAGE, "3", "refused", "bio", "2026-10-01T06"),
                ("m1", MATCH_PANELS_STAGE, "10", "succeeded", None, "2026-10-01T07"),
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
    assert genea.unmatched_associations == 1

    assert [(s.panel_name, [r.disease_label for r in s.reasons]) for s in genea.missing_panels] == [
        ("Epilepsy", ["GENEA-related biallelic disease"])
    ]
    assert [
        (s.panel_name, [r.disease_label for r in s.reasons]) for s in genea.existing_panels
    ] == [("Ataxia", ["GENEA-related ataxia variant", "disease A"])]


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


def _render(db_path: Path, results: GeneAssessmentResults) -> str:
    panel_validation = PanelValidationResult(0, 0, [], [], [], 0.0, 0.0)
    return generate_html_report(
        novel_genes=results.novel_genes,
        known_genes=results.known_genes,
        statistics=calculate_comprehensive_statistics(db_path, results, panel_validation),
        panel_validation=panel_validation,
        low_confidence_papers=[],
        manual_download_papers=[],
        favorite_journal_papers=FavoriteJournalSections([], [], [], [], [], [], [], [], []),
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
    assert "Matched panels:" in html
    assert "not matched yet" in html
    assert "Suggested panels:" in html
    assert "Panel Suggestions:" in html


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
        rating=new,
        disputes=[],
        matched_panels=None,
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
        unmatched_associations=1,
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
