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
    ExistingPanelReviews,
    FavoriteJournalSections,
    Finding,
    FindingKind,
    GeneAssessment,
    GeneAssessmentResults,
    PanelValidationResult,
    QuoteRef,
    ReportAssociation,
    StageState,
    association_finding,
    association_sort_key,
    build_gene_assessment_results,
    build_report_config,
    calculate_comprehensive_statistics,
    format_inheritance,
    gene_findings,
    generate_html_report,
    group_genes,
    is_highlighted_new_moi,
    load_low_confidence_irrelevant_papers,
    load_panel_publications_validation,
    load_quote_refs,
    report_order,
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
    PrefillData,
    calculate_association_rating,
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
    panelapp_rating: str | None = "GREEN",
    independent: int | None = 3,
    green: bool = False,
    mondo_id: str | None = None,
    dispute_status: str = "None",
    inheritance_mode: str = "Monoallelic",
    dois: tuple[str, ...] = ("10.1/a",),
    variants: tuple[str, ...] = (),
    segregation: str = "NR",
    concerns: tuple[str, ...] = (),
) -> str:
    """A stored association: dois instead of paper IDs, criteria as a list.

    *panelapp_rating* is PanelApp's rating of an existing association. Each of its
    quality *concerns* cites Paper A.
    """
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
                "existing_rating": panelapp_rating if status == "existing" else None,
                "basis": "b",
            },
            "dispute_status": dispute_status,
            "patient_count": 4,
            "reported_family_count": independent,
            "family_count": independent,
            "independent_family_count": independent,
            "count_reduction_reasoning": "none",
            "segregation": segregation,
            "functional": "NR",
            "disease_mechanism": "NR",
            "variants": list(variants),
            "evidence_weakening_factors": [],
            "evidence_assessments": _criteria(green),
            "summary": f"Smith2024 reports {description}.",
            "quality_concerns": [
                {
                    "concern": concern,
                    "citations": [{"doi": "10.1/a", "quote": f"Quote for {concern}"}],
                }
                for concern in concerns
            ],
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


def _normalization(variant_id: str, hgvs_c: str, hgvs_p: str) -> str:
    return json.dumps(
        {
            "hgvs_c": hgvs_c,
            "hgvs_p": hgvs_p,
            "original_text": variant_id,
            "total_normalizations": 1,
            "selected_for_max_ac": True,
        }
    )


def _variant(variant_id: str, het: int, hom: int) -> tuple[str, str, str]:
    """A variant row's (variant_id, normalization, gnomad) as extract-evidence stores them."""
    gnomad = {
        "ac": het + 2 * hom,
        "an": 1000,
        "homozygote_count": hom,
        "heterozygote_count": het,
        "hemizygote_count": 0,
        "faf95_popmax": 0.001,
        "faf95_popmax_population": "nfe",
    }
    return (
        variant_id,
        _normalization(variant_id, f"c.{variant_id}", f"p.{variant_id}"),
        json.dumps(gnomad),
    )


# GENEA's variants: common in heterozygotes, common in homozygotes, and one in no association
COMMON_HET = "1-100-A-G"
COMMON_HOM = "1-200-C-T"
UNLISTED = "1-300-G-A"
GENEA_VARIANTS = [
    _variant(COMMON_HET, 40, 2),
    _variant(COMMON_HOM, 5, 20),
    _variant(UNLISTED, 100, 0),
]


def _extraction(hgnc_id: int) -> str:
    return json.dumps({"gene_evaluations": [{"hgnc_id": hgnc_id, "disease_entities": []}]})


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """GENEA (HGNC:1), known, with four associations in mixed states, variants and refused
    papers; GENEB (HGNC:3), novel, with one association; GENEC (HGNC:4) refused by
    assess-genes."""
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
                panelapp_context_json, existing_panel_reviews_json, unassessed_reports_json)
            VALUES (?, '{}', ?, ?, ?, ?)
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
                        variants=(COMMON_HET, COMMON_HOM),
                    ),
                    "MONDO:0000001",
                    "disease A",
                    "panelapp_gencc",
                    None,
                    json.dumps(
                        [
                            {"panel_id": ATAXIA_PANEL_ID, "rationale": "ataxia"},
                            {"panel_id": 4242, "rationale": "panel gone"},
                            {"panel_id": EPILEPSY_PANEL_ID, "rationale": "epilepsy too"},
                        ]
                    ),
                ),
                # AMBER new_moi with 2 families: highlighted
                (
                    11,
                    1,
                    1,
                    _association(
                        "biallelic disease",
                        status="new_moi",
                        independent=2,
                        inheritance_mode="Biallelic",
                        variants=(COMMON_HET, COMMON_HOM, COMMON_HOM),
                    ),
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
                    _association(
                        "ataxia variant",
                        status="new_disease",
                        green=True,
                        segregation="Segregates in both families.",
                    ),
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
            INSERT INTO variant_frequencies (variant_id, hgnc_id, paper_doi, quote,
                                             normalization, gnomad)
            VALUES (?, 1, '10.1/a', 'q', ?, ?)
            """,
            GENEA_VARIANTS,
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


def _load(
    db_path: Path, hgnc_resolver: HgncResolver, genea_rating: int = 2
) -> GeneAssessmentResults:
    """The fixture's genes, with GENEA on the Mendeliome at *genea_rating*."""
    target = PanelGeneData(
        panel_ids=[MENDELIOME_PANEL_ID],
        gene_panel_confidence={1: {MENDELIOME_PANEL_ID: genea_rating}},
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
    assert genea.current_rating == 2
    assert genea.top_rating == 3
    # Strongest first: the GREEN new disease, then the AMBER new disease and new MoI in
    # association order; the RED existing association is rated GREEN in PanelApp.
    assert genea.findings == [
        Finding(FindingKind.NEW_DISEASE, 3, 12, None),
        Finding(FindingKind.NEW_DISEASE, 2, 13, None),
        Finding(FindingKind.NEW_MOI, 2, 11, None),
    ]
    geneb = results.novel_genes[0]
    assert geneb.current_rating is None
    assert geneb.findings == [Finding(FindingKind.NEW_DISEASE, 3, 20, None)]


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


def test_matched_panels_the_gene_is_not_on_come_first(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    genea = results.known_genes[0]
    by_id = {a.id: a for a in genea.associations}
    reused_matches = by_id[10].matched_panels
    assert reused_matches is not None
    # Panel 4242 is gone from PanelApp, so its match is skipped.
    assert [(m.panel_name, m.gene_on_panel) for m in reused_matches] == [
        ("Epilepsy", False),
        ("Ataxia", True),
    ]
    assert by_id[13].matched_panels is None
    assert {a.id: a.panel_matching for a in genea.associations} == {
        10: StageState.STORED,
        11: StageState.STORED,
        12: StageState.STORED,
        13: StageState.NOT_RUN,
    }
    # Associations 10 and 11 both match Epilepsy, which the gene is not on; Ataxia it is on.
    statistics = calculate_comprehensive_statistics(db_path, results, NO_PANEL_VALIDATION)
    assert statistics.total_panel_suggestions == 1


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


def test_review_prefill_unions_the_findings_in_report_order(
    results: GeneAssessmentResults,
) -> None:
    """GENEA's curated RED association 10 is no finding, so its review leaves it out."""
    prefill = json.loads(results.known_genes[0].prefill_json)
    assert (prefill["form_type"], prefill["panel_id"]) == ("review", MENDELIOME_PANEL_ID)
    assert prefill["rating"] == "GREEN"
    assert prefill["moi"] == ENUM_TO_PANELAPP_MOI["Monoallelic_and_biallelic"]
    assert prefill["phenotypes"].split(";") == [
        "GENEA-related ataxia variant, MONDO:0000200",  # broader: the proposed name
        "GENEA-related other disease",
        "GENEA-related biallelic disease",
    ]
    assert prefill["publications"] == "111"
    assert [section.split("\n")[0] for section in prefill["comments"].split("\n\n")] == [
        "GENEA-related ataxia variant, MONDO:0000200 (broader MONDO term: ataxia)"
        " | Monoallelic | GREEN on the papers reviewed",
        "GENEA-related other disease | Monoallelic | AMBER on the papers reviewed",
        "GENEA-related biallelic disease | Biallelic | AMBER on the papers reviewed",
    ]
    assert prefill["comments"].split("\n")[1] == "PMID 111 reports ataxia variant."


def test_review_prefill_of_a_green_gene_lists_only_its_amber_finding(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A GREEN gene with a curated GREEN association and a new AMBER disease: the review
    describes the new disease alone, rated GREEN, the gene's rating on the panel."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM associations WHERE hgnc_id = 1")
    _add_genea_associations(
        db_path,
        [
            (30, 0, _association("curated disease", green=True, inheritance_mode="Biallelic")),
            (31, 1, _association("new disease", status="new_disease", independent=2)),
        ],
    )
    genea = _load(db_path, hgnc_resolver, genea_rating=3).known_genes[0]
    assert [a.id for a in genea.finding_associations] == [31]
    prefill = genea.prefill
    assert (prefill.form_type, prefill.rating) == ("review", "GREEN")
    assert prefill.phenotypes == "GENEA-related new disease"
    # The MoI covers the curated biallelic association too: a review replaces the evaluation's MoI.
    assert prefill.moi == ENUM_TO_PANELAPP_MOI["Monoallelic_and_biallelic"]
    assert prefill.comments == (
        "GENEA-related new disease | Monoallelic | AMBER on the papers reviewed\n"
        "PMID 111 reports new disease."
    )


def test_review_prefill_of_a_gene_without_findings_lists_all_its_associations(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A review that described the findings alone would be empty."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM associations WHERE id IN (11, 12, 13)")
    genea = _load(db_path, hgnc_resolver).known_genes[0]
    assert genea.findings == []
    assert (genea.prefill.form_type, genea.prefill.rating) == ("review", "AMBER")
    assert genea.prefill.phenotypes == "disease A, replacement, MONDO:0000009"


def test_add_prefill_lists_every_association(db_path: Path, hgnc_resolver: HgncResolver) -> None:
    """GENEB's curated association of unknown PanelApp rating is no finding of a new gene,
    yet the add form lists it with the new disease."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO associations (id, hgnc_id, position, assessment_json) VALUES (21, 3, 1, ?)",
            (_association("curated B", panelapp_rating=None, dois=("10.1/b",)),),
        )
    geneb = _load(db_path, hgnc_resolver).novel_genes[0]
    assert [f.association_id for f in geneb.findings] == [20]
    assert (geneb.prefill.form_type, geneb.prefill.panel_id) == ("add", MENDELIOME_PANEL_ID)
    assert geneb.prefill.phenotypes == "GENEA-related B disease;GENEA-related curated B"
    assert geneb.prefill.publications == "222"


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


def _panels(section: str) -> str:
    """The Panels block of an association section: a collapsed list of its matched
    panels, or a note why there is none."""
    match = re.search(r'<(details|p) class="association-panels">.*?</\1>', section, re.DOTALL)
    assert match is not None
    return match.group()


def _strip_tags(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html)


def _panel_summary(section: str) -> str:
    """The summary line of an association's collapsed Panels list, tags stripped."""
    panels = _panels(section)
    assert panels.startswith('<details class="association-panels">')
    match = re.search(r"<summary>(.*?)</summary>", panels, re.DOTALL)
    assert match is not None
    return _strip_tags(match.group(1))


def _panel_items(section: str) -> list[str]:
    """The Panels list items of an association section, tags stripped."""
    return [
        _strip_tags(item) for item in re.findall(r"<li>(.*?)</li>", _panels(section), re.DOTALL)
    ]


def test_each_association_lists_its_panels_after_its_papers(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    assert results.panels_matched
    html = _render(db_path, results)
    reused = _association_section(html, 10)
    assert reused.index("<strong>Papers:</strong>") < reused.index("<strong>Panels:</strong>")
    # The summary names the panels in the list's order, noting those the gene is already on
    assert _panel_summary(reused) == "Panels: Epilepsy, Ataxia (already on)"
    assert (
        '<summary><strong>Panels:</strong> Epilepsy, Ataxia <span class="muted">(already on)</span>'
        "</summary>"
    ) in reused
    assert _panel_items(reused) == [
        "Epilepsy: epilepsy too",
        "Ataxia (gene already on panel): ataxia",
    ]
    assert (
        f'<a href="https://panelapp-aus.org/panels/{EPILEPSY_PANEL_ID}" target="_blank">'
        "<strong>Epilepsy</strong></a>: epilepsy too"
    ) in reused
    assert _panel_summary(_association_section(html, 11)) == "Panels: Epilepsy"
    assert _panel_items(_association_section(html, 11)) == ["Epilepsy: seizures"]
    assert _panel_summary(_association_section(html, 12)) == "Panels: Ataxia (already on)"
    assert _panel_items(_association_section(html, 12)) == [
        "Ataxia (gene already on panel): cerebellar"
    ]
    # Without a match, the note stands in a plain paragraph, not a collapsed list
    assert _panels(_association_section(html, 13)) == (
        '<p class="association-panels"><strong>Panels:</strong> <span class="muted">'
        "not matched yet: match-panels has not run for this association</span></p>"
    )
    assert "<li><strong>Panel Suggestions:</strong> 1</li>" in html
    assert "Suggested panels" not in html


def test_association_matched_to_no_panel_says_so(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE associations SET matched_panels_json = '[]' WHERE id = 13")
    panels = _panels(_association_section(_render(db_path, _load(db_path, hgnc_resolver)), 13))
    assert panels == (
        '<p class="association-panels"><strong>Panels:</strong> '
        '<span class="muted">matched to no panel</span></p>'
    )


def test_report_says_which_associations_were_refused_or_failed(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    refused = ("refused", "2026-10-01T10", FALLBACK_MODEL)
    _add_requests(db_path, MATCH_PANELS_STAGE, "13", [refused])
    _add_requests(db_path, MATCH_PANELS_STAGE, "20", [("errored", "2026-10-01T10", MODEL)])
    _add_requests(db_path, MAP_MONDO_STAGE, "13", [refused])
    _add_requests(db_path, MAP_MONDO_STAGE, "20", [("succeeded", "2026-10-01T10", MODEL)])
    html = _render(db_path, _load(db_path, hgnc_resolver))
    assert (
        "not matched: Opus 5.5 and its fallback Sonnet 5.5 both refused match-panels for this "
        "association, so it is skipped from now on"
    ) in _panels(_association_section(html, 13))
    assert (
        "not matched: match-panels gave no valid answer in any attempt so far; a rerun retries it"
    ) in _panels(_association_section(html, 20))
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
    for text in ('class="association-panels"', "Panels:</strong>", "Panel Suggestions:"):
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
                panelapp_context_json, unassessed_reports_json)
            VALUES (2, '{}', ?, ?, ?)
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
    assert (statistics.total_genes_assessed, statistics.new_genes_count) == (2, 1)
    html = _render(db_path, results)
    assert 'id="novel-gene-3"' in html
    assert "REPEAT1" not in html
    assert "gene-2" not in html


def _report_association(association_id: int, assessment_json: str) -> ReportAssociation:
    assessment = json.loads(assessment_json)
    return ReportAssociation(
        id=association_id,
        position=association_id,
        assessment=assessment,
        disease_label="d",
        mondo=None,
        mondo_mapping=StageState.NOT_RUN,
        rating=calculate_association_rating(assessment),
        disputes=[],
        variants=[],
        matched_panels=None,
        panel_matching=StageState.NOT_RUN,
    )


@pytest.mark.parametrize(
    ("association", "current_rating", "finding"),
    [
        # A new disease of a gene on a target panel, at any corpus rating
        (_association("d", status="new_disease", independent=1), 3, (FindingKind.NEW_DISEASE, 1)),
        # A new gene's new disease too
        (_association("d", status="new_disease", green=True), None, (FindingKind.NEW_DISEASE, 3)),
        # A new MoI only once highlighted
        (_association("d", status="new_moi", independent=2), 3, (FindingKind.NEW_MOI, 2)),
        (_association("d", status="new_moi", independent=1), 1, None),
        (_association("d", status="new_moi", independent=2), None, (FindingKind.NEW_MOI, 2)),
        # An upgrade over PanelApp's rating of the association, whatever the gene's rating
        (_association("d", panelapp_rating="AMBER", green=True), 2, (FindingKind.UPGRADE, 3, 2)),
        (_association("d", panelapp_rating="RED", independent=2), 3, (FindingKind.UPGRADE, 2, 1)),
        (_association("d", panelapp_rating="GREEN", green=True), 1, None),
        (_association("d", panelapp_rating="AMBER", independent=2), 1, None),
        # Without PanelApp's rating of the association, over the gene's current rating
        (_association("d", panelapp_rating=None, independent=2), 1, (FindingKind.UPGRADE, 2, 1)),
        (_association("d", panelapp_rating=None, green=True), 3, None),
        (_association("d", panelapp_rating=None, green=True), None, None),
    ],
)
def test_association_finding(
    association: str,
    current_rating: int | None,
    finding: tuple[FindingKind, int] | tuple[FindingKind, int, int] | None,
) -> None:
    expected = None
    if finding is not None:
        kind, rating, *from_rating = finding
        expected = Finding(kind, rating, 7, from_rating[0] if from_rating else None)
    assert association_finding(_report_association(7, association), current_rating) == expected


def _gene(
    hgnc_id: int,
    current_rating: int | None,
    associations: list[str],
    *,
    reviewed: bool = True,
) -> GeneAssessment:
    """A gene GENE{hgnc_id} with these stored associations, in report order."""
    rated = sorted(
        (_report_association(i, a) for i, a in enumerate(associations)),
        key=association_sort_key,
    )
    findings = gene_findings(rated, current_rating)
    evaluations = [{"rating": "GREEN"}] if reviewed else []
    return GeneAssessment(
        hgnc_id=hgnc_id,
        hgnc_symbol=f"GENE{hgnc_id}",
        associations=report_order(rated, findings),
        unassessed_reports=[],
        current_rating=current_rating,
        findings=findings,
        contributing_papers=[],
        unassociated_variants=[],
        prefill=PrefillData("review", MENDELIOME_PANEL_ID, "HGNC:1", "GREEN", "", None, "", "", ""),
        existing_panel_reviews=(
            []
            if current_rating is None
            else [ExistingPanelReviews(panel_id=MENDELIOME_PANEL_ID, evaluations=evaluations)]
        ),
        refused_papers=[],
    )


GREEN_NEW_DISEASE = _association("d", status="new_disease", green=True)
AMBER_NEW_DISEASE = _association("d", status="new_disease", independent=2)
AMBER_NEW_MOI = _association("d", status="new_moi", independent=2)
RED_NEW_DISEASE = _association("d", status="new_disease", independent=1)
CURATED_GREEN = _association("d", green=True)
CURATED_RED = _association("d", independent=1)


def test_genes_are_placed_once_by_their_strongest_finding() -> None:
    """Within a group, genes go by HGNC symbol, whatever their findings and rating."""
    genes = {
        # GENE100 sorts before GENE11, which has more findings
        "new_amber": _gene(100, None, [AMBER_NEW_DISEASE]),
        "new_amber_moi": _gene(11, None, [AMBER_NEW_DISEASE, AMBER_NEW_MOI]),
        "new_red": _gene(12, None, [RED_NEW_DISEASE]),
        # A GREEN gene with two findings comes before a RED one with a GREEN new disease
        "green_gene": _gene(20, 3, [GREEN_NEW_DISEASE, AMBER_NEW_DISEASE]),
        "red_gene": _gene(21, 1, [GREEN_NEW_DISEASE]),
        # A GREEN gene with only an AMBER new disease: its best finding is AMBER
        "amber_finding": _gene(22, 3, [CURATED_GREEN, AMBER_NEW_DISEASE]),
        "red_finding": _gene(23, 2, [RED_NEW_DISEASE]),
        "unreviewed": _gene(30, 3, [CURATED_GREEN], reviewed=False),
        "reviewed": _gene(31, 1, [CURATED_RED]),
    }
    novel = [g for g in genes.values() if g.is_novel]
    known = [g for g in genes.values() if not g.is_novel]
    groups = group_genes(novel, known, {MENDELIOME_PANEL_ID})

    def layout(gene_groups: list[Any]) -> list[tuple[str, str, list[int], bool, bool]]:
        return [
            (g.key, g.label, [gene.hgnc_id for gene in g.genes], g.collapsed, g.preselected)
            for g in gene_groups
        ]

    assert layout(groups.new_genes) == [
        ("new-amber", "New gene AMBER", [100, 11], False, True),
        ("new-red", "New gene RED", [12], False, True),
    ]
    assert layout(groups.with_findings) == [
        ("finding-green", "Best finding GREEN", [20, 21], False, True),
        ("finding-amber", "Best finding AMBER", [22], False, True),
        ("finding-red", "Best finding RED", [23], False, True),
    ]
    assert layout(groups.without_findings) == [
        ("unreviewed", "Unreviewed on target panel", [30], False, False),
        ("reviewed", "Already reviewed", [31], True, False),
    ]
    assert [(f.kind, f.rating) for f in genes["green_gene"].findings] == [
        (FindingKind.NEW_DISEASE, 3),
        (FindingKind.NEW_DISEASE, 2),
    ]
    # A new gene's new diseases are findings, but it is placed by its top association
    assert {
        name: [(f.kind, f.rating) for f in genes[name].findings]
        for name in ("new_amber", "new_amber_moi", "new_red")
    } == {
        "new_amber": [(FindingKind.NEW_DISEASE, 2)],
        "new_amber_moi": [(FindingKind.NEW_DISEASE, 2), (FindingKind.NEW_MOI, 2)],
        "new_red": [(FindingKind.NEW_DISEASE, 1)],
    }


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
    association["evidence_assessments"][0]["citations"] = [
        {"doi": "10.1/a", "quote": "Located quote.", "commentary": "c"},
        {"doi": "10.1/a", "quote": "Aggregate quote.", "commentary": "c"},
        {"doi": "10.1/b", "quote": "Other paper's quote.", "commentary": "c"},
    ]
    association["quality_concerns"] = [
        {"concern": "c", "citations": [{"doi": "10.1/a", "quote": "Concern quote."}]}
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
            "panelapp_context_json, unassessed_reports_json) "
            "VALUES (1, '{}', '{}', '{}', '[]')"
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
    assert known.count(">\N{HEAVY PLUS SIGN} new MoI</span>") == 1
    assert known.count('class="rating-arrow"') == 3  # the three findings only
    # The novel gene's header already says "Not in panel", so its new disease's heading
    # shows the rating alone; the finding is a header badge only.
    novel = _article(html, "novel-gene-3")
    assert ">new disease</span>" not in novel
    assert 'class="rating-arrow"' not in novel
    assert novel.count("new disease GREEN</a>") == 1
    for removed in ("Basis:", "As described:", "Grouping:", "Matched panels:", "by Sonnet 5.5"):
        assert removed not in html


def _toc(html: str) -> str:
    return html[html.index('<nav class="toc">') : html.index("</nav>")]


def _toc_section(html: str, key: str) -> str:
    """The ToC section with data-section-key *key*."""
    toc = _toc(html)
    start = toc.index(f'data-section-key="{key}"')
    return toc[start : toc.index("</ul>", start)]


def _gene_header(article: str) -> str:
    return article[: article.index("</h3>")]


def test_known_gene_is_placed_once_under_its_best_finding_with_its_findings(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    """GENEA has a GREEN and an AMBER new disease and an AMBER new MoI."""
    html = _render(db_path, results)
    toc = _toc(html)
    assert toc.count('href="#known-gene-1"') == 1
    assert html.count('id="known-gene-1"') == 1
    section = _toc_section(html, "finding-green")
    assert 'data-section-label="Best finding GREEN" data-section-default' in section
    assert re.findall(r'class="finding-chip rating-(\w+)" href="#association-(\d+)"', section) == [
        ("green", "12"),
        ("amber", "13"),
        ("amber", "11"),
    ]
    assert 'data-title="new disease GREEN:&#10;• GENEA-related ataxia variant">D</a>' in section
    assert 'data-title="new MoI AMBER:&#10;• GENEA-related biallelic disease">M</a>' in section
    # The body places it under its group's heading, after the new genes
    body = html[html.index("</nav>") :]
    assert body.index('id="novel-gene-3"') < body.index('id="group-finding-green"')
    assert body.index('id="group-finding-green"') < body.index('id="known-gene-1"')

    header = _gene_header(_article(html, "known-gene-1"))
    assert re.findall(
        r'<a class="finding-badge rating-(\w+)" href="#association-(\d+)"[^>]*>([^<]+)</a>',
        header,
    ) == [
        ("green", "12", "new disease GREEN"),
        ("amber", "13", "new disease AMBER"),
        ("amber", "11", "new MoI AMBER"),
    ]


def test_gene_headers_show_the_current_status_without_an_arrow(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    html = _render(db_path, results)
    known = _gene_header(_article(html, "known-gene-1"))
    assert "rating-arrow" not in known
    order = [
        '<span class="gene-status">On panel</span>',
        '<span class="rating-badge rating-amber" data-title="The gene&#39;s highest rating',
        'class="finding-badge',
        'class="unreviewed-badge"',
    ]
    positions = [known.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "New MoI</span>" not in known

    novel = _gene_header(_article(html, "novel-gene-3"))
    assert "rating-arrow" not in novel
    assert '<span class="rating-badge rating-grey">Not in panel</span>' in novel
    assert re.findall(
        r'<a class="finding-badge rating-(\w+)" href="#association-(\d+)"[^>]*>([^<]+)</a>', novel
    ) == [("green", "20", "new disease GREEN")]
    assert novel.index("Not in panel") < novel.index('class="finding-badge')
    assert "On panel" not in novel


def test_green_gene_with_only_an_amber_finding(db_path: Path, hgnc_resolver: HgncResolver) -> None:
    """A GREEN gene's review keeps GREEN when its best association is AMBER, and the gene
    is placed by its AMBER finding."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM associations WHERE id = 12")
    results = _load(db_path, hgnc_resolver, genea_rating=3)
    genea = results.known_genes[0]
    assert genea.top_rating == 2
    assert (genea.prefill.form_type, genea.prefill.rating) == ("review", "GREEN")
    config = json.loads(build_report_config("r", [MENDELIOME_PANEL_ID], [], [genea]))
    assert config["genes"]["HGNC:1"]["suggested_rating"] == "GREEN"
    novel = results.novel_genes[0]
    assert (novel.prefill.form_type, novel.prefill.rating) == ("add", "GREEN")

    groups = group_genes(results.novel_genes, results.known_genes, {MENDELIOME_PANEL_ID})
    assert [(g.key, [gene.hgnc_id for gene in g.genes]) for g in groups.with_findings] == [
        ("finding-amber", [1])
    ]


def test_statistics_count_genes_and_findings(db_path: Path, results: GeneAssessmentResults) -> None:
    statistics = calculate_comprehensive_statistics(db_path, results, NO_PANEL_VALIDATION)
    assert (statistics.new_genes_count, statistics.known_genes_with_findings) == (1, 1)
    # GENEA's two new diseases and new MoI, and the new gene GENEB's new disease
    assert (
        statistics.new_disease_findings,
        statistics.new_moi_findings,
        statistics.upgrade_findings,
    ) == (3, 1, 0)
    assert "<strong>Findings:</strong> new diseases: 3, new MoIs: 1, upgrades: 0" in _render(
        db_path, results
    )


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


def _association_section(html: str, association_id: int) -> str:
    """The HTML of the association section with id ``association-{association_id}``."""
    start = html.index(f'<section class="association-block" id="association-{association_id}">')
    return html[start : html.index("</section>", start)]


def _heading(section: str) -> str:
    return section[: section.index("</header>")]


def test_association_variants_and_the_variants_in_no_association(
    results: GeneAssessmentResults,
) -> None:
    genea = results.known_genes[0]
    variants = {a.id: [v.variant_id for v in a.variants] for a in genea.associations}
    # In the gene's variant order, each variant once
    assert variants == {10: [COMMON_HET, COMMON_HOM], 11: [COMMON_HET, COMMON_HOM], 12: [], 13: []}
    assert [v.variant_id for v in genea.unassociated_variants] == [UNLISTED]
    assert results.novel_genes[0].unassociated_variants == []


def test_variant_flags_follow_the_association_moi(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    html = _render(db_path, results)

    def flagged(section: str) -> list[str]:
        return re.findall(r'<tr class="high-frequency-variant"><td>([^<]+)</td>', section)

    monoallelic, biallelic = _association_section(html, 10), _association_section(html, 11)
    assert "<summary>Variants (2)</summary>" in monoallelic
    assert "<summary>Variants (2)</summary>" in biallelic
    assert flagged(monoallelic) == [f"c.{COMMON_HET}"]  # het 40 > 30
    assert flagged(biallelic) == [f"c.{COMMON_HOM}"]  # hom 20 > 15
    assert "Variants (" not in _association_section(html, 12)

    known = _article(html, "known-gene-1")
    start = known.index('<details class="unassociated-variants">')
    leftover = known[start : known.index("</details>", start)]
    assert "Variants in no association (1)" in leftover
    assert f"c.{UNLISTED}" in leftover
    assert flagged(leftover) == []  # het 100, but no MoI to judge it by
    assert "Variants in no association" not in _article(html, "novel-gene-3")


def test_variants_come_from_the_papers_the_aggregation_read(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A variant reported only by a paper the preprint gate filtered is left out. A
    variant that two papers number on different transcripts shows both forms and cites
    both papers."""
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO papers (doi, pmid, title, source, source_type, source_details,
                                evidence_extraction_json)
            VALUES (?, ?, ?, 'pubmed', 'initial', 'f.xml', ?)
            """,
            [
                ("10.1/c", 666, "Paper C", _extraction(1)),
                ("10.1/pre", None, "Filtered preprint", _extraction(1)),
            ],
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (1, 'GENEA', ?, 'recent_evidence')",
            [("10.1/c",), ("10.1/pre",)],
        )
        conn.execute(
            "UPDATE gene_aggregations SET paper_id_mapping = ?, filtered_papers_json = ? "
            "WHERE hgnc_id = 1",
            (
                json.dumps({"Smith2024": "10.1/a", "Lee2025": "10.1/c"}),
                json.dumps(
                    [{"doi": "10.1/pre", "reason": "Preprint: 1 families (min 2 required)"}]
                ),
            ),
        )
        conn.executemany(
            """
            INSERT INTO variant_frequencies (variant_id, hgnc_id, paper_doi, quote,
                                             normalization, gnomad)
            VALUES (?, 1, ?, ?, ?, ?)
            """,
            [
                (
                    COMMON_HET,
                    "10.1/c",
                    "q2",
                    _normalization(COMMON_HET, "NM_9.1:c.5del", "NP_9.1:p.Lys2fs"),
                    GENEA_VARIANTS[0][2],
                ),
                (
                    "1-400-T-C",
                    "10.1/pre",
                    "q3",
                    _normalization("1-400-T-C", "c.9del", "p.Ser3fs"),
                    GENEA_VARIANTS[2][2],
                ),
            ],
        )
        conn.executemany(
            "INSERT INTO citation_locations (paper_doi, quote, bboxes_json) VALUES (?, ?, '[]')",
            [("10.1/a", "q"), ("10.1/c", "q2"), ("10.1/pre", "q3")],
        )
    results = _load(db_path, hgnc_resolver)
    genea = results.known_genes[0]
    shown = [v.variant_id for a in genea.associations for v in a.variants]
    assert "1-400-T-C" not in shown + [v.variant_id for v in genea.unassociated_variants]

    row = _association_section(_render(db_path, results), 10)
    row = row[row.index(f"<td>c.{COMMON_HET}") :]
    row = row[: row.index("</tr>")]
    assert (
        f"<td>c.{COMMON_HET}<br>NM_9.1:c.5del</td><td>p.{COMMON_HET}<br>NP_9.1:p.Lys2fs</td>"
        in (row)
    )
    assert re.findall(r"\[(\d+), not located\]", row) == ["111", "666"]


def test_variants_without_gnomad_figures_render_on_one_line(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A variant whose lookup failed keeps its citations but has no gnomAD link: its key
    is the paper's text, which gnomAD cannot open."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO citation_locations (paper_doi, quote, bboxes_json) "
            "VALUES ('10.1/a', 'q', '[]')"
        )
        conn.execute(
            "UPDATE variant_frequencies SET gnomad = ? WHERE variant_id = ?",
            (json.dumps({"variant_not_found": True}), UNLISTED),
        )
        conn.execute(
            """
            INSERT INTO variant_frequencies (variant_id, hgnc_id, paper_doi, quote,
                                             normalization, gnomad)
            VALUES ('E99fs', 1, '10.1/a', 'q', ?, ?)
            """,
            (
                json.dumps(
                    {
                        "original_text": "E99fs",
                        "error_code": "parse",
                        "error_message": "m",
                        "upstream": None,
                    }
                ),
                json.dumps({"normalization_error": True}),
            ),
        )
    known = _article(_render(db_path, _load(db_path, hgnc_resolver)), "known-gene-1")
    start = known.index('<details class="unassociated-variants">')
    rows = re.findall(r"<tr>.*?</tr>", known[start : known.index("</details>", start)], re.DOTALL)
    assert [row.count("<td>") for row in rows if "<td>" in row] == [8, 8]
    not_found, failed = [row for row in rows if "<td>" in row]
    assert "\n" not in not_found and "\n" not in failed
    assert f"<td>c.{UNLISTED}</td><td>p.{UNLISTED}</td><td><em>not found</em></td>" in not_found
    assert not_found.count("<td>—</td>") == 4
    assert failed.startswith("<tr><td><em>E99fs</em></td><td>—</td>")
    assert ">error</span></td><td>—</td><td>—</td><td>—</td><td>—</td>" in failed
    cite = (
        '<a href="viewer/index.html?paper=10.1%252Fa&amp;q=0" target="_blank">'
        "[111, not located]</a>"
    )
    gnomad = (
        f'<a href="https://gnomad.broadinstitute.org/variant/{UNLISTED}?dataset=gnomad_r4" '
        'target="_blank">gnomAD</a>'
    )
    assert not_found.endswith(f"<td>{gnomad} | {cite}</td></tr>")
    assert failed.endswith(f"<td>{cite}</td></tr>")
    assert "gnomad.broadinstitute.org" not in failed


def test_association_listing_an_unknown_variant_fails(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    assessment = json.loads(_association("ataxia variant", variants=("9-9-A-T",)))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE associations SET assessment_json = ? WHERE id = 12", (json.dumps(assessment),)
        )
    with pytest.raises(ValueError, match="9-9-A-T"):
        _load(db_path, hgnc_resolver)


def test_association_heading_rating_disease_moi_dispute_then_mondo(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    html = _render(db_path, results)
    # Association 10 is no finding: its rating, then PanelApp's as a note, without an arrow
    reused = _heading(_association_section(html, 10))
    assert "rating-arrow" not in reused
    order = [
        'class="rating-badge rating-red"',
        '<span class="relation-note" data-title="PanelApp Australia: b">'
        "curated in PanelApp: GREEN</span>",
        'class="association-disease">disease A<',
        'class="moi-pill">Monoallelic<',
        'class="dispute-badge dispute-disputed"',
        'class="mondo-term" href="https://monarchinitiative.org/MONDO:0000001"',
        ">obsolete in MONDO, replaced by MONDO:0000009<",
    ]
    positions = [reused.index(marker) for marker in order]
    assert positions == sorted(positions)

    assert 'class="moi-pill moi-pill-new">Biallelic<' in _heading(_association_section(html, 11))
    broader = _heading(_association_section(html, 12))
    assert (
        '<span class="mondo-term">ataxia (<a href="https://monarchinitiative.org/MONDO:0000200"'
        in (broader)
    )
    assert "most specific term that includes it. no exact term" in broader
    assert ">broader term</span>" in broader
    assert ">No MONDO term yet</span>" in _heading(_association_section(html, 11))
    for removed in ("MONDO exact", "Broader MONDO term", "mondo-badge"):
        assert removed not in html

    # A novel gene's new disease shows the corpus rating alone, first in the heading.
    novel = _heading(_association_section(html, 20))
    assert "rating-arrow" not in novel
    assert novel.index('class="rating-badge rating-green"') < novel.index("association-disease")


def test_association_body_hides_nr_lines(db_path: Path, results: GeneAssessmentResults) -> None:
    html = _render(db_path, results)
    ataxia = _association_section(html, 12)
    assert "<strong>Segregation:</strong> Segregates in both families." in ataxia
    reused = _association_section(html, 10)
    for hidden in ("Segregation:", "Functional:", "Mechanism:"):
        assert hidden not in reused
    assert reused.index("Papers:") < reused.index("Disputed in GenCC by:")
    assert "Cited evidence" not in html


def test_gene_body_association_cards_then_gene_blocks_in_order(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    known = _article(_render(db_path, results), "known-gene-1")
    order = [
        'class="gene-associations"',
        'id="association-11"',  # the last finding
        'class="other-associations"',  # the folded association without a finding
        'id="association-10"',
        'class="unassessed-reports"',
        'class="refused-papers"',
        'class="unassociated-variants"',
        "Contributing papers (1)",
    ]
    positions = [known.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "gnomAD v4 frequencies" not in known
    assert "PanelApp criteria assessment" not in known
    assert "evidence-summary-expanded gene-associations" not in known


def _set_genea_concerns(db_path: Path, concerns: dict[int, list[str]]) -> None:
    """The quality concerns of GENEA's associations by association id, each citing Paper A."""
    with sqlite3.connect(db_path) as conn:
        for association_id, texts in concerns.items():
            (assessment_json,) = conn.execute(
                "SELECT assessment_json FROM associations WHERE id = ?", (association_id,)
            ).fetchone()
            assessment = json.loads(assessment_json)
            assessment["quality_concerns"] = json.loads(_association("d", concerns=tuple(texts)))[
                "quality_concerns"
            ]
            conn.execute(
                "UPDATE associations SET assessment_json = ? WHERE id = ?",
                (json.dumps(assessment), association_id),
            )


def _concern_block(section: str) -> str:
    start = section.index('<div class="quality-concerns association-concerns">')
    return section[start : section.index("</div>", start)]


def test_quality_concerns_show_in_their_associations(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A concern the model states in the finding 13 and the folded curated 10 shows in
    both; the gene has no concerns box of its own."""
    _set_genea_concerns(
        db_path,
        {10: ["Shared cohort."], 13: ["Shared cohort.", "Smith2024 is one group."]},
    )
    results = _load(db_path, hgnc_resolver)
    genea = results.known_genes[0]
    assert {
        a.id: [c["concern"] for c in a.assessment["quality_concerns"]] for a in genea.associations
    } == {
        12: [],
        13: ["Shared cohort.", "PMID 111 is one group."],
        11: [],
        10: ["Shared cohort."],
    }

    html = _render(db_path, results)
    finding = _association_section(html, 13)
    block = _concern_block(finding)
    assert "<strong>⚠️ Quality concerns</strong>" in block
    assert "Shared cohort." in block and "PMID 111 is one group." in block
    assert re.search(r"\[111, not located\]</a>", block)
    papers, panels = finding.index("<strong>Papers:</strong>"), finding.index("Panels:</strong>")
    assert papers < finding.index("association-concerns") < panels
    assert "Shared cohort." in _concern_block(_association_section(html, 10))
    for association_id in (11, 12):
        assert "association-concerns" not in _association_section(html, association_id)

    article = _article(html, "known-gene-1")
    assert article.count("Shared cohort.") == 2
    assert "about the gene as a whole" not in article
    # The folded association's concern folds away with it
    assert article.index('class="other-associations"') < article.index('id="association-10"')


def test_each_association_shows_its_criteria_after_its_toggles(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    html = _render(db_path, results)
    for association_id in (10, 11, 12, 13, 20):
        section = _association_section(html, association_id)
        assert section.count('<details class="association-criteria">') == 1
        assert "<summary>PanelApp criteria</summary>" in section
        assert section.count('class="criterion-header') == 5
    reused = _association_section(html, 10)
    order = ['class="association-panels"', 'class="association-variants"', "PanelApp criteria"]
    positions = [reused.index(marker) for marker in order]
    assert positions == sorted(positions)


def _add_genea_associations(db_path: Path, associations: list[tuple[int, int, str]]) -> None:
    """Add (id, position, assessment_json) associations to GENEA, unmapped and unmatched."""
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO associations (id, hgnc_id, position, assessment_json) VALUES (?, 1, ?, ?)",
            associations,
        )


def _fold_summary(article: str) -> str:
    match = re.search(
        r'<details class="other-associations">\s*<summary>(.*?)</summary>', article, re.DOTALL
    )
    assert match is not None
    return _strip_tags(match.group(1))


def test_findings_with_the_same_badge_merge_and_come_first(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A second AMBER new disease joins association 13's badge, ahead of the AMBER new MoI
    11 that has as many families; a new MoI with one family is no finding."""
    _add_genea_associations(
        db_path,
        [
            (14, 4, _association("second disease", status="new_disease", independent=2)),
            (15, 5, _association("rare moi", status="new_moi", independent=1)),
        ],
    )
    results = _load(db_path, hgnc_resolver)
    genea = results.known_genes[0]
    assert [a.id for a in genea.associations] == [12, 13, 14, 11, 10, 15]
    assert [
        (g.kind, g.rating, g.from_rating, [a.id for a in g.associations])
        for g in genea.finding_groups
    ] == [
        (FindingKind.NEW_DISEASE, 3, None, [12]),
        (FindingKind.NEW_DISEASE, 2, None, [13, 14]),
        (FindingKind.NEW_MOI, 2, None, [11]),
    ]
    assert [a.id for a in genea.other_associations] == [10, 15]
    assert not genea.other_associations_curated

    html = _render(db_path, results)
    article = _article(html, "known-gene-1")
    header = _gene_header(article)
    assert re.findall(
        r'<a class="finding-badge rating-(\w+)" href="#association-(\d+)"[^>]*>([^<]+)</a>',
        header,
    ) == [
        ("green", "12", "new disease GREEN"),
        ("amber", "13", "2\N{MULTIPLICATION SIGN} new disease AMBER"),
        ("amber", "11", "new MoI AMBER"),
    ]
    assert (
        'data-title="Diseases PanelApp Australia does not curate for this gene:'
        '&#10;• GENEA-related other disease&#10;• GENEA-related second disease"'
    ) in header
    assert (
        'data-title="A disease PanelApp Australia does not curate for this gene:'
        '&#10;• GENEA-related ataxia variant"'
    ) in header

    section = _toc_section(html, "finding-green")
    assert re.findall(
        r'class="finding-chip rating-(\w+)" href="#association-(\d+)"[^>]*>(.*?)</a>', section
    ) == [("green", "12", "D"), ("amber", "13", "D<sup>2</sup>"), ("amber", "11", "M")]
    assert (
        'data-title="2\N{MULTIPLICATION SIGN} new disease AMBER:&#10;• GENEA-related other disease'
        '&#10;• GENEA-related second disease">D<sup>2</sup></a>'
    ) in section

    # The findings' blocks in badge order, then the fold with the others
    positions = [article.index(f'id="association-{i}"') for i in (12, 13, 14, 11, 10, 15)]
    assert positions == sorted(positions)
    assert article.index('class="other-associations"') > positions[3]
    assert _fold_summary(article) == "Other associations (2): disease A; GENEA-related rare moi"
    rare_moi = _heading(_association_section(html, 15))
    assert "rating-arrow" not in rare_moi
    assert ">new MoI, fewer than 2 independent families</span>" in rare_moi
    assert rare_moi.index('class="rating-badge rating-red"') < rare_moi.index("relation-note")
    statistics = calculate_comprehensive_statistics(db_path, results, NO_PANEL_VALIDATION)
    assert statistics.new_disease_findings == 4


def test_fold_of_curated_associations_names_their_diseases(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A curated association not rated higher than in PanelApp is folded; a disease
    named twice in the fold shows its MoI."""
    _add_genea_associations(
        db_path,
        [(16, 4, _association("disease A", independent=1, inheritance_mode="Biallelic"))],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE associations SET mondo_id = 'MONDO:0000001', mondo_label = 'disease A', "
            "mondo_match = 'panelapp_gencc' WHERE id = 16"
        )
    results = _load(db_path, hgnc_resolver)
    assert results.known_genes[0].other_associations_curated
    article = _article(_render(db_path, results), "known-gene-1")
    assert _fold_summary(article) == (
        "Already curated in PanelApp (2): disease A (Monoallelic); disease A (Biallelic)"
    )
    assert article.count('<details class="other-associations">') == 1
    assert (
        "<summary><strong>Already curated in PanelApp (2):</strong> disease A "
        '<span class="muted">(Monoallelic)</span>; disease A '
        '<span class="muted">(Biallelic)</span></summary>'
    ) in article


def test_gene_without_findings_shows_its_associations_unfolded(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM associations WHERE id IN (11, 12, 13)")
    results = _load(db_path, hgnc_resolver)
    assert results.known_genes[0].findings == []
    html = _render(db_path, results)
    article = _article(html, "known-gene-1")
    assert "other-associations" not in article
    assert 'id="association-10"' in article
    assert "finding-badge" not in _gene_header(article)
    assert "finding-chip" not in _toc_section(html, "unreviewed")
    heading = _heading(_association_section(html, 10))
    assert "rating-arrow" not in heading
    assert "curated in PanelApp: GREEN</span>" in heading


def test_only_upgrades_of_curated_associations_show_an_arrow(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """16 is RED in PanelApp and AMBER here; 17, with PanelApp's rating unknown, is RED
    here, below the gene's AMBER."""
    _add_genea_associations(
        db_path,
        [
            (16, 4, _association("upgraded", panelapp_rating="RED", independent=2)),
            (17, 5, _association("unknown", panelapp_rating=None, independent=1)),
        ],
    )
    results = _load(db_path, hgnc_resolver)
    genea = results.known_genes[0]
    assert [(g.kind, g.rating, g.from_rating) for g in genea.finding_groups][-1] == (
        FindingKind.UPGRADE,
        2,
        1,
    )
    html = _render(db_path, results)
    upgraded = _heading(_association_section(html, 16))
    order = ['class="rating-badge rating-red"', 'class="rating-arrow"', "rating-amber"]
    positions = [upgraded.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "relation-note" not in upgraded
    assert "upgrade RED→AMBER</a>" in _gene_header(_article(html, "known-gene-1"))

    unknown = _heading(_association_section(html, 17))
    assert "rating-arrow" not in unknown
    assert ">curated in PanelApp, own rating unknown</span>" in unknown
    assert "curated, own rating unknown" not in unknown
