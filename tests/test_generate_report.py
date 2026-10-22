"""Tests for loading and ordering a report's genes and associations."""

import dataclasses
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Literal

import pytest
from anthropic.types import Message

from palit.assess_genes import STAGE as ASSESS_GENES_STAGE
from palit.assess_relevance import STAGE as RELEVANCE_STAGE
from palit.assess_relevance import Refusal, refused_assessment
from palit.extract_evidence import STAGE as EXTRACTION_STAGE
from palit.gencc import MondoRef
from palit.generate_report import (
    MONTHLY_REPORT_TITLE,
    DerivedFamilies,
    FavoriteJournalSections,
    GeneAssessment,
    GeneAssessmentResults,
    ManualDownloadPaper,
    PanelValidationResult,
    PaperSources,
    PapersRead,
    QuoteRef,
    ReadPaper,
    ReportAssociation,
    StageState,
    UnreviewedEntry,
    association_finding,
    association_sort_key,
    build_gene_assessment_results,
    build_report_config,
    calculate_comprehensive_statistics,
    derived_families,
    format_inheritance,
    gene_findings,
    generate_html_report,
    group_genes,
    is_highlighted_new_moi,
    load_favorite_journal_papers,
    load_low_confidence_irrelevant_papers,
    load_manual_download_papers,
    load_panel_publications_validation,
    load_quote_refs,
    load_relevance_refusers,
    prefill_entry,
    report_order,
    unreviewed_target_entries,
    write_viewer_package,
)
from palit.hgnc import HgncResolver
from palit.llm import FALLBACK_MODEL, LIGHT_MODEL, MODEL
from palit.map_mondo import STAGE as MAP_MONDO_STAGE
from palit.match_panels import STAGE as MATCH_PANELS_STAGE
from palit.panel_entries import PanelEntry, ReviewState, entries_by_gene
from palit.panelapp_client import AllPanelsData, PanelGeneData, PanelPublications
from palit.panelapp_integration import (
    ENUM_TO_PANELAPP_MOI,
    MENDELIOME_PANEL_ID,
    PANELAPP_CRITERIA,
    Finding,
    FindingKind,
    PrefillData,
    PrefillEntry,
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


GENCC_TERM = {
    "mondo_id": "MONDO:0000001",
    "disease_title": "disease A",
    "moi_titles": ["Autosomal dominant"],
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


def _context(gencc_terms: list[dict[str, Any]], disputes: list[dict[str, Any]]) -> str:
    return json.dumps({"gencc_terms": gencc_terms, "disputes": disputes, "panel_entries": []})


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
        conn.execute(
            "INSERT INTO panelapp_citations (hgnc_id, paper_doi) VALUES (1, '10.1/seeded')"
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
                    _context([GENCC_TERM], DISPUTES),
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
                    "gencc",
                    None,
                    json.dumps(
                        [
                            {"panel_id": ATAXIA_PANEL_ID, "rationale": "Smith2024 reports ataxia."},
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


def _target_entry(
    panel_id: int, rating: int, review_state: ReviewState = ReviewState.UNREVIEWED
) -> PanelEntry:
    """An entry on target panel *panel_id* that cites nothing."""
    return PanelEntry(
        panel_id=panel_id,
        panel_name={MENDELIOME_PANEL_ID: "Mendeliome", ATAXIA_PANEL_ID: "Ataxia"}[panel_id],
        reference=True,
        entity_type="gene",
        rating={3: "GREEN", 2: "AMBER", 1: "RED"}[rating],
        moi="MONOALLELIC, autosomal or pseudoautosomal, NOT imprinted",
        mode_of_pathogenicity="",
        phenotypes=(),
        publications=(),
        review_publications=(),
        review_state=review_state,
    )


def _load(
    db_path: Path,
    hgnc_resolver: HgncResolver,
    genea_rating: int = 2,
    genea_reviews: ReviewState = ReviewState.UNREVIEWED,
) -> GeneAssessmentResults:
    """The fixture's genes, with GENEA on the Mendeliome at *genea_rating*, its reviews
    there in state *genea_reviews*."""
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
    entries: dict[int, tuple[PanelEntry, ...]] = {
        1: (_target_entry(MENDELIOME_PANEL_ID, genea_rating, genea_reviews),)
    }
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return build_gene_assessment_results(conn, hgnc_resolver, target, all_panels, None, entries)


@pytest.fixture
def results(db_path: Path, hgnc_resolver: HgncResolver) -> GeneAssessmentResults:
    return _load(db_path, hgnc_resolver)


def _load_scoped(
    db_path: Path, hgnc_resolver: HgncResolver, genea_rating: int
) -> GeneAssessmentResults:
    """The fixture's genes in a run scoped to the Ataxia panel, with GENEA on it at
    *genea_rating*, reviewed there. Each association gets the scope_rationale a
    panel-scoped aggregation gives it."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE associations SET assessment_json = json_set(assessment_json, "
            "'$.scope_rationale', 'Smith2024: ataxia in 3 of 4 (association ' || id || ')')"
        )
    target = PanelGeneData(
        panel_ids=[ATAXIA_PANEL_ID], gene_panel_confidence={1: {ATAXIA_PANEL_ID: genea_rating}}
    )
    all_panels = AllPanelsData(
        gene_to_panels={1: {MENDELIOME_PANEL_ID, ATAXIA_PANEL_ID}},
        panel_names={MENDELIOME_PANEL_ID: "Mendeliome", ATAXIA_PANEL_ID: "Ataxia"},
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return build_gene_assessment_results(
            conn,
            hgnc_resolver,
            target,
            all_panels,
            ATAXIA_PANEL_ID,
            {1: (_target_entry(ATAXIA_PANEL_ID, genea_rating, ReviewState.REVIEWED),)},
        )


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
        "gencc",
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
    # A rationale names papers as the summary does.
    assert reused_matches[1].rationale == "PMID 111 reports ataxia."
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
        "New disease: GENEA-related ataxia variant (Monoallelic, Green)",
        "New disease: GENEA-related other disease (Monoallelic, Amber)",
        "New MoI: GENEA-related biallelic disease (Biallelic, Amber)",
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
        "New disease: GENEA-related new disease (Monoallelic, Amber)\nPMID 111 reports new disease."
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
    assert genea.prefill.comments == (
        "Disease A, replacement (Monoallelic, Red)\nPMID 111 reports disease A."
    )


def test_add_prefill_lists_every_association(db_path: Path, hgnc_resolver: HgncResolver) -> None:
    """GENEB's association that PanelApp curates elsewhere is a finding of the new gene
    like its new disease, and the add form lists both."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO associations (id, hgnc_id, position, assessment_json) VALUES (21, 3, 1, ?)",
            (_association("curated B", panelapp_rating=None, dois=("10.1/b",)),),
        )
    geneb = _load(db_path, hgnc_resolver).novel_genes[0]
    assert [f.association_id for f in geneb.findings] == [20, 21]
    assert (geneb.prefill.form_type, geneb.prefill.panel_id) == ("add", MENDELIOME_PANEL_ID)
    assert geneb.prefill.phenotypes == "GENEA-related B disease;GENEA-related curated B"
    assert geneb.prefill.publications == "222"
    # Both are new diseases of a new gene, so their headings don't say so.
    assert geneb.prefill.comments == (
        "GENEA-related B disease (Monoallelic, Green)\nSmith2024 reports B disease.\n\n"
        "GENEA-related curated B (Monoallelic, Amber)\nSmith2024 reports curated B."
    )


def test_add_prefill_of_one_association_comments_its_summary(
    results: GeneAssessmentResults,
) -> None:
    geneb = results.novel_genes[0]
    assert geneb.prefill.form_type == "add"
    assert geneb.prefill.comments == "Smith2024 reports B disease."


def _panel_entry(
    panel_id: int, entity_type: Literal["gene", "str"], moi: str, publications: tuple[str, ...]
) -> PanelEntry:
    return dataclasses.replace(
        _target_entry(panel_id, 2),
        entity_type=entity_type,
        moi=moi,
        phenotypes=(f"{entity_type} phenotype",),
        publications=publications,
    )


def test_a_review_extends_the_gene_entry_on_its_panel() -> None:
    """The panel's gene entry, not its repeat expansion nor the gene's entry on another
    panel; the entry's publications are read as PMIDs and DOIs in their order."""
    repeat = _panel_entry(MENDELIOME_PANEL_ID, "str", "Unknown", ("PMID: 1",))
    gene = _panel_entry(
        MENDELIOME_PANEL_ID,
        "gene",
        ENUM_TO_PANELAPP_MOI["Biallelic"],
        ("PMID: 33333, 22222", "doi: 10.1000/X", "OMIM 601492", "11111"),
    )
    other = _panel_entry(ATAXIA_PANEL_ID, "gene", "Other", ())
    assert prefill_entry((other, repeat, gene), MENDELIOME_PANEL_ID, 3) == PrefillEntry(
        rating=3,
        moi=ENUM_TO_PANELAPP_MOI["Biallelic"],
        phenotypes=("gene phenotype",),
        publications=(22222, 33333, "10.1000/X", 11111),
    )


def test_a_review_of_a_gene_held_only_as_a_repeat_expansion_extends_that_entry() -> None:
    repeat = _panel_entry(MENDELIOME_PANEL_ID, "str", "Unknown", ("PMID: 11111",))
    entry = prefill_entry((repeat,), MENDELIOME_PANEL_ID, 2)
    assert (entry.moi, entry.publications) == ("Unknown", (11111,))


def test_a_review_without_an_entry_on_its_panel_raises() -> None:
    other = _panel_entry(ATAXIA_PANEL_ID, "gene", "Other", ())
    with pytest.raises(ValueError, match="no entry on panel 137"):
        prefill_entry((other,), MENDELIOME_PANEL_ID, 2)


NO_PANEL_VALIDATION = PanelValidationResult(0, 0, [], [], 0, [], 0.0)
NO_FEATURED_PAPERS = FavoriteJournalSections([], [], [], [], [], [], [], [], [])


def _render(
    db_path: Path,
    results: GeneAssessmentResults,
    panel_validation: PanelValidationResult = NO_PANEL_VALIDATION,
    favorite_journal_papers: FavoriteJournalSections = NO_FEATURED_PAPERS,
    manual_download_papers: list[ManualDownloadPaper] | None = None,
) -> str:
    return generate_html_report(
        novel_genes=results.novel_genes,
        known_genes=results.known_genes,
        statistics=calculate_comprehensive_statistics(db_path, results, panel_validation),
        panel_validation=panel_validation,
        low_confidence_papers=[],
        manual_download_papers=manual_download_papers or [],
        favorite_journal_papers=favorite_journal_papers,
        template_dir=ROOT / "templates",
        panel_date="2026-10-01",
        report_id="test",
        target_panel_ids=list(results.target_panel_data.panel_ids),
        target_panel_names=results.target_panel_names,
        refused_genes=results.refused_genes,
        relevance_refusers=load_relevance_refusers(db_path),
        panels_matched=results.panels_matched,
        panelapp_integration=False,
        unreviewed_without_evidence=results.unreviewed_without_evidence,
        scope_panel_id=results.scope_panel_id,
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
        "Ataxia (gene already on panel): PMID 111 reports ataxia.",
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
        reviewed_on=(),
        scope_rationale=None,
    )


@pytest.mark.parametrize(
    ("association", "current_rating", "scoped", "finding"),
    [
        # Every association of a gene on no target panel, whatever PanelApp curates elsewhere
        (
            _association("d", status="new_disease", green=True),
            None,
            False,
            (FindingKind.NEW_DISEASE, 3),
        ),
        (
            _association("d", status="new_moi", independent=1),
            None,
            False,
            (FindingKind.NEW_DISEASE, 1),
        ),
        (
            _association("d", panelapp_rating="GREEN", green=True),
            None,
            True,
            (FindingKind.NEW_DISEASE, 3),
        ),
        (
            _association("d", panelapp_rating="AMBER", independent=1),
            None,
            True,
            (FindingKind.NEW_DISEASE, 1),
        ),
        # A new disease of a gene on a target panel, at any corpus rating
        (
            _association("d", status="new_disease", independent=1),
            3,
            False,
            (FindingKind.NEW_DISEASE, 1),
        ),
        (
            _association("d", status="new_disease", independent=1),
            3,
            True,
            (FindingKind.NEW_DISEASE, 1),
        ),
        # A new MoI only once highlighted
        (_association("d", status="new_moi", independent=2), 3, False, (FindingKind.NEW_MOI, 2)),
        (_association("d", status="new_moi", independent=1), 1, False, None),
        # An upgrade over PanelApp's rating of the association, whatever the gene's rating
        (
            _association("d", panelapp_rating="AMBER", green=True),
            2,
            False,
            (FindingKind.UPGRADE, 3, 2),
        ),
        (
            _association("d", panelapp_rating="RED", independent=2),
            3,
            False,
            (FindingKind.UPGRADE, 2, 1),
        ),
        (_association("d", panelapp_rating="GREEN", green=True), 1, False, None),
        (_association("d", panelapp_rating="AMBER", independent=2), 1, False, None),
        # Without PanelApp's rating of the association, over the gene's current rating
        (
            _association("d", panelapp_rating=None, independent=2),
            1,
            False,
            (FindingKind.UPGRADE, 2, 1),
        ),
        (_association("d", panelapp_rating=None, green=True), 3, False, None),
        # In a panel-scoped run, over the gene's rating on the scope panel, whatever
        # PanelApp rates the association elsewhere
        (
            _association("d", panelapp_rating="GREEN", green=True),
            2,
            True,
            (FindingKind.UPGRADE, 3, 2),
        ),
        (_association("d", panelapp_rating="RED", independent=2), 3, True, None),
    ],
)
def test_association_finding(
    association: str,
    current_rating: int | None,
    scoped: bool,
    finding: tuple[FindingKind, int] | tuple[FindingKind, int, int] | None,
) -> None:
    expected = None
    if finding is not None:
        kind, rating, *from_rating = finding
        expected = Finding(kind, rating, 7, from_rating[0] if from_rating else None)
    actual = association_finding(_report_association(7, association), current_rating, scoped)
    assert actual == expected


@pytest.mark.parametrize(
    "association",
    [
        _association("d", panelapp_rating="AMBER", green=True),
        _association("d", status="new_disease", green=True),
        _association("d", status="new_moi", independent=2),
    ],
)
@pytest.mark.parametrize("scoped", [False, True])
def test_an_association_whose_papers_the_target_panels_reviewed_is_no_finding(
    association: str, scoped: bool
) -> None:
    reviewed = dataclasses.replace(_report_association(7, association), reviewed_on=("Mendeliome",))
    assert association_finding(reviewed, 2, scoped) is None


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
    findings = gene_findings(rated, current_rating, scoped=False)
    review_state = ReviewState.REVIEWED if reviewed else ReviewState.UNREVIEWED
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
        target_entries=(
            ()
            if current_rating is None
            else (_target_entry(MENDELIOME_PANEL_ID, current_rating, review_state),)
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
    groups = group_genes(novel, known)

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
        ("unreviewed", "Unreviewed on target panel", [30], False, False)
    ]
    assert layout(groups.already_reviewed) == [("reviewed", "Already reviewed", [31], True, False)]
    assert [(f.kind, f.rating) for f in genes["green_gene"].findings] == [
        (FindingKind.NEW_DISEASE, 3),
        (FindingKind.NEW_DISEASE, 2),
    ]
    # Each association of a new gene is a finding, and it is placed by its top association
    assert {
        name: [(f.kind, f.rating) for f in genes[name].findings]
        for name in ("new_amber", "new_amber_moi", "new_red")
    } == {
        "new_amber": [(FindingKind.NEW_DISEASE, 2)],
        "new_amber_moi": [(FindingKind.NEW_DISEASE, 2), (FindingKind.NEW_DISEASE, 2)],
        "new_red": [(FindingKind.NEW_DISEASE, 1)],
    }


def _screen(relevant: bool, confidence: str) -> dict[str, Any]:
    return {
        "relevant": relevant,
        "confidence": confidence,
        "rationale": f"Screen rationale {confidence}.",
        "associations": [{"gene_symbol": "GENEA", "disease": "d"}] if relevant else [],
    }


def test_papers_both_models_refused_are_neither_passes_nor_misses(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    """Panel publications: a pass, a screen miss, and two papers refused for good."""
    assessments = {
        "10.1/pass": {"relevant": True, "screen": _screen(True, "HIGH")},
        "10.1/miss": {"relevant": False, "screen": _screen(False, "LOW")},
        "10.1/refused-bio": refused_assessment(Refusal(LIGHT_MODEL, "bio")),
        "10.1/refused-none": refused_assessment(Refusal(LIGHT_MODEL, None)),
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
    # The relevance stage ran on LIGHT_MODEL, whose refusals are final.
    for doi in ("10.1/refused-bio", "10.1/refused-none"):
        _add_requests(
            db_path,
            RELEVANCE_STAGE,
            doi,
            [("refused", "2026-10-01T00:00:00+00:00", LIGHT_MODEL)],
        )
    validation = load_panel_publications_validation(
        db_path, PanelPublications(pmids={900, 901, 902, 903}, dois=set())
    )
    assert [p.doi for p in validation.true_positives] == ["10.1/pass"]
    assert [p.doi for p in validation.screen_misses] == ["10.1/miss"]
    assert sorted(p.doi for p in validation.refused) == ["10.1/refused-bio", "10.1/refused-none"]
    assert validation.panel_papers_in_db == 4
    assert validation.sensitivity_pct == 50.0  # of the two assessed papers
    low_confidence = load_low_confidence_irrelevant_papers(db_path)
    assert [p.doi for p in low_confidence] == ["10.1/miss"]

    html = _render(db_path, results, validation)
    assert "Refused (2)</strong>" in html
    assert "Refused to assess (left out of the sensitivity): 2" in html
    assert "1 assessed relevant, 1 missed by the scope screen." in html
    assert (
        f"Haiku 5.5 refused the scope screen of this paper ({LIGHT_MODEL}, category: bio)"
    ) in html
    assert "category: none given" in html
    # A paper's relevance note is the screen's decision with its rationale.
    assert "<strong>Relevant:</strong> Yes (HIGH confidence). Screen rationale HIGH." in html


def test_panel_publications_match_a_doi_in_any_letter_case(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO papers (doi, title, source, source_type, relevance_assessment_json)
            VALUES ('10.1000/Mixed.Case', 'Title', 'pubmed', 'initial', ?)
            """,
            (json.dumps({"relevant": True, "screen": _screen(True, "HIGH")}),),
        )
    validation = load_panel_publications_validation(
        db_path, PanelPublications(pmids=set(), dois={"10.1000/mixed.CASE"})
    )
    assert [p.doi for p in validation.true_positives] == ["10.1000/Mixed.Case"]


def test_relevance_refusers_name_the_stage_model_and_its_fallback(db_path: Path) -> None:
    _add_requests(
        db_path,
        RELEVANCE_STAGE,
        "10.1/a",
        [("refused", "2026-10-01T00:00:00+00:00", FALLBACK_MODEL)],
    )
    # A run on FALLBACK_MODEL: its refusals are final.
    assert load_relevance_refusers(db_path) == "Sonnet 5.5"
    _add_requests(
        db_path,
        RELEVANCE_STAGE,
        "10.1/b",
        [("refused", "2026-10-01T00:00:00+00:00", MODEL)],
    )
    assert load_relevance_refusers(db_path) == "Opus 5.5 and its fallback Sonnet 5.5 both"


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
    # The novel gene's header already says "Not in panel", so its association's heading
    # shows the rating, then its relation to PanelApp's curation as a quiet note; the
    # finding is a header badge only.
    novel = _article(html, "novel-gene-3")
    assert ">new disease</span>" not in novel
    assert 'class="rating-arrow"' not in novel
    assert ">disease not curated in PanelApp</span>" in novel
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


def test_only_known_gene_sections_say_a_lower_rating_is_no_downgrade(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    html = _render(db_path, results)

    def note(section_id: str) -> str:
        section = html[html.index(f'<section id="{section_id}">') :]
        note = section[section.index('class="section-intro"') :]
        return note[: note.index("</p>")]

    assert "from everything published." in note("new-genes")
    assert "downgrade" not in note("new-genes")
    assert "not a downgrade recommendation" in note("known-genes-with-findings")


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

    groups = group_genes(results.novel_genes, results.known_genes)
    assert [(g.key, [gene.hgnc_id for gene in g.genes]) for g in groups.with_findings] == [
        ("finding-amber", [1])
    ]


def test_statistics_count_genes_and_findings(db_path: Path, results: GeneAssessmentResults) -> None:
    statistics = calculate_comprehensive_statistics(db_path, results, NO_PANEL_VALIDATION)
    assert (statistics.new_genes_count, statistics.known_genes_with_findings) == (1, 1)
    # GENEA's two new diseases and new MoI; the new gene GENEB counts as a new gene
    assert (
        statistics.new_disease_findings,
        statistics.new_moi_findings,
        statistics.upgrade_findings,
    ) == (2, 1, 0)
    assert (
        "<strong>Findings on known genes:</strong> new diseases: 2, new MoIs: 1, upgrades: 0"
        in _render(db_path, results)
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


def _paper_entity(
    description: str, independent: int, summary: str, concerns: tuple[str, ...]
) -> dict[str, Any]:
    """A disease entity of a paper's extraction, in its stored form."""
    return {
        "description": description,
        "variants": [],
        "patient_count": independent,
        "reported_family_count": independent,
        "family_count": independent,
        "independent_family_count": independent,
        "count_reduction_reasoning": f"No reduction ({independent} families)",
        "inheritance_mode": "Monoallelic",
        "inheritance_details": "",
        "evidence_assessments": [
            {"name": name, "result": False, "rationale": "r", "confidence": "LOW", "citations": []}
            for name in PANELAPP_CRITERIA
        ],
        "summary": summary,
        "quality_concerns": [{"concern": concern, "citations": []} for concern in concerns],
    }


def test_paper_evidence_shows_each_entitys_summary_and_concerns(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """A paper's evidence gives each disease entity its own summary and quality concerns."""
    entities = [
        _paper_entity("ataxia", 1, "Reports one individual with ataxia.", ()),
        _paper_entity(
            "neuropathy", 2, "Reports two individuals with neuropathy.", ("No parental samples.",)
        ),
    ]
    extraction = {"gene_evaluations": [{"hgnc_id": 1, "disease_entities": entities}]}
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE papers SET evidence_extraction_json = ? WHERE doi = '10.1/a'",
            (json.dumps(extraction),),
        )

    known = _article(_render(db_path, _load(db_path, hgnc_resolver)), "known-gene-1")
    evidence = known[known.index('<details class="individual-evidence-extraction">') :]
    sections = [
        section[: section.index("</section>")]
        for section in evidence.split('<section class="per-entity-criteria-section">')[1:]
    ]
    # By independent family count, as in the entity table above them.
    neuropathy, ataxia = sections
    assert "neuropathy</h6>" in neuropathy
    assert "<p>Reports two individuals with neuropathy.</p>" in neuropathy
    assert "No parental samples." in neuropathy
    assert "<p>Reports one individual with ataxia.</p>" in ataxia
    assert "quality-concerns" not in ataxia


def test_report_rejects_a_gene_level_extraction(db_path: Path, hgnc_resolver: HgncResolver) -> None:
    gene_level = {
        "gene_evaluations": [
            {"hgnc_id": 1, "summary": "s", "disease_entities": [], "quality_concerns": []}
        ]
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE papers SET evidence_extraction_json = ? WHERE doi = '10.1/a'",
            (json.dumps(gene_level),),
        )
    with pytest.raises(ValueError, match=r"10\.1/a: gene-level extraction shape; re-extract"):
        _load(db_path, hgnc_resolver)


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
        'data-title="Diseases the target panels do not curate for this gene:'
        '&#10;• GENEA-related other disease&#10;• GENEA-related second disease"'
    ) in header
    assert (
        'data-title="A disease the target panels do not curate for this gene:'
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
    assert statistics.new_disease_findings == 3


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
            "mondo_match = 'gencc' WHERE id = 16"
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


def test_a_scoped_run_measures_upgrades_from_the_scope_panel_rating(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """16 is AMBER here and GREEN in PanelApp. GENEA is RED on the scope panel, so the
    scoped run reports 16 as an upgrade from RED; the unscoped run, which starts from
    PanelApp's GREEN, reports nothing. A panel-scoped run has no match-panels."""
    _add_genea_associations(
        db_path, [(16, 4, _association("upgraded", panelapp_rating="GREEN", independent=2))]
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM llm_requests WHERE stage = ?", (MATCH_PANELS_STAGE,))
        conn.execute("UPDATE associations SET matched_panels_json = NULL")

    unscoped = _load(db_path, hgnc_resolver, genea_rating=1)
    assert 16 not in unscoped.known_genes[0].finding_association_ids
    unscoped_statistics = calculate_comprehensive_statistics(db_path, unscoped, NO_PANEL_VALIDATION)
    assert unscoped_statistics.upgrade_findings == 0

    results = _load_scoped(db_path, hgnc_resolver, genea_rating=1)
    genea = results.known_genes[0]
    assert genea.current_rating == 1
    assert Finding(FindingKind.UPGRADE, 2, 16, 1) in genea.findings
    statistics = calculate_comprehensive_statistics(db_path, results, NO_PANEL_VALIDATION)
    assert statistics.upgrade_findings == 1

    geneb = results.novel_genes[0]
    assert geneb.findings == [Finding(FindingKind.NEW_DISEASE, 3, 20, None)]
    assert (geneb.prefill.form_type, geneb.prefill.panel_id, geneb.prefill.rating) == (
        "add",
        ATAXIA_PANEL_ID,
        "GREEN",
    )
    groups = group_genes(results.novel_genes, results.known_genes)
    assert [(g.key, [gene.hgnc_id for gene in g.genes]) for g in groups.new_genes] == [
        ("new-green", [3])
    ]

    assert not results.panels_matched
    html = _render(db_path, results)
    assert 'id="outside-scope"' not in html
    for text in ('class="association-panels"', "Panels:</strong>", "Panel Suggestions:"):
        assert text not in html

    # Each association says why it fits the panel, with paper IDs in display form.
    assert {a.id: a.scope_rationale for a in genea.associations}[16] == (
        "PMID 111: ataxia in 3 of 4 (association 16)"
    )
    assert (
        '<p class="scope-rationale"><strong>Panel fit:</strong> PMID 111: ataxia in 3 of 4 '
        "(association 16)</p>"
    ) in _association_section(html, 16)
    assert "Panel fit" not in _render(db_path, unscoped)
    assert all(a.scope_rationale is None for a in unscoped.known_genes[0].associations)


def test_a_scoped_report_stops_at_an_association_aggregated_without_the_scope_panel(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    target = PanelGeneData(panel_ids=[ATAXIA_PANEL_ID], gene_panel_confidence={})
    all_panels = AllPanelsData(gene_to_panels={}, panel_names={ATAXIA_PANEL_ID: "Ataxia"})
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        with pytest.raises(ValueError, match=r"HGNC:1 association 10 has no scope_rationale"):
            build_gene_assessment_results(
                conn, hgnc_resolver, target, all_panels, ATAXIA_PANEL_ID, {}
            )


def test_a_report_without_a_scope_panel_stops_at_a_panel_scoped_association(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE associations SET assessment_json = "
            "json_set(assessment_json, '$.scope_rationale', 'r') WHERE id = 12"
        )
    with pytest.raises(
        ValueError, match=r"HGNC:1 association 12 was aggregated with a scope panel"
    ):
        _load(db_path, hgnc_resolver)


def test_a_gene_with_any_unreviewed_target_entry_is_unreviewed() -> None:
    """Reviews the snapshot does not hold may exist, so only an entry without reviews
    makes the gene unreviewed, on whichever target panel it is."""
    gene = _gene(30, 3, [CURATED_GREEN], reviewed=False)
    assert gene.unreviewed_target_panels == [MENDELIOME_PANEL_ID]
    mixed = dataclasses.replace(
        gene,
        target_entries=(
            _target_entry(ATAXIA_PANEL_ID, 3, ReviewState.REVIEWED),
            _target_entry(MENDELIOME_PANEL_ID, 3, ReviewState.UNREVIEWED),
        ),
    )
    assert mixed.unreviewed_target_panels == [MENDELIOME_PANEL_ID]
    unknown = dataclasses.replace(
        gene,
        hgnc_id=31,
        target_entries=(_target_entry(MENDELIOME_PANEL_ID, 3, ReviewState.UNKNOWN),),
    )
    assert unknown.unreviewed_target_panels == []
    groups = group_genes([], [mixed, unknown])
    assert [[g.hgnc_id for g in group.genes] for group in groups.without_findings] == [[30]]
    assert [[g.hgnc_id for g in group.genes] for group in groups.already_reviewed] == [[31]]


def test_a_link_into_a_fold_opens_it(db_path: Path, results: GeneAssessmentResults) -> None:
    html = _render(db_path, results)
    assert "window.addEventListener('hashchange'" in html
    assert "fold.open = true" in html


def _add_read_papers(db_path: Path) -> None:
    """Papers with extractions read for GENEC (HGNC:4), screened in or cited, without a
    disease of it, and for GENEE (HGNC:6), with one: a screened initial paper, and an
    expansion paper linked by its extraction alone; and a GENEC paper not extracted."""
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO papers (doi, pmid, title, source, source_type, source_date,
                                evidence_extraction_json)
            VALUES (?, ?, ?, 'pubmed', 'initial', ?, ?)
            """,
            [
                ("10.1/c-screened", 701, "Screened C paper", "2020-01-01", _extraction(4)),
                ("10.1/c-cited", None, "Cited C paper", "2024-01-01", _extraction(4)),
                ("10.1/c-unread", 703, "Unread C paper", "2025-01-01", None),
                ("10.1/e", 705, "E paper", "2023-01-01", _extraction(6)),
                ("10.1/e-cited", None, "Cited E paper", "2022-01-01", _extraction(6)),
            ],
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (?, ?, ?, ?)",
            [
                (4, "GENEC", "10.1/c-screened", "relevance_assessment"),
                (4, "GENEC", "10.1/c-unread", "relevance_assessment"),
                (6, "GENEE", "10.1/e", "relevance_assessment"),
                (6, "GENEE", "10.1/e", "recent_evidence"),
                (6, "GENEE", "10.1/e-cited", "expansion_evidence"),
            ],
        )
        conn.execute(
            "INSERT INTO panelapp_citations (hgnc_id, paper_doi) VALUES (4, '10.1/c-cited')"
        )


def test_unreviewed_target_entries_of_genes_without_associations_are_listed(
    db_path: Path, hgnc_resolver: HgncResolver, panelapp_snapshot: dict[int, dict[str, Any]]
) -> None:
    """On the Mendeliome, GENEA (in the report) and GENEC to GENEE (not in it) are
    unreviewed; GENEB is unreviewed on the Ataxia panel, which is no target panel. GENEC to
    GENEE are listed with the papers read for them: GENEC's without a disease of it, none for
    GENED, and GENEE's with a disease of it that gave no association."""
    mendeliome = panelapp_snapshot[MENDELIOME_PANEL_ID]
    genea = mendeliome["genes"][0]
    genea["evaluations"] = []
    for hgnc_id, symbol in ((4, "GENEC"), (5, "GENED"), (6, "GENEE")):
        gene = {**genea, "gene_data": {"hgnc_id": f"HGNC:{hgnc_id}", "gene_symbol": symbol}}
        gene["confidence_level"] = "2"
        mendeliome["genes"].append(gene)
    panelapp_snapshot[ATAXIA_PANEL_ID]["genes"][0]["evaluations"] = []
    entries = entries_by_gene([panelapp_snapshot[MENDELIOME_PANEL_ID]], [MENDELIOME_PANEL_ID])
    assert sorted(unreviewed_target_entries(entries)) == [1, 4, 5, 6]
    _add_read_papers(db_path)

    target = PanelGeneData(
        panel_ids=[MENDELIOME_PANEL_ID], gene_panel_confidence={1: {MENDELIOME_PANEL_ID: 2}}
    )
    all_panels = AllPanelsData(
        gene_to_panels={1: {MENDELIOME_PANEL_ID}}, panel_names={MENDELIOME_PANEL_ID: "Mendeliome"}
    )
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        results = build_gene_assessment_results(
            conn, hgnc_resolver, target, all_panels, None, entries
        )
    genec_papers = (
        ReadPaper("10.1/c-cited", None, "Cited C paper", disease_extracted=False),
        ReadPaper("10.1/c-screened", 701, "Screened C paper", disease_extracted=False),
    )
    genee_papers = (
        ReadPaper("10.1/e", 705, "E paper", disease_extracted=True),
        ReadPaper("10.1/e-cited", None, "Cited E paper", disease_extracted=True),
    )
    entry = ("Mendeliome", "AMBER")
    assert results.unreviewed_without_evidence == [
        UnreviewedEntry(4, "GENEC", MENDELIOME_PANEL_ID, *entry, genec_papers),
        UnreviewedEntry(5, "GENED", MENDELIOME_PANEL_ID, *entry, ()),
        UnreviewedEntry(6, "GENEE", MENDELIOME_PANEL_ID, *entry, genee_papers),
    ]
    assert [e.papers_read for e in results.unreviewed_without_evidence] == [
        PapersRead.NO_DISEASE,
        PapersRead.NONE,
        PapersRead.NO_ASSOCIATION,
    ]

    def groups(html: str) -> dict[str, str]:
        """Each group's heading and the rows below it, by its id."""
        section = html[html.index('<section id="unreviewed-without-evidence">') :]
        section = section[: section.index("</section>")]
        parts = re.split(r'<h3 id="unreviewed-([a-z-]+)">', section)[1:]
        return dict(zip(parts[::2], parts[1::2], strict=True))

    monthly = groups(_render(db_path, results))
    assert list(monthly) == ["none", "no-disease", "no-association"]
    assert monthly["none"].startswith("No new papers (1)</h3>")
    assert "/gene/GENED/" in monthly["none"]
    assert "Papers read" not in monthly["none"]
    assert monthly["no-disease"].startswith("Papers read, no disease entity (1)</h3>")
    assert (
        '<li><a href="https://doi.org/10.1/c-cited" target="_blank">10.1/c-cited</a>: '
        "Cited C paper</li>"
    ) in monthly["no-disease"]
    assert "701</a>: Screened C paper</li>" in monthly["no-disease"]
    assert "Unread C paper" not in monthly["no-disease"]
    assert monthly["no-association"].startswith("Papers read, no association (1)</h3>")
    assert (
        '705</a>: E paper <span class="muted">(disease extracted)</span></li>'
        in monthly["no-association"]
    )
    scoped = groups(
        _render(db_path, dataclasses.replace(results, scope_panel_id=MENDELIOME_PANEL_ID))
    )
    assert scoped["no-disease"].startswith("Papers read, none fit the panel (1)</h3>")
    assert (
        'href="#unreviewed-without-evidence">Unreviewed panel genes without associations (3)'
        in (_toc(_render(db_path, results)))
    )


def test_already_reviewed_genes_come_last_in_a_folded_section(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """GENEA, reviewed on the Mendeliome and without findings once its new associations
    are gone, comes after every other section, folded."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM associations WHERE id IN (11, 12, 13)")
    results = _load(db_path, hgnc_resolver, genea_reviews=ReviewState.REVIEWED)
    html = _render(db_path, results)
    section = html[html.index('<section id="already-reviewed">') :]
    assert section.index('<details class="folded-section">') < section.index('id="known-gene-1"')
    assert "<summary>Show 1 gene</summary>" in section
    body = html[html.index("</nav>") :]
    assert body.index('<section id="new-genes">') < body.index('<section id="already-reviewed">')
    assert 'data-section-key="reviewed"' in _toc(html)


# Papers of GENEA that the Mendeliome cites: P1 by PMID in its entry, P2 by DOI, in
# another letter case, in a review. It does not cite P3.
P1, P2, P3 = "10.1/p1", "10.1000/p2", "10.1/p3"
REVIEWED = ({"rating": "AMBER", "publications": ("10.1000/P2",)},)


def _load_cited(
    db_path: Path,
    hgnc_resolver: HgncResolver,
    panelapp_snapshot: dict[int, dict[str, Any]],
    associations: list[tuple[int, str]],
    *,
    evaluations: tuple[dict[str, Any], ...] | None = REVIEWED,
) -> GeneAssessmentResults:
    """The fixture's genes with GENEA's associations replaced by (id, assessment_json)
    *associations*, GENEA AMBER on the Mendeliome, which cites P1 and P2. GENEA's reviews
    there are *evaluations*, by default one that cites P2; None when PanelApp could not
    serve them."""
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO papers (doi, pmid, title, source, source_type, evidence_extraction_json) "
            "VALUES (?, ?, 't', 'pubmed', 'initial', ?)",
            [(doi, pmid, _extraction(1)) for doi, pmid in ((P1, 11111), (P2, None), (P3, 33333))],
        )
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (1, 'GENEA', ?, 'recent_evidence')",
            [(P1,), (P2,), (P3,)],
        )
        conn.execute("DELETE FROM associations WHERE hgnc_id = 1")
        conn.executemany(
            "INSERT INTO associations (id, hgnc_id, position, assessment_json) VALUES (?, 1, ?, ?)",
            [(id_, position, a) for position, (id_, a) in enumerate(associations)],
        )
    mendeliome = panelapp_snapshot[MENDELIOME_PANEL_ID]
    genea = mendeliome["genes"][0]
    genea["confidence_level"] = "2"
    genea["publications"] = ["PMID: 11111"]
    genea["evaluations"] = (
        None
        if evaluations is None
        else [{**review, "publications": list(review["publications"])} for review in evaluations]
    )
    target = PanelGeneData(
        panel_ids=[MENDELIOME_PANEL_ID], gene_panel_confidence={1: {MENDELIOME_PANEL_ID: 2}}
    )
    all_panels = AllPanelsData(
        gene_to_panels={1: {MENDELIOME_PANEL_ID}}, panel_names={MENDELIOME_PANEL_ID: "Mendeliome"}
    )
    entries = entries_by_gene([mendeliome], [MENDELIOME_PANEL_ID])
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return build_gene_assessment_results(conn, hgnc_resolver, target, all_panels, None, entries)


# PanelApp curates the association at AMBER; this corpus rates it GREEN.
CITED_UPGRADE = _association("cited disease", panelapp_rating="AMBER", green=True, dois=(P1, P2))
ALREADY_REVIEWED_NOTE = (
    "No finding: every paper of this association is already cited for GENEA on Mendeliome, "
    "where the gene is reviewed."
)


def test_an_upgrade_on_papers_cited_and_reviewed_is_no_finding(
    db_path: Path, hgnc_resolver: HgncResolver, panelapp_snapshot: dict[int, dict[str, Any]]
) -> None:
    """GENEA's only association rests on P1 and P2, which the Mendeliome cites and has
    reviewed: GENEA has no finding and goes under "Already reviewed"."""
    results = _load_cited(db_path, hgnc_resolver, panelapp_snapshot, [(30, CITED_UPGRADE)])
    genea = results.known_genes[0]
    (association,) = genea.associations
    assert (association.rating, association.reviewed_on) == (3, ("Mendeliome",))
    assert genea.findings == []
    groups = group_genes(results.novel_genes, results.known_genes)
    assert [[g.hgnc_id for g in group.genes] for group in groups.already_reviewed] == [[1]]
    assert groups.with_findings == []

    statistics = calculate_comprehensive_statistics(db_path, results, NO_PANEL_VALIDATION)
    assert (
        statistics.known_genes_with_findings,
        statistics.upgrade_findings,
        statistics.already_reviewed_findings,
    ) == (0, 0, 1)
    html = _render(db_path, results)
    assert "<strong>Findings already reviewed:</strong> 1 " in html
    section = html[html.index('<section id="already-reviewed">') :]
    assert ALREADY_REVIEWED_NOTE in _strip_tags(_association_section(section, 30))
    # Every association is already reviewed: the review keeps the gene's AMBER. The
    # entry cites P1 by its PMID, which the association's P1 does not repeat.
    assert (genea.prefill.form_type, genea.prefill.rating) == ("review", "AMBER")
    assert genea.prefill.publications == f"11111;{P2}"
    assert genea.prefill.comments == ""
    assert genea.prefill.phenotypes == ("Some disease MIM#100000;GENEA-related cited disease")
    assert genea.prefill.moi == ENUM_TO_PANELAPP_MOI["Monoallelic_and_biallelic"]


def test_an_uncited_paper_makes_the_upgrade_a_finding(
    db_path: Path, hgnc_resolver: HgncResolver, panelapp_snapshot: dict[int, dict[str, Any]]
) -> None:
    upgrade = _association("cited disease", panelapp_rating="AMBER", green=True, dois=(P1, P2, P3))
    results = _load_cited(db_path, hgnc_resolver, panelapp_snapshot, [(30, upgrade)])
    genea = results.known_genes[0]
    assert genea.associations[0].reviewed_on == ()
    assert genea.findings == [Finding(FindingKind.UPGRADE, 3, 30, 2)]
    assert ALREADY_REVIEWED_NOTE not in _strip_tags(_render(db_path, results))


def test_papers_cited_on_an_unreviewed_entry_still_make_findings(
    db_path: Path, hgnc_resolver: HgncResolver, panelapp_snapshot: dict[int, dict[str, Any]]
) -> None:
    """Nobody has reviewed GENEA on the Mendeliome, so its cited papers are evaluated as
    any others: the upgrade is a finding, flagged as unreviewed."""
    results = _load_cited(
        db_path, hgnc_resolver, panelapp_snapshot, [(30, CITED_UPGRADE)], evaluations=()
    )
    genea = results.known_genes[0]
    assert genea.associations[0].reviewed_on == ()
    assert genea.findings == [Finding(FindingKind.UPGRADE, 3, 30, 2)]
    html = _render(db_path, results)
    assert ALREADY_REVIEWED_NOTE not in _strip_tags(html)
    assert 'class="unreviewed-badge"' in _gene_header(_article(html, "known-gene-1"))


def test_a_reviewed_association_folds_next_to_a_finding_of_its_gene(
    db_path: Path, hgnc_resolver: HgncResolver, panelapp_snapshot: dict[int, dict[str, Any]]
) -> None:
    """A new disease on P3 is a finding, so GENEA is placed by it; the association on
    the cited papers folds under it with the reason it is no finding."""
    new_disease = _association("new disease", status="new_disease", independent=2, dois=(P3,))
    results = _load_cited(
        db_path, hgnc_resolver, panelapp_snapshot, [(30, CITED_UPGRADE), (31, new_disease)]
    )
    genea = results.known_genes[0]
    assert genea.findings == [Finding(FindingKind.NEW_DISEASE, 2, 31, None)]
    assert [a.id for a in genea.other_associations] == [30]
    article = _article(_render(db_path, results), "known-gene-1")
    assert _fold_summary(article) == "Already curated in PanelApp (1): GENEA-related cited disease"
    assert ALREADY_REVIEWED_NOTE in _strip_tags(_association_section(article, 30))


def test_reviews_panelapp_could_not_serve_keep_the_finding(
    db_path: Path, hgnc_resolver: HgncResolver, panelapp_snapshot: dict[int, dict[str, Any]]
) -> None:
    """GENEA's Mendeliome reviews are unknown: it may be unreviewed, and its reviews may
    cite other papers, so the upgrade on P1 alone stays a finding."""
    upgrade = _association("cited disease", panelapp_rating="AMBER", green=True, dois=(P1,))
    results = _load_cited(
        db_path, hgnc_resolver, panelapp_snapshot, [(30, upgrade)], evaluations=None
    )
    genea = results.known_genes[0]
    assert genea.associations[0].reviewed_on == ()
    assert genea.findings == [Finding(FindingKind.UPGRADE, 3, 30, 2)]
    assert genea.unreviewed_target_panels == []


def test_only_reviewed_associations_that_would_be_findings_are_counted(
    db_path: Path, hgnc_resolver: HgncResolver, panelapp_snapshot: dict[int, dict[str, Any]]
) -> None:
    """Both associations rest on cited papers; only the upgrade would have been a finding."""
    same_rating = _association("other cited disease", panelapp_rating="AMBER", dois=(P1,))
    results = _load_cited(
        db_path, hgnc_resolver, panelapp_snapshot, [(30, CITED_UPGRADE), (31, same_rating)]
    )
    assert [a.already_reviewed for a in results.known_genes[0].associations] == [True, True]
    statistics = calculate_comprehensive_statistics(db_path, results, NO_PANEL_VALIDATION)
    assert statistics.already_reviewed_findings == 1


def test_a_gene_without_findings_is_prefilled_without_its_reviewed_associations(
    db_path: Path, hgnc_resolver: HgncResolver, panelapp_snapshot: dict[int, dict[str, Any]]
) -> None:
    """GENEA's AMBER association on P3 is no finding either; the review describes it
    alone, at AMBER, without the already-reviewed GREEN upgrade's papers."""
    curated = _association("uncited disease", panelapp_rating="AMBER", dois=(P3,))
    results = _load_cited(
        db_path, hgnc_resolver, panelapp_snapshot, [(30, CITED_UPGRADE), (31, curated)]
    )
    genea = results.known_genes[0]
    assert genea.findings == []
    prefill = genea.prefill
    # The entry's PMID 11111 first, then the association's paper
    assert (prefill.form_type, prefill.rating, prefill.publications) == (
        "review",
        "AMBER",
        "11111;33333",
    )
    assert prefill.phenotypes == "Some disease MIM#100000;GENEA-related uncited disease"


def test_a_scoped_report_names_its_panel_in_its_title(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    monthly = _render(db_path, _load(db_path, hgnc_resolver))
    assert f"<title>{MONTHLY_REPORT_TITLE}</title>" in monthly
    assert f"<h1>{MONTHLY_REPORT_TITLE}</h1>" in monthly
    scoped = _render(db_path, _load_scoped(db_path, hgnc_resolver, genea_rating=1))
    title = f"Ataxia ({ATAXIA_PANEL_ID}): Literature Assessment Report"
    assert f"<title>{title}</title>" in scoped
    assert f"<h1>{title}</h1>" in scoped


def _rejected() -> str:
    return json.dumps({"relevant": False, "screen": _screen(False, "HIGH")})


def _add_papers(
    db_path: Path,
    papers: list[tuple[str, int | None, str, str | None, str | None]],
    citations: list[tuple[int, str]],
    journal: str | None = None,
) -> None:
    """Add (doi, pmid, source_type, relevance_assessment_json, evidence_extraction_json)
    papers and the (hgnc_id, doi) PanelApp citations of a run's seeding."""
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO papers (doi, pmid, title, journal, source, source_type,
                                relevance_assessment_json, evidence_extraction_json)
            VALUES (?, ?, ?, ?, 'pubmed', ?, ?, ?)
            """,
            [
                (doi, pmid, f"Title {doi}", journal, source_type, relevance, extraction)
                for doi, pmid, source_type, relevance, extraction in papers
            ],
        )
        conn.executemany(
            "INSERT INTO panelapp_citations (hgnc_id, paper_doi) VALUES (?, ?)", citations
        )


def test_featured_journals_leave_out_unread_screen_rejections_in_a_scoped_run(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """In a featured journal: Paper B, which contributes to the new gene GENEB; a paper
    the screen rejected and the run did not read; one it rejected but the run read as
    PanelApp cites it for GENEB, without evidence for the report; and one refused."""
    journal = "Nature genetics"
    _add_papers(
        db_path,
        [
            ("10.1/fj-rejected", None, "initial", _rejected(), None),
            ("10.1/fj-cited", None, "initial", _rejected(), _extraction(3)),
            (
                "10.1/fj-refused",
                None,
                "initial",
                json.dumps(refused_assessment(Refusal(MODEL, "bio"))),
                None,
            ),
        ],
        [(3, "10.1/fj-cited")],
        journal,
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE papers SET journal = ? WHERE doi = '10.1/b'", (journal,))

    def buckets(sections: FavoriteJournalSections) -> dict[str, list[str]]:
        return {
            field.name: [paper.doi for paper in getattr(sections, field.name)]
            for field in dataclasses.fields(sections)
            if getattr(sections, field.name)
        }

    results = _load(db_path, hgnc_resolver)
    monthly = load_favorite_journal_papers(
        db_path, results.novel_genes, results.known_genes, scoped=False
    )
    assert buckets(monthly) == {
        "novel": ["10.1/b"],
        "refused": ["10.1/fj-refused"],
        "not_relevant": ["10.1/fj-rejected"],
        "read_no_evidence": ["10.1/fj-cited"],
    }
    assert monthly.total == 4
    monthly_html = _render(db_path, results, favorite_journal_papers=monthly)
    assert "Papers in Featured Journals (4)</a>" in _toc(monthly_html)
    assert "<strong>Screened out (not relevant)</strong> (1)" in monthly_html
    assert "<strong>Read, no evidence in this report</strong> (1)" in monthly_html

    scoped_results = _load_scoped(db_path, hgnc_resolver, genea_rating=1)
    scoped = load_favorite_journal_papers(
        db_path, scoped_results.novel_genes, scoped_results.known_genes, scoped=True
    )
    assert buckets(scoped) == {
        "novel": ["10.1/b"],
        "refused": ["10.1/fj-refused"],
        "read_no_evidence": ["10.1/fj-cited"],
    }
    assert scoped.total == 3
    scoped_html = _render(db_path, scoped_results, favorite_journal_papers=scoped)
    assert "Papers in Featured Journals (3)</a>" in _toc(scoped_html)
    assert "<h2>Papers in Featured Journals (3)</h2>" in scoped_html
    assert "Screened out" not in scoped_html
    assert (
        "The papers the screen rejected that PanelApp does not cite for any of the run's genes "
        "are left out"
    ) in scoped_html


def test_paper_sources_split_the_read_papers_by_how_they_entered_the_run(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    """Paper B passed the screen and makes GENEB (HGNC:3) a run gene. PanelApp cites for
    GENEB an initial paper the screen rejected, one it passed, and an expansion paper
    seeding added. Another expansion paper is a tournament pick. Paper A, never screened,
    and a rejected paper cited for GENEA, which is no run gene, are outside the corpus."""
    _add_papers(
        db_path,
        [
            ("10.1/cited", None, "initial", _rejected(), _extraction(3)),
            (
                "10.1/cited-relevant",
                None,
                "initial",
                json.dumps({"relevant": True, "screen": _screen(True, "HIGH")}),
                _extraction(3),
            ),
            ("10.1/cited-expansion", None, "expansion", None, _extraction(3)),
            ("10.1/expansion", None, "expansion", None, _extraction(3)),
            ("10.1/cited-other-gene", None, "initial", _rejected(), _extraction(1)),
        ],
        [
            (3, "10.1/cited"),
            (3, "10.1/cited-relevant"),
            (3, "10.1/cited-expansion"),
            (1, "10.1/cited-other-gene"),
        ],
    )
    statistics = calculate_comprehensive_statistics(db_path, results, NO_PANEL_VALIDATION)
    assert statistics.paper_sources == PaperSources(
        relevant=2, cited=2, expansion=1, outside_corpus=2
    )
    html = _render(db_path, results)
    assert "<li>Passed by the scope screen: 2 papers</li>" in html
    assert (
        "<li>Cited by PanelApp for one of the run's genes, not passed by the screen: 2 papers</li>"
    ) in html
    assert "<li>Literature expansion: 1 paper</li>" in html
    assert (
        "<li>Neither passed by the screen nor cited for one of the run's genes: 2 papers</li>"
    ) in html


def test_a_corpus_without_papers_outside_it_or_expansion_papers_lists_only_two_sources(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE papers SET evidence_extraction_json = NULL WHERE doi = '10.1/a'")
    html = _render(db_path, results)
    assert "<li>Passed by the scope screen: 1 paper</li>" in html
    assert "Literature expansion" not in html
    assert "Neither passed by the screen" not in html


def test_panel_validation_counts_the_screen_misses_read_through_panelapp_citations(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    """Of four screen misses, PanelApp cites two for the run gene GENEB, of which the run
    has read one so far; it cites another for GENEA, no run gene, and the fourth not at
    all."""
    _add_papers(
        db_path,
        [
            (
                "10.1/pass",
                900,
                "initial",
                json.dumps({"relevant": True, "screen": _screen(True, "HIGH")}),
                None,
            ),
            ("10.1/miss-read", 901, "initial", _rejected(), _extraction(3)),
            ("10.1/miss-other-gene", 902, "initial", _rejected(), _extraction(1)),
            ("10.1/miss-unread", 903, "initial", _rejected(), None),
            ("10.1/miss-cited-unread", 904, "initial", _rejected(), None),
        ],
        [(3, "10.1/miss-read"), (1, "10.1/miss-other-gene"), (3, "10.1/miss-cited-unread")],
    )
    validation = load_panel_publications_validation(
        db_path, PanelPublications(pmids={900, 901, 902, 903, 904}, dois=set())
    )
    assert len(validation.screen_misses) == 4
    assert validation.screen_misses_read == 1
    assert validation.sensitivity_pct == 20.0
    html = _render(db_path, results, validation)
    assert (
        "<li>Missed by the scope screen: 4, of which 1 was read in full all the same, as "
        "PanelApp cites them for one of the run's genes</li>"
    ) in html
    assert (
        "1 assessed relevant, 4 missed by the scope screen. Of the missed ones, 1 was read in "
        "full all the same, as PanelApp cites them for one of the run's genes."
    ) in html
    assert "<li>Screened in this run: 5</li>" in html


def test_panel_validation_without_read_misses_keeps_the_plain_line(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    _add_papers(db_path, [("10.1/miss-unread", 903, "initial", _rejected(), None)], [])
    validation = load_panel_publications_validation(
        db_path, PanelPublications(pmids={903}, dois=set())
    )
    assert validation.screen_misses_read == 0
    html = _render(db_path, results, validation)
    assert "<li>Missed by the scope screen: 1</li>" in html
    assert "read in full all the same" not in html


def _counts(
    reported: int | None, qualifying: int | None, independent: int | None
) -> dict[str, Any]:
    return {
        "reported_family_count": reported,
        "family_count": qualifying,
        "independent_family_count": independent,
    }


@pytest.mark.parametrize(
    ("entity", "derived"),
    [
        (_counts(3, 3, 2), DerivedFamilies.NOT_EVIDENT),
        # Families failed the qualifying gate; some qualifying ones may still be derived
        (_counts(5, 3, 3), DerivedFamilies.NOT_EVIDENT),
        (_counts(25, 71, 71), DerivedFamilies.SOME),
        (_counts(None, 20, 18), DerivedFamilies.ALL),
        (_counts(None, 0, 0), DerivedFamilies.NOT_EVIDENT),
        (_counts(None, None, None), DerivedFamilies.NOT_EVIDENT),
        ({"family_count": 4, "independent_family_count": 4}, DerivedFamilies.NOT_EVIDENT),
    ],
)
def test_derived_families(entity: dict[str, Any], derived: DerivedFamilies) -> None:
    assert derived_families(entity) == derived


def _family_counts(section: str) -> str:
    match = re.search(r'<p class="entity-count-summary">(.*?)</p>', section, re.DOTALL)
    assert match is not None
    return _strip_tags(match.group(1))


def test_family_counts_say_when_patients_count_as_families(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """Association 30 has fewer reported families than qualifying ones, 31 none reported;
    32 has as many reported as qualifying, and 33 fewer qualifying than reported."""
    associations = {
        30: _counts(25, 71, 71),
        31: _counts(None, 20, 18),
        32: _counts(3, 3, 2),
        33: _counts(5, 3, 3),
    }
    stored = []
    for position, (association_id, counts) in enumerate(associations.items(), start=4):
        assessment = json.loads(_association(f"disease {association_id}")) | counts
        stored.append((association_id, position, json.dumps(assessment)))
    _add_genea_associations(db_path, stored)
    html = _render(db_path, _load(db_path, hgnc_resolver))
    assert _family_counts(_association_section(html, 30)) == (
        "Patients: 4; Families: 25 reported, 71 qualifying (includes patients counted as one "
        "family each, where papers report patients without grouping them into families)"
    )
    assert _family_counts(_association_section(html, 31)) == (
        "Patients: 4; Families: 20 qualifying → 18 independent (patients counted as one "
        "family each, where papers report patients without grouping them into families)"
    )
    assert _family_counts(_association_section(html, 32)) == (
        "Patients: 4; Families: 3 reported, 3 qualifying → 2 independent"
    )
    assert _family_counts(_association_section(html, 33)) == (
        "Patients: 4; Families: 5 reported, 3 qualifying"
    )


def test_paper_entity_table_marks_qualifying_families_counted_per_patient(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    entities = [
        _paper_entity("registry", 20, "A registry.", ()) | _counts(None, 20, 18),
        _paper_entity("mixed", 10, "Families and a cohort.", ()) | _counts(4, 10, 10),
        _paper_entity("families", 3, "Three families.", ()),
    ]
    extraction = {"gene_evaluations": [{"hgnc_id": 1, "disease_entities": entities}]}
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE papers SET evidence_extraction_json = ? WHERE doi = '10.1/a'",
            (json.dumps(extraction),),
        )
    known = _article(_render(db_path, _load(db_path, hgnc_resolver)), "known-gene-1")
    evidence = known[known.index('<details class="individual-evidence-extraction">') :]
    table = evidence[: evidence.index("</table>")]
    rows = [_strip_tags(row) for row in re.findall(r"<tr>(.*?)</tr>", table)[1:]]
    # Description, MoI, patients, reported and qualifying families, by independent families
    assert rows == [
        "registryMonoallelic20NR20 → 18 (per patient)",
        "mixedMonoallelic10410 (partly per patient)",
        "familiesMonoallelic333",
    ]


def _why_considered(html: str, doi: str, pmid: int | None = None) -> str:
    """The "Why considered" reason on the paper card of *doi*, which links to PubMed when
    the paper has a *pmid*."""
    href = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else f"https://doi.org/{doi}"
    card = html[html.index(f'<a href="{href}" target="_blank">') :]
    match = re.search(r"<strong>Why considered:</strong>\s*(.*?)\s*</li>", card, re.DOTALL)
    assert match is not None
    return match.group(1)


def test_a_paper_card_says_when_the_run_read_a_paper_because_panelapp_cites_it(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """GENEB's papers: Paper B, which the screen passed; an initial paper it passed that
    PanelApp also cites; one it rejected that PanelApp cites; and an expansion paper
    seeding added. The screen passing a paper comes before PanelApp citing it. Paper A,
    never screened and cited for no run gene, keeps its initial search."""
    passed = json.dumps({"relevant": True, "screen": _screen(True, "HIGH")})
    _add_papers(
        db_path,
        [
            ("10.1/cited-relevant", None, "initial", passed, _extraction(3)),
            ("10.1/cited", None, "initial", _rejected(), _extraction(3)),
            ("10.1/cited-expansion", None, "expansion", None, _extraction(3)),
        ],
        [(3, "10.1/cited-relevant"), (3, "10.1/cited"), (3, "10.1/cited-expansion")],
    )
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (3, 'GENEB', ?, ?)",
            [
                ("10.1/cited-relevant", "recent_evidence"),
                ("10.1/cited", "recent_evidence"),
                ("10.1/cited-expansion", "expansion_evidence"),
            ],
        )
    results = _load(db_path, hgnc_resolver)
    papers = {p.doi: p for p in results.novel_genes[0].contributing_papers}
    assert {doi: paper.entered_through_citation for doi, paper in papers.items()} == {
        "10.1/b": False,
        "10.1/cited-relevant": False,
        "10.1/cited": True,
        "10.1/cited-expansion": True,
    }
    geneb = _article(_render(db_path, results), "novel-gene-3")
    cited = "Cited by PanelApp for one of the run's genes"
    assert _why_considered(geneb, "10.1/cited") == cited
    assert _why_considered(geneb, "10.1/cited-expansion") == cited
    assert _why_considered(geneb, "10.1/cited-relevant") == "Passed by the scope screen"
    assert _why_considered(geneb, "10.1/b", 222) == "Passed by the scope screen"
    genea = _article(_render(db_path, results), "known-gene-1")
    assert _why_considered(genea, "10.1/a", 111) == "Initial literature search"


def test_a_cited_rejected_featured_paper_awaiting_manual_download_is_due_to_be_read(
    db_path: Path, hgnc_resolver: HgncResolver
) -> None:
    """PanelApp cites for GENEB a featured-journal paper the screen rejected; its PDF needs a
    manual download. Another one cited for GENEB is still being downloaded. Both are in
    the corpus, so neither counts as screened out, in a monthly or a scoped run."""
    journal = "Nature genetics"
    _add_papers(
        db_path,
        [
            ("10.1/fj-manual", None, "initial", _rejected(), None),
            ("10.1/fj-in-progress", None, "initial", _rejected(), None),
        ],
        [(3, "10.1/fj-manual"), (3, "10.1/fj-in-progress")],
        journal,
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE papers SET download_status = 'manual_required' WHERE doi = '10.1/fj-manual'"
        )
        conn.execute(
            "UPDATE papers SET download_status = 'scheduled' WHERE doi = '10.1/fj-in-progress'"
        )
    for scoped in (False, True):
        results = (
            _load_scoped(db_path, hgnc_resolver, genea_rating=1)
            if scoped
            else _load(db_path, hgnc_resolver)
        )
        sections = load_favorite_journal_papers(
            db_path, results.novel_genes, results.known_genes, scoped=scoped
        )
        assert [p.doi for p in sections.manual_download] == ["10.1/fj-manual"]
        assert sections.not_relevant == []
        assert sections.total == 1


def _add_manual_download_papers(
    db_path: Path,
    papers: list[tuple[str, str, str | None, str]],
    metrics: list[tuple[str, float | None]],
) -> None:
    """Initial PubMed papers awaiting manual download as (doi, journal, journal_issn,
    source_date), and journal_metrics rows as (issn, citedness); a None citedness is a
    journal OpenAlex does not know."""
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO papers (doi, pmid, title, journal, journal_issn, source, source_date, "
            "source_type, download_status) "
            "VALUES (?, 1, 'T', ?, ?, 'pubmed', ?, 'initial', 'manual_required')",
            papers,
        )
        conn.executemany(
            "INSERT INTO journal_metrics (issn, openalex_source_id, two_year_mean_citedness, "
            "fetched_at) VALUES (?, CASE WHEN ? IS NULL THEN NULL ELSE 'S' END, ?, 'now')",
            [(issn, citedness, citedness) for issn, citedness in metrics],
        )


def test_manual_downloads_are_ordered_by_journal_citedness_then_newest_first(
    db_path: Path, results: GeneAssessmentResults
) -> None:
    """Papers whose journal has no figure come last, with preprints, which PubMed
    indexes too: a preprint server's figure says nothing of the paper."""
    _add_manual_download_papers(
        db_path,
        [
            ("10.9/low-old", "J", "2222-2222", "2026-01-01"),
            ("10.9/no-issn", "J", None, "2026-03-01"),
            ("10.9/high", "J", "1111-1111", "2026-01-01"),
            (
                "10.9/preprint",
                "bioRxiv : the preprint server for biology",
                "4444-4444",
                "2026-01-15",
            ),
            ("10.9/unknown-journal", "J", "3333-3333", "2026-02-01"),
            ("10.9/low-new", "J", "2222-2222", "2026-02-01"),
        ],
        [("1111-1111", 7.3), ("2222-2222", 0.6), ("3333-3333", None), ("4444-4444", 9.0)],
    )
    papers = load_manual_download_papers(db_path)
    assert [(p.paper.doi, p.journal_citedness) for p in papers] == [
        ("10.9/high", 7.3),
        ("10.9/low-new", 0.6),
        ("10.9/low-old", 0.6),
        ("10.9/no-issn", None),
        ("10.9/unknown-journal", None),
        ("10.9/preprint", None),
    ]

    html = _render(db_path, results, manual_download_papers=papers)
    section = html[html.index('id="manual-download-papers"') :]
    assert "J (OpenAlex 2-year mean citedness 7.3)" in section
    assert "J (no OpenAlex citedness)" in section
    assert "for biology</li>" in section


def test_manual_downloads_need_every_journal_looked_up(db_path: Path) -> None:
    _add_manual_download_papers(db_path, [("10.9/a", "J", "1111-1111", "2026-01-01")], [])
    with pytest.raises(ValueError, match=r"\['1111-1111'\].*palit journal-metrics"):
        load_manual_download_papers(db_path)
