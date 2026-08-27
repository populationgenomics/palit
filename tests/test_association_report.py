"""Tests for the association-level report: loading, grouping, and rendering."""

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from palit.generate_association_report import (
    AssociationStats,
    GeneSection,
    generate_association_report,
    load_association_report_data,
)
from palit.hgnc import HgncResolver

TEMPLATE_DIR = Path("templates")

# The `source` seed-entities stamps on every entity: aggregator, submitter, export.
SOURCE = "gencc:PanelApp Australia:gencc_submissions_2026-06-22.tsv"

# (hgnc_id, symbol) for the genes used below.
_FIXTURE_GENES = [(20, "AARS1"), (4922, "HK1")]

_CRITERIA = ("criterion_A", "criterion_B", "criterion_C", "criterion_D", "criterion_E")

_WEAKENING_FACTORS = (
    "founder_or_recurrent_variant",
    "inherited_no_segregation_with_affected",
    "missense_no_variant_specific_functional",
    "population_frequency_incompatible_with_moi",
    "phenotype_non_mendelian",
    "functional_below_criterion_c_threshold",
)


# --- helpers ----------------------------------------------------------------


@pytest.fixture
def resolver(tmp_path: Path) -> HgncResolver:
    """An HgncResolver over the fixture genes, loaded through its real loader."""
    docs = [
        {
            "hgnc_id": f"HGNC:{hgnc_id}",
            "symbol": symbol,
            "prev_symbol": [],
            "alias_symbol": [],
            "locus_group": "protein-coding gene",
            "location": "16q22.1",
        }
        for hgnc_id, symbol in _FIXTURE_GENES
    ]
    path = tmp_path / "hgnc.json"
    path.write_text(json.dumps({"response": {"docs": docs}}))
    return HgncResolver.from_file(path)


def _criteria(*, met: tuple[str, ...], doi: str, box_id: int) -> list[dict[str, Any]]:
    """The five criteria, with `met` marked true and each citing one passage."""
    return [
        {
            "name": name,
            "result": name in met,
            "rationale": f"Rationale for {name}.",
            "confidence": "HIGH" if name in met else "MEDIUM",
            "citations": [
                {"doi": doi, "box_id": box_id, "commentary": f"Passage supporting {name}."}
            ],
        }
        for name in _CRITERIA
    ]


def _extraction_criteria(*, met: tuple[str, ...], box_id: int) -> list[dict[str, Any]]:
    """The five per-block criteria as extraction writes them: box IDs, no DOI."""
    return [
        {
            "name": name,
            "result": name in met,
            "rationale": f"Extraction rationale for {name}.",
            "confidence": "MEDIUM",
            "citations": [{"box_id": box_id, "commentary": f"Extracted passage for {name}."}],
        }
        for name in _CRITERIA
    ]


def _weakening_factors(*, present: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "factor": factor,
            "present": factor in present,
            "details": (
                "Two families share the same haplotype." if factor in present else "Not present"
            ),
        }
        for factor in _WEAKENING_FACTORS
    ]


def _block(
    *,
    entity_id: int | None,
    entity: dict[str, str] | None,
    description: str,
    reasoning: str,
    inheritance_mode: str,
    box_id: int,
    family_count: int = 4,
    independent_family_count: int = 3,
) -> dict[str, Any]:
    """One extracted disease entity block, as stored in evidence_extraction_json."""
    return {
        "description": description,
        "entity": entity,
        "entity_id": entity_id,
        "entity_match_reasoning": reasoning,
        "patient_count": 7,
        "family_count": family_count,
        "independent_family_count": independent_family_count,
        "count_reduction_reasoning": "One family was reported twice.",
        "disease_mechanism": "Loss of function.",
        "previously_reported_sources": [],
        "inheritance_mode": inheritance_mode,
        "inheritance_details": "",
        "citations": [{"box_id": box_id, "commentary": "Cohort description."}],
        "evidence_weakening_factors": _weakening_factors(present=()),
        "evidence_assessments": _extraction_criteria(met=("criterion_A",), box_id=box_id),
    }


def _extraction(gene_symbol: str, hgnc_id: int, blocks: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "genome_build": "GRCh38",
            "gene_evaluations": [
                {
                    "gene_symbol": gene_symbol,
                    "hgnc_id": hgnc_id,
                    "summary": f"{gene_symbol} variants in a patient cohort.",
                    "variants": [],
                    "families": "Four families.",
                    "disease_entities": blocks,
                    "variants_summary": "Three missense variants.",
                    "population_freq": "Absent from gnomAD.",
                    "segregation": "Segregates in two families.",
                    "functional": "Patient fibroblast assay.",
                    "prior_reports": "None.",
                    "quality_concerns": [],
                }
            ],
        }
    )


def _assessment(
    *,
    summary: str,
    doi: str,
    box_id: int,
    met: tuple[str, ...],
    inheritance_mode: str,
    family_count: int,
    independent_family_count: int,
    weakening_present: tuple[str, ...] = (),
    quality_concerns: list[dict[str, Any]] | None = None,
) -> str:
    return json.dumps(
        {
            "summary": summary,
            "families": f"{family_count} families across the corpus.",
            "variants": "Three missense variants and one frameshift.",
            "population_freq": "Absent from gnomAD v4.",
            "segregation": "Segregates with disease in two families.",
            "functional": "Patient fibroblast assay shows reduced activity.",
            "prior_reports": "No prior curation.",
            "patient_count": 9,
            "family_count": family_count,
            "independent_family_count": independent_family_count,
            "count_reduction_reasoning": (
                "One family was reported in both papers."
                if independent_family_count != family_count
                else "No reduction"
            ),
            "disease_mechanism": "Loss of function.",
            "inheritance_mode": inheritance_mode,
            "inheritance_details": "",
            "citations": [{"doi": doi, "box_id": box_id, "commentary": "Cohort description."}],
            "evidence_weakening_factors": _weakening_factors(present=weakening_present),
            "evidence_assessments": _criteria(met=met, doi=doi, box_id=box_id),
            "quality_concerns": quality_concerns or [],
        }
    )


def _seed_db(path: Path) -> Path:
    """Build a scratch database: two genes, three associations, two papers."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(Path("schema.sql").read_text())

        conn.executemany(
            """
            INSERT INTO gene_disease_entities
                (id, hgnc_id, mondo_id, disease_title, moi, gencc_moi_titles,
                 gencc_classification, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    1,
                    20,
                    "MONDO:0000001",
                    "Alpha syndrome",
                    "Monoallelic",
                    "Autosomal dominant",
                    "Definitive",
                    SOURCE,
                ),
                (
                    2,
                    20,
                    "MONDO:0000001",
                    "Alpha syndrome",
                    "Biallelic",
                    "Autosomal recessive",
                    "Limited",
                    SOURCE,
                ),
                (
                    3,
                    4922,
                    "MONDO:0000002",
                    "Beta syndrome",
                    "Biallelic",
                    "Autosomal recessive",
                    "Moderate",
                    SOURCE,
                ),
            ],
        )

        bbox_mapping = json.dumps({"1": {"page": 3}, "2": {"page": 5}})
        conn.executemany(
            """
            INSERT INTO papers
                (doi, pmid, title, abstract, authors, journal, source, source_date,
                 source_type, source_details, download_status, evidence_extraction_json,
                 bbox_mapping)
            VALUES (?, ?, ?, ?, ?, ?, 'pubmed', ?, 'initial', 'search', 'downloaded', ?, ?)
            """,
            [
                (
                    "10.1000/one",
                    111,
                    "Monoallelic AARS1 variants cause Alpha syndrome",
                    "An abstract for the first paper.",
                    "Smith, Jane; Doe, John",
                    "Journal of Testing",
                    "2020-04-01",
                    _extraction(
                        "AARS1",
                        20,
                        [
                            _block(
                                entity_id=1,
                                entity={"mondo_id": "MONDO:0000001", "moi": "Monoallelic"},
                                description="Alpha syndrome, dominant form",
                                reasoning="Dominant Alpha syndrome matches the listed association.",
                                inheritance_mode="Monoallelic",
                                box_id=1,
                            ),
                            _block(
                                entity_id=None,
                                entity=None,
                                description="Isolated optic atrophy",
                                reasoning="No listed association covers isolated optic atrophy.",
                                inheritance_mode="Monoallelic",
                                box_id=2,
                            ),
                        ],
                    ),
                    bbox_mapping,
                ),
                (
                    "10.1000/two",
                    222,
                    "Biallelic AARS1 variants in Alpha syndrome",
                    "An abstract for the second paper.",
                    "Brown, Ada",
                    "Journal of Testing",
                    "2021-06-01",
                    _extraction(
                        "AARS1",
                        20,
                        [
                            _block(
                                entity_id=2,
                                entity={"mondo_id": "MONDO:0000001", "moi": "Biallelic"},
                                description="Alpha syndrome, recessive form",
                                reasoning="Recessive Alpha syndrome matches the listed association.",
                                inheritance_mode="Monoallelic",
                                box_id=1,
                                family_count=2,
                                independent_family_count=1,
                            )
                        ],
                    ),
                    bbox_mapping,
                ),
            ],
        )

        conn.executemany(
            """
            INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source)
            VALUES (?, ?, ?, 'recent_evidence')
            """,
            [(20, "AARS1", "10.1000/one"), (20, "AARS1", "10.1000/two")],
        )

        conn.executemany(
            "INSERT INTO entity_mentions (entity_id, paper_doi) VALUES (?, ?)",
            [(1, "10.1000/one"), (2, "10.1000/two")],
        )

        conn.executemany(
            """
            INSERT INTO gene_disease_assessments
                (entity_id, hgnc_id, assessment_raw, assessment_json, paper_id_mapping,
                 filtered_papers_json)
            VALUES (?, ?, 'raw', ?, ?, ?)
            """,
            [
                (
                    1,
                    20,
                    _assessment(
                        summary="Smith2020 reports three dominant families with Alpha syndrome.",
                        doi="10.1000/one",
                        box_id=1,
                        met=("criterion_A", "criterion_D", "criterion_E"),
                        inheritance_mode="Monoallelic",
                        family_count=3,
                        independent_family_count=3,
                        weakening_present=("founder_or_recurrent_variant",),
                    ),
                    json.dumps({"Smith2020": "10.1000/one"}),
                    None,
                ),
                (
                    2,
                    20,
                    _assessment(
                        summary="Brown2021 reports two recessive families, one shared with Smith2020.",
                        doi="10.1000/two",
                        box_id=1,
                        met=("criterion_E",),
                        inheritance_mode="Biallelic",
                        family_count=2,
                        independent_family_count=1,
                        quality_concerns=[
                            {
                                "concern": "The two papers appear to share a family.",
                                "dois": ["10.1000/two"],
                                "citations": [
                                    {
                                        "doi": "10.1000/two",
                                        "box_id": 2,
                                        "commentary": "Overlapping pedigree.",
                                    }
                                ],
                            }
                        ],
                    ),
                    json.dumps({"Brown2021": "10.1000/two"}),
                    json.dumps([{"doi": "10.1000/three", "reason": "Preprint family gate"}]),
                ),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture
def report_data(
    tmp_path: Path, resolver: HgncResolver
) -> tuple[list[GeneSection], AssociationStats]:
    return load_association_report_data(_seed_db(tmp_path / "db.sqlite"), resolver)


def _visible_text(html: str) -> str:
    """Strip tags so assertions run against what a reader actually sees."""
    return re.sub(r"<[^>]+>", " ", html)


# --- loading ----------------------------------------------------------------


def test_genes_are_grouped_and_sorted_by_symbol(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    sections, _ = report_data
    assert [s.hgnc_symbol for s in sections] == ["AARS1", "HK1"]
    assert sections[0].paper_count == 2


def test_associations_sort_by_rating_then_independent_families(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    sections, _ = report_data
    aars1 = sections[0]
    assert [(a.entity.moi, a.rating) for a in aars1.associations] == [
        ("Monoallelic", 3),
        ("Biallelic", 1),
    ]


def test_same_disease_under_two_modes_stays_two_associations(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    sections, _ = report_data
    mondo_ids = {a.entity.mondo_id for a in sections[0].associations}
    assert mondo_ids == {"MONDO:0000001"}
    assert len(sections[0].associations) == 2


def test_association_without_evidence_is_listed_as_unassessed(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    sections, stats = report_data
    hk1 = sections[1]
    assert hk1.associations == []
    assert [e.disease_title for e in hk1.unassessed] == ["Beta syndrome"]
    assert stats.unassessed == 1


def test_unattributed_block_is_collected_with_its_reasoning(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    sections, stats = report_data
    unattributed = sections[0].unattributed
    assert len(unattributed) == 1
    assert unattributed[0].display_label == "PMID 111"
    assert unattributed[0].block["description"] == "Isolated optic atrophy"
    assert "isolated optic atrophy" in unattributed[0].block["entity_match_reasoning"]
    assert stats.unattributed_blocks == 1


def test_cross_tab_counts_our_rating_against_the_source_classification(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    _, stats = report_data
    # Definitive (3) assessed GREEN (3); Limited (1) assessed RED (1).
    assert stats.rating_vs_gencc == {(3, 3): 1, (1, 1): 1}
    assert (stats.green, stats.amber, stats.red) == (1, 0, 1)
    assert stats.assessed == 2
    assert stats.contributing_papers == 2
    assert stats.source_label == SOURCE


def test_moi_mismatch_flags_papers_reporting_another_mode(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    sections, _ = report_data
    monoallelic, biallelic = sections[0].associations
    assert monoallelic.moi_mismatch_dois == []
    assert not monoallelic.assessed_moi_mismatch
    # The paper reports monoallelic evidence under the biallelic association.
    assert biallelic.moi_mismatch_dois == ["10.1000/two"]
    assert not biallelic.assessed_moi_mismatch


def test_paper_ids_are_rewritten_to_display_ids(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    sections, _ = report_data
    monoallelic = sections[0].associations[0]
    assert "PMID 111" in monoallelic.assessment_json["summary"]
    assert "Smith2020" not in monoallelic.assessment_json["summary"]
    assert [p.display_id for p in monoallelic.contributing_papers] == ["PMID 111"]


def test_filtered_papers_are_carried_through(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    sections, _ = report_data
    biallelic = sections[0].associations[1]
    assert [f.doi for f in biallelic.filtered_papers] == ["10.1000/three"]
    assert biallelic.filtered_papers[0].reason == "Preprint family gate"


# --- rendering --------------------------------------------------------------


def test_report_renders_every_section(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    sections, stats = report_data
    html = generate_association_report(sections, stats, TEMPLATE_DIR, pdf_links=True)
    text = _visible_text(html)

    # The same disease under two inheritance modes is two separate articles.
    assert 'id="assoc-1"' in html
    assert 'id="assoc-2"' in html
    assert "Alpha syndrome (monoallelic)" in text
    assert "Alpha syndrome (biallelic)" in text
    # The prompt gloss carries a legacy parenthetical; display labels must not.
    assert "(autosomal dominant)" not in text
    assert "(autosomal recessive)" not in text

    assert "crosstab-table" in html
    assert "Strong-equiv" in text
    assert "Founder or recurrent variant" in text
    assert "Checked and not present (5)" in text
    assert "Unattributed evidence (1)" in text
    assert "Isolated optic atrophy" in text
    assert "Beta syndrome (biallelic) — no evidence in this corpus" in text
    assert "Preprint family gate" in text
    assert "The two papers appear to share a family." in text
    assert "3 fixed gene" in text
    assert f"associations sourced from {SOURCE}" in text

    # Citations link into the symlinked PDF directory, by page.
    assert "papers/10.1000%252Fone.pdf#page=3" in html
    assert "None" not in text


def test_gene_header_counts_fixed_associations_then_assessed(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    sections, stats = report_data
    text = _visible_text(generate_association_report(sections, stats, TEMPLATE_DIR, pdf_links=True))

    # AARS1: both fixed associations were assessed, so no parenthetical.
    assert "AARS1 HGNC:20 — 2 associations · 2 papers" in " ".join(text.split())
    # HK1: one fixed association, none assessed.
    assert "HK1 HGNC:4922 — 1 association (0 assessed) · 0 papers" in " ".join(text.split())


def test_report_without_pdf_links_renders_plain_citations(
    report_data: tuple[list[GeneSection], AssociationStats],
) -> None:
    sections, stats = report_data
    html = generate_association_report(sections, stats, TEMPLATE_DIR, pdf_links=False)

    assert ".pdf#page=" not in html
    assert '<span class="citation-ref">[PMID 111:p3]</span>' in html
