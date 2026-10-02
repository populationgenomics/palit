"""Tests for the GenCC loader on a small export and MONDO fixture."""

import csv
from pathlib import Path

import pronto
import pytest

from palit.gencc import (
    DisputeSubmission,
    GenccIndex,
    GeneGencc,
    PaaAssociation,
    load_gencc,
)

# The export's columns, in order.
COLUMNS = [
    "uuid",
    "gene_curie",
    "gene_symbol",
    "disease_curie",
    "disease_title",
    "disease_original_curie",
    "disease_original_title",
    "classification_curie",
    "classification_title",
    "moi_curie",
    "moi_title",
    "submitter_curie",
    "submitter_title",
    "submitted_as_hgnc_id",
    "submitted_as_hgnc_symbol",
    "submitted_as_disease_id",
    "submitted_as_disease_name",
    "submitted_as_moi_id",
    "submitted_as_moi_name",
    "submitted_as_submitter_id",
    "submitted_as_submitter_name",
    "submitted_as_classification_id",
    "submitted_as_classification_name",
    "submitted_as_date",
    "submitted_as_public_report_url",
    "submitted_as_notes",
    "submitted_as_pmids",
    "submitted_as_assertion_criteria_url",
    "submitted_as_submission_id",
    "submitted_run_date",
]

MONDO_OBO = """\
format-version: 1.2
ontology: mondo

[Term]
id: MONDO:0000001
name: disease A
def: "A disease of the first kind." []

[Term]
id: MONDO:0000002
name: disease B
def: "A disease of the second kind." []

[Term]
id: MONDO:0000003
name: disease C
"""

PAA_DATE = "2025-01-17T00:00:00.000000Z"


def _row(
    hgnc_id: int,
    mondo_id: str,
    disease_title: str,
    classification: str,
    moi: str,
    submitter: str = "PanelApp Australia",
    date: str = PAA_DATE,
    pmids: str = "",
    notes: str = "",
    symbol: str = "SYMBOL",
) -> dict[str, str]:
    row = dict.fromkeys(COLUMNS, "")
    row |= {
        "gene_curie": f"HGNC:{hgnc_id}",
        "gene_symbol": symbol,
        "disease_curie": mondo_id,
        "disease_title": disease_title,
        "classification_title": classification,
        "moi_title": moi,
        "submitter_title": submitter,
        "submitted_as_date": date,
        "submitted_as_notes": notes,
        "submitted_as_pmids": pmids,
    }
    return row


def _write_export(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def mondo(tmp_path: Path) -> pronto.Ontology:
    path = tmp_path / "mondo.obo"
    path.write_text(MONDO_OBO, encoding="utf-8")
    return pronto.Ontology(str(path), encoding="utf-8")


def _load(tmp_path: Path, mondo: pronto.Ontology, rows: list[dict[str, str]]) -> GenccIndex:
    path = tmp_path / "gencc_submissions.tsv"
    _write_export(path, rows)
    return load_gencc(path, mondo)


def test_paa_rows_map_class_and_moi(tmp_path: Path, mondo: pronto.Ontology) -> None:
    index = _load(
        tmp_path,
        mondo,
        [
            _row(10, "MONDO:0000001", "disease A", "Strong", "Autosomal recessive"),
            _row(10, "MONDO:0000002", "disease B", "Moderate", "Semidominant"),
            _row(10, "MONDO:0000003", "disease C", "Limited", "X-linked"),
        ],
    )
    assert index.for_gene(10).paa_associations == (
        PaaAssociation(
            mondo_id="MONDO:0000001",
            disease_title="disease A",
            moi_title="Autosomal recessive",
            moi="Biallelic",
            classification="Strong",
            rating="GREEN",
            date="2025-01-17",
            definition="A disease of the first kind.",
        ),
        PaaAssociation(
            mondo_id="MONDO:0000002",
            disease_title="disease B",
            moi_title="Semidominant",
            moi=None,
            classification="Moderate",
            rating="AMBER",
            date="2025-01-17",
            definition="A disease of the second kind.",
        ),
        PaaAssociation(
            mondo_id="MONDO:0000003",
            disease_title="disease C",
            moi_title="X-linked",
            moi="X-linked",
            classification="Limited",
            rating="RED",
            date="2025-01-17",
            definition="",
        ),
    )
    assert index.for_gene(10).disputes == ()


def test_genes_are_keyed_by_hgnc_id_not_symbol(tmp_path: Path, mondo: pronto.Ontology) -> None:
    index = _load(
        tmp_path,
        mondo,
        [
            _row(10, "MONDO:0000001", "disease A", "Strong", "Autosomal dominant", symbol="SAME"),
            _row(20, "MONDO:0000002", "disease B", "Strong", "Autosomal dominant", symbol="SAME"),
        ],
    )
    assert set(index.genes) == {10, 20}
    assert [a.mondo_id for a in index.for_gene(10).paa_associations] == ["MONDO:0000001"]
    assert [a.mondo_id for a in index.for_gene(20).paa_associations] == ["MONDO:0000002"]


def test_unknown_gene_has_no_records(tmp_path: Path, mondo: pronto.Ontology) -> None:
    index = _load(
        tmp_path, mondo, [_row(10, "MONDO:0000001", "disease A", "Strong", "Autosomal dominant")]
    )
    assert index.for_gene(99) == GeneGencc()


def test_disputes_come_from_every_submitter(tmp_path: Path, mondo: pronto.Ontology) -> None:
    index = _load(
        tmp_path,
        mondo,
        [
            _row(
                10,
                "MONDO:0000001",
                "disease A",
                "Disputed Evidence",
                "Autosomal dominant",
                submitter="ClinGen",
                date="2023-05-10 09:12:00",
                pmids="111, 222",
                notes="Variants exceed the\nallele-frequency threshold.\n",
            ),
            _row(
                10,
                "MONDO:0000002",
                "disease B",
                "Refuted Evidence",
                "Autosomal recessive",
                pmids="333",
            ),
            _row(
                10,
                "MONDO:0000002",
                "disease B",
                "Definitive",
                "Autosomal recessive",
                submitter="Ambry Genetics",
            ),
        ],
    )
    gene = index.for_gene(10)
    assert gene.disputes == (
        DisputeSubmission(
            submitter="ClinGen",
            mondo_id="MONDO:0000001",
            disease_title="disease A",
            moi_title="Autosomal dominant",
            status="Disputed",
            date="2023-05-10",
            pmids=(111, 222),
            rationale="Variants exceed the\nallele-frequency threshold.",
        ),
        DisputeSubmission(
            submitter="PanelApp Australia",
            mondo_id="MONDO:0000002",
            disease_title="disease B",
            moi_title="Autosomal recessive",
            status="Refuted",
            date="2025-01-17",
            pmids=(333,),
            rationale="",
        ),
    )
    # PanelApp Australia's own Refuted row is also one of its GenCC rows, without a rating.
    [paa_row] = gene.paa_associations
    assert paa_row.classification == "Refuted Evidence"
    assert paa_row.rating is None


def test_other_submitters_without_dispute_are_ignored(
    tmp_path: Path, mondo: pronto.Ontology
) -> None:
    index = _load(
        tmp_path,
        mondo,
        [_row(10, "MONDO:0000001", "disease A", "Strong", "Autosomal dominant", "ClinGen")],
    )
    assert index.genes == {}


def test_identical_paa_rows_collapse(tmp_path: Path, mondo: pronto.Ontology) -> None:
    row = _row(10, "MONDO:0000001", "disease A", "Strong", "Autosomal recessive")
    index = _load(tmp_path, mondo, [row, row])
    assert len(index.for_gene(10).paa_associations) == 1


def test_records_are_sorted_by_disease(tmp_path: Path, mondo: pronto.Ontology) -> None:
    index = _load(
        tmp_path,
        mondo,
        [
            _row(10, "MONDO:0000002", "disease B", "Strong", "Autosomal recessive"),
            _row(10, "MONDO:0000001", "disease A", "Strong", "Autosomal recessive"),
            _row(10, "MONDO:0000001", "disease A", "Moderate", "Autosomal dominant"),
        ],
    )
    assert [(a.disease_title, a.moi_title) for a in index.for_gene(10).paa_associations] == [
        ("disease A", "Autosomal dominant"),
        ("disease A", "Autosomal recessive"),
        ("disease B", "Autosomal recessive"),
    ]


def test_unexpected_paa_class_raises(tmp_path: Path, mondo: pronto.Ontology) -> None:
    with pytest.raises(ValueError, match="Definitive"):
        _load(
            tmp_path,
            mondo,
            [_row(10, "MONDO:0000001", "disease A", "Definitive", "Autosomal dominant")],
        )


def test_paa_row_with_unknown_mondo_term_raises(tmp_path: Path, mondo: pronto.Ontology) -> None:
    with pytest.raises(KeyError):
        _load(
            tmp_path,
            mondo,
            [_row(10, "MONDO:0009999", "disease Z", "Strong", "Autosomal dominant")],
        )
