"""Tests for the GenCC loader on a small export and MONDO fixture."""

import csv
from pathlib import Path

import pronto
import pytest

from palit.gencc import (
    NO_GENCC_TERMS,
    DisputeSubmission,
    GenccIndex,
    GenccTerm,
    GeneGencc,
    MondoRef,
    format_gencc_terms,
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

[Term]
id: MONDO:0000004
name: obsolete disease D
is_obsolete: true
replaced_by: MONDO:0000003

[Term]
id: MONDO:0000005
name: obsolete disease E
is_obsolete: true
consider: MONDO:0000001
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


def test_terms_of_every_submitter_merge_their_mois(tmp_path: Path, mondo: pronto.Ontology) -> None:
    index = _load(
        tmp_path,
        mondo,
        [
            _row(10, "MONDO:0000001", "disease A", "Strong", "Autosomal recessive"),
            _row(10, "MONDO:0000001", "disease A", "Definitive", "Autosomal dominant", "ClinGen"),
            _row(10, "MONDO:0000002", "disease B", "Limited", "Semidominant", "G2P"),
        ],
    )
    assert index.for_gene(10).terms == (
        GenccTerm(
            mondo_id="MONDO:0000001",
            disease_title="disease A",
            moi_titles=("Autosomal dominant", "Autosomal recessive"),
            definition="A disease of the first kind.",
            obsolete=False,
            replaced_by=(),
        ),
        GenccTerm(
            mondo_id="MONDO:0000002",
            disease_title="disease B",
            moi_titles=("Semidominant",),
            definition="A disease of the second kind.",
            obsolete=False,
            replaced_by=(),
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
    assert [t.mondo_id for t in index.for_gene(10).terms] == ["MONDO:0000001"]
    assert [t.mondo_id for t in index.for_gene(20).terms] == ["MONDO:0000002"]


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
    # Disputed and refuted diseases are terms of the gene too.
    assert [t.mondo_id for t in gene.terms] == ["MONDO:0000001", "MONDO:0000002"]


def test_terms_are_sorted_by_title(tmp_path: Path, mondo: pronto.Ontology) -> None:
    index = _load(
        tmp_path,
        mondo,
        [
            _row(10, "MONDO:0000002", "disease B", "Strong", "Autosomal recessive"),
            _row(10, "MONDO:0000001", "disease A", "Strong", "Autosomal recessive"),
        ],
    )
    assert [t.disease_title for t in index.for_gene(10).terms] == ["disease A", "disease B"]


def test_term_unknown_to_mondo_raises(tmp_path: Path, mondo: pronto.Ontology) -> None:
    with pytest.raises(KeyError):
        _load(
            tmp_path,
            mondo,
            [_row(10, "MONDO:0009999", "disease Z", "Strong", "Autosomal dominant")],
        )


def test_obsolete_mondo_terms_keep_the_submitted_id(tmp_path: Path, mondo: pronto.Ontology) -> None:
    index = _load(
        tmp_path,
        mondo,
        [
            _row(10, "MONDO:0000001", "disease A", "Strong", "Autosomal recessive"),
            _row(10, "MONDO:0000004", "disease D", "Strong", "Autosomal recessive"),
            _row(10, "MONDO:0000005", "disease E", "Limited", "Autosomal dominant"),
        ],
    )
    terms = {t.mondo_id: t for t in index.for_gene(10).terms}
    assert (terms["MONDO:0000001"].obsolete, terms["MONDO:0000001"].replaced_by) == (False, ())
    assert (terms["MONDO:0000004"].obsolete, terms["MONDO:0000004"].replaced_by) == (
        True,
        (MondoRef("MONDO:0000003", "disease C"),),
    )
    # A "consider" suggestion is not a replacement.
    assert (terms["MONDO:0000005"].obsolete, terms["MONDO:0000005"].replaced_by) == (True, ())


def test_format_lists_terms_with_their_mois(gencc_index: GenccIndex) -> None:
    assert format_gencc_terms(gencc_index.for_gene(1)) == (
        "- disease A (MONDO:0000001) | Autosomal recessive\n"
        "- disease B (MONDO:0000002) | Autosomal dominant, Autosomal recessive"
    )
    assert format_gencc_terms(gencc_index.for_gene(99)) == NO_GENCC_TERMS
