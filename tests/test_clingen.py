"""Tests for the ClinGen loader on a small table and archive fixture."""

import csv
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from palit.clingen import TABLE_COLUMNS, ClingenCuration, load_clingen

GENEA = 1
GENEB = 3
UUID_LUMPED = "11111111-1111-4111-8111-111111111111"
UUID_SPLIT = "22222222-2222-4222-8222-222222222222"
UUID_NOT_IN_ARCHIVE = "33333333-3333-4333-8333-333333333333"
UUID_NO_SUMMARY = "44444444-4444-4444-8444-444444444444"
REPORTS = "https://search.clinicalgenome.org/kb/gene-validity"
LUMPED_SUMMARY = (
    "Per criteria outlined by the ClinGen Lumping and Splitting Working Group, the following "
    "disease entities have been lumped into one disease entity, disease A and disease B."
)


def _report_url(uuid: str) -> str:
    return f"{REPORTS}/CGGV:assertion_{uuid}-2025-06-23T160000.000Z"


def _row(
    hgnc_id: int,
    disease: str,
    mondo_id: str,
    moi: str,
    classification: str,
    report_url: str,
    date: str,
) -> list[str]:
    return [
        f"GENE{hgnc_id}",
        f"HGNC:{hgnc_id}",
        disease,
        mondo_id,
        moi,
        "SOP11",
        classification,
        report_url,
        date,
        "Test Gene Curation Expert Panel",
    ]


ROWS = [
    _row(
        GENEA,
        "GENEA-related disease",
        "MONDO:0000010",
        "AD",
        "Definitive",
        _report_url(UUID_LUMPED),
        "2025-06-23T16:00:00.000Z",
    ),
    _row(
        GENEA,
        "disease C",
        "MONDO:0000003",
        "AR",
        "Limited",
        _report_url(UUID_SPLIT),
        "2024-01-02T17:00:00.000Z",
    ),
    _row(
        GENEA,
        "disease C",
        "MONDO:0000003",
        "AD",
        "Disputed",
        _report_url(UUID_NOT_IN_ARCHIVE),
        "2026-11-18T17:00:00.000Z",
    ),
    _row(
        GENEB,
        "disease D",
        "MONDO:0000004",
        "XL",
        "Moderate",
        f"{REPORTS}/CGGCIEX:assertion_10062",
        "2018-10-12T21:07:42.630Z",
    ),
    _row(
        GENEB,
        "disease E",
        "MONDO:0000005",
        "SD",
        "Strong",
        _report_url(UUID_NO_SUMMARY),
        "2023-03-04T05:00:00.000Z",
    ),
]


def _record(uuid: str, summary: str | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": f"https://genegraph.clinicalgenome.org/r/statement-of-{uuid}v1.0.0",
        "GCISnapshot": f"https://genegraph.clinicalgenome.org/r/{uuid}",
        "classification": "Definitive",
    }
    if summary is not None:
        record["dc:description"] = summary
    return record


RECORDS = [
    _record(UUID_LUMPED, f"  {LUMPED_SUMMARY}\n"),
    _record(UUID_SPLIT, "Disease C was split from GENEA-related disease."),
    _record(UUID_NO_SUMMARY, None),
]


def _write_table(path: Path, rows: list[list[str]]) -> None:
    filler = [""] * (len(TABLE_COLUMNS) - 1)
    separators = ["+++++"] * len(TABLE_COLUMNS)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["CLINGEN GENE DISEASE VALIDITY CURATIONS", *filler])
        writer.writerow(["FILE CREATED: 2026-11-19", *filler])
        writer.writerow([f"WEBPAGE: {REPORTS}", *filler])
        writer.writerow(separators)
        writer.writerow(TABLE_COLUMNS)
        writer.writerow(separators)
        writer.writerows(rows)


def _write_archive(path: Path, records: list[dict[str, Any]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for i, record in enumerate(records):
            data = json.dumps(record).encode()
            member = tarfile.TarInfo(f"record-{i}--1.0.0.json")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


@pytest.fixture
def table_path(tmp_path: Path) -> Path:
    path = tmp_path / "clingen.csv"
    _write_table(path, ROWS)
    return path


@pytest.fixture
def archive_path(tmp_path: Path) -> Path:
    path = tmp_path / "clingen.tar.gz"
    _write_archive(path, RECORDS)
    return path


def test_curations_join_the_table_to_the_archive_by_snapshot_uuid(
    table_path: Path, archive_path: Path
) -> None:
    index = load_clingen(table_path, archive_path)
    assert index.for_gene(GENEA)[0] == ClingenCuration(
        mondo_id="MONDO:0000010",
        disease_label="GENEA-related disease",
        moi="Autosomal dominant",
        classification="Definitive",
        date="2025-06-23",
        expert_panel="Test Gene Curation Expert Panel",
        report_url=_report_url(UUID_LUMPED),
        evidence_summary=LUMPED_SUMMARY,
    )


def test_a_genes_curations_are_sorted_by_disease_then_moi(
    table_path: Path, archive_path: Path
) -> None:
    index = load_clingen(table_path, archive_path)
    assert [(c.disease_label, c.moi) for c in index.for_gene(GENEA)] == [
        ("GENEA-related disease", "Autosomal dominant"),
        ("disease C", "Autosomal dominant"),
        ("disease C", "Autosomal recessive"),
    ]
    assert [(c.disease_label, c.moi) for c in index.for_gene(GENEB)] == [
        ("disease D", "X-linked"),
        ("disease E", "Semidominant"),
    ]


def test_a_curation_without_an_archive_record_has_an_empty_summary(
    table_path: Path, archive_path: Path
) -> None:
    index = load_clingen(table_path, archive_path)
    summaries = {
        (c.disease_label, c.moi): c.evidence_summary
        for hgnc_id in (GENEA, GENEB)
        for c in index.for_gene(hgnc_id)
    }
    # Published after the archive was built, a Gene Curation Express curation, and a
    # record without a summary.
    assert summaries[("disease C", "Autosomal dominant")] == ""
    assert summaries[("disease D", "X-linked")] == ""
    assert summaries[("disease E", "Semidominant")] == ""
    assert summaries[("disease C", "Autosomal recessive")] == (
        "Disease C was split from GENEA-related disease."
    )


def test_a_gene_without_curations_has_none(table_path: Path, archive_path: Path) -> None:
    assert load_clingen(table_path, archive_path).for_gene(999) == ()


def test_two_archive_records_for_one_snapshot_are_an_error(
    tmp_path: Path, table_path: Path
) -> None:
    archive_path = tmp_path / "duplicate.tar.gz"
    _write_archive(archive_path, [*RECORDS, _record(UUID_LUMPED, "Another version.")])
    with pytest.raises(ValueError, match=f"two records for snapshot {UUID_LUMPED}"):
        load_clingen(table_path, archive_path)


def test_a_gene_id_that_is_not_an_hgnc_id_is_an_error(tmp_path: Path, archive_path: Path) -> None:
    table_path = tmp_path / "other-ids.csv"
    row = list(ROWS[0])
    row[1] = "NCBIGene:1"
    _write_table(table_path, [row])
    with pytest.raises(ValueError, match="not an HGNC id"):
        load_clingen(table_path, archive_path)


def test_an_unknown_moi_code_is_an_error(tmp_path: Path, archive_path: Path) -> None:
    table_path = tmp_path / "other-moi.csv"
    row = list(ROWS[0])
    row[4] = "ZZ"
    _write_table(table_path, [row])
    with pytest.raises(KeyError, match="ZZ"):
        load_clingen(table_path, archive_path)
