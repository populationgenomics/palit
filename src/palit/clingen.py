"""ClinGen gene-disease validity curations, keyed by HGNC ID.

A ClinGen expert panel curates one disease of a gene, with one MoI, and classifies
the evidence for it. Its evidence summary often says which disease entities the
panel lumped into the curated disease or split from it. The prompts show a gene's
curations as the precedent for lumping and splitting its associations. Expert
panels curate only some diseases of some genes, so a gene or disease without a
curation has not been judged.

ClinGen publishes the curations in two files. The table (CSV) has one row per
curation. The archive holds one JSON-LD record per curation, with the evidence
summary. A row and its record share the UUID of the curation's snapshot in
ClinGen's curation interface, which is part of the row's report URL. Curations
from the older Gene Curation Express have no record, and a curation published
since the archive was last built has none yet; both have an empty summary.

Genes are matched on the table's HGNC id, never on the symbol.
"""

import csv
import json
import logging
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from palit.reference_files import download_if_stale

logger = logging.getLogger(__name__)

CLINGEN_TABLE_URL = "https://search.clinicalgenome.org/kb/gene-validity/download"
CLINGEN_ARCHIVE_URL = (
    "https://storage.googleapis.com/genegraph-public/clingen-gene-validity-json-latest.tar.gz"
)
CLINGEN_TABLE_FILENAME = "clingen_gene_validity.csv"
CLINGEN_ARCHIVE_FILENAME = "clingen_gene_validity_json.tar.gz"

# The table's header row, which follows a banner of varying content.
TABLE_COLUMNS = [
    "GENE SYMBOL",
    "GENE ID (HGNC)",
    "DISEASE LABEL",
    "DISEASE ID (MONDO)",
    "MOI",
    "SOP",
    "CLASSIFICATION",
    "ONLINE REPORT",
    "CLASSIFICATION DATE",
    "GCEP",
]

MOI_TITLES = {
    "AD": "Autosomal dominant",
    "AR": "Autosomal recessive",
    "XL": "X-linked",
    "SD": "Semidominant",
    "MT": "Mitochondrial",
    "UD": "Undetermined",
}

# The snapshot UUID in a report URL such as
# .../kb/gene-validity/CGGV:assertion_<uuid>-2025-06-23T160000.000Z
_SNAPSHOT_UUID = re.compile(r"CGGV:assertion_([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})-")


@dataclass(frozen=True)
class ClingenCuration:
    """One gene-disease validity curation by a ClinGen expert panel."""

    mondo_id: str
    disease_label: str
    moi: str  # "Autosomal dominant", "Autosomal recessive", ...; see MOI_TITLES
    classification: str  # "Definitive", "Limited", "Disputed", "No Known Disease Relationship", ...
    date: str  # ISO date of the classification
    expert_panel: str
    report_url: str
    evidence_summary: str  # empty when the archive has no record of the curation


@dataclass(frozen=True)
class ClingenIndex:
    """The ClinGen curations of every curated gene."""

    genes: dict[int, tuple[ClingenCuration, ...]]

    def for_gene(self, hgnc_id: int) -> tuple[ClingenCuration, ...]:
        """The gene's curations; empty when no expert panel has curated it."""
        return self.genes.get(hgnc_id, ())


def _hgnc_id(gene_id: str) -> int:
    prefix, _, number = gene_id.partition(":")
    if prefix != "HGNC":
        raise ValueError(f"ClinGen gene id is not an HGNC id: {gene_id!r}")
    return int(number)


def _table_rows(table_path: Path) -> list[dict[str, str]]:
    """The table's data rows by column name.

    A row of ``+`` separators follows the header row.
    """
    with table_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows.index(TABLE_COLUMNS)
    return [dict(zip(TABLE_COLUMNS, row, strict=True)) for row in rows[header + 2 :]]


def _evidence_summaries(archive_path: Path) -> dict[str, str]:
    """The evidence summary of every record in the archive, by snapshot UUID.

    A record without a summary is left out. The records are read from the archive
    without unpacking it.
    """
    summaries: dict[str, str] = {}
    seen: set[str] = set()
    with tarfile.open(archive_path) as archive:
        for member in archive:
            if not member.isfile():
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Cannot read {member.name} from {archive_path}")
            record = json.load(extracted)
            uuid = record["GCISnapshot"].rsplit("/", 1)[1]
            if uuid in seen:
                raise ValueError(f"ClinGen archive has two records for snapshot {uuid}")
            seen.add(uuid)
            summary = record.get("dc:description", "").strip()
            if summary:
                summaries[uuid] = summary
    return summaries


def _summary(report_url: str, summaries: dict[str, str]) -> str:
    match = _SNAPSHOT_UUID.search(report_url)
    return "" if match is None else summaries.get(match.group(1), "")


def load_clingen(table_path: Path, archive_path: Path) -> ClingenIndex:
    """Every curated gene's ClinGen curations, each with its evidence summary.

    A gene's curations are sorted by disease, MoI and date, so the order does not
    depend on the table's.
    """
    summaries = _evidence_summaries(archive_path)
    genes: defaultdict[int, list[ClingenCuration]] = defaultdict(list)
    for row in _table_rows(table_path):
        genes[_hgnc_id(row["GENE ID (HGNC)"])].append(
            ClingenCuration(
                mondo_id=row["DISEASE ID (MONDO)"],
                disease_label=row["DISEASE LABEL"],
                moi=MOI_TITLES[row["MOI"]],
                classification=row["CLASSIFICATION"],
                date=datetime.fromisoformat(row["CLASSIFICATION DATE"]).date().isoformat(),
                expert_panel=row["GCEP"],
                report_url=row["ONLINE REPORT"],
                evidence_summary=_summary(row["ONLINE REPORT"], summaries),
            )
        )
    curations = [c for gene in genes.values() for c in gene]
    logger.info(
        "Loaded ClinGen: %d curations on %d genes, %d with an evidence summary",
        len(curations),
        len(genes),
        sum(bool(c.evidence_summary) for c in curations),
    )
    return ClingenIndex(
        {
            hgnc_id: tuple(sorted(gene, key=lambda c: (c.disease_label, c.moi, c.date)))
            for hgnc_id, gene in genes.items()
        }
    )


def fetch_clingen(data_dir: Path) -> ClingenIndex:
    """ClinGen's curations from *data_dir*, each file downloaded first when missing or stale."""
    table_path = data_dir / CLINGEN_TABLE_FILENAME
    archive_path = data_dir / CLINGEN_ARCHIVE_FILENAME
    download_if_stale(CLINGEN_TABLE_URL, table_path)
    download_if_stale(CLINGEN_ARCHIVE_URL, archive_path)
    return load_clingen(table_path, archive_path)
