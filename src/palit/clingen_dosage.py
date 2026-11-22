"""ClinGen dosage sensitivity curations, keyed by HGNC ID.

ClinGen's Dosage Sensitivity working group rates, per gene, the evidence that the
loss of one copy (haploinsufficiency) or an extra copy (triplosensitivity) causes
disease. The Gene Curation Coalition's recommendation for curating a loss-of-function
mechanism takes "sufficient evidence for haploinsufficiency" as an established
monoallelic loss of function, so assess-mechanisms shows a gene's curation to the
model. Most genes have none.

ClinGen publishes the curations as one table (CSV) with one row per gene. Genes are
matched on the table's HGNC id, never on the symbol.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from palit.reference_files import download_if_stale

logger = logging.getLogger(__name__)

DOSAGE_TABLE_URL = "https://search.clinicalgenome.org/kb/gene-dosage/download"
DOSAGE_TABLE_FILENAME = "clingen_dosage_sensitivity.csv"

TABLE_COLUMNS = [
    "GENE SYMBOL",
    "HGNC ID",
    "HAPLOINSUFFICIENCY",
    "TRIPLOSENSITIVITY",
    "ONLINE REPORT",
    "DATE",
]


@dataclass(frozen=True)
class DosageCuration:
    """ClinGen's dosage sensitivity curation of one gene."""

    haploinsufficiency: str  # e.g. "Sufficient Evidence for Haploinsufficiency"
    triplosensitivity: str  # e.g. "No Evidence for Triplosensitivity"
    report_url: str
    date: str  # YYYY-MM-DD


def load_dosage(table_path: Path) -> dict[int, DosageCuration]:
    """The table's curations by HGNC id.

    A row of ``+`` separators follows the header row. Raises ValueError for a gene
    id that is not an HGNC id, and for a gene with two rows.
    """
    with table_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows.index(TABLE_COLUMNS)
    curations: dict[int, DosageCuration] = {}
    for row in rows[header + 2 :]:
        _, gene_id, haploinsufficiency, triplosensitivity, report_url, date = row
        prefix, _, number = gene_id.partition(":")
        if prefix != "HGNC":
            raise ValueError(f"ClinGen gene id is not an HGNC id: {gene_id!r}")
        hgnc_id = int(number)
        if hgnc_id in curations:
            raise ValueError(f"Two ClinGen dosage sensitivity rows for HGNC:{hgnc_id}")
        curations[hgnc_id] = DosageCuration(
            haploinsufficiency=haploinsufficiency,
            triplosensitivity=triplosensitivity,
            report_url=report_url,
            date=date[:10],
        )
    logger.info("Loaded %s ClinGen dosage sensitivity curations", f"{len(curations):,}")
    return curations


def fetch_dosage(data_dir: Path) -> dict[int, DosageCuration]:
    """ClinGen's dosage sensitivity curations from *data_dir*, the table downloaded
    first when missing or stale."""
    table_path = data_dir / DOSAGE_TABLE_FILENAME
    download_if_stale(DOSAGE_TABLE_URL, table_path)
    return load_dosage(table_path)
