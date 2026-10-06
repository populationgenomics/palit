"""A gene's distinct variants with their gnomAD v4.1 frequencies.

extract-evidence looks up every variant a paper reports and stores one
``variant_frequencies`` row per (variant, paper, gene). A variant is identified by
its ``variant_id``: the pseudo-VCF (chr-pos-ref-alt) when the variant-lookup
service normalised it, else the paper's own text. Rows of different papers with
the same ``variant_id`` are the same variant, and their gnomAD figures, looked
up by genomic coordinate, are identical.
"""

import json
import sqlite3
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


@dataclass(frozen=True)
class Normalization:
    """How the variant-lookup service normalised one paper's text of a variant."""

    hgvs_c: str  # with transcript, e.g. "NM_003579.4:c.1375+1G>T"
    hgvs_p: str  # with protein accession, e.g. "NP_003570.2:p.?"
    # How many DNA changes the text could stand for; above 1 for a protein change
    # that back-translates to several, of which the one most frequent in gnomAD is kept.
    candidate_count: int


@dataclass(frozen=True)
class VariantReport:
    """One paper's report of a variant: one ``variant_frequencies`` row."""

    doi: str
    quote: str  # the extraction's quote for the variant
    original_text: str  # the variant as the extraction wrote it
    normalization: Normalization | None  # None when the service could not normalise it


class GnomadMiss(StrEnum):
    """Why a variant has no gnomAD figures."""

    NOT_FOUND = "not_found"  # normalised, but absent from gnomAD
    LOOKUP_FAILED = "lookup_failed"  # not normalised, so never looked up


@dataclass(frozen=True)
class GnomadFrequency:
    """A variant's gnomAD v4.1 figures."""

    ac: int
    an: int
    homozygote_count: int
    heterozygote_count: int
    hemizygote_count: int
    faf95_popmax: float | None  # None where gnomAD computes none
    faf95_popmax_population: str | None  # gnomAD population code, e.g. "nfe"

    @property
    def allele_frequency(self) -> float:
        return self.ac / self.an


@dataclass(frozen=True)
class GeneVariant:
    """One distinct variant of a gene and every report of it by the given papers."""

    variant_id: str  # ``variant_frequencies.variant_id``
    gnomad: GnomadFrequency | GnomadMiss
    reports: tuple[VariantReport, ...]  # by DOI, then quote

    @property
    def dois(self) -> frozenset[str]:
        """The papers that report this variant."""
        return frozenset(r.doi for r in self.reports)


def _normalization(normalization: dict[str, Any]) -> Normalization | None:
    if "hgvs_c" not in normalization:
        return None
    return Normalization(
        hgvs_c=normalization["hgvs_c"],
        hgvs_p=normalization["hgvs_p"],
        candidate_count=normalization["total_normalizations"],
    )


def _gnomad(gnomad: dict[str, Any]) -> GnomadFrequency | GnomadMiss:
    if gnomad.get("normalization_error"):
        return GnomadMiss.LOOKUP_FAILED
    if gnomad.get("variant_not_found"):
        return GnomadMiss.NOT_FOUND
    return GnomadFrequency(**gnomad)


def load_gene_variants(
    cursor: sqlite3.Cursor, hgnc_id: int, dois: Collection[str]
) -> list[GeneVariant]:
    """The gene's distinct variants reported by any of *dois*, by ``variant_id``."""
    rows = cursor.execute(
        f"""
        SELECT variant_id, paper_doi, quote, normalization, gnomad
        FROM variant_frequencies
        WHERE hgnc_id = ? AND paper_doi IN ({",".join("?" * len(dois))})
        ORDER BY variant_id, paper_doi, quote
        """,
        (hgnc_id, *dois),
    ).fetchall()
    reports: dict[str, list[VariantReport]] = {}
    gnomad: dict[str, GnomadFrequency | GnomadMiss] = {}
    for variant_id, doi, quote, normalization_json, gnomad_json in rows:
        normalization = json.loads(normalization_json)
        reports.setdefault(variant_id, []).append(
            VariantReport(
                doi=doi,
                quote=quote,
                original_text=normalization["original_text"],
                normalization=_normalization(normalization),
            )
        )
        gnomad.setdefault(variant_id, _gnomad(json.loads(gnomad_json)))
    return [
        GeneVariant(
            variant_id=variant_id, gnomad=gnomad[variant_id], reports=tuple(variant_reports)
        )
        for variant_id, variant_reports in reports.items()
    ]
