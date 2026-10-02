"""GenCC submissions that describe PanelApp Australia's curation, keyed by HGNC ID.

PanelApp Australia submits its Mendeliome to GenCC as one row per (gene, MONDO
disease, MoI) with a class. These rows are the only record of its curation that
rates each association separately. Disputed and Refuted submissions are kept from
every submitter, because they gate the criteria of the associations they apply to.

Genes are matched on ``gene_curie`` (``HGNC:<id>``), never on the symbol. The
module also formats a gene's PanelApp Australia rows, with their caveat, for the
prompts that show them.
"""

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast, get_args

import pronto

from palit.mondo_lookup import GENCC_URL, MONDO_OBO_URL, download_if_stale

logger = logging.getLogger(__name__)

GENCC_FILENAME = "gencc_submissions.tsv"
MONDO_FILENAME = "mondo.obo"

PAA_SUBMITTER = "PanelApp Australia"

PaaClassification = Literal[
    "Strong", "Moderate", "Limited", "Disputed Evidence", "Refuted Evidence"
]
PanelRating = Literal["GREEN", "AMBER", "RED"]
DisputeClass = Literal["Disputed", "Refuted"]
# Values of the assessment's MoI enum that a GenCC MoI title can map onto.
GenccMoi = Literal["Monoallelic", "Biallelic", "X-linked", "Mitochondrial"]

DISPUTE_CLASSIFICATIONS: dict[str, DisputeClass] = {
    "Disputed Evidence": "Disputed",
    "Refuted Evidence": "Refuted",
}

# PanelApp Australia's GenCC classes and the panel ratings they correspond to.
# Disputed and Refuted rows have no panel rating.
PAA_CLASS_TO_RATING: dict[PaaClassification, PanelRating] = {
    "Strong": "GREEN",
    "Moderate": "AMBER",
    "Limited": "RED",
}

# GenCC MoI titles that map onto one value of the assessment's MoI enum.
# Semidominant and Unknown have no single counterpart and stay unmapped.
GENCC_MOI_TO_ENUM: dict[str, GenccMoi] = {
    "Autosomal dominant": "Monoallelic",
    "Autosomal recessive": "Biallelic",
    "X-linked": "X-linked",
    "X-linked recessive": "X-linked",
    "X-linked dominant": "X-linked",
    "Mitochondrial": "Mitochondrial",
}


@dataclass(frozen=True)
class PaaAssociation:
    """One PanelApp Australia GenCC row: a classified (gene, disease, MoI) association."""

    mondo_id: str
    disease_title: str
    moi_title: str
    moi: GenccMoi | None  # None when the GenCC MoI title has no single enum value
    classification: PaaClassification
    rating: PanelRating | None  # None for Disputed and Refuted rows
    date: str  # ISO date of the submission
    definition: str  # MONDO definition, empty when the term has none
    obsolete: bool  # the submitted MONDO term is obsolete in the loaded ontology
    replaced_by: tuple[str, ...]  # MONDO's replacements for an obsolete term; often empty


@dataclass(frozen=True)
class DisputeSubmission:
    """One Disputed or Refuted GenCC submission, from any submitter."""

    submitter: str
    mondo_id: str
    disease_title: str
    moi_title: str
    status: DisputeClass
    date: str  # ISO date of the submission
    pmids: tuple[int, ...]  # PMIDs the submitter evaluated
    rationale: str


@dataclass(frozen=True)
class GeneGencc:
    """A gene's PanelApp Australia GenCC rows and its Disputed/Refuted submissions."""

    paa_associations: tuple[PaaAssociation, ...] = ()
    disputes: tuple[DisputeSubmission, ...] = ()


_NO_SUBMISSIONS = GeneGencc()


@dataclass(frozen=True)
class GenccIndex:
    """GenCC records of every gene that has a PanelApp Australia row or a dispute."""

    genes: dict[int, GeneGencc]

    def for_gene(self, hgnc_id: int) -> GeneGencc:
        """The gene's records; empty when GenCC has none of either kind."""
        return self.genes.get(hgnc_id, _NO_SUBMISSIONS)


def _hgnc_id(gene_curie: str) -> int:
    prefix, _, number = gene_curie.partition(":")
    if prefix != "HGNC":
        raise ValueError(f"GenCC gene_curie is not an HGNC id: {gene_curie!r}")
    return int(number)


def _iso_date(raw: str) -> str:
    """The calendar date of a GenCC timestamp, which comes in several ISO forms."""
    return datetime.fromisoformat(raw).date().isoformat()


def _pmids(raw: str) -> tuple[int, ...]:
    return tuple(int(token) for token in raw.split(",") if token.strip())


def _paa_classification(title: str) -> PaaClassification:
    if title not in get_args(PaaClassification):
        raise ValueError(f"unexpected PanelApp Australia GenCC class: {title!r}")
    return cast(PaaClassification, title)


def _definition(term: pronto.Term) -> str:
    return str(term.definition).strip() if term.definition else ""


def load_gencc(gencc_path: Path, mondo: pronto.Ontology) -> GenccIndex:
    """PanelApp Australia's GenCC rows and all Disputed/Refuted submissions, per gene.

    The export is a quoted TSV whose notes contain newlines, so it is read with csv.
    A PanelApp Australia row whose MONDO term is missing from *mondo* raises KeyError.
    A row whose term is obsolete keeps its submitted id; MONDO's replacements, when it
    names any, are recorded next to it.
    Records are sorted by disease, so the order does not depend on the export's.
    """
    paa: defaultdict[int, set[PaaAssociation]] = defaultdict(set)
    disputes: defaultdict[int, list[DisputeSubmission]] = defaultdict(list)

    with gencc_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            submitter = row["submitter_title"]
            dispute_class = DISPUTE_CLASSIFICATIONS.get(row["classification_title"])
            if submitter != PAA_SUBMITTER and dispute_class is None:
                continue
            hgnc_id = _hgnc_id(row["gene_curie"])
            mondo_id = row["disease_curie"]
            date = _iso_date(row["submitted_as_date"])
            if submitter == PAA_SUBMITTER:
                classification = _paa_classification(row["classification_title"])
                term = mondo.get_term(mondo_id)
                paa[hgnc_id].add(
                    PaaAssociation(
                        mondo_id=mondo_id,
                        disease_title=row["disease_title"],
                        moi_title=row["moi_title"],
                        moi=GENCC_MOI_TO_ENUM.get(row["moi_title"]),
                        classification=classification,
                        rating=PAA_CLASS_TO_RATING.get(classification),
                        date=date,
                        definition=_definition(term),
                        obsolete=term.obsolete,
                        replaced_by=tuple(sorted(term.replaced_by.ids)),
                    )
                )
            if dispute_class is not None:
                disputes[hgnc_id].append(
                    DisputeSubmission(
                        submitter=submitter,
                        mondo_id=mondo_id,
                        disease_title=row["disease_title"],
                        moi_title=row["moi_title"],
                        status=dispute_class,
                        date=date,
                        pmids=_pmids(row["submitted_as_pmids"]),
                        rationale=row["submitted_as_notes"].strip(),
                    )
                )

    genes = {
        hgnc_id: GeneGencc(
            paa_associations=tuple(
                sorted(
                    paa.get(hgnc_id, ()), key=lambda a: (a.disease_title, a.moi_title, a.mondo_id)
                )
            ),
            disputes=tuple(
                sorted(
                    disputes.get(hgnc_id, ()),
                    key=lambda d: (d.disease_title, d.moi_title, d.submitter, d.date),
                )
            ),
        )
        for hgnc_id in paa.keys() | disputes.keys()
    }
    logger.info(
        "Loaded GenCC: %d PanelApp Australia rows on %d genes, %d disputed or refuted "
        "submissions on %d genes",
        sum(len(rows) for rows in paa.values()),
        len(paa),
        sum(len(rows) for rows in disputes.values()),
        len(disputes),
    )
    return GenccIndex(genes)


# Caveat for prompts that show a gene's PanelApp Australia GenCC rows as formatted by
# format_paa_associations.
PAA_ASSOCIATIONS_CAVEAT = (
    "The GenCC associations are PanelApp Australia's submissions to GenCC, a projection of"
    " the Mendeliome. They are an early-2025 snapshot and lack curation from about March 2025"
    " onward. Unlike PanelApp's panel entries, they rate each gene-disease-MoI association"
    " separately. Their classes Strong, Moderate and Limited correspond to the PanelApp"
    " ratings GREEN, AMBER and RED; Disputed Evidence and Refuted Evidence mark associations"
    " PanelApp Australia questions or rejects."
)

NO_PAA_ASSOCIATIONS = "PanelApp Australia has no GenCC submissions for this gene."


def format_paa_associations(gene: GeneGencc) -> str:
    """The gene's PanelApp Australia GenCC rows, one ``disease | MoI | class`` line each."""
    if not gene.paa_associations:
        return NO_PAA_ASSOCIATIONS
    return "\n".join(
        f"- {a.disease_title} ({a.mondo_id}) | {a.moi_title} | {a.classification}"
        for a in gene.paa_associations
    )


def fetch_gencc(data_dir: Path) -> GenccIndex:
    """Download the GenCC export and MONDO into *data_dir* when stale, then load them."""
    gencc_path = data_dir / GENCC_FILENAME
    mondo_path = data_dir / MONDO_FILENAME
    download_if_stale(GENCC_URL, gencc_path)
    download_if_stale(MONDO_OBO_URL, mondo_path)
    logger.info("Loading MONDO ontology from %s", mondo_path)
    return load_gencc(gencc_path, pronto.Ontology(str(mondo_path), encoding="utf-8"))
