"""GenCC disease terms and disputes, keyed by HGNC ID.

GenCC submissions name each gene's diseases by MONDO term. The terms of every
submitter are offered to the prompts as suggestions for naming and for lumping
or splitting associations; their classes are not shown, because PanelApp
Australia's own submissions are an early-2025 export, and current PanelApp is the
record of what it curates and at what rating. Disputed and Refuted submissions are
kept from every submitter, because they gate the criteria of the associations they
apply to.

Genes are matched on ``gene_curie`` (``HGNC:<id>``), never on the symbol. The
module also downloads the GenCC export and MONDO when they are stale. MONDO is
fetched separately, so a stage that also needs the ontology itself parses it once.
"""

import csv
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx2
import pronto

logger = logging.getLogger(__name__)

GENCC_URL = "https://search.thegencc.org/download/action/submissions-export-tsv"
MONDO_OBO_URL = "https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo"
GENCC_FILENAME = "gencc_submissions.tsv"
MONDO_FILENAME = "mondo.obo"
MAX_DOWNLOAD_AGE_SECONDS = 7 * 24 * 3600

DisputeClass = Literal["Disputed", "Refuted"]

DISPUTE_CLASSIFICATIONS: dict[str, DisputeClass] = {
    "Disputed Evidence": "Disputed",
    "Refuted Evidence": "Refuted",
}


@dataclass(frozen=True)
class MondoRef:
    """A MONDO term by id and label."""

    mondo_id: str
    label: str


@dataclass(frozen=True)
class GenccTerm:
    """A MONDO disease term that GenCC submitters use for a gene, over all submitters."""

    mondo_id: str
    disease_title: str
    moi_titles: tuple[str, ...]  # the GenCC MoI titles submitted with this term, sorted
    definition: str  # MONDO definition, empty when the term has none
    obsolete: bool  # the term is obsolete in the loaded ontology
    replaced_by: tuple[MondoRef, ...]  # MONDO's replacements for an obsolete term; often empty


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
    """A gene's GenCC disease terms and its Disputed/Refuted submissions."""

    terms: tuple[GenccTerm, ...] = ()
    disputes: tuple[DisputeSubmission, ...] = ()


_NO_SUBMISSIONS = GeneGencc()


@dataclass(frozen=True)
class GenccIndex:
    """GenCC records of every gene with a GenCC submission."""

    genes: dict[int, GeneGencc]

    def for_gene(self, hgnc_id: int) -> GeneGencc:
        """The gene's records; empty when GenCC has no submission for it."""
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


def _definition(term: pronto.Term) -> str:
    return str(term.definition).strip() if term.definition else ""


def _replacements(term: pronto.Term) -> tuple[MondoRef, ...]:
    """MONDO's replacements for *term*, by id, with their labels."""
    refs = []
    for replacement in sorted(term.replaced_by, key=lambda t: t.id):
        if replacement.name is None:
            raise ValueError(f"MONDO term {replacement.id} has no label")
        refs.append(MondoRef(mondo_id=replacement.id, label=replacement.name))
    return tuple(refs)


def load_gencc(gencc_path: Path, mondo: pronto.Ontology) -> GenccIndex:
    """Every gene's GenCC disease terms and all Disputed/Refuted submissions.

    The export is a quoted TSV whose notes contain newlines, so it is read with csv.
    A term missing from *mondo* raises KeyError. An obsolete term keeps its
    submitted id; MONDO's replacements, when it names any, are recorded next to it
    with their labels. Terms are sorted by title and disputes by disease, so the
    order does not depend on the export's.
    """
    titles: dict[tuple[int, str], str] = {}
    mois: defaultdict[tuple[int, str], set[str]] = defaultdict(set)
    disputes: defaultdict[int, list[DisputeSubmission]] = defaultdict(list)

    with gencc_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            hgnc_id = _hgnc_id(row["gene_curie"])
            mondo_id = row["disease_curie"]
            titles[(hgnc_id, mondo_id)] = row["disease_title"]
            mois[(hgnc_id, mondo_id)].add(row["moi_title"])
            dispute_class = DISPUTE_CLASSIFICATIONS.get(row["classification_title"])
            if dispute_class is not None:
                disputes[hgnc_id].append(
                    DisputeSubmission(
                        submitter=row["submitter_title"],
                        mondo_id=mondo_id,
                        disease_title=row["disease_title"],
                        moi_title=row["moi_title"],
                        status=dispute_class,
                        date=_iso_date(row["submitted_as_date"]),
                        pmids=_pmids(row["submitted_as_pmids"]),
                        rationale=row["submitted_as_notes"].strip(),
                    )
                )

    terms: defaultdict[int, list[GenccTerm]] = defaultdict(list)
    for (hgnc_id, mondo_id), title in titles.items():
        term = mondo.get_term(mondo_id)
        terms[hgnc_id].append(
            GenccTerm(
                mondo_id=mondo_id,
                disease_title=title,
                moi_titles=tuple(sorted(mois[(hgnc_id, mondo_id)])),
                definition=_definition(term),
                obsolete=term.obsolete,
                replaced_by=_replacements(term),
            )
        )

    genes = {
        hgnc_id: GeneGencc(
            terms=tuple(sorted(gene_terms, key=lambda t: (t.disease_title, t.mondo_id))),
            disputes=tuple(
                sorted(
                    disputes.get(hgnc_id, ()),
                    key=lambda d: (d.disease_title, d.moi_title, d.submitter, d.date),
                )
            ),
        )
        for hgnc_id, gene_terms in terms.items()
    }
    logger.info(
        "Loaded GenCC: %d disease terms on %d genes, %d disputed or refuted submissions "
        "on %d genes",
        sum(len(gene_terms) for gene_terms in terms.values()),
        len(terms),
        sum(len(rows) for rows in disputes.values()),
        len(disputes),
    )
    return GenccIndex(genes)


def format_gencc_terms(gene: GeneGencc) -> str:
    """The gene's GenCC disease terms, one ``title (MONDO id) | MoIs`` line each."""
    if not gene.terms:
        return NO_GENCC_TERMS
    return "\n".join(
        f"- {t.disease_title} ({t.mondo_id}) | {', '.join(t.moi_titles)}" for t in gene.terms
    )


NO_GENCC_TERMS = "GenCC has no submissions for this gene."


def download_if_stale(url: str, path: Path) -> None:
    """Download *url* to *path* unless *path* exists and is less than a week old."""
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < MAX_DOWNLOAD_AGE_SECONDS:
            logger.info("Using cached %s (age: %.0fh)", path, age / 3600)
            return
        logger.info("Re-downloading stale %s (age: %.0fh)", path, age / 3600)
    else:
        logger.info("Downloading %s", url)

    with httpx2.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(url)
        response.raise_for_status()
        path.write_bytes(response.content)
    logger.info("Downloaded %s bytes to %s", f"{len(response.content):,}", path)


def fetch_mondo(data_dir: Path) -> pronto.Ontology:
    """MONDO from *data_dir*, downloaded first when missing or stale."""
    mondo_path = data_dir / MONDO_FILENAME
    download_if_stale(MONDO_OBO_URL, mondo_path)
    logger.info("Loading MONDO ontology from %s", mondo_path)
    return pronto.Ontology(str(mondo_path), encoding="utf-8")


def fetch_gencc(data_dir: Path, mondo: pronto.Ontology) -> GenccIndex:
    """The GenCC export from *data_dir*, downloaded first when missing or stale.

    *mondo* is the ontology from :func:`fetch_mondo`, so a stage that also needs
    MONDO itself parses it once.
    """
    gencc_path = data_dir / GENCC_FILENAME
    download_if_stale(GENCC_URL, gencc_path)
    return load_gencc(gencc_path, mondo)
