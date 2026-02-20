"""Shared paper data model and utilities."""

import enum
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SkipReason(enum.Enum):
    """Why a PubMed article was not extracted."""

    NO_DOI = "no_doi"
    NO_DATE = "no_date"
    NO_TITLE = "no_title"
    NO_ABSTRACT = "no_abstract"


def doi_to_path(doi: str, base_dir: Path, suffix: str = ".pdf") -> Path:
    """Convert a DOI to a filesystem path.

    The single slash in a DOI (e.g. '10.1234/xyz') creates natural two-level nesting:
      base_dir/10.1234/xyz.pdf
    """
    return base_dir / f"{doi}{suffix}"


@dataclass
class Paper:
    """Paper metadata extracted from a bibliographic source."""

    doi: str
    pmid: int | None
    title: str
    abstract: str
    authors: str
    journal: str
    source: str
    source_date: str
    source_metadata: dict[str, object]
    source_type: str
    source_details: str


# Preprint detection

PREPRINT_SERVERS = {
    "biorxiv",
    "medrxiv",
    "arxiv",
    "ssrn",
    "research square",
    "preprints",
    "chemrxiv",
    "eartharxiv",
    "psyarxiv",
    "socarxiv",
    "osf preprints",
}


def is_preprint(journal: str | None, pmid: int | None) -> bool:
    """Detect whether a paper is a preprint.

    A paper is considered a preprint if its journal matches a known preprint server
    or if it has no PMID (peer-reviewed papers indexed in PubMed always have one).
    """
    if pmid is None:
        return True
    if journal:
        journal_lower = journal.lower()
        return any(server in journal_lower for server in PREPRINT_SERVERS)
    return False


# Paper ID generation ({LastName}{Year} format)


def _extract_first_author_last_name(authors: str) -> str:
    """Extract first author's last name from PubMed-style authors string.

    Handles formats like "Smith JA, Doe B, ..." -> "Smith"
    and "van der Berg A, Doe B, ..." -> "vanderBerg".
    """
    if not authors:
        return "Unknown"
    first_author = authors.split(",")[0].strip()
    parts = first_author.split()
    # Collect name parts before the initials (single uppercase letters)
    name_parts = []
    for part in parts:
        if len(part) <= 2 and part.isupper():
            break
        name_parts.append(part)
    if not name_parts:
        return "Unknown"
    # Join multi-word last names, capitalize each part, remove non-alpha
    return "".join(re.sub(r"[^A-Za-z]", "", p).capitalize() for p in name_parts)


def generate_paper_ids(
    evidence_list: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str]]:
    """Generate {LastName}{Year} paper IDs from evidence list.

    Each item in evidence_list must have 'doi', 'authors', and 'date' keys.

    Returns:
        Tuple of (paper_id_to_doi, doi_to_paper_id) mappings
    """
    # Group DOIs by base ID
    base_id_to_dois: dict[str, list[str]] = {}
    doi_to_base: dict[str, str] = {}

    for evidence in evidence_list:
        doi = evidence["doi"]
        authors = evidence.get("authors", "")
        date = evidence.get("date", "")
        last_name = _extract_first_author_last_name(authors)
        year = date[:4] if date and len(date) >= 4 else "Unknown"
        base_id = f"{last_name}{year}"
        base_id_to_dois.setdefault(base_id, []).append(doi)
        doi_to_base[doi] = base_id

    # Assign final IDs, disambiguating with letter suffixes when needed
    paper_id_to_doi: dict[str, str] = {}
    doi_to_paper_id: dict[str, str] = {}
    for base_id, dois in base_id_to_dois.items():
        if len(dois) == 1:
            paper_id_to_doi[base_id] = dois[0]
            doi_to_paper_id[dois[0]] = base_id
        else:
            for i, doi in enumerate(sorted(dois)):
                suffixed_id = f"{base_id}{chr(ord('a') + i)}"
                paper_id_to_doi[suffixed_id] = doi
                doi_to_paper_id[doi] = suffixed_id

    return paper_id_to_doi, doi_to_paper_id
