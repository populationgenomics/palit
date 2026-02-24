"""Shared paper data model and utilities."""

import enum
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


class SkipReason(enum.Enum):
    """Why a PubMed article was not extracted."""

    NO_DOI = "no_doi"
    NO_DATE = "no_date"
    NO_TITLE = "no_title"
    NO_ABSTRACT = "no_abstract"


def doi_to_path(doi: str, base_dir: Path, suffix: str = ".pdf") -> Path:
    """Convert a DOI to a filesystem path.

    The DOI is percent-encoded so special characters (parentheses, angle
    brackets, semicolons, slashes) don't interfere with filesystem paths.

    Examples:
        doi_to_path('10.1038/s41586-020-2308-7', base, '.pdf')
          → base / '10.1038%2Fs41586-020-2308-7.pdf'

        doi_to_path('10.1002/(SICI)1098-1004(200001)15:1<121::AID-HUMU37>3.0.CO;2-U', base, '.pdf')
          → base / '10.1002%2F%28SICI%291098-1004%28200001%2915%3A1%3C121%3A%3AAID-HUMU37%3E3.0.CO%3B2-U.pdf'
    """
    return base_dir / f"{quote(doi, safe='')}{suffix}"


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
    """Extract first author's last name from authors string.

    Authors are semicolon-separated in "Last, First" format:
    "Smith, John A; Doe, Jane B; ..." -> "Smith"
    "van der Berg, Anna; Doe, Jane; ..." -> "VanderBerg"
    """
    if not authors:
        return "Unknown"
    first_author = authors.split(";")[0].strip()
    # Take everything before the comma (the last name portion)
    last_name = first_author.split(",")[0].strip()
    if not last_name:
        return "Unknown"
    parts = last_name.split()
    # Join multi-word last names, capitalize each part, remove non-alpha
    return "".join(re.sub(r"[^A-Za-z]", "", p).capitalize() for p in parts)


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
