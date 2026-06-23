"""Shared paper data model and utilities."""

import enum
import json
import re
from dataclasses import asdict, dataclass
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
class PubmedMetadata:
    """Source metadata for PubMed papers."""

    pmcid: str | None = None


@dataclass
class RxivMetadata:
    """Source metadata for bioRxiv/medRxiv papers."""

    version: int
    category: str
    license: str | None = None
    jatsxml_url: str | None = None
    published_doi: str | None = None


@dataclass
class ResearchSquareMetadata:
    """Source metadata for Research Square papers."""

    version: int
    versioned_doi: str


@dataclass
class CrossrefMetadata:
    """Source metadata for papers fetched directly from CrossRef."""


SourceMetadata = PubmedMetadata | RxivMetadata | ResearchSquareMetadata | CrossrefMetadata


def serialize_source_metadata(metadata: SourceMetadata) -> str:
    """Serialize source metadata to JSON string for DB storage."""
    return json.dumps({k: v for k, v in asdict(metadata).items() if v is not None})


def deserialize_source_metadata(source: str, json_str: str | None) -> SourceMetadata:
    """Deserialize source metadata from DB JSON string."""
    data = json.loads(json_str) if json_str else {}
    if source in ("biorxiv", "medrxiv"):
        return RxivMetadata(**data)
    elif source == "researchsquare":
        return ResearchSquareMetadata(**data)
    elif source == "pubmed":
        return PubmedMetadata(**data)
    elif source == "crossref":
        return CrossrefMetadata(**data)
    else:
        raise ValueError(f"Unknown source: {source}")


# CrossRef response parsing utilities


def strip_xml_tags(text: str) -> str:
    """Strip XML/HTML tags from CrossRef abstract text and collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


def parse_crossref_date(date_obj: dict[str, Any]) -> str:
    """Parse CrossRef date-parts format to YYYY-MM-DD string."""
    parts: list[int] = date_obj["date-parts"][0]
    year = parts[0]
    month = parts[1] if len(parts) > 1 else 1
    day = parts[2] if len(parts) > 2 else 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def format_crossref_authors(authors: list[dict[str, Any]]) -> str:
    """Convert CrossRef author objects to 'Last, First; Last, First' format."""
    formatted: list[str] = []
    for author in authors:
        family = author.get("family")
        given = author.get("given")
        if family and given:
            formatted.append(f"{family}, {given}")
        elif family:
            formatted.append(family)
        elif name := author.get("name"):
            formatted.append(name)
    return "; ".join(formatted)


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
    source_metadata: SourceMetadata
    source_type: str
    source_details: str


# Preprint quality gate: minimum unrelated families required
MIN_PREPRINT_FAMILIES = 5

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


def build_display_ids(
    paper_id_to_doi: dict[str, str],
    doi_to_pmid: dict[str, int],
) -> dict[str, str]:
    """Map AuthorYear paper IDs to human display format.

    Papers with a PMID become "PMID {pmid}".
    Preprints (no PMID) keep their AuthorYear ID.
    """
    display_ids: dict[str, str] = {}
    for paper_id, doi in paper_id_to_doi.items():
        pmid = doi_to_pmid.get(doi)
        if pmid is not None:
            display_ids[paper_id] = f"PMID {pmid}"
        else:
            display_ids[paper_id] = paper_id
    return display_ids


def replace_paper_ids_for_display(
    data: dict[str, Any],
    display_ids: dict[str, str],
) -> dict[str, Any]:
    """Replace AuthorYear paper IDs with display format in a JSON-serializable dict.

    Serializes to JSON, regex-replaces all AuthorYear occurrences, deserializes back.
    Keys are sorted by length descending so "Smith2024a" matches before "Smith2024".
    """
    ids_to_replace = [(pid, did) for pid, did in display_ids.items() if pid != did]
    if not ids_to_replace:
        return data
    # Sort by length descending so "Smith2024a" matches before "Smith2024" in the regex
    ids_to_replace.sort(key=lambda x: len(x[0]), reverse=True)
    replacements = dict(ids_to_replace)  # preserves sort order (Python 3.7+)
    pattern = re.compile("|".join(rf"\b{re.escape(pid)}\b" for pid in replacements))
    text = json.dumps(data)
    text = pattern.sub(lambda m: replacements[m.group()], text)
    result: dict[str, Any] = json.loads(text)
    return result
