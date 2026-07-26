#!/usr/bin/env python3
"""Parse PubMed XML (`PubmedArticleSet`) into `Paper` records.

The same `PubmedArticleSet` DTD is served by the live efetch API, the FTP
baseline, and the FTP update files, so this module is the single XML→`Paper`
extraction path shared by the ingest command, the FTP-updatefile ledger sync,
the baseline screener, and citation discovery.
"""

import io
import logging
from dataclasses import dataclass

from lxml import etree

from palit.papers import Paper, PubmedMetadata, SkipReason

logger = logging.getLogger(__name__)


def extract_text_content(element: etree._Element | None) -> str:
    """Extract all text content from an XML element, including nested elements."""
    if element is None:
        return ""

    # Get all text content, including from nested elements
    text_parts = []
    if element.text:
        text_parts.append(element.text)

    for child in element:
        text_parts.append(extract_text_content(child))
        if child.tail:
            text_parts.append(child.tail)

    return "".join(text_parts).strip()


def extract_authors(article_elem: etree._Element) -> str:
    """Extract authors from AuthorList element."""
    author_list = article_elem.find(".//MedlineCitation/Article/AuthorList")
    if author_list is None:
        return ""

    authors = []
    for author in author_list.findall("Author"):
        last_name_elem = author.find("LastName")
        first_name_elem = author.find("ForeName")

        if last_name_elem is not None and last_name_elem.text:
            if first_name_elem is not None and first_name_elem.text:
                authors.append(f"{last_name_elem.text}, {first_name_elem.text}")
            else:
                authors.append(last_name_elem.text)

    return "; ".join(authors)


def extract_journal(article_elem: etree._Element) -> str:
    """Extract journal title from Journal element."""
    journal_elem = article_elem.find(".//MedlineCitation/Article/Journal/Title")
    if journal_elem is not None and journal_elem.text is not None:
        return str(journal_elem.text).strip()
    return ""


def extract_languages(article_elem: etree._Element) -> list[str]:
    """Extract the article's language codes from Language elements.

    An article carries one Language per language it is published in, as a
    three-letter code ('eng', 'chi', 'ger'). Bilingual articles list several
    (e.g. 'eng' + 'spa'). The element is absent from some unindexed records.
    """
    return [
        elem.text.strip()
        for elem in article_elem.findall(".//MedlineCitation/Article/Language")
        if elem.text
    ]


def extract_date(date_element: etree._Element | None) -> str | None:
    """Extract date from PubMedPubDate element."""
    if date_element is None:
        return None

    year_elem = date_element.find("Year")
    month_elem = date_element.find("Month")
    day_elem = date_element.find("Day")

    if year_elem is None or month_elem is None or day_elem is None:
        return None

    try:
        year = int(year_elem.text)
        month = int(month_elem.text)
        day = int(day_elem.text)
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, TypeError):
        return None


def extract_paper(
    article_elem: etree._Element,
    source_type: str,
    source_details: str,
    require_abstract: bool = True,
) -> Paper | SkipReason:
    """Extract paper data from PubmedArticle element.

    Articles published only in a language other than English are skipped.

    Args:
        article_elem: The XML element containing the paper data
        source_type: Type of source (e.g., "initial", "expansion")
        source_details: Details about the source
        require_abstract: If True, skip papers without abstracts. Default True.
    """
    # Extract article IDs (DOI, PMID, PMCID) from ArticleIdList
    article_ids: dict[str, str] = {}
    for article_id in article_elem.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = article_id.get("IdType")
        if id_type and article_id.text:
            article_ids[id_type] = article_id.text.strip()

    doi = article_ids.get("doi")
    if not doi:
        return SkipReason.NO_DOI

    # Extract entrez date from PubmedData/History/PubMedPubDate[@PubStatus="entrez"]
    entrez_date_elem = article_elem.find('.//PubmedData/History/PubMedPubDate[@PubStatus="entrez"]')
    source_date = extract_date(entrez_date_elem)
    if not source_date:
        return SkipReason.NO_DATE

    # Extract title from MedlineCitation/Article/ArticleTitle
    title_elem = article_elem.find(".//MedlineCitation/Article/ArticleTitle")
    title = extract_text_content(title_elem)
    if not title:
        return SkipReason.NO_TITLE

    # Extract abstract from MedlineCitation/Article/Abstract/AbstractText elements
    abstract_elems = article_elem.findall(".//MedlineCitation/Article/Abstract/AbstractText")
    abstract_parts = [extract_text_content(elem) for elem in abstract_elems]
    abstract = " ".join(abstract_parts).strip()

    if require_abstract and not abstract:
        return SkipReason.NO_ABSTRACT

    # Non-English articles are out of scope: their full text is unavailable through
    # the PMC OA path, so a relevant one only ever reaches manual retrieval. Checked
    # after the gates above so the NON_ENGLISH tally counts exactly what this filter
    # costs. A record with no Language element is kept.
    languages = extract_languages(article_elem)
    if languages and "eng" not in languages:
        return SkipReason.NON_ENGLISH

    # Extract authors and journal
    authors = extract_authors(article_elem)
    journal = extract_journal(article_elem)

    pmid_str = article_ids.get("pubmed")
    pmid = int(pmid_str) if pmid_str else None

    return Paper(
        doi=doi,
        pmid=pmid,
        title=title,
        abstract=abstract,
        authors=authors,
        journal=journal,
        source="pubmed",
        source_date=source_date,
        source_metadata=PubmedMetadata(pmcid=article_ids.get("pmc")),
        source_type=source_type,
        source_details=source_details,
    )


@dataclass
class ExtractionStats:
    """Statistics from paper extraction."""

    total_articles: int
    extracted: int
    skipped: dict[SkipReason, int]


def extract_papers_from_xml(
    xml_content: bytes,
    source_type: str,
    source_details: str,
    require_abstract: bool = True,
    min_year: int | None = None,
) -> tuple[list[Paper], ExtractionStats]:
    """Extract papers from XML bytes content.

    Args:
        xml_content: XML content as bytes
        source_type: Type of source (e.g., "initial", "expansion")
        source_details: Details about the source (e.g., filename, gene name)
        require_abstract: If True, skip papers without abstracts. Default True.
        min_year: If provided, only include papers with source_date >= this year

    Returns:
        Tuple of (list of Paper objects, extraction statistics)
    """
    # Parse XML
    parser = etree.XMLParser(recover=True, resolve_entities=False)
    tree = etree.parse(io.BytesIO(xml_content), parser)
    root = tree.getroot()
    if root is None:
        return [], ExtractionStats(total_articles=0, extracted=0, skipped={})

    # Find all PubmedArticle elements
    papers = []
    skipped: dict[SkipReason, int] = {}
    article_elements = root.findall(".//PubmedArticle")

    for article_elem in article_elements:
        result = extract_paper(article_elem, source_type, source_details, require_abstract)
        if isinstance(result, SkipReason):
            skipped[result] = skipped.get(result, 0) + 1
            continue

        # Filter by year if min_year is specified
        if min_year is not None:
            paper_year = int(result.source_date[:4])
            if paper_year < min_year:
                continue

        papers.append(result)

    stats = ExtractionStats(
        total_articles=len(article_elements),
        extracted=len(papers),
        skipped=skipped,
    )
    if skipped:
        skip_summary = ", ".join(
            f"{r.value}: {n}" for r, n in sorted(skipped.items(), key=lambda x: -x[1])
        )
        logger.info(
            f"Skipped {sum(skipped.values())}/{len(article_elements)} articles "
            f"from {source_details} ({skip_summary})"
        )

    return papers, stats
