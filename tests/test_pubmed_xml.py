"""Tests for PubMed XML extraction: skip gates and their tallies."""

from palit.papers import SkipReason
from palit.pubmed_xml import extract_papers_from_xml

# --- helpers ----------------------------------------------------------------


def _article(
    doi: str | None = "10.1000/a",
    *,
    languages: list[str] | None = None,
    title: str = "A title",
    abstract: str | None = "An abstract",
    day: str = "2026-01-15",
) -> str:
    """Build one PubmedArticle element; `languages=None` omits Language entirely."""
    y, m, d = day.split("-")
    lang_xml = "".join(f"<Language>{code}</Language>" for code in languages or [])
    abstract_xml = (
        f"<Abstract><AbstractText>{abstract}</AbstractText></Abstract>" if abstract else ""
    )
    doi_xml = f'<ArticleId IdType="doi">{doi}</ArticleId>' if doi else ""
    return f"""
        <PubmedArticle>
          <MedlineCitation>
            <Article>
              <ArticleTitle>{title}</ArticleTitle>
              {abstract_xml}
              <AuthorList><Author><LastName>Smith</LastName>
                <ForeName>Jane</ForeName></Author></AuthorList>
              <Journal><Title>Test Journal</Title></Journal>
              {lang_xml}
            </Article>
          </MedlineCitation>
          <PubmedData>
            <History><PubMedPubDate PubStatus="entrez">
              <Year>{y}</Year><Month>{m}</Month><Day>{d}</Day>
            </PubMedPubDate></History>
            <ArticleIdList>
              <ArticleId IdType="pubmed">12345</ArticleId>
              {doi_xml}
            </ArticleIdList>
          </PubmedData>
        </PubmedArticle>
        """


def _articleset(*articles: str) -> bytes:
    return f"<PubmedArticleSet>{''.join(articles)}</PubmedArticleSet>".encode()


def _extract(*articles: str) -> tuple[list[str], dict[SkipReason, int]]:
    """Extract and return (kept DOIs, skip tally)."""
    papers, stats = extract_papers_from_xml(_articleset(*articles), "initial", "test")
    return [p.doi for p in papers], stats.skipped


# --- language gate ----------------------------------------------------------


def test_english_article_is_kept() -> None:
    dois, skipped = _extract(_article("10.1000/eng", languages=["eng"]))
    assert dois == ["10.1000/eng"]
    assert skipped == {}


def test_non_english_articles_are_skipped() -> None:
    dois, skipped = _extract(
        _article("10.1000/chi", languages=["chi"]),
        _article("10.1000/ger", languages=["ger"]),
        _article("10.1000/rus", languages=["rus"]),
    )
    assert dois == []
    assert skipped == {SkipReason.NON_ENGLISH: 3}


def test_bilingual_article_including_english_is_kept() -> None:
    """'eng,spa' records carry English text, so they are in scope."""
    dois, skipped = _extract(_article("10.1000/bi", languages=["eng", "spa"]))
    assert dois == ["10.1000/bi"]
    assert skipped == {}


def test_missing_language_element_is_kept() -> None:
    """Fail open: unindexed records can lack Language and must not be dropped."""
    dois, skipped = _extract(_article("10.1000/nolang", languages=None))
    assert dois == ["10.1000/nolang"]
    assert skipped == {}


def test_language_gate_runs_after_the_earlier_gates() -> None:
    """A non-English article that also fails an earlier gate keeps that gate's reason.

    Ordering keeps the NON_ENGLISH tally a true count of what this filter costs:
    records already dropped for a missing DOI or abstract are not reattributed to it.
    """
    _, skipped = _extract(
        _article(doi=None, languages=["chi"]),
        _article("10.1000/noabs", languages=["ger"], abstract=None),
    )
    assert skipped == {SkipReason.NO_DOI: 1, SkipReason.NO_ABSTRACT: 1}


def test_mixed_set_keeps_only_english() -> None:
    dois, skipped = _extract(
        _article("10.1000/a", languages=["eng"]),
        _article("10.1000/b", languages=["jpn"]),
        _article("10.1000/c", languages=None),
        _article("10.1000/d", languages=["fre"]),
    )
    assert dois == ["10.1000/a", "10.1000/c"]
    assert skipped == {SkipReason.NON_ENGLISH: 2}
