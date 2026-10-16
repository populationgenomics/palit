"""Tests for reading PMIDs and DOIs out of PanelApp's free-text publication fields.

Every example is a publication string from the PanelApp Australia snapshot of
2026-10-13.
"""

import pytest

from palit.panelapp_client import clean_panel_publications, ordered_panel_publications


@pytest.mark.parametrize(
    ("publication", "pmids"),
    [
        ("29531722", {29531722}),
        ("PMID: 29531722", {29531722}),
        ("PMID:12525535", {12525535}),
        ("PMID:\xa016917943, PMID:\xa023919265", {16917943, 23919265}),
        ("(PMID: 26401682): (PMID: 17603483)", {26401682, 17603483}),
        ("PMID: 26303607PMID: 24878975", {26303607, 24878975}),
        ("PubMed: 11781871, 12325024,,", {11781871, 12325024}),
        ("PMID: 33272083, 2918525,  20453517", {33272083, 2918525, 20453517}),
        ("33510603: 33976153", {33510603, 33976153}),
        ("11017805.", {11017805}),
        ("16141009)", {16141009}),
        ("[7795645", {7795645}),
        ("​14559215", {14559215}),
        ("16060907 (Camilot et al., 2005 report subclinical hypothyroid subjects", {16060907}),
        ("PMID: 22127393 (review paper): PubMed: 20077426 (source paper)", {22127393, 20077426}),
        ("https://www.ncbi.nlm.nih.gov/pubmed/16794080", {16794080}),
        ("https://pubmed.ncbi.nlm.nih.gov/16794080/", {16794080}),
        ("PubMed ID: 16794080", {16794080}),
        # A paper from the late 1970s, cited by its six-digit PMID.
        ("469896", {469896}),
    ],
)
def test_reads_bare_and_labelled_pmids(publication: str, pmids: set[int]) -> None:
    assert clean_panel_publications([publication]).pmids == pmids


@pytest.mark.parametrize(
    "publication",
    [
        # Phenotypes and OMIM numbers
        "Ehlers-Danlos syndrome, spondylodysplastic type, 1, 130070",
        "Epileptic encephalopathy, early infantile, 54, MIM#617391",
        "OMIM 120130",
        "MIM # 203800",
        "Mucopolysaccharidosis type IX, MIM# 601492",
        "Mental retardation, X-linked syndromic, Raymond typeMIM# 300799",
        # Other identifiers
        "bioRxiv 773028",
        "NBK131811",
        "Corpus ID: 24567050",
        "ClinVar: SCV000965687.1",
        "https://search.clinicalgenome.org/CCID:004015",
        # Years, volumes and pages, and PMIDs cut short (PMIDs of 1 to 4 digits exist,
        # but every such number labelled as a PMID in the snapshot is a truncation)
        "and online publication GiM Feb 2024",
        "Patel et al. 2012 Mol Genet Metab 105(1):34-43 (PMID: 2207",
        "0980531",
        # Longer than any PMID
        "251920460",
        "2097801822775483",
    ],
)
def test_ignores_numbers_that_are_not_pmids(publication: str) -> None:
    assert clean_panel_publications([publication]).pmids == set()


@pytest.mark.parametrize(
    ("publication", "doi"),
    [
        ("10.1101/2025.01.06.631535", "10.1101/2025.01.06.631535"),
        ("doi:10.1016/j.bdcasr.2024.100043", "10.1016/j.bdcasr.2024.100043"),
        ("doi.org/10.1038/gim.2017.218", "10.1038/gim.2017.218"),
        ("http://dx.doi.org/10.1136/jmedgenet-2021-107843", "10.1136/jmedgenet-2021-107843"),
        ("https://doi.org/10.1016/j.ymgme.2024.108362.", "10.1016/j.ymgme.2024.108362"),
        ("https://doi.org/10.1016/S2096-6911(21)00044-3", "10.1016/S2096-6911(21)00044-3"),
        (
            "https://www.neurology.org/doi/full/10.1212/NXG.0000000000200101 (No PMID)",
            "10.1212/NXG.0000000000200101",
        ),
        (
            "https://www.biorxiv.org/content/10.1101/2021.04.27.441137v1.full.pdf",
            "10.1101/2021.04.27.441137",
        ),
        (
            "https://www.medrxiv.org/content/10.1101/2022.04.04.22273309v1",
            "10.1101/2022.04.04.22273309",
        ),
        # Research Square's DOIs carry the version themselves.
        ("10.21203/rs.3.rs-9516499/v1", "10.21203/rs.3.rs-9516499/v1"),
    ],
)
def test_reads_a_doi_wherever_it_appears(publication: str, doi: str) -> None:
    cited = clean_panel_publications([publication])
    assert cited.dois == {doi}
    assert cited.pmids == set()


def test_reads_a_pmid_and_a_doi_from_one_string() -> None:
    cited = clean_panel_publications(
        ["PMID: 38022338, 36944331, https://doi.org/10.1016/j.jaci.2022.10.022"]
    )
    assert cited.pmids == {38022338, 36944331}
    assert cited.dois == {"10.1016/j.jaci.2022.10.022"}


def test_reads_a_pmid_after_a_doi_url() -> None:
    cited = clean_panel_publications(
        ["PMID: 38022338, https://doi.org/10.1016/j.jaci.2022.10.022, 36944331"]
    )
    assert cited.pmids == {38022338, 36944331}
    assert cited.dois == {"10.1016/j.jaci.2022.10.022"}


def test_does_not_read_a_dois_digits_as_a_pmid() -> None:
    cited = clean_panel_publications(
        [
            "bioRxiv 773028 doi: https://doi.org/10.1101/773028",
            "10.64898/2026.02.18.706528",
            "DOI:10.1002/mds.28915 Corpus ID: 245670502",
        ]
    )
    assert cited.pmids == set()
    assert cited.dois == {"10.1101/773028", "10.64898/2026.02.18.706528", "10.1002/mds.28915"}


def test_ordered_publications_keep_the_strings_order() -> None:
    """Each string's PMIDs, then its DOIs, each sorted; a repeat in a later string stays."""
    assert ordered_panel_publications(
        [
            "PMID: 38022338, 36944331, https://doi.org/10.1016/j.jaci.2022.10.022",
            "OMIM 601492",
            "10.1101/773028",
            "36944331",
        ]
    ) == [36944331, 38022338, "10.1016/j.jaci.2022.10.022", "10.1101/773028", 36944331]
