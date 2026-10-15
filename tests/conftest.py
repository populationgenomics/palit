"""Shared fixtures: a small PanelApp snapshot, its genes' entries, GenCC index, HGNC
resolver, and a builder of one-page PDFs with a text layer."""

from collections.abc import Callable
from typing import Any

import pytest

from palit.gencc import GenccIndex, GenccTerm, GeneGencc
from palit.hgnc import HgncEntry, HgncResolver
from palit.panel_entries import PanelEntry, entries_by_gene
from palit.panelapp_integration import INCIDENTALOME_PANEL_ID, MENDELIOME_PANEL_ID

OTHER_PANEL_ID = 9001
REFERENCE_PANEL_IDS = [MENDELIOME_PANEL_ID, INCIDENTALOME_PANEL_ID]


def _entity(hgnc_id: int | None, symbol: str, **fields: Any) -> dict[str, Any]:
    gene_data = (
        {"gene_symbol": symbol}
        if hgnc_id is None
        else {"hgnc_id": f"HGNC:{hgnc_id}", "gene_symbol": symbol}
    )
    return {
        "gene_data": gene_data,
        "entity_name": symbol,
        "confidence_level": fields.get("confidence_level", "3"),
        "mode_of_inheritance": fields.get("moi", "BIALLELIC, autosomal or pseudoautosomal"),
        "phenotypes": fields.get("phenotypes", ["Some disease MIM#100000"]),
        "publications": fields.get("publications", []),
        "evaluations": fields.get("evaluations", [{"rating": "GREEN", "comments": []}]),
    }


def _panel(
    panel_id: int, name: str, genes: list[dict[str, Any]], strs: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "id": panel_id,
        "name": name,
        "genes": genes,
        "strs": strs or [],
        "evaluations_fetched_at": "2026-09-01T09:00:00+00:00",
    }


@pytest.fixture
def panelapp_snapshot() -> dict[int, dict[str, Any]]:
    """Mendeliome with a gene and a repeat expansion, an Incidentalome gene, and that
    gene's entry on another panel next to an entity without an HGNC ID."""
    return {
        MENDELIOME_PANEL_ID: _panel(
            MENDELIOME_PANEL_ID,
            "Mendeliome",
            [_entity(1, "GENEA", publications=["12345", "10.1000/Cited"])],
            strs=[
                _entity(
                    2, "REPEAT1", moi="MONOALLELIC, autosomal or pseudoautosomal, NOT imprinted"
                )
            ],
        ),
        INCIDENTALOME_PANEL_ID: _panel(
            INCIDENTALOME_PANEL_ID, "Incidentalome", [_entity(3, "GENEB", phenotypes=[])]
        ),
        OTHER_PANEL_ID: _panel(
            OTHER_PANEL_ID,
            "Ataxia",
            [
                _entity(3, "GENEB", phenotypes=["Paediatric disease MIM#200000"]),
                _entity(None, "NOID"),
            ],
        ),
    }


@pytest.fixture
def hgnc_resolver() -> HgncResolver:
    entries = {
        symbol: HgncEntry(
            hgnc_id=hgnc_id,
            symbol=symbol,
            name=f"{symbol.lower()} protein",
            prev_symbols=(),
            alias_symbols=(),
            locus_group="protein-coding gene",
            location="1p1",
            chromosome="1",
        )
        for symbol, hgnc_id in (("GENEA", 1), ("REPEAT1", 2), ("GENEB", 3), ("GENEC", 4))
    }
    return HgncResolver(
        _by_symbol=entries,
        _by_prev={},
        _by_alias={},
        _by_hgnc_id={e.hgnc_id: e for e in entries.values()},
    )


@pytest.fixture
def gencc_index() -> GenccIndex:
    """GENEA (HGNC:1) has two GenCC disease terms; no other gene has any."""
    disease_a = GenccTerm(
        mondo_id="MONDO:0000001",
        disease_title="disease A",
        moi_titles=("Autosomal recessive",),
        definition="",
        obsolete=False,
        replaced_by=(),
    )
    disease_b = GenccTerm(
        mondo_id="MONDO:0000002",
        disease_title="disease B",
        moi_titles=("Autosomal dominant", "Autosomal recessive"),
        definition="",
        obsolete=False,
        replaced_by=(),
    )
    return GenccIndex({1: GeneGencc(terms=(disease_a, disease_b))})


@pytest.fixture
def panel_entries(
    panelapp_snapshot: dict[int, dict[str, Any]],
) -> dict[int, tuple[PanelEntry, ...]]:
    """Each gene's entries on every panel; Mendeliome and Incidentalome are the reference panels."""
    return entries_by_gene(list(panelapp_snapshot.values()), REFERENCE_PANEL_IDS)


def _text_pdf(lines: list[str]) -> bytes:
    """A one-page PDF with one line of Helvetica text per entry in *lines*."""
    text = "\n".join(
        "(" + line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") Tj T*"
        for line in lines
    )
    stream = f"BT\n/F1 8 Tf\n10 TL\n40 800 Td\n{text}\nET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length %d >>\nstream\n%b\nendstream" % (len(stream), stream),
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%b\nendobj\n" % (number, body)
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(pdf)


@pytest.fixture
def text_pdf() -> Callable[[list[str]], bytes]:
    """Builds a one-page PDF with one line of Helvetica text per entry in its argument."""
    return _text_pdf
