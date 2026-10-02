"""Shared fixtures: a small PanelApp snapshot, GenCC index, HGNC resolver and curated record."""

from typing import Any

import pytest

from palit.gencc import GenccIndex, GeneGencc, PaaAssociation
from palit.hgnc import HgncEntry, HgncResolver
from palit.panelapp_check import CuratedRecord
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
    }


def _panel(
    panel_id: int, name: str, genes: list[dict[str, Any]], strs: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {"id": panel_id, "name": name, "genes": genes, "strs": strs or []}


@pytest.fixture
def panelapp_snapshot() -> dict[int, dict[str, Any]]:
    """Mendeliome with a gene and a repeat expansion, an Incidentalome-only gene, and
    that gene's entry on another panel next to an entity without an HGNC ID."""
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
    """GENEA (HGNC:1) has a Strong and a Limited PanelApp Australia GenCC row; no other gene
    has any."""
    strong = PaaAssociation(
        mondo_id="MONDO:0000001",
        disease_title="disease A",
        moi_title="Autosomal recessive",
        moi="Biallelic",
        classification="Strong",
        rating="GREEN",
        date="2025-01-17",
        definition="",
        obsolete=False,
        replaced_by=(),
    )
    limited = PaaAssociation(
        mondo_id="MONDO:0000002",
        disease_title="disease B",
        moi_title="Autosomal dominant",
        moi="Monoallelic",
        classification="Limited",
        rating="RED",
        date="2025-01-17",
        definition="",
        obsolete=False,
        replaced_by=(),
    )
    return GenccIndex({1: GeneGencc(paa_associations=(strong, limited))})


@pytest.fixture
def curated_record(
    panelapp_snapshot: dict[int, dict[str, Any]], gencc_index: GenccIndex
) -> CuratedRecord:
    """Reference panels Mendeliome and Incidentalome, with the Incidentalome fallback."""
    return CuratedRecord.build(
        panelapp_snapshot,
        "2026-09-01",
        REFERENCE_PANEL_IDS,
        gencc_index,
        incidentalome_fallback=True,
    )
