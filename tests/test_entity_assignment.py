"""Tests for routing extracted evidence into a paper's fixed disease associations."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from palit.entities import DiseaseEntity
from palit.extract_evidence import (
    annotate_entity_ids,
    drop_off_target_evaluations,
    validate_entity_assignments,
)
from palit.hgnc import HgncResolver

# (hgnc_id, symbol, prev_symbols) for the genes used below.
_FIXTURE_GENES = [
    (20, "AARS1", ["AARS"]),
    (4922, "HK1", []),
    (12407, "TUBA4A", ["TUBA1"]),
]

AARS1_DOMINANT = "MONDO:0013212|Monoallelic"
AARS1_RECESSIVE = "MONDO:0100000|Biallelic"
TUBA4A_BOTH = "MONDO:0979231|Monoallelic_and_biallelic"
SHARED_RECESSIVE = "MONDO:0009212|Biallelic"


@pytest.fixture
def resolver(tmp_path: Path) -> HgncResolver:
    """An HgncResolver over a handful of genes, loaded through its real loader."""
    docs = [
        {
            "hgnc_id": f"HGNC:{hgnc_id}",
            "symbol": symbol,
            "prev_symbol": prev_symbols,
            "alias_symbol": [],
            "locus_group": "protein-coding gene",
            "location": "16q22.1",
        }
        for hgnc_id, symbol, prev_symbols in _FIXTURE_GENES
    ]
    path = tmp_path / "hgnc.json"
    path.write_text(json.dumps({"response": {"docs": docs}}))
    return HgncResolver.from_file(path)


def _entity(id_: int, hgnc_id: int, ref: str) -> DiseaseEntity:
    mondo_id, moi = ref.split("|")
    return DiseaseEntity(
        id=id_,
        hgnc_id=hgnc_id,
        mondo_id=mondo_id,
        disease_title="some disease",
        moi=moi,
        gencc_moi_titles="Autosomal dominant",
        gencc_classification="Definitive",
        source="gencc:PanelApp Australia",
    )


@pytest.fixture
def entities_for_paper() -> dict[int, list[DiseaseEntity]]:
    """Two genes with two associations each, sharing one (disease, inheritance) pair."""
    return {
        20: [_entity(1, 20, AARS1_DOMINANT), _entity(2, 20, AARS1_RECESSIVE)],
        4922: [_entity(3, 4922, SHARED_RECESSIVE)],
        12407: [_entity(4, 12407, TUBA4A_BOTH), _entity(5, 12407, SHARED_RECESSIVE)],
    }


def _block(ref: str | None, inheritance_mode: str = "Monoallelic") -> dict[str, Any]:
    return {"entity_ref": ref, "inheritance_mode": inheritance_mode}


def _parsed(*gene_evaluations: dict[str, Any]) -> dict[str, Any]:
    return {"gene_evaluations": list(gene_evaluations)}


def _gene_eval(symbol: str, *blocks: dict[str, Any]) -> dict[str, Any]:
    return {"gene_symbol": symbol, "disease_entities": list(blocks)}


# --- reference validation ---------------------------------------------------


def test_listed_references_pass(
    entities_for_paper: dict[int, list[DiseaseEntity]], resolver: HgncResolver
) -> None:
    parsed = _parsed(
        _gene_eval("AARS1", _block(AARS1_DOMINANT), _block(AARS1_RECESSIVE, "Biallelic")),
        _gene_eval("HK1", _block(SHARED_RECESSIVE, "Biallelic")),
    )
    result = validate_entity_assignments(parsed, entities_for_paper, resolver)
    assert result.errors == []
    assert result.off_target_symbols == []


def test_previous_symbol_resolves_to_its_gene(
    entities_for_paper: dict[int, list[DiseaseEntity]], resolver: HgncResolver
) -> None:
    parsed = _parsed(_gene_eval("AARS", _block(AARS1_DOMINANT)))
    result = validate_entity_assignments(parsed, entities_for_paper, resolver)
    assert result.errors == []
    assert result.off_target_symbols == []


def test_null_reference_passes_and_may_repeat(
    entities_for_paper: dict[int, list[DiseaseEntity]], resolver: HgncResolver
) -> None:
    parsed = _parsed(_gene_eval("AARS1", _block(None), _block(None, "Biallelic")))
    result = validate_entity_assignments(parsed, entities_for_paper, resolver)
    assert result.errors == []


def test_unknown_reference_is_an_error(
    entities_for_paper: dict[int, list[DiseaseEntity]], resolver: HgncResolver
) -> None:
    parsed = _parsed(_gene_eval("AARS1", _block("MONDO:0000001|Monoallelic")))
    result = validate_entity_assignments(parsed, entities_for_paper, resolver)
    assert len(result.errors) == 1
    assert "MONDO:0000001|Monoallelic" in result.errors[0]
    assert result.off_target_symbols == []


def test_another_genes_reference_is_an_error(
    entities_for_paper: dict[int, list[DiseaseEntity]], resolver: HgncResolver
) -> None:
    # HK1's association is real, but not one of AARS1's.
    parsed = _parsed(_gene_eval("AARS1", _block(SHARED_RECESSIVE, "Biallelic")))
    result = validate_entity_assignments(parsed, entities_for_paper, resolver)
    assert len(result.errors) == 1
    assert SHARED_RECESSIVE in result.errors[0]


def test_repeated_reference_within_one_gene_is_an_error(
    entities_for_paper: dict[int, list[DiseaseEntity]], resolver: HgncResolver
) -> None:
    parsed = _parsed(_gene_eval("AARS1", _block(AARS1_DOMINANT), _block(AARS1_DOMINANT)))
    result = validate_entity_assignments(parsed, entities_for_paper, resolver)
    assert len(result.errors) == 1
    assert "more than one disease entity" in result.errors[0]


def test_same_reference_under_two_genes_is_fine(
    entities_for_paper: dict[int, list[DiseaseEntity]], resolver: HgncResolver
) -> None:
    parsed = _parsed(
        _gene_eval("HK1", _block(SHARED_RECESSIVE, "Biallelic")),
        _gene_eval("TUBA4A", _block(SHARED_RECESSIVE, "Biallelic")),
    )
    result = validate_entity_assignments(parsed, entities_for_paper, resolver)
    assert result.errors == []


# --- off-target genes -------------------------------------------------------


def test_gene_outside_this_papers_associations_is_off_target(
    entities_for_paper: dict[int, list[DiseaseEntity]], resolver: HgncResolver
) -> None:
    parsed = _parsed(
        _gene_eval("AARS1", _block(AARS1_DOMINANT)),
        _gene_eval("HK1", _block(SHARED_RECESSIVE, "Biallelic")),
    )
    del entities_for_paper[4922]

    result = validate_entity_assignments(parsed, entities_for_paper, resolver)
    assert result.errors == []
    assert result.off_target_symbols == ["HK1"]


def test_unresolvable_symbol_is_off_target(
    entities_for_paper: dict[int, list[DiseaseEntity]], resolver: HgncResolver
) -> None:
    parsed = _parsed(_gene_eval("NOTAGENE", _block(AARS1_DOMINANT)))
    result = validate_entity_assignments(parsed, entities_for_paper, resolver)
    assert result.errors == []
    assert result.off_target_symbols == ["NOTAGENE"]


def test_drop_off_target_evaluations_keeps_the_rest() -> None:
    parsed = _parsed(
        _gene_eval("AARS1", _block(AARS1_DOMINANT)),
        _gene_eval("NOTAGENE", _block(None)),
        _gene_eval("HK1", _block(SHARED_RECESSIVE, "Biallelic")),
    )
    drop_off_target_evaluations(parsed, ["NOTAGENE", "HK1"])
    assert [ge["gene_symbol"] for ge in parsed["gene_evaluations"]] == ["AARS1"]


# --- inheritance mismatches -------------------------------------------------


def test_inheritance_mismatch_warns_without_failing(
    entities_for_paper: dict[int, list[DiseaseEntity]],
    resolver: HgncResolver,
    caplog: pytest.LogCaptureFixture,
) -> None:
    parsed = _parsed(_gene_eval("AARS1", _block(AARS1_DOMINANT, "Biallelic")))
    with caplog.at_level(logging.WARNING, logger="palit.extract_evidence"):
        result = validate_entity_assignments(parsed, entities_for_paper, resolver)

    assert result.errors == []
    assert result.off_target_symbols == []
    assert "differs from" in caplog.text
    assert AARS1_DOMINANT in caplog.text


@pytest.mark.parametrize("inheritance_mode", ["NR", "Other"])
def test_unstated_inheritance_does_not_warn(
    entities_for_paper: dict[int, list[DiseaseEntity]],
    resolver: HgncResolver,
    caplog: pytest.LogCaptureFixture,
    inheritance_mode: str,
) -> None:
    parsed = _parsed(_gene_eval("AARS1", _block(AARS1_DOMINANT, inheritance_mode)))
    with caplog.at_level(logging.WARNING, logger="palit.extract_evidence"):
        validate_entity_assignments(parsed, entities_for_paper, resolver)

    assert caplog.text == ""


@pytest.mark.parametrize("inheritance_mode", ["Monoallelic", "Biallelic"])
def test_combined_association_accepts_either_inheritance(
    entities_for_paper: dict[int, list[DiseaseEntity]],
    resolver: HgncResolver,
    caplog: pytest.LogCaptureFixture,
    inheritance_mode: str,
) -> None:
    parsed = _parsed(_gene_eval("TUBA4A", _block(TUBA4A_BOTH, inheritance_mode)))
    with caplog.at_level(logging.WARNING, logger="palit.extract_evidence"):
        result = validate_entity_assignments(parsed, entities_for_paper, resolver)

    assert result.errors == []
    assert caplog.text == ""


# --- entity id resolution ---------------------------------------------------


def test_entity_ids_resolve_per_gene(
    entities_for_paper: dict[int, list[DiseaseEntity]], resolver: HgncResolver
) -> None:
    # Both genes reference the same (disease, inheritance) pair; each must get its own id.
    parsed = _parsed(
        _gene_eval("HK1", _block(SHARED_RECESSIVE, "Biallelic")),
        _gene_eval("TUBA4A", _block(SHARED_RECESSIVE, "Biallelic"), _block(None)),
    )
    annotate_entity_ids(parsed, entities_for_paper, resolver)

    hk1, tuba4a = parsed["gene_evaluations"]
    assert [block["entity_id"] for block in hk1["disease_entities"]] == [3]
    assert [block["entity_id"] for block in tuba4a["disease_entities"]] == [5, None]


def test_entity_ids_require_off_target_evaluations_to_be_dropped(
    entities_for_paper: dict[int, list[DiseaseEntity]], resolver: HgncResolver
) -> None:
    parsed = _parsed(_gene_eval("NOTAGENE", _block(None)))
    with pytest.raises(ValueError, match="NOTAGENE"):
        annotate_entity_ids(parsed, entities_for_paper, resolver)
