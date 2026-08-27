"""Tests for fixed gene-disease entities: GenCC seeding, storage, prompt block."""

import json
import re
import sqlite3
from pathlib import Path

import pytest

from palit.entities import (
    GENCC_MOI_TO_ENUM,
    MOI_DISPLAY,
    MOI_PROMPT_GLOSS,
    DiseaseEntity,
    build_entities,
    entities_by_gene,
    entity_ref,
    format_entity_block,
    insert_entities,
    load_entities,
    load_entities_by_doi,
    load_gencc_rows,
)
from palit.hgnc import HgncResolver

SUBMITTER = "PanelApp Australia"
SOURCE = f"gencc:{SUBMITTER}:gencc_submissions_2026-06-22.tsv"

# (hgnc_id, symbol, prev_symbols) for the genes used below.
_FIXTURE_GENES = [
    (20, "AARS1", ["AARS"]),
    (4922, "HK1", []),
    (6990, "MECP2", ["RTT"]),
    (12407, "TUBA4A", ["TUBA1"]),
]

# --- helpers ----------------------------------------------------------------


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


def _row(
    symbol: str,
    hgnc_id: int,
    mondo_id: str,
    moi_title: str,
    *,
    disease_title: str = "some disease",
    classification: str = "Strong",
    submitter: str = SUBMITTER,
    gene_curie: str | None = None,
) -> dict[str, str]:
    """Build the subset of GenCC columns build_entities reads."""
    return {
        "gene_curie": gene_curie if gene_curie is not None else f"HGNC:{hgnc_id}",
        "gene_symbol": symbol,
        "disease_curie": mondo_id,
        "disease_title": disease_title,
        "classification_title": classification,
        "moi_title": moi_title,
        "submitter_title": submitter,
    }


def _new_db(tmp_path: Path) -> Path:
    path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(Path("schema.sql").read_text())
    finally:
        conn.close()
    return path


def _schema_moi_values() -> set[str]:
    """The MOI values the gene_disease_entities CHECK constraint permits."""
    schema = Path("schema.sql").read_text()
    match = re.search(r"CHECK\(moi IN \(([^)]*)\)\)", schema)
    assert match is not None
    return set(re.findall(r"'([^']*)'", match.group(1)))


# --- MOI vocabulary ---------------------------------------------------------


def test_gencc_moi_map_covers_the_gencc_vocabulary() -> None:
    assert set(GENCC_MOI_TO_ENUM) == {
        "Autosomal dominant",
        "Autosomal recessive",
        "Semidominant",
        "X-linked",
        "X-linked recessive",
        "Mitochondrial",
        "Unknown",
        "Y-linked inheritance",
    }


def test_mapped_moi_values_are_accepted_by_the_schema() -> None:
    allowed = _schema_moi_values()
    assert set(GENCC_MOI_TO_ENUM.values()) <= allowed
    assert set(MOI_PROMPT_GLOSS) == allowed


def test_prompt_and_display_glosses_cover_the_same_modes() -> None:
    assert set(MOI_DISPLAY) == set(MOI_PROMPT_GLOSS)


def test_display_glosses_carry_no_parentheses() -> None:
    # Reports wrap the display gloss in parentheses of their own.
    assert not any("(" in gloss for gloss in MOI_DISPLAY.values())


def test_unknown_moi_title_raises(resolver: HgncResolver) -> None:
    rows = [_row("AARS1", 20, "MONDO:0013212", "Autosomal codominant")]
    with pytest.raises(KeyError):
        build_entities(rows, SUBMITTER, ["AARS1"], resolver, SOURCE)


# --- entity identity --------------------------------------------------------


def test_dominant_and_recessive_are_separate_entities(resolver: HgncResolver) -> None:
    rows = [
        _row("TUBA4A", 12407, "MONDO:0979231", "Autosomal dominant"),
        _row("TUBA4A", 12407, "MONDO:0979231", "Autosomal recessive"),
    ]
    entities = build_entities(rows, SUBMITTER, ["TUBA4A"], resolver, SOURCE)
    assert [(e.moi, e.gencc_moi_titles) for e in entities] == [
        ("Biallelic", "Autosomal recessive"),
        ("Monoallelic", "Autosomal dominant"),
    ]
    assert {e.mondo_id for e in entities} == {"MONDO:0979231"}
    assert {e.source for e in entities} == {SOURCE}


def test_source_titles_sharing_one_enum_dedupe(resolver: HgncResolver) -> None:
    rows = [
        _row("MECP2", 6990, "MONDO:0010726", "X-linked"),
        _row("MECP2", 6990, "MONDO:0010726", "X-linked recessive"),
    ]
    entities = build_entities(rows, SUBMITTER, ["MECP2"], resolver, SOURCE)
    assert len(entities) == 1
    assert entities[0].moi == "X-linked"
    assert entities[0].gencc_moi_titles == "X-linked; X-linked recessive"


def test_identical_rows_dedupe_to_one_entity(resolver: HgncResolver) -> None:
    rows = [
        _row("HK1", 4922, "MONDO:0009212", "Autosomal recessive"),
        _row("HK1", 4922, "MONDO:0009212", "Autosomal recessive"),
    ]
    entities = build_entities(rows, SUBMITTER, ["HK1"], resolver, SOURCE)
    assert len(entities) == 1
    assert entities[0].moi == "Biallelic"
    assert entities[0].gencc_moi_titles == "Autosomal recessive"


def test_conflicting_classifications_within_one_entity_raise(resolver: HgncResolver) -> None:
    rows = [
        _row("MECP2", 6990, "MONDO:0010726", "X-linked", classification="Definitive"),
        _row("MECP2", 6990, "MONDO:0010726", "X-linked recessive", classification="Limited"),
    ]
    with pytest.raises(ValueError, match="conflicting classifications"):
        build_entities(rows, SUBMITTER, ["MECP2"], resolver, SOURCE)


def test_classifications_may_differ_across_inheritance_modes(resolver: HgncResolver) -> None:
    # The rating a gene-disease pair earns as dominant says nothing about its
    # recessive evidence, so the two entities keep their own classifications.
    rows = [
        _row("TUBA4A", 12407, "MONDO:0979231", "Autosomal dominant", classification="Limited"),
        _row("TUBA4A", 12407, "MONDO:0979231", "Autosomal recessive", classification="Strong"),
    ]
    entities = build_entities(rows, SUBMITTER, ["TUBA4A"], resolver, SOURCE)
    assert [(e.moi, e.gencc_classification) for e in entities] == [
        ("Biallelic", "Strong"),
        ("Monoallelic", "Limited"),
    ]


# --- filtering and validation -----------------------------------------------


def test_only_the_requested_submitter_and_genes_are_kept(resolver: HgncResolver) -> None:
    rows = [
        _row("AARS1", 20, "MONDO:0013212", "Autosomal dominant"),
        _row("AARS1", 20, "MONDO:0100000", "Autosomal recessive", submitter="Ambry Genetics"),
        _row("HK1", 4922, "MONDO:0009212", "Autosomal recessive"),
    ]
    entities = build_entities(rows, SUBMITTER, ["AARS1"], resolver, SOURCE)
    assert [(e.hgnc_id, e.mondo_id) for e in entities] == [(20, "MONDO:0013212")]


def test_requested_symbols_are_resolved_before_matching(resolver: HgncResolver) -> None:
    # "AARS" is a previous symbol; the export carries the current one.
    rows = [_row("AARS1", 20, "MONDO:0013212", "Autosomal dominant")]
    entities = build_entities(rows, SUBMITTER, ["AARS"], resolver, SOURCE)
    assert len(entities) == 1


def test_unresolvable_requested_symbol_raises(resolver: HgncResolver) -> None:
    with pytest.raises(ValueError, match="NOTAGENE"):
        build_entities([], SUBMITTER, ["NOTAGENE"], resolver, SOURCE)


def test_gene_curie_mismatch_raises(resolver: HgncResolver) -> None:
    rows = [_row("AARS1", 20, "MONDO:0013212", "Autosomal dominant", gene_curie="HGNC:99999")]
    with pytest.raises(ValueError, match="HGNC:99999"):
        build_entities(rows, SUBMITTER, ["AARS1"], resolver, SOURCE)


def test_non_mondo_disease_curie_raises(resolver: HgncResolver) -> None:
    rows = [_row("AARS1", 20, "OMIM:613287", "Autosomal dominant")]
    with pytest.raises(ValueError, match="not a MONDO term"):
        build_entities(rows, SUBMITTER, ["AARS1"], resolver, SOURCE)


# --- TSV parsing ------------------------------------------------------------


def test_load_gencc_rows_handles_embedded_newlines(tmp_path: Path) -> None:
    path = tmp_path / "gencc.tsv"
    path.write_text(
        "gene_symbol\tdisease_title\tmoi_title\n"
        'AARS1\t"a title\nspanning lines"\tAutosomal dominant\n'
        "HK1\tplain title\tAutosomal recessive\n"
    )
    rows = load_gencc_rows(path)
    assert len(rows) == 2
    assert rows[0]["disease_title"] == "a title\nspanning lines"
    assert rows[1]["gene_symbol"] == "HK1"


# --- storage ----------------------------------------------------------------


def test_insert_and_load_round_trip(tmp_path: Path, resolver: HgncResolver) -> None:
    db_path = _new_db(tmp_path)
    rows = [
        _row("HK1", 4922, "MONDO:0009212", "Autosomal recessive"),
        _row("AARS1", 20, "MONDO:0013212", "Autosomal dominant", disease_title="CMT2N"),
    ]
    written = insert_entities(
        db_path, build_entities(rows, SUBMITTER, ["AARS1", "HK1"], resolver, SOURCE)
    )
    assert written == 2

    entities = load_entities(db_path)
    assert [(e.hgnc_id, e.mondo_id) for e in entities] == [
        (20, "MONDO:0013212"),
        (4922, "MONDO:0009212"),
    ]
    assert entities[0].disease_title == "CMT2N"
    assert entities[0].moi == "Monoallelic"
    assert entities_by_gene(entities) == {20: [entities[0]], 4922: [entities[1]]}


def test_one_disease_under_two_mois_stores_two_rows(tmp_path: Path, resolver: HgncResolver) -> None:
    db_path = _new_db(tmp_path)
    rows = [
        _row("TUBA4A", 12407, "MONDO:0979231", "Autosomal dominant"),
        _row("TUBA4A", 12407, "MONDO:0979231", "Autosomal recessive"),
    ]
    insert_entities(db_path, build_entities(rows, SUBMITTER, ["TUBA4A"], resolver, SOURCE))

    entities = load_entities(db_path)
    assert [entity_ref(e.mondo_id, e.moi) for e in entities] == [
        "MONDO:0979231|Biallelic",
        "MONDO:0979231|Monoallelic",
    ]
    assert len({e.id for e in entities}) == 2


def test_reseeding_updates_in_place_and_keeps_ids(tmp_path: Path, resolver: HgncResolver) -> None:
    db_path = _new_db(tmp_path)
    rows = [_row("AARS1", 20, "MONDO:0013212", "Autosomal dominant", classification="Limited")]
    insert_entities(db_path, build_entities(rows, SUBMITTER, ["AARS1"], resolver, SOURCE))
    original_id = load_entities(db_path)[0].id

    updated = [_row("AARS1", 20, "MONDO:0013212", "Autosomal dominant", classification="Strong")]
    insert_entities(db_path, build_entities(updated, SUBMITTER, ["AARS1"], resolver, SOURCE))

    entities = load_entities(db_path)
    assert len(entities) == 1
    assert entities[0].id == original_id
    assert entities[0].gencc_classification == "Strong"


def test_load_entities_on_empty_table_raises(tmp_path: Path) -> None:
    db_path = _new_db(tmp_path)
    with pytest.raises(ValueError, match="run seed-entities first"):
        load_entities(db_path)


def test_load_entities_by_doi_follows_relevance_gene_mentions(
    tmp_path: Path, resolver: HgncResolver
) -> None:
    db_path = _new_db(tmp_path)
    rows = [
        _row("AARS1", 20, "MONDO:0013212", "Autosomal dominant"),
        _row("AARS1", 20, "MONDO:0100000", "Autosomal recessive"),
        _row("HK1", 4922, "MONDO:0009212", "Autosomal recessive"),
    ]
    insert_entities(db_path, build_entities(rows, SUBMITTER, ["AARS1", "HK1"], resolver, SOURCE))

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO papers (doi, title, source) VALUES ('10.1/a', 'A', 'pubmed')",
        )
        conn.executemany(
            """
            INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source)
            VALUES (?, ?, '10.1/a', ?)
            """,
            [(20, "AARS1", "relevance_assessment"), (4922, "HK1", "expansion_evidence")],
        )

    by_doi = load_entities_by_doi(db_path)
    assert list(by_doi) == ["10.1/a"]
    # Only the relevance-assessment mention contributes, and both AARS1 entities come with it.
    assert [e.mondo_id for e in by_doi["10.1/a"][20]] == ["MONDO:0013212", "MONDO:0100000"]
    assert 4922 not in by_doi["10.1/a"]


# --- entity references ------------------------------------------------------


def _entity(id_: int, hgnc_id: int, mondo_id: str, title: str, moi: str) -> DiseaseEntity:
    return DiseaseEntity(
        id=id_,
        hgnc_id=hgnc_id,
        mondo_id=mondo_id,
        disease_title=title,
        moi=moi,
        gencc_moi_titles="Autosomal dominant",
        gencc_classification="Definitive",
        source=SOURCE,
    )


def test_entity_ref_labels_the_disease_and_inheritance_pair() -> None:
    entity = _entity(1, 12407, "MONDO:0979231", "maturation arrest 23", "Monoallelic_and_biallelic")
    assert entity_ref(entity.mondo_id, entity.moi) == "MONDO:0979231|Monoallelic_and_biallelic"


# --- prompt block -----------------------------------------------------------


def test_format_entity_block(resolver: HgncResolver) -> None:
    by_gene = {
        4922: [_entity(4, 4922, "MONDO:0009212", "hemolytic anemia", "Biallelic")],
        12407: [
            _entity(3, 12407, "MONDO:0979231", "maturation arrest 23", "Monoallelic"),
            _entity(2, 12407, "MONDO:0979231", "maturation arrest 23", "Biallelic"),
        ],
        20: [_entity(1, 20, "MONDO:0013212", "Charcot-Marie-Tooth type 2N", "Monoallelic")],
    }

    assert format_entity_block(by_gene, resolver) == (
        "FIXED DISEASE ASSOCIATIONS\n"
        "\n"
        "Extract evidence only for the genes listed here. Assign every disease entity\n"
        "block you emit to exactly one of the gene's listed associations: set its\n"
        "`entity` to that association's mondo_id and moi, copied from the two labelled\n"
        "values below, or to null if no listed association fits. The indented line under\n"
        "each association describes it — disease name and inheritance — so you can match\n"
        "the paper against it; that text is never copied into `entity`.\n"
        "\n"
        "AARS1 (HGNC:20)\n"
        "  - mondo_id: MONDO:0013212 | moi: Monoallelic\n"
        "    Charcot-Marie-Tooth type 2N — monoallelic (autosomal dominant)\n"
        "\n"
        "HK1 (HGNC:4922)\n"
        "  - mondo_id: MONDO:0009212 | moi: Biallelic\n"
        "    hemolytic anemia — biallelic (autosomal recessive)\n"
        "\n"
        "TUBA4A (HGNC:12407)\n"
        "  - mondo_id: MONDO:0979231 | moi: Biallelic\n"
        "    maturation arrest 23 — biallelic (autosomal recessive)\n"
        "  - mondo_id: MONDO:0979231 | moi: Monoallelic\n"
        "    maturation arrest 23 — monoallelic (autosomal dominant)"
    )
