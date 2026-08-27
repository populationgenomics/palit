#!/usr/bin/env python3
"""Fixed gene-disease associations ("entities") the pipeline assesses against.

Entities are seeded once from a GenCC submissions export and are read-only for
every downstream stage: extraction assigns its disease entity blocks to one of a
gene's listed associations, and aggregation runs per entity rather than per gene.
"""

import csv
import logging
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import typer

from palit.hgnc import HgncResolver

app = typer.Typer(help="Seed fixed gene-disease entities from a GenCC submissions export")
logger = logging.getLogger(__name__)

# GenCC `moi_title` values mapped onto the palit inheritance vocabulary, which is
# the evidence extraction enum plus the combined value understood by
# panelapp_integration.decompose_moi. Covers every value GenCC emits; an
# unmapped title is a data surprise and raises KeyError.
GENCC_MOI_TO_ENUM: dict[str, str] = {
    "Autosomal dominant": "Monoallelic",
    "Autosomal recessive": "Biallelic",
    "Semidominant": "Monoallelic_and_biallelic",
    "X-linked": "X-linked",
    "X-linked recessive": "X-linked",
    "Mitochondrial": "Mitochondrial",
    "Unknown": "Other",
    "Y-linked inheritance": "Other",
}

# Prose rendering of each inheritance enum for the prompts. The legacy
# parentheticals are deliberate: they give the model the wording it meets in the
# literature. Reports use MOI_DISPLAY instead.
MOI_PROMPT_GLOSS: dict[str, str] = {
    "Monoallelic": "monoallelic (autosomal dominant)",
    "Biallelic": "biallelic (autosomal recessive)",
    "Monoallelic_and_biallelic": "both monoallelic and biallelic",
    "X-linked": "X-linked",
    "Mitochondrial": "mitochondrial",
    "Other": "other/unknown inheritance",
}

# Reader-facing rendering of each inheritance enum, parenthesis-free so a report
# can wrap it in parentheses of its own.
MOI_DISPLAY: dict[str, str] = {
    "Monoallelic": "monoallelic",
    "Biallelic": "biallelic",
    "Monoallelic_and_biallelic": "both monoallelic and biallelic",
    "X-linked": "X-linked",
    "Mitochondrial": "mitochondrial",
    "Other": "other",
}

ENTITY_BLOCK_HEADER = """FIXED DISEASE ASSOCIATIONS

Extract evidence only for the genes listed here. Assign every disease entity
block you emit to exactly one of the gene's listed associations: set its
`entity` to that association's mondo_id and moi, copied from the two labelled
values below, or to null if no listed association fits. The indented line under
each association describes it — disease name and inheritance — so you can match
the paper against it; that text is never copied into `entity`."""


@dataclass(frozen=True)
class GeneDiseaseEntityInput:
    """A gene-disease entity before it is written, hence without a database id."""

    hgnc_id: int
    mondo_id: str
    disease_title: str
    moi: str
    gencc_moi_titles: str
    gencc_classification: str
    source: str


@dataclass(frozen=True)
class DiseaseEntity:
    """A gene-disease entity as stored in `gene_disease_entities`."""

    id: int
    hgnc_id: int
    mondo_id: str
    disease_title: str
    moi: str
    gencc_moi_titles: str
    gencc_classification: str
    source: str


def entity_ref(mondo_id: str, moi: str) -> str:
    """Render the (disease, inheritance mode) pair that identifies an association.

    Entities are keyed by (gene, disease, inheritance mode), so the disease alone
    does not identify one: a gene curated as both dominant and recessive for the
    same MONDO term has two entities. The gene is implied by the block being
    resolved, leaving "MONDO:0979231|Monoallelic" as a lookup key within a gene
    and as a compact label for logs and validation messages.
    """
    return f"{mondo_id}|{moi}"


def load_gencc_rows(path: Path) -> list[dict[str, str]]:
    """Read a GenCC submissions TSV export.

    Free-text fields (notes, disease titles) contain embedded newlines inside
    quoted values, so the file must be parsed by the csv module rather than by
    splitting on lines.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    logger.info(f"Loaded {len(rows)} GenCC rows from {path}")
    return rows


def _dedupe_group(
    key: tuple[int, str, str],
    group: list[dict[str, str]],
    source: str,
) -> GeneDiseaseEntityInput:
    """Collapse the GenCC rows behind one (gene, disease, inheritance mode) key.

    Several source titles can share a palit enum ("X-linked" and "X-linked
    recessive" both map to X-linked), which yields one entity carrying both
    titles. Rows that agree on the mode but disagree on the classification are
    rejected rather than silently picked from.
    """
    hgnc_id, mondo_id, moi = key

    classifications = {row["classification_title"] for row in group}
    if len(classifications) > 1:
        raise ValueError(
            f"HGNC:{hgnc_id} {mondo_id} [{moi}]: "
            f"conflicting classifications {sorted(classifications)}"
        )

    titles: list[str] = []
    for row in group:
        if row["moi_title"] not in titles:
            titles.append(row["moi_title"])

    return GeneDiseaseEntityInput(
        hgnc_id=hgnc_id,
        mondo_id=mondo_id,
        disease_title=group[0]["disease_title"],
        moi=moi,
        gencc_moi_titles="; ".join(titles),
        gencc_classification=group[0]["classification_title"],
        source=source,
    )


def build_entities(
    rows: Sequence[dict[str, str]],
    submitter: str,
    symbols: Sequence[str],
    resolver: HgncResolver,
    source: str,
) -> list[GeneDiseaseEntityInput]:
    """Select one submitter's curations for the requested genes as entities.

    One entity per (gene, disease, inheritance mode): a gene-disease pair GenCC
    curates as both dominant and recessive becomes two entities, each keeping its
    own classification, since a rating earned under one mode says nothing about
    the other. Requested symbols are resolved to their current HGNC symbol first,
    so an alias or withdrawn symbol still matches the export's `gene_symbol`.

    `source` is stamped on every entity and reported verbatim, so it must name the
    export the curations came from, not just the submitter.
    """
    wanted: set[str] = set()
    for symbol in symbols:
        entry = resolver.resolve(symbol)
        if entry is None:
            raise ValueError(f"Requested gene symbol does not resolve to HGNC: {symbol}")
        wanted.add(entry.symbol)

    groups: dict[tuple[int, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["submitter_title"] != submitter or row["gene_symbol"] not in wanted:
            continue

        entry = resolver.resolve(row["gene_symbol"])
        if entry is None:
            raise ValueError(f"GenCC gene symbol does not resolve to HGNC: {row['gene_symbol']}")
        if row["gene_curie"] != f"HGNC:{entry.hgnc_id}":
            raise ValueError(
                f"GenCC gene_curie {row['gene_curie']} disagrees with HGNC:{entry.hgnc_id} "
                f"for symbol {row['gene_symbol']}"
            )

        mondo_id = row["disease_curie"]
        if not mondo_id.startswith("MONDO:"):
            raise ValueError(f"HGNC:{entry.hgnc_id}: disease_curie is not a MONDO term: {mondo_id}")

        moi = GENCC_MOI_TO_ENUM[row["moi_title"]]
        groups.setdefault((entry.hgnc_id, mondo_id, moi), []).append(row)

    entities = [_dedupe_group(key, group, source) for key, group in groups.items()]
    entities.sort(key=lambda e: (e.hgnc_id, e.mondo_id, e.moi))
    selected = sum(len(group) for group in groups.values())
    logger.info(f"Built {len(entities)} entities from {selected} matching GenCC rows")
    return entities


def insert_entities(db_path: Path, entities: Sequence[GeneDiseaseEntityInput]) -> int:
    """Write entities, refreshing the non-key columns of any that already exist.

    Re-seeding is idempotent and keeps ids stable, so `entity_mentions` and
    `gene_disease_assessments` rows survive a refreshed export.
    """
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO gene_disease_entities
                (hgnc_id, mondo_id, disease_title, moi, gencc_moi_titles,
                 gencc_classification, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(hgnc_id, mondo_id, moi) DO UPDATE SET
                disease_title = excluded.disease_title,
                gencc_moi_titles = excluded.gencc_moi_titles,
                gencc_classification = excluded.gencc_classification,
                source = excluded.source
            """,
            [
                (
                    e.hgnc_id,
                    e.mondo_id,
                    e.disease_title,
                    e.moi,
                    e.gencc_moi_titles,
                    e.gencc_classification,
                    e.source,
                )
                for e in entities
            ],
        )
    return len(entities)


def _row_to_entity(row: sqlite3.Row) -> DiseaseEntity:
    return DiseaseEntity(
        id=row["id"],
        hgnc_id=row["hgnc_id"],
        mondo_id=row["mondo_id"],
        disease_title=row["disease_title"],
        moi=row["moi"],
        gencc_moi_titles=row["gencc_moi_titles"],
        gencc_classification=row["gencc_classification"],
        source=row["source"],
    )


def load_entities(db_path: Path) -> list[DiseaseEntity]:
    """Load every seeded entity, ordered by gene, disease, then inheritance mode."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, hgnc_id, mondo_id, disease_title, moi, gencc_moi_titles,
                   gencc_classification, source
            FROM gene_disease_entities
            ORDER BY hgnc_id, mondo_id, moi
            """
        ).fetchall()

    if not rows:
        raise ValueError(f"{db_path} has no gene_disease_entities — run seed-entities first")

    return [_row_to_entity(row) for row in rows]


def entities_by_gene(entities: Sequence[DiseaseEntity]) -> dict[int, list[DiseaseEntity]]:
    """Group entities by HGNC ID, preserving the input order within each gene."""
    by_gene: dict[int, list[DiseaseEntity]] = {}
    for entity in entities:
        by_gene.setdefault(entity.hgnc_id, []).append(entity)
    return by_gene


def load_entities_by_doi(db_path: Path) -> dict[str, dict[int, list[DiseaseEntity]]]:
    """Map each paper to the entities of the genes its relevance assessment named."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT gm.paper_doi, e.id, e.hgnc_id, e.mondo_id, e.disease_title, e.moi,
                   e.gencc_moi_titles, e.gencc_classification, e.source
            FROM gene_mentions gm
            JOIN gene_disease_entities e ON e.hgnc_id = gm.hgnc_id
            WHERE gm.source = 'relevance_assessment'
            ORDER BY gm.paper_doi, e.hgnc_id, e.mondo_id, e.moi
            """
        ).fetchall()

    by_doi: dict[str, dict[int, list[DiseaseEntity]]] = {}
    for row in rows:
        entity = _row_to_entity(row)
        by_doi.setdefault(row["paper_doi"], {}).setdefault(entity.hgnc_id, []).append(entity)
    return by_doi


def format_entity_block(by_gene: dict[int, list[DiseaseEntity]], resolver: HgncResolver) -> str:
    """Render the fixed associations of one paper's genes for the extraction prompt.

    Each association spans two lines: its two identifying fields under the labels
    the model must emit them under, then an indented description of the disease.
    Labelling the fields separately gives the model two values to copy across
    rather than one string to reassemble, and keeps the descriptive text visibly
    outside them.

    Genes are ordered by symbol and entities by MONDO ID then inheritance mode so
    the block is stable across runs. The GenCC classification is deliberately left
    out: the model must weigh the literature, not restate someone else's verdict.
    """
    lines = [ENTITY_BLOCK_HEADER]
    for hgnc_id in sorted(by_gene, key=resolver.get_symbol):
        lines.append("")
        lines.append(f"{resolver.get_symbol(hgnc_id)} (HGNC:{hgnc_id})")
        for entity in sorted(by_gene[hgnc_id], key=lambda e: (e.mondo_id, e.moi)):
            gloss = MOI_PROMPT_GLOSS[entity.moi]
            lines.append(f"  - mondo_id: {entity.mondo_id} | moi: {entity.moi}")
            lines.append(f"    {entity.disease_title} — {gloss}")
    return "\n".join(lines)


@app.callback(invoke_without_command=True)
def seed_entities_command(
    db_path: Path = typer.Option(
        Path("data/db.sqlite"),
        "--db-path",
        help="Path to SQLite database",
    ),
    gencc_tsv: Path = typer.Option(
        ...,
        "--gencc-tsv",
        help="GenCC submissions TSV export",
    ),
    submitter: str = typer.Option(
        ...,
        "--submitter",
        help="GenCC submitter_title whose curations define the entities",
    ),
    genes: list[str] = typer.Option(
        ...,
        "--gene",
        help="Gene symbol to seed (repeatable)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Log the entities that would be written, then exit without writing",
    ),
) -> None:
    """Seed fixed gene-disease entities from a GenCC submissions export."""
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)
    if not gencc_tsv.exists():
        logger.error(f"GenCC export not found: {gencc_tsv}")
        raise typer.Exit(1)

    resolver = HgncResolver.from_file()
    source = f"gencc:{submitter}:{gencc_tsv.name}"
    entities = build_entities(load_gencc_rows(gencc_tsv), submitter, genes, resolver, source)

    by_gene: dict[int, list[GeneDiseaseEntityInput]] = {}
    for entity in entities:
        by_gene.setdefault(entity.hgnc_id, []).append(entity)

    if dry_run:
        logger.info(f"Dry run - {len(entities)} entities over {len(by_gene)} genes:")
        for entity in entities:
            logger.info(
                f"  {resolver.get_symbol(entity.hgnc_id)} {entity.mondo_id} "
                f"[{entity.moi}] {entity.disease_title}"
            )
        return

    written = insert_entities(db_path, entities)
    logger.info(f"Seeded {written} entities over {len(by_gene)} genes into {db_path}")
    for hgnc_id, gene_entities in sorted(
        by_gene.items(), key=lambda kv: resolver.get_symbol(kv[0])
    ):
        logger.info(f"  {resolver.get_symbol(hgnc_id)} (HGNC:{hgnc_id}): {len(gene_entities)}")


if __name__ == "__main__":
    app()
