"""Tests for expand-literature's handling of the corpus before its tournaments."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from palit import expand_literature

ROOT = Path(__file__).resolve().parents[1]


def test_force_all_clears_the_links_before_deleting_expansion_papers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "run.sqlite"
    baseline_path = tmp_path / "baseline.sqlite"
    sqlite3.connect(baseline_path).close()
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.executemany(
            "INSERT INTO papers (doi, title, source, source_type) VALUES (?, 't', 'pubmed', ?)",
            [("10.1/seeded", "expansion"), ("10.1/cited", "initial")],
        )
        conn.executemany(
            "INSERT INTO panelapp_citations (hgnc_id, paper_doi) VALUES (1, ?)",
            [("10.1/seeded",), ("10.1/cited",)],
        )
        # Deleting a paper that is still linked fails, as it would with foreign keys on.
        conn.execute(
            """
            CREATE TRIGGER linked_paper_deleted BEFORE DELETE ON papers
            WHEN EXISTS (SELECT 1 FROM panelapp_citations WHERE paper_doi = OLD.doi)
            BEGIN SELECT RAISE(ABORT, 'a linked paper was deleted'); END
            """
        )

    seen: list[tuple[list[str], int]] = []

    def seed(db: Path, panels: dict[int, dict[str, Any]], baseline: Path, _: Any) -> None:
        with sqlite3.connect(db) as conn:
            dois = [row[0] for row in conn.execute("SELECT doi FROM papers ORDER BY doi")]
            (links,) = conn.execute("SELECT COUNT(*) FROM panelapp_citations").fetchone()
        seen.append((dois, links))

    monkeypatch.setattr(expand_literature, "seed_panelapp_publications", seed)
    monkeypatch.setattr(expand_literature, "load_panels_with_reviews", lambda *_: {})
    monkeypatch.setattr(expand_literature.HgncResolver, "from_file", lambda: None)
    monkeypatch.chdir(ROOT)

    result = CliRunner().invoke(
        expand_literature.app,
        [
            "--cutoff-date",
            "2026-10-01",
            "--panel-date",
            "2026-10-15",
            "--db-path",
            str(db_path),
            "--baseline-db-path",
            str(baseline_path),
            "--force-all",
        ],
    )

    assert result.exit_code == 0, result.output
    # The run has no genes, so seeding is the last step and no tournament runs.
    assert seen == [(["10.1/cited"], 0)]
