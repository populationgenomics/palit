"""Tests for expand-literature's handling of the corpus before its tournaments, and of
panel-scoped runs, whose expansion is PanelApp citation seeding only."""

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from click.testing import Result
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

    seen: list[tuple[list[str], int, list[int]]] = []

    def seed(
        db: Path, panels: dict[int, dict[str, Any]], baseline: Path, _: Any, hgnc_ids: list[int]
    ) -> None:
        with sqlite3.connect(db) as conn:
            dois = [row[0] for row in conn.execute("SELECT doi FROM papers ORDER BY doi")]
            (links,) = conn.execute("SELECT COUNT(*) FROM panelapp_citations").fetchone()
        seen.append((dois, links, hgnc_ids))

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
    # The run has no genes, so none is triaged or seeded and no tournament runs.
    assert seen == [(["10.1/cited"], 0, [])]


def _run_with_a_gene_to_expand(tmp_path: Path) -> tuple[Path, Path]:
    """A run database whose gene 1 has recent evidence from a relevant paper and no
    tournament result, and an empty baseline."""
    db_path = tmp_path / "run.sqlite"
    baseline_path = tmp_path / "baseline.sqlite"
    sqlite3.connect(baseline_path).close()
    with sqlite3.connect(db_path) as conn:
        conn.executescript((ROOT / "schema.sql").read_text())
        conn.execute(
            "INSERT INTO papers (doi, title, source, source_type, relevance_assessment_json) "
            "VALUES ('10.1/relevant', 't', 'pubmed', 'initial', '{\"relevant\": true}')"
        )
        conn.execute(
            "INSERT INTO gene_mentions (hgnc_id, paper_gene_symbol, paper_doi, source) "
            "VALUES (1, 'A', '10.1/relevant', 'recent_evidence')"
        )
    return db_path, baseline_path


def _refuse_client() -> None:
    raise AssertionError("a panel-scoped expansion should run no tournament")


def _refuse_seed(
    db: Path, panels: dict[int, dict[str, Any]], baseline: Path, _: Any, hgnc_ids: list[int]
) -> None:
    raise AssertionError("expand-literature should have stopped before seeding")


def _expand(db_path: Path, baseline_path: Path, *args: str) -> Result:
    return CliRunner().invoke(
        expand_literature.app,
        [
            "--panel-date",
            "2026-10-15",
            "--db-path",
            str(db_path),
            "--baseline-db-path",
            str(baseline_path),
            *args,
        ],
    )


def test_scoped_expansion_seeds_from_every_panel_and_the_baseline_without_a_tournament(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, baseline_path = _run_with_a_gene_to_expand(tmp_path)
    calls: list[tuple[Path, list[int], Path, list[int]]] = []

    def seed(
        db: Path, panels: dict[int, dict[str, Any]], baseline: Path, _: Any, hgnc_ids: list[int]
    ) -> None:
        calls.append((db, sorted(panels), baseline, hgnc_ids))

    monkeypatch.setattr(expand_literature, "seed_panelapp_publications", seed)
    monkeypatch.setattr(expand_literature, "load_panels_with_reviews", lambda *_: {1: {}, 2: {}})
    monkeypatch.setattr(expand_literature.HgncResolver, "from_file", lambda: None)
    monkeypatch.setattr(expand_literature, "make_client", _refuse_client)
    # Outside the repository the tournament's prompt is not found.
    monkeypatch.chdir(tmp_path)

    result = _expand(db_path, baseline_path, "--scope-panel-id", "2")

    assert result.exit_code == 0, result.output
    # A panel-scoped run seeds every run gene, without a triage.
    assert calls == [(db_path, [1, 2], baseline_path, [1])]
    # Gene 1 is still without a tournament result, which a monthly run would run.
    assert expand_literature.genes_to_expand(db_path, [1]) == [1]


def test_scoped_expansion_needs_the_panel_in_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db_path, baseline_path = _run_with_a_gene_to_expand(tmp_path)
    monkeypatch.setattr(expand_literature, "seed_panelapp_publications", _refuse_seed)
    monkeypatch.setattr(expand_literature, "load_panels_with_reviews", lambda *_: {1: {}})

    result = _expand(db_path, baseline_path, "--scope-panel-id", "2")

    assert result.exit_code == 1
    assert "Panel 2 is not in the 2026-10-15 snapshot" in caplog.text


def test_scoped_expansion_rejects_a_cutoff_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db_path, baseline_path = _run_with_a_gene_to_expand(tmp_path)
    monkeypatch.setattr(expand_literature, "seed_panelapp_publications", _refuse_seed)

    result = _expand(db_path, baseline_path, "--scope-panel-id", "2", "--cutoff-date", "2026-10-01")

    assert result.exit_code == 1
    assert "runs no tournament" in caplog.text


def test_expansion_without_a_scope_panel_needs_a_cutoff_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db_path, baseline_path = _run_with_a_gene_to_expand(tmp_path)
    monkeypatch.setattr(expand_literature, "seed_panelapp_publications", _refuse_seed)

    result = _expand(db_path, baseline_path)

    assert result.exit_code == 1
    assert "--cutoff-date is required without --scope-panel-id" in caplog.text


def test_scoped_expansion_needs_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db_path, _ = _run_with_a_gene_to_expand(tmp_path)
    monkeypatch.setattr(expand_literature, "seed_panelapp_publications", _refuse_seed)

    result = _expand(db_path, tmp_path / "missing.sqlite", "--scope-panel-id", "2")

    assert result.exit_code == 1
    assert "Baseline database not found" in caplog.text


def test_monthly_expansion_needs_every_run_gene_triaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    db_path, baseline_path = _run_with_a_gene_to_expand(tmp_path)
    monkeypatch.setattr(expand_literature, "seed_panelapp_publications", _refuse_seed)
    monkeypatch.setattr(expand_literature, "load_panels_with_reviews", lambda *_: {})
    monkeypatch.setattr(expand_literature.HgncResolver, "from_file", lambda: None)

    result = _expand(db_path, baseline_path, "--cutoff-date", "2026-10-01")

    assert result.exit_code == 1
    assert "1 run gene(s) have no expansion triage, e.g. HGNC:1" in caplog.text


def test_the_brief_names_each_pursued_association_and_its_curation() -> None:
    def association(status: str, rating: str | None, description: str) -> dict[str, Any]:
        return {
            "description": description,
            "inheritance_mode": "Biallelic",
            "panelapp_relation": {"status": status, "existing_rating": rating, "basis": "b"},
            "evidence_brief": f"earlier {description}",
        }

    brief = expand_literature.association_brief(
        [
            association("existing", "AMBER", "myopathy"),
            association("existing", None, "ataxia"),
            association("new_disease", None, "cardiomyopathy"),
            association("new_moi", None, "neuropathy"),
        ]
    )
    assert brief.splitlines() == [
        "- myopathy (Biallelic); PanelApp Australia: curated, rated AMBER. "
        "Earlier literature that settles it: earlier myopathy",
        "- ataxia (Biallelic); PanelApp Australia: curated, without a rating of its own. "
        "Earlier literature that settles it: earlier ataxia",
        "- cardiomyopathy (Biallelic); PanelApp Australia: not curated for this gene. "
        "Earlier literature that settles it: earlier cardiomyopathy",
        "- neuropathy (Biallelic); PanelApp Australia: curated only with another mode of "
        "inheritance. Earlier literature that settles it: earlier neuropathy",
    ]
