"""Tests for building PanelApp snapshots with reviews, against a mock API."""

import asyncio
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
from click.testing import Result
from typer.testing import CliRunner

from palit import panelapp_snapshot
from palit.panelapp_client import (
    EVALUATIONS_FETCHED_AT,
    PanelAppClient,
    format_panel_for_prompt,
    snapshot_path,
)
from palit.panelapp_snapshot import app, build_snapshot, version_on, write_snapshot


def _gene(hgnc_id: str) -> dict[str, Any]:
    return {"gene_data": {"hgnc_id": hgnc_id}, "entity_name": hgnc_id, "confidence_level": "3"}


def _review(created: str, rating: str) -> dict[str, Any]:
    return {"created": created, "rating": rating, "comments": []}


PANEL_LIST = {
    "/api/v1/panels/": {
        "results": [
            {"id": 1, "status": "public", "child_panel_ids": []},
            {"id": 2, "status": "retired", "child_panel_ids": []},
        ],
        "next": "https://panelapp-aus.org/api/v1/panels/?page=2",
    },
    "/api/v1/panels/?page=2": {
        "results": [
            {"id": 3, "status": "public", "child_panel_ids": [1]},
            {"id": 4, "status": "public", "child_panel_ids": []},
        ],
        "next": None,
    },
}


class FakePanelApp:
    """The few PanelApp endpoints the snapshot reads, with canned responses."""

    def __init__(self, reviews: dict[str, Any]) -> None:
        self.reviews = reviews  # evaluations path -> response body, or None for a 404
        self.requested: list[str] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.raw_path.decode()
        self.requested.append(path)
        if path in PANEL_LIST:
            return httpx2.Response(200, json=PANEL_LIST[path])
        if path == "/api/v1/panels/1/activities/":
            return httpx2.Response(
                200,
                json=[
                    {"created": "2026-01-01T00:00:00Z", "panel_version": "1.0"},
                    {"created": "2026-03-01T00:00:00Z", "panel_version": "1.5"},
                    {"created": "2026-06-01T00:00:00Z", "panel_version": "2.0"},
                ],
            )
        if path == "/api/v1/panels/4/activities/":
            return httpx2.Response(
                200, json=[{"created": "2026-09-01T00:00:00Z", "panel_version": "0.1"}]
            )
        if path == "/api/v1/panels/1/?version=1.5":
            return httpx2.Response(
                200,
                json={
                    "id": 1,
                    "genes": [
                        _gene("HGNC:10"),
                        _gene("HGNC:20"),
                        _gene("HGNC:30"),
                        _gene("HGNC:30"),
                    ],
                },
            )
        if path in self.reviews:
            body = self.reviews[path]
            return httpx2.Response(404) if body is None else httpx2.Response(200, json=body)
        raise AssertionError(f"unexpected request {path}")


def _reviews_path(panel_id: int, hgnc_id: str) -> str:
    return f"/api/v1/panels/{panel_id}/genes/{hgnc_id}/evaluations/?include_comments=true"


def _build(api: FakePanelApp) -> dict[str, dict[str, Any]]:
    return asyncio.run(build_snapshot("2026-04-01", transport=httpx2.MockTransport(api.handle)))


def test_version_on_takes_the_latest_version_up_to_the_cutoff() -> None:
    activities = [
        {"created": "2026-03-01T00:00:00Z", "panel_version": "1.5"},
        {"created": "2026-01-01T00:00:00Z", "panel_version": "0.0"},
        {"created": "2026-02-01T00:00:00Z"},
        {"created": "2026-06-01T00:00:00Z", "panel_version": "2.0"},
    ]
    cutoff = datetime(2026, 4, 1, tzinfo=UTC)
    assert version_on(activities, cutoff) == "1.5"
    assert version_on(activities, datetime(2026, 1, 15, tzinfo=UTC)) is None


def test_snapshot_holds_public_panels_at_their_version_with_reviews() -> None:
    api = FakePanelApp(
        {
            _reviews_path(1, "HGNC:10"): {
                "count": 2,
                "next": None,
                "results": [
                    _review("2020-01-01T00:00:00Z", "AMBER"),
                    _review("2025-01-01T00:00:00Z", "GREEN"),
                ],
            },
            _reviews_path(1, "HGNC:20"): None,
        }
    )
    panels = _build(api)

    # Retired panel 2 and super-panel 3 are left out; panel 4 did not exist yet.
    assert list(panels) == ["1"]
    panel = panels["1"]
    assert [r["rating"] for r in panel["genes"][0]["evaluations"]] == ["GREEN", "AMBER"]
    # PanelApp cannot serve the reviews of a gene the panel has dropped since that
    # version, nor of one listed twice, which is never requested.
    assert panel["genes"][1]["evaluations"] is None
    assert [g["evaluations"] for g in panel["genes"][2:]] == [None, None]
    assert not any("HGNC:30" in path for path in api.requested)
    datetime.fromisoformat(panel[EVALUATIONS_FETCHED_AT])


def test_a_second_page_of_reviews_raises() -> None:
    api = FakePanelApp(
        {
            _reviews_path(1, "HGNC:10"): {"count": 101, "next": "more", "results": []},
            _reviews_path(1, "HGNC:20"): {"count": 0, "next": None, "results": []},
        }
    )
    with pytest.raises(ValueError, match="more than one page"):
        _build(api)


def _write(tmp_path: Path, panels: dict[str, dict[str, Any]]) -> PanelAppClient:
    write_snapshot(snapshot_path(tmp_path, "2026-04-01"), panels)
    return PanelAppClient("2026-04-01", snapshot_dir=tmp_path)


def test_client_reads_reviews_from_the_snapshot(tmp_path: Path) -> None:
    gene = _gene("HGNC:10") | {"evaluations": [_review("2025-01-01T00:00:00Z", "GREEN")]}
    client = _write(
        tmp_path,
        {"1": {"id": 1, "genes": [gene], EVALUATIONS_FETCHED_AT: "2026-04-01T09:00:00+00:00"}},
    )
    assert client.gene_reviews(1, 10) == [_review("2025-01-01T00:00:00Z", "GREEN")]
    assert client.reviews_fetched_at(1) == "2026-04-01T09:00:00+00:00"
    with pytest.raises(ValueError, match="does not hold"):
        client.gene_reviews(1, 99)


def test_client_returns_none_for_reviews_panelapp_could_not_serve(tmp_path: Path) -> None:
    gene = _gene("HGNC:10") | {"evaluations": None}
    client = _write(
        tmp_path,
        {"1": {"id": 1, "genes": [gene], EVALUATIONS_FETCHED_AT: "2026-04-01T09:00:00+00:00"}},
    )
    assert client.holds_gene(1, 10)
    assert client.gene_reviews(1, 10) is None
    assert not client.holds_gene(1, 99)


def test_client_rejects_a_snapshot_without_reviews(tmp_path: Path) -> None:
    client = _write(tmp_path, {"1": {"id": 1, "genes": [_gene("HGNC:10")]}})
    with pytest.raises(KeyError, match="add-reviews"):
        client.gene_reviews(1, 10)


def test_client_needs_a_built_snapshot(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="palit panelapp snapshot"):
        PanelAppClient("2026-04-01", snapshot_dir=tmp_path).get_panel_data(1)


def test_written_snapshot_leaves_no_partial_file(tmp_path: Path) -> None:
    path = snapshot_path(tmp_path, "2026-04-01")
    write_snapshot(path, {"1": {"id": 1, "genes": []}})
    assert json.loads(path.read_text()) == {"1": {"id": 1, "genes": []}}
    assert [p.name for p in tmp_path.iterdir()] == ["2026-04-01.json"]


def test_panel_prompt_restores_relevant_disorders_split_at_commas() -> None:
    panel = {
        "name": "Leukodystrophy",
        "relevant_disorders": [
            "Leukodystrophy",
            "HP:0002415; Abnormal CNS myelination",
            "HP:0011400",
        ],
        "disease_group": "Neurology and neurodevelopmental disorders",
        "disease_sub_group": "",
        "description": "Genes that cause leukodystrophies.",
    }
    assert format_panel_for_prompt(298, panel) == (
        '<panel id="298">\n'
        "Name: Leukodystrophy\n"
        "Relevant disorders: Leukodystrophy, HP:0002415; Abnormal CNS myelination, HP:0011400\n"
        "Disease group: Neurology and neurodevelopmental disorders\n"
        "Description: Genes that cause leukodystrophies.\n"
        "</panel>"
    )


def _refuse_ingest(db_path: Path, panels: dict[int, dict[str, Any]], panel_date: str) -> None:
    raise AssertionError("ingest-citations should have stopped before the intake")


def _ingest_citations(tmp_path: Path, db_path: Path) -> Result:
    return CliRunner().invoke(
        app,
        [
            "ingest-citations",
            "--panel-date",
            "2026-04-01",
            "--snapshot-dir",
            str(tmp_path),
            "--db-path",
            str(db_path),
        ],
    )


def test_ingest_citations_needs_the_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(panelapp_snapshot, "ingest_snapshot_citations", _refuse_ingest)
    write_snapshot(
        snapshot_path(tmp_path, "2026-04-01"),
        {"1": {"id": 1, "genes": [], "strs": [], EVALUATIONS_FETCHED_AT: "2026-04-01T09:00:00"}},
    )

    result = _ingest_citations(tmp_path, tmp_path / "missing.sqlite")

    assert result.exit_code == 1
    assert "Database not found" in caplog.text


def test_ingest_citations_needs_a_snapshot_with_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(panelapp_snapshot, "ingest_snapshot_citations", _refuse_ingest)
    db_path = tmp_path / "baseline.sqlite"
    sqlite3.connect(db_path).close()
    write_snapshot(
        snapshot_path(tmp_path, "2026-04-01"),
        {
            "1": {"id": 1, "genes": [], "strs": [], EVALUATIONS_FETCHED_AT: "2026-04-01T09:00:00"},
            "2": {"id": 2, "genes": [_gene("HGNC:10")], "strs": []},
        },
    )

    result = _ingest_citations(tmp_path, db_path)

    assert result.exit_code == 1
    assert "no reviews on 1 panels" in caplog.text


def _seed_citations(tmp_path: Path, db_path: Path, baseline_path: Path) -> Result:
    return CliRunner().invoke(
        app,
        [
            "seed-citations",
            "--panel-date",
            "2026-04-01",
            "--snapshot-dir",
            str(tmp_path),
            "--db-path",
            str(db_path),
            "--baseline-db-path",
            str(baseline_path),
        ],
    )


def test_seed_citations_seeds_the_run_from_every_panel_and_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Path, list[int], Path]] = []

    def seed(db_path: Path, panels: dict[int, dict[str, Any]], baseline_path: Path, _: Any) -> None:
        calls.append((db_path, sorted(panels), baseline_path))

    monkeypatch.setattr(panelapp_snapshot, "seed_panelapp_publications", seed)
    monkeypatch.setattr(panelapp_snapshot.HgncResolver, "from_file", lambda: None)
    db_path = tmp_path / "run.sqlite"
    baseline_path = tmp_path / "baseline.sqlite"
    for path in (db_path, baseline_path):
        sqlite3.connect(path).close()
    fetched_at = {EVALUATIONS_FETCHED_AT: "2026-04-01T09:00:00"}
    write_snapshot(
        snapshot_path(tmp_path, "2026-04-01"),
        {
            "1": {"id": 1, "genes": [], "strs": [], **fetched_at},
            "2": {"id": 2, "genes": [], "strs": [], **fetched_at},
        },
    )

    result = _seed_citations(tmp_path, db_path, baseline_path)

    assert result.exit_code == 0, result.output
    assert calls == [(db_path, [1, 2], baseline_path)]


def test_seed_citations_needs_the_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(panelapp_snapshot, "seed_panelapp_publications", _refuse_ingest)
    db_path = tmp_path / "run.sqlite"
    sqlite3.connect(db_path).close()

    result = _seed_citations(tmp_path, db_path, tmp_path / "missing.sqlite")

    assert result.exit_code == 1
    assert "Database not found" in caplog.text
