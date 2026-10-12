"""Tests for building PanelApp snapshots with reviews, against a mock API."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest

from palit.panelapp_client import EVALUATIONS_FETCHED_AT, PanelAppClient, snapshot_path
from palit.panelapp_snapshot import build_snapshot, version_on, write_snapshot


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
