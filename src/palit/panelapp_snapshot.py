"""Build the dated PanelApp snapshot that every stage of a run reads.

A snapshot, ``data/panelapp/{panel_date}.json``, holds every public panel that is
not a super-panel, at its latest version on or before the panel date, keyed by
panel ID. Each gene entry also carries its reviews as ``evaluations``: the
results of PanelApp's evaluations endpoint with their comments, most recent
first. PanelApp serves reviews only as they are now, and a curator's edit
overwrites a review's rating, phenotypes and MoI in place, so a panel's reviews
reflect the time they were fetched, which the panel records as
``evaluations_fetched_at``. Built on the panel date, as runs do on their setup
day, the reviews and the panel versions agree to within hours. An entry whose
reviews PanelApp cannot serve has ``evaluations`` null: a gene listed twice on a
panel, which the evaluations endpoint rejects with a server error, and a gene the
panel has dropped since the snapshot's version.

``snapshot`` builds a new snapshot. ``add-reviews`` adds the reviews to a
snapshot written before they were captured; its reviews are those of the day it
runs.
"""

import asyncio
import json
import logging
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx2
import tenacity
import typer

from palit.panelapp_client import DEFAULT_SNAPSHOT_DIR, EVALUATIONS_FETCHED_AT, snapshot_path
from palit.progress import LoggingProgress as Progress

logger = logging.getLogger(__name__)

app = typer.Typer(help="Build the dated PanelApp snapshot that a run reads")

PANELAPP_BASE_URL = "https://panelapp-aus.org/api/v1"
MAX_IN_FLIGHT = 8
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT_S = 60.0


def _is_transient(exc: BaseException) -> bool:
    """Rate limiting, server errors and transport failures are worth another attempt."""
    if isinstance(exc, httpx2.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx2.TransportError)


def _log_retry(state: tenacity.RetryCallState) -> None:
    if state.outcome is None:
        raise AssertionError("before_sleep runs after an attempt")
    logger.warning(
        "PanelApp request failed (%s); attempt %d/%d in %.1f s",
        state.outcome.exception(),
        state.attempt_number + 1,
        MAX_ATTEMPTS,
        state.upcoming_sleep,
    )


class PanelAppFetcher:
    """GET requests to the PanelApp API, MAX_IN_FLIGHT at a time, retrying transient failures."""

    def __init__(self, http: httpx2.AsyncClient) -> None:
        self._http = http
        self._slots = asyncio.Semaphore(MAX_IN_FLIGHT)

    async def get(self, url: str) -> Any:
        """The JSON body of *url*, a path below the API root or an absolute URL."""
        async with self._slots:
            async for attempt in tenacity.AsyncRetrying(
                stop=tenacity.stop_after_attempt(MAX_ATTEMPTS),
                wait=tenacity.wait_exponential_jitter(initial=2, max=60),
                retry=tenacity.retry_if_exception(_is_transient),
                before_sleep=_log_retry,
                reraise=True,
            ):
                with attempt:
                    response = await self._http.get(url)
                    response.raise_for_status()
                    return response.json()
        raise AssertionError("tenacity returns or re-raises")


def _http_client(transport: httpx2.AsyncBaseTransport | None = None) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(
        base_url=PANELAPP_BASE_URL, timeout=REQUEST_TIMEOUT_S, transport=transport
    )


async def fetch_panel_ids(fetcher: PanelAppFetcher) -> list[int]:
    """The IDs of all public panels that are not super-panels."""
    panel_ids: list[int] = []
    url: str | None = "/panels/"
    while url is not None:
        page = await fetcher.get(url)
        panel_ids.extend(
            panel["id"]
            for panel in page["results"]
            if panel["status"] == "public" and not panel["child_panel_ids"]
        )
        url = page["next"]
    return panel_ids


def version_on(activities: list[dict[str, Any]], cutoff: datetime) -> str | None:
    """The latest panel version created on or before *cutoff*; None if there was none yet."""
    version = None
    for activity in sorted(activities, key=lambda a: a["created"]):
        if datetime.fromisoformat(activity["created"].replace("Z", "+00:00")) > cutoff:
            break
        if activity.get("panel_version") and activity["panel_version"] != "0.0":
            version = activity["panel_version"]
    return version


async def fetch_panel(
    fetcher: PanelAppFetcher, panel_id: int, cutoff: datetime
) -> dict[str, Any] | None:
    """The panel at its version on *cutoff*; None for a panel created after it."""
    activities = await fetcher.get(f"/panels/{panel_id}/activities/")
    version = version_on(activities, cutoff)
    if version is None:
        logger.debug("Skipping panel %d: no version on or before %s", panel_id, cutoff.date())
        return None
    panel: dict[str, Any] = await fetcher.get(f"/panels/{panel_id}/?version={version}")
    return panel


async def fetch_reviews(
    fetcher: PanelAppFetcher, panel_id: int, hgnc_id: str
) -> list[dict[str, Any]] | None:
    """A gene's current reviews on a panel, most recent first; None if the panel no
    longer holds the gene."""
    try:
        page = await fetcher.get(
            f"/panels/{panel_id}/genes/{hgnc_id}/evaluations/?include_comments=true"
        )
    except httpx2.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None
        raise
    if page["next"] is not None:
        raise ValueError(
            f"{hgnc_id} on panel {panel_id} has more than one page of reviews "
            f"({page['count']}); the snapshot reads one page"
        )
    reviews: list[dict[str, Any]] = page["results"]
    return sorted(reviews, key=lambda review: review["created"], reverse=True)


async def add_reviews(fetcher: PanelAppFetcher, panels: dict[str, dict[str, Any]]) -> None:
    """Set the ``evaluations`` of every gene entry and the ``evaluations_fetched_at`` of
    every panel in *panels*, in place.

    The ``evaluations`` of a gene listed twice on a panel, and of a gene the panel
    has dropped since the snapshot's version, are None.
    """
    duplicated = {
        (panel_id, hgnc_id)
        for panel_id, panel in panels.items()
        for hgnc_id, count in Counter(e["gene_data"]["hgnc_id"] for e in panel["genes"]).items()
        if count > 1
    }
    if duplicated:
        logger.warning(
            "%d genes are listed twice on a panel, which PanelApp cannot serve reviews for: %s",
            len(duplicated),
            ", ".join(f"{hgnc_id} on panel {panel_id}" for panel_id, hgnc_id in sorted(duplicated)),
        )
    removed: list[str] = []
    with Progress() as progress:
        task = progress.add_task(
            "Fetching reviews", total=sum(len(panel["genes"]) for panel in panels.values())
        )

        async def entry_reviews(panel_id: int, entry: dict[str, Any]) -> None:
            hgnc_id = entry["gene_data"]["hgnc_id"]
            if (str(panel_id), hgnc_id) in duplicated:
                entry["evaluations"] = None
            else:
                entry["evaluations"] = await fetch_reviews(fetcher, panel_id, hgnc_id)
                if entry["evaluations"] is None:
                    removed.append(f"{hgnc_id} on panel {panel_id}")
            progress.advance(task)

        async def panel_reviews(panel_id: int, panel: dict[str, Any]) -> None:
            await asyncio.gather(*(entry_reviews(panel_id, entry) for entry in panel["genes"]))
            panel[EVALUATIONS_FETCHED_AT] = datetime.now(UTC).isoformat(timespec="seconds")

        await asyncio.gather(
            *(panel_reviews(int(panel_id), panel) for panel_id, panel in panels.items())
        )
    if removed:
        logger.warning(
            "%d genes are no longer on their panel, so their reviews are unavailable: %s",
            len(removed),
            ", ".join(sorted(removed)),
        )


async def build_snapshot(
    panel_date: str, transport: httpx2.AsyncBaseTransport | None = None
) -> dict[str, dict[str, Any]]:
    """Every public panel at its version on *panel_date*, with its genes' current reviews,
    keyed by panel ID as a string."""
    cutoff = datetime.fromisoformat(panel_date).replace(tzinfo=UTC)
    async with _http_client(transport) as http:
        fetcher = PanelAppFetcher(http)
        panel_ids = await fetch_panel_ids(fetcher)
        logger.info("Fetching %d panels at their version on %s", len(panel_ids), panel_date)
        fetched = await asyncio.gather(
            *(fetch_panel(fetcher, panel_id, cutoff) for panel_id in panel_ids)
        )
        panels = {
            str(panel_id): panel
            for panel_id, panel in zip(panel_ids, fetched, strict=True)
            if panel is not None
        }
        await add_reviews(fetcher, panels)
    return panels


def _warn_if_not_today(panel_date: str) -> None:
    today = date.today().isoformat()
    if panel_date != today:
        logger.warning(
            "Capturing reviews on %s for the %s snapshot: they are the reviews of %s",
            today,
            panel_date,
            today,
        )


def write_snapshot(path: Path, panels: dict[str, dict[str, Any]]) -> None:
    """Write *panels* to *path* through a temporary file, so a failed write leaves no
    partial snapshot."""
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w") as f:
        json.dump(panels, f, indent=2)
    partial.replace(path)
    logger.info("Wrote %d panels to %s", len(panels), path)


PANEL_DATE_OPTION = typer.Option(
    ..., "--panel-date", help="Date of the snapshot (YYYY-MM-DD), normally the run's setup day"
)
SNAPSHOT_DIR_OPTION = typer.Option(
    DEFAULT_SNAPSHOT_DIR, "--snapshot-dir", help="Directory holding the snapshots"
)


@app.command()
def snapshot(panel_date: str = PANEL_DATE_OPTION, snapshot_dir: Path = SNAPSHOT_DIR_OPTION) -> None:
    """Fetch every panel at its version on the panel date, with its genes' current reviews."""
    path = snapshot_path(snapshot_dir, panel_date)
    if path.exists():
        logger.error("%s exists; move it aside to build the snapshot again", path)
        raise typer.Exit(1)
    _warn_if_not_today(panel_date)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    write_snapshot(path, asyncio.run(build_snapshot(panel_date)))


@app.command("add-reviews")
def add_reviews_command(
    panel_date: str = PANEL_DATE_OPTION, snapshot_dir: Path = SNAPSHOT_DIR_OPTION
) -> None:
    """Add today's reviews to a snapshot built without them."""
    path = snapshot_path(snapshot_dir, panel_date)
    panels: dict[str, dict[str, Any]] = json.loads(path.read_text())
    captured = [panel_id for panel_id, panel in panels.items() if EVALUATIONS_FETCHED_AT in panel]
    if captured:
        logger.error("%s already has reviews on %d panels", path, len(captured))
        raise typer.Exit(1)
    _warn_if_not_today(panel_date)

    async def run() -> None:
        async with _http_client() as http:
            await add_reviews(PanelAppFetcher(http), panels)

    asyncio.run(run())
    write_snapshot(path, panels)
