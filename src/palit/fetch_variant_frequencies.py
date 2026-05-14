#!/usr/bin/env python3
"""Look up variant frequencies via the variant-lookup service.

One HTTPS request per extracted variant against the gateway's
``POST /v1/variant`` endpoint. Concurrency is bounded by a semaphore
(default 8, matching the service's nginx ``limit_conn`` cap).
Variants are sorted by gene chromosome before issuing requests so
consecutive calls hit the same per-chromosome echtvar archive and
the same Mutalyzer reference-sequence cache.
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import tenacity
import typer
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.progress import (
    BarColumn,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from palit.hgnc import HgncResolver
from palit.progress import LoggingProgress as Progress

logger = logging.getLogger(__name__)

app = typer.Typer(help="Look up variant frequencies via the variant-lookup service")


# Per-request timeout. Mutalyzer's first-touch on a cold reference-sequence
# cache can take well over a minute (NCBI E-fetch + back-translate fan-out).
# Steady-state requests finish in ~2-3s; we trade a few wasted seconds on
# truly stuck calls for not dropping legitimate cold-cache traffic.
_REQUEST_TIMEOUT_S = 180.0

# Default in-flight cap on the client side. The service enforces 8
# cluster-wide via nginx limit_conn; staying at parity avoids burning
# tenacity retries on 429s in the common case.
_DEFAULT_MAX_IN_FLIGHT = 8


@dataclass
class ExtractedVariant:
    """A variant extracted from evidence extraction."""

    hgnc_id: int
    hgnc_symbol: str
    variant_text: str
    genome_build: str | None  # From evidence extraction; None if unknown
    box_id: int


@dataclass
class VariantFrequencyResult:
    """One row destined for the ``variant_frequencies`` table."""

    variant_id: str  # pseudo-VCF when normalization succeeded; original_text on failure
    hgnc_id: int
    doi: str
    box_id: int
    normalization: dict[str, Any]
    gnomad: dict[str, Any]


# ---------------------------------------------------------------------------
# Settings — pydantic-settings, .env is read automatically.
# ---------------------------------------------------------------------------


class VariantLookupSettings(BaseSettings):
    """Loaded from environment + .env at command startup.

    Fails the command with a clean message if either variable is missing —
    see ``_load_settings`` below.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_url: str = Field(..., alias="VARIANT_LOOKUP_BASE_URL")
    api_key: str = Field(..., alias="VARIANT_LOOKUP_API_KEY")


def _load_settings() -> VariantLookupSettings:
    try:
        return VariantLookupSettings()  # type: ignore[call-arg]
    except ValidationError as e:
        missing = sorted({str(err["loc"][0]) for err in e.errors() if err["type"] == "missing"})
        logger.error(
            "Required environment variables not set: %s. See README.md § 'External services'.",
            ", ".join(missing),
        )
        raise typer.Exit(1) from e


# ---------------------------------------------------------------------------
# Async HTTP client.
# ---------------------------------------------------------------------------


def _is_5xx_or_transport(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


class VariantLookupClient:
    """Async wrapper around ``POST /v1/variant``.

    Concurrency: ``max_in_flight`` semaphore. 429s honour ``Retry-After``
    (handled in-band); 5xx and transport errors retry via tenacity.
    """

    def __init__(self, settings: VariantLookupSettings, max_in_flight: int) -> None:
        self._endpoint = f"{settings.base_url.rstrip('/')}/v1/variant"
        self._headers = {"Authorization": f"Bearer {settings.api_key}"}
        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._http = httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S, follow_redirects=True)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def lookup_one(self, body: dict[str, Any]) -> dict[str, Any]:
        async with self._semaphore:
            return await self._post_with_retries(body)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(7),
        wait=tenacity.wait_exponential_jitter(initial=1, max=30),
        retry=tenacity.retry_if_exception(_is_5xx_or_transport),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
    )
    async def _post_with_retries(self, body: dict[str, Any]) -> dict[str, Any]:
        # Inner 429 loop: nginx limit_conn returns 429 + Retry-After when the
        # cluster cap is exceeded; honour the header rather than backing off
        # blindly. Bounded so a misbehaving service doesn't spin forever.
        for _ in range(10):
            response = await self._http.post(self._endpoint, json=body, headers=self._headers)
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()  # type: ignore[no-any-return]
            retry_after = float(response.headers.get("Retry-After", "1.0"))
            await asyncio.sleep(retry_after)
        # Still 429 after 10 honoured retries — let tenacity see the 5xx-shaped
        # error so the caller's resumable rerun picks it up later.
        response.raise_for_status()
        raise RuntimeError("unreachable: 429 loop fell through without raising")


# ---------------------------------------------------------------------------
# DB helpers (extraction → storage).
# ---------------------------------------------------------------------------


def load_extracted_variants(
    db_path: Path, hgnc_resolver: HgncResolver
) -> dict[str, list[ExtractedVariant]]:
    """Load variants from evidence extractions, grouped by paper DOI."""
    logger.info(f"Loading extracted variants from {db_path}...")
    variants_by_doi: dict[str, list[ExtractedVariant]] = {}

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT doi, evidence_extraction_json
            FROM papers
            WHERE evidence_extraction_json IS NOT NULL
            ORDER BY doi DESC
        """)

        for row in cursor.fetchall():
            doi = row["doi"]
            try:
                extraction_data = json.loads(row["evidence_extraction_json"])
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse evidence extraction for DOI {doi}")
                continue

            genome_build: str | None = extraction_data.get("genome_build")
            if genome_build == "unknown":
                genome_build = None

            for eval_data in extraction_data.get("gene_evaluations", []):
                hgnc_id = eval_data.get("hgnc_id")
                if hgnc_id is None:
                    continue
                hgnc_symbol = hgnc_resolver.get_symbol(hgnc_id)
                for variant_entry in eval_data.get("variants", []):
                    variant_text = variant_entry.get("variant")
                    box_id = variant_entry.get("box_id")
                    if variant_text and box_id is not None:
                        variants_by_doi.setdefault(doi, []).append(
                            ExtractedVariant(
                                hgnc_id=hgnc_id,
                                hgnc_symbol=hgnc_symbol,
                                variant_text=variant_text,
                                genome_build=genome_build,
                                box_id=box_id,
                            )
                        )

    total = sum(len(v) for v in variants_by_doi.values())
    logger.info(f"Loaded {total} variants from {len(variants_by_doi)} papers")
    return variants_by_doi


def get_processed_variant_keys(db_path: Path) -> set[tuple[str, int, str]]:
    """Return ``(paper_doi, box_id, original_text)`` triples already stored.

    A variant counts as processed once any row exists for it (including a
    ``normalization_error`` sentinel). Use ``--retry-errors`` to re-run
    error rows; a plain rerun fills in only variants that have no row at
    all, so partially-processed papers with newly extracted variants get
    cheaply backfilled without churning rows that already landed.
    """
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT paper_doi, box_id, json_extract(normalization, '$.original_text')
            FROM variant_frequencies
        """)
        return {(doi, box_id, original_text) for doi, box_id, original_text in cursor.fetchall()}


def store_results_for_doi(
    doi: str,
    results: list[VariantFrequencyResult],
    db_path: Path,
    old_ids_to_delete: list[int],
) -> None:
    """Persist a paper's results atomically.

    ``old_ids_to_delete`` carries the IDs of previously-errored rows that
    this batch is replacing (retry-errors mode); they are deleted in the
    same transaction as the inserts. Per-paper atomicity means a crash
    while another paper is still resolving leaves that paper's old rows
    in place, so the next retry-errors run can pick them up.

    Two layers of dedup on ``(variant_id, paper_doi, box_id)``:

    1. Within-batch: a paper can cite the same variant via different aliases
       at the same box (e.g. ``c.770C>T`` and ``p.Ser257Leu``); we keep the
       first.
    2. Against the DB via ``INSERT OR IGNORE``: a retry-errors run can
       re-resolve a previously-errored alias to a pseudo-VCF that another
       alias already landed under in an earlier run, and a re-extraction
       can add a new alias for an already-stored variant. Both cases mean
       the row is already captured — silently no-op.
    """
    if not results and not old_ids_to_delete:
        logger.debug(f"No results to store for DOI {doi}")
        return

    seen_keys: set[tuple[str, str, int]] = set()
    unique: list[VariantFrequencyResult] = []
    for r in results:
        key = (r.variant_id, r.doi, r.box_id)
        if key not in seen_keys:
            seen_keys.add(key)
            unique.append(r)
    duplicates = len(results) - len(unique)
    if duplicates:
        logger.debug(f"Skipped {duplicates} duplicate entries for DOI {doi}")

    inserted = 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        if old_ids_to_delete:
            placeholders = ",".join("?" * len(old_ids_to_delete))
            cursor.execute(
                f"DELETE FROM variant_frequencies WHERE id IN ({placeholders})",
                old_ids_to_delete,
            )
        for r in unique:
            cursor.execute(
                """
                INSERT OR IGNORE INTO variant_frequencies
                  (variant_id, hgnc_id, paper_doi, box_id, normalization, gnomad)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    r.variant_id,
                    r.hgnc_id,
                    r.doi,
                    r.box_id,
                    json.dumps(r.normalization),
                    json.dumps(r.gnomad),
                ),
            )
            inserted += cursor.rowcount
        conn.commit()
    ignored = len(unique) - inserted
    if ignored:
        logger.debug(
            f"DOI {doi}: stored {inserted} new rows, skipped {ignored} already present "
            f"under a different alias at the same box"
        )
    else:
        logger.debug(f"Stored {inserted} variant frequency results for DOI {doi}")


# ---------------------------------------------------------------------------
# Per-variant resolution: request body → response → max-AC pick → DB row.
# ---------------------------------------------------------------------------


def _build_request_body(ev: ExtractedVariant) -> dict[str, Any]:
    body: dict[str, Any] = {
        "gene": ev.hgnc_symbol,
        "variant": ev.variant_text,
    }
    if ev.genome_build is not None:
        body["genome_build"] = ev.genome_build
    return body


def _pick_max_ac_normalized(normalized: list[dict[str, Any]]) -> tuple[dict[str, Any], int | None]:
    """Pick the back-translation with the highest gnomAD AC.

    Tiebreak: lexicographically first pseudo_vcf — preserves the old
    ``fetch_variant_frequencies.py:289-291`` deterministic-on-tie rule.
    Returns the selected entry plus the AC that drove the choice (None
    when every translation is "not found in gnomAD").
    """
    best: dict[str, Any] | None = None
    best_ac = -1
    for entry in normalized:
        freq = entry.get("frequency")
        ac = freq["ac"] if freq is not None else -1
        if ac > best_ac or (
            ac == best_ac and (best is None or entry["pseudo_vcf"] < best["pseudo_vcf"])
        ):
            best = entry
            best_ac = ac
    assert best is not None, "_pick_max_ac_normalized called with empty list"
    return best, (best_ac if best_ac >= 0 else None)


def _result_from_response(
    ev: ExtractedVariant, doi: str, response: dict[str, Any]
) -> VariantFrequencyResult:
    error = response.get("error")
    if error is not None:
        return VariantFrequencyResult(
            variant_id=ev.variant_text,
            hgnc_id=ev.hgnc_id,
            doi=doi,
            box_id=ev.box_id,
            normalization={
                "original_text": ev.variant_text,
                "error_code": error.get("code", ""),
                "error_message": error.get("message", ""),
                "upstream": error.get("upstream"),
            },
            gnomad={"normalization_error": True},
        )

    normalized = response.get("normalized") or []
    if not normalized:
        return VariantFrequencyResult(
            variant_id=ev.variant_text,
            hgnc_id=ev.hgnc_id,
            doi=doi,
            box_id=ev.box_id,
            normalization={
                "original_text": ev.variant_text,
                "error_code": "NO_NORMALIZED_VARIANTS",
                "error_message": "service returned an empty normalized[] list",
                "upstream": None,
            },
            gnomad={"normalization_error": True},
        )

    picked, selected_ac = _pick_max_ac_normalized(normalized)
    freq = picked.get("frequency")
    gnomad_payload: dict[str, Any] = freq if freq is not None else {"variant_not_found": True}

    return VariantFrequencyResult(
        variant_id=picked["pseudo_vcf"],
        hgnc_id=ev.hgnc_id,
        doi=doi,
        box_id=ev.box_id,
        normalization={
            "hgvs_c": picked.get("hgvs_c"),
            "hgvs_p": picked.get("hgvs_p"),
            "original_text": ev.variant_text,
            "total_normalizations": len(normalized),
            "selected_for_max_ac": selected_ac,
        },
        gnomad=gnomad_payload,
    )


# ---------------------------------------------------------------------------
# Async fan-out + per-paper flush.
# ---------------------------------------------------------------------------


@dataclass
class _PaperBucket:
    """In-memory state tracking a paper's in-flight variants."""

    remaining: int
    results: list[VariantFrequencyResult] = field(default_factory=list)


def _chromosome_sort_key(
    ev: ExtractedVariant, hgnc_resolver: HgncResolver
) -> tuple[int, str, int, int]:
    """Sort key for backend cache locality.

    Tuple of (chromosome rank, chromosome label, hgnc_id, box_id):
    - chromosome rank groups same-chromosome variants (echtvar archive +
      MANE-select transcript cache locality)
    - hgnc_id within a chromosome groups same-gene variants (Mutalyzer
      reference-sequence cache locality)
    - box_id final tiebreaker for determinism
    """
    chrom = hgnc_resolver.get_chromosome(ev.hgnc_id)
    # Numeric chromosomes first (1..22), then X, then Y, then MT, then unknown.
    if chrom is None:
        rank = 1000
        label = ""
    elif chrom.isdigit():
        rank = int(chrom)
        label = chrom
    elif chrom == "X":
        rank = 100
        label = chrom
    elif chrom == "Y":
        rank = 101
        label = chrom
    elif chrom == "MT":
        rank = 102
        label = chrom
    else:
        rank = 999
        label = chrom
    return (rank, label, ev.hgnc_id, ev.box_id)


async def _resolve_all(
    flat_sorted: list[tuple[str, ExtractedVariant]],
    client: VariantLookupClient,
    db_path: Path,
    progress: Progress,
    task_id: TaskID,
    old_ids_by_key: dict[tuple[str, int, str], int],
) -> list[VariantFrequencyResult]:
    """Issue one ``/v1/variant`` request per variant, flush each paper
    atomically as its last variant lands, return all results for stats.

    ``old_ids_by_key`` maps each variant's ``(doi, box_id, original_text)``
    to the existing errored row's ID (retry-errors mode; empty otherwise).
    Each per-paper flush deletes the IDs whose lookup actually produced a
    result and inserts the new rows in one transaction. Lookups that ended
    in a transport error contribute neither a result nor a delete, so the
    old errored row stays for the next retry-errors run.
    """
    pending: dict[str, _PaperBucket] = {}
    for doi, _ in flat_sorted:
        bucket = pending.setdefault(doi, _PaperBucket(remaining=0))
        bucket.remaining += 1

    all_results: list[VariantFrequencyResult] = []

    def _flush(doi: str) -> None:
        bucket = pending[doi]
        old_ids = [
            old_ids_by_key[(doi, r.box_id, r.normalization["original_text"])]
            for r in bucket.results
            if (doi, r.box_id, r.normalization["original_text"]) in old_ids_by_key
        ]
        store_results_for_doi(doi, bucket.results, db_path, old_ids)
        all_results.extend(bucket.results)
        del pending[doi]
        progress.update(task_id, advance=1, description=f"[green]Completed DOI {doi}")

    async def _one(doi: str, ev: ExtractedVariant) -> None:
        try:
            body = _build_request_body(ev)
            response = await client.lookup_one(body)
            result = _result_from_response(ev, doi, response)
        except (httpx.HTTPError, tenacity.RetryError) as e:
            logger.warning(
                "Variant %r for gene %s in DOI %s failed after retries: %s: %s",
                ev.variant_text,
                ev.hgnc_symbol,
                doi,
                type(e).__name__,
                e,
            )
            # Leave the row absent so the next resumable rerun retries it.
            bucket = pending[doi]
            bucket.remaining -= 1
            if bucket.remaining == 0:
                _flush(doi)
            return

        bucket = pending[doi]
        bucket.results.append(result)
        bucket.remaining -= 1
        if bucket.remaining == 0:
            _flush(doi)

    tasks = [asyncio.create_task(_one(doi, ev)) for doi, ev in flat_sorted]
    await asyncio.gather(*tasks)
    return all_results


# ---------------------------------------------------------------------------
# Stats + retry-errors path.
# ---------------------------------------------------------------------------


def print_summary_statistics(results: list[VariantFrequencyResult]) -> None:
    total = len(results)
    normalization_errors = sum(1 for r in results if r.gnomad.get("normalization_error"))
    successful = total - normalization_errors

    found = 0
    not_found = 0
    for r in results:
        if r.gnomad.get("normalization_error"):
            continue
        if r.gnomad.get("variant_not_found"):
            not_found += 1
        elif "ac" in r.gnomad:
            found += 1

    print("\n" + "=" * 60)
    print("VARIANT FREQUENCY LOOKUP SUMMARY")
    print("=" * 60)
    print(f"Total variants processed: {total}")
    if total > 0:
        print(f"Successfully normalized: {successful} ({successful / total * 100:.1f}%)")
        print(
            f"Failed normalization: {normalization_errors} "
            f"({normalization_errors / total * 100:.1f}%)"
        )

    print("\ngnomAD v4 lookup results:")
    print(f"  Found in gnomAD: {found}")
    print(f"  Not found in gnomAD: {not_found}")

    high_freq = [
        (r.variant_id, r.gnomad["ac"], r.hgnc_id)
        for r in results
        if "ac" in r.gnomad and r.gnomad.get("ac", 0) > 30
    ]
    if high_freq:
        print(f"\n⚠️  High-frequency variants (AC > 30): {len(high_freq)}")
        for vid, ac, hgnc_id in high_freq[:5]:
            print(f"    HGNC:{hgnc_id}: {vid} (AC={ac})")
        if len(high_freq) > 5:
            print(f"    ... and {len(high_freq) - 5} more")


async def _retry_errored_variants(
    db_path: Path, client: VariantLookupClient, hgnc_resolver: HgncResolver
) -> None:
    """Re-resolve rows whose ``gnomad`` payload is ``{"normalization_error": true}``.

    Reads ``normalization.original_text`` and the paper's ``genome_build``,
    re-submits through the same async fan-out. The replacement is per-paper
    inside ``store_results_for_doi``: each paper's flush deletes the old
    errored rows and inserts the new ones in the same transaction. A crash
    while another paper is still resolving leaves that paper's old rows in
    the DB for the next ``--retry-errors`` run.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, variant_id, hgnc_id, paper_doi, box_id, normalization
            FROM variant_frequencies
            WHERE json_extract(gnomad, '$.normalization_error') IS NOT NULL
        """)
        rows = cursor.fetchall()

    if not rows:
        print("No errored variants found.")
        return

    paper_dois = {row["paper_doi"] for row in rows}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" * len(paper_dois))
        cur = conn.execute(
            f"SELECT doi, evidence_extraction_json FROM papers WHERE doi IN ({placeholders})",
            list(paper_dois),
        )
        paper_genome_build: dict[str, str | None] = {}
        for paper_row in cur.fetchall():
            extraction = json.loads(paper_row["evidence_extraction_json"])
            build = extraction.get("genome_build")
            if build == "unknown":
                build = None
            paper_genome_build[paper_row["doi"]] = build

    flat: list[tuple[str, ExtractedVariant]] = []
    old_ids_by_key: dict[tuple[str, int, str], int] = {}
    for row in rows:
        doi = row["paper_doi"]
        normalization = json.loads(row["normalization"])
        original_text = normalization["original_text"]
        flat.append(
            (
                doi,
                ExtractedVariant(
                    hgnc_id=row["hgnc_id"],
                    hgnc_symbol=hgnc_resolver.get_symbol(row["hgnc_id"]),
                    variant_text=original_text,
                    genome_build=paper_genome_build.get(doi),
                    box_id=row["box_id"],
                ),
            )
        )
        old_ids_by_key[(doi, row["box_id"], original_text)] = row["id"]

    flat.sort(key=lambda pair: _chromosome_sort_key(pair[1], hgnc_resolver))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
    ) as progress:
        # Per-paper completion drives the progress bar — counts papers, not variants.
        paper_count = len({doi for doi, _ in flat})
        task_id = progress.add_task("Retrying errored variants...", total=paper_count)
        results = await _resolve_all(flat, client, db_path, progress, task_id, old_ids_by_key)

    fixed = sum(1 for r in results if not r.gnomad.get("normalization_error"))
    still_errored = len(results) - fixed
    print(f"\nDone. Fixed: {fixed}, still errored: {still_errored}")


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


async def _main(
    db_path: Path, settings: VariantLookupSettings, max_in_flight: int, retry_errors: bool
) -> None:
    hgnc_resolver = HgncResolver.from_file()
    client = VariantLookupClient(settings, max_in_flight=max_in_flight)
    try:
        if retry_errors:
            await _retry_errored_variants(db_path, client, hgnc_resolver)
            return

        variants_by_doi = load_extracted_variants(db_path, hgnc_resolver)
        if not variants_by_doi:
            logger.warning("No variants found in database")
            print("No variants found in database")
            return

        processed_keys = get_processed_variant_keys(db_path)
        skipped = 0
        for doi in list(variants_by_doi.keys()):
            remaining = [
                v
                for v in variants_by_doi[doi]
                if (doi, v.box_id, v.variant_text) not in processed_keys
            ]
            skipped += len(variants_by_doi[doi]) - len(remaining)
            if remaining:
                variants_by_doi[doi] = remaining
            else:
                del variants_by_doi[doi]

        if not variants_by_doi:
            msg = "All variants already attempted (rerun with --retry-errors to retry error rows)"
            logger.info(msg)
            print(msg)
            return

        if skipped:
            logger.info(f"Skipping {skipped} already-attempted variants")

        flat: list[tuple[str, ExtractedVariant]] = [
            (doi, ev) for doi, evs in variants_by_doi.items() for ev in evs
        ]
        flat.sort(key=lambda pair: _chromosome_sort_key(pair[1], hgnc_resolver))
        logger.info(
            f"Processing {len(flat)} variants across {len(variants_by_doi)} papers "
            f"with up to {max_in_flight} in-flight requests"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("Processing papers...", total=len(variants_by_doi))
            results = await _resolve_all(flat, client, db_path, progress, task_id, {})

        print_summary_statistics(results)
        logger.info("✅ Variant frequency lookup complete")
    finally:
        await client.aclose()


@app.callback(invoke_without_command=True)
def lookup(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database with extracted variants",
    ),
    max_in_flight: int = typer.Option(
        default=_DEFAULT_MAX_IN_FLIGHT,
        help="Cap on concurrent in-flight requests to the variant-lookup service.",
    ),
    retry_errors: bool = typer.Option(
        default=False,
        help='Re-resolve rows whose gnomad payload is {"normalization_error": true}.',
    ),
) -> None:
    """Look up variant frequencies for all extracted variants.

    Issues one HTTPS request per variant to the variant-lookup service
    (``POST /v1/variant``), bounded by an in-flight semaphore. Variants
    are sorted by gene chromosome to maximise cache locality in the
    backend. Resumable: rerunning skips variants that already have any
    row in ``variant_frequencies`` (use ``--retry-errors`` to retry
    rows that landed as ``normalization_error``).
    """
    settings = _load_settings()
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        raise typer.Exit(1)

    asyncio.run(_main(db_path, settings, max_in_flight, retry_errors))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
