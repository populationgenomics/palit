#!/usr/bin/env python3
"""Look up variant frequencies from gnomAD v4 for extracted variants."""

import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import tenacity
import typer
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from palit.hgnc import HgncResolver
from palit.normalize_variants import VariantNormalizer

logger = logging.getLogger(__name__)

app = typer.Typer(help="Look up variant frequencies from gnomAD v4")


@dataclass
class ExtractedVariant:
    """A variant extracted from evidence extraction."""

    hgnc_id: int
    hgnc_symbol: str  # Current HGNC symbol, needed for variant normalizer
    variant_text: str
    genome_build: str | None  # From evidence extraction (e.g. "GRCh38"), None if unknown
    box_id: int


@dataclass
class VariantFrequencyResult:
    """Result of looking up frequency for a single variant."""

    variant_id: str  # gnomAD pseudo-VCF format
    hgnc_id: int
    doi: str
    box_id: int
    normalization: dict[str, Any]  # HGVS c/p information
    gnomad: dict[str, Any]  # gnomAD response or error
    error: str | None = None


@dataclass
class VariantProcessingResults:
    """Results from processing variants including statistics."""

    results: list[VariantFrequencyResult]
    total_variants: int
    failed_normalizations: int


def load_extracted_variants(
    db_path: Path, hgnc_resolver: HgncResolver
) -> dict[str, list[ExtractedVariant]]:
    """Load variants from evidence extractions in the database.

    Returns dict mapping DOI to list of ExtractedVariant objects.
    """
    logger.info(f"Loading extracted variants from {db_path}...")

    variants_by_doi: dict[str, list[ExtractedVariant]] = {}

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all papers with evidence extraction
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

            # Extract variants from each gene evaluation (only resolved ones with hgnc_id)
            gene_evaluations = extraction_data.get("gene_evaluations", [])

            for eval_data in gene_evaluations:
                hgnc_id = eval_data.get("hgnc_id")
                if hgnc_id is None:
                    continue

                hgnc_symbol = hgnc_resolver.get_symbol(hgnc_id)

                # Get variants array with box_ids
                variant_entries = eval_data.get("variants", [])

                for variant_entry in variant_entries:
                    variant_text = variant_entry.get("variant")
                    box_id = variant_entry.get("box_id")

                    if variant_text and box_id is not None:
                        if doi not in variants_by_doi:
                            variants_by_doi[doi] = []
                        variants_by_doi[doi].append(
                            ExtractedVariant(
                                hgnc_id=hgnc_id,
                                hgnc_symbol=hgnc_symbol,
                                variant_text=variant_text,
                                genome_build=genome_build,
                                box_id=box_id,
                            )
                        )

    total_variants = sum(len(variants) for variants in variants_by_doi.values())
    logger.info(f"Loaded {total_variants} variants from {len(variants_by_doi)} papers")
    return variants_by_doi


def get_processed_dois(db_path: Path) -> set[str]:
    """Get set of all DOIs that have already been processed."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT paper_doi FROM variant_frequencies")
        return {row[0] for row in cursor.fetchall()}


@tenacity.retry(
    stop=tenacity.stop_after_attempt(5),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=30),
    retry=tenacity.retry_if_exception_type(requests.exceptions.RequestException),
    before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
)
def _gnomad_request(url: str, payload: dict[str, Any]) -> requests.Response:
    """Make a single gnomAD API request, retrying on network errors."""
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response


def query_gnomad_v4(variant_id: str) -> dict[str, Any]:
    """Query gnomAD v4 API for variant frequency information.

    Args:
        variant_id: Pseudo-VCF format like "17-43045606-C-G"

    Returns:
        Dict with gnomAD data or error information
    """
    # GraphQL endpoint for gnomAD v4
    url = "https://gnomad.broadinstitute.org/api"

    # GraphQL query for variant information
    query = """
    query GnomadVariant($variantId: String!, $datasetId: DatasetId!) {
      variant(variantId: $variantId, dataset: $datasetId) {
        joint {
          ac
          an
          homozygote_count
          hemizygote_count
          faf95 {
            popmax
            popmax_population
          }
        }
      }
    }
    """

    # Variables for the query
    variables = {"variantId": variant_id, "datasetId": "gnomad_r4"}
    payload = {"query": query, "variables": variables}

    try:
        response = _gnomad_request(url, payload)
        result = response.json()

        if "errors" in result:
            error_msg = result["errors"][0].get("message", "Unknown error")
            if "Variant not found" in error_msg:
                # This is expected - many variants won't be in gnomAD
                return {"variant_not_found": True}
            else:
                return {"error": f"gnomAD API error: {error_msg}"}

        data = result.get("data")
        return data if data is not None else {}

    except requests.exceptions.RequestException as e:
        logger.warning(f"Network error querying gnomAD for {variant_id} after retries: {e}")
        return {"error": f"Network error: {e!s}"}
    except Exception as e:
        logger.warning(f"Unexpected error querying gnomAD for {variant_id}: {e}")
        return {"error": f"Unexpected error: {e!s}"}


def process_variants_for_doi(
    doi: str, variants: list[ExtractedVariant], variant_normalizer: VariantNormalizer
) -> VariantProcessingResults:
    """Process all variants for a single paper through normalization and gnomAD lookup.

    Processes variants sequentially within a paper. Papers themselves are processed
    in parallel by the caller.
    """
    results = []
    failed_normalizations = 0
    total_variants = len(variants)

    for variant in variants:
        hgnc_id = variant.hgnc_id
        hgnc_symbol = variant.hgnc_symbol
        variant_text = variant.variant_text
        genome_build = variant.genome_build
        box_id = variant.box_id

        try:
            # Step 1: Normalize variant to get HGVS and pseudo-VCF
            logger.debug(
                f"Normalizing variant '{variant_text}' for gene {hgnc_symbol} from DOI {doi}"
            )

            hgvs_variant = variant_normalizer.hgvs(variant_text, hgnc_symbol, genome_build)
            p_vcfs = variant_normalizer.pseudo_vcf(hgvs_variant)

            if not p_vcfs:
                raise ValueError("No pseudo-VCF coordinates generated")

            # Multiple pseudo-VCF results can occur when back-translating ambiguous protein changes.
            # For rare disease assessment, we need the MAXIMUM allele count across all possible
            # translations to properly evaluate variant rarity. A variant is only as rare as its
            # most common translation in the population.
            best_p_vcf = None
            best_gnomad_result = None
            max_ac = -1

            for p_vcf in p_vcfs:
                variant_id = p_vcf.p_vcf

                logger.debug(
                    f"Querying gnomAD for variant {variant_id} ({len(p_vcfs)} total translations)"
                )
                gnomad_result = query_gnomad_v4(variant_id)

                # Extract allele count for comparison
                current_ac = -1
                if "error" not in gnomad_result and "variant_not_found" not in gnomad_result:
                    variant_info: dict[str, Any] | None = gnomad_result.get("variant")
                    if variant_info and variant_info.get("joint", {}).get("ac") is not None:
                        current_ac = variant_info["joint"]["ac"]

                # Select variant with highest AC (most common = least rare)
                # On tie, prefer lexicographically first variant_id for consistency
                if current_ac > max_ac or (
                    current_ac == max_ac and (best_p_vcf is None or variant_id < best_p_vcf.p_vcf)
                ):
                    max_ac = current_ac
                    best_p_vcf = p_vcf
                    best_gnomad_result = gnomad_result

            # Use the variant with maximum AC
            if best_p_vcf is None or best_gnomad_result is None:
                raise ValueError("No valid variant result found")

            p_vcf = best_p_vcf
            gnomad_result = best_gnomad_result
            variant_id = p_vcf.p_vcf

            # Prepare normalization data
            normalization = {
                "hgvs_variant": str(hgvs_variant),
                "hgvs_c": p_vcf.hgvs_c,
                "hgvs_p": p_vcf.hgvs_p,
                "original_text": variant_text,
                "total_translations": len(p_vcfs),
                "selected_for_max_ac": max_ac if max_ac >= 0 else None,
            }

            result = VariantFrequencyResult(
                variant_id=variant_id,
                hgnc_id=hgnc_id,
                doi=doi,
                box_id=box_id,
                normalization=normalization,
                gnomad=gnomad_result,
            )
            results.append(result)

        except Exception as e:
            logger.debug(
                f"Failed to process variant '{variant_text}' for gene {hgnc_symbol} from DOI {doi}: {e}"
            )
            failed_normalizations += 1

    return VariantProcessingResults(
        results=results, total_variants=total_variants, failed_normalizations=failed_normalizations
    )


def store_results_for_doi(doi: str, results: list[VariantFrequencyResult], db_path: Path) -> None:
    """Store variant frequency results for a single paper atomically in a transaction."""
    if not results:
        logger.debug(f"No results to store for DOI {doi}")
        return

    # Deduplicate by (variant_id, doi, box_id) - keep first occurrence
    seen_keys: set[tuple[str, str, int]] = set()
    unique_results = []
    duplicates_skipped = 0

    for result in results:
        key = (result.variant_id, result.doi, result.box_id)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_results.append(result)
        else:
            duplicates_skipped += 1

    if duplicates_skipped > 0:
        logger.debug(f"Skipped {duplicates_skipped} duplicate entries for DOI {doi}")

    logger.debug(f"Storing {len(unique_results)} variant frequency results for DOI {doi}...")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Insert all unique results for this paper atomically
        for result in unique_results:
            cursor.execute(
                """
                INSERT INTO variant_frequencies
                (variant_id, hgnc_id, paper_doi, box_id, normalization, gnomad)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    result.variant_id,
                    result.hgnc_id,
                    result.doi,
                    result.box_id,
                    json.dumps(result.normalization),
                    json.dumps(result.gnomad),
                ),
            )

        conn.commit()

    logger.debug(f"Stored {len(unique_results)} variant frequency results for DOI {doi}")


def print_summary_statistics(processing_results: VariantProcessingResults) -> None:
    """Print summary statistics about the variant frequency lookup."""

    total = processing_results.total_variants
    normalization_errors = processing_results.failed_normalizations
    successful_normalizations = total - normalization_errors

    # Count gnomAD lookup results (only for successfully normalized variants)
    found_in_gnomad = 0
    not_found = 0
    api_errors = 0

    for result in processing_results.results:
        if not result.error:
            if "error" in result.gnomad:
                api_errors += 1
            elif "variant_not_found" in result.gnomad:
                not_found += 1
            elif "variant" in result.gnomad:
                found_in_gnomad += 1

    print("\n" + "=" * 60)
    print("VARIANT FREQUENCY LOOKUP SUMMARY")
    print("=" * 60)

    print(f"Total variants processed: {total}")
    if total > 0:
        print(
            f"Successfully normalized: {successful_normalizations} ({successful_normalizations / total * 100:.1f}%)"
        )
        print(
            f"Failed normalization: {normalization_errors} ({normalization_errors / total * 100:.1f}%)"
        )

    print("\ngnomAD v4 lookup results:")
    print(f"  Found in gnomAD: {found_in_gnomad}")
    print(f"  Not found in gnomAD: {not_found}")
    print(f"  API/network errors: {api_errors}")

    # Show some high-frequency variants if any
    high_freq_variants = []
    for result in processing_results.results:
        if (
            not result.error
            and "error" not in result.gnomad
            and "variant_not_found" not in result.gnomad
        ):
            variant_data = result.gnomad.get("variant")
            if variant_data and variant_data.get("joint", {}).get("ac", 0) > 30:
                high_freq_variants.append(
                    (result.variant_id, variant_data["joint"]["ac"], result.hgnc_id)
                )

    if high_freq_variants:
        print(f"\n⚠️  High-frequency variants (AC > 30): {len(high_freq_variants)}")
        for variant_id, ac, hgnc_id in high_freq_variants[:5]:
            print(f"    HGNC:{hgnc_id}: {variant_id} (AC={ac})")
        if len(high_freq_variants) > 5:
            print(f"    ... and {len(high_freq_variants) - 5} more")


def _retry_errored_variants(db_path: Path) -> None:
    """Re-query gnomAD for variants that previously failed with network/timeout errors."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, variant_id, hgnc_id
            FROM variant_frequencies
            WHERE json_extract(gnomad, '$.error') IS NOT NULL
        """)
        error_rows = cursor.fetchall()

    if not error_rows:
        print("No errored variants found.")
        return

    print(f"Found {len(error_rows)} variants with errors. Retrying...")

    fixed = 0
    still_errored = 0
    for row in error_rows:
        row_id = row["id"]
        variant_id = row["variant_id"]
        hgnc_id = row["hgnc_id"]

        gnomad_result = query_gnomad_v4(variant_id)

        if "error" in gnomad_result:
            still_errored += 1
            print(f"  Still failing: HGNC:{hgnc_id} {variant_id}: {gnomad_result['error']}")
        else:
            fixed += 1
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE variant_frequencies SET gnomad = ? WHERE id = ?",
                    (json.dumps(gnomad_result), row_id),
                )
                conn.commit()
            print(f"  Fixed: HGNC:{hgnc_id} {variant_id}")

    print(f"\nDone. Fixed: {fixed}, still errored: {still_errored}")


@app.callback(invoke_without_command=True)
def lookup(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"), help="Path to SQLite database with extracted variants"
    ),
    max_workers: int = typer.Option(default=5, help="Number of papers to process in parallel"),
    retry_errors: bool = typer.Option(
        default=False, help="Re-query gnomAD for variants that previously failed with errors"
    ),
) -> None:
    """Look up variant frequencies from gnomAD v4 for all extracted variants.

    Processes papers in parallel with resumability - already processed papers are skipped.
    """

    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        raise typer.Exit(1)

    if retry_errors:
        _retry_errored_variants(db_path)
        return

    # Step 1: Load variants from evidence extractions, grouped by DOI
    hgnc_resolver = HgncResolver.from_file()
    variants_by_doi = load_extracted_variants(db_path, hgnc_resolver)

    if not variants_by_doi:
        logger.warning("No variants found in database")
        print("No variants found in database")
        return

    # Step 2: Filter out already processed DOIs
    processed_dois = get_processed_dois(db_path)
    dois_to_process = [doi for doi in variants_by_doi if doi not in processed_dois]

    if not dois_to_process:
        logger.info("All papers already processed")
        print("All papers already processed")
        return

    skipped_count = len(variants_by_doi) - len(dois_to_process)
    if skipped_count > 0:
        logger.info(f"Skipping {skipped_count} already processed papers")

    logger.info(f"Processing {len(dois_to_process)} papers with {max_workers} parallel workers")

    # Step 3: Process papers in parallel
    variant_normalizer = VariantNormalizer()
    all_results: list[VariantFrequencyResult] = []
    total_failed_normalizations = 0
    total_variants_processed = 0

    def process_and_store_doi(doi: str) -> VariantProcessingResults:
        """Process a single paper and store results (called in worker thread)."""
        variants = variants_by_doi[doi]

        # Process all variants for this paper
        processing_results = process_variants_for_doi(doi, variants, variant_normalizer)

        # Store results atomically for this paper
        store_results_for_doi(doi, processing_results.results, db_path)

        return processing_results

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Processing papers...", total=len(dois_to_process))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all papers to the worker pool
            future_to_doi = {
                executor.submit(process_and_store_doi, doi): doi for doi in dois_to_process
            }

            # Main thread: sequentially pull completed results from queue
            for future in as_completed(future_to_doi):
                doi = future_to_doi[future]
                processing_results = future.result()
                num_variants = processing_results.total_variants

                # Accumulate statistics in main thread (no lock needed)
                all_results.extend(processing_results.results)
                total_failed_normalizations += processing_results.failed_normalizations
                total_variants_processed += processing_results.total_variants

                # Update progress
                progress.update(
                    task,
                    advance=1,
                    description=f"[green]Completed DOI {doi} ({num_variants} variants)",
                )

    # Step 4: Print summary statistics
    combined_results = VariantProcessingResults(
        results=all_results,
        total_variants=total_variants_processed,
        failed_normalizations=total_failed_normalizations,
    )
    print_summary_statistics(combined_results)

    logger.info("✅ Variant frequency lookup complete")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
