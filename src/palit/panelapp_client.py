#!/usr/bin/env python3
"""Reusable PanelApp API client for gene-panel operations."""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from palit.panelapp_integration import TARGET_PANEL_IDS

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


def _request_with_retry(
    client: httpx.Client,
    url: str,
    timeout: float = 30.0,
    max_retries: int = MAX_RETRIES,
) -> httpx.Response:
    """Make an HTTP GET request with retry logic and exponential backoff.

    Only retries on transient errors (5xx server errors and connection issues).
    Client errors (4xx) are considered permanent and are not retried.

    Args:
        client: httpx Client instance
        url: URL to fetch
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts

    Returns:
        httpx.Response on success

    Raises:
        httpx.HTTPStatusError: For client errors (4xx) or after all retries exhausted
        httpx.RequestError: After all retries exhausted for connection errors
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            # Only retry on transient errors: 429 (rate limit) and 5xx (server errors)
            # Other 4xx errors (400, 401, 403, 404, etc.) are permanent
            if e.response.status_code != 429 and e.response.status_code < 500:
                raise
            last_exception = e
            if attempt < max_retries:
                sleep_time = RETRY_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    f"Request to {url} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {sleep_time}s..."
                )
                time.sleep(sleep_time)
            else:
                logger.error(f"Request to {url} failed after {max_retries + 1} attempts: {e}")
        except httpx.RequestError as e:
            # Retry connection/timeout errors
            last_exception = e
            if attempt < max_retries:
                sleep_time = RETRY_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    f"Request to {url} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {sleep_time}s..."
                )
                time.sleep(sleep_time)
            else:
                logger.error(f"Request to {url} failed after {max_retries + 1} attempts: {e}")

    # Should never reach here, but satisfy type checker
    raise last_exception  # type: ignore[misc]


# PanelApp API configuration
PANELAPP_BASE_URL = "https://panelapp-aus.org/api/v1"


@dataclass(frozen=True)
class ResolvedGene:
    """A gene symbol resolved from paper mention to PanelApp form."""

    panelapp_symbol: str  # Symbol as used in PanelApp (for matching/lookup)
    paper_symbol: str  # Original symbol mentioned in paper


@dataclass
class NewNameTaggedGene:
    """A gene tagged with 'new gene name' in PanelApp."""

    panelapp_symbol: str
    hgnc_id: str


@dataclass
class PanelGeneData:
    """Combined gene data from PanelApp panels."""

    panel_ids: list[int]  # Ordered list of panel IDs
    combined_gene_symbols: set[
        str
    ]  # All unique gene symbols (PanelApp symbols + current HGNC aliases)
    gene_confidence: dict[str, int]  # PanelApp symbol -> confidence level (integer)
    hgnc_to_panelapp: dict[str, str]  # Current HGNC symbol -> PanelApp symbol mapping
    gene_panel_mapping: dict[str, set[int]]  # Gene symbol -> set of panel IDs
    gene_moi: dict[str, str]  # PanelApp symbol -> mode of inheritance


@dataclass
class AllPanelsData:
    """Gene data from ALL panels with panel information."""

    gene_to_panels: dict[str, set[int]]  # gene_symbol -> set of panel_ids
    panel_names: dict[int, str]  # panel_id -> panel name


def clean_panel_publications(pmids: list[str]) -> list[int]:
    """Clean and extract PMIDs from panel publication strings.

    Splits on both comma and space, filters for digits only.
    """
    result: list[int] = []
    for pmid in pmids:
        # Split on both comma and space, filter for digits only
        tokens = pmid.replace(",", " ").split()
        result.extend(int(token) for token in tokens if token.isdigit())
    return result


def extract_panel_pmids(panel_data: dict[str, Any]) -> list[int]:
    """Extract all PMIDs from panel genes, strs, and regions publications."""
    all_pmids = []

    # Extract from genes
    genes = panel_data.get("genes", [])
    for gene in genes:
        publications = gene.get("publications", [])
        if publications:
            clean_pmids = clean_panel_publications(publications)
            all_pmids.extend(clean_pmids)

    # Extract from STRs
    strs = panel_data.get("strs", [])
    for str_entry in strs:
        publications = str_entry.get("publications", [])
        if publications:
            clean_pmids = clean_panel_publications(publications)
            all_pmids.extend(clean_pmids)

    # Extract from regions
    regions = panel_data.get("regions", [])
    for region in regions:
        publications = region.get("publications", [])
        if publications:
            clean_pmids = clean_panel_publications(publications)
            all_pmids.extend(clean_pmids)

    # Remove duplicates while preserving order
    unique_pmids = list(dict.fromkeys(all_pmids))
    return unique_pmids


def resolve_gene_symbols(gene_symbols: set[str], panel_data: PanelGeneData) -> set[ResolvedGene]:
    """Resolve a set of gene symbols using PanelApp alias mappings.

    Args:
        gene_symbols: Set of gene symbols from paper (case-insensitive)
        panel_data: PanelApp gene data with alias mappings

    Returns:
        Set of ResolvedGene objects with both paper and panelapp symbols
    """
    resolved_genes = set()

    for paper_symbol in gene_symbols:
        if not paper_symbol:
            continue

        paper_symbol_upper = paper_symbol.upper()
        # Resolve to PanelApp symbol if it's an alias
        panelapp_symbol = panel_data.hgnc_to_panelapp.get(paper_symbol_upper, paper_symbol_upper)
        resolved_genes.add(ResolvedGene(panelapp_symbol, paper_symbol_upper))

    return resolved_genes


def format_panel_for_prompt(panel_id: int, panel_info: dict[str, Any]) -> str:
    """Format a single panel description for LLM prompt.

    Args:
        panel_id: Panel ID
        panel_info: Panel information dictionary with keys like 'name', 'level',
                    'relevant_disorders', 'description', etc.

    Returns:
        Formatted panel description string
    """
    formatted = []
    formatted.append(f'<panel id="{panel_id}">')
    formatted.append(f"Name: {panel_info['name']}")
    if panel_info.get("level"):
        formatted.append(f"Level: {panel_info['level']}")
    if panel_info.get("relevant_disorders"):
        formatted.append(f"Relevant disorders: {panel_info['relevant_disorders']}")
    if panel_info.get("disease_group"):
        formatted.append(f"Disease group: {panel_info['disease_group']}")
    if panel_info.get("disease_sub_group"):
        formatted.append(f"Disease subgroup: {panel_info['disease_sub_group']}")
    if panel_info.get("description"):
        formatted.append(f"Description: {panel_info['description']}")
    formatted.append("</panel>")
    return "\n".join(formatted)


class PanelAppClient:
    """Client for interacting with PanelApp API with caching support."""

    def __init__(
        self,
        panel_date: str,
        cache_dir: Path | str = "data/panelapp",
        timeout: float = 60.0,
    ):
        """Initialize the PanelApp client with cached data for a specific date.

        Args:
            panel_date: Date in YYYY-MM-DD format for which to load/fetch panel data
            cache_dir: Directory to store cache files (default: data/panelapp)
            timeout: Request timeout in seconds for API calls
        """
        self.panel_date = panel_date
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.base_url = PANELAPP_BASE_URL

        # Cache will be loaded lazily on first access
        self._panel_data: dict[int, dict[str, Any]] | None = None

    def _ensure_cache_loaded(self) -> dict[int, dict[str, Any]]:
        """Ensure the cache is loaded, loading it if necessary."""
        if self._panel_data is None:
            self._panel_data = self._load_or_fetch_all_panels()
        return self._panel_data

    def _load_or_fetch_all_panels(self) -> dict[int, dict[str, Any]]:
        """Load cached panel data or fetch from API if cache doesn't exist.

        Returns:
            Dictionary mapping panel IDs to their complete data
        """
        # Create cache directory if needed
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build cache file path
        cache_file = self.cache_dir / f"{self.panel_date}.json"

        # Try to load existing cache
        if cache_file.exists():
            logger.info(f"Loading cached panel data from {cache_file}")
            with open(cache_file) as f:
                # Convert string keys back to integers
                data = json.load(f)
                return {int(k): v for k, v in data.items()}

        # Fetch all panels if no cache exists
        logger.info(f"No cache found for {self.panel_date}, fetching all panels...")

        # Get list of all panel IDs
        panel_ids = self._get_all_panel_ids()
        logger.info(f"Found {len(panel_ids)} panels to fetch")

        # Fetch each panel at the correct version for the date
        all_panels = {}
        cutoff_date = datetime.fromisoformat(self.panel_date).replace(tzinfo=UTC)

        with httpx.Client(timeout=self.timeout) as client:
            for panel_id in panel_ids:
                # Find the correct version for this date
                activities_url = f"{self.base_url}/panels/{panel_id}/activities/"
                response = _request_with_retry(client, activities_url, timeout=self.timeout)
                activities = response.json()

                # Find latest version at or before cutoff date
                target_version = None
                for activity in sorted(activities, key=lambda x: x["created"]):
                    activity_date = datetime.fromisoformat(
                        activity["created"].replace("Z", "+00:00")
                    )
                    if activity_date <= cutoff_date:
                        panel_version = activity.get("panel_version")
                        if panel_version and panel_version != "0.0":
                            target_version = panel_version
                    else:
                        break

                if not target_version:
                    # Panel didn't exist at this date - skip it
                    logger.debug(
                        f"Skipping panel {panel_id}: no version exists at or before {self.panel_date} "
                        f"(panel has {len(activities)} activities, likely created after cutoff)"
                    )
                    continue

                # Fetch the panel at this version
                panel_url = f"{self.base_url}/panels/{panel_id}/?version={target_version}"
                response = _request_with_retry(client, panel_url, timeout=self.timeout)

                all_panels[panel_id] = response.json()
                logger.debug(f"Fetched panel {panel_id} version {target_version}")

        # Save to cache
        logger.info(f"Saving {len(all_panels)} panels to cache: {cache_file}")
        with open(cache_file, "w") as f:
            # Convert integer keys to strings for JSON serialization
            json.dump({str(k): v for k, v in all_panels.items()}, f, indent=2)

        return all_panels

    def get_genes_with_new_name_tag(self) -> list[NewNameTaggedGene]:
        """Extract all genes with 'new gene name' tag from cached panel data."""
        panel_data_cache = self._ensure_cache_loaded()

        # First collect all genes with the tag to check for collisions
        genes_by_hgnc = defaultdict(list)

        for panel_id, panel_data in panel_data_cache.items():
            all_entities = panel_data.get("genes", []) + panel_data.get("strs", [])

            for entity in all_entities:
                if "new gene name" in entity.get("tags", []):
                    gene_data = entity.get("gene_data", {})
                    gene_symbol = entity.get("entity_name")
                    hgnc_id = gene_data.get("hgnc_id")

                    if gene_symbol and hgnc_id:
                        genes_by_hgnc[hgnc_id].append(
                            {"panelapp_symbol": gene_symbol, "panel_id": panel_id}
                        )

        # Detect collisions: same HGNC ID with different PanelApp symbols
        collisions = []
        for hgnc_id, gene_list in genes_by_hgnc.items():
            panelapp_symbols = {g["panelapp_symbol"] for g in gene_list}
            if len(panelapp_symbols) > 1:
                collisions.append((hgnc_id, gene_list))

        if collisions:
            error_msg = f"CRITICAL: Found {len(collisions)} HGNC ID collisions with different PanelApp symbols:\n"
            for hgnc_id, gene_list in collisions:
                error_msg += f"  HGNC {hgnc_id}:\n"
                for gene in gene_list:
                    error_msg += f"    Panel {gene['panel_id']}: {gene['panelapp_symbol']}\n"
            error_msg += "This would cause incorrect alias mappings. Fix data inconsistency before proceeding."
            raise ValueError(error_msg)

        # Convert to list of unique genes (deduplicated by HGNC ID)
        unique_genes = []
        seen_hgnc_ids = set()

        for hgnc_id, gene_list in genes_by_hgnc.items():
            if hgnc_id not in seen_hgnc_ids:
                seen_hgnc_ids.add(hgnc_id)
                # Take the first occurrence (they should all be the same after collision check)
                unique_genes.append(
                    NewNameTaggedGene(
                        panelapp_symbol=gene_list[0]["panelapp_symbol"], hgnc_id=hgnc_id
                    )
                )

        return unique_genes

    def load_hgnc_lookup_by_id(self) -> dict[str, str]:
        """Build HGNC ID → current symbol lookup from local HGNC data."""
        hgnc_file = Path("data/hgnc_complete_set.json")

        if not hgnc_file.exists():
            raise FileNotFoundError(f"HGNC data file not found at {hgnc_file}")

        logger.info(f"Loading HGNC data from {hgnc_file}...")

        with open(hgnc_file) as f:
            hgnc_data: dict[str, Any] = json.load(f)

        hgnc_by_id = {}
        docs = hgnc_data.get("response", {}).get("docs", [])

        for doc in docs:
            hgnc_id = doc.get("hgnc_id")
            symbol = doc.get("symbol")
            if hgnc_id and symbol:
                hgnc_by_id[hgnc_id] = symbol

        logger.info(f"Loaded {len(hgnc_by_id)} HGNC ID mappings")
        return hgnc_by_id

    def get_hgnc_to_panelapp_mappings(self) -> dict[str, str]:
        """Get mappings from current HGNC symbols to outdated PanelApp symbols.

        For genes tagged with 'new gene name' in PanelApp, this fetches the current
        HGNC symbol and creates a mapping to the outdated symbol still used in PanelApp.

        Returns:
            Dict mapping current HGNC symbol (uppercase) -> outdated PanelApp symbol (uppercase)
            Example: {'RXYLT1': 'TMEM5', 'LARS1': 'LARS', 'CRPPA': 'ISPD', ...}
        """
        try:
            tagged_genes = self.get_genes_with_new_name_tag()
            hgnc_lookup = self.load_hgnc_lookup_by_id()

            if not hgnc_lookup:
                return {}

            aliases = {}
            no_change_genes = set()

            for gene in tagged_genes:
                current_symbol = hgnc_lookup.get(gene.hgnc_id)

                if current_symbol:
                    if current_symbol.upper() == gene.panelapp_symbol.upper():
                        no_change_genes.add(f"{gene.panelapp_symbol} ({gene.hgnc_id})")
                    else:
                        aliases[current_symbol.upper()] = gene.panelapp_symbol.upper()

            if no_change_genes:
                logger.warning(
                    f"{len(no_change_genes)} genes tagged 'new gene name' but have unchanged HGNC symbols: {', '.join(sorted(no_change_genes))}"
                )

            logger.info(f"Found {len(aliases)} HGNC-based gene aliases")
            return aliases

        except Exception as e:
            logger.warning(f"Failed to get HGNC aliases: {e}")
            return {}

    def get_panel_data(self, panel_id: int) -> dict[str, Any]:
        """Get panel data from cache.

        Args:
            panel_id: PanelApp panel ID

        Returns:
            Panel data with genes and metadata
        """
        panel_data_cache = self._ensure_cache_loaded()
        if panel_id not in panel_data_cache:
            raise ValueError(
                f"Panel {panel_id} not found in cached data for date {self.panel_date}"
            )
        return panel_data_cache[panel_id]

    def get_target_panels_genes(self, panel_ids: list[int] | None = None) -> PanelGeneData:
        """Get gene symbols and confidence levels from specified panels.

        This includes both genes and STRs (Short Tandem Repeats) from the panels.
        Also collects gene aliases for improved matching.

        Args:
            panel_ids: List of panel IDs to include. If None, uses TARGET_PANEL_IDS.

        Returns:
            PanelGeneData containing all gene symbols, confidence levels, and alias mappings.
        """
        if panel_ids is None:
            panel_ids = TARGET_PANEL_IDS

        panel_data_cache = self._ensure_cache_loaded()

        combined_gene_symbols: set[str] = set()
        combined_gene_confidence: dict[str, int] = {}
        combined_gene_moi: dict[str, str] = {}  # Track mode of inheritance
        hgnc_to_panelapp: dict[str, str] = {}  # Maps current HGNC symbols to PanelApp symbols
        gene_sources: dict[str, set[int]] = {}  # Track which panels contribute each gene
        gene_panel_mapping: dict[str, set[int]] = {}  # Track gene to panel mappings
        panel_gene_counts: dict[int, int] = {}

        for panel_id in panel_ids:
            panel_data = panel_data_cache[panel_id]

            # Process all entities (genes and STRs) for this panel
            panel_gene_count = 0
            all_entities = panel_data.get("genes", []) + panel_data.get("strs", [])

            for entity in all_entities:
                gene_data = entity.get("gene_data", {})
                gene_symbol = gene_data.get("gene_symbol")
                confidence_level = entity.get("confidence_level")
                if confidence_level is None:
                    raise ValueError(
                        f"Gene '{gene_symbol}' has no confidence_level in API response"
                    )

                if gene_symbol:
                    gene_symbol_upper = gene_symbol.upper()
                    combined_gene_symbols.add(gene_symbol_upper)
                    panel_gene_count += 1

                    # Track which panels contribute this gene
                    if gene_symbol_upper not in gene_sources:
                        gene_sources[gene_symbol_upper] = set()
                        gene_panel_mapping[gene_symbol_upper] = set()
                    gene_sources[gene_symbol_upper].add(panel_id)
                    gene_panel_mapping[gene_symbol_upper].add(panel_id)

                    # Handle confidence level conflicts - use highest confidence
                    # Also track MoI from the same entity that provides the highest confidence
                    confidence_int = int(confidence_level)
                    if (
                        gene_symbol_upper not in combined_gene_confidence
                        or confidence_int > combined_gene_confidence[gene_symbol_upper]
                    ):
                        combined_gene_confidence[gene_symbol_upper] = confidence_int
                        # Store MoI from same entity that provides confidence
                        moi = entity.get("mode_of_inheritance") or "Unknown"
                        combined_gene_moi[gene_symbol_upper] = moi

            panel_gene_counts[panel_id] = panel_gene_count

        # Add current HGNC symbols that map to outdated PanelApp symbols
        all_hgnc_to_panelapp = self.get_hgnc_to_panelapp_mappings()

        # Filter to only include aliases where the PanelApp symbol is in our target panels
        hgnc_to_panelapp = {
            hgnc_symbol: panelapp_symbol
            for hgnc_symbol, panelapp_symbol in all_hgnc_to_panelapp.items()
            if panelapp_symbol in combined_gene_confidence
        }

        logger.info(
            f"Filtered aliases to target panels: {len(hgnc_to_panelapp)} (from {len(all_hgnc_to_panelapp)})"
        )
        combined_gene_symbols = combined_gene_symbols.union(hgnc_to_panelapp.keys())

        # Update gene panel mappings for aliases
        for hgnc_symbol, panelapp_symbol in hgnc_to_panelapp.items():
            if panelapp_symbol in gene_panel_mapping:
                gene_panel_mapping[hgnc_symbol] = gene_panel_mapping[panelapp_symbol].copy()

        # Calculate overlap
        len([gene for gene, sources in gene_sources.items() if len(sources) > 1])

        return PanelGeneData(
            panel_ids=panel_ids,
            combined_gene_symbols=combined_gene_symbols,
            gene_confidence=combined_gene_confidence,
            hgnc_to_panelapp=hgnc_to_panelapp,
            gene_panel_mapping=gene_panel_mapping,
            gene_moi=combined_gene_moi,
        )

    def get_all_panels_genes(self) -> AllPanelsData:
        """Get genes from ALL panels using cached data.

        Returns:
            AllPanelsData containing gene-to-panels mapping and panel names
        """
        panel_data_cache = self._ensure_cache_loaded()

        # Build mapping: gene -> panels it belongs to and collect panel names
        gene_to_panels: dict[str, set[int]] = {}
        panel_names: dict[int, str] = {}

        for panel_id, panel_data in panel_data_cache.items():
            # Store panel name
            panel_names[panel_id] = panel_data.get("name", f"Panel {panel_id}")

            # Process all entities (genes and STRs) for this panel
            all_entities = panel_data.get("genes", []) + panel_data.get("strs", [])

            for entity in all_entities:
                gene_data = entity.get("gene_data", {})
                gene_symbol = gene_data.get("gene_symbol")

                if gene_symbol:
                    gene_symbol_upper = gene_symbol.upper()

                    # Initialize gene entry if needed
                    if gene_symbol_upper not in gene_to_panels:
                        gene_to_panels[gene_symbol_upper] = set()

                    # Store panel membership
                    gene_to_panels[gene_symbol_upper].add(panel_id)

        return AllPanelsData(gene_to_panels=gene_to_panels, panel_names=panel_names)

    def _get_all_panel_ids(self) -> list[int]:
        """Get list of all active panel IDs from PanelApp.

        Returns:
            List of panel IDs for all public, non-superpanel panels
        """
        panel_ids = []
        url: str | None = f"{PANELAPP_BASE_URL}/panels/"

        with httpx.Client(timeout=self.timeout) as client:
            while url:
                logger.debug(f"Fetching panel list from: {url}")
                response = _request_with_retry(client, url, timeout=30.0)
                data = response.json()

                for panel in data["results"]:
                    # Skip archived and superpanels
                    if panel.get("status") != "public":
                        continue
                    # Super-panels have child_panel_ids (may be non-empty list or empty list)
                    # We want to exclude panels with non-empty child_panel_ids
                    if panel.get("child_panel_ids"):
                        continue

                    panel_ids.append(panel["id"])

                url = data.get("next")  # Pagination

        return panel_ids

    def get_all_panel_descriptions(self) -> dict[int, dict[str, Any]]:
        """Get descriptions of all cached panels, excluding super-panels.

        Super-panels are aggregations of other panels and should be excluded
        from panel matching to avoid redundancy.

        Returns:
            Dict mapping panel_id to panel information:
            {
                137: {
                    "name": "Mendeliome",
                    "description": "Panel description text",
                    "level": "Level 1: Core panels",
                    "relevant_disorders": "Rare genetic disorders",
                    "disease_group": "Multiple systems",
                    "disease_sub_group": "General genetic disorders"
                },
                ...
            }
        """
        panel_data_cache = self._ensure_cache_loaded()

        panels = {}
        excluded_count = 0

        for panel_id, panel_data in panel_data_cache.items():
            # Skip super-panels (panels with non-empty child_panel_ids)
            child_panel_ids = panel_data.get("child_panel_ids", [])
            if child_panel_ids:
                excluded_count += 1
                logger.debug(f"Excluding super-panel {panel_id}: {panel_data.get('name')}")
                continue

            panels[panel_id] = {
                "name": panel_data.get("name", ""),
                "description": panel_data.get("description", ""),
                "level": panel_data.get("level", ""),
                "relevant_disorders": panel_data.get("relevant_disorders", ""),
                "disease_group": panel_data.get("disease_group", ""),
                "disease_sub_group": panel_data.get("disease_sub_group", ""),
            }

        if excluded_count > 0:
            logger.info(f"Excluded {excluded_count} super-panels from panel descriptions")

        return panels

    def get_gene_evaluations(self, panel_id: int, gene_symbol: str) -> list[dict[str, Any]]:
        """Fetch evaluations with comments for a gene in a panel.

        Makes a live API call (not cached) to get the current evaluations
        with review comments for a specific gene in a panel.

        Args:
            panel_id: Panel ID
            gene_symbol: Gene symbol to look up

        Returns:
            List of evaluation dicts with comments, ordered by most recent first.
            Returns empty list if gene is not in the panel (404).
        """
        url = f"{self.base_url}/panels/{panel_id}/genes/{gene_symbol}/evaluations/?include_comments=true"

        with httpx.Client(timeout=self.timeout) as client:
            try:
                response = _request_with_retry(client, url, timeout=self.timeout)
                data = response.json()
                results: list[dict[str, Any]] = data.get("results", [])
                # Sort by created date descending (most recent first)
                results.sort(key=lambda x: x.get("created", ""), reverse=True)
                return results
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # Gene not in panel - this is expected for novel genes
                    logger.debug(f"Gene {gene_symbol} not found in panel {panel_id}")
                    return []
                raise


def get_current_panel_pmids(panel_ids: list[int] | None = None, timeout: float = 60.0) -> set[int]:
    """Fetch PMIDs from current/latest panel versions without caching.

    This function always fetches fresh data from the API and is used for
    validation against the most current panel publications.

    Args:
        panel_ids: List of panel IDs to extract PMIDs from. If None, uses TARGET_PANEL_IDS.
        timeout: Request timeout in seconds

    Returns:
        Set of unique PMIDs from the specified panels.
    """
    if panel_ids is None:
        panel_ids = TARGET_PANEL_IDS

    all_pmids: set[int] = set()

    with httpx.Client(timeout=timeout) as client:
        for panel_id in panel_ids:
            # Fetch current panel data (no version parameter)
            panel_url = f"{PANELAPP_BASE_URL}/panels/{panel_id}/"
            response = _request_with_retry(client, panel_url, timeout=timeout)
            panel_data = response.json()

            # Extract PMIDs using the free function
            panel_pmids = extract_panel_pmids(panel_data)
            all_pmids.update(panel_pmids)

    return all_pmids
