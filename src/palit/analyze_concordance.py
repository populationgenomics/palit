#!/usr/bin/env python3
"""Analyze concordance between literature assessment suggestions and PanelApp curation outcomes.

This script compares gene suggestions and rating recommendations from monthly reports against
the actual Mendeliome panel state ~2 months later to measure concordance rates.
"""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, cast

import httpx
import plotly.graph_objects as go
import typer
from jinja2 import Environment, FileSystemLoader
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

console = Console()
app = typer.Typer(help="Analyze concordance between suggestions and PanelApp curation outcomes")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# Constants
MENDELIOME_PANEL_ID = 137
PANELAPP_BASE_URL = "https://panelapp-aus.org/api/v1"
CONCORDANCE_DATA_DIR = Path("data/concordance")


class ConcordanceStatus(Enum):
    """Status of concordance between suggestion and actual outcome."""

    CONCORDANT = "concordant"  # Executed with matching rating
    DISCORDANT = "discordant"  # Executed with different rating
    NOT_EXECUTED = "not_executed"  # Not added/not upgraded


@dataclass
class NovelGene:
    """A gene suggested for addition to the panel."""

    gene: str
    suggested_rating: str  # GREEN, AMBER, or RED


@dataclass
class Upgrade:
    """A gene suggested for rating upgrade."""

    gene: str
    current: str  # RED, AMBER, or GREEN
    suggested: str  # RED, AMBER, or GREEN


@dataclass
class Suggestions:
    """Suggestions extracted from a report."""

    novel_genes: list[NovelGene]
    upgrades: list[Upgrade]


@dataclass
class GeneStatus:
    """Concordance status for a novel gene suggestion."""

    gene: str
    status: ConcordanceStatus
    suggested_rating: str
    actual_rating: str | None  # None if not executed


@dataclass
class UpgradeGeneStatus:
    """Concordance status for an upgrade suggestion."""

    gene: str
    status: ConcordanceStatus
    current_rating: str
    suggested_rating: str
    actual_rating: str


@dataclass
class NovelConcordance:
    """Concordance results for novel gene suggestions."""

    gene_statuses: list[GeneStatus]
    suggested: int
    added: int
    concordant: int


@dataclass
class UpgradeConcordance:
    """Concordance results for upgrade suggestions."""

    gene_statuses: list[UpgradeGeneStatus]
    suggested: int
    executed: int
    concordant: int


@dataclass
class ConcordanceResults:
    """Concordance analysis results for one period."""

    month: str
    novel_suggested: int
    novel_added: int
    novel_added_concordant: int
    upgrade_suggested: int
    upgrade_executed: int
    upgrade_executed_concordant: int


@dataclass
class MonthlyPanelActivity:
    """Overall panel activity for a given month (all sources, not just our suggestions)."""

    month: str  # YYYY-MM format
    additions_by_rating: dict[int, list[str]]  # rating -> list of gene names
    upgrades_by_rating: dict[int, list[str]]  # rating -> list of gene names


@dataclass
class PanelSnapshot:
    """Snapshot of panel genes at a specific point in time."""

    month: str  # YYYY-MM format
    version: str  # Panel version (e.g., "1.3383")
    genes: dict[str, int]  # gene name -> confidence_level (1=RED, 2=AMBER, 3=GREEN)


def _is_upgrade(current: str, suggested: str) -> bool:
    """Check if a rating change is an upgrade (higher confidence).

    Upgrades: RED → AMBER, RED → GREEN, AMBER → GREEN
    Downgrades: GREEN → AMBER, GREEN → RED, AMBER → RED
    """
    rating_order = {"RED": 0, "AMBER": 1, "GREEN": 2}
    return rating_order[suggested] > rating_order[current]


def merge_suggestions(suggestions_list: list[Suggestions]) -> Suggestions:
    """Merge multiple Suggestions objects into one.

    Args:
        suggestions_list: List of Suggestions to merge

    Returns:
        Merged Suggestions with combined novel_genes and upgrades
    """
    merged_novel_genes: list[NovelGene] = []
    merged_upgrades: list[Upgrade] = []

    for suggestions in suggestions_list:
        merged_novel_genes.extend(suggestions.novel_genes)
        merged_upgrades.extend(suggestions.upgrades)

    return Suggestions(novel_genes=merged_novel_genes, upgrades=merged_upgrades)


def log_suggestions_summary(
    novel_genes: list[NovelGene],
    upgrades: list[Upgrade],
    novel_concordance: NovelConcordance | None = None,
    upgrade_concordance: UpgradeConcordance | None = None,
) -> None:
    """Log detailed summary of novel genes and upgrades found in report.

    Args:
        novel_genes: Novel gene suggestions from report
        upgrades: Upgrade suggestions from report
        novel_concordance: Optional concordance results for novel genes
        upgrade_concordance: Optional concordance results for upgrades
    """
    logger.info(f"  Found {len(novel_genes)} novel genes and {len(upgrades)} upgrades")

    if novel_genes:
        # Create table for novel genes
        novel_table = Table(title="Novel Genes", show_header=True, header_style="bold magenta")
        novel_table.add_column("Gene", style="cyan")
        novel_table.add_column("Suggested Rating", style="green")

        # Group by rating
        by_rating: dict[str, list[str]] = {"GREEN": [], "AMBER": [], "RED": []}

        # Build status map if concordance data available (already sorted)
        status_map: dict[str, ConcordanceStatus] = {}
        if novel_concordance:
            status_map = {gs.gene: gs.status for gs in novel_concordance.gene_statuses}
            # Use pre-sorted gene order from concordance
            gene_order = {gs.gene: i for i, gs in enumerate(novel_concordance.gene_statuses)}
        else:
            gene_order = {}

        for ng in novel_genes:
            by_rating[ng.suggested_rating].append(ng.gene)

        # Add rows in rating order
        for rating in ["GREEN", "AMBER", "RED"]:
            genes = by_rating[rating]
            if genes:
                # Sort using pre-computed order from concordance, or alphabetically
                if gene_order:
                    sorted_genes = sorted(genes, key=lambda g: gene_order[g])
                else:
                    sorted_genes = sorted(genes)

                # Apply styling based on concordance status
                styled_genes = []
                for gene in sorted_genes:
                    status = status_map.get(gene)
                    if status == ConcordanceStatus.CONCORDANT:
                        styled_genes.append(f"[black on green]{gene}[/]")
                    elif status == ConcordanceStatus.DISCORDANT:
                        styled_genes.append(f"[black on yellow]{gene}[/]")
                    elif status == ConcordanceStatus.NOT_EXECUTED:
                        styled_genes.append(f"[dim]{gene}[/]")
                    else:
                        # No concordance data
                        styled_genes.append(gene)

                genes_str = ", ".join(styled_genes)
                rating_style = {
                    "GREEN": "[green]🟢 GREEN[/green]",
                    "AMBER": "[yellow]🟡 AMBER[/yellow]",
                    "RED": "[red]🔴 RED[/red]",
                }[rating]
                novel_table.add_row(genes_str, rating_style)

        console.print(novel_table)

    if upgrades:
        # Create table for upgrades
        upgrade_table = Table(title="Upgrades", show_header=True, header_style="bold magenta")
        upgrade_table.add_column("Gene", style="cyan")
        upgrade_table.add_column("Current", style="dim")
        upgrade_table.add_column("", style="dim", width=3)
        upgrade_table.add_column("Suggested", style="bold")

        # Build status map if concordance data available (already sorted)
        upgrade_status_map: dict[str, ConcordanceStatus] = {}
        if upgrade_concordance:
            upgrade_status_map = {gs.gene: gs.status for gs in upgrade_concordance.gene_statuses}
            # Use pre-sorted gene order from concordance
            upgrade_gene_order = {
                gs.gene: i for i, gs in enumerate(upgrade_concordance.gene_statuses)
            }
        else:
            upgrade_gene_order = {}

        # Group by transition type
        transitions: dict[str, list[str]] = {}
        for upg in upgrades:
            key = f"{upg.current} → {upg.suggested}"
            if key not in transitions:
                transitions[key] = []
            transitions[key].append(upg.gene)

        # Add rows for each transition type
        for transition in sorted(transitions.keys()):
            genes = transitions[transition]
            current, suggested = transition.split(" → ")

            # Sort using pre-computed order from concordance, or alphabetically
            if upgrade_gene_order:
                sorted_genes = sorted(genes, key=lambda g: upgrade_gene_order[g])
            else:
                sorted_genes = sorted(genes)

            # Apply styling based on concordance status
            styled_genes = []
            for gene in sorted_genes:
                status = upgrade_status_map.get(gene)
                if status == ConcordanceStatus.CONCORDANT:
                    styled_genes.append(f"[black on green]{gene}[/]")
                elif status == ConcordanceStatus.DISCORDANT:
                    styled_genes.append(f"[black on yellow]{gene}[/]")
                elif status == ConcordanceStatus.NOT_EXECUTED:
                    styled_genes.append(f"[dim]{gene}[/]")
                else:
                    # No concordance data
                    styled_genes.append(gene)

            genes_str = ", ".join(styled_genes)

            # Style ratings with colors
            current_styled = {
                "RED": "[red]🔴 RED[/red]",
                "AMBER": "[yellow]🟡 AMBER[/yellow]",
                "GREEN": "[green]🟢 GREEN[/green]",
            }[current]
            suggested_styled = {
                "RED": "[red]🔴 RED[/red]",
                "AMBER": "[yellow]🟡 AMBER[/yellow]",
                "GREEN": "[green]🟢 GREEN[/green]",
            }[suggested]

            upgrade_table.add_row(genes_str, current_styled, "→", suggested_styled)

        console.print(upgrade_table)


def extract_old_format_report(report_path: Path) -> Suggestions:
    """Extract suggestions from old format reports (Aug, Sept H1).

    Format: Paper-by-paper with Overall Rating in <p> tags
    """
    logger.info(f"Parsing report: {report_path}")

    with open(report_path, encoding="utf-8") as f:
        html = f.read()

    novel_genes: list[NovelGene] = []
    upgrades: list[Upgrade] = []

    # Pattern: <h4...><a...>GENE → ...</a></h4>...<p> or <li><strong>Overall Rating</strong>: EMOJI RATING</p> or </li>
    gene_rating_pattern = r"<h4[^>]*>(?:<a[^>]*>)?(\w+)\s+→[^<]+(?:</a>)?</h4>.*?<(?:p|li)><strong>Overall Rating</strong>:\s*[🟢🔴🟡]?\s*(GREEN|AMBER|RED)</(?:p|li)>"

    # Extract novel section (between novel header and known header)
    novel_match = re.search(
        r'<h3 id="novel-gene-disease-associations">.*?(?=<h3 id="known-gene-evidence">)',
        html,
        re.DOTALL,
    )
    if novel_match:
        novel_html = novel_match.group(0)
        # Split into papers (each ends with <hr />)
        for paper in re.split(r"<hr\s*/>", novel_html):
            # Skip if it's a known gene paper
            if "<strong>Known Genes</strong>:" in paper:
                continue
            # Extract all gene-rating pairs
            for gene, rating in re.findall(gene_rating_pattern, paper, re.DOTALL):
                novel_genes.append(NovelGene(gene=gene, suggested_rating=rating))

    # Extract known genes from each rating section
    for section_id, current_rating in [
        ("known-high-red", "RED"),
        ("known-high-amber", "AMBER"),
        ("known-high-green", "GREEN"),
    ]:
        # Extract section (from h5 to next h5 or end)
        section_match = re.search(
            f'<h5 id="{section_id}">.*?(?=<h5 id="known-high-|$)',
            html,
            re.DOTALL,
        )
        if section_match:
            section_html = section_match.group(0)
            # Extract all gene-rating pairs
            for gene, suggested_rating in re.findall(gene_rating_pattern, section_html, re.DOTALL):
                # Only include upgrades (higher confidence), not downgrades
                if _is_upgrade(current_rating, suggested_rating):
                    upgrades.append(
                        Upgrade(gene=gene, current=current_rating, suggested=suggested_rating)
                    )

    return Suggestions(novel_genes=novel_genes, upgrades=upgrades)


def extract_new_format_report(report_path: Path, report_name: str) -> Suggestions:
    """Extract suggestions from new format reports (gene-by-gene with HTML badges).

    Format: Gene-by-gene organization with HTML rating badges
    Used for: September H2 2025, October H1 2025, and later reports

    Args:
        report_path: Path to the HTML report file
        report_name: Human-readable name for logging

    Returns:
        Suggestions extracted from the report
    """
    logger.info(f"Parsing {report_name} report: {report_path}")

    with open(report_path, encoding="utf-8") as f:
        html = f.read()

    novel_genes: list[NovelGene] = []
    upgrades: list[Upgrade] = []

    # Pattern for novel genes: extract gene from article id, skip first badge ("Not in panel"),
    # capture second badge (suggested rating). Handles both formats: bare gene name or <a>-wrapped,
    # and allows extra content (prefill buttons, etc.) before </h3>.
    novel_pattern = r'<article[^>]+id="novel-gene-(\w+)"[^>]*>.*?<h3>.*?<span class="rating-badge[^"]*">[^<]+</span>.*?<span class="rating-badge[^"]*">([^<]+)</span>.*?</h3>'
    for gene, rating in re.findall(novel_pattern, html, re.DOTALL):
        novel_genes.append(NovelGene(gene=gene, suggested_rating=rating.strip().upper()))

    # Pattern for known genes (upgrades): extract gene from article id, capture both rating badges
    # (current and suggested). Handles both formats: bare gene name or <a>-wrapped.
    upgrade_pattern = r'<article[^>]+id="known-gene-(\w+)"[^>]*>.*?<h3>.*?<span class="rating-badge[^"]*">([^<]+)</span>.*?<span class="rating-badge[^"]*">([^<]+)</span>'
    for gene, current, suggested in re.findall(upgrade_pattern, html, re.DOTALL):
        current = current.strip().upper()
        suggested = suggested.strip().upper()
        # Only include upgrades (higher confidence), not downgrades
        if _is_upgrade(current, suggested):
            upgrades.append(Upgrade(gene=gene, current=current, suggested=suggested))

    return Suggestions(novel_genes=novel_genes, upgrades=upgrades)


def extract_sept_h2_2025() -> Suggestions:
    """Extract suggestions from September H2 2025 report (new format, HTML badges).

    Report file: data/concordance/report_2025-10-01.html
    """
    return extract_new_format_report(
        CONCORDANCE_DATA_DIR / "report_2025-10-01.html", "September H2 2025"
    )


def extract_oct_h1_2025() -> Suggestions:
    """Extract suggestions from October H1 2025 report (new format, HTML badges).

    Report file: data/concordance/report_2025-10-15.html
    """
    return extract_new_format_report(
        CONCORDANCE_DATA_DIR / "report_2025-10-15.html", "October H1 2025"
    )


def extract_oct_h2_2025() -> Suggestions:
    """Extract suggestions from October H2 2025 report (new format, HTML badges).

    Report file: data/concordance/report_2025-11-01.html
    """
    return extract_new_format_report(
        CONCORDANCE_DATA_DIR / "report_2025-11-01.html", "October H2 2025"
    )


def extract_nov_h1_2025() -> Suggestions:
    """Extract suggestions from November H1 2025 report (new format, HTML badges).

    Report file: data/concordance/report_2025-11-16.html
    """
    return extract_new_format_report(
        CONCORDANCE_DATA_DIR / "report_2025-11-16.html", "November H1 2025"
    )


def extract_nov_h2_2025() -> Suggestions:
    """Extract suggestions from November H2 2025 report (new format, HTML badges).

    Report file: data/concordance/report_2025-12-01.html
    """
    return extract_new_format_report(
        CONCORDANCE_DATA_DIR / "report_2025-12-01.html", "November H2 2025"
    )


def extract_dec_h1_2025() -> Suggestions:
    """Extract suggestions from December H1 2025 report (new format, HTML badges).

    Report file: data/concordance/report_2025-12-16.html
    """
    return extract_new_format_report(
        CONCORDANCE_DATA_DIR / "report_2025-12-16.html", "December H1 2025"
    )


def fetch_mendeliome_at_date(target_date: datetime) -> dict[str, str]:
    """Fetch Mendeliome panel genes at a specific date (or closest available).

    Args:
        target_date: Target date to fetch panel version

    Returns:
        Dictionary mapping gene name to rating (GREEN/AMBER/RED)
    """
    # Clamp to today if in the future
    today = datetime.now(UTC)
    if target_date > today:
        logger.info(f"  Clamping target date {target_date.date()} to today {today.date()}")
        target_date = today

    logger.info(f"Fetching Mendeliome panel at {target_date.date()}...")

    # Fetch all activities to find the appropriate version
    url = f"{PANELAPP_BASE_URL}/panels/{MENDELIOME_PANEL_ID}/activities/"
    response = httpx.get(url, timeout=60.0)
    response.raise_for_status()
    activities: list[dict[str, Any]] = response.json()

    # Find first version at or after target date
    version = None
    version_date = None
    for activity in sorted(activities, key=lambda x: x.get("created", "")):
        created_str = activity.get("created")
        panel_version = activity.get("panel_version")

        if not created_str or not panel_version or panel_version == "0.0":
            continue

        activity_date = datetime.fromisoformat(created_str.replace("Z", "+00:00"))

        if activity_date >= target_date:
            version = panel_version
            version_date = activity_date
            break

    if not version:
        # If no version found after target date, use the latest
        for activity in sorted(activities, key=lambda x: x.get("created", ""), reverse=True):
            panel_version = activity.get("panel_version")
            if panel_version and panel_version != "0.0":
                version = panel_version
                created_str = activity.get("created")
                if created_str:
                    version_date = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                break

    if not version:
        logger.error("Could not find any valid panel version")
        return {}

    logger.info(
        f"  Using version {version} from {version_date.date() if version_date else 'unknown'}"
    )

    # Fetch panel genes for this version
    url = f"{PANELAPP_BASE_URL}/panels/{MENDELIOME_PANEL_ID}/?version={version}"
    response = httpx.get(url, timeout=60.0)
    response.raise_for_status()
    panel = response.json()

    genes = {}
    for gene in panel.get("genes", []):
        gene_name = gene["entity_name"]
        confidence_level = int(gene["confidence_level"])

        # Map confidence level to rating
        rating_map = {1: "RED", 2: "AMBER", 3: "GREEN"}
        if confidence_level in rating_map:
            # Skip mitochondrial genes
            if not gene_name.startswith("MT-"):
                genes[gene_name] = rating_map[confidence_level]

    logger.info(f"  Found {len(genes)} genes in Mendeliome")
    return genes


def fetch_panel_genes_numeric(version: str, exclude_mito: bool = True) -> dict[str, int]:
    """Fetch all genes and their numeric ratings for a specific panel version.

    Args:
        version: Panel version string (e.g., "1.3383")
        exclude_mito: If True, exclude mitochondrial genes (starting with "MT-")

    Returns:
        Dictionary mapping gene name -> confidence_level (1=RED, 2=AMBER, 3=GREEN)
    """
    url = f"{PANELAPP_BASE_URL}/panels/{MENDELIOME_PANEL_ID}/?version={version}"
    response = httpx.get(url, timeout=60.0)
    response.raise_for_status()
    panel = response.json()

    genes = {}
    for gene in panel.get("genes", []):
        gene_name = gene["entity_name"]
        confidence_level = int(gene["confidence_level"])

        # Filter out spurious gray levels (0)
        if confidence_level not in (1, 2, 3):
            continue

        # Filter out mitochondrial genes if requested
        if exclude_mito and gene_name.startswith("MT-"):
            continue

        genes[gene_name] = confidence_level

    return genes


def fetch_panel_activity(
    start_month: str, end_month: str, activities: list[dict[str, Any]]
) -> list[MonthlyPanelActivity]:
    """Fetch overall panel activity (additions and upgrades) for each month.

    Args:
        start_month: Start month (YYYY-MM)
        end_month: End month (YYYY-MM)
        activities: List of activity dictionaries from API

    Returns:
        List of MonthlyPanelActivity objects
    """
    logger.info(f"Fetching panel activity from {start_month} to {end_month}...")

    # Find earliest version for each month
    month_to_earliest: dict[str, tuple[datetime, str]] = {}

    for activity in activities:
        version = activity.get("panel_version")
        created = activity.get("created")

        if not version or not created or version == "0.0":
            continue

        activity_date = datetime.fromisoformat(created.replace("Z", "+00:00"))
        month_key = activity_date.strftime("%Y-%m")

        # Keep the earliest date/version for this month
        if month_key not in month_to_earliest or activity_date < month_to_earliest[month_key][0]:
            month_to_earliest[month_key] = (activity_date, version)

    # Calculate next month after end_month (needed for comparison)
    end_dt = datetime.strptime(end_month, "%Y-%m")
    next_month = (end_dt + timedelta(days=32)).replace(day=1)
    next_month_str = next_month.strftime("%Y-%m")

    # Fetch snapshots for months in range plus one extra for comparison
    snapshots: list[PanelSnapshot] = []
    for month_key in sorted(month_to_earliest.keys()):
        if start_month <= month_key <= next_month_str:
            version = month_to_earliest[month_key][1]  # Extract only version, not date
            logger.info(f"  Fetching genes for {month_key} (version {version})...")
            genes = fetch_panel_genes_numeric(version, exclude_mito=True)
            snapshots.append(PanelSnapshot(month=month_key, version=version, genes=genes))

    # Analyze changes between consecutive months
    monthly_activity = []

    for i in range(1, len(snapshots)):
        prev_snapshot = snapshots[i - 1]
        curr_snapshot = snapshots[i]
        month = prev_snapshot.month

        # Skip if month is after our range
        if month > end_month:
            break

        # Find additions (by rating)
        prev_genes = prev_snapshot.genes
        curr_genes = curr_snapshot.genes

        additions_by_rating: dict[int, list[str]] = {1: [], 2: [], 3: []}
        for gene_name in sorted(set(curr_genes.keys()) - set(prev_genes.keys())):
            rating = curr_genes[gene_name]
            additions_by_rating[rating].append(gene_name)

        # Find upgrades (by destination rating)
        upgrades_by_rating: dict[int, list[str]] = {1: [], 2: [], 3: []}
        for gene_name in set(curr_genes.keys()) & set(prev_genes.keys()):
            old_rating = prev_genes[gene_name]
            new_rating = curr_genes[gene_name]

            # Only count upgrades (higher rating)
            if new_rating > old_rating:
                upgrades_by_rating[new_rating].append(gene_name)

        monthly_activity.append(
            MonthlyPanelActivity(
                month=month,
                additions_by_rating=additions_by_rating,
                upgrades_by_rating=upgrades_by_rating,
            )
        )

        total_additions = sum(len(genes) for genes in additions_by_rating.values())
        total_upgrades = sum(len(genes) for genes in upgrades_by_rating.values())
        logger.info(f"  {month}: {total_additions} additions, {total_upgrades} upgrades")

    return monthly_activity


def create_interactive_stacked_bar(
    monthly_activity: list[MonthlyPanelActivity],
    data_type: str,
    title: str,
) -> str:
    """Create an interactive stacked bar chart with gene lists on hover.

    Args:
        monthly_activity: List of MonthlyPanelActivity objects
        data_type: Either "additions" or "upgrades"
        title: Chart title

    Returns:
        HTML string containing the Plotly chart
    """
    if not monthly_activity:
        return ""

    months = [activity.month for activity in monthly_activity]

    # Rating colors (lighter, more vibrant shades)
    rating_colors = {1: "#f87171", 2: "#fbbf24", 3: "#4ade80"}
    rating_names = {1: "Red", 2: "Amber", 3: "Green"}

    # Extract data based on type
    data_by_rating: dict[int, tuple[list[int], list[str]]] = {}

    for rating in [1, 2, 3]:
        counts = []
        hover_texts = []

        for activity in monthly_activity:
            if data_type == "additions":
                genes = activity.additions_by_rating[rating]
            else:  # upgrades
                genes = activity.upgrades_by_rating[rating]

            counts.append(len(genes))

            # Create hover text with gene list
            if genes:
                # Show up to 20 genes, then indicate if there are more
                display_genes = genes[:20]
                gene_list = "<br>".join(display_genes)
                if len(genes) > 20:
                    gene_list += f"<br>... and {len(genes) - 20} more"
                hover_texts.append(gene_list)
            else:
                hover_texts.append("None")

        data_by_rating[rating] = (counts, hover_texts)

    # Create figure
    fig = go.Figure()

    # Add bars for each rating (stack from bottom to top: Green, Amber, Red)
    for rating in [3, 2, 1]:
        counts, hover_texts = data_by_rating[rating]

        fig.add_trace(
            go.Bar(
                name=rating_names[rating],
                x=months,
                y=counts,
                marker_color=rating_colors[rating],
                hovertemplate="<b>%{x}</b><br>"
                + f"{rating_names[rating]}: %{{y}}<br>"
                + "%{customdata}<extra></extra>",
                customdata=hover_texts,
            )
        )

    # Update layout
    fig.update_layout(
        barmode="stack",
        title={"text": title, "font": {"size": 16, "color": "#2c3e50"}},
        xaxis_title="",
        yaxis_title="Count",
        hovermode="closest",
        showlegend=False,  # Remove redundant color legend
        font={"family": "Arial, sans-serif", "size": 12},
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
        margin={"l": 60, "r": 40, "t": 60, "b": 80},
    )

    # Grid styling and x-axis configuration
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor="#dee2e6",
        tickmode="array",  # Use explicit tick values
        tickvals=months,  # One tick per month
        ticktext=months,  # Show the month labels
        tickangle=-45,  # Rotate labels 45 degrees
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f0f0f0", zeroline=False)

    # Return as HTML div (not full page)
    return cast(
        str,
        fig.to_html(
            include_plotlyjs="cdn", div_id=f"chart_{data_type}", config={"displayModeBar": False}
        ),
    )


def _sort_gene_statuses[GeneStatusT: (GeneStatus, UpgradeGeneStatus)](
    statuses: list[GeneStatusT],
) -> list[GeneStatusT]:
    """Sort gene statuses by concordance (concordant, discordant, not_executed) then alphabetically."""
    status_order = {
        ConcordanceStatus.CONCORDANT: 0,
        ConcordanceStatus.DISCORDANT: 1,
        ConcordanceStatus.NOT_EXECUTED: 2,
    }
    return sorted(statuses, key=lambda s: (status_order[s.status], s.gene))


def check_concordance(
    suggestions: Suggestions, mendeliome: dict[str, str]
) -> tuple[NovelConcordance, UpgradeConcordance]:
    """Check concordance between suggestions and actual Mendeliome state.

    Args:
        suggestions: Extracted suggestions from report
        mendeliome: Actual Mendeliome state {gene: rating}

    Returns:
        Tuple of (novel_concordance, upgrade_concordance) with sorted gene_statuses
    """
    # Novel genes - track per-gene status
    novel_gene_statuses: list[GeneStatus] = []
    novel_suggested = len(suggestions.novel_genes)
    novel_added = 0
    novel_added_concordant = 0

    for suggestion in suggestions.novel_genes:
        gene = suggestion.gene
        suggested_rating = suggestion.suggested_rating

        if gene in mendeliome:
            novel_added += 1
            actual_rating = mendeliome[gene]
            if actual_rating == suggested_rating:
                novel_added_concordant += 1
                status = ConcordanceStatus.CONCORDANT
            else:
                status = ConcordanceStatus.DISCORDANT

            novel_gene_statuses.append(
                GeneStatus(
                    gene=gene,
                    status=status,
                    suggested_rating=suggested_rating,
                    actual_rating=actual_rating,
                )
            )
        else:
            novel_gene_statuses.append(
                GeneStatus(
                    gene=gene,
                    status=ConcordanceStatus.NOT_EXECUTED,
                    suggested_rating=suggested_rating,
                    actual_rating=None,
                )
            )

    # Upgrades - track per-gene status
    upgrade_gene_statuses: list[UpgradeGeneStatus] = []
    upgrade_suggested = len(suggestions.upgrades)
    upgrade_executed = 0
    upgrade_executed_concordant = 0

    for upgrade in suggestions.upgrades:
        gene = upgrade.gene
        current_rating = upgrade.current
        suggested_rating = upgrade.suggested

        if gene in mendeliome:
            actual_rating = mendeliome[gene]
            # Check if rating changed from current
            if actual_rating != current_rating:
                upgrade_executed += 1
                # Check if it matches our suggestion
                if actual_rating == suggested_rating:
                    upgrade_executed_concordant += 1
                    status = ConcordanceStatus.CONCORDANT
                else:
                    status = ConcordanceStatus.DISCORDANT
            else:
                status = ConcordanceStatus.NOT_EXECUTED

            upgrade_gene_statuses.append(
                UpgradeGeneStatus(
                    gene=gene,
                    status=status,
                    current_rating=current_rating,
                    suggested_rating=suggested_rating,
                    actual_rating=actual_rating,
                )
            )
        else:
            # Gene not in panel at all
            upgrade_gene_statuses.append(
                UpgradeGeneStatus(
                    gene=gene,
                    status=ConcordanceStatus.NOT_EXECUTED,
                    current_rating=current_rating,
                    suggested_rating=suggested_rating,
                    actual_rating=current_rating,  # Assume it stayed the same
                )
            )

    # Sort gene statuses by concordance outcome
    novel_gene_statuses = _sort_gene_statuses(novel_gene_statuses)
    upgrade_gene_statuses = _sort_gene_statuses(upgrade_gene_statuses)

    novel_concordance = NovelConcordance(
        gene_statuses=novel_gene_statuses,
        suggested=novel_suggested,
        added=novel_added,
        concordant=novel_added_concordant,
    )

    upgrade_concordance = UpgradeConcordance(
        gene_statuses=upgrade_gene_statuses,
        suggested=upgrade_suggested,
        executed=upgrade_executed,
        concordant=upgrade_executed_concordant,
    )

    return novel_concordance, upgrade_concordance


@app.callback(invoke_without_command=True)
def analyze() -> None:
    """Analyze concordance for all reporting periods."""
    logger.info("Starting concordance analysis...")

    results: list[ConcordanceResults] = []

    # August 2025 (report released Sept 1, check 2 months later on Nov 1)
    logger.info("\n=== August 2025 ===")
    aug_suggestions = extract_old_format_report(CONCORDANCE_DATA_DIR / "report_2025-09-01.html")
    aug_target_date = datetime(2025, 11, 1, tzinfo=UTC)
    aug_mendeliome = fetch_mendeliome_at_date(aug_target_date)
    aug_novel, aug_upgrade = check_concordance(aug_suggestions, aug_mendeliome)

    # Display suggestions with concordance styling
    log_suggestions_summary(
        aug_suggestions.novel_genes,
        aug_suggestions.upgrades,
        aug_novel,
        aug_upgrade,
    )

    results.append(
        ConcordanceResults(
            month="August 2025",
            novel_suggested=aug_novel.suggested,
            novel_added=aug_novel.added,
            novel_added_concordant=aug_novel.concordant,
            upgrade_suggested=aug_upgrade.suggested,
            upgrade_executed=aug_upgrade.executed,
            upgrade_executed_concordant=aug_upgrade.concordant,
        )
    )

    # September 2025 (merge H1 and H2, check 2 months later on Dec 1)
    logger.info("\n=== September 2025 ===")
    # Parse both September reports
    sept_h1_suggestions = extract_old_format_report(CONCORDANCE_DATA_DIR / "report_2025-09-16.html")
    sept_h2_suggestions = extract_sept_h2_2025()
    # Merge them into a single period
    sept_suggestions = merge_suggestions([sept_h1_suggestions, sept_h2_suggestions])
    # Check concordance against December 1st state
    sept_target_date = datetime(2025, 12, 1, tzinfo=UTC)
    sept_mendeliome = fetch_mendeliome_at_date(sept_target_date)
    sept_novel, sept_upgrade = check_concordance(sept_suggestions, sept_mendeliome)

    # Display suggestions with concordance styling
    log_suggestions_summary(
        sept_suggestions.novel_genes,
        sept_suggestions.upgrades,
        sept_novel,
        sept_upgrade,
    )

    results.append(
        ConcordanceResults(
            month="September 2025",
            novel_suggested=sept_novel.suggested,
            novel_added=sept_novel.added,
            novel_added_concordant=sept_novel.concordant,
            upgrade_suggested=sept_upgrade.suggested,
            upgrade_executed=sept_upgrade.executed,
            upgrade_executed_concordant=sept_upgrade.concordant,
        )
    )

    # October 2025 (merge H1 and H2, check 2 months later on Jan 1)
    logger.info("\n=== October 2025 ===")
    # Parse both October reports
    oct_h1_suggestions = extract_oct_h1_2025()
    oct_h2_suggestions = extract_oct_h2_2025()
    # Merge them into a single period
    oct_suggestions = merge_suggestions([oct_h1_suggestions, oct_h2_suggestions])
    # Check concordance against January 1st state
    oct_target_date = datetime(2026, 1, 1, tzinfo=UTC)
    oct_mendeliome = fetch_mendeliome_at_date(oct_target_date)
    oct_novel, oct_upgrade = check_concordance(oct_suggestions, oct_mendeliome)

    # Display suggestions with concordance styling
    log_suggestions_summary(
        oct_suggestions.novel_genes,
        oct_suggestions.upgrades,
        oct_novel,
        oct_upgrade,
    )

    results.append(
        ConcordanceResults(
            month="October 2025",
            novel_suggested=oct_novel.suggested,
            novel_added=oct_novel.added,
            novel_added_concordant=oct_novel.concordant,
            upgrade_suggested=oct_upgrade.suggested,
            upgrade_executed=oct_upgrade.executed,
            upgrade_executed_concordant=oct_upgrade.concordant,
        )
    )

    # November 2025 (merge H1 and H2, check 2 months later on Feb 1)
    logger.info("\n=== November 2025 ===")
    # Parse both November reports
    nov_h1_suggestions = extract_nov_h1_2025()
    nov_h2_suggestions = extract_nov_h2_2025()
    # Merge them into a single period
    nov_suggestions = merge_suggestions([nov_h1_suggestions, nov_h2_suggestions])
    # Check concordance against February 1st state
    nov_target_date = datetime(2026, 2, 1, tzinfo=UTC)
    nov_mendeliome = fetch_mendeliome_at_date(nov_target_date)
    nov_novel, nov_upgrade = check_concordance(nov_suggestions, nov_mendeliome)

    # Display suggestions with concordance styling
    log_suggestions_summary(
        nov_suggestions.novel_genes,
        nov_suggestions.upgrades,
        nov_novel,
        nov_upgrade,
    )

    results.append(
        ConcordanceResults(
            month="November 2025",
            novel_suggested=nov_novel.suggested,
            novel_added=nov_novel.added,
            novel_added_concordant=nov_novel.concordant,
            upgrade_suggested=nov_upgrade.suggested,
            upgrade_executed=nov_upgrade.executed,
            upgrade_executed_concordant=nov_upgrade.concordant,
        )
    )

    # December H1 2025 (check 2 months later on Mar 1)
    logger.info("\n=== December H1 2025 ===")
    dec_h1_suggestions = extract_dec_h1_2025()
    # Check concordance against March 1st state
    dec_target_date = datetime(2026, 3, 1, tzinfo=UTC)
    dec_mendeliome = fetch_mendeliome_at_date(dec_target_date)
    dec_novel, dec_upgrade = check_concordance(dec_h1_suggestions, dec_mendeliome)

    # Display suggestions with concordance styling
    log_suggestions_summary(
        dec_h1_suggestions.novel_genes,
        dec_h1_suggestions.upgrades,
        dec_novel,
        dec_upgrade,
    )

    results.append(
        ConcordanceResults(
            month="December H1 2025",
            novel_suggested=dec_novel.suggested,
            novel_added=dec_novel.added,
            novel_added_concordant=dec_novel.concordant,
            upgrade_suggested=dec_upgrade.suggested,
            upgrade_executed=dec_upgrade.executed,
            upgrade_executed_concordant=dec_upgrade.concordant,
        )
    )

    # Fetch overall panel activity for the past 12 months
    logger.info("\nFetching overall panel activity...")

    # Fetch activities once (reuse from concordance checks)
    url = f"{PANELAPP_BASE_URL}/panels/{MENDELIOME_PANEL_ID}/activities/"
    response = httpx.get(url, timeout=60.0)
    response.raise_for_status()
    activities: list[dict[str, Any]] = response.json()

    # Get activity for the last 2 years
    end_month = datetime.now(UTC).replace(day=1) - timedelta(days=1)  # Last complete month
    start_month = end_month - timedelta(days=730)  # 2 years

    monthly_activity = fetch_panel_activity(
        start_month.strftime("%Y-%m"),
        end_month.strftime("%Y-%m"),
        activities,
    )

    # Generate interactive charts
    logger.info("Generating interactive charts...")
    additions_chart_html = create_interactive_stacked_bar(
        monthly_activity,
        "additions",
        "New Genes",
    )

    upgrades_chart_html = create_interactive_stacked_bar(
        monthly_activity,
        "upgrades",
        "Rating Upgrades",
    )

    # Generate HTML report
    logger.info("\nGenerating HTML concordance report...")

    # Prepare period data (gene_statuses already sorted by check_concordance())
    # Sort from most recent to oldest
    periods = [
        {
            "id": "december-h1",
            "name": "December H1 2025",
            "novel_concordance": dec_novel,
            "upgrade_concordance": dec_upgrade,
        },
        {
            "id": "november",
            "name": "November 2025",
            "novel_concordance": nov_novel,
            "upgrade_concordance": nov_upgrade,
        },
        {
            "id": "october",
            "name": "October 2025",
            "novel_concordance": oct_novel,
            "upgrade_concordance": oct_upgrade,
        },
        {
            "id": "september",
            "name": "September 2025",
            "novel_concordance": sept_novel,
            "upgrade_concordance": sept_upgrade,
        },
        {
            "id": "august",
            "name": "August 2025",
            "novel_concordance": aug_novel,
            "upgrade_concordance": aug_upgrade,
        },
    ]

    # Load template and CSS
    template_dir = Path("templates")
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("concordance_report.html")

    css_file = template_dir / "concordance_report_styles.css"
    custom_css = css_file.read_text()

    # Render HTML (reverse results to show most recent first)
    html_content = template.render(
        results=list(reversed(results)),
        periods=periods,
        custom_css=custom_css,
        additions_chart_html=additions_chart_html,
        upgrades_chart_html=upgrades_chart_html,
    )

    # Save HTML report
    output_path = Path("data/concordance_report.html")
    output_path.write_text(html_content)

    # Print summary (most recent first)
    console.print("\n[bold]Concordance Analysis Summary[/bold]\n")
    for result in reversed(results):
        console.print(f"[cyan]{result.month}:[/cyan]")
        console.print(f"  Novel genes suggested: {result.novel_suggested}")
        console.print(f"  Novel genes added: {result.novel_added}")
        console.print(
            f"  Novel genes added with concordant rating: {result.novel_added_concordant}"
        )
        console.print(f"  Upgrades suggested: {result.upgrade_suggested}")
        console.print(f"  Upgrades executed: {result.upgrade_executed}")
        console.print(
            f"  Upgrades executed with concordant target: {result.upgrade_executed_concordant}"
        )
        console.print()

    console.print(f"[green]✓[/green] HTML report saved to {output_path}")


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
