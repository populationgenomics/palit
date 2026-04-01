#!/usr/bin/env python3
"""PanelApp Literature - Main CLI entry point."""

import logging

import typer
from rich.logging import RichHandler

# Always-available commands (no optional dependencies)
from palit import (
    analyze_concordance,
    annotate_pdfs,
    discover_citations,
    docling,
    download_papers,
    fetch_variant_frequencies,
    generate_report,
    ingest_preprints,
    ingest_pubmed,
    normalize_variants,
    scan_mechanisms,
)

app = typer.Typer(
    name="palit",
    help="PanelApp Literature - LLM-based literature curation for rare disease genes",
)


def setup_logging(log_level: str) -> None:
    """Configure logging for all commands."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(name)s - %(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


@app.callback()
def main(
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
) -> None:
    """PanelApp Literature CLI."""
    setup_logging(log_level)


# Register always-available commands
app.add_typer(ingest_pubmed.app, name="ingest-pubmed")
app.add_typer(ingest_preprints.app, name="ingest-preprints")
app.add_typer(discover_citations.app, name="discover-citations")
app.add_typer(normalize_variants.app, name="normalize-variants")
app.add_typer(fetch_variant_frequencies.app, name="fetch-variant-frequencies")
app.add_typer(annotate_pdfs.app, name="annotate-pdfs")
app.add_typer(analyze_concordance.app, name="analyze-concordance")
app.add_typer(download_papers.app, name="download-papers")
app.add_typer(docling.app, name="docling")
app.add_typer(generate_report.app, name="generate-report")
app.add_typer(scan_mechanisms.app, name="scan-mechanisms")

# Conditionally register ML commands (require vLLM, torch, transformers)
try:
    from palit import (
        assess_genes,
        assess_relevance,
        expand_literature,
        extract_evidence,
        match_panels,
        reduce_literature,
        screen_pubmed,
    )
    from palit.screening_classifier import cli as screening_cli

    app.add_typer(assess_relevance.app, name="assess-relevance")
    app.add_typer(extract_evidence.app, name="extract-evidence")
    app.add_typer(assess_genes.app, name="assess-genes")
    app.add_typer(match_panels.app, name="match-panels")
    app.add_typer(expand_literature.app, name="expand-literature")
    app.add_typer(reduce_literature.app, name="reduce-literature")
    app.add_typer(screen_pubmed.app, name="screen-pubmed")
    app.add_typer(screening_cli.app, name="screening-classifier")
except ImportError:
    pass  # ML commands not available without: uv sync --extra ml


if __name__ == "__main__":
    app()
