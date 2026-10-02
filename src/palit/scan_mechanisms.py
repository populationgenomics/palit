#!/usr/bin/env python3
"""Scan PanelApp curator review comments for mechanism-of-disease mentions."""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import jinja2
import typer
from anthropic import AsyncAnthropic
from anthropic.types.output_config_param import OutputConfigParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from palit.llm import (
    FALLBACK_MODEL,
    MODEL,
    Effort,
    ImmediateTransport,
    LlmRequest,
    ResultStatus,
    json_output_config,
    make_client,
    parse_json_output,
)
from palit.panelapp_client import PanelAppClient
from palit.panelapp_integration import INCIDENTALOME_PANEL_ID, MENDELIOME_PANEL_ID
from palit.progress import LoggingProgress as Progress

logger = logging.getLogger(__name__)

app = typer.Typer(help="Scan PanelApp reviews for mechanism-of-disease mentions.")

SCAN_PANEL_IDS = [MENDELIOME_PANEL_ID, INCIDENTALOME_PANEL_ID]
STAGE = "scan_mechanisms"
EFFORT: Effort = "high"
MAX_TOKENS = 16000
PROMPT_TEMPLATE_PATH = Path(__file__).parents[2] / "prompts" / "mechanism_scan.j2"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneText:
    hgnc_id: int
    gene_symbol: str  # Display symbol used in output files (PanelApp gene_symbol or geneName).
    text: str


class TextSource(Protocol):
    def get_gene_texts(self) -> list[GeneText]:
        """Return texts to scan, one entry per gene."""
        ...


class MechanismScanResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gain_of_function: bool = Field(
        description=(
            "Whether the text asserts gain-of-function as a mechanism of disease for this gene"
        )
    )
    dominant_negative: bool = Field(
        description=(
            "Whether the text asserts a dominant negative effect "
            "as a mechanism of disease for this gene"
        )
    )


@dataclass
class GeneEvaluations:
    """Cached PanelApp evaluation API responses for a single gene."""

    hgnc_id: int
    gene_symbol: str
    evaluations: list[dict[str, Any]]


@dataclass
class EvaluationsCache:
    """All fetched gene evaluations, serialisable to/from JSON."""

    genes: list[GeneEvaluations]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "hgnc_id": g.hgnc_id,
                "gene_symbol": g.gene_symbol,
                "evaluations": g.evaluations,
            }
            for g in self.genes
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved evaluations cache (%d genes) to %s", len(self.genes), path)

    @staticmethod
    def load(path: Path) -> "EvaluationsCache":
        logger.info("Loading evaluations cache from %s", path)
        with open(path) as f:
            data: list[dict[str, Any]] = json.load(f)
        return EvaluationsCache(
            genes=[
                GeneEvaluations(
                    hgnc_id=entry["hgnc_id"],
                    gene_symbol=entry["gene_symbol"],
                    evaluations=entry["evaluations"],
                )
                for entry in data
            ]
        )


@dataclass
class GeneFailure:
    gene_symbol: str
    error: str


# ---------------------------------------------------------------------------
# Text source: PanelApp evaluations
# ---------------------------------------------------------------------------


def _format_evaluations(evaluations: list[dict[str, Any]]) -> str:
    """Extract free-text comments from evaluations."""
    texts: list[str] = []
    for review in evaluations:
        for comment in review.get("comments", []):
            text = comment.get("comment", "").strip()
            if text:
                texts.append(text)
    return "\n\n".join(texts)


class PanelAppEvaluationSource:
    """Fetch gene evaluation comments from PanelApp API."""

    def __init__(
        self,
        panelapp_client: PanelAppClient,
        cache_path: Path,
        no_cache: bool = False,
    ) -> None:
        self._client = panelapp_client
        self._cache_path = cache_path
        self._no_cache = no_cache

    def _fetch_evaluations(self) -> EvaluationsCache:
        """Enumerate genes across scan panels and fetch their evaluations."""
        panel_data_cache = self._client._ensure_cache_loaded()

        # Collect unique genes with their panel memberships
        genes: dict[int, str] = {}  # hgnc_id -> gene_symbol
        gene_panels: dict[int, list[int]] = {}  # hgnc_id -> [panel_ids]
        for panel_id in SCAN_PANEL_IDS:
            panel_data = panel_data_cache[panel_id]
            for entity in panel_data.get("genes", []) + panel_data.get("strs", []):
                gene_data = entity.get("gene_data", {})
                hgnc_id_str = gene_data.get("hgnc_id")
                if not hgnc_id_str:
                    continue
                hgnc_id = int(hgnc_id_str.removeprefix("HGNC:"))
                genes[hgnc_id] = gene_data["gene_symbol"]
                gene_panels.setdefault(hgnc_id, []).append(panel_id)

        logger.info("Enumerated %d unique genes across %d panels", len(genes), len(SCAN_PANEL_IDS))

        results: list[GeneEvaluations] = []
        with Progress() as progress:
            task = progress.add_task("Fetching evaluations", total=len(genes))
            for hgnc_id, symbol in genes.items():
                all_evals: list[dict[str, Any]] = []
                for panel_id in gene_panels.get(hgnc_id, []):
                    all_evals.extend(self._client.get_gene_evaluations(panel_id, hgnc_id))
                results.append(GeneEvaluations(hgnc_id, symbol, all_evals))
                progress.update(task, advance=1)

        return EvaluationsCache(genes=results)

    def get_gene_texts(self) -> list[GeneText]:
        if not self._no_cache and self._cache_path.exists():
            cache = EvaluationsCache.load(self._cache_path)
        else:
            cache = self._fetch_evaluations()
            cache.save(self._cache_path)

        gene_texts: list[GeneText] = []
        skipped = 0
        for gene in cache.genes:
            text = _format_evaluations(gene.evaluations)
            if not text:
                skipped += 1
                continue
            gene_texts.append(GeneText(gene.hgnc_id, gene.gene_symbol, text))

        logger.info("Built %d gene texts (%d skipped — no comments)", len(gene_texts), skipped)
        return gene_texts


# ---------------------------------------------------------------------------
# Text source: Gene profiles (curation-service JSON dump)
# ---------------------------------------------------------------------------


class GeneProfileSource:
    """Read gene profiles from a curation-service JSON dump."""

    def __init__(self, profiles_path: Path) -> None:
        self._path = profiles_path

    def get_gene_texts(self) -> list[GeneText]:
        with open(self._path) as f:
            data: list[dict[str, Any]] = json.load(f)

        gene_texts: list[GeneText] = []
        skipped = 0
        for entry in data:
            abstract = entry.get("geneAbstract") or ""
            if not abstract.strip():
                skipped += 1
                continue
            hgnc_id = int(entry["hgncId"].removeprefix("HGNC:"))
            gene_texts.append(GeneText(hgnc_id, entry["geneName"], abstract))

        logger.info(
            "Loaded %d gene profiles (%d skipped — no abstract) from %s",
            len(gene_texts),
            skipped,
            self._path,
        )
        return gene_texts


# ---------------------------------------------------------------------------
# LLM scanning
# ---------------------------------------------------------------------------


def _load_prompt_template() -> jinja2.Template:
    template_text = PROMPT_TEMPLATE_PATH.read_text()
    return jinja2.Template(template_text)


def scan_request(
    gene: GeneText, template: jinja2.Template, output_config: OutputConfigParam, model: str
) -> LlmRequest:
    return LlmRequest(
        subject=str(gene.hgnc_id),
        params={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {
                    "role": "user",
                    "content": template.render(gene_symbol=gene.gene_symbol, review_text=gene.text),
                }
            ],
            "output_config": output_config,
        },
    )


async def scan_all(
    gene_texts: list[GeneText],
    client: AsyncAnthropic,
    output_dir: Path,
    concurrency: int,
) -> None:
    """Scan every gene's text; genes MODEL refuses are scanned again on FALLBACK_MODEL."""
    template = _load_prompt_template()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_config = json_output_config(MechanismScanResult.model_json_schema(), EFFORT)
    by_id = {str(gene.hgnc_id): gene for gene in gene_texts}
    transport = ImmediateTransport(client, workers=concurrency)
    results = await transport.run(
        STAGE, 1, [scan_request(gene, template, output_config, MODEL) for gene in gene_texts]
    )
    refused = [by_id[result.subject] for result in results if result.goes_to_fallback]
    if refused:
        logger.warning(
            "%s refused %d genes; scanning them with %s", MODEL, len(refused), FALLBACK_MODEL
        )
        results = [result for result in results if not result.goes_to_fallback]
        results += await transport.run(
            STAGE,
            1,
            [scan_request(gene, template, output_config, FALLBACK_MODEL) for gene in refused],
        )

    gof_genes: list[str] = []
    dn_genes: list[str] = []
    failures: list[GeneFailure] = []
    for result in results:
        gene = by_id[result.subject]
        if result.status != ResultStatus.SUCCEEDED or result.message is None:
            failures.append(
                GeneFailure(
                    gene_symbol=gene.gene_symbol, error=f"{result.status.value} by {result.model}"
                )
            )
            continue
        try:
            scan = MechanismScanResult.model_validate(parse_json_output(result.message))
        except (ValueError, ValidationError) as e:
            failures.append(GeneFailure(gene_symbol=gene.gene_symbol, error=str(e)))
            continue
        (raw_dir / f"{gene.hgnc_id}.json").write_text(result.message.to_json())
        logger.info(
            "%s: GoF=%s DN=%s", gene.gene_symbol, scan.gain_of_function, scan.dominant_negative
        )
        if scan.gain_of_function:
            gof_genes.append(gene.gene_symbol)
        if scan.dominant_negative:
            dn_genes.append(gene.gene_symbol)

    gof_path = output_dir / "gain_of_function.txt"
    dn_path = output_dir / "dominant_negative.txt"
    gof_path.write_text("\n".join(sorted(gof_genes)) + "\n" if gof_genes else "")
    dn_path.write_text("\n".join(sorted(dn_genes)) + "\n" if dn_genes else "")

    logger.info(
        "Scan complete: %d genes scanned, %d gain-of-function, %d dominant-negative, %d errors",
        len(gene_texts),
        len(gof_genes),
        len(dn_genes),
        len(failures),
    )
    if failures:
        logger.error("Failed genes:")
        for f in sorted(failures, key=lambda x: x.gene_symbol):
            logger.error("  %s: %s", f.gene_symbol, f.error)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def scan(
    source: str = typer.Option(
        "panelapp-evaluations",
        "--source",
        help="Text source: panelapp-evaluations or gene-profiles",
    ),
    panel_date: str = typer.Option(
        None,
        "--panel-date",
        help="PanelApp snapshot date (YYYY-MM-DD), required for panelapp source",
    ),
    profiles_path: Path = typer.Option(
        None,
        "--profiles-path",
        help="Path to gene profiles JSON dump, required for gene-profiles source",
    ),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Output directory (default: data/mechanism_scan/{source})"
    ),
    concurrency: int = typer.Option(30, "--concurrency", help="Max parallel LLM calls"),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Force re-fetch of PanelApp evaluations"
    ),
) -> None:
    """Scan gene curation text for gain-of-function and dominant negative mentions."""
    if output_dir is None:
        output_dir = Path("data/mechanism_scan") / source
    output_dir.mkdir(parents=True, exist_ok=True)

    text_source: TextSource
    if source == "panelapp-evaluations":
        if not panel_date:
            raise typer.BadParameter("--panel-date is required for panelapp-evaluations source")
        cache_path = output_dir / "evaluations_cache.json"
        panelapp_client = PanelAppClient(panel_date=panel_date)
        text_source = PanelAppEvaluationSource(panelapp_client, cache_path, no_cache=no_cache)
    elif source == "gene-profiles":
        if not profiles_path:
            raise typer.BadParameter("--profiles-path is required for gene-profiles source")
        text_source = GeneProfileSource(profiles_path)
    else:
        raise typer.BadParameter(f"Unknown source: {source}")

    gene_texts = text_source.get_gene_texts()

    if not gene_texts:
        logger.warning("No genes with evaluation comments found — nothing to scan")
        return

    logger.info("Scanning %d genes for mechanism-of-disease mentions", len(gene_texts))

    async def run() -> None:
        await scan_all(gene_texts, make_client(), output_dir, concurrency)

    asyncio.run(run())
