#!/usr/bin/env python3
"""Scan PanelApp curator review comments for mechanism-of-disease mentions."""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import boto3
import jinja2
import typer
from botocore.config import Config
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.providers.bedrock import BedrockModelProfile, BedrockProvider

from palit.panelapp_client import PanelAppClient
from palit.panelapp_integration import INCIDENTALOME_PANEL_ID, MENDELIOME_PANEL_ID
from palit.progress import LoggingProgress as Progress

logger = logging.getLogger(__name__)

app = typer.Typer(help="Scan PanelApp reviews for mechanism-of-disease mentions.")

SCAN_PANEL_IDS = [MENDELIOME_PANEL_ID, INCIDENTALOME_PANEL_ID]
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


def _create_agent(model_id: str, concurrency: int) -> Agent[None, MechanismScanResult]:
    session = boto3.Session()
    bedrock_client = session.client(
        "bedrock-runtime",
        config=Config(max_pool_connections=concurrency, read_timeout=300, connect_timeout=60),
    )
    model = BedrockConverseModel(
        model_id,
        provider=BedrockProvider(bedrock_client=bedrock_client),
        profile=BedrockModelProfile(
            bedrock_supports_tool_choice=False,
            bedrock_send_back_thinking_parts=True,
        ),
    )
    return Agent(
        model,
        output_type=MechanismScanResult,
        retries=3,
        instructions="Always return your response by calling the final_result tool.",
        model_settings=BedrockModelSettings(
            max_tokens=16_000,
            bedrock_additional_model_requests_fields={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
        ),
    )


async def _scan_gene(
    agent: Agent[None, MechanismScanResult],
    template: jinja2.Template,
    gene: GeneText,
    raw_dir: Path,
) -> MechanismScanResult:
    prompt = template.render(gene_symbol=gene.gene_symbol, review_text=gene.text)
    t0 = time.monotonic()
    result = await agent.run(prompt)
    elapsed = time.monotonic() - t0
    logger.info(
        "%s: GoF=%s DN=%s (%.1fs)",
        gene.gene_symbol,
        result.output.gain_of_function,
        result.output.dominant_negative,
        elapsed,
    )
    # Store raw LLM conversation
    raw_path = raw_dir / f"{gene.hgnc_id}.json"
    raw_path.write_bytes(result.all_messages_json())
    return result.output


async def scan_all(
    gene_texts: list[GeneText],
    agent: Agent[None, MechanismScanResult],
    output_dir: Path,
    concurrency: int,
) -> None:
    template = _load_prompt_template()
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    gof_genes: list[str] = []
    dn_genes: list[str] = []
    failures: list[GeneFailure] = []
    semaphore = asyncio.Semaphore(concurrency)

    with Progress() as progress:
        task = progress.add_task("Scanning genes", total=len(gene_texts))

        async def process(gene: GeneText) -> None:
            async with semaphore:
                try:
                    result = await _scan_gene(agent, template, gene, raw_dir)
                    if result.gain_of_function:
                        gof_genes.append(gene.gene_symbol)
                    if result.dominant_negative:
                        dn_genes.append(gene.gene_symbol)
                except Exception:
                    logger.exception("Failed to scan %s (HGNC:%d)", gene.gene_symbol, gene.hgnc_id)
                    failures.append(
                        GeneFailure(gene_symbol=gene.gene_symbol, error=str(gene.hgnc_id))
                    )
                finally:
                    progress.update(task, advance=1)

        async with asyncio.TaskGroup() as tg:
            for gene in gene_texts:
                tg.create_task(process(gene))

    # Write output files
    gof_path = output_dir / "gain_of_function.txt"
    dn_path = output_dir / "dominant_negative.txt"
    gof_path.write_text("\n".join(sorted(gof_genes)) + "\n" if gof_genes else "")
    dn_path.write_text("\n".join(sorted(dn_genes)) + "\n" if dn_genes else "")

    # Summary
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
            logger.error("  %s (HGNC:%s)", f.gene_symbol, f.error)


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
    model_id: str = typer.Option(
        "au.anthropic.claude-opus-4-6-v1", "--model-id", help="Bedrock model ID"
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

    agent = _create_agent(model_id, concurrency)
    asyncio.run(scan_all(gene_texts, agent, output_dir, concurrency))
