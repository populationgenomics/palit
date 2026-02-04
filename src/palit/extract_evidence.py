#!/usr/bin/env python3
"""Extract evidence from full-text papers using vLLM inference."""

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from jinja2 import Environment, FileSystemLoader
from openai_harmony import HarmonyEncoding
from tqdm import tqdm

from palit.docling import serialize_with_bbox_ids
from palit.llm import HarmonyBatchProcessor, PromptResult
from palit.panelapp_client import (
    PanelAppClient,
    PanelGeneData,
    ResolvedGene,
    format_panel_for_prompt,
    resolve_gene_symbols,
)

app = typer.Typer(help="Extract structured evidence from full-text papers using vLLM")
logger = logging.getLogger(__name__)


@dataclass
class PaperPrompt:
    """A paper prepared for processing with its metadata."""

    prompt: str
    pmid: int
    bbox_mapping: dict[int, dict[str, Any]]


@dataclass
class DeepAnalysisPreparation:
    """Result from preparing deep analysis prompts."""

    paper_prompts: list[PaperPrompt]
    missing_pmids: list[int]


def validate_box_ids(data: Any, valid_box_ids: set[int]) -> bool:
    """
    Recursively check all 'box_id' fields in data structure are valid.
    Returns False immediately on first invalid box_id (fail-fast).

    Args:
        data: JSON data structure (dict, list, or primitive)
        valid_box_ids: Set of valid box IDs for this paper

    Returns:
        True if all box_ids are valid, False otherwise
    """

    def recurse(obj: Any) -> bool:
        if isinstance(obj, dict):
            # Check if this dict has a 'box_id' field
            if "box_id" in obj:
                box_id = obj["box_id"]
                if isinstance(box_id, int) and box_id not in valid_box_ids:
                    return False
            # Recurse into all values
            for value in obj.values():
                if not recurse(value):
                    return False
        elif isinstance(obj, list):
            # Recurse into all list items
            for item in obj:
                if not recurse(item):
                    return False
        # Primitives: return True
        return True

    return recurse(data)


class PaperBatchProcessor:
    """Handle database operations for evidence extraction."""

    def __init__(self, db_path: Path):
        """Initialize with database path."""
        self.db_path = db_path

    def get_papers_for_deep_analysis(self) -> list[dict[str, Any]]:
        """
        Get papers that have been downloaded and haven't been processed for evidence extraction.

        Returns list of papers with pmid, title, abstract.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT pmid, entrez_date, title, abstract
                FROM papers
                WHERE download_status IN ('pmc_downloaded', 'manual_downloaded')
                AND evidence_extraction_json IS NULL
                ORDER BY pmid
                """
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_paper_evidence_extraction(
        self,
        paper_prompts: list[PaperPrompt],
        results: list[PromptResult | None],
        panel_data: PanelGeneData,
    ) -> None:
        """
        Update evidence extraction and automatically sync gene_mentions with source.

        Args:
            paper_prompts: List of PaperPrompt objects with PMIDs and bbox mappings
            results: List of PromptResult objects or None, same order as paper_prompts
            panel_data: PanelApp gene data for symbol resolution
        """
        if not paper_prompts or not results:
            return

        successful_updates = 0
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Update each paper with its result
            for paper_prompt, result in zip(paper_prompts, results, strict=True):
                if result is not None:
                    pmid = paper_prompt.pmid
                    bbox_mapping_str = json.dumps(paper_prompt.bbox_mapping)

                    # Get source_type to determine gene_mentions source
                    cursor.execute("SELECT source_type FROM papers WHERE pmid = ?", (pmid,))
                    row = cursor.fetchone()
                    if not row:
                        raise ValueError(f"Paper PMID {pmid} not found in database")

                    source_type = row["source_type"]
                    if not source_type:
                        raise ValueError(f"Paper PMID {pmid} has NULL source_type")

                    # Map source_type to gene_mentions source
                    if source_type == "initial":
                        gene_source = "recent_evidence"
                    elif source_type == "expansion":
                        gene_source = "expansion_evidence"
                    else:
                        raise ValueError(
                            f"Unknown source_type '{source_type}' for PMID {pmid}. "
                            f"Expected 'initial' or 'expansion'"
                        )

                    # Normalize gene names to uppercase for consistency with gene_mentions
                    if "gene_evaluations" in result.parsed_json:
                        for gene_eval in result.parsed_json["gene_evaluations"]:
                            if "gene" in gene_eval:
                                gene_eval["gene"] = gene_eval["gene"].upper()

                    # Update papers table
                    cursor.execute(
                        """
                        UPDATE papers
                        SET evidence_extraction_raw = ?,
                            evidence_extraction_json = ?,
                            bbox_mapping = ?
                        WHERE pmid = ?
                    """,
                        (
                            result.raw_response,
                            json.dumps(result.parsed_json),
                            bbox_mapping_str,
                            pmid,
                        ),
                    )

                    # Sync gene_mentions for this paper
                    # Extract genes from evidence JSON
                    genes = self._extract_genes_from_evidence(result.parsed_json, panel_data)

                    # Delete existing gene_mentions for this PMID and source only
                    cursor.execute(
                        "DELETE FROM gene_mentions WHERE pmid = ? AND source = ?",
                        (pmid, gene_source),
                    )

                    # Insert new gene_mentions for each gene found with appropriate source
                    # Only include genes with disease_entities (patient data)
                    for resolved_gene in genes:
                        # Skip genes without disease_entities (mechanistic only)
                        has_phenotypes = any(
                            g["gene"] == resolved_gene.paper_symbol and g["disease_entities"]
                            for g in result.parsed_json["gene_evaluations"]
                        )
                        if not has_phenotypes:
                            continue

                        cursor.execute(
                            """
                            INSERT INTO gene_mentions (panelapp_gene_symbol, paper_gene_symbol, pmid, source)
                            VALUES (?, ?, ?, ?)
                            """,
                            (
                                resolved_gene.panelapp_symbol,
                                resolved_gene.paper_symbol,
                                pmid,
                                gene_source,
                            ),
                        )

                    successful_updates += 1

            conn.commit()
            logger.info(
                f"Updated {successful_updates} papers with evidence extraction and synchronized gene_mentions"
            )

    def _extract_genes_from_evidence(
        self, evidence_data: dict[str, Any], panel_data: PanelGeneData
    ) -> set[ResolvedGene]:
        """Extract gene symbols from evidence extraction JSON with alias resolution."""

        # Extract gene symbols from evidence evaluations
        gene_symbols = set()
        if "gene_evaluations" in evidence_data:
            for evaluation in evidence_data["gene_evaluations"]:
                paper_gene_symbol = evaluation.get("gene")
                if paper_gene_symbol:
                    gene_symbols.add(paper_gene_symbol)

        # Use shared resolution logic
        resolved: set[ResolvedGene] = resolve_gene_symbols(gene_symbols, panel_data)
        return resolved

    def get_deep_analysis_statistics(self) -> dict[str, int]:
        """Get statistics about deep analysis processing progress."""
        logger.debug("Querying database for deep analysis statistics...")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Papers eligible for deep analysis (downloaded full text)
            logger.debug("  Counting papers with downloaded full text...")
            cursor.execute(
                """
                SELECT COUNT(*) FROM papers
                WHERE download_status IN ('pmc_downloaded', 'manual_downloaded')
                """
            )
            eligible_papers = cursor.fetchone()[0]

            # Papers already processed for gene rating
            logger.debug("  Counting already processed papers...")
            cursor.execute(
                """
                SELECT COUNT(*) FROM papers
                WHERE evidence_extraction_json IS NOT NULL
            """
            )
            processed_papers = cursor.fetchone()[0]

            # Remaining papers to process
            logger.debug("  Counting remaining papers to process...")
            cursor.execute(
                """
                SELECT COUNT(*) FROM papers
                WHERE download_status IN ('pmc_downloaded', 'manual_downloaded')
                AND evidence_extraction_json IS NULL
            """
            )
            remaining_papers = cursor.fetchone()[0]

            return {
                "eligible_papers": eligible_papers,
                "processed_papers": processed_papers,
                "remaining_papers": remaining_papers,
            }


def prepare_deep_analysis_prompts(
    papers: list[dict[str, Any]],
    template_path: Path,
    panel_formatted: str,
    papers_dir: Path,
    encoding: HarmonyEncoding,
    max_model_len: int,
    max_tokens: int,
) -> DeepAnalysisPreparation:
    """
    Prepare prompts for deep analysis with full text from Docling JSON files.

    Args:
        papers: List of paper dicts with pmid, title, abstract, entrez_date
        template_path: Path to Jinja2 prompt template
        panel_formatted: Formatted panel description (empty string if not scoping)
        papers_dir: Directory containing Docling JSON files
        encoding: Tokenizer encoding for length checking
        max_model_len: Maximum model context length
        max_tokens: Maximum tokens to generate

    Returns:
        DeepAnalysisPreparation with prepared prompts and missing PMIDs
    """
    # Load Jinja2 template
    env = Environment(loader=FileSystemLoader(template_path.parent), autoescape=False)
    template = env.get_template(template_path.name)

    paper_prompts = []
    missing_pmids = []

    for paper in papers:
        pmid = paper["pmid"]
        json_file = papers_dir / f"{pmid}.json"

        if not json_file.exists():
            logger.warning(f"Missing JSON file for PMID {pmid}: {json_file}")
            missing_pmids.append(pmid)
            continue

        try:
            # Use the Docling serializer to get text with bbox IDs
            full_text, bbox_mapping = serialize_with_bbox_ids(json_file)
        except Exception as e:
            logger.error(f"Failed to serialize JSON file for PMID {pmid}: {e}")
            missing_pmids.append(pmid)
            continue

        # Truncate full_text if it exceeds token budget
        overhead_tokens = max_tokens + 2000
        available_tokens = max_model_len - overhead_tokens
        full_text_tokens = encoding.encode(full_text)

        if len(full_text_tokens) > available_tokens:
            logger.warning(
                f"Truncating PMID {pmid}: {len(full_text_tokens)} -> {available_tokens} tokens"
            )
            truncated_tokens = full_text_tokens[:available_tokens]
            full_text = (
                encoding.decode(truncated_tokens)
                + "\n\n[NOTE: Paper truncated to fit context window]"
            )

        # Render the Jinja2 template
        prompt = template.render(
            title=paper["title"],
            date=paper["entrez_date"],
            abstract=paper["abstract"],
            full_text=full_text,
            panel_formatted=panel_formatted,
        )

        paper_prompts.append(PaperPrompt(prompt, pmid, bbox_mapping))

    return DeepAnalysisPreparation(paper_prompts, missing_pmids)


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database",
    ),
    papers_dir: Path = typer.Option(
        Path("data/papers"),
        "--papers-dir",
        help="Directory containing full text files",
    ),
    model: str = typer.Option(
        "openai/gpt-oss-120b",
        "--model",
        "-m",
        help="Model name for vLLM",
    ),
    temperature: float = typer.Option(
        1.0,
        "--temperature",
        "-t",
        help="Sampling temperature",
    ),
    max_tokens: int = typer.Option(
        30000,
        "--max-tokens",
        help="Maximum tokens to generate",
    ),
    tensor_parallel_size: int = typer.Option(
        1,
        "--tensor-parallel-size",
        help="Tensor parallelism size",
    ),
    max_model_len: int = typer.Option(
        131072,
        "--max-model-len",
        help="Maximum model context length",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    ),
    prompt_path: Path = typer.Option(
        Path("prompts/evidence_extraction_prompt.j2"),
        "--prompt-path",
        "-p",
        help="Path to Jinja2 prompt template file",
    ),
    schema_path: Path = typer.Option(
        Path("prompts/evidence_extraction_schema.json"),
        "--schema-path",
        "-s",
        help="Path to response schema file",
    ),
    panel_date: str = typer.Option(
        ...,
        "--panel-date",
        help="Panel state date (YYYY-MM-DD) for gene alias resolution",
    ),
    scope_panel_id: int | None = typer.Option(
        None,
        "--scope-panel-id",
        help="Panel ID for panel-scoped evidence extraction (only extracts genes relevant to this panel)",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of retry attempts for failed papers",
    ),
) -> None:
    """Extract evidence from full text papers using vLLM inference."""
    # Validate inputs
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if not papers_dir.exists():
        logger.error(f"Papers directory not found: {papers_dir}")
        raise typer.Exit(1)

    if not prompt_path.exists():
        logger.error(f"Prompt template not found: {prompt_path}")
        raise typer.Exit(1)

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        raise typer.Exit(1)

    # Load schema (template is loaded via Jinja2 in prepare_deep_analysis_prompts)
    logger.info("Loading schema...")
    schema: dict[str, Any] = json.loads(schema_path.read_text())
    logger.info(f"  Loaded schema from {schema_path}")

    # Initialize components
    logger.info("Initializing database processor...")
    db_processor = PaperBatchProcessor(db_path)
    logger.info(f"  Connected to database at {db_path}")

    # Fetch PanelApp data once for gene symbol resolution
    logger.info(f"Fetching PanelApp gene data for {panel_date}...")
    client = PanelAppClient(panel_date)
    panel_data = client.get_target_panels_genes()
    logger.info(f"  Loaded {len(panel_data.combined_gene_symbols)} genes from PanelApp")

    # Build panel_formatted for template (empty string if not scoping to a panel)
    if scope_panel_id is not None:
        logger.info(f"Fetching description for scope panel {scope_panel_id}...")
        try:
            panel_info = client.get_panel_data(scope_panel_id)
        except ValueError as e:
            logger.error(f"Panel {scope_panel_id} not found in PanelApp data for {panel_date}")
            raise typer.Exit(1) from e

        panel_formatted = format_panel_for_prompt(scope_panel_id, panel_info)
        logger.info(f"  Panel-scoped mode: {panel_info.get('name', 'Unknown')}")
    else:
        panel_formatted = ""

    logger.info("Initializing Harmony batch processor...")
    inference_engine = HarmonyBatchProcessor(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )

    # Get initial statistics
    logger.info("Fetching database statistics...")
    stats = db_processor.get_deep_analysis_statistics()
    logger.info("Deep analysis statistics:")
    logger.info(f"  Papers with downloaded full text: {stats['eligible_papers']:,}")
    logger.info(f"  Already processed: {stats['processed_papers']:,}")
    logger.info(f"  Remaining to process: {stats['remaining_papers']:,}")

    if stats["remaining_papers"] == 0:
        logger.info("No papers remaining to process!")
        return

    initial_remaining = stats["remaining_papers"]
    total_processed = 0
    all_missing_pmids = set()
    consecutive_failures = 0
    retry_attempt = 0

    with tqdm(total=initial_remaining, desc="Processing papers") as pbar:
        while retry_attempt < max_retries:
            # Check remaining work
            stats = db_processor.get_deep_analysis_statistics()
            if stats["remaining_papers"] == 0:
                logger.info("All papers successfully processed!")
                break

            if retry_attempt > 0:
                logger.info(
                    f"Retry attempt {retry_attempt} - {stats['remaining_papers']} papers remaining"
                )

            # Get all papers for deep analysis
            logger.info("Fetching papers for deep analysis...")
            papers = db_processor.get_papers_for_deep_analysis()
            logger.info(f"  Retrieved {len(papers)} papers for processing")

            if not papers:
                logger.info("No more papers to process")
                break

            # Prepare prompts with full text
            logger.info("Preparing prompts with full text...")
            preparation = prepare_deep_analysis_prompts(
                papers,
                prompt_path,
                panel_formatted,
                papers_dir,
                inference_engine.get_encoding(),
                max_model_len,
                max_tokens,
            )
            logger.info(f"  Successfully prepared {len(preparation.paper_prompts)} prompts")
            logger.info(f"  Missing JSON files: {len(preparation.missing_pmids)}")

            all_missing_pmids.update(preparation.missing_pmids)

            if not preparation.paper_prompts:
                logger.error("No papers could be processed - all JSON files are missing")
                break

            # Process papers individually
            pass_processed = 0
            failed_papers = []

            for paper_prompt in preparation.paper_prompts:
                logger.info(f"Processing PMID {paper_prompt.pmid}")

                # Process single paper
                results = inference_engine.process_batch([paper_prompt.prompt], schema)

                # Validate and update database
                if results and results[0] is not None:
                    result = results[0]
                    # Validate box IDs against paper's bbox_mapping
                    valid_box_ids = set(paper_prompt.bbox_mapping.keys())
                    if not validate_box_ids(result.parsed_json, valid_box_ids):
                        logger.warning(f"Invalid box IDs for PMID {paper_prompt.pmid}")
                        failed_papers.append(paper_prompt.pmid)
                    else:
                        db_processor.update_paper_evidence_extraction(
                            [paper_prompt], [result], panel_data
                        )
                        pass_processed += 1
                        pbar.update(1)
                else:
                    logger.warning(f"Failed to process PMID {paper_prompt.pmid}")
                    failed_papers.append(paper_prompt.pmid)

            total_processed += pass_processed

            # Check if we made progress
            if pass_processed == 0:
                consecutive_failures += 1
                logger.warning(f"No progress made in retry attempt {retry_attempt}")
                if consecutive_failures >= 2:
                    logger.error("Multiple consecutive attempts with no progress - stopping")
                    break
            else:
                consecutive_failures = 0

            retry_attempt += 1

    # Final statistics
    final_stats = db_processor.get_deep_analysis_statistics()
    logger.info("Deep analysis complete!")
    logger.info("Final statistics:")
    logger.info(f"  Successfully processed: {total_processed:,}")
    logger.info(f"  Missing JSON files: {len(all_missing_pmids):,}")
    logger.info(f"  Still remaining: {final_stats['remaining_papers']:,}")

    if all_missing_pmids:
        missing_list = list(all_missing_pmids)
        logger.info(
            f"  PMIDs with missing JSON files: {missing_list[:10]}..."
            if len(missing_list) > 10
            else f"  PMIDs with missing JSON files: {missing_list}"
        )

    if final_stats["remaining_papers"] > 0:
        logger.warning(
            f"Failed to process {final_stats['remaining_papers']} papers after {max_retries} attempts"
        )


if __name__ == "__main__":
    app()
