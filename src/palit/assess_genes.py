#!/usr/bin/env python3
"""Aggregate evidence assessment across papers for each gene."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

import typer
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from palit.panelapp_client import PanelAppClient, PanelGeneData, format_panel_for_prompt

app = typer.Typer(help="Aggregate evidence assessment across papers for each gene")
logger = logging.getLogger(__name__)


def find_gene_panel(
    gene_symbol: str,
    target_panel_ids: list[int],
    panel_data: PanelGeneData,
) -> int | None:
    """Find first target panel containing this gene.

    Iterates through target panels in the given order and returns the first
    panel that contains the gene.

    Args:
        gene_symbol: Gene symbol to look up
        target_panel_ids: Ordered list of panel IDs to check
        panel_data: PanelApp gene data with panel mappings

    Returns:
        Panel ID if found, None if gene is novel (not in any target panel).
    """
    gene_panels = panel_data.gene_panel_mapping.get(gene_symbol, set())
    for panel_id in target_panel_ids:
        if panel_id in gene_panels:
            return panel_id
    return None


def validate_box_ids_with_pmid(data: Any, valid_box_ids_by_pmid: dict[int, set[int]]) -> bool:
    """
    Recursively check all (pmid, box_id) pairs in citation structures are valid.
    Returns False immediately on first invalid pair (fail-fast).

    Args:
        data: JSON data structure (dict, list, or primitive)
        valid_box_ids_by_pmid: Map from PMID to set of valid box IDs for that paper

    Returns:
        True if all pairs are valid, False otherwise
    """

    def recurse(obj: Any) -> bool:
        if isinstance(obj, dict):
            # Check if this dict has both 'pmid' and 'box_id' fields (citation structure)
            if "pmid" in obj and "box_id" in obj:
                pmid = obj["pmid"]
                box_id = obj["box_id"]
                if isinstance(pmid, int) and isinstance(box_id, int):
                    valid_box_ids = valid_box_ids_by_pmid.get(pmid)
                    if valid_box_ids is None or box_id not in valid_box_ids:
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


def fetch_valid_box_ids_by_pmid(
    db_path: Path, evidence_list: list[dict[str, Any]]
) -> dict[int, set[int]]:
    """
    Query database to get valid box IDs for each paper in evidence_list.

    Args:
        db_path: Path to SQLite database
        evidence_list: List of evidence dicts containing PMIDs

    Returns:
        Map from PMID to set of valid box IDs for that paper
    """
    # Extract PMIDs from evidence list
    pmids = {evidence["pmid"] for evidence in evidence_list}

    if not pmids:
        return {}

    valid_box_ids_by_pmid: dict[int, set[int]] = {}

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Fetch bbox_mapping for each PMID
        for pmid in pmids:
            cursor.execute("SELECT bbox_mapping FROM papers WHERE pmid = ?", (pmid,))
            row = cursor.fetchone()

            if row and row[0]:
                try:
                    bbox_mapping = json.loads(row[0])
                    # Store the set of valid box IDs for this PMID
                    valid_box_ids_by_pmid[pmid] = {int(box_id) for box_id in bbox_mapping.keys()}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Error parsing bbox_mapping for PMID {pmid}: {e}")
                    continue

    return valid_box_ids_by_pmid


class PaperBatchProcessor:
    """Handle database operations for aggregate gene assessment."""

    def __init__(self, db_path: Path):
        """Initialize with database path."""
        self.db_path = db_path

    def get_evidence_for_gene(self, panelapp_gene_symbol: str) -> list[dict[str, Any]]:
        """Get all evidence extractions for a specific gene.

        Args:
            panelapp_gene_symbol: PanelApp gene symbol to search for

        Returns:
            List of dicts with pmid, evidence_extraction_json, and metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get evidence from papers that mention this gene
            cursor.execute(
                """
                SELECT DISTINCT p.pmid, p.entrez_date, p.title, p.evidence_extraction_json, gm.paper_gene_symbol
                FROM papers p
                JOIN gene_mentions gm ON p.pmid = gm.pmid
                WHERE gm.panelapp_gene_symbol = ?
                AND p.evidence_extraction_json IS NOT NULL
                ORDER BY p.pmid
            """,
                (panelapp_gene_symbol,),
            )

            evidence_list = []
            for row in cursor.fetchall():
                try:
                    evidence_json = json.loads(row["evidence_extraction_json"])
                    # Filter to only include evaluations for this specific gene
                    paper_gene_symbol = row["paper_gene_symbol"]
                    filtered_evaluations = []
                    for gene_eval in evidence_json.get("gene_evaluations", []):
                        if gene_eval.get("gene", "").upper() == paper_gene_symbol.upper():
                            del gene_eval["variants"]  # Drop detailed variants from prompt.
                            filtered_evaluations.append(gene_eval)

                    if filtered_evaluations:
                        evidence_list.append(
                            {
                                "pmid": row["pmid"],
                                "date": row["entrez_date"],
                                "title": row["title"],
                                "paper_gene_symbol": paper_gene_symbol,  # Include the actual symbol used in the paper
                                "gene_evaluations": filtered_evaluations,
                            }
                        )

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error parsing evidence for PMID {row['pmid']}: {e}")
                    continue

            logger.debug(
                f"Found {len(evidence_list)} papers with evidence for gene {panelapp_gene_symbol}"
            )
            return evidence_list

    def update_gene_assessment(
        self,
        panelapp_gene_symbol: str,
        assessment_data: tuple[str, dict[str, Any] | None],
    ) -> None:
        """Store aggregate assessment result in gene_assessments table.

        Args:
            panelapp_gene_symbol: PanelApp gene symbol
            assessment_data: Tuple of (raw_response, parsed_json)
        """
        raw_response, json_data = assessment_data

        if not json_data:
            logger.warning(f"No valid JSON data for {panelapp_gene_symbol}")
            return

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO gene_assessments
                    (panelapp_gene_symbol, assessment_raw, assessment_json)
                    VALUES (?, ?, ?)
                """,
                    (panelapp_gene_symbol, raw_response, json.dumps(json_data)),
                )

                conn.commit()
                logger.info(f"Stored aggregate assessment for {panelapp_gene_symbol}")

            except sqlite3.Error as e:
                logger.error(f"Error storing aggregate assessment for {panelapp_gene_symbol}: {e}")

    def get_aggregate_assessment_statistics(self) -> dict[str, int]:
        """Get statistics about aggregate assessment progress."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total unique genes in the working set (recent_evidence)
            cursor.execute(
                """
                SELECT COUNT(DISTINCT panelapp_gene_symbol)
                FROM gene_mentions
                WHERE source = 'recent_evidence'
            """
            )
            genes_with_evidence = cursor.fetchone()[0]

            # Already assessed genes
            cursor.execute("SELECT COUNT(*) FROM gene_assessments")
            assessed_genes = cursor.fetchone()[0]

            return {
                "genes_with_evidence": genes_with_evidence,
                "assessed_genes": assessed_genes,
                "remaining_to_assess": genes_with_evidence - assessed_genes,
            }


def format_previous_reviews_data(existing_reviews: list[dict[str, Any]]) -> str:
    """Format existing PanelApp reviews as XML-tagged data for the prompt.

    Args:
        existing_reviews: List of evaluation dicts from PanelApp API, ordered most recent first

    Returns:
        Empty string if no reviews, otherwise XML-tagged formatted review data.
    """
    if not existing_reviews:
        return ""

    lines = ["<previous_reviews>"]

    for i, review in enumerate(existing_reviews, 1):
        lines.append(f"Review {i}:")

        rating = review.get("rating")
        if rating:
            lines.append(f"  Rating: {rating}")

        moi = review.get("moi")
        if moi:
            lines.append(f"  Mode of Inheritance: {moi}")

        phenotypes = review.get("phenotypes")
        if phenotypes:
            lines.append(f"  Phenotypes: {', '.join(phenotypes)}")

        publications = review.get("publications")
        if publications:
            lines.append(f"  Publications: {'; '.join(publications)}")

        comments = review.get("comments", [])
        if comments:
            lines.append("  Comments:")
            for comment in comments:
                user = comment.get("user_name", "Unknown")
                date = comment.get("created", "")
                text = comment.get("comment", "")
                lines.append(f"    [{date}] {user}: {text}")

        lines.append("")

    lines.append("</previous_reviews>")
    return "\n".join(lines)


def prepare_aggregate_assessment_prompt(
    panelapp_gene_symbol: str,
    evidence_list: list[dict[str, Any]],
    template_path: Path,
    existing_reviews: list[dict[str, Any]],
    panel_formatted: str,
) -> str:
    """
    Prepare aggregate assessment prompt using Jinja2 template.

    Args:
        panelapp_gene_symbol: PanelApp gene symbol being assessed
        evidence_list: List of evidence extractions from multiple papers
        template_path: Path to Jinja2 template file
        existing_reviews: List of existing PanelApp reviews (empty for novel genes)
        panel_formatted: Formatted panel description for panel-scoped mode (empty string if not scoping)

    Returns:
        Rendered prompt string
    """
    # Extract unique paper gene symbols (aliases) from evidence
    paper_symbols = set()
    for evidence in evidence_list:
        if "paper_gene_symbol" in evidence:
            paper_symbols.add(evidence["paper_gene_symbol"])

    # Format gene symbol with aliases if they differ from PanelApp symbol
    if paper_symbols and paper_symbols != {panelapp_gene_symbol}:
        gene_symbol_with_aliases = f"{panelapp_gene_symbol} (also referred to as: {', '.join(sorted(paper_symbols))} in the papers)"
    else:
        gene_symbol_with_aliases = panelapp_gene_symbol

    # Create structured JSON with all evidence
    evidence_extractions = json.dumps(evidence_list, indent=2)

    # Format previous reviews data (empty string for novel genes)
    previous_reviews_section = format_previous_reviews_data(existing_reviews)

    # Load and render Jinja2 template
    env = Environment(loader=FileSystemLoader(template_path.parent), autoescape=False)
    template = env.get_template(template_path.name)

    return template.render(
        gene_symbol=gene_symbol_with_aliases,
        evidence_extractions=evidence_extractions,
        has_previous_reviews=bool(existing_reviews),
        previous_reviews_section=previous_reviews_section,
        panel_formatted=panel_formatted,
    )


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database",
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
        65536,
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
        Path("prompts/aggregate_assessment_prompt.j2"),
        "--prompt-path",
        "-p",
        help="Path to Jinja2 prompt template file",
    ),
    schema_path: Path = typer.Option(
        Path("prompts/aggregate_assessment_schema.json"),
        "--schema-path",
        "-s",
        help="Path to response schema file",
    ),
    max_retries: int = typer.Option(
        5,
        "--max-retries",
        help="Maximum number of retry attempts for failed genes",
    ),
    panel_date: str = typer.Option(
        ...,
        "--panel-date",
        help="Panel state date (YYYY-MM-DD) for checking gene panel membership",
    ),
    target_panel_ids: list[int] | None = typer.Option(
        None,
        "--target-panel-ids",
        help="Panel IDs to check for existing genes. Can be specified multiple times. Defaults to TARGET_PANEL_IDS.",
    ),
    scope_panel_id: int | None = typer.Option(
        None,
        "--scope-panel-id",
        help="Panel ID for panel-scoped assessment. When set, the summary must explain why the gene is relevant to this panel's scope.",
    ),
) -> None:
    """Perform aggregate assessment of genes using evidence from multiple papers."""
    # Validate inputs
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if not prompt_path.exists():
        logger.error(f"Prompt template not found: {prompt_path}")
        raise typer.Exit(1)

    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        raise typer.Exit(1)

    # Load schema (template is loaded via Jinja2 in prepare_aggregate_assessment_prompt)
    logger.info("Loading schema...")
    schema: dict[str, Any] = json.loads(schema_path.read_text())
    logger.info(f"  Loaded schema from {schema_path}")

    # Initialize PanelApp client and fetch panel data
    logger.info(f"Fetching PanelApp gene data for {panel_date}...")
    panelapp_client = PanelAppClient(panel_date)
    panel_data = panelapp_client.get_target_panels_genes(target_panel_ids)
    logger.info(
        f"  Loaded {len(panel_data.combined_gene_symbols)} genes from {len(panel_data.panel_ids)} target panels"
    )

    # Build panel_formatted for template (empty string if not scoping to a panel)
    if scope_panel_id is not None:
        logger.info(f"Fetching description for scope panel {scope_panel_id}...")
        try:
            panel_info = panelapp_client.get_panel_data(scope_panel_id)
        except ValueError as e:
            logger.error(f"Panel {scope_panel_id} not found in PanelApp data for {panel_date}")
            raise typer.Exit(1) from e

        panel_formatted = format_panel_for_prompt(scope_panel_id, panel_info)
        logger.info(f"  Panel-scoped mode: {panel_info.get('name', 'Unknown')}")
    else:
        panel_formatted = ""

    # Initialize components
    logger.info("Initializing database processor...")
    db_processor = PaperBatchProcessor(db_path)
    logger.info(f"  Connected to database at {db_path}")

    logger.info("Initializing Harmony batch processor...")
    from palit.llm import HarmonyBatchProcessor

    inference_engine = HarmonyBatchProcessor(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )

    # Get initial statistics
    logger.info("Fetching aggregate assessment statistics...")
    stats = db_processor.get_aggregate_assessment_statistics()
    logger.info("Aggregate assessment statistics:")
    logger.info(f"  Genes with evidence: {stats['genes_with_evidence']:,}")
    logger.info(f"  Already assessed: {stats['assessed_genes']:,}")
    logger.info(f"  Remaining to assess: {stats['remaining_to_assess']:,}")

    if stats["remaining_to_assess"] == 0:
        logger.info("No genes remaining to assess!")
        return

    initial_remaining = stats["remaining_to_assess"]
    total_processed = 0
    genes_without_evidence = set()
    consecutive_failures = 0
    retry_attempt = 0

    with tqdm(total=initial_remaining, desc="Processing genes") as pbar:
        while retry_attempt < max_retries:
            # Re-fetch statistics to check remaining work
            stats = db_processor.get_aggregate_assessment_statistics()
            if stats["remaining_to_assess"] == 0:
                logger.info("All genes successfully assessed!")
                break

            if retry_attempt > 0:
                logger.info(
                    f"Retry attempt {retry_attempt} - {stats['remaining_to_assess']} genes remaining"
                )

            # Get all unique genes from the working set (recent_evidence) that haven't been assessed
            logger.info("Fetching genes with evidence...")
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT DISTINCT panelapp_gene_symbol
                    FROM gene_mentions
                    WHERE source = 'recent_evidence'
                    ORDER BY panelapp_gene_symbol
                """
                )
                genes = [row[0] for row in cursor.fetchall()]

                # Check which genes already assessed
                cursor.execute(
                    """
                    SELECT panelapp_gene_symbol
                    FROM gene_assessments
                """
                )
                already_assessed = {row[0] for row in cursor.fetchall()}

            genes_to_assess = [g for g in genes if g not in already_assessed]
            logger.info(f"Found {len(genes_to_assess)} genes to assess")

            if not genes_to_assess:
                logger.info("No genes need assessment!")
                break

            # Process each gene
            pass_processed = 0
            failed_genes = []

            for gene_symbol in genes_to_assess:
                logger.info(f"Processing {gene_symbol}")

                # Get evidence for this gene
                evidence_list = db_processor.get_evidence_for_gene(gene_symbol)

                if not evidence_list:
                    logger.warning(f"No evidence found for {gene_symbol}")
                    genes_without_evidence.add(gene_symbol)
                    continue

                # Check if gene exists in any target panel
                existing_panel_id = find_gene_panel(gene_symbol, panel_data.panel_ids, panel_data)

                # Fetch existing reviews for non-novel genes
                existing_reviews: list[dict[str, Any]] = []
                if existing_panel_id is not None:
                    existing_reviews = panelapp_client.get_gene_evaluations(
                        existing_panel_id, gene_symbol
                    )
                    logger.info(
                        f"  Found {len(existing_reviews)} existing reviews in panel {existing_panel_id}"
                    )

                # Prepare aggregate assessment prompt
                prompt = prepare_aggregate_assessment_prompt(
                    gene_symbol, evidence_list, prompt_path, existing_reviews, panel_formatted
                )

                # Process aggregate assessment
                results = inference_engine.process_batch([prompt], schema)

                # Validate and update database
                if results and results[0] is not None:
                    result = results[0]
                    # Validate (pmid, box_id) pairs against database
                    valid_box_ids_by_pmid = fetch_valid_box_ids_by_pmid(db_path, evidence_list)
                    if not validate_box_ids_with_pmid(result.parsed_json, valid_box_ids_by_pmid):
                        logger.warning(f"Invalid (pmid, box_id) pairs for {gene_symbol}")
                        failed_genes.append(gene_symbol)
                    else:
                        db_processor.update_gene_assessment(
                            gene_symbol, (result.raw_response, result.parsed_json)
                        )
                        pass_processed += 1
                        pbar.update(1)
                else:
                    logger.warning(f"Failed to process aggregate assessment for {gene_symbol}")
                    failed_genes.append(gene_symbol)

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
    final_stats = db_processor.get_aggregate_assessment_statistics()
    logger.info("Aggregate assessment complete!")
    logger.info("Final statistics:")
    logger.info(f"  Successfully processed: {total_processed:,}")
    logger.info(f"  Genes without evidence: {len(genes_without_evidence):,}")
    logger.info(f"  Still remaining: {final_stats['remaining_to_assess']:,}")

    if genes_without_evidence:
        no_evidence_list = list(genes_without_evidence)
        logger.info(
            f"  Genes without evidence: {no_evidence_list[:10]}..."
            if len(no_evidence_list) > 10
            else f"  Genes without evidence: {no_evidence_list}"
        )

    if final_stats["remaining_to_assess"] > 0:
        logger.warning(
            f"Failed to assess {final_stats['remaining_to_assess']} genes after {max_retries} attempts"
        )


if __name__ == "__main__":
    app()
