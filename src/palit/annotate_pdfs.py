#!/usr/bin/env python3
"""Create annotated PDFs from gene assessment results with citation highlighting."""

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Highlight
from pypdf.generic import (
    ArrayObject,
    FloatObject,
    NameObject,
    TextStringObject,
)

from palit.docling import parse_bbox_mapping_from_json
from palit.panelapp_integration import PANELAPP_CRITERIA

app = typer.Typer(help="Create annotated PDFs with citation highlighting")
logger = logging.getLogger(__name__)


@dataclass
class AnnotationCitation:
    """Represents a citation to be highlighted in a PDF."""

    gene: str
    title: str
    content: str
    box_id: int
    page: int


def extract_citations_from_individual_assessment(
    evidence_json: dict[str, Any], bbox_mapping: dict[int, dict[str, Any]], current_pmid: int
) -> list[AnnotationCitation]:
    """
    Extract citations from an individual paper's evidence extraction.

    Args:
        evidence_json: The evidence extraction JSON for one paper
        bbox_mapping: Bbox mapping for the current paper
        current_pmid: PMID of the current paper being processed

    Returns:
        List of AnnotationCitation objects
    """
    citations = []

    for gene_eval in evidence_json.get("gene_evaluations", []):
        gene_symbol = gene_eval.get("gene", "Unknown")

        for criterion_name in PANELAPP_CRITERIA:
            if criterion_name not in gene_eval:
                continue

            criterion = gene_eval[criterion_name]

            for citation in criterion.get("citations", []):
                box_id = citation["box_id"]

                # Check if box_id exists in current paper's bbox_mapping
                if box_id not in bbox_mapping:
                    logger.warning(
                        f"Box ID {box_id} not found in bbox mapping for PMID {current_pmid}"
                    )
                    continue

                bbox_info = bbox_mapping[box_id]
                result_status = "PASS" if criterion.get("result", False) else "FAIL"

                citations.append(
                    AnnotationCitation(
                        gene=gene_symbol,
                        title=f"{gene_symbol} - {criterion_name} ({result_status}) - Individual",
                        content=citation["commentary"],
                        box_id=box_id,
                        page=bbox_info["page"],
                    )
                )

    # Extract phenotype citations
    for gene_eval in evidence_json.get("gene_evaluations", []):
        gene_symbol = gene_eval.get("gene", "Unknown")

        for phenotype_group in gene_eval.get("phenotype_groups", []):
            phenotype = phenotype_group.get("phenotype", "Unknown phenotype")

            for citation in phenotype_group.get("citations", []):
                box_id = citation["box_id"]

                # Check if box_id exists in current paper's bbox_mapping
                if box_id not in bbox_mapping:
                    logger.warning(
                        f"Phenotype box ID {box_id} not found in bbox mapping for PMID {current_pmid}"
                    )
                    continue

                bbox_info = bbox_mapping[box_id]

                citations.append(
                    AnnotationCitation(
                        gene=gene_symbol,
                        title=f"{gene_symbol} - Phenotype: {phenotype} - Individual",
                        content=citation["commentary"],
                        box_id=box_id,
                        page=bbox_info["page"],
                    )
                )

    # Extract quality concern citations
    for gene_eval in evidence_json.get("gene_evaluations", []):
        gene_symbol = gene_eval.get("gene", "Unknown")

        for concern in gene_eval.get("quality_concerns", []):
            concern_text = concern.get("concern", "Quality concern")

            for citation in concern.get("citations", []):
                box_id = citation["box_id"]

                # Check if box_id exists in current paper's bbox_mapping
                if box_id not in bbox_mapping:
                    logger.warning(
                        f"Quality concern box ID {box_id} not found in bbox mapping for PMID {current_pmid}"
                    )
                    continue

                bbox_info = bbox_mapping[box_id]

                citations.append(
                    AnnotationCitation(
                        gene=gene_symbol,
                        title=f"{gene_symbol} - Quality Concern - Individual",
                        content=f"{concern_text}: {citation['commentary']}",
                        box_id=box_id,
                        page=bbox_info["page"],
                    )
                )

    return citations


def extract_citations_from_aggregate_assessment(
    assessment_json: dict[str, Any], bbox_mapping: dict[int, dict[str, Any]], current_pmid: int
) -> list[AnnotationCitation]:
    """
    Extract citations from an aggregate assessment for a specific gene and paper.

    Args:
        assessment_json: The aggregate assessment JSON for one gene
        bbox_mapping: Bbox mapping for the current paper only
        current_pmid: PMID of the current paper being processed

    Returns:
        List of AnnotationCitation objects for citations that belong to the current paper only
    """
    citations = []
    gene_symbol = assessment_json["gene"]

    for criterion_name in PANELAPP_CRITERIA:
        criterion = assessment_json[criterion_name]

        for citation in criterion["citations"]:
            # Only process citations from the current paper
            if citation["pmid"] != current_pmid:
                continue

            box_id = citation["box_id"]
            bbox_info = bbox_mapping[box_id]
            result_status = "PASS" if criterion["result"] else "FAIL"

            citations.append(
                AnnotationCitation(
                    gene=gene_symbol,
                    title=f"{gene_symbol} - {criterion_name} ({result_status}) - Aggregate",
                    content=citation["commentary"],
                    box_id=box_id,
                    page=bbox_info["page"],
                )
            )

    # Extract phenotype citations
    for phenotype_group in assessment_json.get("phenotype_groups", []):
        phenotype = phenotype_group.get("phenotype", "Unknown phenotype")

        for citation in phenotype_group.get("citations", []):
            # Only process citations from the current paper
            if citation["pmid"] != current_pmid:
                continue

            box_id = citation["box_id"]
            bbox_info = bbox_mapping[box_id]

            citations.append(
                AnnotationCitation(
                    gene=gene_symbol,
                    title=f"{gene_symbol} - Phenotype: {phenotype} - Aggregate",
                    content=citation["commentary"],
                    box_id=box_id,
                    page=bbox_info["page"],
                )
            )

    # Extract quality concern citations
    for concern in assessment_json.get("quality_concerns", []):
        concern_text = concern.get("concern", "Quality concern")

        for citation in concern.get("citations", []):
            # Only process citations from the current paper
            if citation["pmid"] != current_pmid:
                continue

            box_id = citation["box_id"]
            bbox_info = bbox_mapping[box_id]

            citations.append(
                AnnotationCitation(
                    gene=gene_symbol,
                    title=f"{gene_symbol} - Quality Concern - Aggregate",
                    content=f"{concern_text}: {citation['commentary']}",
                    box_id=box_id,
                    page=bbox_info["page"],
                )
            )

    return citations


def extract_variant_citations(
    db_cursor: Any, bbox_mapping: dict[int, dict[str, Any]], current_pmid: int
) -> list[AnnotationCitation]:
    """
    Extract variant citations from the variant_frequencies table for the current paper.

    Args:
        db_cursor: Database cursor for querying variant frequencies
        bbox_mapping: Bbox mapping for the current paper
        current_pmid: PMID of the current paper being processed

    Returns:
        List of AnnotationCitation objects
    """
    citations = []

    # Query variant frequencies for this paper
    db_cursor.execute(
        """
        SELECT
            vf.variant_id,
            vf.box_id,
            vf.panelapp_gene_symbol,
            vf.normalization,
            vf.gnomad
        FROM variant_frequencies vf
        WHERE vf.pmid = ?
        ORDER BY vf.panelapp_gene_symbol, vf.variant_id
    """,
        (current_pmid,),
    )

    for row in db_cursor.fetchall():
        box_id = row["box_id"]
        gene_symbol = row["panelapp_gene_symbol"]
        variant_id = row["variant_id"]

        # Check if box_id exists in current paper's bbox_mapping
        if box_id not in bbox_mapping:
            logger.warning(
                f"Variant box ID {box_id} not found in bbox mapping for PMID {current_pmid}"
            )
            continue

        bbox_info = bbox_mapping[box_id]

        # Parse normalization and gnomAD data for commentary
        try:
            normalization = json.loads(row["normalization"])
            gnomad = json.loads(row["gnomad"])

            # Create informative commentary
            variant_display = normalization.get("hgvs_c") or normalization.get(
                "original_text", variant_id
            )

            if "error" in gnomad:
                frequency_info = f"gnomAD: {gnomad['error']}"
            elif "variant_not_found" in gnomad:
                frequency_info = "gnomAD: not found"
            else:
                variant_data = gnomad.get("variant")
                if variant_data and variant_data.get("joint", {}).get("ac") is not None:
                    ac = variant_data["joint"]["ac"]
                    frequency_info = f"gnomAD AC: {ac}"
                    if ac > 30:
                        frequency_info += " ⚠️ HIGH FREQUENCY"
                else:
                    frequency_info = "gnomAD: No data"

            commentary = f"Variant: {variant_display} | {frequency_info}"

        except Exception as e:
            logger.warning(
                f"Failed to parse variant data for PMID {current_pmid}, variant {variant_id}: {e}"
            )
            commentary = f"Variant: {variant_id} | Data parsing error"

        citations.append(
            AnnotationCitation(
                gene=gene_symbol,
                title=f"{gene_symbol} - Variant Evidence",
                content=commentary,
                box_id=box_id,
                page=bbox_info["page"],
            )
        )

    return citations


def create_pdf_annotations(
    pdf_path: Path,
    citations: list[AnnotationCitation],
    bbox_mapping: dict[int, dict[str, Any]],
    output_path: Path,
) -> bool:
    """
    Create annotated PDF with square annotations and named destinations for citations.

    Args:
        pdf_path: Path to original PDF
        citations: List of citation dicts with box_id and commentary
        bbox_mapping: Mapping from box_id to bbox info
        output_path: Path for annotated PDF output

    Returns:
        True if successful, False otherwise
    """
    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        # Process each page
        for page_num, page in enumerate(reader.pages, 1):
            writer.add_page(page)

            # Get page dimensions
            float(page.mediabox.height)

            # Find citations for this page
            page_citations = []
            for citation in citations:
                if citation.box_id in bbox_mapping:
                    bbox_info = bbox_mapping[citation.box_id]
                    if bbox_info.get("page") == page_num:
                        page_citations.append((citation, bbox_info))

            # Add annotations and named destinations for this page
            for citation, bbox_info in page_citations:
                bbox = bbox_info["bbox"]
                # Use bbox coordinates directly (both Docling and pypdf use BOTTOMLEFT)
                x1, y1, x2, y2 = bbox["l"], bbox["b"], bbox["r"], bbox["t"]

                # Create highlight annotation with quad_points for the rectangle
                quad_points = ArrayObject(
                    [
                        FloatObject(x1),
                        FloatObject(y2),  # Top-left
                        FloatObject(x2),
                        FloatObject(y2),  # Top-right
                        FloatObject(x1),
                        FloatObject(y1),  # Bottom-left
                        FloatObject(x2),
                        FloatObject(y1),  # Bottom-right
                    ]
                )

                highlight_annotation = Highlight(
                    rect=(x1, y1, x2, y2),
                    quad_points=quad_points,
                    highlight_color="ffeb3b",  # Light yellow highlighter color
                )

                # Add content and title using proper pypdf objects
                highlight_annotation.update(
                    {
                        NameObject("/Contents"): TextStringObject(citation.content),
                        NameObject("/T"): TextStringObject(citation.title),
                        NameObject("/NM"): TextStringObject(
                            f"citation_{citation.box_id}"
                        ),  # Name for linking
                    }
                )

                # Add annotation to the page
                writer.add_annotation(page_number=page_num - 1, annotation=highlight_annotation)

        # Write annotated PDF
        with open(output_path, "wb") as output_file:
            writer.write(output_file)

        logger.info(f"Created annotated PDF: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to create annotated PDF for {pdf_path}: {e}")
        return False


@app.callback(invoke_without_command=True)
def main(
    db_path: Path = typer.Option(
        default=Path("data/db.sqlite"),
        help="Path to SQLite database",
    ),
    papers_dir: Path = typer.Option(
        Path("data/papers"),
        "--papers-dir",
        "-p",
        help="Directory containing original PDFs",
    ),
    output_dir: Path = typer.Option(
        Path("data/papers/annotated"),
        "--output-dir",
        "-o",
        help="Base directory for hierarchical annotated PDFs",
    ),
) -> None:
    """Create gene-centric annotated PDFs with simple directory structure.

    Creates annotated PDFs in structure: {output_dir}/{gene}/{pmid}.pdf
    Each PDF contains highlights relevant to the specific gene.
    """

    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        raise typer.Exit(1)

    if not papers_dir.exists():
        logger.error(f"Papers directory not found: {papers_dir}")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading gene-panel assessments and paper data...")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all gene assessments with their paper citations (all sources)
        cursor.execute("""
            SELECT DISTINCT
                ga.panelapp_gene_symbol,
                ga.assessment_json,
                p.pmid,
                p.bbox_mapping
            FROM gene_assessments ga
            JOIN gene_mentions gm ON ga.panelapp_gene_symbol = gm.panelapp_gene_symbol
            JOIN papers p ON gm.pmid = p.pmid
            WHERE p.bbox_mapping IS NOT NULL
            ORDER BY ga.panelapp_gene_symbol, p.pmid
        """)

        # Process gene assessments
        gene_data = {}
        for row in cursor.fetchall():
            gene_symbol = row["panelapp_gene_symbol"]
            pmid = row["pmid"]
            assessment_json = json.loads(row["assessment_json"])
            bbox_mapping = parse_bbox_mapping_from_json(row["bbox_mapping"])

            key = (gene_symbol, pmid)
            if key not in gene_data:
                gene_data[key] = {
                    "gene_symbol": gene_symbol,
                    "pmid": pmid,
                    "assessment_json": assessment_json,
                    "bbox_mapping": bbox_mapping,
                    "type": "aggregate",
                }

        # Get all papers with individual evidence extractions (separate query, not gene-scoped)
        cursor.execute("""
            SELECT
                p.pmid,
                p.bbox_mapping,
                p.evidence_extraction_json
            FROM papers p
            WHERE p.bbox_mapping IS NOT NULL
            AND p.evidence_extraction_json IS NOT NULL
            ORDER BY p.pmid
        """)

        # Process individual assessments
        individual_data = {}
        for row in cursor.fetchall():
            pmid = row["pmid"]
            evidence_json = json.loads(row["evidence_extraction_json"])
            bbox_mapping = parse_bbox_mapping_from_json(row["bbox_mapping"])

            individual_data[pmid] = {
                "pmid": pmid,
                "evidence_json": evidence_json,
                "bbox_mapping": bbox_mapping,
                "type": "individual",
            }

        # Combine both types
        all_annotation_tasks = list(gene_data.values()) + list(individual_data.values())

    logger.info(
        f"Found {len(gene_data)} gene assessments and {len(individual_data)} individual assessments to process"
    )

    successful_annotations = 0
    failed_annotations = 0
    skipped_annotations = 0

    for task in all_annotation_tasks:
        pmid = task["pmid"]
        task_type = task["type"]

        if task_type == "aggregate":
            gene_symbol = task["gene_symbol"]
            logger.debug(f"Processing gene assessment: {gene_symbol} / PMID {pmid}")

            # Create output directory structure for gene
            gene_dir = output_dir / gene_symbol
            gene_dir.mkdir(parents=True, exist_ok=True)
            output_path = gene_dir / f"{pmid}.pdf"

            # Extract aggregate citations
            citations = extract_citations_from_aggregate_assessment(
                task["assessment_json"], task["bbox_mapping"], pmid
            )

            # Add variant citations from variant_frequencies table
            variant_citations = extract_variant_citations(cursor, task["bbox_mapping"], pmid)
            citations.extend(variant_citations)

        else:  # individual
            logger.debug(f"Processing individual: PMID {pmid}")

            # Create output directory structure for individual
            individual_dir = output_dir / "individual"
            individual_dir.mkdir(parents=True, exist_ok=True)
            output_path = individual_dir / f"{pmid}.pdf"

            # Extract individual citations
            citations = extract_citations_from_individual_assessment(
                task["evidence_json"], task["bbox_mapping"], pmid
            )

            # Add variant citations from variant_frequencies table
            variant_citations = extract_variant_citations(cursor, task["bbox_mapping"], pmid)
            citations.extend(variant_citations)

        # Check if original PDF exists
        pdf_path = papers_dir / f"{pmid}.pdf"
        if not pdf_path.exists():
            logger.warning(f"PDF not found for PMID {pmid}: {pdf_path}")
            failed_annotations += 1
            continue

        # Skip if already exists
        if output_path.exists():
            logger.debug(f"Already exists: {output_path}")
            skipped_annotations += 1
            continue

        if not citations:
            logger.debug(f"No citations for {task_type} assessment PMID {pmid}")
            skipped_annotations += 1
            continue

        logger.info(f"Creating {len(citations)} {task_type} annotations for PMID {pmid}")

        # Create annotated PDF
        success = create_pdf_annotations(pdf_path, citations, task["bbox_mapping"], output_path)

        if success:
            successful_annotations += 1
        else:
            failed_annotations += 1

    logger.info("Gene annotation complete!")
    logger.info(f"Successfully annotated: {successful_annotations} PDFs")
    logger.info(f"Skipped (existing/no citations): {skipped_annotations} PDFs")
    logger.info(f"Failed annotations: {failed_annotations} PDFs")
    logger.info(f"Annotated PDFs saved to: {output_dir}")


if __name__ == "__main__":
    app()
