#!/usr/bin/env python3
"""Docling PDF conversion and text serialization."""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, cast

import typer
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownPictureSerializer,
    MarkdownTableSerializer,
    MarkdownTextSerializer,
)
from docling_core.types.doc import DoclingDocument, PictureItem, TableItem, TextItem
from pydantic import Field
from rich.console import Console
from rich.progress import Progress

logger = logging.getLogger(__name__)

console = Console()
app = typer.Typer(help="Docling PDF conversion and text serialization")


def parse_bbox_mapping_from_json(bbox_mapping_json: str | None) -> dict[int, dict[str, Any]]:
    """
    Parse bbox_mapping from JSON string, converting string keys back to integers.

    Args:
        bbox_mapping_json: JSON string containing bbox mapping data

    Returns:
        Dict with integer keys mapping to bbox info dicts
    """
    if not bbox_mapping_json:
        return {}

    bbox_mapping_raw = json.loads(bbox_mapping_json)
    # Convert bbox_mapping string keys back to integers (JSON converts int keys to strings)
    return {int(k): v for k, v in bbox_mapping_raw.items()}


class BboxMarkdownTextSerializer(MarkdownTextSerializer):
    """Text serializer that wraps output with bbox IDs."""

    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        result = super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
        wrapped = cast(BboxMarkdownDocSerializer, doc_serializer).wrap_with_bbox(item, result.text)
        return create_ser_result(text=wrapped, span_source=[result])


class BboxMarkdownTableSerializer(MarkdownTableSerializer):
    """Table serializer that wraps output with bbox IDs."""

    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        result = super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
        wrapped = cast(BboxMarkdownDocSerializer, doc_serializer).wrap_with_bbox(item, result.text)
        return create_ser_result(text=wrapped, span_source=[result])


class BboxMarkdownPictureSerializer(MarkdownPictureSerializer):
    """Picture serializer that wraps output with bbox IDs."""

    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        result = super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
        wrapped = cast(BboxMarkdownDocSerializer, doc_serializer).wrap_with_bbox(item, result.text)
        return create_ser_result(text=wrapped, span_source=[result])


class BboxMarkdownDocSerializer(MarkdownDocSerializer):
    """Document serializer that tracks bbox mappings and wraps items with IDs."""

    text_serializer: BboxMarkdownTextSerializer = BboxMarkdownTextSerializer()
    table_serializer: BboxMarkdownTableSerializer = BboxMarkdownTableSerializer()
    picture_serializer: BboxMarkdownPictureSerializer = BboxMarkdownPictureSerializer()

    box_id_counter: int = 1
    bbox_mapping: dict[int, dict[str, Any]] = Field(default_factory=dict)

    def wrap_with_bbox(self, item: Any, text: str) -> str:
        """
        Wrap text with bbox ID if item has provenance information and meaningful content.

        Args:
            item: Document item (TextItem, TableItem, PictureItem, etc.)
            text: Serialized text to wrap

        Returns:
            Text wrapped with <b id=N>...</b> tags if bbox available and content is meaningful,
            empty string if content is placeholder-only, otherwise unchanged
        """
        if not text:
            return text

        # Skip placeholder-only content (e.g., "<!-- image -->", "logo\n\n<!-- image -->")
        # Strip HTML comments to see if there's actual content
        text_without_comments = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
        if not text_without_comments.strip():
            # No actual content besides HTML comments, return empty string to skip it
            return ""

        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0]
            if hasattr(prov, "bbox") and hasattr(prov, "page_no"):
                box_id = self.box_id_counter
                self.box_id_counter += 1

                self.bbox_mapping[box_id] = {
                    "page": prov.page_no,
                    "bbox": {
                        "l": float(prov.bbox.l),
                        "t": float(prov.bbox.t),
                        "r": float(prov.bbox.r),
                        "b": float(prov.bbox.b),
                    },
                }

                if hasattr(prov.bbox, "coord_origin"):
                    self.bbox_mapping[box_id]["coord_origin"] = str(prov.bbox.coord_origin)

                return f"<b id={box_id}>{text}</b>"

        return text


def serialize_with_bbox_ids(docling_json_path: Path) -> tuple[str, dict[int, dict[str, Any]]]:
    """
    Serialize Docling JSON document to token-efficient text with bbox IDs.

    Args:
        docling_json_path: Path to Docling JSON file

    Returns:
        Tuple of (serialized_text, bbox_mapping_dict)

    The bbox_mapping_dict maps box_id to:
    {
        "page": page_number,
        "bbox": {"l": left, "t": top, "r": right, "b": bottom}
    }
    """
    logger.debug(f"Loading Docling document from {docling_json_path}")

    # Load the Docling document
    doc = DoclingDocument.load_from_json(docling_json_path)

    # Create custom serializer that tracks bbox mappings
    serializer = BboxMarkdownDocSerializer(doc=doc)
    result = serializer.serialize()

    logger.info(f"Generated serialized text with {len(serializer.bbox_mapping)} bbox IDs")

    return result.text, serializer.bbox_mapping


def extract_text_only(docling_json_path: Path) -> str:
    """
    Extract plain text from Docling JSON document without bbox IDs.

    Args:
        docling_json_path: Path to Docling JSON file

    Returns:
        Plain text content of the document
    """
    logger.debug(f"Loading Docling document from {docling_json_path}")

    # Load the Docling document
    doc = DoclingDocument.load_from_json(docling_json_path)

    # Export to markdown without any modifications
    text = doc.export_to_markdown()

    logger.info(f"Extracted text content ({len(text)} characters)")

    return text


@app.command("convert")
def convert_pdfs(
    papers_dir: Path = typer.Option(
        default=Path("data/papers"), help="Directory containing PDF files"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-convert PDFs even if JSON already exists"
    ),
) -> None:
    """Convert PDF files to Docling JSON format for full-text analysis."""

    if not papers_dir.exists():
        console.print(f"[red]Directory not found: {papers_dir}[/red]")
        raise typer.Exit(1)

    pdf_files = list(papers_dir.glob("*.pdf"))

    if not pdf_files:
        console.print(f"[red]No PDF files found in {papers_dir}[/red]")
        raise typer.Exit(1)

    # Filter PDFs that need conversion
    pdfs_to_convert = []
    for pdf_path in pdf_files:
        json_path = pdf_path.with_suffix(".json")
        if force or not json_path.exists():
            pdfs_to_convert.append(pdf_path)

    if not pdfs_to_convert:
        console.print("[green]All PDFs already converted to Docling format![/green]")
        return

    console.print(f"[bold]Converting {len(pdfs_to_convert)} PDFs to Docling format[/bold]")

    # Configure Docling pipeline
    # NOTE: Picture enrichment is disabled because it generates placeholder-only content
    # that wastes tokens and bbox IDs. For actual image transcription, use a remote VLM
    # like Qwen 3 VL. See: https://docling-project.github.io/docling/examples/pictures_description/
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable OCR as default
    pipeline_options.images_scale = 1.0
    pipeline_options.generate_page_images = True  # Needed for visual extraction
    pipeline_options.generate_picture_images = False  # Disabled - generates useless placeholders

    # Enable only code and formula enrichment
    pipeline_options.do_code_enrichment = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.do_picture_classification = False  # Disabled - wastes tokens
    pipeline_options.do_picture_description = False  # Disabled - wastes tokens

    # Initialize converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    # Convert all PDFs in batch for better performance
    converted = 0
    failed = 0

    with Progress(console=console) as progress:
        task = progress.add_task("Converting PDFs...", total=len(pdfs_to_convert))

        conv_results_iter = iter(
            converter.convert_all(
                pdfs_to_convert,
                raises_on_error=False,  # Continue processing other files even if one fails
            )
        )

        for pdf_path in pdfs_to_convert:
            progress.update(task, description=f"Converting {pdf_path.name}")

            # Fetch the next result - blocks here until conversion completes
            conv_result = next(conv_results_iter)

            try:
                if conv_result.status.name == "SUCCESS":
                    # Export to JSON using the save_as_json method
                    json_path = pdf_path.with_suffix(".json")
                    conv_result.document.save_as_json(json_path)

                    logger.info(f"✅ Converted {pdf_path.name} to Docling JSON")
                    converted += 1
                else:
                    logger.error(f"❌ Failed to convert {pdf_path.name}: {conv_result.status}")
                    failed += 1

            except Exception as e:
                logger.error(f"❌ Failed to convert {pdf_path.name}: {e}")
                failed += 1

            progress.advance(task)

    console.print("\n[bold]Conversion Summary:[/bold]")
    console.print(f"  ✅ Successfully converted: {converted}")
    if failed > 0:
        console.print(f"  ❌ Failed conversions: {failed}")

    console.print("\n[green]PDFs have been converted to Docling JSON format![/green]")


@app.command("serialize")
def extract_text(
    json_file: Path = typer.Argument(..., help="Path to Docling JSON file"),
) -> None:
    """Extract plain text from a Docling JSON document."""

    if not json_file.exists():
        logger.error(f"File not found: {json_file}")
        raise typer.Exit(1)

    try:
        text = extract_text_only(json_file)
        print(text)
    except Exception as e:
        logger.error(f"Failed to extract text from {json_file}: {e}")
        raise typer.Exit(1) from e


@app.command("extract-with-bboxes")
def extract_text_with_bboxes(
    json_file: Path = typer.Argument(..., help="Path to Docling JSON file"),
    output_bbox: Path = typer.Option(
        None, "--bbox-output", "-b", help="Output file for bbox mapping JSON"
    ),
) -> None:
    """Extract text with bbox IDs for PDF highlighting."""

    if not json_file.exists():
        logger.error(f"File not found: {json_file}")
        raise typer.Exit(1)

    try:
        enhanced_markdown, bbox_mapping = serialize_with_bbox_ids(json_file)

        # Output the enhanced markdown to stdout
        print(enhanced_markdown)

        # Save bbox mapping to file if specified
        if output_bbox:
            with open(output_bbox, "w") as f:
                json.dump(bbox_mapping, f, indent=2)
            print(f"Bbox mapping saved to {output_bbox}", file=sys.stderr)

    except Exception as e:
        logger.error(f"Failed to serialize {json_file}: {e}")
        raise typer.Exit(1) from e


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
