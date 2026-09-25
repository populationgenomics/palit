"""Verbatim quotes against a paper's PDF: grounding checks and highlight boxes.

Extraction cites evidence as verbatim quotes. ``anchorite`` aligns each quote
(Smith-Waterman over normalised text, so ligatures, hyphenation and whitespace
differences don't matter) against the PDF's text layer, which gives both a
grounding check and the per-line bounding boxes the report's PDF viewer draws.
"""

import logging
from dataclasses import dataclass
from typing import Any

import anchorite
import pypdfium2 as pdfium

logger = logging.getLogger(__name__)

MIN_QUOTE_CHARS = 30

# Share of a quote's normalised characters that must align to the PDF text.
# Verbatim quotes reach ~1.0; the margin absorbs text-layer quirks.
MIN_QUOTE_COVERAGE = 0.9

# Below this many text-layer characters the PDF is treated as a scan without
# usable text: quotes can be neither checked nor highlighted.
MIN_TEXT_LAYER_CHARS = 2000


@dataclass(frozen=True)
class QuoteCheck:
    """Quotes that fail the grounding check, and whether a check was possible."""

    rejected: list[str]
    text_layer: bool


class PaperQuotes:
    """Grounding and location of quotes in one PDF.

    Construction extracts the text layer and builds anchorite's index, which is
    the expensive step (seconds for a long paper). PDFium isn't thread-safe, so
    build instances one at a time.
    """

    def __init__(self, pdf_bytes: bytes) -> None:
        document = pdfium.PdfDocument(pdf_bytes)
        self._text = "\n".join(
            document[i].get_textpage().get_text_range() for i in range(len(document))
        )
        self._index = anchorite.PdfIndex(pdf_bytes)

    @property
    def has_text_layer(self) -> bool:
        return len(self._text.strip()) >= MIN_TEXT_LAYER_CHARS

    def check(self, quotes: list[str]) -> QuoteCheck:
        """Quotes that are too short or don't appear verbatim in the PDF."""
        too_short = [q for q in quotes if len(q.strip()) < MIN_QUOTE_CHARS]
        if not self.has_text_layer:
            return QuoteCheck(rejected=too_short, text_layer=False)
        ungrounded = [
            q
            for q in quotes
            if len(q.strip()) >= MIN_QUOTE_CHARS
            and not anchorite.is_quote_grounded(
                self._text, q, fail_coverage=MIN_QUOTE_COVERAGE, strip_html=False
            )
        ]
        return QuoteCheck(rejected=too_short + ungrounded, text_layer=True)

    def locate(self, quotes: list[str]) -> dict[str, list[dict[str, Any]]]:
        """Highlight boxes per quote: 1-based page and 0-1000 page coordinates.

        One box per matched visual line; ``[]`` when the quote can't be placed.
        """
        resolved = self._index.resolve(sorted(set(quotes)))
        return {
            quote: [
                {
                    "page": page + 1,
                    "top": box.top,
                    "left": box.left,
                    "bottom": box.bottom,
                    "right": box.right,
                }
                for page, box in boxes
            ]
            for quote, boxes in resolved.items()
        }
