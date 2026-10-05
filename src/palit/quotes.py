"""Verbatim quotes against a paper's PDF: highlight boxes, and placeholders.

Extraction cites evidence as verbatim quotes. ``anchorite`` aligns each quote
(Smith-Waterman over normalised text, so ligatures, hyphenation and whitespace
differences don't matter) against the PDF's text layer, which gives the per-line
bounding boxes the report's PDF viewer draws. A quote that can't be placed is
still kept and shown: the viewer says it could not be located.
"""

from typing import Any

import anchorite

# Quotes that stand in for a missing quote rather than quoting anything,
# compared after stripping whitespace and case-folding.
PLACEHOLDER_QUOTES = frozenset({"placeholder", "pending", "n/a", "tbd"})
# A quote with fewer letters and digits than this quotes nothing: "", "x", ":", "-".
MIN_QUOTE_ALNUM_CHARS = 2


def is_placeholder(quote: str) -> bool:
    """Whether *quote* is a placeholder for a missing quote rather than a quote."""
    if sum(char.isalnum() for char in quote) < MIN_QUOTE_ALNUM_CHARS:
        return True
    return quote.strip().casefold() in PLACEHOLDER_QUOTES


class PaperQuotes:
    """Location of quotes in one PDF.

    Construction builds anchorite's index, which is the expensive step (seconds
    for a long paper). PDFium isn't thread-safe, so build instances one at a time.
    """

    def __init__(self, pdf_bytes: bytes) -> None:
        self._index = anchorite.PdfIndex(pdf_bytes)

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
