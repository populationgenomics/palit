"""Tests for quote grounding against a PDF built in memory."""

from palit.quotes import PaperQuotes

TABLE_ROW = "c.1061C>T p.T354M 20 46 3"
FIGURE_LABEL = "KMT2D exon31:c.6752C>T"
FILLER = [
    f"Paragraph {i}: the cohort comprised unrelated probands referred for exome sequencing."
    for i in range(30)
]


def _pdf(lines: list[str]) -> bytes:
    """A one-page PDF with one line of Helvetica text per entry in *lines*."""
    text = "\n".join(
        "(" + line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") Tj T*"
        for line in lines
    )
    stream = f"BT\n/F1 8 Tf\n10 TL\n40 800 Td\n{text}\nET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Length %d >>\nstream\n%b\nendstream" % (len(stream), stream),
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%b\nendobj\n" % (number, body)
    xref = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(pdf)


def _paper() -> PaperQuotes:
    quotes = PaperQuotes(_pdf([*FILLER[:15], TABLE_ROW, FIGURE_LABEL, *FILLER[15:]]))
    assert quotes.has_text_layer
    return quotes


def test_short_verbatim_quotes_are_grounded() -> None:
    check = _paper().check([TABLE_ROW, FIGURE_LABEL])
    assert check.text_layer
    assert check.rejected == []


def test_short_quote_not_in_the_paper_is_rejected() -> None:
    absent = "BRCA1 exon11:c.1234C>T"
    assert _paper().check([TABLE_ROW, absent]).rejected == [absent]


def test_short_verbatim_quotes_are_located() -> None:
    locations = _paper().locate([TABLE_ROW, FIGURE_LABEL])
    assert [len(boxes) for boxes in locations.values()] == [1, 1]
    assert all(box["page"] == 1 for boxes in locations.values() for box in boxes)


def test_without_text_layer_nothing_is_rejected() -> None:
    quotes = PaperQuotes(_pdf([TABLE_ROW]))
    check = quotes.check([TABLE_ROW, "BRCA1 exon11:c.1234C>T"])
    assert not check.text_layer
    assert check.rejected == []
