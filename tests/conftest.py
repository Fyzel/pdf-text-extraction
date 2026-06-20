"""Session-scoped PDF fixture generation via PyMuPDF."""
from pathlib import Path

import fitz
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _make_text_page(doc: fitz.Document, text: str) -> None:
    """Append a text-only page carrying ``text`` to ``doc``.

    :param doc: Open document to append the page to. Required.
    :type doc: fitz.Document
    :param text: Body text to place on the page. Required.
    :type text: str
    :return: ``None``.
    :rtype: None
    """
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), text, fontsize=12)


def _make_diagram_page(doc: fitz.Document, text: str) -> None:
    """Append a page with body text and a simple drawn diagram to ``doc``.

    :param doc: Open document to append the page to. Required.
    :type doc: fitz.Document
    :param text: Body text to place on the page. Required.
    :type text: str
    :return: ``None``.
    :rtype: None
    """
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 100), text, fontsize=12)
    page.draw_rect(fitz.Rect(100, 200, 400, 500), color=(0, 0, 0), width=2)
    page.draw_line((100, 350), (400, 350), color=(0, 0, 0))


def _make_table_page(doc: fitz.Document, cols: int, rows: int) -> None:
    """Append a page with a drawn ``cols`` x ``rows`` ruled table to ``doc``.

    :param doc: Open document to append the page to. Required.
    :type doc: fitz.Document
    :param cols: Number of table columns. Required.
    :type cols: int
    :param rows: Number of table rows. Required.
    :type rows: int
    :return: ``None``.
    :rtype: None
    """
    page = doc.new_page(width=595, height=842)
    col_w = 100
    row_h = 30
    x0, y0 = 72, 100
    for r in range(rows):
        for c in range(cols):
            x = x0 + c * col_w
            y = y0 + r * row_h
            page.draw_rect(fitz.Rect(x, y, x + col_w, y + row_h), color=(0, 0, 0))
            page.insert_text((x + 5, y + 20), f"R{r}C{c}", fontsize=10)


@pytest.fixture(scope="session", autouse=True)
def generate_fixtures() -> None:
    """Generate the shared sample PDFs under ``fixtures/`` once per session.

    Creates ``simple``, ``multipage``, ``diagrams``, ``mixed``, ``tables``, and
    ``corrupt`` PDFs if they do not already exist. Runs automatically for the
    whole test session.

    :return: ``None``.
    :rtype: None
    """
    FIXTURES_DIR.mkdir(exist_ok=True)

    # simple.pdf — 1 page, plain text
    path = FIXTURES_DIR / "simple.pdf"
    if not path.exists():
        doc = fitz.open()
        _make_text_page(doc, "Simple page one. Hello world.")
        doc.save(str(path))
        doc.close()

    # multipage.pdf — 10 pages, plain text
    path = FIXTURES_DIR / "multipage.pdf"
    if not path.exists():
        doc = fitz.open()
        for i in range(1, 11):
            _make_text_page(doc, f"Page {i} of multipage document.")
        doc.save(str(path))
        doc.close()

    # diagrams.pdf — 3 pages with diagrams
    path = FIXTURES_DIR / "diagrams.pdf"
    if not path.exists():
        doc = fitz.open()
        for i in range(1, 4):
            _make_diagram_page(doc, f"Diagram page {i}.")
        doc.save(str(path))
        doc.close()

    # mixed.pdf — 5 pages: pages 1,3,5 text-only; 2,4 have diagrams
    path = FIXTURES_DIR / "mixed.pdf"
    if not path.exists():
        doc = fitz.open()
        for i in range(1, 6):
            if i % 2 == 0:
                _make_diagram_page(doc, f"Mixed page {i} with diagram.")
            else:
                _make_text_page(doc, f"Mixed page {i} text only.")
        doc.save(str(path))
        doc.close()

    # tables.pdf — 3 pages with table grids
    path = FIXTURES_DIR / "tables.pdf"
    if not path.exists():
        doc = fitz.open()
        _make_table_page(doc, cols=3, rows=4)   # simple
        _make_table_page(doc, cols=5, rows=6)   # multi-column
        _make_table_page(doc, cols=4, rows=3)   # complex
        doc.save(str(path))
        doc.close()

    # corrupt.pdf — intentionally malformed
    path = FIXTURES_DIR / "corrupt.pdf"
    if not path.exists():
        path.write_bytes(b"not a pdf file at all \x00\x01\x02")
