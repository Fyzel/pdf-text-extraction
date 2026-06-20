"""Native table extraction for per-page OCR output.

The vision model (Phase 2) transcribes tables unreliably — most often it drops a
visually-empty leading cell (e.g. the blank top-left corner of a table with a
row-label column), producing a header row with fewer columns than the body rows.
The result is not a valid Markdown table.

For digital PDFs the true table structure is recoverable directly from the page
via PyMuPDF's ``find_tables``. This module extracts those tables, renders them as
aligned GitHub-flavoured Markdown, and splices them over the model's table blocks
in the page text. Scanned/image-only pages yield no tables here, so the model's
own output is left in place as a fallback.
"""
import re

import fitz

from pdf_extractor.pdf_errors import PDF_ERRORS, open_guarded

# Silence the one-time "Consider using the pymupdf_layout package …" notice that
# find_tables prints to stdout; it would otherwise clutter every run.
if hasattr(fitz, "no_recommend_layout"):
    fitz.no_recommend_layout()


def _clean_cell(value: str | None) -> str:
    """Collapse a raw table cell to single-line Markdown-safe text.

    Newlines inside a cell become spaces, runs of whitespace collapse to one,
    and pipe characters are escaped so they do not break the table.

    :param value: Raw cell text from ``find_tables``. Required, but may be
        ``None`` for an empty cell.
    :type value: str | None
    :return: Cleaned single-line cell string.
    :rtype: str
    """
    text: str = (value or "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("|", r"\|")


def _render_table(rows: list[list[str]]) -> str:
    """Render extracted table rows as aligned GitHub-flavoured Markdown.

    The first row is treated as the header. Columns are padded to their widest
    cell so the output matches the project's hand-authored table fixtures.

    :param rows: Table contents as a list of rows, each a list of cell strings.
        Required. All rows are assumed to share the same column count.
    :type rows: list[list[str]]
    :return: Markdown table string (no trailing newline); empty string if
        ``rows`` is empty.
    :rtype: str
    """
    if not rows:
        return ""

    cols: int = max(len(r) for r in rows)
    norm: list[list[str]] = [
        [_clean_cell(r[c] if c < len(r) else "") for c in range(cols)] for r in rows
    ]
    widths: list[int] = [
        max(len(row[c]) for row in norm) for c in range(cols)
    ]

    def _fmt(cells: list[str]) -> str:
        """Render one row's cells as a padded Markdown table line.

        :param cells: Cleaned cell strings for the row. Required.
        :type cells: list[str]
        :return: A ``| a | b |`` line with each cell padded to its column width.
        :rtype: str
        """
        return "| " + " | ".join(cells[c].ljust(widths[c]) for c in range(cols)) + " |"

    header: str = _fmt(norm[0])
    delim: str = "|" + "|".join("-" * (widths[c] + 2) for c in range(cols)) + "|"
    body: list[str] = [_fmt(row) for row in norm[1:]]
    return "\n".join([header, delim, *body])


def extract_tables_markdown(pdf_path: str, page_num: int) -> list[str]:
    """Extract every table on a page as aligned Markdown, top-to-bottom.

    :param pdf_path: Path to the source PDF file. Required.
    :type pdf_path: str
    :param page_num: 1-based page number. Required.
    :type page_num: int
    :return: List of Markdown table strings in page reading order; empty if the
        page has no detectable tables (e.g. a scanned/image-only page).
    :rtype: list[str]
    """
    try:
        with open_guarded(pdf_path) as doc:
            page: fitz.Page = doc[page_num - 1]
            found = page.find_tables()
            tables = sorted(found.tables, key=lambda t: (round(t.bbox[1]), round(t.bbox[0])))
            return [_render_table(t.extract()) for t in tables if t.row_count]
    except PDF_ERRORS:
        # never let table extraction fail a page
        return []


def _table_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Return (start, end) index spans of contiguous Markdown table blocks.

    A table block is a maximal run of lines whose first non-space character is a
    pipe (``|``).

    :param lines: Page text split into lines. Required.
    :type lines: list[str]
    :return: List of half-open ``(start, end)`` index ranges, in order.
    :rtype: list[tuple[int, int]]
    """
    blocks: list[tuple[int, int]] = []
    i: int = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j: int = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def splice_tables(text: str, tables_md: list[str]) -> str:
    """Replace the model's Markdown table blocks with extracted tables.

    Blocks and extracted tables are matched 1:1 in reading order. Any extracted
    tables beyond the number of model blocks are appended at the end of the text
    (covers the rare case where the model omits a table entirely). Model blocks
    beyond the number of extracted tables are left untouched.

    :param text: Per-page Markdown text from the OCR response. Required.
    :type text: str
    :param tables_md: Extracted Markdown tables for the page, in reading order.
        Required (may be empty).
    :type tables_md: list[str]
    :return: Page text with table blocks substituted; unchanged if ``tables_md``
        is empty.
    :rtype: str
    """
    if not tables_md:
        return text

    lines: list[str] = text.split("\n")
    blocks: list[tuple[int, int]] = _table_blocks(lines)

    out: list[str] = []
    cursor: int = 0
    for idx, (start, end) in enumerate(blocks):
        out.extend(lines[cursor:start])
        if idx < len(tables_md):
            out.extend(tables_md[idx].split("\n"))
        else:
            out.extend(lines[start:end])
        cursor = end
    out.extend(lines[cursor:])

    result: str = "\n".join(out)

    # Append any tables the model never emitted a block for.
    for extra in tables_md[len(blocks):]:
        result = f"{result}\n\n{extra}"

    return result
