"""Unit tests for pdf_extractor/tables.py."""
from pathlib import Path

import fitz

from pdf_extractor.tables import (
    _clean_cell,
    _render_table,
    _table_blocks,
    extract_tables_markdown,
    splice_tables,
)


# ---------------------------------------------------------------------------
# _clean_cell
# ---------------------------------------------------------------------------

def test_clean_cell_none_is_empty():
    """A ``None`` cell becomes an empty string.

    :return: ``None``.
    :rtype: None
    """
    assert _clean_cell(None) == ""


def test_clean_cell_collapses_newlines_and_whitespace():
    """Newlines and runs of whitespace collapse to single spaces.

    :return: ``None``.
    :rtype: None
    """
    assert _clean_cell("quaestio posidonium\nno") == "quaestio posidonium no"
    assert _clean_cell("  a   b  ") == "a b"


def test_clean_cell_escapes_pipe():
    """A pipe character in a cell is escaped.

    :return: ``None``.
    :rtype: None
    """
    assert _clean_cell("a|b") == r"a\|b"


# ---------------------------------------------------------------------------
# _render_table
# ---------------------------------------------------------------------------

def test_render_table_empty():
    """Rendering no rows yields an empty string.

    :return: ``None``.
    :rtype: None
    """
    assert not _render_table([])


def test_render_table_aligns_columns_and_keeps_empty_corner():
    """Columns are padded to their widest cell and the empty corner is kept.

    :return: ``None``.
    :rtype: None
    """
    rows = [
        ["", "Column 1", "Column 2"],
        ["Row 1", "usu ad discere", "oporteat ut"],
    ]
    expected = (
        "|       | Column 1       | Column 2    |\n"
        "|-------|----------------|-------------|\n"
        "| Row 1 | usu ad discere | oporteat ut |"
    )
    assert _render_table(rows) == expected


def test_render_table_pads_short_rows_to_column_count():
    """Short rows are padded so every rendered row has equal column count.

    :return: ``None``.
    :rtype: None
    """
    rows = [["A", "B", "C"], ["1"]]
    out = _render_table(rows).split("\n")
    # Every rendered row has the same pipe count.
    assert len({line.count("|") for line in out}) == 1


# ---------------------------------------------------------------------------
# _table_blocks
# ---------------------------------------------------------------------------

def test_table_blocks_finds_contiguous_pipe_runs():
    """Contiguous pipe-prefixed lines are returned as index spans.

    :return: ``None``.
    :rtype: None
    """
    lines = [
        "intro",
        "| a | b |",
        "|---|---|",
        "| 1 | 2 |",
        "outro",
        "| x |",
    ]
    assert _table_blocks(lines) == [(1, 4), (5, 6)]


def test_table_blocks_none():
    """Prose-only lines yield no table blocks.

    :return: ``None``.
    :rtype: None
    """
    assert not _table_blocks(["just", "prose"])


# ---------------------------------------------------------------------------
# splice_tables
# ---------------------------------------------------------------------------

def test_splice_no_tables_returns_unchanged():
    """With no extracted tables, the text is returned unchanged.

    :return: ``None``.
    :rtype: None
    """
    text = "para\n\n| bad |\n| --- |"
    assert splice_tables(text, []) == text


def test_splice_replaces_block_in_place():
    """An extracted table replaces the model's table block in place.

    :return: ``None``.
    :rtype: None
    """
    text = "before\n\n| Column 1 | Column 2 |\n| --- | --- |\n| 1 | 2 | 3 |\n\nafter"
    table = "|  | Column 1 | Column 2 |\n|--|----------|----------|\n| R | 1 | 2 |"
    out = splice_tables(text, [table])
    assert "before" in out and "after" in out
    assert table in out
    assert "| --- | --- |" not in out  # old malformed delimiter gone
    assert "| 1 | 2 | 3 |" not in out  # old body row gone


def test_splice_appends_extra_tables_when_model_omits_block():
    """An extracted table with no matching block is appended at the end.

    :return: ``None``.
    :rtype: None
    """
    text = "only prose, no table here"
    table = "| a |\n|---|\n| 1 |"
    out = splice_tables(text, [table])
    assert out.startswith("only prose")
    assert out.endswith(table)


def test_splice_leaves_extra_model_blocks_untouched():
    """A model block with no extracted match is left untouched.

    :return: ``None``.
    :rtype: None
    """
    text = "| keep | me |\n|------|----|\n| 1 | 2 |\n\n| second |\n|--------|\n| x |"
    table = "| A |\n|---|\n| z |"
    out = splice_tables(text, [table])
    assert table in out
    assert "| second |" in out  # second model block had no extracted match


# ---------------------------------------------------------------------------
# extract_tables_markdown (real PyMuPDF)
# ---------------------------------------------------------------------------

def _make_pdf_with_table(path: Path) -> None:
    """Write a one-page PDF containing a findable 2x2 ruled table.

    :param path: Destination path for the PDF. Required.
    :type path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    # Draw a 2x2 grid and place text in each cell to make a findable table.
    page.draw_line(fitz.Point(50, 100), fitz.Point(350, 100))
    page.draw_line(fitz.Point(50, 150), fitz.Point(350, 150))
    page.draw_line(fitz.Point(50, 200), fitz.Point(350, 200))
    page.draw_line(fitz.Point(50, 100), fitz.Point(50, 200))
    page.draw_line(fitz.Point(200, 100), fitz.Point(200, 200))
    page.draw_line(fitz.Point(350, 100), fitz.Point(350, 200))
    page.insert_text((60, 120), "H1")
    page.insert_text((210, 120), "H2")
    page.insert_text((60, 170), "a")
    page.insert_text((210, 170), "b")
    doc.save(str(path))
    doc.close()


def test_extract_tables_markdown_from_real_pdf(tmp_path):
    """A ruled table in a real PDF is extracted as Markdown.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "t.pdf"
    _make_pdf_with_table(pdf)
    tables = extract_tables_markdown(str(pdf), 1)
    assert len(tables) >= 1
    first = tables[0]
    assert first.count("\n") >= 2  # header + delimiter + at least one row
    assert "H1" in first and "H2" in first


def test_extract_tables_markdown_no_table_returns_empty(tmp_path):
    """A page with no table yields no extracted tables.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "plain.pdf"
    doc = fitz.open()
    doc.new_page(width=400, height=300).insert_text((50, 50), "just prose")
    doc.save(str(pdf))
    doc.close()
    assert not extract_tables_markdown(str(pdf), 1)


def test_extract_tables_markdown_bad_page_returns_empty(tmp_path):
    """An out-of-range page yields no tables instead of raising.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "t.pdf"
    _make_pdf_with_table(pdf)
    # Out-of-range page must not raise.
    assert not extract_tables_markdown(str(pdf), 99)
