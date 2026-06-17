"""Unit tests for pdf_extractor/annotations.py."""
from pathlib import Path

import fitz

from pdf_extractor.annotations import extract_comments_markdown


def _pdf_with_annotations(path: Path) -> None:
    """Write a one-page PDF with a sticky note and a highlight-with-note."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_text((50, 50), "body text")

    note = page.add_text_annot((100, 100), "a sticky note")
    note.set_info(title="Alice")
    note.update()

    hl = page.add_highlight_annot(fitz.Rect(40, 45, 120, 60))
    hl.set_info(content="highlight comment", title="Bob")
    hl.update()

    doc.save(str(path))
    doc.close()


def test_extract_lists_text_bearing_annotations(tmp_path):
    pdf = tmp_path / "annotated.pdf"
    _pdf_with_annotations(pdf)
    md = extract_comments_markdown(str(pdf), 1)
    assert md.startswith("## Comments")
    assert "**Alice** (Text): a sticky note" in md
    assert "**Bob** (Highlight): highlight comment" in md


def test_extract_empty_when_no_annotations(tmp_path):
    doc = fitz.open()
    doc.new_page(width=200, height=200)
    pdf = tmp_path / "plain.pdf"
    doc.save(str(pdf))
    doc.close()
    assert extract_comments_markdown(str(pdf), 1) == ""


def test_extract_skips_contentless_markup(tmp_path):
    # A highlight with no popup note carries no comment text — skip it.
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 20), "x")
    page.add_highlight_annot(fitz.Rect(15, 10, 60, 25))  # no content set
    pdf = tmp_path / "nohl.pdf"
    doc.save(str(pdf))
    doc.close()
    assert extract_comments_markdown(str(pdf), 1) == ""


def test_extract_unknown_author_label(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    note = page.add_text_annot((50, 50), "anonymous note")
    note.update()  # no title/author set
    pdf = tmp_path / "anon.pdf"
    doc.save(str(pdf))
    doc.close()
    md = extract_comments_markdown(str(pdf), 1)
    assert "**Unknown** (Text): anonymous note" in md


def test_extract_bad_pdf_returns_empty(tmp_path):
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"not a real pdf \x00\x01")
    assert extract_comments_markdown(str(bad), 1) == ""
