"""Unit tests for pdf_extractor/headings.py — PDF-driven heading correction."""
from pathlib import Path

import fitz

from pdf_extractor.headings import extract_heading_scale, fix_headings


_BODY = (
    "This is ordinary body text that makes up the bulk of the page so that the "
    "body font size is the most common one measured across all the spans here."
)


def _make_pdf(path: Path) -> None:
    """Build a 1-page PDF: 18pt title, 14pt subheading, 11pt body."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Big Title", fontsize=18)
    page.insert_text((72, 140), "Sub Heading", fontsize=14)
    page.insert_text((72, 200), _BODY, fontsize=11)
    page.insert_text((72, 300), _BODY, fontsize=11)
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# extract_heading_scale
# ---------------------------------------------------------------------------

def test_scale_ranks_heading_sizes_desc(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    assert extract_heading_scale(str(pdf)) == [18.0, 14.0]


def test_scale_excludes_near_body_sizes(tmp_path):
    # A 12pt footer over 11pt body is only 1.09x — below the 1.15 threshold.
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 80), "Heading Big", fontsize=20)
    page.insert_text((72, 760), "Page 1 of 3", fontsize=12)
    page.insert_text((72, 200), _BODY, fontsize=11)
    page.insert_text((72, 300), _BODY, fontsize=11)
    pdf = tmp_path / "footer.pdf"
    doc.save(str(pdf))
    doc.close()
    assert extract_heading_scale(str(pdf)) == [20.0]


def test_scale_empty_for_textless_pdf(tmp_path):
    # A page with no extractable text (e.g. scanned/image-only) yields no scale.
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    pdf = tmp_path / "blank.pdf"
    doc.save(str(pdf))
    doc.close()
    assert extract_heading_scale(str(pdf)) == []


# ---------------------------------------------------------------------------
# fix_headings — relevel / promote / demote
# ---------------------------------------------------------------------------

def test_relevels_model_heading(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    out = fix_headings("# Sub Heading", str(pdf), 1, scale)
    assert out == "## Sub Heading"


def test_keeps_correct_top_level(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    assert fix_headings("# Big Title", str(pdf), 1, scale) == "# Big Title"


def test_promotes_plain_line_that_is_a_heading(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    assert fix_headings("Sub Heading", str(pdf), 1, scale) == "## Sub Heading"


def test_demotes_unbacked_model_heading(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    assert fix_headings("# Not In The Pdf", str(pdf), 1, scale) == "Not In The Pdf"


def test_strips_bold_wrap_when_promoting(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    assert fix_headings("**Sub Heading**", str(pdf), 1, scale) == "## Sub Heading"


def test_fuzzy_match_tolerates_typo(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    # "Sub Headng" (missing 'i') is within the match ratio of "Sub Heading".
    assert fix_headings("# Sub Headng", str(pdf), 1, scale) == "## Sub Headng"


def test_body_line_left_untouched(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    assert fix_headings(_BODY, str(pdf), 1, scale) == _BODY


# ---------------------------------------------------------------------------
# fallbacks
# ---------------------------------------------------------------------------

def test_noop_when_scale_empty(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    assert fix_headings("# Whatever", str(pdf), 1, []) == "# Whatever"


def test_noop_when_pdf_none(tmp_path):
    assert fix_headings("# Whatever", None, 1, [18.0]) == "# Whatever"
