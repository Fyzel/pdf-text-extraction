"""Unit tests for pdf_extractor/headings.py — PDF-driven heading correction."""
from pathlib import Path

import fitz

from pdf_extractor.headings import extract_heading_scale, fix_headings


_BODY = (
    "This is ordinary body text that makes up the bulk of the page so that the "
    "body font size is the most common one measured across all the spans here."
)


def _make_pdf(path: Path) -> None:
    """Build a 1-page PDF: 18pt title, 14pt subheading, 11pt body.

    :param path: Destination path for the generated PDF file. Required.
    :type path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
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
    """The heading scale lists the document's heading font sizes largest first.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    assert extract_heading_scale(str(pdf)) == [18.0, 14.0]


def test_scale_excludes_near_body_sizes(tmp_path):
    """Font sizes only marginally larger than body (below the 1.15x ratio) are excluded.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
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
    """A PDF with no extractable text (scanned/image-only) yields an empty scale.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    # A page with no extractable text (e.g. scanned/image-only) yields no scale.
    doc = fitz.open()
    doc.new_page(width=595, height=842)
    pdf = tmp_path / "blank.pdf"
    doc.save(str(pdf))
    doc.close()
    assert extract_heading_scale(str(pdf)) == []


def test_scale_empty_for_missing_pdf(tmp_path):
    """A missing/unreadable PDF path must not raise — it yields an empty scale.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    missing = tmp_path / "does-not-exist.pdf"
    assert extract_heading_scale(str(missing)) == []


def test_scale_empty_for_unreadable_pdf(tmp_path):
    """A file that is not a valid PDF must not raise — it yields an empty scale.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    bad = tmp_path / "garbage.pdf"
    bad.write_bytes(b"this is not a pdf")
    assert extract_heading_scale(str(bad)) == []


# ---------------------------------------------------------------------------
# fix_headings — relevel / promote / demote
# ---------------------------------------------------------------------------

def test_relevels_model_heading(tmp_path):
    """A model heading is releveled to its PDF-derived level (``#`` → ``##``).

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    out = fix_headings("# Sub Heading", str(pdf), 1, scale)
    assert out == "## Sub Heading"


def test_keeps_correct_top_level(tmp_path):
    """A heading already at the correct top level is left unchanged.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    assert fix_headings("# Big Title", str(pdf), 1, scale) == "# Big Title"


def test_promotes_plain_line_that_is_a_heading(tmp_path):
    """A plain line that matches a PDF heading is promoted to that heading level.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    assert fix_headings("Sub Heading", str(pdf), 1, scale) == "## Sub Heading"


def test_demotes_unbacked_model_heading(tmp_path):
    """A model heading the PDF does not back is demoted to plain text.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    assert fix_headings("# Not In The Pdf", str(pdf), 1, scale) == "Not In The Pdf"


def test_strips_bold_wrap_when_promoting(tmp_path):
    """Whole-line bold wrapping is stripped when a line is promoted to a heading.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    assert fix_headings("**Sub Heading**", str(pdf), 1, scale) == "## Sub Heading"


def test_fuzzy_match_tolerates_typo(tmp_path):
    """A heading with minor OCR drift still matches and is releveled.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    # "Sub Headng" (missing 'i') is within the match ratio of "Sub Heading".
    assert fix_headings("# Sub Headng", str(pdf), 1, scale) == "## Sub Headng"


def test_body_line_left_untouched(tmp_path):
    """Ordinary body prose that matches no PDF heading is returned unchanged.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    assert fix_headings(_BODY, str(pdf), 1, scale) == _BODY


# ---------------------------------------------------------------------------
# fallbacks
# ---------------------------------------------------------------------------

def test_noop_when_scale_empty(tmp_path):
    """With an empty heading scale, the text is returned untouched.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    assert fix_headings("# Whatever", str(pdf), 1, []) == "# Whatever"


def test_noop_when_pdf_none():
    """With no source PDF (``pdf_path`` is ``None``), the text is returned untouched.

    Takes no fixtures: the no-PDF code path needs no file on disk.

    :return: ``None``.
    :rtype: None
    """
    assert fix_headings("# Whatever", None, 1, [18.0]) == "# Whatever"


# ---------------------------------------------------------------------------
# fenced code blocks left untouched (issue #72)
# ---------------------------------------------------------------------------

def test_skips_heading_like_line_inside_fence(tmp_path):
    """A heading-like line inside a fenced code block is left verbatim.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    # "# Sub Heading" inside a fence is a code comment, not a heading — verbatim.
    text = "```\n# Sub Heading\n```"
    assert fix_headings(text, str(pdf), 1, scale) == text


def test_skips_promotion_inside_fence(tmp_path):
    """A plain line matching a PDF heading is not promoted inside a fence.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    # A plain line matching a PDF heading must not be promoted inside a fence.
    text = "```python\nSub Heading\n```"
    assert fix_headings(text, str(pdf), 1, scale) == text


def test_skips_tilde_fence(tmp_path):
    """A tilde-delimited (``~~~``) fenced block is also skipped.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    text = "~~~\n# Sub Heading\n~~~"
    assert fix_headings(text, str(pdf), 1, scale) == text


def test_corrects_heading_after_closed_fence(tmp_path):
    """A heading after a closed fence is corrected once fence state toggles off.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    # Fence state toggles closed; the heading after it is still corrected.
    text = "```\n# Sub Heading\n```\n# Sub Heading"
    expected = "```\n# Sub Heading\n```\n## Sub Heading"
    assert fix_headings(text, str(pdf), 1, scale) == expected


def test_different_marker_does_not_close_fence(tmp_path):
    """A ``~~~`` line does not close a ```` ``` ````-opened fence.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    # A ~~~ line inside a ```-opened block must not close it; the heading-like
    # line that follows is still code and must stay verbatim.
    text = "```\n~~~\n# Sub Heading\n```"
    assert fix_headings(text, str(pdf), 1, scale) == text


def test_shorter_marker_does_not_close_fence(tmp_path):
    """A closing run shorter than the opening run does not close the fence.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    # Closing run must be at least as long as the opening run (4 backticks).
    text = "````\n```\n# Sub Heading\n````"
    assert fix_headings(text, str(pdf), 1, scale) == text


def test_info_string_does_not_close_fence(tmp_path):
    """A fence line carrying an info string is not a valid close.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    # A fence line carrying extra text is not a valid close — still inside code.
    text = "```\n``` not a close\n# Sub Heading\n```"
    assert fix_headings(text, str(pdf), 1, scale) == text


def test_longer_marker_closes_fence(tmp_path):
    """A longer same-character run is a valid close; the next heading is corrected.

    :param tmp_path: pytest temporary-directory fixture. Required; injected by
        pytest.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    scale = extract_heading_scale(str(pdf))
    # A longer same-char run is a valid close; the heading after it is corrected.
    text = "```\n# Sub Heading\n`````\n# Sub Heading"
    expected = "```\n# Sub Heading\n`````\n## Sub Heading"
    assert fix_headings(text, str(pdf), 1, scale) == expected
