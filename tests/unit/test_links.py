"""Unit tests for pdf_extractor/links.py."""
from pathlib import Path

import fitz

from pdf_extractor.links import extract_links, splice_links


def _pdf_with_link(path: Path, text: str, uri: str, kind: int = fitz.LINK_URI) -> None:
    """Write a one-page PDF whose text is covered by a single link rect."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 50), text, fontsize=12)
    rect = fitz.Rect(48, 38, 250, 56)  # cover the inserted text line
    link: dict = {"kind": kind, "from": rect}
    if kind == fitz.LINK_URI:
        link["uri"] = uri
    else:
        link["page"] = 0
    page.insert_link(link)
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# extract_links
# ---------------------------------------------------------------------------

def test_extract_returns_anchor_and_uri(tmp_path):
    pdf = tmp_path / "link.pdf"
    _pdf_with_link(pdf, "Visit example", "https://example.com")
    links = extract_links(str(pdf), 1)
    assert len(links) == 1
    anchor, uri = links[0]
    assert "Visit example" in anchor
    assert uri == "https://example.com"


def test_extract_orders_links_top_to_bottom(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=300, height=300)
    page.insert_text((50, 250), "second", fontsize=12)
    page.insert_text((50, 80), "first", fontsize=12)
    page.insert_link(
        {"kind": fitz.LINK_URI, "from": fitz.Rect(48, 70, 150, 88), "uri": "https://1.test"}
    )
    page.insert_link(
        {"kind": fitz.LINK_URI, "from": fitz.Rect(48, 240, 150, 258), "uri": "https://2.test"}
    )
    pdf = tmp_path / "ordered.pdf"
    doc.save(str(pdf))
    doc.close()
    links = extract_links(str(pdf), 1)
    assert [u for _, u in links] == ["https://1.test", "https://2.test"]


def test_extract_skips_internal_goto(tmp_path):
    pdf = tmp_path / "goto.pdf"
    _pdf_with_link(pdf, "jump", "", kind=fitz.LINK_GOTO)
    assert extract_links(str(pdf), 1) == []


def test_extract_skips_link_over_blank_area(tmp_path):
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    # link rect over an empty region — no anchor text to attach
    page.insert_link(
        {"kind": fitz.LINK_URI, "from": fitz.Rect(10, 10, 60, 30), "uri": "https://x.test"}
    )
    pdf = tmp_path / "blanklink.pdf"
    doc.save(str(pdf))
    doc.close()
    assert extract_links(str(pdf), 1) == []


def test_extract_bad_pdf_returns_empty(tmp_path):
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"not a real pdf \x00\x01")
    assert extract_links(str(bad), 1) == []


# ---------------------------------------------------------------------------
# splice_links
# ---------------------------------------------------------------------------

def test_splice_rewrites_anchor():
    out = splice_links("Visit example here", [("example", "https://e.test")])
    assert out == "Visit [example](https://e.test) here"


def test_splice_empty_links_unchanged():
    text = "nothing to do"
    assert splice_links(text, []) == text


def test_splice_anchor_not_found_dropped():
    text = "no match in this line"
    assert splice_links(text, [("absent", "https://e.test")]) == text


def test_splice_url_as_text_becomes_full_link():
    out = splice_links("see https://e.test now", [("https://e.test", "https://e.test")])
    assert out == "see [https://e.test](https://e.test) now"


def test_splice_skips_fenced_code():
    text = "```\nexample\n```"
    assert splice_links(text, [("example", "https://e.test")]) == text


def test_splice_skips_table_rows():
    text = "| example | x |"
    assert splice_links(text, [("example", "https://e.test")]) == text


def test_splice_skips_existing_markdown_link():
    text = "[example](https://old.test) and example"
    # the first (already-linked) occurrence is protected; the plain one is linked
    out = splice_links(text, [("example", "https://new.test")])
    assert out == "[example](https://old.test) and [example](https://new.test)"


def test_splice_skips_inline_code_span():
    text = "`example` then example"
    out = splice_links(text, [("example", "https://e.test")])
    assert out == "`example` then [example](https://e.test)"


def test_splice_escapes_parens_in_uri():
    # Wikipedia-style URLs with parentheses must not truncate the destination.
    out = splice_links("see x now", [("x", "https://e.test/a_(b)")])
    assert out == r"see [x](https://e.test/a_\(b\)) now"


def test_splice_escapes_brackets_in_anchor():
    # Brackets in the anchor must not break the [...] text span.
    out = splice_links("a [b] c", [("[b]", "https://e.test")])
    assert out == r"a [\[b\]](https://e.test) c"


def test_splice_escapes_backslash():
    out = splice_links("path here", [("path", "https://e.test/a\\b")])
    assert out == "[path](https://e.test/a\\\\b) here"


def test_splice_duplicate_anchor_consumed_in_order():
    text = "link and link again"
    out = splice_links(
        text, [("link", "https://one.test"), ("link", "https://two.test")]
    )
    assert out == "[link](https://one.test) and [link](https://two.test) again"