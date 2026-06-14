"""Unit tests for pdf_extractor/render.py."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitz

from pdf_extractor.render import _page_filename, _render_page_worker, render_pages, get_page_count


# ---------------------------------------------------------------------------
# _page_filename
# ---------------------------------------------------------------------------

def test_page_filename_single_digit_doc():
    assert _page_filename(1, 9) == "page_1.jpg"


def test_page_filename_pads_to_page_count_width():
    assert _page_filename(1, 100) == "page_001.jpg"
    assert _page_filename(10, 100) == "page_010.jpg"
    assert _page_filename(100, 100) == "page_100.jpg"


def test_page_filename_ten_pages():
    assert _page_filename(1, 10) == "page_01.jpg"
    assert _page_filename(10, 10) == "page_10.jpg"


# ---------------------------------------------------------------------------
# _render_page_worker
# ---------------------------------------------------------------------------

def test_render_page_worker_success(tmp_path):
    fixtures = Path(__file__).parent.parent / "fixtures"
    pdf = fixtures / "simple.pdf"
    args = (str(pdf), str(tmp_path), 1, 1)
    page_num, success, error = _render_page_worker(args)
    assert page_num == 1
    assert success
    assert error == ""
    assert (tmp_path / "page_1.jpg").is_file()


def test_render_page_worker_bad_pdf(tmp_path):
    args = (str(tmp_path / "nonexistent.pdf"), str(tmp_path), 1, 1)
    page_num, success, error = _render_page_worker(args)
    assert page_num == 1
    assert not success
    assert error != ""


def test_render_page_worker_corrupt_pdf(tmp_path):
    fixtures = Path(__file__).parent.parent / "fixtures"
    corrupt = fixtures / "corrupt.pdf"
    args = (str(corrupt), str(tmp_path), 1, 1)
    _, success, error = _render_page_worker(args)
    assert not success
    assert error != ""


# ---------------------------------------------------------------------------
# render_pages
# ---------------------------------------------------------------------------

def test_render_pages_simple(tmp_path):
    fixtures = Path(__file__).parent.parent / "fixtures"
    pdf = fixtures / "simple.pdf"
    pages_dir = tmp_path / "pages"
    results = render_pages(pdf, pages_dir, 1, [1], max_workers=1)
    assert len(results) == 1
    page_num, success, error = results[0]
    assert page_num == 1
    assert success
    assert (pages_dir / "page_1.jpg").is_file()


def test_render_pages_multipage(tmp_path):
    fixtures = Path(__file__).parent.parent / "fixtures"
    pdf = fixtures / "multipage.pdf"
    pages_dir = tmp_path / "pages"
    results = render_pages(pdf, pages_dir, 10, list(range(1, 11)), max_workers=2)
    assert len(results) == 10
    assert all(success for _, success, _ in results)
    jpegs = list(pages_dir.glob("*.jpg"))
    assert len(jpegs) == 10
    # no duplicates
    names = {j.name for j in jpegs}
    assert len(names) == 10


def test_render_pages_creates_dir(tmp_path):
    fixtures = Path(__file__).parent.parent / "fixtures"
    pdf = fixtures / "simple.pdf"
    pages_dir = tmp_path / "pages" / "sub"
    render_pages(pdf, pages_dir, 1, [1], max_workers=1)
    assert pages_dir.is_dir()


def test_render_pages_partial_failure(tmp_path):
    fixtures = Path(__file__).parent.parent / "fixtures"
    pdf = fixtures / "simple.pdf"
    pages_dir = tmp_path / "pages"
    # page 99 doesn't exist in a 1-page PDF
    results = render_pages(pdf, pages_dir, 99, [99], max_workers=1)
    _, success, error = results[0]
    assert not success
    assert error != ""


# ---------------------------------------------------------------------------
# get_page_count
# ---------------------------------------------------------------------------

def test_get_page_count_simple():
    fixtures = Path(__file__).parent.parent / "fixtures"
    assert get_page_count(fixtures / "simple.pdf") == 1


def test_get_page_count_multipage():
    fixtures = Path(__file__).parent.parent / "fixtures"
    assert get_page_count(fixtures / "multipage.pdf") == 10


def test_get_page_count_corrupt_raises():
    fixtures = Path(__file__).parent.parent / "fixtures"
    with pytest.raises(Exception):
        get_page_count(fixtures / "corrupt.pdf")
