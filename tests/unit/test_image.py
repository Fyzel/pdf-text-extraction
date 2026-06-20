"""Unit tests for pdf_extractor/render.py."""
from pathlib import Path

import fitz
import pytest

from pdf_extractor.render import (
    _page_filename,
    _render_page_worker,
    get_page_count,
    render_pages,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# _page_filename
# ---------------------------------------------------------------------------

def test_page_filename_single_digit_doc():
    """A single-digit page count yields an unpadded filename.

    :return: ``None``.
    :rtype: None
    """
    assert _page_filename(1, 9) == "page_1.jpg"


def test_page_filename_pads_to_page_count_width():
    """Filenames are zero-padded to the page-count width.

    :return: ``None``.
    :rtype: None
    """
    assert _page_filename(1, 100) == "page_001.jpg"
    assert _page_filename(10, 100) == "page_010.jpg"
    assert _page_filename(100, 100) == "page_100.jpg"


def test_page_filename_ten_pages():
    """A ten-page document pads filenames to two digits.

    :return: ``None``.
    :rtype: None
    """
    assert _page_filename(1, 10) == "page_01.jpg"
    assert _page_filename(10, 10) == "page_10.jpg"


# ---------------------------------------------------------------------------
# _render_page_worker
# ---------------------------------------------------------------------------

def test_render_page_worker_success(tmp_path):
    """The worker renders a valid page to JPEG and reports success.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = FIXTURES / "simple.pdf"
    args = (str(pdf), str(tmp_path), 1, 1, 2.0)
    page_num, success, error = _render_page_worker(args)
    assert page_num == 1
    assert success
    assert error == ""
    assert (tmp_path / "page_1.jpg").is_file()


def test_render_page_worker_bad_pdf(tmp_path):
    """A missing PDF makes the worker report failure with a message.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    args = (str(tmp_path / "nonexistent.pdf"), str(tmp_path), 1, 1, 2.0)
    page_num, success, error = _render_page_worker(args)
    assert page_num == 1
    assert not success
    assert error != ""


def test_render_page_worker_corrupt_pdf(tmp_path):
    """A corrupt PDF makes the worker report failure with a message.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    corrupt = FIXTURES / "corrupt.pdf"
    args = (str(corrupt), str(tmp_path), 1, 1, 2.0)
    _, success, error = _render_page_worker(args)
    assert not success
    assert error != ""


def test_render_page_worker_dpi_scale_affects_size(tmp_path):
    """A higher DPI scale produces a proportionally wider rendered image.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = FIXTURES / "simple.pdf"
    lo = tmp_path / "lo"
    lo.mkdir()
    hi = tmp_path / "hi"
    hi.mkdir()
    _render_page_worker((str(pdf), str(lo), 1, 1, 1.0))
    _render_page_worker((str(pdf), str(hi), 1, 1, 4.0))
    w_lo = fitz.Pixmap(str(lo / "page_1.jpg")).width
    w_hi = fitz.Pixmap(str(hi / "page_1.jpg")).width
    assert w_hi == w_lo * 4


# ---------------------------------------------------------------------------
# render_pages
# ---------------------------------------------------------------------------

def test_render_pages_simple(tmp_path):
    """Rendering a single page returns one success result and the JPEG.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = FIXTURES / "simple.pdf"
    pages_dir = tmp_path / "pages"
    results = render_pages(pdf, pages_dir, 1, [1], max_workers=1)
    assert len(results) == 1
    page_num, success, _ = results[0]
    assert page_num == 1
    assert success
    assert (pages_dir / "page_1.jpg").is_file()


def test_render_pages_multipage(tmp_path):
    """Rendering ten pages yields ten unique JPEGs.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = FIXTURES / "multipage.pdf"
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
    """The pages directory is created when absent.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = FIXTURES / "simple.pdf"
    pages_dir = tmp_path / "pages" / "sub"
    render_pages(pdf, pages_dir, 1, [1], max_workers=1)
    assert pages_dir.is_dir()


def test_render_pages_partial_failure(tmp_path):
    """An out-of-range page is reported as a failure result.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = FIXTURES / "simple.pdf"
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
    """A single-page PDF reports a page count of 1.

    :return: ``None``.
    :rtype: None
    """
    assert get_page_count(FIXTURES / "simple.pdf") == 1


def test_get_page_count_multipage():
    """A ten-page PDF reports a page count of 10.

    :return: ``None``.
    :rtype: None
    """
    assert get_page_count(FIXTURES / "multipage.pdf") == 10


def test_get_page_count_corrupt_raises():
    """A corrupt PDF raises when its page count is requested.

    :return: ``None``.
    :rtype: None
    """
    with pytest.raises(RuntimeError):
        get_page_count(FIXTURES / "corrupt.pdf")
