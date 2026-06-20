"""Integration tests for Phase 1 — real PyMuPDF rendering."""
from pathlib import Path

from pdf_extractor.pdf_errors import PDF_ERRORS
from pdf_extractor.render import get_page_count, render_pages
from pdf_extractor.state import StateManager
from tests.helpers import apply_render_results

FIXTURES = Path(__file__).parent.parent / "fixtures"
DATA = Path(__file__).parent.parent / "data"


def test_phase1_simple_produces_jpeg(tmp_path):
    """Rendering a single-page PDF writes one JPEG and reports success.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = FIXTURES / "simple.pdf"
    pages_dir = tmp_path / "pages"
    results = render_pages(pdf, pages_dir, 1, [1], max_workers=1)
    assert len(results) == 1
    _, success, _ = results[0]
    assert success
    assert (pages_dir / "page_1.jpg").is_file()


def test_phase1_multipage_all_jpegs(tmp_path):
    """Rendering the 10-page fixture writes correctly zero-padded JPEGs.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = FIXTURES / "multipage.pdf"
    count = get_page_count(pdf)
    assert count == 10
    pages_dir = tmp_path / "pages"
    results = render_pages(pdf, pages_dir, count, list(range(1, count + 1)), max_workers=2)
    assert all(success for _, success, _ in results)
    jpegs = sorted(pages_dir.glob("*.jpg"))
    assert len(jpegs) == 10
    # unique names, correct zero-padding
    names = {j.name for j in jpegs}
    assert "page_01.jpg" in names
    assert "page_10.jpg" in names


def test_phase1_multipage_no_duplicates(tmp_path):
    """Concurrent rendering returns each page exactly once.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = FIXTURES / "multipage.pdf"
    count = get_page_count(pdf)
    pages_dir = tmp_path / "pages"
    results = render_pages(pdf, pages_dir, count, list(range(1, count + 1)), max_workers=4)
    page_nums = [pn for pn, _, _ in results]
    assert len(page_nums) == len(set(page_nums))


def test_phase1_corrupt_pdf_all_fail(tmp_path):
    """A corrupt PDF fails to render rather than producing output.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    corrupt = FIXTURES / "corrupt.pdf"
    pages_dir = tmp_path / "pages"
    # corrupt PDF may fail at get_page_count; render_pages uses page 1 directly
    try:
        results = render_pages(corrupt, pages_dir, 1, [1], max_workers=1)
        _, success, error = results[0]
        assert not success
        assert error != ""
    except PDF_ERRORS:
        pass  # may also raise during open — acceptable


def test_phase1_diagrams_pdf(tmp_path):
    """The 3-page diagrams fixture renders all pages to JPEG.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = FIXTURES / "diagrams.pdf"
    count = get_page_count(pdf)
    assert count == 3
    pages_dir = tmp_path / "pages"
    results = render_pages(pdf, pages_dir, count, list(range(1, count + 1)), max_workers=1)
    assert all(success for _, success, _ in results)
    assert len(list(pages_dir.glob("*.jpg"))) == 3


def test_phase1_state_updated_after_render(tmp_path):
    """A successful render flips ``image_done`` (and not ``image_failed``).

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = FIXTURES / "simple.pdf"
    out = tmp_path / "out"
    out.mkdir()
    pages_dir = out / "pages"
    sm = StateManager(out)
    st = sm.load_or_init(pdf, 1)
    results = render_pages(pdf, pages_dir, 1, [1], max_workers=1)
    apply_render_results(sm, st, results)
    assert st.pages["1"].image_done
    assert not st.pages["1"].image_failed


# ---------------------------------------------------------------------------
# tests/data — real PDFs
# ---------------------------------------------------------------------------

def test_phase1_data_001_one_page_text(tmp_path):
    """The one-page data PDF reports one page and renders one JPEG.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = DATA / "test-001--one-page-text.pdf"
    count = get_page_count(pdf)
    assert count == 1
    pages_dir = tmp_path / "pages"
    results = render_pages(pdf, pages_dir, count, list(range(1, count + 1)), max_workers=1)
    assert all(success for _, success, _ in results)
    assert len(list(pages_dir.glob("*.jpg"))) == 1


def test_phase1_data_002_two_page_text(tmp_path):
    """The two-page data PDF reports two pages and renders two JPEGs.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = DATA / "test-002--two-page-text.pdf"
    count = get_page_count(pdf)
    assert count == 2
    pages_dir = tmp_path / "pages"
    results = render_pages(pdf, pages_dir, count, list(range(1, count + 1)), max_workers=1)
    assert all(success for _, success, _ in results)
    assert len(list(pages_dir.glob("*.jpg"))) == 2


def test_phase1_data_003_three_page_text_diagram(tmp_path):
    """Data PDF 003 reports three pages and renders three JPEGs.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = DATA / "test-003--three-page-text-diagram.pdf"
    count = get_page_count(pdf)
    assert count == 3
    pages_dir = tmp_path / "pages"
    results = render_pages(pdf, pages_dir, count, list(range(1, count + 1)), max_workers=1)
    assert all(success for _, success, _ in results)
    assert len(list(pages_dir.glob("*.jpg"))) == 3


def test_phase1_data_004_three_page_text_diagram(tmp_path):
    """Data PDF 004 reports three pages and renders three JPEGs.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = DATA / "test-004--three-page-text-diagram.pdf"
    count = get_page_count(pdf)
    assert count == 3
    pages_dir = tmp_path / "pages"
    results = render_pages(pdf, pages_dir, count, list(range(1, count + 1)), max_workers=1)
    assert all(success for _, success, _ in results)
    assert len(list(pages_dir.glob("*.jpg"))) == 3
