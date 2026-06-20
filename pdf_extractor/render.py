"""Phase 1 — PDF page rendering to JPEG using PyMuPDF (fitz)."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fitz

from pdf_extractor.pdf_errors import PDF_ERRORS

_DPI_SCALE: float = 2.0  # default render scale (2.0 = 144 DPI); override via --dpi-scale


def _page_filename(page_num: int, page_count: int) -> str:
    """Return the zero-padded JPEG filename for a page.

    :param page_num: 1-based page number. Required.
    :type page_num: int
    :param page_count: Total pages in the document, used to determine padding
        width. Required.
    :type page_count: int
    :return: Filename string such as ``page_001.jpg`` for a 100-page document.
    :rtype: str
    """
    width: int = len(str(page_count))
    return f"page_{page_num:0{width}d}.jpg"


def _render_page_worker(args: tuple[str, str, int, int, float]) -> tuple[int, bool, str]:
    """Render one PDF page to JPEG.

    Top-level function required for ``ProcessPoolExecutor`` pickling on Windows.
    ``dpi_scale`` is passed in the args tuple rather than read from the module
    global, since worker processes spawned on Windows re-import this module and
    would otherwise see the default rather than a caller-supplied value.

    :param args: Tuple of ``(pdf_path, pages_dir, page_num, page_count,
        dpi_scale)``. Required.
    :type args: tuple[str, str, int, int, float]
    :return: Tuple of ``(page_num, success, error_message)``;
        ``error_message`` is an empty string on success.
    :rtype: tuple[int, bool, str]
    """
    pdf_path_str, pages_dir_str, page_num, page_count, dpi_scale = args
    output_path: Path = Path(pages_dir_str) / _page_filename(page_num, page_count)
    try:
        doc: fitz.Document = fitz.open(pdf_path_str)
        page: fitz.Page = doc[page_num - 1]
        mat: fitz.Matrix = fitz.Matrix(dpi_scale, dpi_scale)
        pix: fitz.Pixmap = page.get_pixmap(matrix=mat)
        pix.save(str(output_path))
        doc.close()
        return page_num, True, ""
    except PDF_ERRORS as exc:
        return page_num, False, str(exc)


def render_pages(
    pdf_path: Path,
    pages_dir: Path,
    page_count: int,
    pending: list[int],
    max_workers: int,
    dpi_scale: float = _DPI_SCALE,
) -> list[tuple[int, bool, str]]:
    """Render a subset of PDF pages to JPEG using a process pool.

    Results are returned only after all workers complete, so the caller can
    perform a single serial state-update pass with no inter-process locking.

    :param pdf_path: Path to the source PDF file. Required.
    :type pdf_path: pathlib.Path
    :param pages_dir: Directory where JPEG files are written (created if
        absent). Required.
    :type pages_dir: pathlib.Path
    :param page_count: Total pages in the PDF, used for zero-padded filenames.
        Required.
    :type page_count: int
    :param pending: 1-based page numbers to render (pages already done are
        excluded by the caller before calling this function). Required.
    :type pending: list[int]
    :param max_workers: Maximum number of parallel render processes. Required.
    :type max_workers: int
    :param dpi_scale: Render scale factor (2.0 = 144 DPI). Higher values yield
        sharper page images at the cost of size and render time. Optional;
        defaults to the module ``_DPI_SCALE`` (2.0).
    :type dpi_scale: float
    :return: List of ``(page_num, success, error_message)`` tuples, one per
        page in ``pending``, in submission order.
    :rtype: list[tuple[int, bool, str]]
    """
    pages_dir.mkdir(parents=True, exist_ok=True)

    args_list: list[tuple[str, str, int, int, float]] = [
        (str(pdf_path), str(pages_dir), page_num, page_count, dpi_scale)
        for page_num in pending
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results: list[tuple[int, bool, str]] = list(
            executor.map(_render_page_worker, args_list)
        )

    return results


def get_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF file.

    :param pdf_path: Path to the PDF file. Required.
    :type pdf_path: pathlib.Path
    :return: Total page count.
    :rtype: int
    :raises fitz.FileNotFoundError: If the PDF cannot be opened.
    :raises Exception: If PyMuPDF raises any other error while reading the file.
    """
    doc: fitz.Document = fitz.open(str(pdf_path))
    count: int = doc.page_count
    doc.close()
    return count
