"""Phase 1 — PDF page rendering to JPEG using PyMuPDF (fitz)."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fitz

_DPI_SCALE: float = 2.0  # 144 DPI — sufficient resolution for qwen2.5vl OCR


def _page_filename(page_num: int, page_count: int) -> str:
    """Return the zero-padded JPEG filename for a page.

    Args:
        page_num: 1-based page number.
        page_count: Total pages in the document, used to determine padding width.

    Returns:
        Filename string such as ``page_001.jpg`` for a 100-page document.
    """
    width: int = len(str(page_count))
    return f"page_{page_num:0{width}d}.jpg"


def _render_page_worker(args: tuple[str, str, int, int]) -> tuple[int, bool, str]:
    """Render one PDF page to JPEG.

    Top-level function required for ``ProcessPoolExecutor`` pickling on Windows.

    Args:
        args: Tuple of ``(pdf_path, pages_dir, page_num, page_count)``.

    Returns:
        Tuple of ``(page_num, success, error_message)``.
        ``error_message`` is an empty string on success.
    """
    pdf_path_str, pages_dir_str, page_num, page_count = args
    output_path: Path = Path(pages_dir_str) / _page_filename(page_num, page_count)
    try:
        doc: fitz.Document = fitz.open(pdf_path_str)
        page: fitz.Page = doc[page_num - 1]
        mat: fitz.Matrix = fitz.Matrix(_DPI_SCALE, _DPI_SCALE)
        pix: fitz.Pixmap = page.get_pixmap(matrix=mat)
        pix.save(str(output_path))
        doc.close()
        return page_num, True, ""
    except Exception as exc:  # noqa: BLE001
        return page_num, False, str(exc)


def render_pages(
    pdf_path: Path,
    pages_dir: Path,
    page_count: int,
    pending: list[int],
    max_workers: int,
) -> list[tuple[int, bool, str]]:
    """Render a subset of PDF pages to JPEG using a process pool.

    Results are returned only after all workers complete, so the caller can
    perform a single serial state-update pass with no inter-process locking.

    Args:
        pdf_path: Path to the source PDF file.
        pages_dir: Directory where JPEG files are written (created if absent).
        page_count: Total pages in the PDF, used for zero-padded filenames.
        pending: 1-based page numbers to render (pages already done are excluded
            by the caller before calling this function).
        max_workers: Maximum number of parallel render processes.

    Returns:
        List of ``(page_num, success, error_message)`` tuples, one per page in
        ``pending``, in submission order.
    """
    pages_dir.mkdir(parents=True, exist_ok=True)

    args_list: list[tuple[str, str, int, int]] = [
        (str(pdf_path), str(pages_dir), page_num, page_count)
        for page_num in pending
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results: list[tuple[int, bool, str]] = list(
            executor.map(_render_page_worker, args_list)
        )

    return results


def get_page_count(pdf_path: Path) -> int:
    """Return the number of pages in a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Total page count.

    Raises:
        fitz.FileNotFoundError: If the PDF cannot be opened.
        Exception: If PyMuPDF raises any other error while reading the file.
    """
    doc: fitz.Document = fitz.open(str(pdf_path))
    count: int = doc.page_count
    doc.close()
    return count
