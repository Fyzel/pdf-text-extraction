"""Shared helpers for the test suite.

Small utilities reused across test modules: building sample PDFs, applying
Phase 1 render results to state, and scanning Markdown table blocks. Centralised
here so the individual test modules do not each carry a copy.
"""
from pathlib import Path

import fitz

from pdf_extractor.state import AppState, StateManager


def make_text_pdf(path: Path, pages: int = 1) -> None:
    """Write a minimal text PDF with ``pages`` pages labelled ``Page N``.

    :param path: Destination path for the PDF. Required.
    :type path: pathlib.Path
    :param pages: Number of pages to create. Optional; defaults to ``1``.
    :type pages: int
    :return: ``None``.
    :rtype: None
    """
    doc = fitz.open()
    for i in range(pages):
        pg = doc.new_page(width=595, height=842)
        pg.insert_text((72, 100), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()


def apply_render_results(
    state_mgr: StateManager,
    state: AppState,
    results: list[tuple[int, bool, str]],
) -> None:
    """Record Phase 1 render results into the processing state.

    :param state_mgr: State manager used to persist each update. Required.
    :type state_mgr: pdf_extractor.state.StateManager
    :param state: Application state to mutate. Required.
    :type state: pdf_extractor.state.AppState
    :param results: ``(page_num, success, error)`` tuples from ``render_pages``.
        Required.
    :type results: list[tuple[int, bool, str]]
    :return: ``None``.
    :rtype: None
    """
    for page_num, success, _ in results:
        if success:
            state_mgr.update_page(state, page_num, image_done=True)
        else:
            state_mgr.update_page(state, page_num, image_failed=True)


def markdown_table_blocks(content: str) -> list[list[str]]:
    """Return each contiguous run of pipe-prefixed lines as a list of lines.

    :param content: Markdown content to scan. Required.
    :type content: str
    :return: Each table block as its list of lines, in document order.
    :rtype: list[list[str]]
    """
    lines = content.split("\n")
    blocks: list[list[str]] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            blocks.append(lines[i:j])
            i = j
        else:
            i += 1
    return blocks
