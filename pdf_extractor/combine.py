"""Phase 3 — Combine per-page markdown files into a single output file."""
from pathlib import Path

from pdf_extractor.state import AppState, StateManager


def run_phase3(
    pdf_path: Path,
    output_dir: Path,
    page_count: int,
    state: AppState,
    state_mgr: StateManager,
) -> tuple[bool, str]:
    """Combine OCR'd per-page markdown files into one output file.

    Pages with ``ocr_failed=True`` are skipped. Each page section is preceded
    by a ``--- PAGE N ---`` separator. The output file is written to the same
    directory as the source PDF, named ``<stem>.md``.

    :param pdf_path: Path to the original PDF file. Required.
    :type pdf_path: pathlib.Path
    :param output_dir: Working directory containing the ``pages/``
        subdirectory. Required.
    :type output_dir: pathlib.Path
    :param page_count: Total page count for the iteration range. Required.
    :type page_count: int
    :param state: Shared application state — read for per-page flags, written on
        success. Required.
    :type state: pdf_extractor.state.AppState
    :param state_mgr: State manager for atomic state persistence. Required.
    :type state_mgr: pdf_extractor.state.StateManager
    :return: Tuple of ``(success, error_message)``; on success
        ``error_message`` is an empty string.
    :rtype: tuple[bool, str]
    """
    pages_dir: Path = output_dir / "pages"
    output_path: Path = pdf_path.parent / f"{pdf_path.stem}.md"

    parts: list[str] = []
    for page_num in range(1, page_count + 1):
        page_state = state.pages[str(page_num)]
        if not page_state.ocr_done:
            continue
        width: int = len(str(page_count))
        stem: str = f"page_{page_num:0{width}d}"
        md_path: Path = pages_dir / f"{stem}.md"
        content: str = md_path.read_text(encoding="utf-8") if md_path.is_file() else ""
        # Per-page md stores diagram refs relative to output_dir (``diagrams/...``).
        # The combined file lives one level up, alongside ``<stem>/``, so rewrite
        # refs to be relative to it: ``<stem>/diagrams/...``.
        content = content.replace("](diagrams/", f"]({output_dir.name}/diagrams/")
        parts.append(f"--- PAGE {page_num} ---\n{content}")

    combined: str = "\n\n".join(parts)

    try:
        output_path.write_text(combined, encoding="utf-8")
    except OSError as exc:
        return False, str(exc)

    state_mgr.mark_combined_done(state)
    return True, ""
