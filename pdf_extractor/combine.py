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

    Args:
        pdf_path: Path to the original PDF file.
        output_dir: Working directory containing ``pages/`` subdirectory.
        page_count: Total page count for iteration range.
        state: Shared AppState — read for per-page flags, written on success.
        state_mgr: StateManager for atomic state persistence.

    Returns:
        Tuple of ``(success, error_message)``.
        On success ``error_message`` is an empty string.
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
        parts.append(f"--- PAGE {page_num} ---\n{content}")

    combined: str = "\n\n".join(parts)

    try:
        output_path.write_text(combined, encoding="utf-8")
    except OSError as exc:
        return False, str(exc)

    state_mgr.mark_combined_done(state)
    return True, ""
