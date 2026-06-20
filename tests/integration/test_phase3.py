"""Integration tests for Phase 3 — combine per-page markdown."""
from pathlib import Path

from pdf_extractor.combine import run_phase3
from pdf_extractor.state import StateManager

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _write_pages(pages_dir: Path, page_count: int, skip: list[int] | None = None) -> None:
    """Write a per-page markdown file for each page, optionally skipping some.

    :param pages_dir: Directory to write the ``page_*.md`` files into. Required.
    :type pages_dir: pathlib.Path
    :param page_count: Total number of pages (sets the zero-pad width). Required.
    :type page_count: int
    :param skip: 1-based page numbers to leave unwritten. Optional; defaults to
        none.
    :type skip: list[int] | None
    :return: ``None``.
    :rtype: None
    """
    pages_dir.mkdir(parents=True, exist_ok=True)
    skip = skip or []
    width = len(str(page_count))
    for i in range(1, page_count + 1):
        if i not in skip:
            stem = f"page_{i:0{width}d}"
            (pages_dir / f"{stem}.md").write_text(
                f"# Page {i}\n\nContent of page {i}.", encoding="utf-8"
            )


def test_phase3_combined_file_structure(tmp_path):
    """The combined file carries a separator and content for every page.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    out = tmp_path / "out"
    out.mkdir()
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    sm = StateManager(out)
    st = sm.load_or_init(pdf, 5)
    _write_pages(out / "pages", 5)
    for i in range(1, 6):
        sm.update_page(st, i, image_done=True, ocr_done=True)

    ok, _ = run_phase3(pdf, out, 5, st, sm)
    assert ok
    content = (tmp_path / "doc.md").read_text(encoding="utf-8")
    for i in range(1, 6):
        assert f"--- PAGE {i} ---" in content
        assert f"Content of page {i}" in content


def test_phase3_gap_pages_skipped(tmp_path):
    """A page whose OCR failed is omitted from the combined output.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    out = tmp_path / "out"
    out.mkdir()
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    sm = StateManager(out)
    st = sm.load_or_init(pdf, 4)
    _write_pages(out / "pages", 4, skip=[2])
    sm.update_page(st, 1, image_done=True, ocr_done=True)
    sm.update_page(st, 2, image_done=True, ocr_failed=True)
    sm.update_page(st, 3, image_done=True, ocr_done=True)
    sm.update_page(st, 4, image_done=True, ocr_done=True)

    run_phase3(pdf, out, 4, st, sm)
    content = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "--- PAGE 1 ---" in content
    assert "--- PAGE 2 ---" not in content
    assert "--- PAGE 3 ---" in content
    assert "--- PAGE 4 ---" in content


def test_phase3_correct_page_order(tmp_path):
    """Pages appear in ascending order in the combined file.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    out = tmp_path / "out"
    out.mkdir()
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    sm = StateManager(out)
    st = sm.load_or_init(pdf, 3)
    _write_pages(out / "pages", 3)
    for i in range(1, 4):
        sm.update_page(st, i, image_done=True, ocr_done=True)

    run_phase3(pdf, out, 3, st, sm)
    content = (tmp_path / "doc.md").read_text(encoding="utf-8")
    pos1 = content.index("--- PAGE 1 ---")
    pos2 = content.index("--- PAGE 2 ---")
    pos3 = content.index("--- PAGE 3 ---")
    assert pos1 < pos2 < pos3


def test_phase3_rewrites_diagram_refs(tmp_path):
    """Diagram references are rewritten relative to the combined file's location.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    out = tmp_path / "out"
    out.mkdir()
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    sm = StateManager(out)
    st = sm.load_or_init(pdf, 1)
    pages_dir = out / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "page_1.md").write_text(
        "# Page 1\n\n![Diagram 1](diagrams/page_1_diagram_1.jpg)", encoding="utf-8"
    )
    sm.update_page(st, 1, image_done=True, ocr_done=True)

    run_phase3(pdf, out, 1, st, sm)
    content = (tmp_path / "doc.md").read_text(encoding="utf-8")
    # Combined file sits beside the ``out/`` dir, so refs gain the dir-name prefix.
    assert "![Diagram 1](out/diagrams/page_1_diagram_1.jpg)" in content
    assert "](diagrams/" not in content


def test_phase3_marks_combined_done(tmp_path):
    """A successful combine sets ``combined_done`` and persists it.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    out = tmp_path / "out"
    out.mkdir()
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    sm = StateManager(out)
    st = sm.load_or_init(pdf, 2)
    _write_pages(out / "pages", 2)
    for i in range(1, 3):
        sm.update_page(st, i, image_done=True, ocr_done=True)

    run_phase3(pdf, out, 2, st, sm)
    assert st.combined_done

    sm2 = StateManager(out)
    st2 = sm2.load_or_init(pdf, 2)
    assert sm2.status(st2) == "complete"
