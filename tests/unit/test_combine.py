"""Unit tests for pdf_extractor/combine.py."""
import pytest
from pathlib import Path

from pdf_extractor.combine import run_phase3
from pdf_extractor.state import StateManager


def _setup(tmp_path: Path, page_count: int, ocr_done: list[int], ocr_failed: list[int] | None = None) -> tuple[Path, StateManager, object]:
    out = tmp_path / "out"; out.mkdir()
    pages = out / "pages"; pages.mkdir()
    pdf = tmp_path / "doc.pdf"; pdf.touch()
    sm = StateManager(out)
    st = sm.load_or_init(pdf, page_count)
    width = len(str(page_count))
    for i in ocr_done:
        stem = f"page_{i:0{width}d}"
        (pages / f"{stem}.md").write_text(f"Content {i}", encoding="utf-8")
        sm.update_page(st, i, image_done=True, ocr_done=True)
    for i in (ocr_failed or []):
        sm.update_page(st, i, image_done=True, ocr_failed=True)
    return pdf, sm, st


# ---------------------------------------------------------------------------
# Basic output
# ---------------------------------------------------------------------------

def test_combine_creates_output_file(tmp_path):
    pdf, sm, st = _setup(tmp_path, 3, ocr_done=[1, 2, 3])
    ok, err = run_phase3(pdf, tmp_path / "out", 3, st, sm)
    assert ok
    assert err == ""
    assert (tmp_path / "doc.md").is_file()


def test_combine_page_separators(tmp_path):
    pdf, sm, st = _setup(tmp_path, 3, ocr_done=[1, 2, 3])
    run_phase3(pdf, tmp_path / "out", 3, st, sm)
    content = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "--- PAGE 1 ---" in content
    assert "--- PAGE 2 ---" in content
    assert "--- PAGE 3 ---" in content


def test_combine_page_content_included(tmp_path):
    pdf, sm, st = _setup(tmp_path, 2, ocr_done=[1, 2])
    run_phase3(pdf, tmp_path / "out", 2, st, sm)
    content = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "Content 1" in content
    assert "Content 2" in content


def test_combine_ascending_order(tmp_path):
    pdf, sm, st = _setup(tmp_path, 3, ocr_done=[1, 2, 3])
    run_phase3(pdf, tmp_path / "out", 3, st, sm)
    content = (tmp_path / "doc.md").read_text(encoding="utf-8")
    p1 = content.index("--- PAGE 1 ---")
    p2 = content.index("--- PAGE 2 ---")
    p3 = content.index("--- PAGE 3 ---")
    assert p1 < p2 < p3


# ---------------------------------------------------------------------------
# Skipping failed pages
# ---------------------------------------------------------------------------

def test_combine_skips_ocr_failed(tmp_path):
    pdf, sm, st = _setup(tmp_path, 3, ocr_done=[1, 3], ocr_failed=[2])
    run_phase3(pdf, tmp_path / "out", 3, st, sm)
    content = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "--- PAGE 1 ---" in content
    assert "--- PAGE 2 ---" not in content
    assert "--- PAGE 3 ---" in content


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

def test_combine_output_sibling_to_pdf(tmp_path):
    pdf, sm, st = _setup(tmp_path, 1, ocr_done=[1])
    run_phase3(pdf, tmp_path / "out", 1, st, sm)
    assert (tmp_path / "doc.md").is_file()


# ---------------------------------------------------------------------------
# combined_done state
# ---------------------------------------------------------------------------

def test_combine_sets_combined_done(tmp_path):
    pdf, sm, st = _setup(tmp_path, 2, ocr_done=[1, 2])
    run_phase3(pdf, tmp_path / "out", 2, st, sm)
    assert st.combined_done


def test_combine_combined_done_persisted(tmp_path):
    pdf, sm, st = _setup(tmp_path, 2, ocr_done=[1, 2])
    run_phase3(pdf, tmp_path / "out", 2, st, sm)
    sm2 = StateManager(tmp_path / "out")
    st2 = sm2.load_or_init(pdf, 2)
    assert st2.combined_done
    assert sm2.status(st2) == "complete"


# ---------------------------------------------------------------------------
# Write error → exit 7
# ---------------------------------------------------------------------------

def test_combine_write_error_returns_failure(tmp_path):
    pdf, sm, st = _setup(tmp_path, 1, ocr_done=[1])
    # make output path a directory so write_text fails
    (tmp_path / "doc.md").mkdir()
    ok, err = run_phase3(pdf, tmp_path / "out", 1, st, sm)
    assert not ok
    assert err != ""


# ---------------------------------------------------------------------------
# Zero-padded filenames
# ---------------------------------------------------------------------------

def test_combine_ten_page_zero_padding(tmp_path):
    pdf, sm, st = _setup(tmp_path, 10, ocr_done=list(range(1, 11)))
    ok, _ = run_phase3(pdf, tmp_path / "out", 10, st, sm)
    assert ok
    content = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert content.count("--- PAGE") == 10
