"""Live end-to-end tests — require a reachable Ollama instance with qwen2.5vl:7b.

Run with:
    pytest -m live

Skip in CI with:
    pytest -m "not live"
"""
import difflib
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from pdf_extractor.cli import run
from pdf_extractor.config import OllamaInstance
from pdf_extractor.health import probe_instances
from pdf_extractor.mdlint import normalize_markdown

pytestmark = pytest.mark.live

DATA = Path(__file__).parent.parent / "data"
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_COMPARE_SIZE: int = 128  # px — both images downsampled to this before comparison


def _image_similarity(path_a: Path, path_b: Path) -> float:
    """Pixel-level grayscale similarity between two images. Returns 0.0–1.0.

    Both images are downsampled to _COMPARE_SIZE x _COMPARE_SIZE to normalise
    resolution and aspect-ratio differences before comparison.
    """
    def _samples(p: Path) -> bytes:
        doc = fitz.open(str(p))
        page = doc[0]
        sx = _COMPARE_SIZE / page.rect.width
        sy = _COMPARE_SIZE / page.rect.height
        pix = page.get_pixmap(matrix=fitz.Matrix(sx, sy), colorspace=fitz.csGRAY)
        doc.close()
        return pix.samples

    a = _samples(path_a)
    b = _samples(path_b)
    if len(a) != len(b):
        return 0.0
    diff = sum(abs(x - y) for x, y in zip(a, b))
    return 1.0 - diff / (255 * len(a))


def _assert_tables_well_formed(content: str) -> None:
    """Assert every Markdown table block has a consistent column count.

    Scans contiguous runs of pipe-prefixed lines and checks that the header,
    delimiter, and all body rows share the same number of cells. Guards against
    the malformed-table regression in issue #44.
    """
    lines = content.split("\n")
    blocks = 0
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            block = lines[i:j]
            counts = {line.count("|") for line in block}
            assert len(counts) == 1, (
                f"Table block has inconsistent column counts {counts}:\n"
                + "\n".join(block)
            )
            assert len(block) >= 2, "Table block missing delimiter row"
            blocks += 1
            i = j
        else:
            i += 1
    assert blocks >= 1, "Expected at least one Markdown table in the output"


def _load_project_config() -> dict:
    cfg_path = _PROJECT_ROOT / "ollama.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {"instances": [{"url": "http://localhost:11434", "model": "qwen2.5vl:7b"}]}


@pytest.fixture(scope="module")
def live_config() -> dict:
    cfg = _load_project_config()
    instances = [
        OllamaInstance(url=i["url"], model=i.get("model", "qwen2.5vl:7b"))
        for i in cfg["instances"]
    ]
    if not probe_instances(instances):
        pytest.skip("No Ollama instances reachable — skipping live tests")
    return cfg


def _run_pipeline(tmp_path: Path, pdf_name: str, cfg: dict) -> tuple[int, str]:
    src = DATA / pdf_name
    dst = tmp_path / pdf_name
    shutil.copy2(src, dst)
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    with patch.object(sys, "argv", ["main.py", str(dst)]):
        code = run()
    stem = Path(pdf_name).stem
    out_md = tmp_path / f"{stem}.md"
    content = out_md.read_text(encoding="utf-8") if out_md.exists() else ""
    return code, content


# ---------------------------------------------------------------------------
# test-001 — 1 page, text only
# ---------------------------------------------------------------------------

def test_live_001_one_page_text(tmp_path, monkeypatch, live_config):
    monkeypatch.chdir(tmp_path)
    code, content = _run_pipeline(tmp_path, "test-001--one-page-text.pdf", live_config)

    assert code == 0
    assert content.count("--- PAGE") == 1
    assert "--- PAGE 1 ---" in content
    assert "> " not in content, "Blockquote prefix '> ' must not appear in OCR output"

    expected = (DATA / "test-001--one-page-text-expected.md").read_text(encoding="utf-8")
    ratio = difflib.SequenceMatcher(None, content.strip(), expected.strip()).ratio()
    assert ratio >= 0.85, f"OCR output similarity to expected is only {ratio:.2%}"

    diag_dir = tmp_path / "test-001--one-page-text" / "diagrams"
    assert not diag_dir.exists() or not any(diag_dir.glob("*.jpg"))


# ---------------------------------------------------------------------------
# test-002 — 2 pages, text only
# ---------------------------------------------------------------------------

def test_live_002_two_page_text(tmp_path, monkeypatch, live_config):
    monkeypatch.chdir(tmp_path)
    code, content = _run_pipeline(tmp_path, "test-002--two-page-text.pdf", live_config)

    assert code == 0
    assert content.count("--- PAGE") == 2
    assert content.index("--- PAGE 1 ---") < content.index("--- PAGE 2 ---")
    assert "> " not in content, "Blockquote prefix '> ' must not appear in OCR output"

    expected = (DATA / "test-002--two-page-text-expected.md").read_text(encoding="utf-8")
    ratio = difflib.SequenceMatcher(None, content.strip(), expected.strip()).ratio()
    assert ratio >= 0.85, f"OCR output similarity to expected is only {ratio:.2%}"

    diag_dir = tmp_path / "test-002--two-page-text" / "diagrams"
    assert not diag_dir.exists() or not any(diag_dir.glob("*.jpg"))


# ---------------------------------------------------------------------------
# test-003 — 3 pages, text + diagram
# ---------------------------------------------------------------------------

def test_live_003_three_page_text_diagram(tmp_path, monkeypatch, live_config):
    monkeypatch.chdir(tmp_path)
    code, content = _run_pipeline(
        tmp_path, "test-003--three-page-text-diagram.pdf", live_config
    )

    assert code == 0
    assert content.count("--- PAGE") == 3
    assert content.index("--- PAGE 1 ---") < content.index("--- PAGE 2 ---") < content.index("--- PAGE 3 ---")
    assert "> " not in content, "Blockquote prefix '> ' must not appear in OCR output"

    expected = (DATA / "test-003--three-page-text-diagram-expected.md").read_text(encoding="utf-8")
    ratio = difflib.SequenceMatcher(None, content.strip(), expected.strip()).ratio()
    assert ratio >= 0.85, f"OCR output similarity to expected is only {ratio:.2%}"

    diag_dir = tmp_path / "test-003--three-page-text-diagram" / "diagrams"
    assert diag_dir.is_dir(), "Expected diagram directory to be created"
    extracted = sorted(diag_dir.glob("*.jpg"))
    assert len(extracted) > 0, "Expected at least one diagram image"
    assert "![Diagram" in content, "Expected diagram Markdown reference in output"

    reference = DATA / "images" / "test-003-diagram1.png"
    sim = _image_similarity(extracted[0], reference)
    assert sim >= 0.75, f"Extracted diagram similarity to reference is only {sim:.2%}"


# ---------------------------------------------------------------------------
# test-004 — 3 pages, text + diagram
# ---------------------------------------------------------------------------

def test_live_004_three_page_text_diagram(tmp_path, monkeypatch, live_config):
    monkeypatch.chdir(tmp_path)
    code, content = _run_pipeline(
        tmp_path, "test-004--three-page-text-diagram.pdf", live_config
    )

    assert code == 0
    assert content.count("--- PAGE") == 3
    assert content.index("--- PAGE 1 ---") < content.index("--- PAGE 2 ---") < content.index("--- PAGE 3 ---")
    assert "> " not in content, "Blockquote prefix '> ' must not appear in OCR output"

    expected = (DATA / "test-004--three-page-text-diagram-expected.md").read_text(encoding="utf-8")
    ratio = difflib.SequenceMatcher(None, content.strip(), expected.strip()).ratio()
    assert ratio >= 0.85, f"OCR output similarity to expected is only {ratio:.2%}"

    diag_dir = tmp_path / "test-004--three-page-text-diagram" / "diagrams"
    assert diag_dir.is_dir(), "Expected diagram directory to be created"
    extracted = sorted(diag_dir.glob("*.jpg"))
    assert len(extracted) > 0, "Expected at least one diagram image"
    assert "![Diagram" in content, "Expected diagram Markdown reference in output"

    reference = DATA / "images" / "test-004-diagram1.png"
    sim = _image_similarity(extracted[0], reference)
    assert sim >= 0.75, f"Extracted diagram similarity to reference is only {sim:.2%}"


# ---------------------------------------------------------------------------
# test-005 — 3 pages, text + diagram + table
# ---------------------------------------------------------------------------

def test_live_005_three_page_text_diagram_table(tmp_path, monkeypatch, live_config):
    monkeypatch.chdir(tmp_path)
    code, content = _run_pipeline(
        tmp_path, "test-005--three-page-text-diagram-table.pdf", live_config
    )

    assert code == 0
    assert content.count("--- PAGE") == 3
    assert content.index("--- PAGE 1 ---") < content.index("--- PAGE 2 ---") < content.index("--- PAGE 3 ---")
    assert "> " not in content, "Blockquote prefix '> ' must not appear in OCR output"
    assert "|" in content, "Expected a Markdown table in the output text"
    _assert_tables_well_formed(content)

    expected = (DATA / "test-005--three-page-text-diagram-table-expected.md").read_text(encoding="utf-8")
    ratio = difflib.SequenceMatcher(None, content.strip(), expected.strip()).ratio()
    assert ratio >= 0.85, f"OCR output similarity to expected is only {ratio:.2%}"

    diag_dir = tmp_path / "test-005--three-page-text-diagram-table" / "diagrams"
    assert diag_dir.is_dir(), "Expected diagram directory to be created"
    extracted = sorted(diag_dir.glob("*.jpg"))
    assert len(extracted) > 0, "Expected at least one diagram image"
    assert "![Diagram" in content, "Expected diagram Markdown reference in output"

    reference = DATA / "images" / "test-005-diagram1.png"
    sim = _image_similarity(extracted[0], reference)
    assert sim >= 0.75, f"Extracted diagram similarity to reference is only {sim:.2%}"


# ---------------------------------------------------------------------------
# test-006 — 3 pages, text + diagram + table + bullets
# ---------------------------------------------------------------------------

def test_live_006_three_page_text_diagram_table_bullets(tmp_path, monkeypatch, live_config):
    monkeypatch.chdir(tmp_path)
    code, content = _run_pipeline(
        tmp_path, "test-006--three-page-text-diagram-table-bullets.pdf", live_config
    )

    assert code == 0
    assert content.count("--- PAGE") == 3
    assert content.index("--- PAGE 1 ---") < content.index("--- PAGE 2 ---") < content.index("--- PAGE 3 ---")
    assert "> " not in content, "Blockquote prefix '> ' must not appear in OCR output"
    assert "|" in content, "Expected a Markdown table in the output text"
    assert any(
        line.lstrip().startswith(("- ", "* ")) for line in content.splitlines()
    ), "Expected a Markdown bullet list in the output text"

    # Page output must already be list-normalised (issue #39): re-running the
    # normaliser is a no-op when markers and numbering are valid.
    assert normalize_markdown(content) == content, (
        "Combined output contains list markup the normaliser would still fix"
    )
    # Tables must be well-formed (issue #44).
    _assert_tables_well_formed(content)

    expected = (DATA / "test-006--three-page-text-diagram-table-bullets-expected.md").read_text(encoding="utf-8")
    ratio = difflib.SequenceMatcher(None, content.strip(), expected.strip()).ratio()
    assert ratio >= 0.85, f"OCR output similarity to expected is only {ratio:.2%}"

    diag_dir = tmp_path / "test-006--three-page-text-diagram-table-bullets" / "diagrams"
    assert diag_dir.is_dir(), "Expected diagram directory to be created"
    extracted = sorted(diag_dir.glob("*.jpg"))
    assert len(extracted) > 0, "Expected at least one diagram image"
    assert "![Diagram" in content, "Expected diagram Markdown reference in output"

    reference = DATA / "images" / "test-006-diagram1.png"
    sim = _image_similarity(extracted[0], reference)
    assert sim >= 0.75, f"Extracted diagram similarity to reference is only {sim:.2%}"
