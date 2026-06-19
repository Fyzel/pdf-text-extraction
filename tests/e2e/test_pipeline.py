"""End-to-end pipeline tests with mocked Ollama."""
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

from pdf_extractor.cli import run
from tests.helpers import markdown_table_blocks
from tests.ollama_mock import start_ollama_mock

FIXTURES = Path(__file__).parent.parent / "fixtures"
DATA = Path(__file__).parent.parent / "data"

_DEFAULT_BODY = {"text": "page text", "diagrams": []}


def _copy_fixture(tmp_path: Path, name: str) -> Path:
    """Copy a fixture PDF into ``tmp_path`` so output lands there, not in fixtures/.

    :param tmp_path: Destination directory. Required.
    :type tmp_path: pathlib.Path
    :param name: Fixture filename to copy. Required.
    :type name: str
    :return: Path to the copied PDF.
    :rtype: pathlib.Path
    """
    src = FIXTURES / name
    dst = tmp_path / name
    shutil.copy2(src, dst)
    return dst


def _copy_data(tmp_path: Path, name: str) -> Path:
    """Copy a data PDF into ``tmp_path`` so output lands there, not in data/.

    :param tmp_path: Destination directory. Required.
    :type tmp_path: pathlib.Path
    :param name: Data filename to copy. Required.
    :type name: str
    :return: Path to the copied PDF.
    :rtype: pathlib.Path
    """
    src = DATA / name
    dst = tmp_path / name
    shutil.copy2(src, dst)
    return dst


def _run(tmp_path: Path, pdf: Path, port: int) -> int:
    """Write an ``ollama.json`` pointing at ``port`` and run the pipeline on ``pdf``.

    :param tmp_path: Working directory holding ``ollama.json``. Required.
    :type tmp_path: pathlib.Path
    :param pdf: PDF to process. Required.
    :type pdf: pathlib.Path
    :param port: Loopback port of the running mock Ollama server. Required.
    :type port: int
    :return: The pipeline exit code from :func:`pdf_extractor.cli.run`.
    :rtype: int
    """
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    with patch.object(sys, "argv", ["main.py", str(pdf)]):
        return run()


# ---------------------------------------------------------------------------
# simple.pdf
# ---------------------------------------------------------------------------

def test_e2e_simple(tmp_path, monkeypatch):
    """A single-page PDF runs end to end and writes a combined file.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    server = start_ollama_mock(0, _DEFAULT_BODY)
    try:
        pdf = _copy_fixture(tmp_path, "simple.pdf")
        code = _run(tmp_path, pdf, server.server_address[1])
        assert code == 0
        out_md = tmp_path / "simple.md"
        assert out_md.is_file()
        content = out_md.read_text(encoding="utf-8")
        assert "--- PAGE 1 ---" in content
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# multipage.pdf — 10 sections in order
# ---------------------------------------------------------------------------

def test_e2e_multipage(tmp_path, monkeypatch):
    """A ten-page PDF produces ten page sections in order.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    server = start_ollama_mock(0, _DEFAULT_BODY)
    try:
        pdf = _copy_fixture(tmp_path, "multipage.pdf")
        code = _run(tmp_path, pdf, server.server_address[1])
        assert code == 0
        content = (tmp_path / "multipage.md").read_text(encoding="utf-8")
        assert content.count("--- PAGE") == 10
        positions = [content.index(f"--- PAGE {i} ---") for i in range(1, 11)]
        assert positions == sorted(positions)
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# diagrams.pdf — image refs + diagram files
# ---------------------------------------------------------------------------

def test_e2e_diagrams(tmp_path, monkeypatch):
    """A diagrams PDF yields diagram references and cropped image files.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    bbox = {"x": 10, "y": 10, "width": 100, "height": 80}
    server = start_ollama_mock(0, {"text": "diagram page", "diagrams": [bbox]})
    try:
        pdf = _copy_fixture(tmp_path, "diagrams.pdf")
        code = _run(tmp_path, pdf, server.server_address[1])
        assert code == 0
        content = (tmp_path / "diagrams.md").read_text(encoding="utf-8")
        assert "![Diagram 1]" in content
        diag_dir = tmp_path / "diagrams" / "diagrams"
        assert diag_dir.is_dir()
        assert len(list(diag_dir.glob("*.jpg"))) > 0
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# mixed.pdf — diagram pages have refs, text-only pages do not
# ---------------------------------------------------------------------------

def test_e2e_mixed(tmp_path, monkeypatch):
    """A mixed PDF alternates between diagram and text-only page responses.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    call_num = [0]

    def ocr_fn() -> dict:
        """Return a diagram body on even calls, a text-only body on odd calls.

        :return: The OCR body dict for the current call.
        :rtype: dict
        """
        call_num[0] += 1
        if call_num[0] % 2 == 0:
            return {
                "text": "diagram page",
                "diagrams": [{"x": 10, "y": 10, "width": 50, "height": 50}],
            }
        return {"text": "text only page", "diagrams": []}

    server = start_ollama_mock(0, ocr_fn)
    try:
        pdf = _copy_fixture(tmp_path, "mixed.pdf")
        code = _run(tmp_path, pdf, server.server_address[1])
        assert code == 0
        content = (tmp_path / "mixed.md").read_text(encoding="utf-8")
        assert content.count("--- PAGE") == 5
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# tables.pdf — markdown tables in output, no diagram directory
# ---------------------------------------------------------------------------

def test_e2e_tables(tmp_path, monkeypatch):
    """A tables PDF replaces the model table with the PDF-extracted grid.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    body = {"text": "| Col1 | Col2 |\n|------|------|\n| A    | B    |", "diagrams": []}
    server = start_ollama_mock(0, body)
    try:
        pdf = _copy_fixture(tmp_path, "tables.pdf")
        code = _run(tmp_path, pdf, server.server_address[1])
        assert code == 0
        content = (tmp_path / "tables.md").read_text(encoding="utf-8")
        # Tables are sourced from the PDF, not the model text (issue #44): the
        # model's table block is replaced with the grid extracted from the page.
        assert "R0C0" in content, "Expected PDF-extracted table content"
        assert "| Col1 | Col2 |" not in content, "Model table block should be replaced"
        # Every Markdown table block has a consistent column count.
        for block in markdown_table_blocks(content):
            counts = {line.count("|") for line in block}
            assert len(counts) == 1, f"Inconsistent columns: {block}"
        diag_dir = tmp_path / "tables" / "diagrams"
        assert not diag_dir.exists()
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Re-run on completed PDF → exit 0 immediately
# ---------------------------------------------------------------------------

def test_e2e_rerun_exits_0(tmp_path, monkeypatch):
    """Re-running a completed PDF returns ``0`` again.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    server = start_ollama_mock(0, _DEFAULT_BODY)
    try:
        pdf = _copy_fixture(tmp_path, "simple.pdf")
        port = server.server_address[1]
        assert _run(tmp_path, pdf, port) == 0
        assert _run(tmp_path, pdf, port) == 0
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# --rerun-pages: archives prior artifacts, reprocesses, reassembles → exit 0
# ---------------------------------------------------------------------------

def test_e2e_rerun_pages_archives_and_reprocesses(tmp_path, monkeypatch):
    """``--rerun-pages`` archives prior artifacts and rebuilds the output.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    server = start_ollama_mock(0, _DEFAULT_BODY)
    try:
        pdf = _copy_fixture(tmp_path, "simple.pdf")
        port = server.server_address[1]
        cfg = {"instances": [{"url": f"http://127.0.0.1:{port}", "model": "qwen2.5vl:7b"}]}
        (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")

        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            assert run() == 0
        out_md = tmp_path / "simple.md"
        assert out_md.is_file()

        with patch.object(sys, "argv", ["main.py", str(pdf), "--rerun-pages", "1"]):
            assert run() == 0

        # Prior artifacts archived under _archive/v1, output rebuilt.
        archive = tmp_path / "simple" / "_archive" / "v1"
        assert archive.is_dir()
        assert (archive / "simple.md").is_file()
        assert out_md.is_file()
    finally:
        server.shutdown()


def test_e2e_rerun_pages_no_state_exit_1(tmp_path, monkeypatch):
    """``--rerun-pages`` without a prior run exits ``1``.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    server = start_ollama_mock(0, _DEFAULT_BODY)
    try:
        pdf = _copy_fixture(tmp_path, "simple.pdf")
        port = server.server_address[1]
        cfg = {"instances": [{"url": f"http://127.0.0.1:{port}", "model": "qwen2.5vl:7b"}]}
        (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
        with patch.object(sys, "argv", ["main.py", str(pdf), "--rerun-pages", "1"]):
            assert run() == 1
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# corrupt.pdf → exit 5 (or 3)
# ---------------------------------------------------------------------------

def test_e2e_corrupt_pdf(tmp_path, monkeypatch):
    """A corrupt PDF exits ``3`` (unreadable) or ``5`` (all pages fail).

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    server = start_ollama_mock(0, _DEFAULT_BODY)
    try:
        pdf = _copy_fixture(tmp_path, "corrupt.pdf")
        code = _run(tmp_path, pdf, server.server_address[1])
        assert code in (3, 5)
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Missing argument → exit 1
# ---------------------------------------------------------------------------

def test_e2e_missing_argument():
    """No PDF argument exits ``1``.

    :return: ``None``.
    :rtype: None
    """
    with patch.object(sys, "argv", ["main.py"]):
        assert run() == 1


# ---------------------------------------------------------------------------
# Non-existent path → exit 2
# ---------------------------------------------------------------------------

def test_e2e_nonexistent_path(tmp_path):
    """A non-existent PDF path exits ``2``.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    with patch.object(sys, "argv", ["main.py", str(tmp_path / "no.pdf")]):
        assert run() == 2


# ---------------------------------------------------------------------------
# No Ollama reachable → exit 4
# ---------------------------------------------------------------------------

def test_e2e_no_ollama(tmp_path, monkeypatch):
    """An unreachable Ollama instance exits ``4``.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    cfg = {"instances": [{"url": "http://127.0.0.1:19598", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    with patch.object(sys, "argv", ["main.py", str(FIXTURES / "simple.pdf")]):
        assert run() == 4


# ---------------------------------------------------------------------------
# tests/data — real PDFs
# ---------------------------------------------------------------------------

def test_e2e_data_001_one_page_text(tmp_path, monkeypatch):
    """Data PDF 001 runs end to end with one page section.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    server = start_ollama_mock(0, _DEFAULT_BODY)
    try:
        pdf = _copy_data(tmp_path, "test-001--one-page-text.pdf")
        code = _run(tmp_path, pdf, server.server_address[1])
        assert code == 0
        content = (tmp_path / "test-001--one-page-text.md").read_text(encoding="utf-8")
        assert content.count("--- PAGE") == 1
        assert "--- PAGE 1 ---" in content
    finally:
        server.shutdown()


def test_e2e_data_002_two_page_text(tmp_path, monkeypatch):
    """Data PDF 002 runs end to end with two ordered page sections.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    server = start_ollama_mock(0, _DEFAULT_BODY)
    try:
        pdf = _copy_data(tmp_path, "test-002--two-page-text.pdf")
        code = _run(tmp_path, pdf, server.server_address[1])
        assert code == 0
        content = (tmp_path / "test-002--two-page-text.md").read_text(encoding="utf-8")
        assert content.count("--- PAGE") == 2
        assert content.index("--- PAGE 1 ---") < content.index("--- PAGE 2 ---")
    finally:
        server.shutdown()


def test_e2e_data_003_three_page_text_diagram(tmp_path, monkeypatch):
    """Data PDF 003 runs end to end with diagrams and three page sections.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    bbox = {"x": 10, "y": 10, "width": 100, "height": 80}
    server = start_ollama_mock(0, {"text": "diagram page", "diagrams": [bbox]})
    try:
        pdf = _copy_data(tmp_path, "test-003--three-page-text-diagram.pdf")
        code = _run(tmp_path, pdf, server.server_address[1])
        assert code == 0
        content = (tmp_path / "test-003--three-page-text-diagram.md").read_text(encoding="utf-8")
        assert content.count("--- PAGE") == 3
        assert "![Diagram 1]" in content
        diag_dir = tmp_path / "test-003--three-page-text-diagram" / "diagrams"
        assert diag_dir.is_dir()
        assert len(list(diag_dir.glob("*.jpg"))) > 0
    finally:
        server.shutdown()


def test_e2e_data_004_three_page_text_diagram(tmp_path, monkeypatch):
    """Data PDF 004 runs end to end with diagrams and three page sections.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    bbox = {"x": 10, "y": 10, "width": 100, "height": 80}
    server = start_ollama_mock(0, {"text": "diagram page", "diagrams": [bbox]})
    try:
        pdf = _copy_data(tmp_path, "test-004--three-page-text-diagram.pdf")
        code = _run(tmp_path, pdf, server.server_address[1])
        assert code == 0
        content = (tmp_path / "test-004--three-page-text-diagram.md").read_text(encoding="utf-8")
        assert content.count("--- PAGE") == 3
        assert "![Diagram 1]" in content
        diag_dir = tmp_path / "test-004--three-page-text-diagram" / "diagrams"
        assert diag_dir.is_dir()
        assert len(list(diag_dir.glob("*.jpg"))) > 0
    finally:
        server.shutdown()
