"""End-to-end pipeline tests with mocked Ollama."""
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from pdf_extractor.cli import run

FIXTURES = Path(__file__).parent.parent / "fixtures"
DATA = Path(__file__).parent.parent / "data"

_NEXT_PORT = 19600


def _alloc_port() -> int:
    global _NEXT_PORT
    p = _NEXT_PORT
    _NEXT_PORT += 1
    return p


def _start_mock(port: int, ocr_fn=None) -> HTTPServer:
    default_body = json.dumps({"text": "page text", "diagrams": []})

    def _default(_path):
        return default_body

    fn = ocr_fn or _default

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            resp = json.dumps({"models": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(resp))
            self.end_headers()
            self.wfile.write(resp)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            payload = json.dumps({"response": fn(self.path)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _copy_fixture(tmp_path: Path, name: str) -> Path:
    """Copy a fixture PDF into tmp_path so output lands there, not in fixtures/."""
    src = FIXTURES / name
    dst = tmp_path / name
    shutil.copy2(src, dst)
    return dst


def _markdown_table_blocks(content: str) -> list[list[str]]:
    """Return each contiguous run of pipe-prefixed lines as a list of lines."""
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


def _copy_data(tmp_path: Path, name: str) -> Path:
    """Copy a data PDF into tmp_path so output lands there, not in data/."""
    src = DATA / name
    dst = tmp_path / name
    shutil.copy2(src, dst)
    return dst


def _run(tmp_path: Path, pdf: Path, port: int, ocr_fn=None):
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    with patch.object(sys, "argv", ["main.py", str(pdf)]):
        return run()


# ---------------------------------------------------------------------------
# simple.pdf
# ---------------------------------------------------------------------------

def test_e2e_simple(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    server = _start_mock(port)
    try:
        pdf = _copy_fixture(tmp_path, "simple.pdf")
        code = _run(tmp_path, pdf, port)
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
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    server = _start_mock(port)
    try:
        pdf = _copy_fixture(tmp_path, "multipage.pdf")
        code = _run(tmp_path, pdf, port)
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
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    bbox = {"x": 10, "y": 10, "width": 100, "height": 80}

    def ocr_fn(_path):
        return json.dumps({"text": "diagram page", "diagrams": [bbox]})

    server = _start_mock(port, ocr_fn)
    try:
        pdf = _copy_fixture(tmp_path, "diagrams.pdf")
        code = _run(tmp_path, pdf, port)
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
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    call_num = [0]

    def ocr_fn(_path):
        call_num[0] += 1
        if call_num[0] % 2 == 0:
            return json.dumps({"text": "diagram page", "diagrams": [{"x": 10, "y": 10, "width": 50, "height": 50}]})
        return json.dumps({"text": "text only page", "diagrams": []})

    server = _start_mock(port, ocr_fn)
    try:
        pdf = _copy_fixture(tmp_path, "mixed.pdf")
        code = _run(tmp_path, pdf, port)
        assert code == 0
        content = (tmp_path / "mixed.md").read_text(encoding="utf-8")
        assert content.count("--- PAGE") == 5
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# tables.pdf — markdown tables in output, no diagram directory
# ---------------------------------------------------------------------------

def test_e2e_tables(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()

    def ocr_fn(_path):
        return json.dumps({"text": "| Col1 | Col2 |\n|------|------|\n| A    | B    |", "diagrams": []})

    server = _start_mock(port, ocr_fn)
    try:
        pdf = _copy_fixture(tmp_path, "tables.pdf")
        code = _run(tmp_path, pdf, port)
        assert code == 0
        content = (tmp_path / "tables.md").read_text(encoding="utf-8")
        # Tables are sourced from the PDF, not the model text (issue #44): the
        # model's table block is replaced with the grid extracted from the page.
        assert "R0C0" in content, "Expected PDF-extracted table content"
        assert "| Col1 | Col2 |" not in content, "Model table block should be replaced"
        # Every Markdown table block has a consistent column count.
        for block in _markdown_table_blocks(content):
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
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    server = _start_mock(port)
    try:
        pdf = _copy_fixture(tmp_path, "simple.pdf")
        code1 = _run(tmp_path, pdf, port)
        assert code1 == 0
        code2 = _run(tmp_path, pdf, port)
        assert code2 == 0
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# --rerun-pages: archives prior artifacts, reprocesses, reassembles → exit 0
# ---------------------------------------------------------------------------

def test_e2e_rerun_pages_archives_and_reprocesses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    server = _start_mock(port)
    try:
        pdf = _copy_fixture(tmp_path, "simple.pdf")
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
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    server = _start_mock(port)
    try:
        pdf = _copy_fixture(tmp_path, "simple.pdf")
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
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    server = _start_mock(port)
    try:
        pdf = _copy_fixture(tmp_path, "corrupt.pdf")
        code = _run(tmp_path, pdf, port)
        assert code in (3, 5)
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Missing argument → exit 1
# ---------------------------------------------------------------------------

def test_e2e_missing_argument():
    with patch.object(sys, "argv", ["main.py"]):
        assert run() == 1


# ---------------------------------------------------------------------------
# Non-existent path → exit 2
# ---------------------------------------------------------------------------

def test_e2e_nonexistent_path(tmp_path):
    with patch.object(sys, "argv", ["main.py", str(tmp_path / "no.pdf")]):
        assert run() == 2


# ---------------------------------------------------------------------------
# No Ollama reachable → exit 4
# ---------------------------------------------------------------------------

def test_e2e_no_ollama(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {"instances": [{"url": "http://127.0.0.1:19598", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    with patch.object(sys, "argv", ["main.py", str(FIXTURES / "simple.pdf")]):
        assert run() == 4


# ---------------------------------------------------------------------------
# tests/data — real PDFs
# ---------------------------------------------------------------------------

def test_e2e_data_001_one_page_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    server = _start_mock(port)
    try:
        pdf = _copy_data(tmp_path, "test-001--one-page-text.pdf")
        code = _run(tmp_path, pdf, port)
        assert code == 0
        content = (tmp_path / "test-001--one-page-text.md").read_text(encoding="utf-8")
        assert content.count("--- PAGE") == 1
        assert "--- PAGE 1 ---" in content
    finally:
        server.shutdown()


def test_e2e_data_002_two_page_text(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    server = _start_mock(port)
    try:
        pdf = _copy_data(tmp_path, "test-002--two-page-text.pdf")
        code = _run(tmp_path, pdf, port)
        assert code == 0
        content = (tmp_path / "test-002--two-page-text.md").read_text(encoding="utf-8")
        assert content.count("--- PAGE") == 2
        assert content.index("--- PAGE 1 ---") < content.index("--- PAGE 2 ---")
    finally:
        server.shutdown()


def test_e2e_data_003_three_page_text_diagram(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    bbox = {"x": 10, "y": 10, "width": 100, "height": 80}

    def ocr_fn(_path):
        return json.dumps({"text": "diagram page", "diagrams": [bbox]})

    server = _start_mock(port, ocr_fn)
    try:
        pdf = _copy_data(tmp_path, "test-003--three-page-text-diagram.pdf")
        code = _run(tmp_path, pdf, port)
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
    monkeypatch.chdir(tmp_path)
    port = _alloc_port()
    bbox = {"x": 10, "y": 10, "width": 100, "height": 80}

    def ocr_fn(_path):
        return json.dumps({"text": "diagram page", "diagrams": [bbox]})

    server = _start_mock(port, ocr_fn)
    try:
        pdf = _copy_data(tmp_path, "test-004--three-page-text-diagram.pdf")
        code = _run(tmp_path, pdf, port)
        assert code == 0
        content = (tmp_path / "test-004--three-page-text-diagram.md").read_text(encoding="utf-8")
        assert content.count("--- PAGE") == 3
        assert "![Diagram 1]" in content
        diag_dir = tmp_path / "test-004--three-page-text-diagram" / "diagrams"
        assert diag_dir.is_dir()
        assert len(list(diag_dir.glob("*.jpg"))) > 0
    finally:
        server.shutdown()
