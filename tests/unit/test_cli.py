"""Unit tests for pdf_extractor/cli.py — argument parsing and exit codes."""
import sys
import json
import threading
import pytest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import fitz

from pdf_extractor.cli import _parse_args, run
from pdf_extractor.render import _DPI_SCALE


def _make_pdf(path: Path, pages: int = 1) -> None:
    doc = fitz.open()
    for i in range(pages):
        pg = doc.new_page(width=595, height=842)
        pg.insert_text((72, 100), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()


def _start_ollama_mock(port: int, ocr_body: dict) -> HTTPServer:
    ocr_str = json.dumps(ocr_body)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # health probe
            body = json.dumps({"models": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            payload = json.dumps({"response": ocr_str}).encode()
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


# ---------------------------------------------------------------------------
# Exit 1 — missing argument
# ---------------------------------------------------------------------------

def test_exit1_no_argument():
    with patch.object(sys, "argv", ["main.py"]):
        assert run() == 1


# ---------------------------------------------------------------------------
# Argument parsing — _parse_args / --dpi-scale
# ---------------------------------------------------------------------------

def test_parse_args_pdf_only():
    pdf, dpi, err = _parse_args(["doc.pdf"])
    assert pdf == "doc.pdf"
    assert dpi == _DPI_SCALE
    assert err is None


def test_parse_args_dpi_scale_space():
    pdf, dpi, err = _parse_args(["doc.pdf", "--dpi-scale", "4.0"])
    assert pdf == "doc.pdf"
    assert dpi == 4.0
    assert err is None


def test_parse_args_dpi_scale_equals():
    pdf, dpi, err = _parse_args(["--dpi-scale=3.5", "doc.pdf"])
    assert pdf == "doc.pdf"
    assert dpi == 3.5
    assert err is None


def test_parse_args_missing_value():
    pdf, _, err = _parse_args(["doc.pdf", "--dpi-scale"])
    assert pdf is None
    assert err is not None


def test_parse_args_invalid_value():
    pdf, _, err = _parse_args(["doc.pdf", "--dpi-scale", "huge"])
    assert pdf is None
    assert err is not None


def test_parse_args_non_positive_value():
    pdf, _, err = _parse_args(["doc.pdf", "--dpi-scale", "0"])
    assert pdf is None
    assert err is not None


def test_parse_args_unexpected_extra():
    pdf, _, err = _parse_args(["doc.pdf", "extra.pdf"])
    assert pdf is None
    assert err is not None


def test_parse_args_no_pdf():
    pdf, _, err = _parse_args(["--dpi-scale", "2.0"])
    assert pdf is None
    assert err is None


def test_exit1_bad_dpi_scale():
    with patch.object(sys, "argv", ["main.py", "doc.pdf", "--dpi-scale", "nope"]):
        assert run() == 1


def test_help_exits_zero(capsys):
    with patch.object(sys, "argv", ["main.py", "--help"]):
        assert run() == 0
    out = capsys.readouterr().out
    assert "--dpi-scale" in out


# ---------------------------------------------------------------------------
# Exit 2 — file not found
# ---------------------------------------------------------------------------

def test_exit2_file_not_found(tmp_path):
    with patch.object(sys, "argv", ["main.py", str(tmp_path / "missing.pdf")]):
        assert run() == 2


# ---------------------------------------------------------------------------
# Exit 3 — file not readable
# ---------------------------------------------------------------------------

def test_exit3_not_a_file(tmp_path):
    # passing a directory satisfies exists() but not is_file()
    with patch.object(sys, "argv", ["main.py", str(tmp_path)]):
        assert run() == 3


# ---------------------------------------------------------------------------
# Exit 4 — no Ollama instances reachable
# ---------------------------------------------------------------------------

def test_exit4_no_ollama(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)
    cfg = {"instances": [{"url": "http://127.0.0.1:19599", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    with patch.object(sys, "argv", ["main.py", str(pdf)]):
        assert run() == 4


# ---------------------------------------------------------------------------
# Exit 5 — all pages fail rendering (corrupt PDF)
# ---------------------------------------------------------------------------

def test_exit5_corrupt_pdf(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    port = 19590
    server = _start_ollama_mock(port, {"text": "x", "diagrams": []})
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")

    corrupt = tmp_path / "bad.pdf"
    corrupt.write_bytes(b"not a real pdf \x00\x01")

    try:
        with patch.object(sys, "argv", ["main.py", str(corrupt)]):
            # corrupt PDF may exit 3 (can't read page count) or 5
            code = run()
            assert code in (3, 5)
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Exit 0 — already complete
# ---------------------------------------------------------------------------

def test_exit0_already_complete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    port = 19591
    server = _start_ollama_mock(port, {"text": "hello", "diagrams": []})
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")

    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf)

    try:
        # first run
        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            code = run()
        assert code == 0

        # second run — should return 0 immediately (combined_done)
        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            code = run()
        assert code == 0
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Full successful run — exit 0
# ---------------------------------------------------------------------------

def test_exit0_full_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    port = 19592
    server = _start_ollama_mock(port, {"text": "page text", "diagrams": []})
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")

    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=2)

    try:
        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            code = run()
        assert code == 0
        assert (tmp_path / "doc.md").is_file()
        content = (tmp_path / "doc.md").read_text(encoding="utf-8")
        assert "--- PAGE 1 ---" in content
        assert "--- PAGE 2 ---" in content
    finally:
        server.shutdown()
