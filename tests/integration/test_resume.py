"""Integration tests for resumability via state.json."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

from pdf_extractor.cli import run
from pdf_extractor.render import render_pages, get_page_count
from pdf_extractor.state import StateManager

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _make_pdf(path: Path, pages: int = 3) -> None:
    doc = fitz.open()
    for i in range(pages):
        pg = doc.new_page(width=595, height=842)
        pg.insert_text((72, 100), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()


def _start_mock() -> HTTPServer:
    """Start a mock Ollama server on an OS-allocated port (avoids collisions).

    Read the chosen port from ``server.server_address[1]``.
    """
    body = json.dumps({"text": "resumed page", "diagrams": []})

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
            payload = json.dumps({"response": body}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_resume_phase1_partial(tmp_path):
    """Pages already image_done are skipped on second pass."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=3)
    out = tmp_path / "doc"
    out.mkdir()
    pages_dir = out / "pages"

    sm = StateManager(out)
    st = sm.load_or_init(pdf, 3)

    # pre-render page 1 manually
    results = render_pages(pdf, pages_dir, 3, [1], max_workers=1)
    sm.update_page(st, 1, image_done=True)

    # pending for phase 1 should exclude page 1
    pending = [
        i for i in range(1, 4)
        if not st.pages[str(i)].image_done and not st.pages[str(i)].image_failed
    ]
    assert 1 not in pending
    assert 2 in pending
    assert 3 in pending


def test_resume_phase2_partial(tmp_path, monkeypatch):
    """Pages already ocr_done are skipped; no re-rendering happens."""
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=2)
    out = tmp_path / "doc"
    out.mkdir()

    sm = StateManager(out)
    st = sm.load_or_init(pdf, 2)
    # simulate phase 1 done, page 1 ocr done
    sm.update_page(st, 1, image_done=True, ocr_done=True)
    sm.update_page(st, 2, image_done=True)

    # pending for phase 2 excludes page 1
    ocr_pending = [
        i for i in range(1, 3)
        if st.pages[str(i)].image_done
        and not st.pages[str(i)].ocr_done
        and not st.pages[str(i)].ocr_failed
    ]
    assert 1 not in ocr_pending
    assert 2 in ocr_pending


def test_resume_combined_done_exits_0(tmp_path, monkeypatch):
    """combined_done=True → run() returns 0 without any processing."""
    monkeypatch.chdir(tmp_path)
    server = _start_mock()
    port = server.server_address[1]
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)

    try:
        # first run to completion
        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            assert run() == 0

        # verify state is complete
        out = tmp_path / "doc"
        sm = StateManager(out)
        st = sm.load_or_init(pdf, 1)
        assert sm.status(st) == "complete"

        # second run — exits 0 immediately
        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            assert run() == 0
    finally:
        server.shutdown()


def test_state_mismatch_page_count_exits_8(tmp_path, monkeypatch):
    """A state.json whose page count no longer matches the PDF → run() returns 8."""
    monkeypatch.chdir(tmp_path)
    server = _start_mock()
    port = server.server_address[1]
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)

    try:
        # first run to completion writes state.json for a 1-page document
        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            assert run() == 0

        # the PDF at the same path is replaced by a 2-page document
        _make_pdf(pdf, pages=2)

        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            assert run() == 8
    finally:
        server.shutdown()


def test_resume_phase3_reruns_if_not_combined(tmp_path, monkeypatch):
    """combined_done=False but all OCR done → Phase 3 re-runs, output regenerated."""
    monkeypatch.chdir(tmp_path)
    server = _start_mock()
    port = server.server_address[1]
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, pages=1)

    try:
        # full run
        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            assert run() == 0

        # delete combined output but leave combined_done=False
        out_md = tmp_path / "doc.md"
        out = tmp_path / "doc"
        sm = StateManager(out)
        st = sm.load_or_init(pdf, 1)
        # reset combined_done in state
        st.combined_done = False
        sm._write(st)  # noqa: SLF001
        out_md.unlink()
        assert not out_md.exists()

        # re-run — Phase 3 should regenerate the file
        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            assert run() == 0
        assert out_md.is_file()
    finally:
        server.shutdown()
