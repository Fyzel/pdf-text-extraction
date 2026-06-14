"""Integration tests for Phase 2 — OCR with mocked Ollama."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from pdf_extractor.config import OllamaInstance
from pdf_extractor.ocr import run_phase2
from pdf_extractor.render import render_pages, get_page_count
from pdf_extractor.state import StateManager

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _start_mock(port: int, response_fn=None):
    def _default_response(path):
        return {"text": "OCR text content", "diagrams": []}

    fn = response_fn or _default_response

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = json.dumps({"response": json.dumps(fn(self.path))}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _render_all(pdf: Path, out: Path) -> tuple[int, StateManager, object]:
    count = get_page_count(pdf)
    pages_dir = out / "pages"
    sm = StateManager(out)
    st = sm.load_or_init(pdf, count)
    results = render_pages(pdf, pages_dir, count, list(range(1, count + 1)), max_workers=1)
    for pn, success, _ in results:
        if success:
            sm.update_page(st, pn, image_done=True)
        else:
            sm.update_page(st, pn, image_failed=True)
    return count, sm, st


def test_phase2_simple_pdf(tmp_path):
    server = _start_mock(19510)
    try:
        out = tmp_path / "out"; out.mkdir()
        count, sm, st = _render_all(FIXTURES / "simple.pdf", out)
        pending = [i for i in range(1, count + 1) if st.pages[str(i)].image_done]
        inst = OllamaInstance("http://127.0.0.1:19510", "m")
        run_phase2(out, count, pending, [inst], st, sm)
        assert st.pages["1"].ocr_done
        assert (out / "pages" / "page_1.md").is_file()
    finally:
        server.shutdown()


def test_phase2_multipage_pdf(tmp_path):
    server = _start_mock(19511)
    try:
        out = tmp_path / "out"; out.mkdir()
        count, sm, st = _render_all(FIXTURES / "multipage.pdf", out)
        pending = [i for i in range(1, count + 1) if st.pages[str(i)].image_done]
        assert len(pending) == 10
        inst = OllamaInstance("http://127.0.0.1:19511", "m")
        run_phase2(out, count, pending, [inst], st, sm)
        done = sum(1 for i in range(1, count + 1) if st.pages[str(i)].ocr_done)
        assert done == 10
        mds = list((out / "pages").glob("*.md"))
        assert len(mds) == 10
    finally:
        server.shutdown()


def test_phase2_diagrams_pdf_crops_images(tmp_path):
    bbox = {"x": 10, "y": 10, "width": 100, "height": 80}

    def resp(_path):
        return {"text": "diagram page", "diagrams": [bbox]}

    server = _start_mock(19512, resp)
    try:
        out = tmp_path / "out"; out.mkdir()
        count, sm, st = _render_all(FIXTURES / "diagrams.pdf", out)
        pending = [i for i in range(1, count + 1) if st.pages[str(i)].image_done]
        inst = OllamaInstance("http://127.0.0.1:19512", "m")
        run_phase2(out, count, pending, [inst], st, sm)
        diag_dir = out / "diagrams"
        assert diag_dir.is_dir()
        assert len(list(diag_dir.glob("*.jpg"))) == count
        width = len(str(count))
        for i in range(1, count + 1):
            stem = f"page_{i:0{width}d}"
            md = (out / "pages" / f"{stem}.md").read_text(encoding="utf-8")
            assert "![Diagram 1]" in md
    finally:
        server.shutdown()


def test_phase2_tables_pdf_no_diagram_files(tmp_path):
    def resp(_path):
        return {"text": "| A | B |\n|---|---|\n| 1 | 2 |", "diagrams": []}

    server = _start_mock(19513, resp)
    try:
        out = tmp_path / "out"; out.mkdir()
        count, sm, st = _render_all(FIXTURES / "tables.pdf", out)
        pending = [i for i in range(1, count + 1) if st.pages[str(i)].image_done]
        inst = OllamaInstance("http://127.0.0.1:19513", "m")
        run_phase2(out, count, pending, [inst], st, sm)
        assert not (out / "diagrams").exists()
        for i in range(1, count + 1):
            md = (out / "pages" / f"page_{i}.md").read_text(encoding="utf-8")
            assert "| A | B |" in md
    finally:
        server.shutdown()


def test_phase2_one_instance_failure_marked(tmp_path):
    call_count = [0]

    def resp(_path):
        call_count[0] += 1
        if call_count[0] == 1:
            return "bad json that will fail"
        return {"text": "ok", "diagrams": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            r = resp(self.path)
            body = json.dumps({"response": r if isinstance(r, str) else json.dumps(r)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 19514), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "out"; out.mkdir()
        count, sm, st = _render_all(FIXTURES / "simple.pdf", out)
        pending = [i for i in range(1, count + 1) if st.pages[str(i)].image_done]
        inst = OllamaInstance("http://127.0.0.1:19514", "m")
        run_phase2(out, count, pending, [inst], st, sm)
        # only one instance, first call fails → page marked ocr_failed
        assert st.pages["1"].ocr_failed
    finally:
        server.shutdown()
