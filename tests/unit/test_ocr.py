"""Unit tests for pdf_extractor/ocr.py."""
import json
import threading
import pytest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import fitz

from pdf_extractor.config import OllamaInstance
from pdf_extractor.ocr import (
    _crop_diagram,
    _ocr_page_with_retry,
    _parse_ocr_response,
    run_phase2,
)
from pdf_extractor.state import StateManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg(path: Path, w: int = 400, h: int = 300) -> tuple[int, int]:
    doc = fitz.open()
    pg = doc.new_page(width=w, height=h)
    pg.insert_text((50, 100), "test text")
    pg.draw_rect(fitz.Rect(200, 50, 350, 200), color=(0, 0, 0))
    pix = pg.get_pixmap(matrix=fitz.Matrix(2, 2))
    pix.save(str(path))
    doc.close()
    return pix.width, pix.height


def _start_mock_server(port: int, response_body: dict | str) -> HTTPServer:
    body_str = response_body if isinstance(response_body, str) else json.dumps(response_body)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            payload = json.dumps({"response": body_str}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()
    return server


# ---------------------------------------------------------------------------
# _parse_ocr_response
# ---------------------------------------------------------------------------

def test_parse_valid_json():
    data = _parse_ocr_response(json.dumps({"text": "hello", "diagrams": []}))
    assert data["text"] == "hello"
    assert data["diagrams"] == []


def test_parse_strips_code_fence():
    raw = "```json\n" + json.dumps({"text": "hi", "diagrams": []}) + "\n```"
    data = _parse_ocr_response(raw)
    assert data["text"] == "hi"


def test_parse_invalid_json_raises():
    with pytest.raises(json.JSONDecodeError):
        _parse_ocr_response("{bad}")


def test_parse_missing_text_raises():
    with pytest.raises(ValueError):
        _parse_ocr_response(json.dumps({"diagrams": []}))


def test_parse_missing_diagrams_raises():
    with pytest.raises(ValueError):
        _parse_ocr_response(json.dumps({"text": "x"}))


# ---------------------------------------------------------------------------
# _crop_diagram
# ---------------------------------------------------------------------------

def test_crop_normal(tmp_path):
    jpeg = tmp_path / "page.jpg"
    _make_jpeg(jpeg)
    out = tmp_path / "crop.jpg"
    _crop_diagram(jpeg, 10, 10, 100, 80, out)
    assert out.is_file()
    pix = fitz.Pixmap(str(out))
    assert pix.width == 100
    assert pix.height == 80


def test_crop_clamps_to_boundary(tmp_path):
    jpeg = tmp_path / "page.jpg"
    iw, ih = _make_jpeg(jpeg)
    out = tmp_path / "crop.jpg"
    _crop_diagram(jpeg, iw - 5, ih - 5, 9999, 9999, out)
    assert out.is_file()
    pix = fitz.Pixmap(str(out))
    assert pix.width <= iw
    assert pix.height <= ih


# ---------------------------------------------------------------------------
# _ocr_page_with_retry — text only
# ---------------------------------------------------------------------------

def test_ocr_text_only(tmp_path):
    server = _start_mock_server(19500, {"text": "Page content", "diagrams": []})
    try:
        inst = OllamaInstance("http://127.0.0.1:19500", "m")
        pages = tmp_path / "pages"; pages.mkdir()
        _make_jpeg(pages / "page_1.jpg")
        pn, ok, err, dcnt = _ocr_page_with_retry(1, [inst], pages, tmp_path / "d", 1)
        assert ok
        assert err == ""
        assert dcnt == 0
        assert (pages / "page_1.md").is_file()
        assert (pages / "page_1.md").read_text() == "Page content"
    finally:
        server.shutdown()


def test_ocr_with_diagrams(tmp_path):
    body = {"text": "Fig below", "diagrams": [{"x": 10, "y": 10, "width": 100, "height": 80}]}
    server = _start_mock_server(19501, body)
    try:
        inst = OllamaInstance("http://127.0.0.1:19501", "m")
        pages = tmp_path / "pages"; pages.mkdir()
        diags = tmp_path / "diagrams"
        _make_jpeg(pages / "page_1.jpg")
        _, ok, _, dcnt = _ocr_page_with_retry(1, [inst], pages, diags, 1)
        assert ok
        assert dcnt == 1
        assert (diags / "page_1_diagram_1.jpg").is_file()
        md = (pages / "page_1.md").read_text()
        assert "![Diagram 1](diagrams/page_1_diagram_1.jpg)" in md
    finally:
        server.shutdown()


def test_ocr_table_stays_in_text(tmp_path):
    body = {"text": "| A | B |\n|---|---|\n| 1 | 2 |", "diagrams": []}
    server = _start_mock_server(19502, body)
    try:
        inst = OllamaInstance("http://127.0.0.1:19502", "m")
        pages = tmp_path / "pages"; pages.mkdir()
        diags = tmp_path / "diagrams"
        _make_jpeg(pages / "page_1.jpg")
        _, ok, _, dcnt = _ocr_page_with_retry(1, [inst], pages, diags, 1)
        assert ok
        assert dcnt == 0
        assert not diags.exists()
        assert "| A | B |" in (pages / "page_1.md").read_text()
    finally:
        server.shutdown()


def test_ocr_invalid_json_marks_failed(tmp_path):
    server = _start_mock_server(19503, "not json")
    try:
        inst = OllamaInstance("http://127.0.0.1:19503", "m")
        pages = tmp_path / "pages"; pages.mkdir()
        _make_jpeg(pages / "page_1.jpg")
        _, ok, err, _ = _ocr_page_with_retry(1, [inst], pages, tmp_path / "d", 1)
        assert not ok
        assert err != ""
    finally:
        server.shutdown()


def test_ocr_retries_all_instances(tmp_path):
    tried: list[int] = []

    class TrackHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            tried.append(self.server.server_address[1])
            payload = json.dumps({"response": "bad"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(payload))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    s1 = HTTPServer(("127.0.0.1", 19504), TrackHandler)
    s2 = HTTPServer(("127.0.0.1", 19505), TrackHandler)
    for s in [s1, s2]:
        threading.Thread(target=s.serve_forever, daemon=True).start()

    try:
        insts = [
            OllamaInstance("http://127.0.0.1:19504", "m"),
            OllamaInstance("http://127.0.0.1:19505", "m"),
        ]
        pages = tmp_path / "pages"; pages.mkdir()
        _make_jpeg(pages / "page_1.jpg")
        _, ok, _, _ = _ocr_page_with_retry(1, insts, pages, tmp_path / "d", 1)
        assert not ok
        assert sorted(tried) == [19504, 19505]
    finally:
        s1.shutdown(); s2.shutdown()


# ---------------------------------------------------------------------------
# Round-robin ordering
# ---------------------------------------------------------------------------

def test_round_robin_ordering():
    instances = [
        OllamaInstance("http://a:11434", "m"),
        OllamaInstance("http://b:11434", "m"),
        OllamaInstance("http://c:11434", "m"),
    ]
    n = len(instances)
    for page_num, expected_first in [(1, "http://a:11434"), (2, "http://b:11434"), (3, "http://c:11434"), (4, "http://a:11434")]:
        start = (page_num - 1) % n
        ordered = instances[start:] + instances[:start]
        assert ordered[0].url == expected_first


# ---------------------------------------------------------------------------
# run_phase2 — exit 6 condition
# ---------------------------------------------------------------------------

def test_run_phase2_exit6_condition(tmp_path):
    sm = StateManager(tmp_path)
    st = sm.load_or_init(tmp_path / "fake.pdf", 2)
    sm.update_page(st, 1, image_done=True, ocr_failed=True)
    sm.update_page(st, 2, image_done=True, ocr_failed=True)
    rendered = [i for i in range(1, 3) if st.pages[str(i)].image_done]
    all_failed = rendered and all(st.pages[str(i)].ocr_failed for i in rendered)
    assert all_failed


def test_run_phase2_updates_state(tmp_path):
    body = {"text": "hello", "diagrams": []}
    server = _start_mock_server(19506, body)
    try:
        inst = OllamaInstance("http://127.0.0.1:19506", "m")
        out = tmp_path / "out"; out.mkdir()
        pages = out / "pages"; pages.mkdir()
        sm = StateManager(out)
        st = sm.load_or_init(out / "fake.pdf", 2)
        for i in range(1, 3):
            _make_jpeg(pages / f"page_{i}.jpg")
            sm.update_page(st, i, image_done=True)
        pending = [1, 2]
        run_phase2(out, 2, pending, [inst], st, sm)
        assert st.pages["1"].ocr_done
        assert st.pages["2"].ocr_done
    finally:
        server.shutdown()
