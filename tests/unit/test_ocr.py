"""Unit tests for pdf_extractor/ocr.py."""
import json
import threading
import pytest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import fitz

from pdf_extractor.config import OllamaInstance
from pdf_extractor.ocr import (
    _BLANK_PAGE_MARKER,
    _crop_diagram,
    _crop_pdf_region,
    _embedded_image_rects,
    _is_blank_page,
    _ocr_page_with_retry,
    _page_white_ratio,
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


def _make_pdf_with_image(
    path: Path,
    rect: fitz.Rect = fitz.Rect(100, 120, 300, 260),
    page_count: int = 1,
    image_page: int = 1,
) -> None:
    """Write a PDF whose ``image_page`` embeds one raster image at ``rect``."""
    img = fitz.open()
    ip = img.new_page(width=200, height=140)
    ip.draw_rect(ip.rect, color=(1, 0, 0), fill=(1, 0, 0))
    pix = ip.get_pixmap()
    img.close()

    doc = fitz.open()
    for n in range(1, page_count + 1):
        pg = doc.new_page(width=612, height=792)
        pg.insert_text((72, 72), f"page {n} text")
        if n == image_page:
            pg.insert_image(rect, pixmap=pix)
    doc.save(str(path))
    doc.close()


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

# ---------------------------------------------------------------------------
# _embedded_image_rects / _crop_pdf_region
# ---------------------------------------------------------------------------

def test_embedded_image_rects_returns_rect(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf_with_image(pdf, rect=fitz.Rect(100, 120, 300, 260))
    rects = _embedded_image_rects(pdf, 1)
    assert len(rects) == 1
    r = rects[0]
    assert (round(r.x0), round(r.y0), round(r.x1), round(r.y1)) == (100, 120, 300, 260)


def test_embedded_image_rects_empty_when_no_image(tmp_path):
    pdf = tmp_path / "doc.pdf"
    _make_pdf_with_image(pdf, page_count=2, image_page=1)
    assert _embedded_image_rects(pdf, 2) == []


def test_crop_pdf_region_matches_rect_at_dpi_scale(tmp_path):
    pdf = tmp_path / "doc.pdf"
    rect = fitz.Rect(100, 120, 300, 260)
    _make_pdf_with_image(pdf, rect=rect)
    out = tmp_path / "crop.jpg"
    _crop_pdf_region(pdf, 1, rect, out)
    assert out.is_file()
    pix = fitz.Pixmap(str(out))
    # render.py renders at _DPI_SCALE = 2.0
    assert pix.width == round(rect.width * 2)
    assert pix.height == round(rect.height * 2)


def test_ocr_prefers_pdf_rects_over_model_bbox(tmp_path):
    # Model reports a wildly greedy bbox; the crop must follow the exact PDF rect.
    body = {"text": "Fig", "diagrams": [{"x": 0, "y": 0, "width": 9999, "height": 9999}]}
    server = _start_mock_server(19510, body)
    try:
        inst = OllamaInstance("http://127.0.0.1:19510", "m")
        pdf = tmp_path / "doc.pdf"
        rect = fitz.Rect(100, 120, 300, 260)
        _make_pdf_with_image(pdf, rect=rect)
        pages = tmp_path / "pages"; pages.mkdir()
        _make_jpeg(pages / "page_1.jpg")
        diags = tmp_path / "diagrams"
        _, ok, _, dcnt = _ocr_page_with_retry(
            1, [inst], pages, diags, 1, pdf_path=pdf
        )
        assert ok
        assert dcnt == 1
        crop = diags / "page_1_diagram_1.jpg"
        assert crop.is_file()
        pix = fitz.Pixmap(str(crop))
        assert pix.width == round(rect.width * 2)
        assert pix.height == round(rect.height * 2)
    finally:
        server.shutdown()


def test_ocr_falls_back_to_model_bbox_for_vector_figure(tmp_path):
    # PDF page has no embedded raster; crop must use the model bbox from the JPEG.
    body = {"text": "Fig", "diagrams": [{"x": 10, "y": 10, "width": 100, "height": 80}]}
    server = _start_mock_server(19511, body)
    try:
        inst = OllamaInstance("http://127.0.0.1:19511", "m")
        pdf = tmp_path / "doc.pdf"
        _make_pdf_with_image(pdf, page_count=2, image_page=1)  # page 2 has no image
        pages = tmp_path / "pages"; pages.mkdir()
        _make_jpeg(pages / "page_2.jpg")
        diags = tmp_path / "diagrams"
        _, ok, _, dcnt = _ocr_page_with_retry(
            2, [inst], pages, diags, 2, pdf_path=pdf
        )
        assert ok
        assert dcnt == 1
        assert (diags / "page_2_diagram_1.jpg").is_file()
    finally:
        server.shutdown()


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


def test_ocr_timeout_passed_through(tmp_path):
    received_timeout: list[int] = []

    class TimeoutCapture(BaseHTTPRequestHandler):
        def do_POST(self):
            # We can't directly capture the timeout from the server side,
            # but we verify the call succeeds when timeout is explicitly set.
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = json.dumps({"response": json.dumps({"text": "ok", "diagrams": []})}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 19508), TimeoutCapture)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        inst = OllamaInstance("http://127.0.0.1:19508", "m")
        pages = tmp_path / "pages"; pages.mkdir()
        _make_jpeg(pages / "page_1.jpg")
        pn, ok, err, _ = _ocr_page_with_retry(1, [inst], pages, tmp_path / "d", 1, ocr_timeout=30)
        assert ok
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# Blank-page detection (issue #49)
# ---------------------------------------------------------------------------

def _make_white_jpeg(path: Path, w: int = 400, h: int = 300) -> None:
    """Write a JPEG of a completely empty (white) page."""
    doc = fitz.open()
    doc.new_page(width=w, height=h)
    pix = doc[0].get_pixmap(matrix=fitz.Matrix(2, 2))
    pix.save(str(path))
    doc.close()


def _make_blank_pdf(path: Path, pages: int = 1) -> None:
    """Write a PDF with empty pages (no text, no drawings, no images)."""
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page(width=612, height=792)
    doc.save(str(path))
    doc.close()


def test_page_white_ratio_all_white(tmp_path):
    jpeg = tmp_path / "white.jpg"
    _make_white_jpeg(jpeg)
    assert _page_white_ratio(jpeg) >= 0.999


def test_page_white_ratio_with_content(tmp_path):
    jpeg = tmp_path / "content.jpg"
    _make_jpeg(jpeg)  # has text and a black rectangle
    assert _page_white_ratio(jpeg) < 0.999


def test_is_blank_page_white_no_pdf(tmp_path):
    jpeg = tmp_path / "white.jpg"
    _make_white_jpeg(jpeg)
    assert _is_blank_page(None, 1, jpeg) is True


def test_is_blank_page_content_no_pdf(tmp_path):
    jpeg = tmp_path / "content.jpg"
    _make_jpeg(jpeg)
    assert _is_blank_page(None, 1, jpeg) is False


def test_is_blank_page_blank_pdf(tmp_path):
    pdf = tmp_path / "blank.pdf"
    _make_blank_pdf(pdf)
    jpeg = tmp_path / "white.jpg"
    _make_white_jpeg(jpeg)
    assert _is_blank_page(pdf, 1, jpeg) is True


def test_is_blank_page_text_pdf_not_blank(tmp_path):
    pdf = tmp_path / "text.pdf"
    _make_pdf_with_image(pdf)  # page has "page 1 text"
    jpeg = tmp_path / "white.jpg"
    _make_white_jpeg(jpeg)  # even a white render is not blank when the PDF has text
    assert _is_blank_page(pdf, 1, jpeg) is False


def test_page_white_ratio_missing_file_returns_zero(tmp_path):
    # Unreadable image must not raise — returns 0.0 so the page is treated as non-blank.
    assert _page_white_ratio(tmp_path / "does_not_exist.jpg") == 0.0


def test_is_blank_page_unreadable_pdf_falls_back_to_whiteness(tmp_path):
    # A corrupt PDF must not abort the precheck; it falls through to the pixel check.
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"not a real pdf \x00\x01")
    jpeg = tmp_path / "white.jpg"
    _make_white_jpeg(jpeg)
    assert _is_blank_page(bad_pdf, 1, jpeg) is True


def test_is_blank_page_out_of_range_page_falls_back(tmp_path):
    pdf = tmp_path / "blank.pdf"
    _make_blank_pdf(pdf, pages=1)
    jpeg = tmp_path / "white.jpg"
    _make_white_jpeg(jpeg)
    # page_num beyond the document must not raise.
    assert _is_blank_page(pdf, 99, jpeg) is True


def test_ocr_skips_blank_page(tmp_path):
    # Point at a port with no server: if OCR were attempted it would fail.
    # A successful return therefore proves the Ollama call was skipped.
    inst = OllamaInstance("http://127.0.0.1:19599", "m")
    pdf = tmp_path / "blank.pdf"
    _make_blank_pdf(pdf)
    pages = tmp_path / "pages"; pages.mkdir()
    _make_white_jpeg(pages / "page_1.jpg")
    pn, ok, err, dcnt = _ocr_page_with_retry(
        1, [inst], pages, tmp_path / "d", 1, pdf_path=pdf
    )
    assert ok
    assert err == ""
    assert dcnt == 0
    assert (pages / "page_1.md").read_text(encoding="utf-8") == _BLANK_PAGE_MARKER


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
