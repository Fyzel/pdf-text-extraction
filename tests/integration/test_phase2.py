"""Integration tests for Phase 2 — OCR with mocked Ollama."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from pdf_extractor.config import OllamaInstance
from pdf_extractor.ocr import run_phase2
from pdf_extractor.render import get_page_count, render_pages
from pdf_extractor.state import AppState, StateManager
from tests.helpers import apply_render_results
from tests.ollama_mock import start_ollama_mock

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _render_all(pdf: Path, out: Path) -> tuple[int, StateManager, AppState]:
    """Render every page of ``pdf`` into ``out`` and return the resulting state.

    :param pdf: Source PDF to render. Required.
    :type pdf: pathlib.Path
    :param out: Per-document working directory. Required.
    :type out: pathlib.Path
    :return: ``(page_count, state_manager, state)`` after Phase 1.
    :rtype: tuple[int, pdf_extractor.state.StateManager, pdf_extractor.state.AppState]
    """
    count = get_page_count(pdf)
    pages_dir = out / "pages"
    sm = StateManager(out)
    st = sm.load_or_init(pdf, count)
    results = render_pages(pdf, pages_dir, count, list(range(1, count + 1)), max_workers=1)
    apply_render_results(sm, st, results)
    return count, sm, st


def test_phase2_simple_pdf(tmp_path):
    """A single-page PDF is OCR'd and its page markdown is written.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    server = start_ollama_mock(19510, {"text": "OCR text content", "diagrams": []})
    try:
        out = tmp_path / "out"
        out.mkdir()
        count, sm, st = _render_all(FIXTURES / "simple.pdf", out)
        pending = [i for i in range(1, count + 1) if st.pages[str(i)].image_done]
        inst = OllamaInstance("http://127.0.0.1:19510", "m")
        run_phase2(out, count, pending, [inst], st, sm)
        assert st.pages["1"].ocr_done
        assert (out / "pages" / "page_1.md").is_file()
    finally:
        server.shutdown()


def test_phase2_multipage_pdf(tmp_path):
    """All ten pages of the multipage fixture are OCR'd.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    server = start_ollama_mock(19511, {"text": "OCR text content", "diagrams": []})
    try:
        out = tmp_path / "out"
        out.mkdir()
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
    """A diagram bbox in every response yields one cropped image per page.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    bbox = {"x": 10, "y": 10, "width": 100, "height": 80}
    server = start_ollama_mock(19512, {"text": "diagram page", "diagrams": [bbox]})
    try:
        out = tmp_path / "out"
        out.mkdir()
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
    """A table-only response produces no diagram files and keeps the table text.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    body = {"text": "| A | B |\n|---|---|\n| 1 | 2 |", "diagrams": []}
    server = start_ollama_mock(19513, body)
    try:
        out = tmp_path / "out"
        out.mkdir()
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
    """When the only instance's first reply is unusable, the page is marked failed.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    call_count = [0]

    def resp(_path: str) -> dict | str:
        """Return a bad reply on the first call, then a valid one.

        :param _path: Request path, ignored. Required.
        :type _path: str
        :return: ``"bad json..."`` on the first call, else a valid OCR dict.
        :rtype: dict | str
        """
        call_count[0] += 1
        if call_count[0] == 1:
            return "bad json that will fail"
        return {"text": "ok", "diagrams": []}

    class Handler(BaseHTTPRequestHandler):
        """Handler whose reply varies per call via the ``resp`` closure."""

        def do_POST(self) -> None:
            """Reply with the next ``resp`` value wrapped as an Ollama response.

            :return: ``None``.
            :rtype: None
            """
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            r = resp(self.path)
            body = json.dumps(
                {"response": r if isinstance(r, str) else json.dumps(r)}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            """Silence per-request logging.

            :param args: Base-class log arguments, ignored. Optional.
            :type args: object
            :return: ``None``.
            :rtype: None
            """

    server = HTTPServer(("127.0.0.1", 19514), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "out"
        out.mkdir()
        count, sm, st = _render_all(FIXTURES / "simple.pdf", out)
        pending = [i for i in range(1, count + 1) if st.pages[str(i)].image_done]
        inst = OllamaInstance("http://127.0.0.1:19514", "m")
        run_phase2(out, count, pending, [inst], st, sm)
        # only one instance, first call fails → page marked ocr_failed
        assert st.pages["1"].ocr_failed
    finally:
        server.shutdown()
