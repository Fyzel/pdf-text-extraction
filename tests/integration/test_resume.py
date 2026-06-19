"""Integration tests for resumability via state.json."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from pdf_extractor.cli import run
from pdf_extractor.render import render_pages
from pdf_extractor.state import StateManager
from tests.helpers import make_text_pdf
from tests.ollama_mock import start_ollama_mock

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_resume_phase1_partial(tmp_path):
    """Pages already marked ``image_done`` are excluded from Phase 1 pending.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    make_text_pdf(pdf, pages=3)
    out = tmp_path / "doc"
    out.mkdir()
    pages_dir = out / "pages"

    sm = StateManager(out)
    st = sm.load_or_init(pdf, 3)

    # pre-render page 1 manually
    render_pages(pdf, pages_dir, 3, [1], max_workers=1)
    sm.update_page(st, 1, image_done=True)

    # pending for phase 1 should exclude page 1
    pending = [
        i for i in range(1, 4)
        if not st.pages[str(i)].image_done and not st.pages[str(i)].image_failed
    ]
    assert 1 not in pending
    assert 2 in pending
    assert 3 in pending


def test_resume_phase2_partial(tmp_path):
    """Pages already marked ``ocr_done`` are excluded from Phase 2 pending.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    make_text_pdf(pdf, pages=2)
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
    """A completed run exits ``0`` on re-invocation without reprocessing.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    server = start_ollama_mock(0, {"text": "resumed page", "diagrams": []})
    port = server.server_address[1]
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    make_text_pdf(pdf, pages=1)

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
    """A state.json whose page count no longer matches the PDF makes run() exit 8.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    server = start_ollama_mock(0, {"text": "resumed page", "diagrams": []})
    port = server.server_address[1]
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    make_text_pdf(pdf, pages=1)

    try:
        # first run to completion writes state.json for a 1-page document
        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            assert run() == 0

        # the PDF at the same path is replaced by a 2-page document
        make_text_pdf(pdf, pages=2)

        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            assert run() == 8
    finally:
        server.shutdown()


def test_resume_phase3_reruns_if_not_combined(tmp_path, monkeypatch):
    """With OCR done but ``combined_done`` false, Phase 3 regenerates the output.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    server = start_ollama_mock(0, {"text": "resumed page", "diagrams": []})
    port = server.server_address[1]
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    make_text_pdf(pdf, pages=1)

    try:
        # full run
        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            assert run() == 0

        # clear combined_done in state.json directly, then delete the output
        out_md = tmp_path / "doc.md"
        out = tmp_path / "doc"
        state_path = out / "state.json"
        data = json.loads(state_path.read_text(encoding="utf-8"))
        data["combined_done"] = False
        state_path.write_text(json.dumps(data), encoding="utf-8")
        out_md.unlink()
        assert not out_md.exists()

        # re-run — Phase 3 should regenerate the file
        with patch.object(sys, "argv", ["main.py", str(pdf)]):
            assert run() == 0
        assert out_md.is_file()
    finally:
        server.shutdown()
