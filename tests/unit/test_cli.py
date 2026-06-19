"""Unit tests for pdf_extractor/cli.py — argument parsing and exit codes."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from pdf_extractor.cli import (
    _archive_page_artifacts,
    _next_archive_dir,
    _parse_args,
    _parse_page_spec,
    run,
)
from pdf_extractor.render import _DPI_SCALE
from tests.helpers import make_text_pdf
from tests.ollama_mock import start_ollama_mock


# ---------------------------------------------------------------------------
# Exit 1 — missing argument
# ---------------------------------------------------------------------------

def test_exit1_no_argument():
    """With no PDF argument, :func:`run` exits ``1``.

    :return: ``None``.
    :rtype: None
    """
    with patch.object(sys, "argv", ["main.py"]):
        assert run() == 1


# ---------------------------------------------------------------------------
# Argument parsing — _parse_args / --dpi-scale
# ---------------------------------------------------------------------------

def test_parse_args_pdf_only():
    """A lone PDF path parses with the default DPI scale and no error.

    :return: ``None``.
    :rtype: None
    """
    pdf, dpi, _, _, _, err = _parse_args(["doc.pdf"])
    assert pdf == "doc.pdf"
    assert dpi == _DPI_SCALE
    assert err is None


def test_parse_args_dpi_scale_space():
    """``--dpi-scale N`` (space form) sets the render scale.

    :return: ``None``.
    :rtype: None
    """
    pdf, dpi, _, _, _, err = _parse_args(["doc.pdf", "--dpi-scale", "4.0"])
    assert pdf == "doc.pdf"
    assert dpi == 4.0
    assert err is None


def test_parse_args_dpi_scale_equals():
    """``--dpi-scale=N`` (equals form) sets the render scale.

    :return: ``None``.
    :rtype: None
    """
    pdf, dpi, _, _, _, err = _parse_args(["--dpi-scale=3.5", "doc.pdf"])
    assert pdf == "doc.pdf"
    assert dpi == 3.5
    assert err is None


def test_parse_args_missing_value():
    """``--dpi-scale`` with no value is a parse error.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, _, _, _, err = _parse_args(["doc.pdf", "--dpi-scale"])
    assert pdf is None
    assert err is not None


def test_parse_args_invalid_value():
    """A non-numeric ``--dpi-scale`` value is a parse error.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, _, _, _, err = _parse_args(["doc.pdf", "--dpi-scale", "huge"])
    assert pdf is None
    assert err is not None


def test_parse_args_non_positive_value():
    """A non-positive ``--dpi-scale`` value is a parse error.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, _, _, _, err = _parse_args(["doc.pdf", "--dpi-scale", "0"])
    assert pdf is None
    assert err is not None


def test_parse_args_unexpected_extra():
    """A second positional argument is a parse error.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, _, _, _, err = _parse_args(["doc.pdf", "extra.pdf"])
    assert pdf is None
    assert err is not None


def test_parse_args_no_pdf():
    """Flags without a PDF path parse cleanly with a ``None`` path.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, _, _, _, err = _parse_args(["--dpi-scale", "2.0"])
    assert pdf is None
    assert err is None


def test_parse_args_include_comments_default_false():
    """``--include-comments`` defaults to ``False`` when absent.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, include_comments, _, _, err = _parse_args(["doc.pdf"])
    assert pdf == "doc.pdf"
    assert include_comments is False
    assert err is None


def test_parse_args_include_comments_flag():
    """``--include-comments`` sets the flag to ``True``.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, include_comments, _, _, err = _parse_args(["doc.pdf", "--include-comments"])
    assert pdf == "doc.pdf"
    assert include_comments is True
    assert err is None


def test_parse_args_include_comments_with_dpi():
    """``--include-comments`` combines with ``--dpi-scale`` and a PDF path.

    :return: ``None``.
    :rtype: None
    """
    pdf, dpi, include_comments, _, _, err = _parse_args(
        ["--include-comments", "--dpi-scale", "3", "doc.pdf"]
    )
    assert pdf == "doc.pdf"
    assert dpi == 3.0
    assert include_comments is True
    assert err is None


def test_parse_args_include_links_default_false():
    """``--include-links`` defaults to ``False`` when absent.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, _, include_links, _, err = _parse_args(["doc.pdf"])
    assert pdf == "doc.pdf"
    assert include_links is False
    assert err is None


def test_parse_args_include_links_flag():
    """``--include-links`` sets the flag to ``True``.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, _, include_links, _, err = _parse_args(["doc.pdf", "--include-links"])
    assert pdf == "doc.pdf"
    assert include_links is True
    assert err is None


# ---------------------------------------------------------------------------
# Page spec parsing — _parse_page_spec / --rerun-pages
# ---------------------------------------------------------------------------

def test_page_spec_single_list():
    """A comma list of single pages parses to the matching set.

    :return: ``None``.
    :rtype: None
    """
    assert _parse_page_spec("3,5,7") == {3, 5, 7}


def test_page_spec_range():
    """An ``N-M`` range parses to the inclusive set.

    :return: ``None``.
    :rtype: None
    """
    assert _parse_page_spec("7-9") == {7, 8, 9}


def test_page_spec_mixed():
    """A mix of singles and ranges parses to their union.

    :return: ``None``.
    :rtype: None
    """
    assert _parse_page_spec("3,5,7-9") == {3, 5, 7, 8, 9}


def test_page_spec_whitespace_and_overlap():
    """Whitespace is ignored and overlapping pages collapse in the set.

    :return: ``None``.
    :rtype: None
    """
    assert _parse_page_spec(" 1 , 2-4 , 3 ") == {1, 2, 3, 4}


@pytest.mark.parametrize("bad", ["", "x", "3,x", "5-", "-5", "4-2", "0", "3,0", "1-0"])
def test_page_spec_malformed_raises(bad):
    """A malformed page spec raises :class:`ValueError`.

    :param bad: Malformed spec string under test. Required (parametrized).
    :type bad: str
    :return: ``None``.
    :rtype: None
    """
    with pytest.raises(ValueError):
        _parse_page_spec(bad)


def test_parse_args_rerun_pages_space():
    """``--rerun-pages SPEC`` (space form) parses to the page set.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, _, _, rerun, err = _parse_args(["doc.pdf", "--rerun-pages", "3,5,7-9"])
    assert pdf == "doc.pdf"
    assert rerun == {3, 5, 7, 8, 9}
    assert err is None


def test_parse_args_rerun_pages_equals():
    """``--rerun-pages=SPEC`` (equals form) parses to the page set.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, _, _, rerun, err = _parse_args(["--rerun-pages=2-4", "doc.pdf"])
    assert pdf == "doc.pdf"
    assert rerun == {2, 3, 4}
    assert err is None


def test_parse_args_rerun_pages_default_none():
    """``rerun_pages`` is ``None`` when the flag is absent.

    :return: ``None``.
    :rtype: None
    """
    _, _, _, _, rerun, err = _parse_args(["doc.pdf"])
    assert rerun is None
    assert err is None


def test_parse_args_rerun_pages_missing_value():
    """``--rerun-pages`` with no value is a parse error.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, _, _, _, err = _parse_args(["doc.pdf", "--rerun-pages"])
    assert pdf is None
    assert err is not None


def test_parse_args_rerun_pages_malformed():
    """A malformed ``--rerun-pages`` value is a parse error.

    :return: ``None``.
    :rtype: None
    """
    pdf, _, _, _, _, err = _parse_args(["doc.pdf", "--rerun-pages", "3,x"])
    assert pdf is None
    assert err is not None


# ---------------------------------------------------------------------------
# Archive helpers — _next_archive_dir / _archive_page_artifacts
# ---------------------------------------------------------------------------

def test_next_archive_dir_first(tmp_path):
    """With no archive yet, the next version directory is ``v1``.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    assert _next_archive_dir(tmp_path).name == "v1"


def test_next_archive_dir_increments(tmp_path):
    """The next version is one past the highest existing ``vN``.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    (tmp_path / "_archive" / "v1").mkdir(parents=True)
    (tmp_path / "_archive" / "v2").mkdir(parents=True)
    assert _next_archive_dir(tmp_path).name == "v3"


def test_next_archive_dir_ignores_non_version(tmp_path):
    """Non ``vN`` archive entries are ignored when choosing the next version.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    (tmp_path / "_archive" / "v5").mkdir(parents=True)
    (tmp_path / "_archive" / "notes").mkdir(parents=True)
    assert _next_archive_dir(tmp_path).name == "v6"


def _seed_page_files(output_dir: Path, pdf: Path, page_count: int, page_num: int) -> None:
    """Create page/diagram/combined artifacts for one page, for archive tests.

    :param output_dir: Per-document working directory to seed. Required.
    :type output_dir: pathlib.Path
    :param pdf: Source PDF path (used to locate the combined output). Required.
    :type pdf: pathlib.Path
    :param page_count: Total page count, for the zero-padded stem. Required.
    :type page_count: int
    :param page_num: 1-based page number to seed. Required.
    :type page_num: int
    :return: ``None``.
    :rtype: None
    """
    stem = f"page_{page_num:0{len(str(page_count))}d}"
    (output_dir / "pages").mkdir(parents=True, exist_ok=True)
    (output_dir / "diagrams").mkdir(parents=True, exist_ok=True)
    (output_dir / "pages" / f"{stem}.jpg").write_bytes(b"jpg")
    (output_dir / "pages" / f"{stem}.md").write_text("md", encoding="utf-8")
    (output_dir / "diagrams" / f"{stem}_diagram_1.jpg").write_bytes(b"d1")
    (output_dir / "diagrams" / f"{stem}_diagram_2.jpg").write_bytes(b"d2")
    (pdf.parent / f"{pdf.stem}.md").write_text("combined", encoding="utf-8")


def test_archive_moves_not_deletes(tmp_path):
    """Archiving moves a page's artifacts into ``v1`` and removes the originals.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    out = tmp_path / "doc"
    _seed_page_files(out, pdf, 10, 3)

    arch = _archive_page_artifacts(out, pdf, 10, [3])

    assert arch is not None and arch.name == "v1"
    # originals gone
    assert not (out / "pages" / "page_03.jpg").exists()
    assert not (out / "pages" / "page_03.md").exists()
    assert not (out / "diagrams" / "page_03_diagram_1.jpg").exists()
    assert not (pdf.parent / "doc.md").exists()
    # archived copies present
    assert (arch / "pages" / "page_03.jpg").read_bytes() == b"jpg"
    assert (arch / "pages" / "page_03.md").read_text(encoding="utf-8") == "md"
    assert (arch / "diagrams" / "page_03_diagram_1.jpg").exists()
    assert (arch / "diagrams" / "page_03_diagram_2.jpg").exists()
    assert (arch / "doc.md").read_text(encoding="utf-8") == "combined"


def test_archive_missing_files_returns_none(tmp_path):
    """Archiving when nothing exists to move returns ``None``.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    out = tmp_path / "doc"
    out.mkdir()
    assert _archive_page_artifacts(out, pdf, 10, [3]) is None


def test_archive_second_rerun_uses_v2(tmp_path):
    """A second archive of the same page lands in ``v2``.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF")
    out = tmp_path / "doc"
    _seed_page_files(out, pdf, 10, 3)
    _archive_page_artifacts(out, pdf, 10, [3])
    _seed_page_files(out, pdf, 10, 3)
    arch2 = _archive_page_artifacts(out, pdf, 10, [3])
    assert arch2 is not None and arch2.name == "v2"


def test_exit1_bad_dpi_scale():
    """An invalid ``--dpi-scale`` makes :func:`run` exit ``1``.

    :return: ``None``.
    :rtype: None
    """
    with patch.object(sys, "argv", ["main.py", "doc.pdf", "--dpi-scale", "nope"]):
        assert run() == 1


def test_help_exits_zero(capsys):
    """``--help`` prints usage and exits ``0``.

    :param capsys: pytest captured-output fixture. Required.
    :type capsys: _pytest.capture.CaptureFixture
    :return: ``None``.
    :rtype: None
    """
    with patch.object(sys, "argv", ["main.py", "--help"]):
        assert run() == 0
    out = capsys.readouterr().out
    assert "--dpi-scale" in out


# ---------------------------------------------------------------------------
# Exit 2 — file not found
# ---------------------------------------------------------------------------

def test_exit2_file_not_found(tmp_path):
    """A missing PDF path makes :func:`run` exit ``2``.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    with patch.object(sys, "argv", ["main.py", str(tmp_path / "missing.pdf")]):
        assert run() == 2


# ---------------------------------------------------------------------------
# Exit 3 — file not readable
# ---------------------------------------------------------------------------

def test_exit3_not_a_file(tmp_path):
    """A path that exists but is not a file makes :func:`run` exit ``3``.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    # passing a directory satisfies exists() but not is_file()
    with patch.object(sys, "argv", ["main.py", str(tmp_path)]):
        assert run() == 3


# ---------------------------------------------------------------------------
# Exit 4 — no Ollama instances reachable
# ---------------------------------------------------------------------------

def test_exit4_no_ollama(tmp_path, monkeypatch):
    """With no reachable Ollama instance, :func:`run` exits ``4``.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "doc.pdf"
    make_text_pdf(pdf)
    cfg = {"instances": [{"url": "http://127.0.0.1:19599", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")
    with patch.object(sys, "argv", ["main.py", str(pdf)]):
        assert run() == 4


# ---------------------------------------------------------------------------
# Exit 5 — all pages fail rendering (corrupt PDF)
# ---------------------------------------------------------------------------

def test_exit5_corrupt_pdf(tmp_path, monkeypatch):
    """A corrupt PDF makes :func:`run` exit ``3`` (unreadable) or ``5`` (all fail).

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    port = 19590
    server = start_ollama_mock(port, {"text": "x", "diagrams": []})
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
    """A second run of a completed document returns ``0`` immediately.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    port = 19591
    server = start_ollama_mock(port, {"text": "hello", "diagrams": []})
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")

    pdf = tmp_path / "doc.pdf"
    make_text_pdf(pdf)

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
    """A full two-page run writes the combined output and exits ``0``.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    port = 19592
    server = start_ollama_mock(port, {"text": "page text", "diagrams": []})
    cfg = {"instances": [{"url": f"http://127.0.0.1:{port}", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg), encoding="utf-8")

    pdf = tmp_path / "doc.pdf"
    make_text_pdf(pdf, pages=2)

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
