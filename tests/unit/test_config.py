"""Unit tests for pdf_extractor/config.py."""
import json

import pytest

from pdf_extractor.config import (
    DEFAULT_MODEL,
    DEFAULT_OCR_TIMEOUT,
    DEFAULT_URL,
    _parse,
    load_config,
)


# ---------------------------------------------------------------------------
# _parse
# ---------------------------------------------------------------------------

def test_parse_basic():
    """A full instance entry parses its URL and model verbatim.

    :return: ``None``.
    :rtype: None
    """
    data = {"instances": [{"url": "http://host:11434", "model": "qwen2.5vl:7b"}]}
    cfg = _parse(data, cpu_count=4)
    assert len(cfg.instances) == 1
    assert cfg.instances[0].url == "http://host:11434"
    assert cfg.instances[0].model == "qwen2.5vl:7b"


def test_parse_instance_default_model():
    """An instance without a model falls back to :data:`DEFAULT_MODEL`.

    :return: ``None``.
    :rtype: None
    """
    data = {"instances": [{"url": "http://host:11434"}]}
    cfg = _parse(data, cpu_count=4)
    assert cfg.instances[0].model == DEFAULT_MODEL


def test_parse_missing_instances_raises():
    """A config without ``instances`` raises :class:`ValueError`.

    :return: ``None``.
    :rtype: None
    """
    with pytest.raises(ValueError, match="instances"):
        _parse({}, cpu_count=4)


def test_parse_empty_instances_raises():
    """An empty ``instances`` list raises :class:`ValueError`.

    :return: ``None``.
    :rtype: None
    """
    with pytest.raises(ValueError, match="instances"):
        _parse({"instances": []}, cpu_count=4)


def test_parse_instance_missing_url_raises():
    """An instance without a ``url`` raises :class:`ValueError`.

    :return: ``None``.
    :rtype: None
    """
    with pytest.raises(ValueError, match="url"):
        _parse({"instances": [{"model": "qwen2.5vl:7b"}]}, cpu_count=4)


def test_parse_max_render_workers_absent_uses_cpu_count():
    """An absent ``max_render_workers`` defaults to the CPU count.

    :return: ``None``.
    :rtype: None
    """
    data = {"instances": [{"url": "http://h:11434"}]}
    cfg = _parse(data, cpu_count=8)
    assert cfg.max_render_workers == 8


def test_parse_max_render_workers_capped_to_cpu_count():
    """A ``max_render_workers`` above the CPU count is capped to it.

    :return: ``None``.
    :rtype: None
    """
    data = {"instances": [{"url": "http://h:11434"}], "max_render_workers": 16}
    cfg = _parse(data, cpu_count=8)
    assert cfg.max_render_workers == 8


def test_parse_max_render_workers_below_cpu_count():
    """A ``max_render_workers`` below the CPU count is kept as-is.

    :return: ``None``.
    :rtype: None
    """
    data = {"instances": [{"url": "http://h:11434"}], "max_render_workers": 4}
    cfg = _parse(data, cpu_count=8)
    assert cfg.max_render_workers == 4


def test_parse_max_render_workers_zero_raises():
    """A zero ``max_render_workers`` raises :class:`ValueError`.

    :return: ``None``.
    :rtype: None
    """
    data = {"instances": [{"url": "http://h:11434"}], "max_render_workers": 0}
    with pytest.raises(ValueError, match="positive"):
        _parse(data, cpu_count=4)


def test_parse_max_render_workers_negative_raises():
    """A negative ``max_render_workers`` raises :class:`ValueError`.

    :return: ``None``.
    :rtype: None
    """
    data = {"instances": [{"url": "http://h:11434"}], "max_render_workers": -2}
    with pytest.raises(ValueError, match="positive"):
        _parse(data, cpu_count=4)


def test_parse_ocr_timeout_absent_uses_default():
    """An absent ``ocr_timeout`` defaults to :data:`DEFAULT_OCR_TIMEOUT`.

    :return: ``None``.
    :rtype: None
    """
    data = {"instances": [{"url": "http://h:11434"}]}
    cfg = _parse(data, cpu_count=4)
    assert cfg.ocr_timeout == DEFAULT_OCR_TIMEOUT


def test_parse_ocr_timeout_explicit():
    """An explicit ``ocr_timeout`` is kept as-is.

    :return: ``None``.
    :rtype: None
    """
    data = {"instances": [{"url": "http://h:11434"}], "ocr_timeout": 900}
    cfg = _parse(data, cpu_count=4)
    assert cfg.ocr_timeout == 900


def test_parse_ocr_timeout_zero_raises():
    """A zero ``ocr_timeout`` raises :class:`ValueError`.

    :return: ``None``.
    :rtype: None
    """
    data = {"instances": [{"url": "http://h:11434"}], "ocr_timeout": 0}
    with pytest.raises(ValueError, match="positive"):
        _parse(data, cpu_count=4)


def test_parse_ocr_timeout_negative_raises():
    """A negative ``ocr_timeout`` raises :class:`ValueError`.

    :return: ``None``.
    :rtype: None
    """
    data = {"instances": [{"url": "http://h:11434"}], "ocr_timeout": -60}
    with pytest.raises(ValueError, match="positive"):
        _parse(data, cpu_count=4)


def test_parse_multiple_instances():
    """Multiple instances parse, each defaulting its model independently.

    :return: ``None``.
    :rtype: None
    """
    data = {
        "instances": [
            {"url": "http://a:11434", "model": "qwen2.5vl:32b"},
            {"url": "http://b:11434"},
        ]
    }
    cfg = _parse(data, cpu_count=4)
    assert len(cfg.instances) == 2
    assert cfg.instances[1].model == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_from_pdf_dir(tmp_path):
    """``ollama.json`` beside the PDF is loaded.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    cfg_data = {"instances": [{"url": "http://pdf-dir:11434", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg_data), encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    cfg = load_config(pdf)
    assert cfg.instances[0].url == "http://pdf-dir:11434"


def test_load_config_from_cwd(tmp_path, monkeypatch):
    """``ollama.json`` in the cwd is loaded when none sits beside the PDF.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    cfg_data = {"instances": [{"url": "http://cwd:11434", "model": "qwen2.5vl:7b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg_data), encoding="utf-8")
    pdf = tmp_path / "subdir" / "doc.pdf"
    pdf.parent.mkdir()
    pdf.touch()
    cfg = load_config(pdf)
    assert cfg.instances[0].url == "http://cwd:11434"


def test_load_config_defaults_when_no_file(tmp_path, monkeypatch):
    """With no config file, built-in defaults are used.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    cfg = load_config(pdf)
    assert len(cfg.instances) == 1
    assert cfg.instances[0].url == DEFAULT_URL
    assert cfg.instances[0].model == DEFAULT_MODEL


def test_load_config_invalid_json_raises(tmp_path):
    """Invalid JSON in ``ollama.json`` raises :class:`json.JSONDecodeError`.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    (tmp_path / "ollama.json").write_text("{bad json}", encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    with pytest.raises(json.JSONDecodeError):
        load_config(pdf)


def test_load_config_schema_error_raises(tmp_path):
    """A schema-invalid ``ollama.json`` raises :class:`ValueError`.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    (tmp_path / "ollama.json").write_text(json.dumps({"instances": []}), encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    with pytest.raises(ValueError):
        load_config(pdf)


def test_load_config_pdf_dir_takes_priority_over_cwd(tmp_path, monkeypatch):
    """A config beside the PDF takes priority over one in the cwd.

    :param tmp_path: pytest temporary-directory fixture. Required.
    :type tmp_path: pathlib.Path
    :param monkeypatch: pytest monkeypatch fixture. Required.
    :type monkeypatch: _pytest.monkeypatch.MonkeyPatch
    :return: ``None``.
    :rtype: None
    """
    cwd_dir = tmp_path / "cwd"
    pdf_dir = tmp_path / "pdf"
    cwd_dir.mkdir()
    pdf_dir.mkdir()
    monkeypatch.chdir(cwd_dir)
    (cwd_dir / "ollama.json").write_text(
        json.dumps({"instances": [{"url": "http://cwd:11434"}]}), encoding="utf-8"
    )
    (pdf_dir / "ollama.json").write_text(
        json.dumps({"instances": [{"url": "http://pdfdir:11434"}]}), encoding="utf-8"
    )
    pdf = pdf_dir / "doc.pdf"
    pdf.touch()
    cfg = load_config(pdf)
    assert cfg.instances[0].url == "http://pdfdir:11434"
