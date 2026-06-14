"""Unit tests for pdf_extractor/config.py."""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from pdf_extractor.config import (
    DEFAULT_MODEL,
    DEFAULT_URL,
    AppConfig,
    OllamaInstance,
    _parse,
    load_config,
)


# ---------------------------------------------------------------------------
# _parse
# ---------------------------------------------------------------------------

def test_parse_basic(tmp_path):
    data = {"instances": [{"url": "http://host:11434", "model": "qwen3-vl:8b"}]}
    cfg = _parse(data, cpu_count=4)
    assert len(cfg.instances) == 1
    assert cfg.instances[0].url == "http://host:11434"
    assert cfg.instances[0].model == "qwen3-vl:8b"


def test_parse_instance_default_model(tmp_path):
    data = {"instances": [{"url": "http://host:11434"}]}
    cfg = _parse(data, cpu_count=4)
    assert cfg.instances[0].model == DEFAULT_MODEL


def test_parse_missing_instances_raises():
    with pytest.raises(ValueError, match="instances"):
        _parse({}, cpu_count=4)


def test_parse_empty_instances_raises():
    with pytest.raises(ValueError, match="instances"):
        _parse({"instances": []}, cpu_count=4)


def test_parse_instance_missing_url_raises():
    with pytest.raises(ValueError, match="url"):
        _parse({"instances": [{"model": "qwen3-vl:8b"}]}, cpu_count=4)


def test_parse_max_render_workers_absent_uses_cpu_count():
    data = {"instances": [{"url": "http://h:11434"}]}
    cfg = _parse(data, cpu_count=8)
    assert cfg.max_render_workers == 8


def test_parse_max_render_workers_capped_to_cpu_count():
    data = {"instances": [{"url": "http://h:11434"}], "max_render_workers": 16}
    cfg = _parse(data, cpu_count=8)
    assert cfg.max_render_workers == 8


def test_parse_max_render_workers_below_cpu_count():
    data = {"instances": [{"url": "http://h:11434"}], "max_render_workers": 4}
    cfg = _parse(data, cpu_count=8)
    assert cfg.max_render_workers == 4


def test_parse_max_render_workers_zero_raises():
    data = {"instances": [{"url": "http://h:11434"}], "max_render_workers": 0}
    with pytest.raises(ValueError, match="positive"):
        _parse(data, cpu_count=4)


def test_parse_max_render_workers_negative_raises():
    data = {"instances": [{"url": "http://h:11434"}], "max_render_workers": -2}
    with pytest.raises(ValueError, match="positive"):
        _parse(data, cpu_count=4)


def test_parse_multiple_instances():
    data = {
        "instances": [
            {"url": "http://a:11434", "model": "qwen3-vl:32b"},
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
    cfg_data = {"instances": [{"url": "http://pdf-dir:11434", "model": "qwen3-vl:8b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg_data), encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    cfg = load_config(pdf)
    assert cfg.instances[0].url == "http://pdf-dir:11434"


def test_load_config_from_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_data = {"instances": [{"url": "http://cwd:11434", "model": "qwen3-vl:8b"}]}
    (tmp_path / "ollama.json").write_text(json.dumps(cfg_data), encoding="utf-8")
    pdf = tmp_path / "subdir" / "doc.pdf"
    pdf.parent.mkdir()
    pdf.touch()
    cfg = load_config(pdf)
    assert cfg.instances[0].url == "http://cwd:11434"


def test_load_config_defaults_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    cfg = load_config(pdf)
    assert len(cfg.instances) == 1
    assert cfg.instances[0].url == DEFAULT_URL
    assert cfg.instances[0].model == DEFAULT_MODEL


def test_load_config_invalid_json_raises(tmp_path):
    (tmp_path / "ollama.json").write_text("{bad json}", encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    with pytest.raises(json.JSONDecodeError):
        load_config(pdf)


def test_load_config_schema_error_raises(tmp_path):
    (tmp_path / "ollama.json").write_text(json.dumps({"instances": []}), encoding="utf-8")
    pdf = tmp_path / "doc.pdf"
    pdf.touch()
    with pytest.raises(ValueError):
        load_config(pdf)


def test_load_config_pdf_dir_takes_priority_over_cwd(tmp_path, monkeypatch):
    cwd_dir = tmp_path / "cwd"
    pdf_dir = tmp_path / "pdf"
    cwd_dir.mkdir(); pdf_dir.mkdir()
    monkeypatch.chdir(cwd_dir)
    (cwd_dir / "ollama.json").write_text(
        json.dumps({"instances": [{"url": "http://cwd:11434"}]}), encoding="utf-8"
    )
    (pdf_dir / "ollama.json").write_text(
        json.dumps({"instances": [{"url": "http://pdfdir:11434"}]}), encoding="utf-8"
    )
    pdf = pdf_dir / "doc.pdf"; pdf.touch()
    cfg = load_config(pdf)
    assert cfg.instances[0].url == "http://pdfdir:11434"
