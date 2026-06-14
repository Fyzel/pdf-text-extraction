"""Configuration loading and validation for pdf-text-extraction."""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL: str = "qwen3-vl:8b"
DEFAULT_URL: str = "http://localhost:11434"
_CONFIG_FILENAME: str = "ollama.json"


@dataclass
class OllamaInstance:
    """A single Ollama endpoint with its assigned model.

    Attributes:
        url: Base URL of the Ollama instance (e.g. ``http://localhost:11434``).
        model: Model name to use for OCR on this instance.
    """

    url: str
    model: str


@dataclass
class AppConfig:
    """Resolved application configuration derived from ollama.json.

    Attributes:
        instances: Ollama instances to distribute OCR work across.
        max_render_workers: Maximum parallel processes for Phase 1 image rendering.
    """

    instances: list[OllamaInstance]
    max_render_workers: int


def _parse(data: dict[str, Any], cpu_count: int) -> AppConfig:
    """Parse and validate a raw config dict into an AppConfig.

    Args:
        data: Decoded JSON object from ollama.json.
        cpu_count: Logical CPU core count used to cap ``max_render_workers``.

    Returns:
        Validated AppConfig with all instances and worker count resolved.

    Raises:
        ValueError: If ``instances`` is missing, empty, or contains an entry
            without a ``url``, or if ``max_render_workers`` is not a positive integer.
    """
    raw: Any = data.get("instances")
    if not isinstance(raw, list) or not raw:
        raise ValueError("'instances' must be a non-empty list")

    instances: list[OllamaInstance] = []
    for item in raw:
        if not isinstance(item, dict) or "url" not in item:
            raise ValueError("each instance must have a 'url'")
        instances.append(
            OllamaInstance(url=item["url"], model=item.get("model", DEFAULT_MODEL))
        )

    raw_workers: Any = data.get("max_render_workers")
    if raw_workers is None:
        workers: int = cpu_count
    else:
        workers = int(raw_workers)
        if workers <= 0:
            raise ValueError("max_render_workers must be a positive integer")
        workers = min(workers, cpu_count)

    return AppConfig(instances=instances, max_render_workers=workers)


def load_config(pdf_path: Path) -> AppConfig:
    """Load ollama.json from alongside the PDF, then cwd, then built-in defaults.

    Search order:
        1. Directory containing the PDF file.
        2. Current working directory.
        3. Built-in defaults (``http://localhost:11434``, model ``qwen3-vl:8b``).

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        Resolved AppConfig from the first found ollama.json, or built-in defaults
        when no config file is present.

    Raises:
        ValueError: If the found ollama.json fails schema validation.
        json.JSONDecodeError: If the found ollama.json is not valid JSON.
    """
    cpu_count: int = os.cpu_count() or 1
    search_dirs: dict[Path, None] = dict.fromkeys(
        [pdf_path.resolve().parent, Path.cwd().resolve()]
    )

    for directory in search_dirs:
        candidate: Path = directory / _CONFIG_FILENAME
        if candidate.is_file():
            print(f"Config: {candidate}")
            with open(candidate, encoding="utf-8") as fh:
                data: dict[str, Any] = json.load(fh)
            return _parse(data, cpu_count)

    print("Config: built-in defaults")
    return AppConfig(
        instances=[OllamaInstance(url=DEFAULT_URL, model=DEFAULT_MODEL)],
        max_render_workers=cpu_count,
    )
