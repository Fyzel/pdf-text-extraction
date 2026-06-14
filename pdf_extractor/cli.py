"""Command-line entry point and startup validation."""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from pdf_extractor.config import AppConfig, OllamaInstance, load_config
from pdf_extractor.health import probe_instances


def run() -> int:
    """Validate arguments, load config, probe Ollama instances, and prepare output directory.

    Reads ``sys.argv[1]`` as the PDF path. Performs all startup checks before
    any processing begins.

    Returns:
        Exit code indicating the outcome of startup validation:

        - ``0``: startup succeeded, ready for processing.
        - ``1``: no PDF path argument supplied.
        - ``2``: PDF file not found.
        - ``3``: PDF file exists but is not readable.
        - ``4``: ollama.json is invalid, or no Ollama instances are reachable.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_path>", file=sys.stderr)
        return 1

    pdf_path: Path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        return 2

    if not pdf_path.is_file() or not os.access(pdf_path, os.R_OK):
        print(f"Error: file not readable: {pdf_path}", file=sys.stderr)
        return 3

    try:
        config: AppConfig = load_config(pdf_path)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Error: invalid ollama.json — {exc}", file=sys.stderr)
        return 4

    print(f"Render workers: {config.max_render_workers}")
    print(f"Probing {len(config.instances)} Ollama instance(s)...")

    live: list[OllamaInstance] = probe_instances(config.instances)
    if not live:
        print("Error: no Ollama instances reachable", file=sys.stderr)
        return 4

    config = replace(config, instances=live)
    print(f"Ready — {len(live)} instance(s) reachable")

    output_dir: Path = pdf_path.parent / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Segment 3 continues here: page count via PyMuPDF → state init → Phase 1
    _ = config
    return 0
