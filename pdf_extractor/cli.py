"""Command-line entry point and startup validation."""
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

from pdf_extractor.combine import run_phase3
from pdf_extractor.config import AppConfig, OllamaInstance, load_config
from pdf_extractor.health import probe_instances
from pdf_extractor.ocr import run_phase2
from pdf_extractor.render import get_page_count, render_pages
from pdf_extractor.state import AppState, StateManager


def run() -> int:
    """Validate arguments, load config, probe Ollama instances, and run Phase 1.

    Reads ``sys.argv[1]`` as the PDF path. Performs all startup checks before
    any processing begins, then renders all PDF pages to JPEG.

    Returns:
        Exit code indicating outcome:

        - ``0``: Phase 1 succeeded (or run was already complete).
        - ``1``: No PDF path argument supplied.
        - ``2``: PDF file not found.
        - ``3``: PDF file exists but is not readable or cannot be opened.
        - ``4``: ollama.json is invalid, or no Ollama instances are reachable.
        - ``5``: All pages failed image rendering.
        - ``6``: All rendered pages failed OCR.
        - ``7``: Combined output file write failed.
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

    # --- Page count ---
    try:
        page_count: int = get_page_count(pdf_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: cannot read PDF: {exc}", file=sys.stderr)
        return 3

    print(f"Pages: {page_count}")

    # --- State init / load ---
    state_mgr: StateManager = StateManager(output_dir)
    state: AppState = state_mgr.load_or_init(pdf_path, page_count)

    run_status: str = state_mgr.status(state)
    if run_status == "complete":
        md_path: Path = pdf_path.parent / f"{pdf_path.stem}.md"
        print(f"Already complete: {md_path}")
        return 0
    if run_status == "partial":
        print("Resuming from partial run")

    # --- Phase 1: image rendering ---
    pending: list[int] = [
        i for i in range(1, page_count + 1)
        if not state.pages[str(i)].image_done
        and not state.pages[str(i)].image_failed
    ]

    if pending:
        pages_dir: Path = output_dir / "pages"
        print(
            f"Phase 1: rendering {len(pending)} page(s) "
            f"with {config.max_render_workers} worker(s)..."
        )
        results: list[tuple[int, bool, str]] = render_pages(
            pdf_path, pages_dir, page_count, pending, config.max_render_workers
        )
        for page_num, success, error in results:
            if success:
                state_mgr.update_page(state, page_num, image_done=True)
            else:
                print(f"  Page {page_num}: render failed — {error}", file=sys.stderr)
                state_mgr.update_page(state, page_num, image_failed=True)

    if all(state.pages[str(i)].image_failed for i in range(1, page_count + 1)):
        print("Error: all pages failed image rendering", file=sys.stderr)
        return 5

    rendered_count: int = sum(
        1 for i in range(1, page_count + 1) if state.pages[str(i)].image_done
    )
    print(f"Phase 1 complete: {rendered_count}/{page_count} page(s) rendered")

    # --- Phase 2: OCR ---
    ocr_pending: list[int] = [
        i for i in range(1, page_count + 1)
        if state.pages[str(i)].image_done
        and not state.pages[str(i)].ocr_done
        and not state.pages[str(i)].ocr_failed
    ]

    if ocr_pending:
        print(
            f"Phase 2: OCR {len(ocr_pending)} page(s) "
            f"with {len(config.instances)} instance(s) "
            f"(timeout {config.ocr_timeout}s)..."
        )
        run_phase2(
            output_dir, page_count, ocr_pending, config.instances,
            state, state_mgr, config.ocr_timeout, pdf_path,
        )

    rendered_pages: list[int] = [
        i for i in range(1, page_count + 1) if state.pages[str(i)].image_done
    ]
    if rendered_pages and all(state.pages[str(i)].ocr_failed for i in rendered_pages):
        print("Error: all pages failed OCR processing", file=sys.stderr)
        return 6

    ocr_count: int = sum(
        1 for i in range(1, page_count + 1) if state.pages[str(i)].ocr_done
    )
    print(f"Phase 2 complete: {ocr_count}/{page_count} page(s) OCR'd")

    # --- Phase 3: combine ---
    print("Phase 3: combining per-page markdown...")
    ok: bool
    err: str
    ok, err = run_phase3(pdf_path, output_dir, page_count, state, state_mgr)
    if not ok:
        print(f"Error: output file write failed — {err}", file=sys.stderr)
        return 7

    output_md: Path = pdf_path.parent / f"{pdf_path.stem}.md"

    # --- Summary ---
    render_ok: int = sum(1 for i in range(1, page_count + 1) if state.pages[str(i)].image_done)
    render_fail: int = sum(1 for i in range(1, page_count + 1) if state.pages[str(i)].image_failed)
    ocr_ok: int = sum(1 for i in range(1, page_count + 1) if state.pages[str(i)].ocr_done)
    ocr_fail: int = sum(1 for i in range(1, page_count + 1) if state.pages[str(i)].ocr_failed)

    print(f"Done: {output_md}")
    print(f"  Pages total:        {page_count}")
    print(f"  Image rendered:     {render_ok}")
    print(f"  Image failed:       {render_fail}")
    print(f"  OCR succeeded:      {ocr_ok}")
    print(f"  OCR failed:         {ocr_fail}")

    return 0
