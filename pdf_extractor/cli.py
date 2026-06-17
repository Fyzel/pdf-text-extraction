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
from pdf_extractor.render import _DPI_SCALE, get_page_count, render_pages
from pdf_extractor.state import AppState, StateManager

_USAGE: str = "Usage: python main.py <pdf_path> [--dpi-scale N]"


def _parse_args(argv: list[str]) -> tuple[str | None, float, str | None]:
    """Parse CLI arguments into a PDF path and render scale.

    Accepts one positional PDF path and an optional ``--dpi-scale N`` (or
    ``--dpi-scale=N``) flag controlling the Phase 1 render resolution.

    Args:
        argv: Argument list excluding the program name (``sys.argv[1:]``).

    Returns:
        Tuple of ``(pdf_path, dpi_scale, error)``. On success ``error`` is
        ``None``; on a parse error ``pdf_path`` is ``None`` and ``error`` holds
        a message. ``dpi_scale`` defaults to the module default when absent.
    """
    pdf_path: str | None = None
    dpi_scale: float = _DPI_SCALE
    i: int = 0
    while i < len(argv):
        arg: str = argv[i]
        if arg == "--dpi-scale" or arg.startswith("--dpi-scale="):
            if "=" in arg:
                raw: str = arg.split("=", 1)[1]
            else:
                i += 1
                if i >= len(argv):
                    return None, dpi_scale, "--dpi-scale requires a value"
                raw = argv[i]
            try:
                dpi_scale = float(raw)
            except ValueError:
                return None, dpi_scale, f"invalid --dpi-scale value: {raw}"
            if dpi_scale <= 0:
                return None, dpi_scale, "--dpi-scale must be a positive number"
        elif pdf_path is None:
            pdf_path = arg
        else:
            return None, dpi_scale, f"unexpected argument: {arg}"
        i += 1

    return pdf_path, dpi_scale, None


def run() -> int:
    """Validate arguments, load config, probe Ollama instances, and run Phase 1.

    Reads the PDF path and optional ``--dpi-scale N`` flag from ``sys.argv``.
    Performs all startup checks before any processing begins, then renders all
    PDF pages to JPEG.

    Returns:
        Exit code indicating outcome:

        - ``0``: Phase 1 succeeded (or run was already complete).
        - ``1``: No PDF path supplied, or invalid command-line arguments.
        - ``2``: PDF file not found.
        - ``3``: PDF file exists but is not readable or cannot be opened.
        - ``4``: ollama.json is invalid, or no Ollama instances are reachable.
        - ``5``: All pages failed image rendering.
        - ``6``: All rendered pages failed OCR.
        - ``7``: Combined output file write failed.
    """
    pdf_arg: str | None
    dpi_scale: float
    arg_err: str | None
    pdf_arg, dpi_scale, arg_err = _parse_args(sys.argv[1:])
    if arg_err is not None:
        print(f"Error: {arg_err}\n{_USAGE}", file=sys.stderr)
        return 1
    if pdf_arg is None:
        print(_USAGE, file=sys.stderr)
        return 1

    pdf_path: Path = Path(pdf_arg)

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
    print(f"DPI scale: {dpi_scale}")
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
            pdf_path, pages_dir, page_count, pending, config.max_render_workers,
            dpi_scale,
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
            state, state_mgr, config.ocr_timeout, pdf_path, dpi_scale,
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
