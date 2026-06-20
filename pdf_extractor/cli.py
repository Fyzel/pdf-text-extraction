"""Command-line entry point and startup validation."""
import json
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from pdf_extractor.combine import run_phase3
from pdf_extractor.config import AppConfig, OllamaInstance, load_config
from pdf_extractor.health import probe_instances
from pdf_extractor.ocr import _page_stem, run_phase2
from pdf_extractor.pdf_errors import PDF_ERRORS
from pdf_extractor.render import _DPI_SCALE, get_page_count, render_pages
from pdf_extractor.state import AppState, StateManager, StateMismatchError

_USAGE: str = (
    "Usage: python main.py <pdf_path> [--dpi-scale N] [--include-comments] "
    "[--include-links] [--rerun-pages SPEC]"
)
_HELP: str = f"""\
{_USAGE}

Extract text and diagrams from a PDF via Ollama vision OCR.

Arguments:
  <pdf_path>          Path to the source PDF file.
  --dpi-scale N       Page render scale factor (default 2.0, ~144 DPI). Higher
                      values give sharper images and better OCR of small text at
                      the cost of size and render time. Also accepts --dpi-scale=N.
  --include-comments  Append PDF comment annotations (sticky notes, highlight
                      notes, etc.) to each page as a "## Comments" section.
  --include-links     Rewrite plain text as Markdown links using the PDF's
                      embedded URI hyperlinks (e.g. [text](https://...)).
  --rerun-pages SPEC  Reprocess specific pages from a previous run. SPEC is a
                      comma-separated list of page numbers and/or N-M ranges,
                      e.g. "3,5,7-9". The page image, diagrams, per-page markdown,
                      and combined output are archived under <stem>/_archive/vN/
                      then regenerated. Requires an existing state.json.
                      Also accepts --rerun-pages=SPEC.
  -h, --help          Show this help message and exit.
"""


def _parse_page_spec(raw: str) -> set[int]:
    """Parse a page specification string into a set of 1-based page numbers.

    The spec is a comma-separated list of individual page numbers and inclusive
    ranges, e.g. ``"3,5,7-9"`` yields ``{3, 5, 7, 8, 9}``. Whitespace around
    tokens is ignored. Page numbers must be positive integers; ranges must be
    ascending (``N <= M``).

    :param raw: The raw spec string, e.g. ``"3,5,7-9"``. Required.
    :type raw: str
    :return: Set of 1-based page numbers.
    :rtype: set[int]
    :raises ValueError: If the spec is empty or contains a malformed token.
    """
    pages: set[int] = set()
    tokens: list[str] = [t.strip() for t in raw.split(",")]
    if not any(tokens):
        raise ValueError("empty page spec")
    for token in tokens:
        if not token:
            raise ValueError("empty page spec token")
        if "-" in token:
            lo_str, _, hi_str = token.partition("-")
            lo_str, hi_str = lo_str.strip(), hi_str.strip()
            if not lo_str or not hi_str:
                raise ValueError(f"malformed range: {token!r}")
            try:
                lo, hi = int(lo_str), int(hi_str)
            except ValueError as exc:
                raise ValueError(f"non-integer in range: {token!r}") from exc
            if lo < 1 or hi < 1:
                raise ValueError(f"page numbers must be positive: {token!r}")
            if lo > hi:
                raise ValueError(f"descending range: {token!r}")
            pages.update(range(lo, hi + 1))
        else:
            try:
                num: int = int(token)
            except ValueError as exc:
                raise ValueError(f"non-integer page: {token!r}") from exc
            if num < 1:
                raise ValueError(f"page numbers must be positive: {token!r}")
            pages.add(num)
    return pages


def _parse_args(
    argv: list[str],
) -> tuple[str | None, float, bool, bool, set[int] | None, str | None]:
    """Parse CLI arguments into a PDF path, render scale, and flags.

    Accepts one positional PDF path, an optional ``--dpi-scale N`` (or
    ``--dpi-scale=N``) flag controlling the Phase 1 render resolution, an
    optional ``--include-comments`` flag, an optional ``--include-links`` flag,
    and an optional ``--rerun-pages SPEC`` (or ``--rerun-pages=SPEC``) flag
    listing pages to reprocess.

    :param argv: Argument list excluding the program name (``sys.argv[1:]``).
        Required.
    :type argv: list[str]
    :return: Tuple of ``(pdf_path, dpi_scale, include_comments, include_links,
        rerun_pages, error)``. On success ``error`` is ``None``; on a parse
        error ``pdf_path`` is ``None`` and ``error`` holds a message.
        ``dpi_scale`` defaults to the module default when absent; ``rerun_pages``
        is ``None`` when the flag is not supplied.
    :rtype: tuple[str | None, float, bool, bool, set[int] | None, str | None]
    """
    pdf_path: str | None = None
    dpi_scale: float = _DPI_SCALE
    include_comments: bool = False
    include_links: bool = False
    rerun_pages: set[int] | None = None

    def _err(msg: str) -> tuple[None, float, bool, bool, set[int] | None, str]:
        """Build the parse-failure return tuple carrying an error message.

        :param msg: Human-readable parse-error message. Required.
        :type msg: str
        :return: The :func:`_parse_args` result tuple with a ``None`` PDF path
            and ``msg`` as the error.
        :rtype: tuple[None, float, bool, bool, set[int] | None, str]
        """
        return None, dpi_scale, include_comments, include_links, rerun_pages, msg

    i: int = 0
    while i < len(argv):
        arg: str = argv[i]
        if arg == "--dpi-scale" or arg.startswith("--dpi-scale="):
            if "=" in arg:
                raw: str = arg.split("=", 1)[1]
            else:
                i += 1
                if i >= len(argv):
                    return _err("--dpi-scale requires a value")
                raw = argv[i]
            try:
                dpi_scale = float(raw)
            except ValueError:
                return _err(f"invalid --dpi-scale value: {raw}")
            if dpi_scale <= 0:
                return _err("--dpi-scale must be a positive number")
        elif arg == "--rerun-pages" or arg.startswith("--rerun-pages="):
            if "=" in arg:
                spec: str = arg.split("=", 1)[1]
            else:
                i += 1
                if i >= len(argv):
                    return _err("--rerun-pages requires a value")
                spec = argv[i]
            try:
                rerun_pages = _parse_page_spec(spec)
            except ValueError as exc:
                return _err(f"invalid --rerun-pages value: {exc}")
        elif arg == "--include-comments":
            include_comments = True
        elif arg == "--include-links":
            include_links = True
        elif pdf_path is None:
            pdf_path = arg
        else:
            return _err(f"unexpected argument: {arg}")
        i += 1

    return pdf_path, dpi_scale, include_comments, include_links, rerun_pages, None


def _next_archive_dir(output_dir: Path) -> Path:
    """Return the next ``_archive/vN`` directory path (not yet created).

    Scans ``<output_dir>/_archive`` for existing ``v<N>`` subdirectories and
    returns a path with ``N`` one greater than the highest found, or ``v1`` if
    none exist.

    :param output_dir: The per-document working directory. Required.
    :type output_dir: pathlib.Path
    :return: Path to the next versioned archive directory.
    :rtype: pathlib.Path
    """
    archive_root: Path = output_dir / "_archive"
    highest: int = 0
    if archive_root.is_dir():
        for child in archive_root.iterdir():
            if child.is_dir() and child.name.startswith("v"):
                try:
                    highest = max(highest, int(child.name[1:]))
                except ValueError:
                    continue
    return archive_root / f"v{highest + 1}"


def _archive_page_artifacts(
    output_dir: Path,
    pdf_path: Path,
    page_count: int,
    page_nums: list[int],
) -> Path | None:
    """Move existing artifacts for the given pages into a new archive version.

    For each page, moves the rendered JPEG, per-page markdown, and any cropped
    diagram images into a fresh ``<output_dir>/_archive/vN/`` directory,
    preserving their relative ``pages/`` and ``diagrams/`` layout. The combined
    ``<stem>.md`` output (which lives alongside the PDF) is archived once at the
    archive root. Missing files are skipped silently — only what exists is moved.

    :param output_dir: The per-document working directory. Required.
    :type output_dir: pathlib.Path
    :param pdf_path: Path to the source PDF (used to locate the combined
        output). Required.
    :type pdf_path: pathlib.Path
    :param page_count: Total page count, for zero-padding filename stems.
        Required.
    :type page_count: int
    :param page_nums: 1-based page numbers whose artifacts should be archived.
        Required.
    :type page_nums: list[int]
    :return: The archive directory that was created, or ``None`` if nothing was
        moved.
    :rtype: pathlib.Path | None
    """
    pages_dir: Path = output_dir / "pages"
    diagrams_dir: Path = output_dir / "diagrams"
    combined_md: Path = pdf_path.parent / f"{pdf_path.stem}.md"
    archive_dir: Path = _next_archive_dir(output_dir)

    moved: int = 0

    def _move(src: Path, dest: Path) -> None:
        """Move one artifact into the archive if it exists, counting the move.

        Creates ``dest``'s parent directory as needed and increments the
        enclosing ``moved`` counter. A missing ``src`` is skipped silently.

        :param src: Source artifact path. Required.
        :type src: pathlib.Path
        :param dest: Destination path inside the archive. Required.
        :type dest: pathlib.Path
        :return: ``None``.
        :rtype: None
        """
        nonlocal moved
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dest)
            moved += 1

    for page_num in page_nums:
        stem: str = _page_stem(page_num, page_count)
        _move(pages_dir / f"{stem}.jpg", archive_dir / "pages" / f"{stem}.jpg")
        _move(pages_dir / f"{stem}.md", archive_dir / "pages" / f"{stem}.md")
        for diag in sorted(diagrams_dir.glob(f"{stem}_diagram_*.jpg")):
            _move(diag, archive_dir / "diagrams" / diag.name)

    _move(combined_md, archive_dir / combined_md.name)

    return archive_dir if moved else None


@dataclass
class _RunContext:
    """Resolved startup settings shared across a run's phases.

    :ivar pdf_path: Path to the validated source PDF.
    :vartype pdf_path: pathlib.Path
    :ivar config: Resolved application configuration with reachable instances.
    :vartype config: AppConfig
    :ivar dpi_scale: Page render scale factor.
    :vartype dpi_scale: float
    :ivar include_comments: Whether to append PDF annotations to each page.
    :vartype include_comments: bool
    :ivar include_links: Whether to rewrite plain anchor text as Markdown links.
    :vartype include_links: bool
    :ivar rerun_pages: Pages to reprocess, or ``None`` for a normal run.
    :vartype rerun_pages: set[int] | None
    :ivar output_dir: Per-document working directory.
    :vartype output_dir: pathlib.Path
    """

    pdf_path: Path
    config: AppConfig
    dpi_scale: float
    include_comments: bool
    include_links: bool
    rerun_pages: set[int] | None
    output_dir: Path


def _startup(argv: list[str]) -> tuple[_RunContext | None, int]:
    """Parse arguments, validate the PDF, load config, and probe instances.

    :param argv: Argument list excluding the program name (``sys.argv[1:]``).
        Required.
    :type argv: list[str]
    :return: ``(context, exit_code)``. On success ``context`` is a
        :class:`_RunContext` and ``exit_code`` is ``0``; on a help request or
        any startup failure ``context`` is ``None`` and ``exit_code`` is the
        code :func:`run` should return.
    :rtype: tuple[_RunContext | None, int]
    """
    if any(a in ("-h", "--help") for a in argv):
        print(_HELP)
        return None, 0

    pdf_arg, dpi_scale, include_comments, include_links, rerun_pages, arg_err = (
        _parse_args(argv)
    )
    if arg_err is not None:
        print(f"Error: {arg_err}\n{_USAGE}", file=sys.stderr)
        return None, 1
    if pdf_arg is None:
        print(_USAGE, file=sys.stderr)
        return None, 1

    pdf_path: Path = Path(pdf_arg)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        return None, 2
    if not pdf_path.is_file() or not os.access(pdf_path, os.R_OK):
        print(f"Error: file not readable: {pdf_path}", file=sys.stderr)
        return None, 3

    try:
        config: AppConfig = load_config(pdf_path)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Error: invalid ollama.json — {exc}", file=sys.stderr)
        return None, 4

    print(f"Render workers: {config.max_render_workers}")
    print(f"DPI scale: {dpi_scale}")
    print(f"Include comments: {include_comments}")
    print(f"Include links: {include_links}")
    print(f"Probing {len(config.instances)} Ollama instance(s)...")

    live: list[OllamaInstance] = probe_instances(config.instances)
    if not live:
        print("Error: no Ollama instances reachable", file=sys.stderr)
        return None, 4

    config = replace(config, instances=live)
    print(f"Ready — {len(live)} instance(s) reachable")

    output_dir: Path = pdf_path.parent / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    ctx: _RunContext = _RunContext(
        pdf_path=pdf_path, config=config, dpi_scale=dpi_scale,
        include_comments=include_comments, include_links=include_links,
        rerun_pages=rerun_pages, output_dir=output_dir,
    )
    return ctx, 0


def _apply_rerun(
    ctx: _RunContext, state: AppState, state_mgr: StateManager, page_count: int,
) -> int:
    """Archive prior artifacts and reset state for the ``--rerun-pages`` selection.

    :param ctx: Resolved startup context (supplies ``rerun_pages``). Required.
    :type ctx: _RunContext
    :param state: Loaded application state, reset in place for the chosen pages.
        Required.
    :type state: AppState
    :param state_mgr: State manager for atomic persistence. Required.
    :type state_mgr: StateManager
    :param page_count: Total page count, for range validation. Required.
    :type page_count: int
    :return: ``0`` on success, or ``1`` if no requested page is in range.
    :rtype: int
    """
    requested: set[int] = ctx.rerun_pages or set()
    in_range: list[int] = sorted(p for p in requested if 1 <= p <= page_count)
    for p in sorted(requested):
        if p < 1 or p > page_count:
            print(f"Warning: --rerun-pages: skipping out-of-range page {p}", file=sys.stderr)
    if not in_range:
        print("Error: --rerun-pages: no valid pages to rerun", file=sys.stderr)
        return 1
    archive_dir: Path | None = _archive_page_artifacts(
        ctx.output_dir, ctx.pdf_path, page_count, in_range,
    )
    if archive_dir is not None:
        print(f"Archived prior artifacts to {archive_dir}")
    state_mgr.reset_pages(state, in_range)
    print(f"Rerunning page(s): {', '.join(str(p) for p in in_range)}")
    return 0


def _prepare_state(
    ctx: _RunContext,
) -> tuple[tuple[AppState, StateManager, int] | None, int]:
    """Resolve the page count and load (or rerun-reset) the processing state.

    :param ctx: Resolved startup context. Required.
    :type ctx: _RunContext
    :return: ``(payload, exit_code)``. On success ``payload`` is
        ``(state, state_manager, page_count)`` and ``exit_code`` is ``0``; when
        the run is already complete or a failure occurs ``payload`` is ``None``
        and ``exit_code`` is the code :func:`run` should return.
    :rtype: tuple[tuple[AppState, StateManager, int] | None, int]
    """
    try:
        page_count: int = get_page_count(ctx.pdf_path)
    except PDF_ERRORS as exc:
        print(f"Error: cannot read PDF: {exc}", file=sys.stderr)
        return None, 3
    print(f"Pages: {page_count}")

    state_mgr: StateManager = StateManager(ctx.output_dir)
    if ctx.rerun_pages is not None and not state_mgr.path.is_file():
        print(
            "Error: --rerun-pages requires a previous run (no state.json found)",
            file=sys.stderr,
        )
        return None, 1

    try:
        state: AppState = state_mgr.load_or_init(ctx.pdf_path, page_count)
    except StateMismatchError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return None, 8

    if ctx.rerun_pages is not None:
        code: int = _apply_rerun(ctx, state, state_mgr, page_count)
        if code != 0:
            return None, code

    run_status: str = state_mgr.status(state)
    if run_status == "complete":
        md_path: Path = ctx.pdf_path.parent / f"{ctx.pdf_path.stem}.md"
        print(f"Already complete: {md_path}")
        return None, 0
    if run_status == "partial":
        print("Resuming from partial run")

    return (state, state_mgr, page_count), 0


def _run_phases(
    ctx: _RunContext, state: AppState, state_mgr: StateManager, page_count: int,
) -> int:
    """Run Phase 1 (render), Phase 2 (OCR) and Phase 3 (combine), then summarise.

    :param ctx: Resolved startup context. Required.
    :type ctx: _RunContext
    :param state: Loaded application state, mutated as phases progress. Required.
    :type state: AppState
    :param state_mgr: State manager for atomic persistence. Required.
    :type state_mgr: StateManager
    :param page_count: Total page count. Required.
    :type page_count: int
    :return: Exit code: ``5`` if all pages failed rendering, ``6`` if all
        rendered pages failed OCR, ``7`` if the combined write failed, else
        ``0``.
    :rtype: int
    """
    pending: list[int] = [
        i for i in range(1, page_count + 1)
        if not state.pages[str(i)].image_done
        and not state.pages[str(i)].image_failed
    ]

    if pending:
        pages_dir: Path = ctx.output_dir / "pages"
        print(
            f"Phase 1: rendering {len(pending)} page(s) "
            f"with {ctx.config.max_render_workers} worker(s)..."
        )
        results: list[tuple[int, bool, str]] = render_pages(
            ctx.pdf_path, pages_dir, page_count, pending,
            ctx.config.max_render_workers, ctx.dpi_scale,
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

    ocr_pending: list[int] = [
        i for i in range(1, page_count + 1)
        if state.pages[str(i)].image_done
        and not state.pages[str(i)].ocr_done
        and not state.pages[str(i)].ocr_failed
    ]

    if ocr_pending:
        print(
            f"Phase 2: OCR {len(ocr_pending)} page(s) "
            f"with {len(ctx.config.instances)} instance(s) "
            f"(timeout {ctx.config.ocr_timeout}s)..."
        )
        run_phase2(
            ctx.output_dir, page_count, ocr_pending, ctx.config.instances,
            state, state_mgr, ctx.config.ocr_timeout, ctx.pdf_path, ctx.dpi_scale,
            ctx.include_comments, ctx.include_links,
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

    print("Phase 3: combining per-page markdown...")
    ok, err = run_phase3(ctx.pdf_path, ctx.output_dir, page_count, state, state_mgr)
    if not ok:
        print(f"Error: output file write failed — {err}", file=sys.stderr)
        return 7

    _print_summary(ctx, state, page_count)
    return 0


def _print_summary(ctx: _RunContext, state: AppState, page_count: int) -> None:
    """Print the final per-phase success/failure counts for the run.

    :param ctx: Resolved startup context (supplies the output path). Required.
    :type ctx: _RunContext
    :param state: Final application state to tally. Required.
    :type state: AppState
    :param page_count: Total page count. Required.
    :type page_count: int
    :return: ``None``.
    :rtype: None
    """
    output_md: Path = ctx.pdf_path.parent / f"{ctx.pdf_path.stem}.md"
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


def run() -> int:
    """Validate arguments, load config, probe Ollama instances, and run all phases.

    Reads the PDF path and optional flags from ``sys.argv``, performs all
    startup checks before any processing begins (delegated to :func:`_startup`
    and :func:`_prepare_state`), then renders, OCRs, and combines the pages
    (:func:`_run_phases`).

    :return: Exit code indicating outcome:

        - ``0``: All phases succeeded (or the run was already complete).
        - ``1``: No PDF path supplied, or invalid command-line arguments.
        - ``2``: PDF file not found.
        - ``3``: PDF file exists but is not readable or cannot be opened.
        - ``4``: ollama.json is invalid, or no Ollama instances are reachable.
        - ``5``: All pages failed image rendering.
        - ``6``: All rendered pages failed OCR.
        - ``7``: Combined output file write failed.
        - ``8``: Existing state.json does not match the current PDF (different
          path or page count).
    :rtype: int
    """
    ctx, code = _startup(sys.argv[1:])
    if ctx is None:
        return code

    prepared, code = _prepare_state(ctx)
    if prepared is None:
        return code

    state, state_mgr, page_count = prepared
    return _run_phases(ctx, state, state_mgr, page_count)
