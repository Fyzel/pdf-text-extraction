"""Prose reflow and stray-emphasis stripping for per-page OCR output.

The vision model (Phase 2) tends to mirror the PDF's *visual* line wrapping —
emitting a hard newline at the end of every rendered line — so a single
paragraph arrives as several short lines. It also sometimes wraps a whole
paragraph in emphasis markers (``*…*``) that are not real emphasis. Both are
formatting artifacts that diverge from clean Markdown (issue #63).

This module is deterministic and side-effect free. ``reflow_prose`` joins the
soft-wrapped lines of each prose paragraph back into a single line and strips
emphasis that spans an entire paragraph. It leaves structural lines untouched:
blank lines (paragraph separators), headings, list items, table rows,
blockquotes, thematic breaks, and anything inside a fenced code block.
"""
import re

# A line that must never be merged into a prose paragraph and always acts as a
# paragraph boundary.
_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]")
_LIST_RE = re.compile(
    r"^[ \t]*([-*+]|[0-9]+[.)]|[A-Za-z][.)]|[ivxlcdmIVXLCDM]+[.)])[ \t]+\S"
)
_TABLE_RE = re.compile(r"^[ \t]*\|")
_QUOTE_RE = re.compile(r"^[ \t]*>")
_FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_HR_RE = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$")

# A paragraph fully wrapped in a single run of 1–3 ``*`` or ``_`` with no
# interior marker of the same kind — i.e. stray whole-paragraph emphasis.
_WRAP_RE = re.compile(r"^(\*{1,3}|_{1,3})(\S.*?\S|\S)\1$")


def _closes_fence(run: str, line: str, opening: str) -> bool:
    """Return True if ``line`` is a valid CommonMark close for ``opening``.

    A closing fence uses the same character as the opening run, is at least as
    long, and carries no info string (only the fence and optional surrounding
    whitespace).

    Args:
        run: The fence run captured from ``line`` (group 1 of :data:`_FENCE_RE`).
        line: Candidate line, already known to match :data:`_FENCE_RE`.
        opening: The opening fence run (e.g. ```` ``` ```` or ``~~~~``).
    """
    return run[0] == opening[0] and len(run) >= len(opening) and line.strip() == run


def _is_boundary(line: str) -> bool:
    """Return True if a line cannot be part of a reflowed prose paragraph."""
    if not line.strip():
        return True
    return bool(
        _HEADING_RE.match(line)
        or _LIST_RE.match(line)
        or _TABLE_RE.match(line)
        or _QUOTE_RE.match(line)
        or _FENCE_RE.match(line)
        or _HR_RE.match(line)
    )


def _strip_wrap_emphasis(text: str) -> str:
    """Strip emphasis markers that wrap an entire paragraph.

    A leading run of ``*`` or ``_`` (1–3 chars) and a matching trailing run are
    removed only when they enclose the whole string and the run's marker does
    not appear inside — so genuine inline emphasis (``a *b* c``) is preserved.
    """
    match = _WRAP_RE.match(text)
    if match is None:
        return text
    marker, inner = match.group(1), match.group(2)
    if marker[0] in inner:
        return text
    return inner


def reflow_prose(text: str) -> str:
    """Join soft-wrapped prose lines and strip whole-paragraph emphasis.

    Consecutive non-structural lines are merged into a single line (joined by a
    space), which is how a Markdown paragraph renders anyway. Blank lines,
    headings, list items, table rows, blockquotes, thematic breaks, and fenced
    code blocks are emitted verbatim and act as paragraph boundaries. Fences are
    tracked per CommonMark: a block opened by a run of three or more backticks
    (or tildes) is closed only by a run of the same character that is at least
    as long and carries no info string, so a different or shorter marker inside
    the block does not close it early and re-enable reflow.

    Args:
        text: Per-page Markdown text from the OCR response.

    Returns:
        Markdown with each prose paragraph on one line; structure preserved.
    """
    out: list[str] = []
    buf: list[str] = []
    fence: str | None = None  # opening fence run while inside a code block

    def _flush() -> None:
        if buf:
            out.append(_strip_wrap_emphasis(" ".join(buf)))
            buf.clear()

    for line in text.split("\n"):
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            _flush()
            run: str = fence_match.group(1)
            if fence is None:
                fence = run
            elif _closes_fence(run, line, fence):
                fence = None
            out.append(line)
            continue
        if fence is not None:
            out.append(line)
            continue
        if _is_boundary(line):
            _flush()
            out.append(line)
            continue
        # Strip per-line first (catches a sentence wrapped on its own line);
        # _flush strips again on the joined paragraph (catches a wrap that spans
        # several lines, with markers only on the first and last).
        buf.append(_strip_wrap_emphasis(line.strip()))

    _flush()
    return "\n".join(out)
