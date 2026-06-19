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

from pdf_extractor.fences import FENCE_RE, next_fence_state

# A line that must never be merged into a prose paragraph and always acts as a
# paragraph boundary.
_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]")
_LIST_RE = re.compile(
    r"^[ \t]*([-*+]|[0-9]+[.)]|[A-Za-z][.)]|[ivxlcdmIVXLCDM]+[.)])[ \t]+\S"
)
_TABLE_RE = re.compile(r"^[ \t]*\|")
_QUOTE_RE = re.compile(r"^[ \t]*>")
_HR_RE = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$")

# A paragraph fully wrapped in a single run of 1–3 ``*`` or ``_`` with no
# interior marker of the same kind — i.e. stray whole-paragraph emphasis.
_WRAP_RE = re.compile(r"^(\*{1,3}|_{1,3})(\S.*?\S|\S)\1$")


def _is_boundary(line: str) -> bool:
    """Return whether a line cannot be part of a reflowed prose paragraph.

    :param line: A single line of page Markdown. Required.
    :type line: str
    :return: ``True`` for a blank line, heading, list item, table row,
        blockquote, fence, or thematic break — anything that ends a paragraph.
    :rtype: bool
    """
    if not line.strip():
        return True
    return bool(
        _HEADING_RE.match(line)
        or _LIST_RE.match(line)
        or _TABLE_RE.match(line)
        or _QUOTE_RE.match(line)
        or FENCE_RE.match(line)
        or _HR_RE.match(line)
    )


def _strip_wrap_emphasis(text: str) -> str:
    """Strip emphasis markers that wrap an entire paragraph.

    A leading run of ``*`` or ``_`` (1–3 chars) and a matching trailing run are
    removed only when they enclose the whole string and the run's marker does
    not appear inside — so genuine inline emphasis (``a *b* c``) is preserved.

    :param text: A single prose line or joined paragraph. Required.
    :type text: str
    :return: ``text`` with whole-string wrapping emphasis removed, else ``text``
        unchanged.
    :rtype: str
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

    :param text: Per-page Markdown text from the OCR response. Required.
    :type text: str
    :return: Markdown with each prose paragraph on one line; structure
        preserved.
    :rtype: str
    """
    out: list[str] = []
    buf: list[str] = []
    fence: str | None = None  # opening fence run while inside a code block

    def _flush() -> None:
        """Emit the buffered paragraph (emphasis-stripped) and clear the buffer.

        :return: ``None``.
        :rtype: None
        """
        if buf:
            out.append(_strip_wrap_emphasis(" ".join(buf)))
            buf.clear()

    for line in text.split("\n"):
        new_fence, is_fence = next_fence_state(line, fence)
        if is_fence or fence is not None:
            # A fence delimiter ends the current paragraph; lines inside an open
            # block are emitted verbatim.
            if is_fence:
                _flush()
            fence = new_fence
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
