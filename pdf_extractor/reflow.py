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

# A table-of-contents / index line: visible text followed by a trailing page
# reference — an arabic page number or a lowercase front-matter roman numeral
# (i…lxxxix). These are complete one-per-line entries, not soft-wrapped prose,
# so reflow must not join them into a running paragraph (issue #85). The roman
# form is deliberately narrow (no m/c/d, length ≥ 2) to avoid matching ordinary
# words like "did" or "mix" or a sentence ending in the pronoun "I".
_ROMAN_PAGE_RE = re.compile(r"^l?x{0,3}(?:ix|iv|v?i{0,3})$")
_INDEX_TAIL_RE = re.compile(r"^[ \t]*\S.*\s(\S+?)[ \t]*$")
# Leading dot leaders the model often glues to a page number ("...89", "…89").
_LEADER_RE = re.compile(r"^[.…]+")

# A run-on table of contents: the model sometimes transcribes a whole contents
# page as one line, "Foreword xv Preface xvii Chapter 1 3 …". reflow only joins
# lines, never splits, so such a line stays a single paragraph (issue #85). A
# run is split back into one entry per line only when it cleanly partitions into
# at least this many short entries that each end in a page reference — a guard
# that keeps ordinary prose (which rarely tiles into number-terminated segments)
# untouched.
_MIN_RUN_ENTRIES: int = 3
_MAX_ENTRY_WORDS: int = 14

# A paragraph fully wrapped in a single run of 1–3 ``*`` or ``_`` with no
# interior marker of the same kind — i.e. stray whole-paragraph emphasis.
_WRAP_RE = re.compile(r"^(\*{1,3}|_{1,3})(\S.*?\S|\S)\1$")


def _is_page_ref(token: str) -> bool:
    """Return whether a token is a page reference (arabic or roman numeral).

    Any leading dot leaders or ellipsis the model glues on (``...89``, ``…89``)
    are stripped before the check.

    :param token: A single whitespace-delimited token. Required.
    :type token: str
    :return: ``True`` for 1–4 arabic digits or a short lowercase roman numeral
        (length ≥ 2, no hundreds/thousands), else ``False``.
    :rtype: bool
    """
    core: str = _LEADER_RE.sub("", token)
    if core.isdigit():
        return len(core) <= 4
    return len(core) >= 2 and bool(_ROMAN_PAGE_RE.match(core))


def _is_index_entry(line: str) -> bool:
    """Return whether a line is a table-of-contents / index entry.

    An index entry is visible text followed by a trailing page reference — an
    arabic page number or a short lowercase roman numeral (front matter). Such
    lines are complete on their own and must not be reflowed into the next line.

    :param line: A single line of page Markdown. Required.
    :type line: str
    :return: ``True`` if the line ends with a page-reference token.
    :rtype: bool
    """
    match = _INDEX_TAIL_RE.match(line)
    if match is None:
        return False
    return _is_page_ref(match.group(1))


def _split_index_run(line: str) -> list[str]:
    """Split a run-on table-of-contents line into one entry per line.

    The model sometimes transcribes a whole contents page as a single line. Each
    entry ends in a page reference, so the line is partitioned greedily: words
    accumulate until a page-reference token closes an entry. The split is applied
    only when the line tiles cleanly into at least :data:`_MIN_RUN_ENTRIES`
    short entries with no leftover words — otherwise the line is returned
    unchanged, so ordinary prose is never broken up.

    :param line: A single line of page Markdown. Required.
    :type line: str
    :return: The entries as separate lines, or ``[line]`` if it is not a run.
    :rtype: list[str]
    """
    words: list[str] = line.split()
    entries: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if _is_page_ref(word):
            entries.append(" ".join(current))
            current = []
    if current:
        # Trailing words with no page reference — not a clean contents run.
        return [line]
    if len(entries) < _MIN_RUN_ENTRIES:
        return [line]
    if any(len(entry.split()) > _MAX_ENTRY_WORDS for entry in entries):
        return [line]
    return entries


def _is_boundary(line: str) -> bool:
    """Return whether a line cannot be part of a reflowed prose paragraph.

    :param line: A single line of page Markdown. Required.
    :type line: str
    :return: ``True`` for a blank line, heading, list item, table row,
        blockquote, fence, thematic break, or table-of-contents entry — anything
        that ends a paragraph.
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
        or _is_index_entry(line)
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

    Table-of-contents and index entries (a line ending in a page reference) are
    kept on their own line, and a whole contents page that the model collapsed
    into one run-on line is split back into one entry per line.

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

        When the model split each contents entry's title and page number onto
        their own lines, the buffer joins them into one run-on line; this is
        split back into one entry per line before emitting.

        :return: ``None``.
        :rtype: None
        """
        if buf:
            joined: str = _strip_wrap_emphasis(" ".join(buf))
            out.extend(_split_index_run(joined))
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
        pieces = _split_index_run(line)
        if len(pieces) > 1:
            # A run-on table of contents: emit each entry on its own line.
            _flush()
            out.extend(pieces)
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
