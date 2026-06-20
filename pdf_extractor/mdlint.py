"""Markdown list normalisation for per-page OCR output.

The vision model (Phase 2) sometimes emits malformed list markup that does not
render as a list under CommonMark — most often a nested item that uses a stray
alphabetic or roman marker (``A.``, ``i.``) under a numeric ordered parent, or
inconsistent ordered numbering. This module normalises list markers and
sequential numbering while preserving the model's indentation, so sub-bullets
become valid nested lists.

The normaliser is deterministic and side-effect free; it touches only lines that
parse as list items and leaves prose, tables, headings, and blank lines untouched.
"""
import re
from dataclasses import dataclass

# A list item: optional indent, a bullet/ordered marker, whitespace, then content.
# Ordered markers cover numeric (``1.`` / ``1)``), single-letter alphabetic
# (``A.`` / ``a)``), and roman (``iv.`` / ``IX)``) forms.
_ITEM_RE = re.compile(
    r"^(?P<indent>[ \t]*)"
    r"(?P<marker>[-*+]|[0-9]+[.)]|[A-Za-z][.)]|[ivxlcdmIVXLCDM]+[.)])"
    r"[ \t]+"
    r"(?P<content>\S.*)$"
)

_UNORDERED: frozenset[str] = frozenset({"-", "*", "+"})


def _is_unordered(marker: str) -> bool:
    """Return whether a marker is an unordered-list bullet (``-``/``*``/``+``).

    :param marker: List marker captured from a line. Required.
    :type marker: str
    :return: ``True`` if ``marker`` is ``-``, ``*``, or ``+``.
    :rtype: bool
    """
    return marker in _UNORDERED


@dataclass
class _Level:
    """One open list level on the nesting stack.

    :ivar raw_indent: Source indentation width, used only to detect nesting
        depth relative to other items.
    :vartype raw_indent: int
    :ivar out_indent: Normalised indentation emitted for items at this level.
    :vartype out_indent: str
    :ivar ordered: Whether this level is an ordered list.
    :vartype ordered: bool
    :ivar counter: Running 1-based item number for ordered levels.
    :vartype counter: int
    :ivar marker_len: Rendered width of the most recent marker at this level
        (e.g. 2 for ``5.``), used to align any child level's indentation.
    :vartype marker_len: int
    """

    raw_indent: int
    out_indent: str
    ordered: bool
    counter: int = 0
    marker_len: int = 1


def normalize_markdown(text: str) -> str:
    """Normalise list markers, numbering, and nesting indentation in markdown.

    Nesting depth is inferred from the *relative* leading indentation of items.
    Within each level, unordered items are rewritten to ``-`` and ordered items
    (numeric, alphabetic, or roman) are renumbered sequentially as ``1.``,
    ``2.``, … Child levels are re-indented to align with the parent item's
    content column (marker width plus one space), which is what CommonMark
    requires for a sub-item to render as a nested list. This corrects both stray
    markers (``A.`` under an ordered parent) and under-indented sub-items
    (``  1.`` where three spaces are needed under ``5. ``).

    :param text: Per-page markdown text from the OCR response. Required.
    :type text: str
    :return: Markdown with normalised list markup; non-list content is
        unchanged.
    :rtype: str
    """
    out: list[str] = []
    stack: list[_Level] = []

    for line in text.split("\n"):
        match = _ITEM_RE.match(line)
        if match is None:
            # A blank line may separate items in the same list, so keep the
            # stack alive; any non-blank line at column 0 ends the open lists.
            if line[:1] not in ("", " ", "\t"):
                stack.clear()
            out.append(line)
            continue

        raw_w: int = len(match.group("indent").expandtabs(4))
        marker: str = match.group("marker")
        content: str = match.group("content")
        ordered: bool = not _is_unordered(marker)

        # Close any deeper levels; a shallower-or-equal indent ends them.
        while stack and stack[-1].raw_indent > raw_w:
            stack.pop()

        if stack and stack[-1].raw_indent == raw_w:
            level = stack[-1]
            if level.ordered != ordered:  # marker type changed at this indent
                level.ordered = ordered
                level.counter = 0
        elif stack and raw_w > stack[-1].raw_indent:
            parent = stack[-1]
            child_indent: str = parent.out_indent + " " * (parent.marker_len + 1)
            level = _Level(raw_w, child_indent, ordered)
            stack.append(level)
        else:
            # First level, or an indented root item with no open parent.
            level = _Level(raw_w, "", ordered)
            stack.append(level)

        level.counter += 1
        new_marker: str = f"{level.counter}." if ordered else "-"
        level.marker_len = len(new_marker)
        out.append(f"{level.out_indent}{new_marker} {content}")

    return "\n".join(out)
