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
    """Return True if the marker is an unordered-list bullet (``-``/``*``/``+``)."""
    return marker in _UNORDERED


def normalize_markdown(text: str) -> str:
    """Normalise list markers and ordered-list numbering in markdown text.

    Nesting is inferred from leading indentation. Within each list level,
    unordered items are rewritten to ``-`` and ordered items (numeric,
    alphabetic, or roman) are renumbered sequentially as ``1.``, ``2.``, …
    The original indentation string is preserved, so a sub-item indented under
    an ordered parent keeps its indent but gains a valid numeric marker.

    Args:
        text: Per-page markdown text from the OCR response.

    Returns:
        Markdown with normalised list markers; non-list content is unchanged.
    """
    out: list[str] = []
    # Stack of open list levels, innermost last. Each entry: [indent_width,
    # ordered(bool), counter].
    stack: list[list] = []

    for line in text.split("\n"):
        match = _ITEM_RE.match(line)
        if match is None:
            # A blank line may separate items in the same list, so keep the
            # stack alive; any non-blank line at column 0 ends the open lists.
            if line[:1] not in ("", " ", "\t"):
                stack.clear()
            out.append(line)
            continue

        indent: str = match.group("indent")
        indent_w: int = len(indent.expandtabs(4))
        marker: str = match.group("marker")
        content: str = match.group("content")
        ordered: bool = not _is_unordered(marker)

        # Drop any deeper levels; a shallower-or-equal indent closes them.
        while stack and stack[-1][0] > indent_w:
            stack.pop()

        if stack and stack[-1][0] == indent_w:
            level = stack[-1]
            if level[1] != ordered:  # marker type changed at this indent
                level[1] = ordered
                level[2] = 0
            level[2] += 1
        else:
            level = [indent_w, ordered, 1]
            stack.append(level)

        new_marker: str = f"{level[2]}." if ordered else "-"
        out.append(f"{indent}{new_marker} {content}")

    return "\n".join(out)
