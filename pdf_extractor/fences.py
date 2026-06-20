"""Shared CommonMark fenced-code-block tracking helpers.

Both :mod:`pdf_extractor.headings` and :mod:`pdf_extractor.reflow` walk per-page
Markdown line by line and must leave fenced code blocks untouched. They share the
same CommonMark fence rules — an opening run of three or more backticks (or
tildes) is closed only by a run of the same character that is at least as long
and carries no info string. This module holds that logic once so the two callers
stay consistent.
"""
import re

#: Matches a line whose first non-space content is a fence run (group 1 is the
#: run of three-or-more backticks or tildes).
FENCE_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


def closes_fence(run: str, line: str, opening: str) -> bool:
    """Return whether ``line`` is a valid CommonMark close for ``opening``.

    A closing fence uses the same character as the opening run, is at least as
    long, and carries no info string (only the fence and optional surrounding
    whitespace).

    :param run: The fence run captured from ``line`` (group 1 of
        :data:`FENCE_RE`). Required.
    :type run: str
    :param line: Candidate line, already known to match :data:`FENCE_RE`.
        Required.
    :type line: str
    :param opening: The opening fence run (e.g. ```` ``` ```` or ``~~~~``).
        Required.
    :type opening: str
    :return: ``True`` if ``line`` validly closes the ``opening`` fence.
    :rtype: bool
    """
    return run[0] == opening[0] and len(run) >= len(opening) and line.strip() == run


def next_fence_state(line: str, fence: str | None) -> tuple[str | None, bool]:
    """Advance the open-fence state for one line of Markdown.

    Given the currently open fence run (or ``None`` when outside any block),
    decide the state after ``line``: a fence line outside a block opens one, a
    valid close (see :func:`closes_fence`) ends the open block, and any other
    fence line leaves the state unchanged. Non-fence lines never change state.

    :param line: A single line of page Markdown. Required.
    :type line: str
    :param fence: The opening fence run while inside a code block, or ``None``
        when outside any block. Required.
    :type fence: str | None
    :return: ``(new_fence, is_fence_line)`` — the updated open-fence run (or
        ``None``) and whether ``line`` was itself a fence delimiter.
    :rtype: tuple[str | None, bool]
    """
    match = FENCE_RE.match(line)
    if match is None:
        return fence, False
    run: str = match.group(1)
    if fence is None:
        return run, True
    if closes_fence(run, line, fence):
        return None, True
    return fence, True
