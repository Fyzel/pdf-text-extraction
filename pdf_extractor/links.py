"""Hyperlink extraction for per-page OCR output.

Phase 2 transcribes a *rendered* page image, so it captures a hyperlink's
visible anchor text but never its target URL — the URI lives only in the PDF's
link objects, not in the rendered pixels. This module reads a page's external
(URI) links via PyMuPDF ``page.get_links()``, recovers each link's anchor text
from the link rectangle, and splices Markdown links (``[text](uri)``) over the
matching plain text in the page's Markdown.

Only external URI links (``LINK_URI`` — http/https/mailto/…) are handled.
Internal jumps (``LINK_GOTO``) are skipped: per-page Markdown files have no
stable cross-page anchor targets.

Extraction is opt-in (CLI ``--include-links``); when disabled the page text is
left exactly as before.
"""
import re

import fitz

from pdf_extractor.pdf_errors import PDF_ERRORS

# Existing Markdown link spans and inline-code spans whose contents must not be
# re-linked. Used to carve a line into "protected" and "plain" segments.
_PROTECTED = re.compile(r"\[[^\]]*\]\([^)]*\)|`[^`]*`")


def _escape_anchor(text: str) -> str:
    """Escape characters that would break a Markdown link's ``[...]`` text.

    Backslashes are escaped first so the escapes added for ``[`` and ``]`` are
    not themselves doubled.

    :param text: Raw anchor text. Required.
    :type text: str
    :return: Anchor text safe to place inside ``[...]``.
    :rtype: str
    """
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _escape_uri(uri: str) -> str:
    """Escape characters that would break a Markdown link's ``(...)`` target.

    Unbalanced parentheses (common in e.g. Wikipedia URLs) would otherwise
    truncate the destination; CommonMark renders ``\\(`` / ``\\)`` as literal
    parentheses, keeping the URL intact.

    :param uri: Raw link target. Required.
    :type uri: str
    :return: URI safe to place inside ``(...)``.
    :rtype: str
    """
    return uri.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def extract_links(pdf_path: str, page_num: int) -> list[tuple[str, str]]:
    """Extract a page's external URI links as ``(anchor_text, uri)`` pairs.

    Links are returned in reading order (top-to-bottom, then left-to-right) so a
    later left-to-right splice consumes them in the order they appear in prose.

    :param pdf_path: Path to the source PDF file. Required.
    :type pdf_path: str
    :param page_num: 1-based page number. Required.
    :type page_num: int
    :return: ``(anchor_text, uri)`` for each external URI link whose rectangle
        covers non-empty text; empty list when the page has no such links or
        cannot be read.
    :rtype: list[tuple[str, str]]
    """
    items: list[tuple[float, float, str, str]] = []
    try:
        doc: fitz.Document = fitz.open(pdf_path)
        try:
            page: fitz.Page = doc[page_num - 1]
            for link in page.get_links():
                if link.get("kind") != fitz.LINK_URI:
                    continue
                uri: str = link.get("uri") or ""
                rect = link.get("from")
                if not uri or rect is None:
                    continue
                anchor: str = re.sub(r"\s+", " ", page.get_textbox(rect)).strip()
                if not anchor:
                    continue
                items.append((rect.y0, rect.x0, anchor, uri))
        finally:
            doc.close()
    except PDF_ERRORS:
        # never let link extraction fail a page
        return []

    items.sort(key=lambda it: (round(it[0]), round(it[1])))
    return [(anchor, uri) for _, _, anchor, uri in items]


def _splice_plain(segment: str, pending: list[tuple[str, str]]) -> str:
    """Link the first unlinked occurrence of each pending anchor in a segment.

    Anchors are consumed left-to-right: each match advances a cursor past the
    inserted link so an identical anchor text later in the segment (or the
    inserted link's own text) is never re-matched. Consumed pairs are removed
    from ``pending`` in place so they are not reused on later segments/lines.

    :param segment: A run of plain (non-link, non-code) Markdown text. Required.
    :type segment: str
    :param pending: Remaining ``(anchor, uri)`` pairs in reading order.
        Required; mutated in place as pairs are consumed.
    :type pending: list[tuple[str, str]]
    :return: The segment with matched anchors rewritten as ``[anchor](uri)``.
    :rtype: str
    """
    out: list[str] = []
    cursor: int = 0
    while pending:
        best_idx: int | None = None
        best_pos: int | None = None
        for k, (anchor, _uri) in enumerate(pending):
            pos: int = segment.find(anchor, cursor)
            if pos != -1 and (best_pos is None or pos < best_pos):
                best_pos = pos
                best_idx = k
        if best_idx is None:
            break
        anchor, uri = pending.pop(best_idx)
        out.append(segment[cursor:best_pos])
        out.append(f"[{_escape_anchor(anchor)}]({_escape_uri(uri)})")
        cursor = best_pos + len(anchor)
    out.append(segment[cursor:])
    return "".join(out)


def splice_links(text: str, links: list[tuple[str, str]]) -> str:
    """Rewrite plain anchor text in page Markdown as Markdown links.

    Each extracted link is matched to the next unlinked occurrence of its anchor
    text, in reading order. Fenced code blocks, Markdown table rows, existing
    Markdown links, and inline-code spans are left untouched. A link whose anchor
    text is not found in the page text is dropped (no synthetic text is added).

    :param text: Per-page Markdown text (after reflow, list, and table
        processing). Required.
    :type text: str
    :param links: ``(anchor, uri)`` pairs from :func:`extract_links`, in reading
        order. Required (may be empty).
    :type links: list[tuple[str, str]]
    :return: Page text with anchor text rewritten as links; unchanged if
        ``links`` is empty.
    :rtype: str
    """
    if not links:
        return text

    pending: list[tuple[str, str]] = list(links)
    lines: list[str] = text.split("\n")
    out: list[str] = []
    fenced: bool = False
    for line in lines:
        stripped: str = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fenced = not fenced
            out.append(line)
            continue
        if fenced or stripped.startswith("|") or not pending:
            out.append(line)
            continue
        # Carve the line into protected (link/code) and plain segments; only the
        # plain segments are eligible for linking.
        parts: list[str] = []
        last: int = 0
        for m in _PROTECTED.finditer(line):
            parts.append(_splice_plain(line[last:m.start()], pending))
            parts.append(m.group(0))
            last = m.end()
        parts.append(_splice_plain(line[last:], pending))
        out.append("".join(parts))
    return "\n".join(out)
