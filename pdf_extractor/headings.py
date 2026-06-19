"""Heading-level correction for per-page OCR output, driven by PDF structure.

The vision model (Phase 2) guesses heading levels from visual appearance and
gets them wrong: it flattens distinct levels to ``#``, or promotes an ordinary
body sentence to a heading (issue #63). The true hierarchy is recoverable from a
digital PDF: heading runs use a larger font than body text, and the distinct
heading sizes rank into levels.

This module reads the span font sizes via PyMuPDF. ``extract_heading_scale``
builds the document-wide size→level ranking once (so a level is consistent
across pages); ``fix_headings`` then, for one page, relevels model headings that
match a real PDF heading, promotes body lines that are actually headings, and
demotes model headings the PDF does not back. Scanned/image-only PDFs yield no
sizes, so the model's own heading markup is left untouched as a fallback.
"""
import collections
import re
from difflib import SequenceMatcher

import fitz

from pdf_extractor.fences import next_fence_state
from pdf_extractor.pdf_errors import PDF_ERRORS

# A heading's font must exceed body size by this ratio. 1.15 cleanly separates
# real headings (≈1.27× body and up) from near-body noise like a 12pt "Page N
# of M" footer over 11pt body (1.09×).
_HEADING_RATIO: float = 1.15
# Minimum normalised-text similarity to treat a model line and a PDF heading as
# the same heading (tolerates minor OCR drift in the heading text).
_MATCH_RATIO: float = 0.85
_MAX_LEVEL: int = 6

_HEADING_RE = re.compile(r"^[ \t]*#{1,6}[ \t]+")
_MARKER_RE = re.compile(r"[#*_`]+")
_WS_RE = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Normalise heading text for comparison: drop markers, case, punctuation.

    :param text: Raw heading or candidate line text. Required.
    :type text: str
    :return: Lower-cased text with Markdown markers, runs of whitespace, and
        edge punctuation removed.
    :rtype: str
    """
    stripped: str = _MARKER_RE.sub("", text)
    stripped = _WS_RE.sub(" ", stripped).strip().lower()
    return stripped.strip(".,:;!?-—–")


def _clean_text(line: str) -> str:
    """Return a line's visible text with leading ``#`` and wrapping emphasis removed.

    :param line: A single Markdown line. Required.
    :type line: str
    :return: The visible text with any heading marker prefix and whole-line
        wrapping emphasis stripped.
    :rtype: str
    """
    text: str = _HEADING_RE.sub("", line).strip()
    match = re.match(r"^(\*{1,3}|_{1,3})(.+?)\1$", text)
    if match and match.group(1)[0] not in match.group(2):
        text = match.group(2)
    return text.strip()


def _line_size_text(line: dict) -> tuple[float, str]:
    """Return a text line's dominant span size (by char count) and joined text.

    :param line: A PyMuPDF text-line dict (a ``lines`` entry from
        ``page.get_text("dict")``). Required.
    :type line: dict
    :return: ``(dominant_size, joined_text)`` where ``dominant_size`` is the
        rounded span size covering the most characters (``0.0`` if the line has
        no text), and ``joined_text`` is the line's concatenated span text.
    :rtype: tuple[float, str]
    """
    sizes: collections.Counter = collections.Counter()
    parts: list[str] = []
    for span in line.get("spans", []):
        txt: str = span["text"]
        parts.append(txt)
        if txt.strip():
            sizes[round(span["size"], 1)] += len(txt.strip())
    joined: str = "".join(parts).strip()
    if not sizes:
        return 0.0, joined
    return sizes.most_common(1)[0][0], joined


def _doc_span_sizes(doc: fitz.Document) -> collections.Counter:
    """Tally span font sizes (rounded) by character count across all pages.

    :param doc: An open PyMuPDF document. Required.
    :type doc: fitz.Document
    :return: Counter mapping each rounded span size to the total number of
        non-whitespace characters rendered at that size.
    :rtype: collections.Counter
    """
    sizes: collections.Counter = collections.Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    txt: str = span["text"].strip()
                    if txt:
                        sizes[round(span["size"], 1)] += len(txt)
    return sizes


def extract_heading_scale(pdf_path: str) -> list[float]:
    """Build the document-wide heading size ranking, largest first.

    Body size is the most common span size across the whole document (by
    character count). Heading sizes are the distinct span sizes at least
    ``_HEADING_RATIO`` times the body size, sorted descending so the index of a
    size is its level (0 → ``#``, 1 → ``##``, …).

    :param pdf_path: Path to the source PDF file. Required.
    :type pdf_path: str
    :return: Heading sizes in descending order; empty if the PDF has no
        extractable text (e.g. scanned/image-only) or no text larger than body.
    :rtype: list[float]
    """
    doc: fitz.Document | None = None
    try:
        doc = fitz.open(pdf_path)
        sizes: collections.Counter = _doc_span_sizes(doc)
        if not sizes:
            return []
        body: float = sizes.most_common(1)[0][0]
        heading_sizes = {s for s in sizes if s >= body * _HEADING_RATIO}
        return sorted(heading_sizes, reverse=True)
    except PDF_ERRORS:
        # never let heading extraction fail a page
        return []
    finally:
        if doc is not None:
            try:
                doc.close()
            except PDF_ERRORS:
                # cleanup must not break the guarantee
                pass


def _page_headings(pdf_path: str, page_num: int, scale: list[float]) -> list[tuple[str, int]]:
    """Return ``(normalised_text, level)`` for each heading line on a page.

    :param pdf_path: Path to the source PDF file. Required.
    :type pdf_path: str
    :param page_num: 1-based page number. Required.
    :type page_num: int
    :param scale: Document-wide heading size ranking from
        :func:`extract_heading_scale`. Required.
    :type scale: list[float]
    :return: ``(normalised_text, level)`` for each line whose dominant size is
        in ``scale``; empty if the page cannot be read.
    :rtype: list[tuple[str, int]]
    """
    doc: fitz.Document | None = None
    try:
        doc = fitz.open(pdf_path)
        page: fitz.Page = doc[page_num - 1]
        result: list[tuple[str, int]] = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                size, text = _line_size_text(line)
                norm: str = _norm(text)
                if norm and size in scale:
                    level: int = min(scale.index(size) + 1, _MAX_LEVEL)
                    result.append((norm, level))
        return result
    except PDF_ERRORS:
        # never let heading extraction fail a page
        return []
    finally:
        if doc is not None:
            try:
                doc.close()
            except PDF_ERRORS:
                # cleanup must not break the guarantee
                pass


def _match_level(candidate: str, headings: list[tuple[str, int]]) -> int | None:
    """Return the level of the best-matching PDF heading, or None if none match.

    :param candidate: Normalised candidate line text (from :func:`_norm`).
        Required.
    :type candidate: str
    :param headings: ``(normalised_text, level)`` pairs for the page's PDF
        headings. Required.
    :type headings: list[tuple[str, int]]
    :return: The level of the heading whose similarity to ``candidate`` is
        highest and at least ``_MATCH_RATIO``, or ``None`` if none qualify.
    :rtype: int | None
    """
    best_level: int | None = None
    best_ratio: float = _MATCH_RATIO
    for norm, level in headings:
        ratio: float = SequenceMatcher(None, candidate, norm).ratio()
        if ratio >= best_ratio:
            best_ratio = ratio
            best_level = level
    return best_level


def fix_headings(text: str, pdf_path: str | None, page_num: int, scale: list[float]) -> str:
    """Correct heading levels in page text using the PDF's font hierarchy.

    For each line: a model heading or a plain line matching a PDF heading is
    rewritten at the PDF-derived level; a model heading that matches no PDF
    heading is demoted to plain text. Non-heading prose is left unchanged.

    Lines inside a fenced code block are emitted verbatim, so a ``#`` comment or
    heading-like line in code is never rewritten. Fences are tracked per
    CommonMark: a block opened by a run of three or more backticks (or tildes)
    is closed only by a run of the same character that is at least as long and
    carries no info string, so a different or shorter marker inside the block
    does not close it early.

    When the document has no heading scale (``scale`` empty) or no source PDF,
    the text is returned untouched so the model's own markup stands as a
    fallback.

    :param text: Per-page Markdown text from the OCR response. Required.
    :type text: str
    :param pdf_path: Path to the source PDF. Required, but may be ``None`` when
        unavailable (the text is then returned untouched).
    :type pdf_path: str | None
    :param page_num: 1-based page number. Required.
    :type page_num: int
    :param scale: Document-wide heading size ranking from
        :func:`extract_heading_scale`. Required (may be empty).
    :type scale: list[float]
    :return: Page text with corrected heading markup.
    :rtype: str
    """
    if pdf_path is None or not scale:
        return text

    headings: list[tuple[str, int]] = _page_headings(pdf_path, page_num, scale)

    out: list[str] = []
    fence: str | None = None  # opening fence run while inside a code block
    for line in text.split("\n"):
        new_fence, is_fence = next_fence_state(line, fence)
        if is_fence or fence is not None:
            # A fence delimiter, or any line inside an open block: emit verbatim.
            fence = new_fence
            out.append(line)
            continue
        is_heading: bool = bool(_HEADING_RE.match(line))
        stripped: str = line.strip()
        if not stripped:
            out.append(line)
            continue

        candidate: str = _norm(line)
        level: int | None = _match_level(candidate, headings) if candidate else None

        if level is not None:
            out.append(f"{'#' * level} {_clean_text(line)}")
        elif is_heading:
            # Model marked a heading the PDF does not back — demote to prose.
            out.append(_clean_text(line))
        else:
            out.append(line)

    return "\n".join(out)
