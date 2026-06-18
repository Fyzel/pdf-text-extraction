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
    """Normalise heading text for comparison: drop markers, case, punctuation."""
    stripped: str = _MARKER_RE.sub("", text)
    stripped = _WS_RE.sub(" ", stripped).strip().lower()
    return stripped.strip(".,:;!?-—–")


def _clean_text(line: str) -> str:
    """Return a line's visible text with leading ``#`` and wrapping emphasis removed."""
    text: str = _HEADING_RE.sub("", line).strip()
    match = re.match(r"^(\*{1,3}|_{1,3})(.+?)\1$", text)
    if match and match.group(1)[0] not in match.group(2):
        text = match.group(2)
    return text.strip()


def _line_size_text(line: dict) -> tuple[float, str]:
    """Return a text line's dominant span size (by char count) and joined text."""
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
    """Tally span font sizes (rounded) by character count across all pages."""
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

    Args:
        pdf_path: Path to the source PDF file.

    Returns:
        Heading sizes in descending order. Empty if the PDF has no extractable
        text (e.g. scanned/image-only) or no text larger than body.
    """
    doc: fitz.Document = fitz.open(pdf_path)
    try:
        sizes: collections.Counter = _doc_span_sizes(doc)
        if not sizes:
            return []
        body: float = sizes.most_common(1)[0][0]
        heading_sizes = {s for s in sizes if s >= body * _HEADING_RATIO}
        return sorted(heading_sizes, reverse=True)
    except Exception:  # noqa: BLE001 — never let heading extraction fail a page
        return []
    finally:
        doc.close()


def _page_headings(pdf_path: str, page_num: int, scale: list[float]) -> list[tuple[str, int]]:
    """Return ``(normalised_text, level)`` for each heading line on a page."""
    doc: fitz.Document = fitz.open(pdf_path)
    try:
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
    except Exception:  # noqa: BLE001
        return []
    finally:
        doc.close()


def _match_level(candidate: str, headings: list[tuple[str, int]]) -> int | None:
    """Return the level of the best-matching PDF heading, or None if none match."""
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
    heading is demoted to plain text. Non-heading prose is left unchanged. When
    the document has no heading scale (``scale`` empty) or no source PDF, the
    text is returned untouched so the model's own markup stands as a fallback.

    Args:
        text: Per-page Markdown text from the OCR response.
        pdf_path: Path to the source PDF, or ``None`` when unavailable.
        page_num: 1-based page number.
        scale: Document-wide heading size ranking from ``extract_heading_scale``.

    Returns:
        Page text with corrected heading markup.
    """
    if pdf_path is None or not scale:
        return text

    headings: list[tuple[str, int]] = _page_headings(pdf_path, page_num, scale)

    out: list[str] = []
    for line in text.split("\n"):
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
