"""PDF annotation ("comment") extraction for per-page OCR output.

Phase 2 transcribes a *rendered* page image. Some annotations (e.g. FreeText
boxes and highlight marks) are drawn into that image, but the textual content of
sticky notes and the popup notes attached to markup annotations is not — it
lives only in the PDF's annotation objects. This module reads those text-bearing
annotations directly so they can be appended to a page's Markdown as a
``## Comments`` section.

Extraction is opt-in (CLI ``--include-comments``); when disabled the page text is
left exactly as before.
"""
import re

import fitz

# Annotation subtypes whose text we surface as a "comment". Purely visual marks
# with no note (e.g. a highlight without a popup) are skipped because they carry
# no content text.
_COMMENT_TYPES: frozenset[str] = frozenset(
    {"Text", "FreeText", "Highlight", "Underline", "StrikeOut", "Squiggly"}
)


def _clean(value: str | None) -> str:
    """Collapse annotation text to a single Markdown-safe line.

    Args:
        value: Raw annotation field (``content`` or ``title``); may be ``None``.

    Returns:
        Whitespace-collapsed, trimmed single-line string.
    """
    return re.sub(r"\s+", " ", (value or "")).strip()


def extract_comments_markdown(pdf_path: str, page_num: int) -> str:
    """Extract a page's text-bearing annotations as a Markdown comments section.

    Args:
        pdf_path: Path to the source PDF file.
        page_num: 1-based page number.

    Returns:
        A ``## Comments`` Markdown block listing each comment as
        ``- **Author** (Type): content``, in annotation order. Empty string when
        the page has no text-bearing annotations or cannot be read.
    """
    items: list[tuple[str, str, str]] = []
    try:
        doc: fitz.Document = fitz.open(pdf_path)
        try:
            page: fitz.Page = doc[page_num - 1]
            for annot in page.annots() or []:
                type_name: str = annot.type[1]
                if type_name not in _COMMENT_TYPES:
                    continue
                info: dict[str, str] = annot.info
                content: str = _clean(info.get("content"))
                if not content:
                    continue
                items.append((_clean(info.get("title")), type_name, content))
        finally:
            doc.close()
    except Exception:  # noqa: BLE001 — never let annotation extraction fail a page
        return ""

    if not items:
        return ""

    lines: list[str] = ["## Comments", ""]
    for author, type_name, content in items:
        who: str = author or "Unknown"
        lines.append(f"- **{who}** ({type_name}): {content}")
    return "\n".join(lines)
