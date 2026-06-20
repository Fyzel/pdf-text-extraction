"""Phase 2 — OCR processing via Ollama vision model."""
import base64
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fitz

from pdf_extractor.annotations import extract_comments_markdown
from pdf_extractor.config import OllamaInstance
from pdf_extractor.headings import extract_heading_scale, fix_headings
from pdf_extractor.links import extract_links, splice_links
from pdf_extractor.mdlint import normalize_markdown
from pdf_extractor.pdf_errors import PDF_ERRORS
from pdf_extractor.reflow import reflow_prose
from pdf_extractor.render import _DPI_SCALE
from pdf_extractor.tables import extract_tables_markdown, splice_tables
from pdf_extractor.state import AppState, StateManager

_DEFAULT_OCR_TIMEOUT: int = 600  # fallback only; overridden by AppConfig.ocr_timeout
_BBOX_TRIM_RATIO: float = 0.05  # trim 5% off right and bottom edges of model-returned bboxes

# Blank-page detection (issue #49): skip the expensive OCR call on empty pages.
_BLANK_PAGE_MARKER: str = "<!-- blank page -->"  # written to a blank page's per-page .md
_WHITE_CHANNEL_MIN: int = 250  # a pixel counts as white when every channel is >= this (0–255)
_BLANK_WHITE_RATIO: float = 0.999  # page is blank when this fraction of pixels are white

_PROMPT: str = """\
Analyze this document page image. Return ONLY a valid JSON object with this exact structure:

{
  "text": "all text content from the page in markdown format",
  "diagrams": [{"x": 0, "y": 0, "width": 0, "height": 0}]
}

Rules:
- text: include ALL text, formatted as markdown (headings, lists, bold, italics). Use plain paragraphs for regular text — do NOT use blockquote syntax (>).
- Write each paragraph as a SINGLE line. Do not insert line breaks inside a paragraph to mirror how the text wraps in the image.
- Do NOT wrap text in emphasis markers (* or _) unless it is visually bold or italic. Regular body text must have no emphasis.
- Use heading markup (#, ##, ###) only for actual headings, matching the visual hierarchy. Ordinary body sentences are never headings.
- diagrams: pixel bounding boxes for figures, charts, illustrations only. Empty array if none. Boxes must be tight — no surrounding whitespace, margins, or text. Do not pad or expand beyond the visible edge of the figure.
- tables: render as markdown table syntax inside the text field. Do NOT add tables to diagrams.
- Return ONLY the JSON object. No explanations, no code fences, no other text.\
"""


def _encode_image(jpeg_path: Path) -> str:
    """Base64-encode a JPEG file for the Ollama multimodal request body.

    :param jpeg_path: Path to the JPEG file. Required.
    :type jpeg_path: pathlib.Path
    :return: Base64-encoded string of the raw file bytes.
    :rtype: str
    """
    return base64.b64encode(jpeg_path.read_bytes()).decode("utf-8")


def _call_ollama(instance: OllamaInstance, image_b64: str, timeout: int) -> str:
    """POST a page image to an Ollama instance and return the raw response text.

    :param instance: Ollama instance to call. Required.
    :type instance: pdf_extractor.config.OllamaInstance
    :param image_b64: Base64-encoded JPEG image string. Required.
    :type image_b64: str
    :param timeout: HTTP request timeout in seconds. Required.
    :type timeout: int
    :return: Raw ``response`` string from the Ollama JSON reply.
    :rtype: str
    :raises urllib.error.URLError: On network or HTTP error.
    :raises ValueError: If the Ollama reply is missing the ``response`` field or
        is empty.
    :raises json.JSONDecodeError: If the Ollama reply body is not valid JSON.
    """
    payload: bytes = json.dumps({
        "model": instance.model,
        "prompt": _PROMPT,
        "images": [image_b64],
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{instance.url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))

    response_text: str = body.get("response", "")
    if not response_text:
        raise ValueError("empty response from Ollama")
    return response_text


def _parse_ocr_response(response_text: str) -> dict[str, Any]:
    """Parse and validate the structured JSON OCR response from Ollama.

    Strips markdown code fences (```json ... ```) if the model wraps its output.

    :param response_text: Raw text returned by the Ollama ``response`` field.
        Required.
    :type response_text: str
    :return: Dict with keys ``text`` (str) and ``diagrams`` (list of bbox
        dicts).
    :rtype: dict[str, typing.Any]
    :raises json.JSONDecodeError: If the text is not valid JSON after fence
        stripping.
    :raises ValueError: If the parsed object is missing ``text`` or ``diagrams``
        keys.
    """
    text: str = response_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])

    data: dict[str, Any] = json.loads(text)
    if "text" not in data:
        raise ValueError("response missing 'text' field")
    if "diagrams" not in data:
        raise ValueError("response missing 'diagrams' field")

    raw_text: str = data["text"]
    lines = raw_text.splitlines()
    data["text"] = "\n".join(
        line[2:] if line.startswith("> ") else line for line in lines
    )
    return data


def _crop_diagram(
    jpeg_path: Path,
    x: int,
    y: int,
    width: int,
    height: int,
    output_path: Path,
) -> None:
    """Crop a bounding box region from a JPEG and save it as a new image.

    Coordinates are clamped to image dimensions when they exceed bounds.
    Uses page-coordinate conversion because fitz opens JPEGs as point-dimensioned
    pages independent of the embedded DPI.

    :param jpeg_path: Source JPEG to crop from. Required.
    :type jpeg_path: pathlib.Path
    :param x: Left edge in pixels. Required.
    :type x: int
    :param y: Top edge in pixels. Required.
    :type y: int
    :param width: Width of the region in pixels. Required.
    :type width: int
    :param height: Height of the region in pixels. Required.
    :type height: int
    :param output_path: Destination path for the cropped JPEG. Required.
    :type output_path: pathlib.Path
    :return: ``None``.
    :rtype: None
    """
    pix_full: fitz.Pixmap = fitz.Pixmap(str(jpeg_path))
    img_w: int = pix_full.width
    img_h: int = pix_full.height

    doc: fitz.Document = fitz.open(str(jpeg_path))
    page: fitz.Page = doc[0]
    page_w: float = page.rect.width
    page_h: float = page.rect.height
    sx: float = img_w / page_w
    sy: float = img_h / page_h

    x0: int = max(0, x)
    y0: int = max(0, y)
    x1: int = min(img_w, x + width)
    y1: int = min(img_h, y + height)

    clip: fitz.Rect = fitz.Rect(x0 / sx, y0 / sy, x1 / sx, y1 / sy)
    cropped: fitz.Pixmap = page.get_pixmap(matrix=fitz.Matrix(sx, sy), clip=clip)
    cropped.save(str(output_path))
    doc.close()


def _embedded_image_rects(pdf_path: Path, page_num: int) -> list[fitz.Rect]:
    """Return the page-coordinate rectangles of raster images embedded in a page.

    These rects come straight from the PDF object model, so they bound the figure
    exactly — unlike the vision model's bounding boxes, which tend to overshoot
    into surrounding captions and body text. Rects are ordered top-to-bottom,
    then left-to-right, to give stable diagram numbering.

    :param pdf_path: Path to the source PDF file. Required.
    :type pdf_path: pathlib.Path
    :param page_num: 1-based page number. Required.
    :type page_num: int
    :return: List of ``fitz.Rect`` in page points; empty if the page embeds no
        images (e.g. a vector-only figure).
    :rtype: list[fitz.Rect]
    """
    doc: fitz.Document = fitz.open(str(pdf_path))
    try:
        page: fitz.Page = doc[page_num - 1]
        rects: list[fitz.Rect] = []
        for img in page.get_images(full=True):
            xref: int = img[0]
            rects.extend(fitz.Rect(r) for r in page.get_image_rects(xref))
    finally:
        doc.close()
    rects.sort(key=lambda r: (round(r.y0), round(r.x0)))
    return rects


def _crop_pdf_region(
    pdf_path: Path,
    page_num: int,
    rect: fitz.Rect,
    output_path: Path,
    dpi_scale: float = _DPI_SCALE,
) -> None:
    """Render a page-coordinate rectangle from the PDF directly to a JPEG.

    Rendering from the source page at the Phase 1 DPI yields a clean, full-quality
    crop with no recompression of the already-rasterised page image.

    :param pdf_path: Path to the source PDF file. Required.
    :type pdf_path: pathlib.Path
    :param page_num: 1-based page number. Required.
    :type page_num: int
    :param rect: Region to crop, in page points. Required.
    :type rect: fitz.Rect
    :param output_path: Destination path for the cropped JPEG. Required.
    :type output_path: pathlib.Path
    :param dpi_scale: Render scale factor; should match the Phase 1 page render.
        Optional; defaults to ``_DPI_SCALE`` (2.0).
    :type dpi_scale: float
    :return: ``None``.
    :rtype: None
    """
    doc: fitz.Document = fitz.open(str(pdf_path))
    try:
        page: fitz.Page = doc[page_num - 1]
        pix: fitz.Pixmap = page.get_pixmap(
            matrix=fitz.Matrix(dpi_scale, dpi_scale), clip=rect
        )
        pix.save(str(output_path))
    finally:
        doc.close()


def _page_stem(page_num: int, page_count: int) -> str:
    """Return the zero-padded filename stem for a page (without extension).

    :param page_num: 1-based page number. Required.
    :type page_num: int
    :param page_count: Total page count, used to determine zero-padding width.
        Required.
    :type page_count: int
    :return: Stem string such as ``page_001`` for a 100-page document.
    :rtype: str
    """
    width: int = len(str(page_count))
    return f"page_{page_num:0{width}d}"


def _page_white_ratio(jpeg_path: Path) -> float:
    """Return the fraction of near-white pixels in a rendered page JPEG.

    The image is shrunk to a small thumbnail first so the per-pixel scan stays
    cheap regardless of render DPI. A pixel is "white" when every colour channel
    is at least ``_WHITE_CHANNEL_MIN``.

    :param jpeg_path: Path to the rendered page JPEG. Required.
    :type jpeg_path: pathlib.Path
    :return: Ratio in ``[0.0, 1.0]`` of white pixels to total pixels; ``0.0``
        for a missing, corrupt, or empty image, so such pages are treated as
        non-blank and still get a real OCR attempt.
    :rtype: float
    """
    try:
        pix: fitz.Pixmap = fitz.Pixmap(str(jpeg_path))
        # Halve the dimensions repeatedly until small; keeps the Python scan fast.
        while pix.width > 200 or pix.height > 200:
            pix.shrink(1)
    except PDF_ERRORS:
        # any decode failure means "treat as non-blank"
        return 0.0

    total: int = pix.width * pix.height
    if total == 0:
        return 0.0

    data: bytes = pix.samples
    stride: int = pix.n
    # Count only colour channels; the alpha channel (if any) is not part of
    # the visible colour, so n=2 (gray+alpha) checks 1 channel, n=4 (RGBA) 3.
    channels: int = max(1, pix.n - pix.alpha)
    white: int = 0
    for i in range(0, len(data), stride):
        if all(data[i + c] >= _WHITE_CHANNEL_MIN for c in range(channels)):
            white += 1
    return white / total


def _is_blank_page(pdf_path: Path | None, page_num: int, jpeg_path: Path) -> bool:
    """Decide whether a page is blank, to skip the expensive OCR call.

    Hybrid check: when a source PDF is available, any extractable text or vector
    drawing means the page is not blank (cheap, no pixel work). Otherwise — and
    for the no-PDF case — the page is blank when its rendered image is
    near-uniformly white, which also catches scanned/image-only blank pages.

    A PDF that cannot be opened or paged is treated as "can't prove non-blank":
    the inspection is skipped and the decision falls through to the whiteness
    heuristic rather than aborting OCR for the page.

    :param pdf_path: Source PDF. Required, but may be ``None`` when unavailable
        (the whiteness check is then used alone).
    :type pdf_path: pathlib.Path | None
    :param page_num: 1-based page number. Required.
    :type page_num: int
    :param jpeg_path: Rendered page JPEG, used for the whiteness fallback.
        Required.
    :type jpeg_path: pathlib.Path
    :return: ``True`` when the page should be treated as blank.
    :rtype: bool
    """
    if pdf_path is not None:
        try:
            doc: fitz.Document = fitz.open(str(pdf_path))
            try:
                page: fitz.Page = doc[page_num - 1]
                if page.get_text().strip() or page.get_drawings():
                    return False
            finally:
                doc.close()
        except PDF_ERRORS:
            # can't inspect PDF; fall through to the pixel-whiteness check
            pass

    return _page_white_ratio(jpeg_path) >= _BLANK_WHITE_RATIO


def _ocr_page_with_retry(
    page_num: int,
    instances_ordered: list[OllamaInstance],
    pages_dir: Path,
    diagrams_dir: Path,
    page_count: int,
    ocr_timeout: int = _DEFAULT_OCR_TIMEOUT,
    pdf_path: Path | None = None,
    dpi_scale: float = _DPI_SCALE,
    include_comments: bool = False,
    heading_scale: list[float] | None = None,
    include_links: bool = False,
) -> tuple[int, bool, str, int]:
    """Attempt OCR on one page, trying each instance in order until one succeeds.

    On success, writes the per-page markdown file and any cropped diagram images.
    Round-robin is achieved by the caller rotating ``instances_ordered`` based on
    page number before calling this function.

    Blank pages (see ``_is_blank_page``) are detected up front and short-circuit
    the whole instance loop: a marker file is written and the function returns
    success without any Ollama call.

    Diagram cropping prefers the PDF's embedded image rects (exact bounds) when
    ``pdf_path`` is supplied and the page embeds raster images; it falls back to
    the vision model's bounding boxes only for vector-only figures or when no PDF
    is available. The model's non-empty ``diagrams`` list gates whether any crop
    happens at all, so pages with purely decorative images stay text-only.

    :param page_num: 1-based page number. Required.
    :type page_num: int
    :param instances_ordered: Instances to try in order (pre-rotated for
        round-robin). Required.
    :type instances_ordered: list[pdf_extractor.config.OllamaInstance]
    :param pages_dir: Directory containing the rendered page JPEG files.
        Required.
    :type pages_dir: pathlib.Path
    :param diagrams_dir: Directory where cropped diagram images will be written.
        Required.
    :type diagrams_dir: pathlib.Path
    :param page_count: Total page count for zero-padded filename generation.
        Required.
    :type page_count: int
    :param ocr_timeout: Per-request HTTP timeout in seconds. Optional; defaults
        to ``_DEFAULT_OCR_TIMEOUT`` (600).
    :type ocr_timeout: int
    :param pdf_path: Source PDF, used for exact embedded-image crops. Optional;
        defaults to ``None``, in which case diagrams are cropped from the
        rendered JPEG using model bboxes.
    :type pdf_path: pathlib.Path | None
    :param dpi_scale: Render scale factor for PDF-region crops; should match
        Phase 1. Optional; defaults to ``_DPI_SCALE`` (2.0).
    :type dpi_scale: float
    :param include_comments: When ``True`` and a source PDF is available, append
        the page's text-bearing annotations as a ``## Comments`` section.
        Optional; defaults to ``False``.
    :type include_comments: bool
    :param heading_scale: Document-wide heading size ranking from
        :func:`pdf_extractor.headings.extract_heading_scale`; passed through to
        ``fix_headings`` to relevel the page's headings. Only takes effect when
        ``pdf_path`` is also supplied; ``fix_headings`` no-ops without a PDF.
        Optional; defaults to ``None``, and ``None`` or empty leaves headings
        as-is.
    :type heading_scale: list[float] | None
    :param include_links: When ``True`` and a source PDF is available, rewrite
        the page's plain anchor text as Markdown links using the PDF's URI
        links. Optional; defaults to ``False``.
    :type include_links: bool
    :return: Tuple of ``(page_num, success, error_message, diagram_count)``;
        ``error_message`` is empty on success and ``diagram_count`` is 0 on
        failure.
    :rtype: tuple[int, bool, str, int]
    """
    stem: str = _page_stem(page_num, page_count)
    jpeg_path: Path = pages_dir / f"{stem}.jpg"
    md_path: Path = pages_dir / f"{stem}.md"
    last_error: str = "no instances available"

    # Skip blank pages entirely — no Ollama call. Write a marker so the page
    # still appears (empty) in the combined output and is recorded as done.
    # A blank page may still carry comment annotations (e.g. a lone sticky note),
    # which are not in the rendered image, so preserve them when requested.
    if _is_blank_page(pdf_path, page_num, jpeg_path):
        print(f"  Page {page_num}: blank — skipped OCR")
        comments_md: str = (
            extract_comments_markdown(str(pdf_path), page_num)
            if include_comments and pdf_path is not None
            else ""
        )
        body: str = _BLANK_PAGE_MARKER
        if comments_md:
            body = f"{_BLANK_PAGE_MARKER}\n\n{comments_md}"
        md_path.write_text(body, encoding="utf-8")
        return page_num, True, "", 0

    for instance in instances_ordered:
        try:
            image_b64: str = _encode_image(jpeg_path)
            raw: str = _call_ollama(instance, image_b64, ocr_timeout)
            data: dict[str, Any] = _parse_ocr_response(raw)
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{instance.url}: {exc}"
            continue

        # Clean up the model's markdown before any PDF-sourced splicing:
        # 1) correct heading levels from the PDF font hierarchy (must run first,
        #    while a mis-flattened heading is still on its own line),
        # 2) reflow soft-wrapped prose and strip stray paragraph emphasis,
        # 3) normalise list markup.
        page_text: str = str(data.get("text", ""))
        page_text = fix_headings(
            page_text, str(pdf_path) if pdf_path is not None else None,
            page_num, heading_scale or [],
        )
        page_text = reflow_prose(page_text)
        page_text = normalize_markdown(page_text)
        # Replace the model's unreliable table transcription with tables read
        # straight from the PDF, when a source PDF is available.
        if pdf_path is not None:
            page_text = splice_tables(
                page_text, extract_tables_markdown(str(pdf_path), page_num)
            )
        # Rewrite plain anchor text as Markdown links using the PDF's URI links.
        if include_links and pdf_path is not None:
            page_text = splice_links(
                page_text, extract_links(str(pdf_path), page_num)
            )
        raw_diagrams: list[dict[str, Any]] = data.get("diagrams", [])

        cropped_count: int = 0
        diagram_refs: list[str] = []

        # Prefer exact embedded-image rects from the PDF; fall back to the model's
        # bounding boxes only for vector-only figures or when no PDF is available.
        # A PDF-scan failure here must not abort the page, so fall back to bboxes.
        rects: list[fitz.Rect] = []
        if raw_diagrams and pdf_path is not None:
            try:
                rects = _embedded_image_rects(pdf_path, page_num)
            except PDF_ERRORS:
                rects = []

        if rects:
            for idx, rect in enumerate(rects, start=1):
                try:
                    diag_filename: str = f"{stem}_diagram_{idx}.jpg"
                    diag_path: Path = diagrams_dir / diag_filename
                    diagrams_dir.mkdir(parents=True, exist_ok=True)
                    _crop_pdf_region(pdf_path, page_num, rect, diag_path, dpi_scale)
                    diagram_refs.append(f"![Diagram {idx}](diagrams/{diag_filename})")
                    cropped_count += 1
                except PDF_ERRORS as exc:
                    print(f"  Page {page_num} diagram {idx}: crop failed — {exc}")
        else:
            for idx, bbox in enumerate(raw_diagrams, start=1):
                try:
                    diag_filename = f"{stem}_diagram_{idx}.jpg"
                    diag_path = diagrams_dir / diag_filename
                    diagrams_dir.mkdir(parents=True, exist_ok=True)
                    raw_w: int = int(bbox.get("width", 0))
                    raw_h: int = int(bbox.get("height", 0))
                    _crop_diagram(
                        jpeg_path,
                        int(bbox.get("x", 0)),
                        int(bbox.get("y", 0)),
                        int(raw_w * (1 - _BBOX_TRIM_RATIO)),
                        int(raw_h * (1 - _BBOX_TRIM_RATIO)),
                        diag_path,
                    )
                    diagram_refs.append(f"![Diagram {idx}](diagrams/{diag_filename})")
                    cropped_count += 1
                except PDF_ERRORS as exc:
                    print(f"  Page {page_num} diagram {idx}: crop failed — {exc}")

        parts: list[str] = [page_text]
        if diagram_refs:
            parts += ["", *diagram_refs]
        if include_comments and pdf_path is not None:
            comments_md = extract_comments_markdown(str(pdf_path), page_num)
            if comments_md:
                parts += ["", comments_md]
        md_path.write_text("\n".join(parts), encoding="utf-8")

        return page_num, True, "", cropped_count

    return page_num, False, last_error, 0


def run_phase2(
    output_dir: Path,
    page_count: int,
    pending: list[int],
    instances: list[OllamaInstance],
    state: AppState,
    state_mgr: StateManager,
    ocr_timeout: int = _DEFAULT_OCR_TIMEOUT,
    pdf_path: Path | None = None,
    dpi_scale: float = _DPI_SCALE,
    include_comments: bool = False,
    include_links: bool = False,
) -> None:
    """Run Phase 2 OCR concurrently across all available Ollama instances.

    Spawns one worker thread per instance. Pages are distributed in round-robin
    order; failed pages are retried on the next instance in the rotation.
    State writes are serialised through the StateManager lock.

    :param output_dir: Working directory containing ``pages/`` and
        ``diagrams/`` subdirs. Required.
    :type output_dir: pathlib.Path
    :param page_count: Total page count for zero-padded filename generation.
        Required.
    :type page_count: int
    :param pending: 1-based page numbers with ``image_done=True`` and
        ``ocr_done=False``. Required.
    :type pending: list[int]
    :param instances: Reachable Ollama instances to distribute OCR work across.
        Required.
    :type instances: list[pdf_extractor.config.OllamaInstance]
    :param state: Shared application state mutated under the StateManager lock.
        Required.
    :type state: pdf_extractor.state.AppState
    :param state_mgr: State manager for serialised, atomic state writes.
        Required.
    :type state_mgr: pdf_extractor.state.StateManager
    :param ocr_timeout: Per-request HTTP timeout in seconds passed to each
        worker. Optional; defaults to ``_DEFAULT_OCR_TIMEOUT`` (600).
    :type ocr_timeout: int
    :param pdf_path: Source PDF, forwarded to workers for exact embedded-image
        crops. Optional; defaults to ``None``.
    :type pdf_path: pathlib.Path | None
    :param dpi_scale: Render scale factor for PDF-region crops; should match
        Phase 1. Optional; defaults to ``_DPI_SCALE`` (2.0).
    :type dpi_scale: float
    :param include_comments: When ``True``, append PDF annotations as a comments
        section to each page; forwarded to every worker. Optional; defaults to
        ``False``.
    :type include_comments: bool
    :param include_links: When ``True``, rewrite plain anchor text as Markdown
        links from the PDF's URI links on each page; forwarded to every worker.
        Optional; defaults to ``False``.
    :type include_links: bool
    :return: ``None``.
    :rtype: None
    """
    pages_dir: Path = output_dir / "pages"
    diagrams_dir: Path = output_dir / "diagrams"
    n: int = len(instances)

    # Build the document-wide heading size ranking once; workers reuse it to
    # correct per-page heading levels consistently.
    heading_scale: list[float] = (
        extract_heading_scale(str(pdf_path)) if pdf_path is not None else []
    )

    def _args(
        page_num: int,
    ) -> tuple[
        int, list[OllamaInstance], Path, Path, int, int, Path | None, float, bool,
        list[float], bool,
    ]:
        """Build the positional arg tuple for one page's OCR worker call.

        Rotates ``instances`` so page ``page_num`` starts on a different
        instance (round-robin), then packs the shared per-run settings.

        :param page_num: 1-based page number to build worker args for. Required.
        :type page_num: int
        :return: The positional argument tuple for :func:`_ocr_page_with_retry`,
            i.e. ``(page_num, ordered_instances, pages_dir, diagrams_dir,
            page_count, ocr_timeout, pdf_path, dpi_scale, include_comments,
            heading_scale, include_links)``.
        :rtype: tuple
        """
        start: int = (page_num - 1) % n
        ordered: list[OllamaInstance] = instances[start:] + instances[:start]
        return (
            page_num, ordered, pages_dir, diagrams_dir, page_count,
            ocr_timeout, pdf_path, dpi_scale, include_comments, heading_scale,
            include_links,
        )

    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {
            executor.submit(_ocr_page_with_retry, *_args(p)): p
            for p in pending
        }
        for future in as_completed(futures):
            page_num, success, error, diagram_count = future.result()
            if success:
                state_mgr.update_page(
                    state, page_num, ocr_done=True, diagram_count=diagram_count
                )
            else:
                print(f"  Page {page_num}: OCR failed — {error}")
                state_mgr.update_page(state, page_num, ocr_failed=True)
