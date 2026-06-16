"""Phase 2 — OCR processing via Ollama vision model."""
import base64
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fitz

from pdf_extractor.config import OllamaInstance
from pdf_extractor.mdlint import normalize_markdown
from pdf_extractor.render import _DPI_SCALE
from pdf_extractor.state import AppState, StateManager

_DEFAULT_OCR_TIMEOUT: int = 600  # fallback only; overridden by AppConfig.ocr_timeout
_BBOX_TRIM_RATIO: float = 0.05  # trim 5% off right and bottom edges of model-returned bboxes

_PROMPT: str = """\
Analyze this document page image. Return ONLY a valid JSON object with this exact structure:

{
  "text": "all text content from the page in markdown format",
  "diagrams": [{"x": 0, "y": 0, "width": 0, "height": 0}]
}

Rules:
- text: include ALL text, formatted as markdown (headings, lists, bold, italics). Use plain paragraphs for regular text — do NOT use blockquote syntax (>).
- diagrams: pixel bounding boxes for figures, charts, illustrations only. Empty array if none. Boxes must be tight — no surrounding whitespace, margins, or text. Do not pad or expand beyond the visible edge of the figure.
- tables: render as markdown table syntax inside the text field. Do NOT add tables to diagrams.
- Return ONLY the JSON object. No explanations, no code fences, no other text.\
"""


def _encode_image(jpeg_path: Path) -> str:
    """Base64-encode a JPEG file for the Ollama multimodal request body.

    Args:
        jpeg_path: Path to the JPEG file.

    Returns:
        Base64-encoded string of the raw file bytes.
    """
    return base64.b64encode(jpeg_path.read_bytes()).decode("utf-8")


def _call_ollama(instance: OllamaInstance, image_b64: str, timeout: int) -> str:
    """POST a page image to an Ollama instance and return the raw response text.

    Args:
        instance: Ollama instance to call.
        image_b64: Base64-encoded JPEG image string.
        timeout: HTTP request timeout in seconds.

    Returns:
        Raw ``response`` string from the Ollama JSON reply.

    Raises:
        urllib.error.URLError: On network or HTTP error.
        ValueError: If the Ollama reply is missing the ``response`` field or is empty.
        json.JSONDecodeError: If the Ollama reply body is not valid JSON.
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

    Args:
        response_text: Raw text returned by the Ollama ``response`` field.

    Returns:
        Dict with keys ``text`` (str) and ``diagrams`` (list of bbox dicts).

    Raises:
        json.JSONDecodeError: If the text is not valid JSON after fence stripping.
        ValueError: If the parsed object is missing ``text`` or ``diagrams`` keys.
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

    Args:
        jpeg_path: Source JPEG to crop from.
        x: Left edge in pixels.
        y: Top edge in pixels.
        width: Width of the region in pixels.
        height: Height of the region in pixels.
        output_path: Destination path for the cropped JPEG.
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

    Args:
        pdf_path: Path to the source PDF file.
        page_num: 1-based page number.

    Returns:
        List of ``fitz.Rect`` in page points; empty if the page embeds no images
        (e.g. a vector-only figure).
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
) -> None:
    """Render a page-coordinate rectangle from the PDF directly to a JPEG.

    Rendering from the source page at the Phase 1 DPI yields a clean, full-quality
    crop with no recompression of the already-rasterised page image.

    Args:
        pdf_path: Path to the source PDF file.
        page_num: 1-based page number.
        rect: Region to crop, in page points.
        output_path: Destination path for the cropped JPEG.
    """
    doc: fitz.Document = fitz.open(str(pdf_path))
    try:
        page: fitz.Page = doc[page_num - 1]
        pix: fitz.Pixmap = page.get_pixmap(
            matrix=fitz.Matrix(_DPI_SCALE, _DPI_SCALE), clip=rect
        )
        pix.save(str(output_path))
    finally:
        doc.close()


def _page_stem(page_num: int, page_count: int) -> str:
    """Return the zero-padded filename stem for a page (without extension).

    Args:
        page_num: 1-based page number.
        page_count: Total page count, used to determine zero-padding width.

    Returns:
        Stem string such as ``page_001`` for a 100-page document.
    """
    width: int = len(str(page_count))
    return f"page_{page_num:0{width}d}"


def _ocr_page_with_retry(
    page_num: int,
    instances_ordered: list[OllamaInstance],
    pages_dir: Path,
    diagrams_dir: Path,
    page_count: int,
    ocr_timeout: int = _DEFAULT_OCR_TIMEOUT,
    pdf_path: Path | None = None,
) -> tuple[int, bool, str, int]:
    """Attempt OCR on one page, trying each instance in order until one succeeds.

    On success, writes the per-page markdown file and any cropped diagram images.
    Round-robin is achieved by the caller rotating ``instances_ordered`` based on
    page number before calling this function.

    Diagram cropping prefers the PDF's embedded image rects (exact bounds) when
    ``pdf_path`` is supplied and the page embeds raster images; it falls back to
    the vision model's bounding boxes only for vector-only figures or when no PDF
    is available. The model's non-empty ``diagrams`` list gates whether any crop
    happens at all, so pages with purely decorative images stay text-only.

    Args:
        page_num: 1-based page number.
        instances_ordered: Instances to try in order (pre-rotated for round-robin).
        pages_dir: Directory containing the rendered page JPEG files.
        diagrams_dir: Directory where cropped diagram images will be written.
        page_count: Total page count for zero-padded filename generation.
        ocr_timeout: Per-request HTTP timeout in seconds.
        pdf_path: Source PDF, used for exact embedded-image crops. When ``None``,
            diagrams are cropped from the rendered JPEG using model bboxes.

    Returns:
        Tuple of ``(page_num, success, error_message, diagram_count)``.
        ``error_message`` is empty on success; ``diagram_count`` is 0 on failure.
    """
    stem: str = _page_stem(page_num, page_count)
    jpeg_path: Path = pages_dir / f"{stem}.jpg"
    md_path: Path = pages_dir / f"{stem}.md"
    last_error: str = "no instances available"

    for instance in instances_ordered:
        try:
            image_b64: str = _encode_image(jpeg_path)
            raw: str = _call_ollama(instance, image_b64, ocr_timeout)
            data: dict[str, Any] = _parse_ocr_response(raw)
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{instance.url}: {exc}"
            continue

        page_text: str = normalize_markdown(str(data.get("text", "")))
        raw_diagrams: list[dict[str, Any]] = data.get("diagrams", [])

        cropped_count: int = 0
        diagram_refs: list[str] = []

        # Prefer exact embedded-image rects from the PDF; fall back to the model's
        # bounding boxes only for vector-only figures or when no PDF is available.
        rects: list[fitz.Rect] = (
            _embedded_image_rects(pdf_path, page_num)
            if raw_diagrams and pdf_path is not None
            else []
        )

        if rects:
            for idx, rect in enumerate(rects, start=1):
                try:
                    diag_filename: str = f"{stem}_diagram_{idx}.jpg"
                    diag_path: Path = diagrams_dir / diag_filename
                    diagrams_dir.mkdir(parents=True, exist_ok=True)
                    _crop_pdf_region(pdf_path, page_num, rect, diag_path)
                    diagram_refs.append(f"![Diagram {idx}](diagrams/{diag_filename})")
                    cropped_count += 1
                except Exception as exc:  # noqa: BLE001
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
                except Exception as exc:  # noqa: BLE001
                    print(f"  Page {page_num} diagram {idx}: crop failed — {exc}")

        parts: list[str] = [page_text]
        if diagram_refs:
            parts += ["", *diagram_refs]
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
) -> None:
    """Run Phase 2 OCR concurrently across all available Ollama instances.

    Spawns one worker thread per instance. Pages are distributed in round-robin
    order; failed pages are retried on the next instance in the rotation.
    State writes are serialised through the StateManager lock.

    Args:
        output_dir: Working directory containing ``pages/`` and ``diagrams/`` subdirs.
        page_count: Total page count for zero-padded filename generation.
        pending: 1-based page numbers with ``image_done=True`` and ``ocr_done=False``.
        instances: Reachable Ollama instances to distribute OCR work across.
        state: Shared AppState mutated under the StateManager lock.
        state_mgr: StateManager for serialised, atomic state writes.
        ocr_timeout: Per-request HTTP timeout in seconds passed to each worker.
        pdf_path: Source PDF, forwarded to workers for exact embedded-image crops.
    """
    pages_dir: Path = output_dir / "pages"
    diagrams_dir: Path = output_dir / "diagrams"
    n: int = len(instances)

    def _args(
        page_num: int,
    ) -> tuple[int, list[OllamaInstance], Path, Path, int, int, Path | None]:
        start: int = (page_num - 1) % n
        ordered: list[OllamaInstance] = instances[start:] + instances[:start]
        return page_num, ordered, pages_dir, diagrams_dir, page_count, ocr_timeout, pdf_path

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
