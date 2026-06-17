# Project Initial Planning

## Purpose

The purpose of this command line application is to have an arbitrarily large PDF file path provided by the user as a command line argument. The result of the command line application will be a single markdown file that contains the OCRed text from each page of the PDF file. Each page in the markdown file will be separated by a page break. The application will also extract any diagrams from the PDF pages and save them as separate image files, with references to these images included in the corresponding markdown text.

## Processing Steps

The processing steps are broken down in `docs/high-level-process.mermaid`.

The application will iterate through each page of the PDF file and save each page as a JPEG file using PyMuPDF.

For each of these JPEG files, the program will call a local or remote Ollama-hosted `qwen2.5vl` instance to perform optical character recognition (OCR) on each page in order. A single Ollama call per page handles both OCR and diagram detection.

State for all processing is tracked in a unified `state.json` file inside the output directory. The state file records which pages have been successfully processed as JPEG files, which have been OCR'd, and whether the final combined file has been written. If multiple Ollama instances are configured, OCR processing is distributed concurrently across the available instances to optimize processing time.

The OCRed text from each page is saved as an individual markdown file. If Ollama flags a diagram on a page, the application crops the figure at its exact bounds — taken from the PDF's embedded image objects (`page.get_image_rects`) — and saves it as a separate image file in a designated directory. Cropping from the embedded image rect keeps the crop tight to the figure, with no surrounding caption or body text; the model's returned bounding box is used only as a fallback for vector-only figures that have no embedded raster image. The markdown file for that page includes references to the extracted diagram images.

Optionally (CLI flag `--include-comments`), the application also reads the page's PDF comment annotations — sticky notes, FreeText boxes, and the popup notes attached to highlight/underline/strikeout markup — directly from the PDF and appends them to the page markdown as a `## Comments` section. This recovers comment text that lives only in the PDF's annotation objects and is therefore absent from the rendered image the model sees. Comments are excluded by default.

If the OCR processing for a page fails, the application logs the error and continues processing the remaining pages. A summary of any pages that failed to process is printed at the end of execution.

When all pages have been processed, the application combines all individual markdown files into a single markdown file. The combined file is saved in the same directory as the original PDF with the same name but a `.md` extension. Each page is preceded by a header in this format:

```Markdown
--- PAGE 1 ---
[Text from page 1]
```

If the application is re-run on the same PDF file, it checks `state.json` to determine which pages have already been processed and resumes from where it left off.

## Configuration

The application reads `ollama.json` from the following locations in order, using the first found:
1. Directory containing the PDF file
2. Project directory
3. Built-in default: single local instance at `http://localhost:11434` with model `qwen2.5vl:7b`

`ollama.json` is excluded from version control (see `.gitignore`). Copy `ollama.sample.json` to `ollama.json` and edit to match your environment.

### `ollama.json` Schema

Each instance is an object with a `url` and optional per-instance `model` override.

```json
{
  "max_render_workers": 4,
  "instances": [
    { "url": "http://host-a:11434", "model": "qwen2.5vl:32b" },
    { "url": "http://host-b:11434", "model": "qwen2.5vl:7b" }
  ]
}
```

- `max_render_workers`: optional; caps Phase 1 process pool size; defaults to `os.cpu_count()` if omitted
- `url`: Ollama base URL for this instance
- `model`: optional; defaults to `qwen2.5vl:7b` if omitted

**Model sizing guidance:**
- Hosts with ≥20GB available memory (unified or VRAM): prefer `qwen2.5vl:32b` for higher accuracy on complex pages
- Hosts with ~16GB VRAM: use `qwen2.5vl:7b` for fast throughput (~60-80 tok/s)

Before processing begins, the app probes each URL with `GET /api/tags`. Unreachable URLs are excluded from the working pool. If zero instances respond, the application exits with code 4.

## Output Directory Structure

```
/path/to/my-document/           ← original PDF location
├── my-document.pdf             ← original PDF (untouched)
├── my-document.md              ← final combined output
└── my-document/                ← working directory
    ├── state.json
    ├── pages/
    │   ├── page_001.jpg
    │   ├── page_001.md
    │   ├── page_002.jpg
    │   ├── page_002.md
    │   └── ...
    └── diagrams/               ← created only when diagrams are detected
        ├── page_001_diagram_1.jpg
        ├── page_003_diagram_1.jpg
        ├── page_003_diagram_2.jpg
        └── ...
```

Diagram numbering (`M`) starts at 1 and is scoped per page.

## State File Schema (`state.json`)

```json
{
  "pdf_path": "/absolute/path/to/my-document.pdf",
  "page_count": 42,
  "combined_done": false,
  "pages": {
    "1": {
      "image_done": false,
      "image_failed": false,
      "ocr_done": false,
      "ocr_failed": false,
      "diagram_count": 0
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `pdf_path` | Absolute path stored for identification |
| `page_count` | Total pages discovered at startup |
| `combined_done` | `true` after Phase 3 completes; triggers exit 0 on re-run |
| `image_done` | `true` after JPEG saved successfully |
| `image_failed` | `true` if JPEG rendering fails; page skipped in OCR phase |
| `ocr_done` | `true` after per-page markdown saved successfully |
| `ocr_failed` | `true` if all Ollama retries exhausted; page skipped in combine phase |
| `diagram_count` | Number of diagrams extracted from this page |

### Resumability Rules

- `combined_done = true` → log "already complete", print output path, exit 0
- `state.json` exists, `combined_done = false` → skip pages where `image_done` or `image_failed` is `true` in Phase 1; skip pages where `ocr_done` or `ocr_failed` is `true` in Phase 2; re-run Phase 3 unconditionally
- No `state.json` → start fresh

## OCR Prompt Strategy

A single Ollama call per page handles both OCR and diagram detection. The page JPEG is sent as a base64-encoded inline image in the `/api/generate` multimodal request body.

The prompt requests a structured JSON response:

```json
{
  "text": "full OCR text in markdown",
  "diagrams": [
    { "x": 100, "y": 200, "width": 400, "height": 300 }
  ]
}
```

- `text`: full page OCR output in markdown format
- `diagrams`: array of bounding boxes in pixels relative to the rendered JPEG; empty array if no diagrams present. A non-empty array acts as a per-page flag that the page contains a figure — the actual crop bounds are taken from the PDF's embedded image rects, not these coordinates (the coordinates are used only as a fallback for vector-only figures with no embedded raster image)

If JSON parsing of the response fails, the page is treated as an OCR failure.

## Concurrency

### Core Detection

At startup the application calls `os.cpu_count()` to determine available logical cores and logs the result. This value drives Phase 1 worker count. An optional `max_render_workers` key in `ollama.json` caps the count for resource-constrained environments; if absent, all available cores are used.

### Phase 1 — Image Rendering (CPU-bound, multi-process)

Uses `concurrent.futures.ProcessPoolExecutor(max_workers=min(cpu_count, max_render_workers))`. Each worker renders one page to JPEG independently. Workers return a result tuple `(page_num, success, error_message)` to the main process. The main process collects all results after the pool completes and writes state updates to `state.json` in a single serial pass — no inter-process locking required.

### Phase 2 — OCR (I/O-bound, multi-thread)

Uses a thread pool with one worker thread per available Ollama instance.
- Pages are placed in a shared work queue; workers pull and call their assigned instance
- On failure: page is re-queued for a different instance; each available instance is tried once before marking `ocr_failed`
- `state.json` writes are serialized with a `threading.Lock`

# Technology Stack

## Application Dependencies

- Python 3.14
- PyMuPDF (`fitz`) — PDF page rendering and JPEG export
- Ollama hosted locally or remotely, running `qwen2.5vl` (7b or 32b per instance)

## Dev / Tooling Dependencies

- pytest — test framework
- pytest-mock — Ollama HTTP call mocking
- pylint — linting
- bandit — security static analysis
- pre-commit — git hook management
- PyYAML — YAML parsing
- trivy — secret detection (installed separately, called via pre-commit)
- actionlint — GitHub Actions workflow linting (installed separately, called via pre-commit)

## Error States

Each error code returned by the command line application has a unique exit code mapping to exactly one error condition.

| Exit Code | Condition                                              |
|-----------|--------------------------------------------------------|
| 0         | Success                                                |
| 1         | Missing PDF path argument                              |
| 2         | PDF file not found                                     |
| 3         | PDF file not readable or inaccessible                  |
| 4         | No Ollama instances reachable                          |
| 5         | All pages failed image rendering                       |
| 6         | All pages failed OCR processing                        |
| 7         | Output file write error (combining markdown)           |

### Exit Code Details

- **Exit 1**: no PDF path argument supplied; application exits before any file I/O
- **Exit 2**: PDF path provided but file does not exist
- **Exit 3**: PDF file exists but cannot be opened or read; no output directory is created
- **Exit 4**: triggered at startup when zero Ollama instances respond to health check
- **Exit 5**: triggered only when every page fails JPEG rendering; partial failures are non-fatal and logged
- **Exit 6**: triggered when every page with a valid JPEG fails OCR after exhausting all instance retries
- **Exit 7**: triggered during Phase 3 write of the combined markdown; per-page `.md` write failures are non-fatal

# Testing

## Test Structure

```
tests/
├── fixtures/
│   ├── simple.pdf          — single page, plain text only
│   ├── multipage.pdf       — 10 pages, plain text, tests pagination and ordering
│   ├── diagrams.pdf        — pages containing embedded diagrams, tests the embedded-image-rect crop path
│   ├── mixed.pdf           — combination of text-only and diagram pages
│   ├── tables.pdf          — 3 pages: simple, multi-column, and complex/irregular tables
│   └── corrupt.pdf         — malformed PDF, triggers exit 5
├── unit/
│   ├── test_cli.py
│   ├── test_config.py
│   ├── test_state.py
│   ├── test_image.py
│   ├── test_ocr.py
│   └── test_combine.py
├── integration/
│   ├── test_phase1.py
│   ├── test_phase2.py
│   ├── test_phase3.py
│   └── test_resume.py
└── e2e/
    └── test_pipeline.py
```

All Ollama HTTP calls are mocked in unit and integration tests. End-to-end tests default to mocked Ollama; a `--live-ollama` pytest flag enables real Ollama calls when an instance is reachable.

## Sample PDF Fixtures

PDF fixtures are **not committed to git** — they are generated programmatically by `conftest.py` at test bootstrap using PyMuPDF and written to `tests/fixtures/` before the test session runs. The `tests/fixtures/` directory is tracked but the generated `.pdf` files are gitignored.

| File | Pages | Purpose |
|------|-------|---------|
| `simple.pdf` | 1 | Baseline OCR path, plain text only, no diagrams or tables |
| `multipage.pdf` | 10 | Page ordering, combined output structure, plain text only |
| `diagrams.pdf` | 3 | Diagram detection, embedded-image-rect crop, image references in markdown |
| `mixed.pdf` | 5 | Mix of text-only and diagram pages |
| `tables.pdf` | 3 | Page 1: simple table; page 2: multi-column table; page 3: complex table with irregular structure |
| `corrupt.pdf` | N/A | Malformed file, triggers exit 5 (all pages fail image rendering) |

## Unit Tests

### `test_cli.py` — argument parsing and exit codes
- Missing argument → exit 1
- Non-existent path → exit 2
- Unreadable file (mocked permissions) → exit 3
- Valid path passes through to processing

### `test_config.py` — `ollama.json` loading and core detection
- File found alongside PDF → loaded correctly
- File found in project dir → loaded correctly
- No file found → built-in defaults applied
- Malformed JSON → raises config error
- Missing `instances` key → raises config error
- Instance with no `model` → defaults to `qwen2.5vl:7b`
- All instances unreachable (mocked health check) → exit 4
- Mix of reachable and unreachable → unreachable excluded, processing continues
- `max_render_workers` absent → worker count equals `os.cpu_count()`
- `max_render_workers` set to 4 on an 8-core machine → pool capped at 4
- `max_render_workers` set higher than `os.cpu_count()` → clamped to `os.cpu_count()`
- `max_render_workers` set to 0 or negative → raises config error

### `test_state.py` — `state.json` lifecycle
- No existing state → initialized with correct schema and page count
- Existing state with `combined_done = true` → detected as complete
- Existing state with some pages done → detected as partial
- Existing state with no pages done → detected as not started
- Concurrent writes serialized correctly (lock prevents corruption)

### `test_image.py` — JPEG rendering
- Successful render → file saved to correct path, state updated
- Render failure → error logged, `image_failed = true`, processing continues
- All pages fail → exit 5
- Partial failure → processing continues to OCR phase
- Output filename format: `page_001.jpg`, `page_010.jpg`, `page_100.jpg` (zero-padded to page count width)
- Multi-core: `multipage.pdf` rendered with pool of N workers → all 10 pages produced, none duplicated or missing
- Results from all workers collected before any state write (no partial state on crash mid-pool)

### `test_ocr.py` — Ollama OCR call and response parsing
- Valid JSON response with text and empty diagrams → `ocr_done = true`, markdown saved
- Valid JSON response flagging a diagram on a page with an embedded image → figure cropped at the exact embedded-image rect, markdown includes image references
- Valid JSON response flagging a diagram on a page with no embedded raster (vector-only) → falls back to cropping the model's bounding box from the JPEG
- Embedded-image rect takes precedence over a greedy model bounding box → crop matches the figure bounds, not the model's overshoot
- Invalid JSON response → page marked `ocr_failed`
- HTTP error from Ollama → retry on next instance
- All instances exhausted → page marked `ocr_failed`
- All pages `ocr_failed` → exit 6
- Bounding box coordinates out of JPEG bounds (fallback path) → cropped to image boundary, logged as warning
- Round-robin distribution: pages assigned to instances in rotation
- Table in OCR response rendered as markdown table (`| col | col |` syntax), not treated as diagram
- Simple table (mocked response) → output markdown contains valid markdown table
- Multi-column table (mocked response) → column count preserved exactly
- Complex/irregular table (mocked response) → output recorded as-is; no crash on irregular structure

### `test_combine.py` — markdown combination
- Pages combined in ascending order
- Each page preceded by `--- PAGE N ---`
- Pages with `ocr_failed = true` skipped in output
- Combined file written to correct path (sibling to PDF)
- File write error → exit 7
- `combined_done` set to `true` in state after successful write

## Integration Tests

### `test_phase1.py` — full image processing phase
- Real PyMuPDF rendering against `simple.pdf` → correct number of JPEGs produced
- Real PyMuPDF rendering against `multipage.pdf` → 10 JPEGs, correctly named
- `corrupt.pdf` → all pages fail, exit 5

### `test_phase2.py` — full OCR phase with mocked Ollama
- Process all page JPEGs from `simple.pdf` with mocked response → per-page markdown produced
- Process `diagrams.pdf` pages with mocked response flagging diagrams → figures cropped at their embedded-image rects and saved, markdown references correct paths
- Process `tables.pdf` pages with mocked response containing markdown tables → per-page markdown contains valid markdown table syntax; no diagram files created for table pages
- Single Ollama instance fails mid-run → failed page marked, processing continues

### `test_phase3.py` — full combine phase
- Pre-generated per-page markdown files → combined file matches expected format
- Gaps (some pages `ocr_failed`) → gaps skipped, remaining pages in correct order

### `test_resume.py` — resumability
- State with Phase 1 partially complete → only unprocessed pages re-rendered
- State with Phase 1 complete, Phase 2 partially complete → no re-rendering, only remaining OCR pages processed
- State with `combined_done = true` → application exits 0 immediately, no processing
- Phase 3 always re-runs if `combined_done = false`, even if all per-page markdown exists

## End-to-End Tests

Run with `pytest tests/e2e/` (mocked Ollama by default; `--live-ollama` for real).

### `test_pipeline.py`
- `simple.pdf` full run → exit 0, output markdown file exists, contains `--- PAGE 1 ---`
- `multipage.pdf` full run → output contains 10 page sections in order
- `diagrams.pdf` full run → output contains image references, diagram files exist in `diagrams/`
- `mixed.pdf` full run → diagram pages include image refs, text-only pages do not
- `tables.pdf` full run → output markdown contains markdown tables; no diagram files created; `diagrams/` directory not created
- Re-run on completed `simple.pdf` → exit 0 immediately, no files re-generated
- Re-run on partially completed `multipage.pdf` → only remaining pages processed, final output correct
- `corrupt.pdf` → exit 5
- Missing path argument → exit 1
- Non-existent path → exit 2
- All Ollama instances unreachable (mocked) → exit 4
