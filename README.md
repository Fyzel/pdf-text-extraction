# pdf-text-extraction

A command line tool that converts PDF files to Markdown using AI-powered OCR via [Ollama](https://ollama.com). Each page is extracted as text, diagrams are saved as separate images with Markdown references, and tables are rendered as Markdown table syntax — all combined into a single `.md` file.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![CI (main)](https://img.shields.io/github/actions/workflow/status/Fyzel/pdf-text-extraction/tests.yml?branch=main&label=CI%20%28main%29)](https://github.com/Fyzel/pdf-text-extraction/actions/workflows/tests.yml)
[![CI (dev)](https://img.shields.io/github/actions/workflow/status/Fyzel/pdf-text-extraction/tests.yml?branch=dev&label=CI%20%28dev%29)](https://github.com/Fyzel/pdf-text-extraction/actions/workflows/tests.yml)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-2cbe4e?logo=dependabot)](https://github.com/Fyzel/pdf-text-extraction/network/updates)

## Features

- AI OCR via `qwen2.5vl` (local or remote Ollama instance)
- Diagram extraction — figures cropped at their exact PDF image bounds and saved as image files
- Table recognition — tables read directly from the PDF (PyMuPDF) and rendered as Markdown table syntax, not images
- Markdown list normalisation — sub-bullets get valid CommonMark markers and indentation so nested lists render correctly
- Blank-page skipping — empty pages are detected and skipped, avoiding a wasted OCR call
- Parallel PDF rendering across all available CPU cores
- Concurrent OCR across multiple Ollama instances
- Resumable — interrupted runs continue from where they left off

## Requirements

- Python 3.14+
- [Ollama](https://ollama.com) running locally or on a remote host
- `qwen2.5vl` model pulled in Ollama

```sh
ollama pull qwen2.5vl:7b    # recommended — lower memory pressure
```

## Installation

```sh
git clone https://github.com/Fyzel/pdf-text-extraction.git
cd pdf-text-extraction
python -m venv .venv
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```sh
source .venv/bin/activate
pip install -r requirements.txt
```

Install pre-commit hooks:

```sh
pre-commit install
```

## Configuration

Copy the sample config and edit it to point at your Ollama instance:

```sh
cp ollama.sample.json ollama.json
```

`ollama.json` is gitignored — it never gets committed.

### `ollama.json` schema

```json
{
  "max_render_workers": 4,
  "instances": [
    { "url": "http://localhost:11434", "model": "qwen2.5vl:7b" }
  ]
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `max_render_workers` | No | All CPU cores | Cap on parallel PDF rendering processes |
| `instances[].url` | Yes | — | Ollama base URL |
| `instances[].model` | No | `qwen2.5vl:7b` | Model for this instance |

Multiple instances are supported — pages are distributed across them concurrently. See the [Configuration wiki page](https://github.com/Fyzel/pdf-text-extraction/wiki/Configuration) for full details.

## Usage

```sh
python main.py /path/to/document.pdf [--dpi-scale N]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `<pdf_path>` | Yes | — | Path to the source PDF |
| `--dpi-scale N` | No | `2.0` | Page render scale factor (`2.0` ≈ 144 DPI). Raise for sharper images and OCR of fine print, at the cost of larger images and slower rendering. Applies to both full-page renders and diagram crops. |

```sh
# render at ~288 DPI for clearer capture of dense or small text
python main.py /path/to/document.pdf --dpi-scale 4
```

Output is written alongside the PDF:

```
/path/to/
├── document.pdf          ← untouched
├── document.md           ← combined OCR output
└── document/
    ├── state.json        ← resume state
    ├── pages/            ← per-page JPEGs and Markdown
    └── diagrams/         ← extracted diagram images
```

Re-running the same command resumes from where processing left off.

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |
| 1 | Missing PDF path, or invalid command-line arguments |
| 2 | PDF file not found |
| 3 | PDF file not readable |
| 4 | No Ollama instances reachable |
| 5 | All pages failed image rendering |
| 6 | All pages failed OCR |
| 7 | Output file write error |

See the [Error Codes wiki page](https://github.com/Fyzel/pdf-text-extraction/wiki/Error-Codes) for remediation steps.

## Testing

```sh
pytest tests/
```

174 tests across unit, integration, and end-to-end layers. No real Ollama instance required — all HTTP calls are mocked. A further 6 live tests are opt-in (see below).

To also run live tests against a real Ollama instance:

```sh
pytest -m live
```

Live tests require `qwen2.5vl:7b` reachable at the URL configured in `ollama.json`. They are automatically skipped if no instance is reachable.

### Cleaning manual test runs

Running `main.py` against the sample PDFs in `tests/data/` leaves generated output behind (`<stem>/` working dirs and `<stem>.md` files). Remove it with:

```sh
bin/clean-test-data
```

Portable POSIX `sh` — works under Git Bash (Windows), Linux, and macOS. Source PDFs and checked-in `*-expected.md` fixtures are left untouched.

See the [Testing wiki page](https://github.com/Fyzel/pdf-text-extraction/wiki/Testing) for details.

## Contributing

Pull requests are reviewed by the code owners listed in [`.github/CODEOWNERS`](.github/CODEOWNERS). GitHub automatically requests their review on any PR that touches owned paths, so the relevant owner is added as a reviewer for you.

## Documentation

Full end-user documentation is available in the [project wiki](https://github.com/Fyzel/pdf-text-extraction/wiki).

## License

GNU Affero General Public License v3.0 — see [LICENSE.txt](LICENSE.txt).

This project uses [PyMuPDF](https://pymupdf.readthedocs.io/) which is also licensed under AGPL-3.0.
