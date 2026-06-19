# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

Full pipeline implemented and tested. `main.py` is the entry point. Core modules in `pdf_extractor/`:

| Module | Purpose |
|--------|---------|
| `config.py` | Load `ollama.json`, fallback defaults, validate schema |
| `health.py` | Probe Ollama instances via `GET /api/tags` |
| `state.py` | Thread-safe `state.json` read/write with atomic rename; `load_or_init` validates an existing `state.json` against the current PDF path + page count, raising `StateMismatchError` on mismatch (exit 8) |
| `render.py` | Phase 1 — PDF→JPEG via PyMuPDF, `ProcessPoolExecutor` |
| `ocr.py` | Phase 2 — Ollama OCR, diagram crop, round-robin + retry; skips blank pages (hybrid text/drawing + pixel-whiteness check) before any OCR call; per page, corrects heading levels via `headings`, reflows prose + strips stray emphasis via `reflow`, normalises list markdown via `mdlint`, replaces model tables with PDF-extracted tables via `tables`, (with `--include-links`) rewrites plain anchor text as Markdown links via `links`, and (with `--include-comments`) appends PDF annotations via `annotations` before writing |
| `headings.py` | Phase 2 helper — derive a document-wide heading size→level scale from PDF font spans (`get_text("dict")`), then per page relevel/promote/demote model headings to match the PDF hierarchy; skips lines inside fenced code blocks; opt-out when the PDF has no heading scale (scanned) |
| `reflow.py` | Phase 2 helper — join soft-wrapped prose lines into one line per paragraph and strip emphasis that wraps a whole paragraph; leaves headings, lists, tables, blockquotes, and fenced code untouched |
| `mdlint.py` | Phase 2 helper — normalise list markers, ordered numbering, and nested-item indentation in per-page markdown (CommonMark) |
| `tables.py` | Phase 2 helper — extract tables from the PDF via PyMuPDF `find_tables`, render as aligned Markdown, splice over the model's table blocks |
| `links.py` | Phase 2 helper — extract external URI hyperlinks from the PDF via PyMuPDF `page.get_links()`, recover each link's anchor text, splice `[text](uri)` over the matching plain text; skips fenced code, table rows, and existing links; opt-in via `--include-links` |
| `annotations.py` | Phase 2 helper — extract text-bearing PDF annotations (comments) via PyMuPDF `page.annots()`, render as a `## Comments` section; opt-in via `--include-comments` |
| `combine.py` | Phase 3 — merge per-page `.md` into single output file |
| `cli.py` | Entry point, phases 1–3, exit codes 0–8 (8 = existing `state.json` does not match the current PDF path/page count); flags `--dpi-scale N`, `--include-comments`, `--include-links`, and `--rerun-pages SPEC` (e.g. `3,5,7-9`) which archives a selected page's image/diagrams/markdown and the combined output under `<stem>/_archive/vN/` (moved, not deleted), resets their state via `StateManager.reset_pages`, then reprocesses and reassembles |

Test suite: `tests/` — unit, integration, and e2e layers. Run with `pytest tests/`.

Clean manual test-run output (generated `<stem>/` dirs and `<stem>.md` under `tests/data/`) with `bin/clean-test-data` — portable POSIX `sh` for Git Bash, Linux, and macOS.

Application dependencies in `requirements.txt`: PyMuPDF, pytest, pytest-mock, pylint, bandit, pre-commit, PyYAML.

## Pre-commit Hooks

Install hooks after cloning:

```sh
pre-commit install
```

Active hooks:

| Hook | Purpose |
|------|---------|
| `actionlint` | GitHub Actions workflow linting |
| `trivy-secrets` | Secret detection across all files |
| `pylint` | Python linting |
| `bandit` | Python security static analysis |

## Environment

Python 3.14 venv at `.venv/`. Activate before running anything:

```powershell
.venv\Scripts\Activate.ps1
```

## Running

```powershell
python main.py
```

## Ollama Automation

Both commit messages and PRs are auto-generated via Ollama (`gemma4:latest`). Instance URLs are read from `ollama-dev.json` (see `ollama-dev.sample.json`); falls back to `http://localhost:11434`. Requires `curl` and `jq`.

Note: `ollama-dev.json` is the dev tooling config (gemma4). `ollama.json` is the application config (qwen2.5vl). They are separate and both gitignored.

**Commit messages** — `.git/hooks/prepare-commit-msg` fires on every `git commit`. Generates title (≤100 chars) + body from staged diff, prepended to any message you typed. Skips on merge, squash, `git commit -m`, and empty diff.

**PR creation** — `bin/create-pr [remote-branch]` generates title (≤72 chars) + body from commit log and diff, then calls `gh pr create`. Requires `gh`. The target branch may be passed as the first argument; omit it to pick interactively from the remote branch list.

```sh
bin/create-pr        # prompt for target branch
bin/create-pr dev    # target dev, no prompt
```
