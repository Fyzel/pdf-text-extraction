# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

Full pipeline implemented and tested. `main.py` is the entry point. Core modules in `pdf_extractor/`:

| Module | Purpose |
|--------|---------|
| `config.py` | Load `ollama.json`, fallback defaults, validate schema |
| `health.py` | Probe Ollama instances via `GET /api/tags` |
| `state.py` | Thread-safe `state.json` read/write with atomic rename |
| `render.py` | Phase 1 — PDF→JPEG via PyMuPDF, `ProcessPoolExecutor` |
| `ocr.py` | Phase 2 — Ollama OCR, diagram crop, round-robin + retry |
| `combine.py` | Phase 3 — merge per-page `.md` into single output file |
| `cli.py` | Entry point, phases 1–3, exit codes 0–7 |

Test suite: `tests/` — 121 tests across unit, integration, and e2e layers. Run with `pytest tests/`.

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

Note: `ollama-dev.json` is the dev tooling config (gemma4). `ollama.json` is the application config (qwen3-vl). They are separate and both gitignored.

**Commit messages** — `.git/hooks/prepare-commit-msg` fires on every `git commit`. Generates title (≤100 chars) + body from staged diff, prepended to any message you typed. Skips on merge, squash, `git commit -m`, and empty diff.

**PR creation** — `bin/create-pr` generates title (≤72 chars) + body from commit log and diff, then calls `gh pr create`. Requires `gh`.

```sh
bin/create-pr
```
