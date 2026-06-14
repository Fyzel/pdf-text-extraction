# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

Early-stage Python project for PDF text extraction. `main.py` is currently a placeholder. Application dependencies (PyMuPDF, Ollama HTTP client) not yet added — `requirements.txt` currently contains dev tooling only (bandit, pylint, pre-commit, PyYAML).

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

Note: `ollama-dev.json` is the dev tooling config (gemma4). `ollama.json` is the application config (qwen2.5-vl). They are separate and both gitignored.

**Commit messages** — `.git/hooks/prepare-commit-msg` fires on every `git commit`. Generates title (≤100 chars) + body from staged diff, prepended to any message you typed. Skips on merge, squash, `git commit -m`, and empty diff.

**PR creation** — `bin/create-pr` generates title (≤72 chars) + body from commit log and diff, then calls `gh pr create`. Requires `gh`.

```sh
bin/create-pr
```
