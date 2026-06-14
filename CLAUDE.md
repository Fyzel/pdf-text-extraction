# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

Early-stage Python project for PDF text extraction. `main.py` is currently a placeholder. No dependencies or tests have been added yet.

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

Both commit messages and PRs are auto-generated via Ollama (`qwen2.5-vl:7b`). Instance URLs are read from `ollama.json` (see `ollama.sample.json`); falls back to `http://localhost:11434`. Requires `curl` and `jq`.

**Commit messages** — `.git/hooks/prepare-commit-msg` fires on every `git commit`. Generates title (≤100 chars) + body from staged diff, prepended to any message you typed. Skips on merge/squash/empty diff.

**PR creation** — `bin/create-pr` generates title (≤72 chars) + body from commit log and diff, then calls `gh pr create`. Requires `gh`.

```sh
bin/create-pr
```
