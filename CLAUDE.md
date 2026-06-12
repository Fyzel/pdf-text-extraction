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

## PR Automation

`bin/create-pr` generates PR title and body via Ollama (`gemma4:latest`), then opens the PR with `gh`. Requires `gh`, `curl`, and `jq`.

- Tries remote Ollama at `http://michaels-mac-mini.local:11434` first, falls back to `http://localhost:11434`.

```sh
bin/create-pr
```
