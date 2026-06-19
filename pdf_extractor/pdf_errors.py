"""Shared exception set for PyMuPDF and PDF I/O guards.

The extraction helpers (heading, table, link, annotation, diagram, blank-page)
must never let a single malformed or unreadable page abort the whole run, so
each wraps its PyMuPDF work in a guard that falls back to a safe default. They
all need the same set of exception types, kept here as one source of truth.

PyMuPDF raises across two families: document-level errors
(``FileDataError``, ``FileNotFoundError``) subclass :class:`RuntimeError`, while
low-level MuPDF errors (``FzErrorSystem`` and siblings) subclass
``pymupdf.mupdf.FzErrorBase`` → :class:`Exception` directly. :data:`PDF_ERRORS`
covers both, plus the standard I/O, value, and indexing errors these helpers can
hit.
"""
import pymupdf.mupdf

#: Exception types PDF reading, rendering, and parsing realistically raise.
#: Catch this tuple (never bare :class:`Exception`) in per-page guards.
PDF_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    OSError,
    ValueError,
    IndexError,
    KeyError,
    pymupdf.mupdf.FzErrorBase,
)
