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
import contextlib
from collections.abc import Iterator

import pymupdf
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


@contextlib.contextmanager
def open_guarded(pdf_path: str) -> Iterator[pymupdf.Document]:
    """Open a PDF, yield the document, and always close it on exit.

    The ``fitz.open`` call runs inside the guarded block so open-time failures
    propagate to the caller's ``except PDF_ERRORS`` rather than escaping it.
    Close-time errors are swallowed so cleanup can never break a helper's
    "never fail a page" guarantee.

    :param pdf_path: Path to the source PDF file. Required.
    :type pdf_path: str
    :return: Context manager yielding the open document.
    :rtype: Iterator[pymupdf.Document]
    """
    doc: pymupdf.Document | None = None
    try:
        doc = pymupdf.open(pdf_path)
        yield doc
    finally:
        if doc is not None:
            try:
                doc.close()
            except PDF_ERRORS:
                pass  # cleanup must not break the guarantee
