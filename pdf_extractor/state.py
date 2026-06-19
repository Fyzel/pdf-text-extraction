"""Persistent processing state management via state.json."""
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

_STATE_FILENAME: str = "state.json"


class StateMismatchError(Exception):
    """Raised when an existing ``state.json`` does not match the current PDF.

    Guards ``--rerun-pages`` and resume against a stale or foreign ``state.json``
    (e.g. a different document with the same file stem, or a PDF whose page count
    changed since the prior run), which would otherwise raise ``KeyError`` while
    indexing :attr:`AppState.pages` or silently reprocess the wrong document.
    """


@dataclass
class PageState:
    """Per-page processing status tracked in state.json.

    :ivar image_done: ``True`` after the page JPEG was saved successfully.
    :vartype image_done: bool
    :ivar image_failed: ``True`` if JPEG rendering failed; page is skipped in
        Phase 2.
    :vartype image_failed: bool
    :ivar ocr_done: ``True`` after the per-page markdown was saved successfully.
    :vartype ocr_done: bool
    :ivar ocr_failed: ``True`` if all Ollama retries were exhausted; page is
        skipped in Phase 3.
    :vartype ocr_failed: bool
    :ivar diagram_count: Number of diagrams cropped from this page.
    :vartype diagram_count: int
    """

    image_done: bool = False
    image_failed: bool = False
    ocr_done: bool = False
    ocr_failed: bool = False
    diagram_count: int = 0


@dataclass
class AppState:
    """Full application state loaded from or written to state.json.

    :ivar pdf_path: Absolute path to the source PDF, stored for identification.
    :vartype pdf_path: str
    :ivar page_count: Total number of pages discovered at startup.
    :vartype page_count: int
    :ivar combined_done: ``True`` after Phase 3 writes the combined markdown file.
    :vartype combined_done: bool
    :ivar pages: Per-page state keyed by 1-based page number string (e.g.
        ``"1"``).
    :vartype pages: dict[str, PageState]
    """

    pdf_path: str
    page_count: int
    combined_done: bool
    pages: dict[str, PageState]


class StateManager:
    """Thread-safe reader and writer for state.json.

    All writes are serialized through a ``threading.Lock`` and performed
    atomically via a ``.tmp`` rename to prevent partial writes.
    """

    def __init__(self, output_dir: Path) -> None:
        """Initialize the state manager.

        :param output_dir: Working directory where ``state.json`` will be
            stored. Required.
        :type output_dir: pathlib.Path
        """
        self._path: Path = output_dir / _STATE_FILENAME
        self._lock: threading.Lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Absolute path to the managed ``state.json`` file.

        :return: Path to ``state.json`` inside the manager's output directory.
        :rtype: pathlib.Path
        """
        return self._path

    def load_or_init(self, pdf_path: Path, page_count: int) -> AppState:
        """Load ``state.json`` if it exists, otherwise create and persist a fresh state.

        A pre-existing ``state.json`` is validated against the current PDF: its
        stored absolute path and page count must match. On a mismatch a
        :class:`StateMismatchError` is raised rather than returning state that
        belongs to a different (or changed) document.

        :param pdf_path: Path to the source PDF; stored as an absolute path.
            Required.
        :type pdf_path: pathlib.Path
        :param page_count: Total number of pages in the PDF. Required.
        :type page_count: int
        :return: Loaded or freshly initialised application state.
        :rtype: AppState
        :raises StateMismatchError: The existing ``state.json`` describes a
            different PDF path or a different page count than the current run.
        """
        if self._path.is_file():
            loaded: AppState = self._load()
            self._validate(loaded, pdf_path, page_count)
            return loaded
        state: AppState = AppState(
            pdf_path=str(pdf_path.resolve()),
            page_count=page_count,
            combined_done=False,
            pages={str(i): PageState() for i in range(1, page_count + 1)},
        )
        self._write(state)
        return state

    def update_page(self, state: AppState, page_num: int, **fields: object) -> None:
        """Update fields on a single page entry and persist atomically under the lock.

        Valid keyword arguments match :class:`PageState` fields:
        ``image_done``, ``image_failed``, ``ocr_done``, ``ocr_failed``,
        ``diagram_count``.

        :param state: Shared application state to mutate. Required.
        :type state: AppState
        :param page_num: 1-based page number to update. Required.
        :type page_num: int
        :param fields: :class:`PageState` field names mapped to their new values.
            Optional; passed as keyword arguments.
        :type fields: object
        :return: ``None``.
        :rtype: None
        """
        with self._lock:
            page: PageState = state.pages[str(page_num)]
            for name, value in fields.items():
                setattr(page, name, value)
            self._write(state)

    def reset_pages(self, state: AppState, page_nums: list[int]) -> None:
        """Clear all processing flags for the given pages and persist atomically.

        For each page, resets ``image_done``, ``image_failed``, ``ocr_done``,
        and ``ocr_failed`` to ``False`` and ``diagram_count`` to ``0``. Also
        clears ``combined_done`` on the whole state so Phase 3 reassembles. Used
        by the ``--rerun-pages`` feature to force reprocessing of specific pages.

        :param state: Shared application state to mutate. Required.
        :type state: AppState
        :param page_nums: 1-based page numbers to reset. Required.
        :type page_nums: list[int]
        :return: ``None``.
        :rtype: None
        """
        with self._lock:
            for page_num in page_nums:
                page: PageState = state.pages[str(page_num)]
                page.image_done = False
                page.image_failed = False
                page.ocr_done = False
                page.ocr_failed = False
                page.diagram_count = 0
            state.combined_done = False
            self._write(state)

    def mark_combined_done(self, state: AppState) -> None:
        """Set ``combined_done = True`` and persist under the lock.

        :param state: Shared application state to mutate. Required.
        :type state: AppState
        :return: ``None``.
        :rtype: None
        """
        with self._lock:
            state.combined_done = True
            self._write(state)

    @staticmethod
    def status(state: AppState) -> Literal["complete", "partial", "not_started"]:
        """Derive the resumability status from the current state.

        :param state: Application state to evaluate. Required.
        :type state: AppState
        :return: One of ``"complete"`` (``combined_done`` is ``True``; nothing
            to do), ``"partial"`` (at least one page attempted — any of
            ``image_done``, ``image_failed``, ``ocr_done``, or ``ocr_failed``),
            or ``"not_started"`` (no page touched yet).
        :rtype: typing.Literal["complete", "partial", "not_started"]
        """
        if state.combined_done:
            return "complete"
        for page in state.pages.values():
            if page.image_done or page.image_failed or page.ocr_done or page.ocr_failed:
                return "partial"
        return "not_started"

    def _validate(self, state: AppState, pdf_path: Path, page_count: int) -> None:
        """Verify a loaded state matches the current PDF, or raise.

        :param state: Application state deserialised from an existing
            ``state.json``. Required.
        :type state: AppState
        :param pdf_path: Path to the PDF for the current run. Required.
        :type pdf_path: pathlib.Path
        :param page_count: Page count discovered for the current run. Required.
        :type page_count: int
        :return: ``None``.
        :rtype: None
        :raises StateMismatchError: The stored absolute PDF path or page count
            does not match the current run.
        """
        current_path: str = str(pdf_path.resolve())
        if state.pdf_path != current_path:
            raise StateMismatchError(
                f"{self._path} belongs to a different PDF "
                f"({state.pdf_path!r}, not {current_path!r}). "
                f"Remove {self._path} to start a fresh run."
            )
        if state.page_count != page_count:
            raise StateMismatchError(
                f"{self._path} page count ({state.page_count}) does not match the "
                f"current PDF ({page_count}); the PDF may have changed. "
                f"Remove {self._path} to start a fresh run."
            )

    def _load(self) -> AppState:
        """Deserialise ``state.json`` from disk into an :class:`AppState`.

        :return: Application state populated from the persisted JSON file.
        :rtype: AppState
        """
        with open(self._path, encoding="utf-8") as fh:
            data: dict = json.load(fh)
        pages: dict[str, PageState] = {
            k: PageState(**v) for k, v in data["pages"].items()
        }
        return AppState(
            pdf_path=data["pdf_path"],
            page_count=data["page_count"],
            combined_done=data["combined_done"],
            pages=pages,
        )

    def _write(self, state: AppState) -> None:
        """Serialise application state to disk atomically via a ``.tmp`` rename.

        :param state: Application state to persist. Required. Caller must hold
            ``self._lock``.
        :type state: AppState
        :return: ``None``.
        :rtype: None
        """
        payload: dict = {
            "pdf_path": state.pdf_path,
            "page_count": state.page_count,
            "combined_done": state.combined_done,
            "pages": {k: asdict(v) for k, v in state.pages.items()},
        }
        tmp: Path = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        tmp.replace(self._path)
