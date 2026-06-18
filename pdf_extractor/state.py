"""Persistent processing state management via state.json."""
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

_STATE_FILENAME: str = "state.json"


@dataclass
class PageState:
    """Per-page processing status tracked in state.json.

    Attributes:
        image_done: ``True`` after the page JPEG was saved successfully.
        image_failed: ``True`` if JPEG rendering failed; page is skipped in Phase 2.
        ocr_done: ``True`` after the per-page markdown was saved successfully.
        ocr_failed: ``True`` if all Ollama retries were exhausted; page is skipped in Phase 3.
        diagram_count: Number of diagrams cropped from this page.
    """

    image_done: bool = False
    image_failed: bool = False
    ocr_done: bool = False
    ocr_failed: bool = False
    diagram_count: int = 0


@dataclass
class AppState:
    """Full application state loaded from or written to state.json.

    Attributes:
        pdf_path: Absolute path to the source PDF, stored for identification.
        page_count: Total number of pages discovered at startup.
        combined_done: ``True`` after Phase 3 writes the combined markdown file.
        pages: Per-page state keyed by 1-based page number string (e.g. ``"1"``).
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

        Args:
            output_dir: Working directory where ``state.json`` will be stored.
        """
        self._path: Path = output_dir / _STATE_FILENAME
        self._lock: threading.Lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Absolute path to ``state.json``."""
        return self._path

    def load_or_init(self, pdf_path: Path, page_count: int) -> AppState:
        """Load ``state.json`` if it exists, otherwise create and persist a fresh state.

        Args:
            pdf_path: Path to the source PDF; stored as an absolute path.
            page_count: Total number of pages in the PDF.

        Returns:
            Loaded or freshly initialised AppState.
        """
        if self._path.is_file():
            return self._load()
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

        Valid keyword arguments match ``PageState`` fields:
        ``image_done``, ``image_failed``, ``ocr_done``, ``ocr_failed``, ``diagram_count``.

        Args:
            state: Shared AppState instance to mutate.
            page_num: 1-based page number to update.
            **fields: ``PageState`` field names and their new values.
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

        Args:
            state: Shared AppState instance to mutate.
            page_nums: 1-based page numbers to reset.
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

        Args:
            state: Shared AppState instance to mutate.
        """
        with self._lock:
            state.combined_done = True
            self._write(state)

    @staticmethod
    def status(state: AppState) -> Literal["complete", "partial", "not_started"]:
        """Derive the resumability status from the current state.

        Args:
            state: AppState to evaluate.

        Returns:
            - ``"complete"``: ``combined_done`` is ``True``; nothing to do.
            - ``"partial"``: at least one page has been attempted (any of
              ``image_done``, ``image_failed``, ``ocr_done``, or ``ocr_failed``).
            - ``"not_started"``: no page has been touched yet.
        """
        if state.combined_done:
            return "complete"
        for page in state.pages.values():
            if page.image_done or page.image_failed or page.ocr_done or page.ocr_failed:
                return "partial"
        return "not_started"

    def _load(self) -> AppState:
        """Deserialise ``state.json`` from disk into an AppState.

        Returns:
            AppState populated from the persisted JSON file.
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
        """Serialise AppState to disk atomically via a ``.tmp`` rename.

        Args:
            state: AppState to persist. Caller must hold ``self._lock``.
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
