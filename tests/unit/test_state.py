"""Unit tests for pdf_extractor/state.py."""
import threading
import pytest
from pathlib import Path

from pdf_extractor.state import (
    AppState,
    PageState,
    StateManager,
    StateMismatchError,
)


def _make_sm(tmp_path: Path) -> tuple[StateManager, AppState]:
    sm = StateManager(tmp_path)
    st = sm.load_or_init(tmp_path / "fake.pdf", page_count=3)
    return sm, st


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_init_creates_state_json(tmp_path):
    sm, _ = _make_sm(tmp_path)
    assert sm.path.is_file()


def test_init_page_count(tmp_path):
    _, st = _make_sm(tmp_path)
    assert st.page_count == 3
    assert set(st.pages.keys()) == {"1", "2", "3"}


def test_init_all_pages_blank(tmp_path):
    _, st = _make_sm(tmp_path)
    for pg in st.pages.values():
        assert not pg.image_done
        assert not pg.image_failed
        assert not pg.ocr_done
        assert not pg.ocr_failed
        assert pg.diagram_count == 0


def test_init_combined_done_false(tmp_path):
    _, st = _make_sm(tmp_path)
    assert not st.combined_done


# ---------------------------------------------------------------------------
# Load existing
# ---------------------------------------------------------------------------

def test_load_existing_state(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.update_page(st, 1, image_done=True)
    sm2 = StateManager(tmp_path)
    st2 = sm2.load_or_init(tmp_path / "fake.pdf", page_count=3)
    assert st2.pages["1"].image_done


# ---------------------------------------------------------------------------
# Validation against the current PDF (issue #71)
# ---------------------------------------------------------------------------

def test_load_mismatched_page_count_raises(tmp_path):
    # state.json from a 3-page run; the PDF now reports 5 pages.
    _make_sm(tmp_path)
    sm2 = StateManager(tmp_path)
    with pytest.raises(StateMismatchError):
        sm2.load_or_init(tmp_path / "fake.pdf", page_count=5)


def test_load_mismatched_pdf_path_raises(tmp_path):
    # state.json belongs to fake.pdf; a different file with the same stem reuses it.
    _make_sm(tmp_path)
    sm2 = StateManager(tmp_path)
    with pytest.raises(StateMismatchError):
        sm2.load_or_init(tmp_path / "other.pdf", page_count=3)


def test_load_matching_pdf_does_not_raise(tmp_path):
    _make_sm(tmp_path)
    sm2 = StateManager(tmp_path)
    st2 = sm2.load_or_init(tmp_path / "fake.pdf", page_count=3)
    assert st2.page_count == 3


def test_fresh_init_skips_validation(tmp_path):
    # No existing state.json → no validation, any page count is accepted.
    sm = StateManager(tmp_path)
    st = sm.load_or_init(tmp_path / "fake.pdf", page_count=9)
    assert st.page_count == 9


# ---------------------------------------------------------------------------
# reset_pages
# ---------------------------------------------------------------------------

def test_reset_pages_clears_flags(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.update_page(st, 2, image_done=True, ocr_done=True, diagram_count=3)
    sm.reset_pages(st, [2])
    pg = st.pages["2"]
    assert not pg.image_done
    assert not pg.image_failed
    assert not pg.ocr_done
    assert not pg.ocr_failed
    assert pg.diagram_count == 0


def test_reset_pages_clears_combined_done(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.mark_combined_done(st)
    assert st.combined_done
    sm.reset_pages(st, [1])
    assert not st.combined_done


def test_reset_pages_leaves_other_pages(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.update_page(st, 1, image_done=True, ocr_done=True)
    sm.update_page(st, 3, image_done=True, ocr_done=True)
    sm.reset_pages(st, [3])
    assert st.pages["1"].image_done and st.pages["1"].ocr_done
    assert not st.pages["3"].image_done and not st.pages["3"].ocr_done


def test_reset_pages_persists(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.update_page(st, 2, image_done=True, ocr_done=True)
    sm.reset_pages(st, [2])
    st2 = StateManager(tmp_path).load_or_init(tmp_path / "fake.pdf", page_count=3)
    assert not st2.pages["2"].image_done
    assert not st2.combined_done


# ---------------------------------------------------------------------------
# Status detection
# ---------------------------------------------------------------------------

def test_status_not_started(tmp_path):
    sm, st = _make_sm(tmp_path)
    assert sm.status(st) == "not_started"


def test_status_partial_image_done(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.update_page(st, 1, image_done=True)
    assert sm.status(st) == "partial"


def test_status_partial_image_failed(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.update_page(st, 1, image_failed=True)
    assert sm.status(st) == "partial"


def test_status_partial_ocr_done(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.update_page(st, 1, ocr_done=True)
    assert sm.status(st) == "partial"


def test_status_partial_ocr_failed(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.update_page(st, 1, ocr_failed=True)
    assert sm.status(st) == "partial"


def test_status_complete(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.mark_combined_done(st)
    assert sm.status(st) == "complete"


# ---------------------------------------------------------------------------
# update_page
# ---------------------------------------------------------------------------

def test_update_page_image_done(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.update_page(st, 2, image_done=True)
    assert st.pages["2"].image_done


def test_update_page_diagram_count(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.update_page(st, 1, ocr_done=True, diagram_count=3)
    assert st.pages["1"].diagram_count == 3


def test_update_page_persists(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.update_page(st, 3, ocr_failed=True)
    sm2 = StateManager(tmp_path)
    st2 = sm2.load_or_init(tmp_path / "fake.pdf", 3)
    assert st2.pages["3"].ocr_failed


# ---------------------------------------------------------------------------
# mark_combined_done
# ---------------------------------------------------------------------------

def test_mark_combined_done(tmp_path):
    sm, st = _make_sm(tmp_path)
    sm.mark_combined_done(st)
    assert st.combined_done
    sm2 = StateManager(tmp_path)
    st2 = sm2.load_or_init(tmp_path / "fake.pdf", 3)
    assert st2.combined_done


# ---------------------------------------------------------------------------
# Concurrent writes
# ---------------------------------------------------------------------------

def test_concurrent_writes_no_corruption(tmp_path):
    sm, st = _make_sm(tmp_path)
    errors: list[Exception] = []

    def worker(page_num: int) -> None:
        try:
            sm.update_page(st, page_num, image_done=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert sm.path.is_file()
    # reload and verify all three pages set
    sm2 = StateManager(tmp_path)
    st2 = sm2.load_or_init(tmp_path / "fake.pdf", 3)
    for i in range(1, 4):
        assert st2.pages[str(i)].image_done
