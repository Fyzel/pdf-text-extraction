"""Unit tests for pdf_extractor/mdlint.py."""
from pdf_extractor.mdlint import normalize_markdown


def test_stray_alpha_marker_under_ordered_parent_becomes_numeric():
    src = (
        "1. soluta labitur\n2. utroque elaboraret\n3. perfecto vix\n"
        "   A. exerci ridens feugait duo ut"
    )
    expected = (
        "1. soluta labitur\n2. utroque elaboraret\n3. perfecto vix\n"
        "   1. exerci ridens feugait duo ut"
    )
    assert normalize_markdown(src) == expected


def test_roman_marker_under_ordered_parent_becomes_numeric():
    src = "1. alpha\n   i. beta\n   ii. gamma"
    expected = "1. alpha\n   1. beta\n   2. gamma"
    assert normalize_markdown(src) == expected


def test_ordered_list_renumbered_sequentially():
    src = "1. one\n3. two\n7. three"
    expected = "1. one\n2. two\n3. three"
    assert normalize_markdown(src) == expected


def test_unordered_markers_normalised_to_dash():
    src = "* one\n+ two\n- three"
    expected = "- one\n- two\n- three"
    assert normalize_markdown(src) == expected


def test_nested_unordered_indentation_preserved():
    src = "- perfecto vix\n  - exerci ridens feugait duo ut"
    assert normalize_markdown(src) == src


def test_counters_independent_per_level():
    src = "1. a\n   A. a1\n   B. a2\n2. b\n   C. b1"
    expected = "1. a\n   1. a1\n   2. a2\n2. b\n   1. b1"
    assert normalize_markdown(src) == expected


def test_blank_line_between_items_keeps_list_open():
    src = "1. a\n\n2. b"
    assert normalize_markdown(src) == src


def test_paragraph_breaks_list_and_resets_numbering():
    src = "1. a\n2. b\n\nSome prose paragraph.\n\n7. c\n9. d"
    expected = "1. a\n2. b\n\nSome prose paragraph.\n\n1. c\n2. d"
    assert normalize_markdown(src) == expected


def test_non_list_content_untouched():
    src = (
        "# Heading\n\nA paragraph of text.\n\n"
        "| col | col |\n|-----|-----|\n| a | b |\n\n"
        "--- PAGE 2 ---"
    )
    assert normalize_markdown(src) == src


def test_emphasis_and_horizontal_rule_not_treated_as_list():
    src = "*emphasis* and **bold**\n\n---\n\ntext"
    assert normalize_markdown(src) == src


def test_idempotent():
    src = "1. a\n   A. a1\n2. b\n* x\n+ y"
    once = normalize_markdown(src)
    assert normalize_markdown(once) == once


def test_empty_string():
    assert normalize_markdown("") == ""