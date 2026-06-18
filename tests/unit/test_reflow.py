"""Unit tests for pdf_extractor/reflow.py — prose reflow and emphasis stripping."""
from pdf_extractor.reflow import reflow_prose


# ---------------------------------------------------------------------------
# Reflow — joining soft-wrapped prose
# ---------------------------------------------------------------------------

def test_joins_wrapped_paragraph():
    assert reflow_prose("one two\nthree four") == "one two three four"


def test_blank_line_separates_paragraphs():
    assert reflow_prose("para one line\n\npara two line") == "para one line\n\npara two line"


def test_multiple_wrapped_lines_join():
    assert reflow_prose("a\nb\nc") == "a b c"


def test_trailing_and_leading_whitespace_collapsed():
    assert reflow_prose("  one  \n  two  ") == "one two"


# ---------------------------------------------------------------------------
# Structural lines are boundaries, left intact
# ---------------------------------------------------------------------------

def test_heading_is_boundary():
    assert reflow_prose("# Title\nbody one\nbody two") == "# Title\nbody one body two"


def test_list_items_untouched():
    src = "- item one\n- item two"
    assert reflow_prose(src) == src


def test_ordered_list_untouched():
    src = "1. first\n2. second"
    assert reflow_prose(src) == src


def test_table_rows_untouched():
    src = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert reflow_prose(src) == src


def test_blockquote_untouched():
    src = "> quoted line one\n> quoted line two"
    assert reflow_prose(src) == src


def test_code_fence_content_untouched():
    src = "```\ncode line one\ncode line two\n```"
    assert reflow_prose(src) == src


def test_thematic_break_untouched():
    assert reflow_prose("text above\n\n---\n\ntext below") == "text above\n\n---\n\ntext below"


# ---------------------------------------------------------------------------
# Whole-paragraph emphasis stripping
# ---------------------------------------------------------------------------

def test_strips_single_line_wrap():
    assert reflow_prose("*whole line emphasised*") == "whole line emphasised"


def test_strips_underscore_wrap():
    assert reflow_prose("_whole line_") == "whole line"


def test_strips_double_asterisk_wrap():
    assert reflow_prose("**whole bold line**") == "whole bold line"


def test_strips_wrap_then_joins_unwrapped_line():
    # Model wrapped only the first sentence on its own line (issue #63 pattern).
    assert reflow_prose("*First sentence.*\nSecond sentence.") == "First sentence. Second sentence."


def test_strips_multiline_wrap():
    # Wrap markers only on the first and last line of a multi-line paragraph.
    assert reflow_prose("*First part\nsecond part*") == "First part second part"


def test_preserves_inline_emphasis():
    assert reflow_prose("plain with *one* emphasised word") == "plain with *one* emphasised word"


def test_does_not_strip_when_interior_marker():
    # Two separate emphasised spans — not a whole-paragraph wrap.
    src = "*one* and *two*"
    assert reflow_prose(src) == src


def test_empty_input():
    assert reflow_prose("") == ""
