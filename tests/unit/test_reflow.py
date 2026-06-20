"""Unit tests for pdf_extractor/reflow.py — prose reflow and emphasis stripping."""
from pdf_extractor.reflow import reflow_prose


# ---------------------------------------------------------------------------
# Reflow — joining soft-wrapped prose
# ---------------------------------------------------------------------------

def test_joins_wrapped_paragraph():
    """Two soft-wrapped lines join into one paragraph line.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("one two\nthree four") == "one two three four"


def test_blank_line_separates_paragraphs():
    """A blank line keeps two paragraphs separate.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("para one line\n\npara two line") == "para one line\n\npara two line"


def test_multiple_wrapped_lines_join():
    """Several consecutive soft-wrapped lines join into one.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("a\nb\nc") == "a b c"


def test_trailing_and_leading_whitespace_collapsed():
    """Leading/trailing whitespace on joined lines is collapsed.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("  one  \n  two  ") == "one two"


# ---------------------------------------------------------------------------
# Structural lines are boundaries, left intact
# ---------------------------------------------------------------------------

def test_heading_is_boundary():
    """A heading ends the paragraph and is emitted verbatim.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("# Title\nbody one\nbody two") == "# Title\nbody one body two"


def test_list_items_untouched():
    """Unordered list items are left untouched.

    :return: ``None``.
    :rtype: None
    """
    src = "- item one\n- item two"
    assert reflow_prose(src) == src


def test_ordered_list_untouched():
    """Ordered list items are left untouched.

    :return: ``None``.
    :rtype: None
    """
    src = "1. first\n2. second"
    assert reflow_prose(src) == src


def test_table_rows_untouched():
    """Table rows are left untouched.

    :return: ``None``.
    :rtype: None
    """
    src = "| a | b |\n|---|---|\n| 1 | 2 |"
    assert reflow_prose(src) == src


def test_blockquote_untouched():
    """Blockquote lines are left untouched.

    :return: ``None``.
    :rtype: None
    """
    src = "> quoted line one\n> quoted line two"
    assert reflow_prose(src) == src


def test_code_fence_content_untouched():
    """Lines inside a fenced code block are left untouched.

    :return: ``None``.
    :rtype: None
    """
    src = "```\ncode line one\ncode line two\n```"
    assert reflow_prose(src) == src


def test_different_marker_does_not_reopen_reflow():
    """A different fence marker inside a block does not re-enable reflow.

    :return: ``None``.
    :rtype: None
    """
    # A ~~~ line inside a ```-opened block must not close it; the lines after it
    # are still code and must stay on their own lines, not be reflowed.
    src = "```\n~~~\nsoft line one\nsoft line two\n```"
    assert reflow_prose(src) == src


def test_shorter_marker_does_not_reopen_reflow():
    """A shorter fence run inside a longer block does not re-enable reflow.

    :return: ``None``.
    :rtype: None
    """
    # Closing run must be at least as long as the 4-backtick opening run.
    src = "````\n```\nsoft line one\nsoft line two\n````"
    assert reflow_prose(src) == src


def test_thematic_break_untouched():
    """A thematic break acts as a boundary and is emitted verbatim.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("text above\n\n---\n\ntext below") == "text above\n\n---\n\ntext below"


# ---------------------------------------------------------------------------
# Whole-paragraph emphasis stripping
# ---------------------------------------------------------------------------

def test_strips_single_line_wrap():
    """Emphasis wrapping a whole single line is stripped.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("*whole line emphasised*") == "whole line emphasised"


def test_strips_underscore_wrap():
    """Underscore emphasis wrapping a whole line is stripped.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("_whole line_") == "whole line"


def test_strips_double_asterisk_wrap():
    """Double-asterisk bold wrapping a whole line is stripped.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("**whole bold line**") == "whole bold line"


def test_strips_wrap_then_joins_unwrapped_line():
    """A wrapped first sentence is stripped and joined with the next line.

    :return: ``None``.
    :rtype: None
    """
    # Model wrapped only the first sentence on its own line (issue #63 pattern).
    assert reflow_prose("*First sentence.*\nSecond sentence.") == "First sentence. Second sentence."


def test_strips_multiline_wrap():
    """Emphasis spanning a multi-line paragraph is stripped after joining.

    :return: ``None``.
    :rtype: None
    """
    # Wrap markers only on the first and last line of a multi-line paragraph.
    assert reflow_prose("*First part\nsecond part*") == "First part second part"


def test_preserves_inline_emphasis():
    """Genuine inline emphasis on one word is preserved.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("plain with *one* emphasised word") == "plain with *one* emphasised word"


def test_does_not_strip_when_interior_marker():
    """Two separate emphasised spans are not treated as a whole-line wrap.

    :return: ``None``.
    :rtype: None
    """
    # Two separate emphasised spans — not a whole-paragraph wrap.
    src = "*one* and *two*"
    assert reflow_prose(src) == src


def test_empty_input():
    """Empty input returns empty output.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("") == ""


# ---------------------------------------------------------------------------
# Index / table-of-contents entries — not reflowed (issue #85)
# ---------------------------------------------------------------------------

def test_toc_entries_not_merged():
    """Consecutive table-of-contents lines each stay on their own line.

    Each entry ends with a page reference (roman front-matter or arabic), so it
    is a complete line, not soft-wrapped prose, and must not be joined.

    :return: ``None``.
    :rtype: None
    """
    src = (
        "Foreword xi\n"
        "Preface xii\n"
        "Chapter 1: Introduction 1\n"
        "Chapter 2: Hassling AI Prompts with Humor 9"
    )
    assert reflow_prose(src) == src


def test_arabic_page_reference_is_boundary():
    """A line ending in an arabic page number is not merged with the next.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("TOC 4 2\nTOC 5 5") == "TOC 4 2\nTOC 5 5"


def test_roman_page_reference_is_boundary():
    """A line ending in a lowercase roman numeral is not merged with the next.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("Acknowledgements xvi\nAppendix xvii") == (
        "Acknowledgements xvi\nAppendix xvii"
    )


def test_roman_lookalike_word_still_reflows():
    """A prose line ending in a roman-looking word is still treated as prose.

    Words like ``mix`` or ``did`` are not page references, so the soft-wrapped
    line is joined as normal.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("stir the mix\nuntil smooth") == "stir the mix until smooth"


def test_single_letter_pronoun_does_not_block_reflow():
    """A prose line ending in the pronoun ``I`` still reflows.

    :return: ``None``.
    :rtype: None
    """
    assert reflow_prose("that is what I\nbelieve today") == (
        "that is what I believe today"
    )
