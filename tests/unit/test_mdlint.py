"""Unit tests for pdf_extractor/mdlint.py."""
from pdf_extractor.mdlint import normalize_markdown


def test_stray_alpha_marker_under_ordered_parent_becomes_numeric():
    """A stray ``A.`` marker under an ordered parent is renumbered to ``1.``.

    :return: ``None``.
    :rtype: None
    """
    src = (
        "1. soluta labitur\n2. utroque elaboraret\n3. perfecto vix\n"
        "   A. exerci ridens feugait duo ut"
    )
    expected = (
        "1. soluta labitur\n2. utroque elaboraret\n3. perfecto vix\n"
        "   1. exerci ridens feugait duo ut"
    )
    assert normalize_markdown(src) == expected


def test_under_indented_ordered_subitem_realigned_to_parent_content_column():
    """An under-indented ordered sub-item is re-aligned to nest correctly.

    :return: ``None``.
    :rtype: None
    """
    # Model emits a 2-space indent under "5. "; CommonMark needs 3 to nest.
    src = "4. utroque\n5. perfecto vix\n  1. exerci ridens feugait duo ut"
    expected = "1. utroque\n2. perfecto vix\n   1. exerci ridens feugait duo ut"
    assert normalize_markdown(src) == expected


def test_unordered_subitem_aligned_to_two_space_column():
    """An over-indented unordered sub-item is aligned to the 2-space column.

    :return: ``None``.
    :rtype: None
    """
    src = "- perfecto vix\n    - exerci"
    expected = "- perfecto vix\n  - exerci"
    assert normalize_markdown(src) == expected


def test_multidigit_parent_widens_child_indent():
    """A two-digit ordered parent widens its child's indent to align.

    :return: ``None``.
    :rtype: None
    """
    src = (
        "\n".join(f"{n}. item{n}" for n in range(1, 11))
        + "\n  1. nested under ten"
    )
    lines = normalize_markdown(src).split("\n")
    assert lines[9] == "10. item10"
    assert lines[10] == "    1. nested under ten"  # 4-space indent under "10. "


def test_roman_marker_under_ordered_parent_becomes_numeric():
    """Roman-numeral sub-markers under an ordered parent renumber to ``1.``, ``2.``.

    :return: ``None``.
    :rtype: None
    """
    src = "1. alpha\n   i. beta\n   ii. gamma"
    expected = "1. alpha\n   1. beta\n   2. gamma"
    assert normalize_markdown(src) == expected


def test_ordered_list_renumbered_sequentially():
    """An ordered list with gaps is renumbered sequentially from ``1``.

    :return: ``None``.
    :rtype: None
    """
    src = "1. one\n3. two\n7. three"
    expected = "1. one\n2. two\n3. three"
    assert normalize_markdown(src) == expected


def test_unordered_markers_normalised_to_dash():
    """Mixed unordered markers are normalised to ``-``.

    :return: ``None``.
    :rtype: None
    """
    src = "* one\n+ two\n- three"
    expected = "- one\n- two\n- three"
    assert normalize_markdown(src) == expected


def test_nested_unordered_indentation_preserved():
    """A correctly nested unordered list is left unchanged.

    :return: ``None``.
    :rtype: None
    """
    src = "- perfecto vix\n  - exerci ridens feugait duo ut"
    assert normalize_markdown(src) == src


def test_counters_independent_per_level():
    """Ordered counters reset independently per nesting level.

    :return: ``None``.
    :rtype: None
    """
    src = "1. a\n   A. a1\n   B. a2\n2. b\n   C. b1"
    expected = "1. a\n   1. a1\n   2. a2\n2. b\n   1. b1"
    assert normalize_markdown(src) == expected


def test_blank_line_between_items_keeps_list_open():
    """A blank line between items keeps the list open (no renumber reset).

    :return: ``None``.
    :rtype: None
    """
    src = "1. a\n\n2. b"
    assert normalize_markdown(src) == src


def test_paragraph_breaks_list_and_resets_numbering():
    """A prose paragraph ends the list and resets numbering for the next one.

    :return: ``None``.
    :rtype: None
    """
    src = "1. a\n2. b\n\nSome prose paragraph.\n\n7. c\n9. d"
    expected = "1. a\n2. b\n\nSome prose paragraph.\n\n1. c\n2. d"
    assert normalize_markdown(src) == expected


def test_non_list_content_untouched():
    """Headings, prose, tables, and separators pass through unchanged.

    :return: ``None``.
    :rtype: None
    """
    src = (
        "# Heading\n\nA paragraph of text.\n\n"
        "| col | col |\n|-----|-----|\n| a | b |\n\n"
        "--- PAGE 2 ---"
    )
    assert normalize_markdown(src) == src


def test_emphasis_and_horizontal_rule_not_treated_as_list():
    """Emphasis and a thematic break are not mistaken for list items.

    :return: ``None``.
    :rtype: None
    """
    src = "*emphasis* and **bold**\n\n---\n\ntext"
    assert normalize_markdown(src) == src


def test_idempotent():
    """Normalising already-normalised markdown is a no-op.

    :return: ``None``.
    :rtype: None
    """
    src = "1. a\n   A. a1\n2. b\n* x\n+ y"
    once = normalize_markdown(src)
    assert normalize_markdown(once) == once


def test_empty_string():
    """Empty input returns empty output.

    :return: ``None``.
    :rtype: None
    """
    assert normalize_markdown("") == ""
