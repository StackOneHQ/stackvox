"""Tests for stackvox.text — speech-text normalization."""

from stackvox.text import (
    apply_pronunciations,
    decimals_to_words,
    ensure_terminal_stop,
    expand_units,
    markdown_to_paragraphs,
    normalize_for_speech,
    shape_pauses,
    strip_emoji,
    strip_thousands_separators,
)

# --- numbers ---------------------------------------------------------------


def test_thousands_separators_removed():
    assert strip_thousands_separators("1,198.9 and 2,294.7") == "1198.9 and 2294.7"


def test_decimals_spoken_digit_by_digit():
    assert decimals_to_words("1198.9") == "1198 point 9"
    assert decimals_to_words("770.72") == "770 point 7 2"


def test_year_is_not_a_decimal():
    assert decimals_to_words("built in 2023") == "built in 2023"


# --- units -----------------------------------------------------------------


def test_currency_and_pence():
    assert expand_units("£1.63") == "1.63 pounds"
    assert expand_units("25p per litre") == "25 pence per litre"
    assert expand_units("167.14p") == "167.14 pence"


def test_kwh_spaced_and_glued_units():
    assert expand_units("13.5 kWh") == "13.5 kilowatt hours"
    assert expand_units("per kWh") == "per kilowatt hours"
    assert expand_units("65kg") == "65 kilograms"
    assert expand_units("a 4km trip") == "a 4 kilometres trip"


def test_math_symbols():
    assert expand_units("770 ÷ 2 × 3 = 5").strip() == "770 divided by 2 times 3 equals 5"


def test_unknown_locale_falls_back_to_en_gb():
    assert expand_units("£5", locale="xx-YY") == "5 pounds"


# --- pronunciations --------------------------------------------------------


def test_pronunciations_whole_word_case_insensitive():
    m = {"Redis": "Reddis", "agy": "antigravity"}
    assert apply_pronunciations("Redis and AGY", m) == "Reddis and antigravity"


def test_pronunciations_do_not_match_substrings():
    # "StackOne" must not partially rewrite "StackOneHQ"
    assert apply_pronunciations("StackOneHQ", {"StackOne": "stack one"}) == "StackOneHQ"


# --- pauses ----------------------------------------------------------------


def test_dash_becomes_ellipsis_pause():
    assert shape_pauses("in control - once I started") == "in control ... once I started"


def test_comma_inserted_before_paren():
    assert shape_pauses("Claude Code (CC)") == "Claude Code, (CC)"


def test_ensure_terminal_stop():
    assert ensure_terminal_stop("About") == "About."
    assert ensure_terminal_stop("Ready?") == "Ready?"
    assert ensure_terminal_stop("A list:") == "A list:"


# --- emoji -----------------------------------------------------------------


def test_strip_emoji():
    assert (
        strip_emoji("done ✅ and 🚀 go").replace("  ", " ").strip()
        == "done  and  go".replace("  ", " ").strip()
    )


# --- markdown --------------------------------------------------------------


def test_markdown_headings_and_lists_are_own_paragraphs():
    md = "# Title\n\nBody line.\n\n- one\n- two"
    assert markdown_to_paragraphs(md) == ["Title", "Body line.", "one", "two"]


def test_markdown_links_and_code_reduced_to_text():
    md = "See [the docs](https://x.example) and run `make build`."
    assert markdown_to_paragraphs(md) == ["See the docs and run make build."]


def test_fenced_code_dropped():
    md = "Before.\n\n```py\nprint('hi')\n```\n\nAfter."
    assert markdown_to_paragraphs(md) == ["Before.", "After."]


def test_tables_dropped_by_default():
    md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    assert markdown_to_paragraphs(md) == []


def test_tables_as_csv():
    md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    assert markdown_to_paragraphs(md, tables="csv") == ["A, B", "1, 2"]


def test_tables_without_outer_pipes_are_recognized():
    md = "A | B\n--- | ---\n1 | 2"
    assert markdown_to_paragraphs(md) == []
    assert markdown_to_paragraphs(md, tables="csv") == ["A, B", "1, 2"]


def test_code_identifiers_are_not_treated_as_emphasis():
    # __init__, **kwargs and *args are not emphasis in CommonMark — keep them.
    md = "The `__init__` method takes **kwargs and *args."
    assert markdown_to_paragraphs(md) == ["The __init__ method takes **kwargs and *args."]


def test_paired_emphasis_markers_are_stripped():
    md = "This is **bold** and *italic* text."
    assert markdown_to_paragraphs(md) == ["This is bold and italic text."]


def test_angle_brackets_in_prose_are_kept():
    # Only real HTML tags are stripped, not comparisons.
    assert markdown_to_paragraphs("Voltage: x < y and y > z.") == ["Voltage: x < y and y > z."]
    assert markdown_to_paragraphs("Wrap <b>x</b> now.") == ["Wrap x now."]


def test_bare_urls_and_reference_definitions_are_dropped():
    md = "See [the docs][docs] here.\n\n[docs]: https://example.com/path"
    assert markdown_to_paragraphs(md) == ["See the docs here."]
    # whitespace left by the stripped URL is collapsed by the full pipeline
    assert normalize_for_speech("Go to https://example.com now.") == "Go to now."


def test_setext_underline_is_dropped():
    assert markdown_to_paragraphs("The Verdict\n===========\n\nBody.") == ["The Verdict", "Body."]


def test_task_list_checkboxes_are_stripped():
    assert markdown_to_paragraphs("- [ ] todo\n- [x] done") == ["todo", "done"]


def test_nested_blockquote_markers_are_stripped():
    assert markdown_to_paragraphs("> > deeply quoted") == ["deeply quoted"]


# --- end to end ------------------------------------------------------------


def test_currency_runs_before_decimal_split():
    # The ordering bug guard: £1.63 -> "1.63 pounds" -> "1 point 6 3 pounds",
    # never "£1 point 63".
    out = normalize_for_speech("It cost £1.63 today.", markdown=False)
    assert "1 point 6 3 pounds" in out
    assert "£" not in out


def test_blog_style_normalization():
    md = "## The Verdict\n\nIt drew 770.72 kWh for £192.68 - a loss."
    out = normalize_for_speech(md, pronunciations={}, tables="drop")
    assert out == ("The Verdict.\nIt drew 770 point 7 2 kilowatt hours for 192 point 6 8 pounds ... a loss.")


def test_speaklast_style_normalization():
    md = "I updated `Redis` ✅ and the CLI (v2)."
    out = normalize_for_speech(
        md,
        pronunciations={"Redis": "Reddis", "CLI": "C L I"},
        tables="csv",
        strip_emoji=True,
    )
    assert "Reddis" in out
    assert "C L I, (v2)" in out
    assert "✅" not in out


def test_plain_text_mode_splits_paragraphs():
    out = normalize_for_speech("First para.\n\nSecond para.", markdown=False)
    assert out == "First para.\nSecond para."


def test_dev_acronyms_are_spelled_out():
    assert normalize_for_speech("Use the CLI.", markdown=False) == "Use the C L I."
    assert "C L I" in normalize_for_speech("the cli tool", markdown=False)  # lowercase too
    assert "A W S" in normalize_for_speech("deploy to AWS", markdown=False)
    assert "U R I" in normalize_for_speech("parse the URI", markdown=False)
    assert "C I C D" in normalize_for_speech("the CI/CD pipeline", markdown=False)


def test_dev_terms_leave_correctly_voiced_acronyms_alone():
    out = normalize_for_speech("Send JSON over HTTP to the API.", markdown=False)
    assert "JSON" in out
    assert "HTTP" in out
    assert "API" in out


def test_dev_terms_can_be_disabled():
    assert normalize_for_speech("Use the CLI.", markdown=False, dev_terms=False) == "Use the CLI."


def test_caller_pronunciations_override_dev_terms():
    out = normalize_for_speech("Use the CLI.", markdown=False, pronunciations={"CLI": "command line"})
    assert "command line" in out
    assert "C L I" not in out
