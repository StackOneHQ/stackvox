"""Turn real-world text — Markdown, Claude responses, prose — into something
that *sounds* right when synthesized.

StackVox speaks whatever string it's given; this module is the pre-step that
makes that string speakable: strip Markdown structure, expand units and
numbers, shape pauses, and apply a caller-supplied pronunciation dictionary.

The primary entry point is :func:`normalize_for_speech`. Each stage is also
exposed for composition and testing. See ``docs/speech-normalization.md``.
"""

from __future__ import annotations

import re

__all__ = [
    "normalize_for_speech",
    "markdown_to_paragraphs",
    "strip_emoji",
    "strip_thousands_separators",
    "decimals_to_words",
    "expand_units",
    "apply_pronunciations",
    "shape_pauses",
    "ensure_terminal_stop",
]

# --------------------------------------------------------------------------- #
# Emoji                                                                       #
# --------------------------------------------------------------------------- #

_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # symbols, emoticons, pictographs, supplemental
    "\U00002600-\U000027bf"  # misc symbols + dingbats
    "\U00002190-\U000021ff"  # arrows
    "\U00002b00-\U00002bff"  # misc symbols and arrows
    "\U0000fe00-\U0000fe0f"  # variation selectors
    "]+",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    return _EMOJI.sub("", text)


# --------------------------------------------------------------------------- #
# Units & symbols (locale-keyed)                                              #
# --------------------------------------------------------------------------- #
# (pattern, replacement). Digit-glued units capture the leading digit and
# re-emit it, so "65kg" -> "65 kilograms". IMPORTANT: currency/unit expansion
# must run BEFORE the decimal-point split, or "£1.63" becomes "£1 point 63".

_UNIT_RULES: dict[str, list[tuple[str, str]]] = {
    "en-GB": [
        (r"£\s?(\d[\d,]*(?:\.\d+)?)", r"\1 pounds"),  # £1.63 -> 1.63 pounds
        (r"\bkWh\b", "kilowatt hours"),
        (r"\bMPG\b", "miles per gallon"),
        (r"(\d)\s?kg\b", r"\1 kilograms"),
        (r"(\d)\s?km\b", r"\1 kilometres"),
        (r"(\d)p\b", r"\1 pence"),  # 25p, 167.14p
        (r"\s*÷\s*", " divided by "),
        (r"\s*×\s*", " times "),
        (r"\s*=\s*", " equals "),
    ],
}
DEFAULT_LOCALE = "en-GB"


def expand_units(text: str, locale: str = DEFAULT_LOCALE) -> str:
    for pattern, repl in _UNIT_RULES.get(locale, _UNIT_RULES[DEFAULT_LOCALE]):
        text = re.sub(pattern, repl, text)
    return text


# --------------------------------------------------------------------------- #
# Numbers                                                                     #
# --------------------------------------------------------------------------- #


def strip_thousands_separators(text: str) -> str:
    """1,198.9 -> 1198.9 (so the whole number is read as one, not split)."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def decimals_to_words(text: str) -> str:
    """1198.9 -> "1198 point 9"; 770.72 -> "770 point 7 2". Removes the bare
    "." between digits, which TTS can otherwise read as a full stop."""
    return re.sub(
        r"(\d+)\.(\d+)",
        lambda m: m.group(1) + " point " + " ".join(m.group(2)),
        text,
    )


# --------------------------------------------------------------------------- #
# Pronunciations                                                              #
# --------------------------------------------------------------------------- #


def apply_pronunciations(text: str, mapping: dict[str, str] | None) -> str:
    """Whole-word, case-insensitive spoken-form substitutions."""
    for written, spoken in (mapping or {}).items():
        text = re.sub(rf"\b{re.escape(written)}\b", spoken, text, flags=re.IGNORECASE)
    return text


# Dev acronyms/terms espeak mispronounces — it reads them as a word ("CLI" ->
# "kligh", "AWS" -> "awz", "URI" -> "yuri") instead of spelling them out.
# Applied by default (dev_terms=True), whole-word and case-insensitive, so
# lowercase "cli" is fixed too. Compound keys (ci/cd) precede their parts (ci)
# so the specific form wins. Only terms espeak gets WRONG are here — API, URL,
# JSON, YAML, HTTP, CRUD, nginx, etc. already voice correctly and are left alone.
_DEV_PRONUNCIATIONS: dict[str, str] = {
    "ci/cd": "C I C D",
    "cli": "C L I",
    "ci": "C I",
    "ide": "I D E",
    "aws": "A W S",
    "uri": "U R I",
    "iam": "I A M",
    "saas": "sass",
    "paas": "pass",
    "tui": "T U I",
    "postgresql": "postgres",
    "kubectl": "kube control",
}


# --------------------------------------------------------------------------- #
# Pauses                                                                      #
# --------------------------------------------------------------------------- #


def shape_pauses(text: str) -> str:
    """Give punctuation the beats it deserves: a dash used as punctuation reads
    as a rushed nothing, and a "(" runs onto the previous word."""
    text = text.replace("→", " to ")
    text = re.sub(r"\s*[—–]\s*", " ... ", text)  # em / en dash
    text = re.sub(r"\s+--?\s+", " ... ", text)  # spaced ASCII hyphen(s)
    text = re.sub(r"(\w)\s*\(", r"\1, (", text)  # comma before "("
    return text


def ensure_terminal_stop(text: str) -> str:
    """Guarantee terminal punctuation so a pause lands before the next line —
    StackVox pauses on punctuation, not on line breaks."""
    text = text.rstrip()
    return text if text.endswith((".", "!", "?", ":", "…")) else text + "."


# --------------------------------------------------------------------------- #
# Markdown -> prose                                                           #
# --------------------------------------------------------------------------- #


def _strip_md_inline(line: str) -> str:
    line = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", line)  # images
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # inline links -> text
    line = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", line)  # reference links -> text
    line = re.sub(r"`([^`]+)`", r"\1", line)  # inline code -> text
    line = re.sub(r"<!--.*?-->", "", line)  # HTML comments
    line = re.sub(r"</?[a-zA-Z][^>]*>", "", line)  # real tags + <url> autolinks; leaves "a < b"
    line = re.sub(r"https?://\S+", "", line)  # bare URLs: strip rather than read them aloud
    # Emphasis: only strip *paired, boundary-flanked* markers, so code-ish tokens
    # that aren't emphasis in CommonMark — *args, **kwargs, __init__, snake_case —
    # survive intact. Underscores are left alone entirely (TTS doesn't voice them).
    line = re.sub(r"(?<![*\w])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![*\w])", r"\1", line)  # bold
    line = re.sub(r"(?<![*\w])\*(?=\S)(.+?)(?<=\S)\*(?![*\w])", r"\1", line)  # italic
    line = re.sub(r"(?<!~)~~(?=\S)(.+?)(?<=\S)~~(?!~)", r"\1", line)  # strikethrough
    return line


def markdown_to_paragraphs(text: str, *, tables: str = "drop", strip_emoji_flag: bool = False) -> list[str]:
    """Reduce Markdown to a list of speakable paragraphs. Headings and list
    items become their own paragraphs (so each gets its own pause). Tables are
    dropped or rendered comma-separated per ``tables``."""
    text = re.sub(r"(?ms)^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", "\n", text)  # fenced code
    text = re.sub(r"```+|~~~+", " ", text)  # stray fences
    if strip_emoji_flag:
        text = strip_emoji(text)

    paragraphs: list[str] = []
    current: list[str] = []
    in_table = False

    def flush() -> None:
        if current:
            paragraphs.append(" ".join(current))
            current.clear()

    def emit_csv_row(cells_source: str) -> None:
        cells = [c.strip() for c in _strip_md_inline(cells_source.strip("|")).split("|")]
        joined = ", ".join(c for c in cells if c)
        if joined:
            paragraphs.append(joined)

    for raw in text.splitlines():
        row = raw.strip()

        if not row:  # blank line ends the current paragraph and any table block
            flush()
            in_table = False
            continue

        # link reference definition ( [id]: https://… ) — invisible when rendered
        if re.match(r"^\[[^\]]+\]:\s*\S", row):
            flush()
            continue

        # table separator row ( |---|---| or --- | --- )
        if re.fullmatch(r"\|?[\s:|-]*-[\s:|-]*\|?", row):
            # Only a *table* separator if the line above held cells; otherwise it's
            # a `---` horizontal rule / setext underline (no table context).
            if current and "|" in current[-1]:
                header = current.pop()
                flush()
                if tables == "csv":
                    emit_csv_row(header)
                in_table = True
            else:
                flush()
                in_table = False
            continue

        # setext underline ( === ) — drop so it isn't voiced as "equals equals…"
        if re.fullmatch(r"=+", row):
            flush()
            continue

        # table row: outer-pipe form, or a bare-pipe row within a table block
        outer_pipe = len(row) >= 2 and row.startswith("|") and row.endswith("|")
        if outer_pipe or (in_table and "|" in row):
            flush()
            if tables == "csv":
                emit_csv_row(row)
            continue

        # horizontal rule ( *** / ___ ; --- is handled by the separator branch )
        if re.fullmatch(r"([-*_])(?:\s*\1){2,}", row):
            flush()
            in_table = False
            continue

        in_table = False
        is_heading = re.match(r"^\s{0,3}#{1,6}\s+", raw)
        is_item = re.match(r"^\s*([-*+]|\d+[.)])\s+", raw)
        cleaned = _strip_md_inline(re.sub(r"^\s{0,3}(#{1,6}\s*|(?:>\s?)+)", "", raw))
        cleaned = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", cleaned).strip()
        if is_item:  # drop task-list checkboxes ( - [ ] / - [x] )
            cleaned = re.sub(r"^\[[ xX]\]\s+", "", cleaned).strip()

        if is_heading or is_item:  # each stands alone -> its own pause
            flush()
            if cleaned:
                paragraphs.append(cleaned)
        elif cleaned:
            current.append(cleaned)
        else:
            flush()

    flush()
    return paragraphs


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #


def _shape_paragraph(
    text: str,
    *,
    pronunciations: dict[str, str] | None,
    expand_units_flag: bool,
    expand_numbers_flag: bool,
    pauses_flag: bool,
    locale: str,
) -> str:
    if expand_numbers_flag:
        text = strip_thousands_separators(text)
    if pauses_flag:
        text = shape_pauses(text)
    if pronunciations:
        text = apply_pronunciations(text, pronunciations)
    if expand_units_flag:  # units BEFORE decimals (see note above)
        text = expand_units(text, locale)
    if expand_numbers_flag:
        text = decimals_to_words(text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def normalize_for_speech(
    text: str,
    *,
    markdown: bool = True,
    pronunciations: dict[str, str] | None = None,
    dev_terms: bool = True,
    expand_units: bool = True,
    expand_numbers: bool = True,
    pauses: bool = True,
    tables: str = "drop",
    strip_emoji: bool = False,
    terminal_stops: bool = True,
    locale: str = DEFAULT_LOCALE,
) -> str:
    """Normalize ``text`` into speakable prose. Returns paragraphs joined by
    newlines. See ``docs/speech-normalization.md`` for the full contract."""
    expand_units_flag, expand_numbers_flag = expand_units, expand_numbers
    # Built-in dev-term fixes first; caller-supplied pronunciations override them
    # (keyed case-insensitively, so a caller's "CLI" beats the default "cli").
    effective_pronunciations: dict[str, str] = dict(_DEV_PRONUNCIATIONS) if dev_terms else {}
    for written, spoken in (pronunciations or {}).items():
        effective_pronunciations[written.lower()] = spoken

    if markdown:
        paragraphs = markdown_to_paragraphs(text, tables=tables, strip_emoji_flag=strip_emoji)
    else:
        # `strip_emoji` (the bool kwarg) shadows the module function here, so
        # reach for the underlying pattern directly.
        body = _EMOJI.sub("", text) if strip_emoji else text
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]

    out = []
    for para in paragraphs:
        shaped = _shape_paragraph(
            para,
            pronunciations=effective_pronunciations,
            expand_units_flag=expand_units_flag,
            expand_numbers_flag=expand_numbers_flag,
            pauses_flag=pauses,
            locale=locale,
        )
        if not shaped:
            continue
        out.append(ensure_terminal_stop(shaped) if terminal_stops else shaped)
    return "\n".join(out)
