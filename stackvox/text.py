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
from collections.abc import Callable

__all__ = [
    "normalize_for_speech",
    "markdown_to_paragraphs",
    "strip_emoji",
    "strip_thousands_separators",
    "versions_to_words",
    "speak_versions",
    "decimals_to_words",
    "speak_file_refs",
    "speak_file_names",
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
        (r"~\s*(?=\d)", "about "),  # ~123 -> "about 123" (else espeak says "tilde 123"); leaves ~/path alone
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


_VERSION = re.compile(r"(?<!\d)\d+(?:\.\d+){2,}(?!\d)")


def versions_to_words(text: str) -> str:
    """1.2.3 -> "1 point 2 point 3"; 0.7.0 -> "0 point 7 point 0".

    A dotted version string has more than one decimal point, which
    `decimals_to_words` can't handle — it splits only the first, leaving a stray
    full stop mid-number ("0.7.0" -> "0 point 7.0"). Run this before it. Two-part
    numbers are left untouched so ordinary decimals still read digit-by-digit,
    and a trailing sentence stop (``upgrade to 1.2.3.``) is preserved.
    """
    return _VERSION.sub(lambda m: m.group(0).replace(".", " point "), text)


# --------------------------------------------------------------------------- #
# Semantic versions                                                           #
# --------------------------------------------------------------------------- #
# `versions_to_words` handles the dotted digits. This handles everything else
# in a semver string, all of which espeak gets wrong:
#
#   * a pre-release suffix GLUES to the core: "1.2.3-rc.1" voices as
#     "one point two point three-arsee-one" (hyphen swallowed, suffix dot silent)
#   * "^", ">" and "<" are SILENT, so "^1.2.3" is indistinguishable from a pin
#   * ">=" is broken by our own `=` rule below, which splits it into "> equals",
#     and a bare ">" voices as nothing, so ">=1.2.3" says "equals 1.2.3",
#     inverting the meaning. espeak reads an intact ">=" correctly, so this stage
#     must consume the operator BEFORE `expand_units` sees the "=".
#   * "1.x" voices as "one ex" (dot silent)
#
# "~" is left to `expand_units`, which already maps it to "about"; accidentally
# the right reading for a tilde range.

# Longest operators first: ">=" must win before ">". Each requires a following
# digit (optionally "v"-prefixed), which keeps the rules in version context and
# off ordinary punctuation. The leading `(\S?)` captures whatever non-space
# character precedes the operator and re-emits it with a space, so glued forms
# like pip's "requests>=2.0" or "arr[i]<5" don't fuse into "requestsat least".
_VERSION_OPERATORS: list[tuple[str, str]] = [
    (r"(\S?)>=\s*(?=v?\d)", "at least "),
    (r"(\S?)<=\s*(?=v?\d)", "at most "),
    (r"(\S?)==\s*(?=v?\d)", "exactly "),
    (r"(\S?)!=\s*(?=v?\d)", "not equal to "),
    (r"(\S?)>\s*(?=v?\d)", "above "),
    (r"(\S?)<\s*(?=v?\d)", "below "),
]

# "^" only at a token start: "x^2" and ")^2" are exponentiation, not a range.
_CARET_RANGE = re.compile(r"(?<![\w)])\^\s*(?=v?\d)")

# A dotted core (>=2 parts, so a bare "100" can't match) followed by a
# "-prerelease" or "+build" suffix.
_VERSION_SUFFIX = re.compile(r"(?<![\w.])(v?\d+(?:\.\d+){1,3})([-+][0-9A-Za-z][0-9A-Za-z.-]*)(?![\w])")

# Wildcard versions: "1.x", "2.*", "1.2.X".
_VERSION_WILDCARD = re.compile(r"(?<![\w.])(v?\d+(?:\.\d+)*)\.([xX*])(?![\w])")


def _operator_repl(spoken: str) -> Callable[[re.Match[str]], str]:
    """Replacement that re-emits the captured boundary character with a space."""

    def repl(match: re.Match[str]) -> str:
        boundary = match.group(1)
        return f"{boundary} {spoken}" if boundary else spoken

    return repl


def speak_versions(text: str) -> str:
    """Voice the non-numeric parts of a semantic version.

    ``1.2.3-rc.1`` -> "1.2.3, rc 1"; ``>=1.2.3`` -> "at least 1.2.3";
    ``^1.2.3`` -> "compatible with 1.2.3"; ``1.x`` -> "1 dot x". The dotted
    digits are left for :func:`versions_to_words`, which runs later, so this
    stage only unglues and names things espeak drops.

    Must run BEFORE :func:`expand_units`, whose ``=`` rule would otherwise split
    ``>=`` into a silent ``>`` plus "equals".
    """
    for pattern, words in _VERSION_OPERATORS:
        text = re.sub(pattern, _operator_repl(words), text)
    text = _CARET_RANGE.sub("compatible with ", text)
    # Suffix: comma for a beat, then the tag's own dots and hyphens as spaces.
    text = _VERSION_SUFFIX.sub(
        lambda m: f"{m.group(1)}, {re.sub(r'[.-]', ' ', m.group(2)[1:])}",
        text,
    )
    return _VERSION_WILDCARD.sub(r"\1 dot \2", text)


def decimals_to_words(text: str) -> str:
    """1198.9 -> "1198 point 9"; 770.72 -> "770 point 7 2". Removes the bare
    "." between digits, which TTS can otherwise read as a full stop."""
    return re.sub(
        r"(\d+)\.(\d+)",
        lambda m: m.group(1) + " point " + " ".join(m.group(2)),
        text,
    )


# --------------------------------------------------------------------------- #
# File & path references                                                      #
# --------------------------------------------------------------------------- #
# espeak already spells extensions correctly on its own: ".md" voices as "em
# dee", ".tf" as "tee eff", ".json" as "jason", so these stages deliberately
# leave the extension alone. What espeak gets wrong is everything around it:
#
#   * the dot between stem and extension is SILENT ("README.md" -> "readmee-emdee")
#   * a hyphen inside a name is swallowed ("speech-normalization" -> one word)
#   * a leading dot is silent, so ".github" reads as "github"
#   * "~/" glues into "tilde-slash"
#   * ".yml" reads as "immle" (".yaml" is fine)
#
# So: voice the dot, space the hyphens, voice a leading dot, say "home" for "~",
# and alias the one bad extension. Directory segments are spoken with "slash"
# between them, which is how a person reads a path aloud.

# Only these extensions mark a dotted token as a filename. An allowlist, not a
# general "word.word" rule, because ordinary prose is full of lookalikes espeak
# ALREADY voices correctly and which must not be touched: attribute access
# (os.path.join, self.assertEqual), abbreviations (e.g., i.e., U.S.), and
# domains (example.com, claude.ai) all fall outside it for free.
# Single-letter extensions (.c, .h, .r) are deliberately absent: they would
# rewrite initials like "J.R.R" into "J dot R dot R". Those files keep espeak's
# existing reading rather than risk a prose regression.
_FILE_EXTENSIONS = frozenset(
    [
        "py",
        "pyi",
        "pyx",
        "ipynb",
        "rb",
        "rs",
        "go",
        "java",
        "kt",
        "kts",
        "swift",
        "cpp",
        "hpp",
        "cc",
        "cxx",
        "cs",
        "php",
        "lua",
        "pl",
        "scala",
        "clj",
        "cljs",
        "ex",
        "exs",
        "erl",
        "vim",
        "el",
        "ts",
        "tsx",
        "js",
        "jsx",
        "mjs",
        "cjs",
        "vue",
        "svelte",
        "astro",
        "json",
        "yaml",
        "yml",
        "toml",
        "ini",
        "cfg",
        "conf",
        "env",
        "xml",
        "csv",
        "tsv",
        "properties",
        "lock",
        "md",
        "mdx",
        "rst",
        "txt",
        "adoc",
        "tex",
        "html",
        "htm",
        "css",
        "scss",
        "sass",
        "less",
        "svg",
        "tf",
        "tfvars",
        "tfstate",
        "mk",
        "cmake",
        "gradle",
        "bzl",
        "proto",
        "graphql",
        "gql",
        "sql",
        "sh",
        "bash",
        "zsh",
        "fish",
        "ps1",
        "bat",
        "dockerfile",
        "gitignore",
        "editorconfig",
        "npmrc",
        "nvmrc",
        "log",
        "wav",
        "mp3",
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
        "zip",
        "tar",
        "gz",
        "whl",
    ]
)

# Extensions espeak mispronounces, mapped to a spelling it reads correctly.
# ".yml" -> "immle"; the "yaml" spelling voices as "yaml".
_EXTENSION_ALIASES: dict[str, str] = {"yml": "yaml"}

# Dotted tokens that LOOK like "name.ext" with a real extension but are read as
# a single product name. Compared lowercased against the whole matched token.
_NOT_FILENAMES = frozenset({"node.js", "next.js", "nuxt.js", "vue.js", "ember.js", "backbone.js"})


def _speak_segment(segment: str) -> str:
    """Space out a hyphenated path segment: "speech-normalization" -> "speech
    normalization". espeak swallows the hyphen and runs the halves together.
    Underscores are left alone; espeak doesn't voice them."""
    return segment.replace("-", " ")


def _speak_basename(name: str) -> str:
    """``speech-normalization.md`` -> "speech normalization dot md".

    Every dot becomes a spoken "dot" (espeak drops it), hyphens become spaces,
    and the final extension is run through the alias map. The extension itself
    is otherwise untouched; espeak spells it correctly already.
    """
    segments = name.split(".")
    segments[-1] = _EXTENSION_ALIASES.get(segments[-1].lower(), segments[-1])
    return " dot ".join(_speak_segment(segment) for segment in segments)


def _speak_dirs(path: str) -> str:
    """``src/lib/`` -> "src slash lib"; ``~/.config/`` -> "home slash dot config".

    A leading dot and a ``~`` are both mis-voiced by espeak (silent, and
    "tilde-slash" respectively), so they're spelled out here.
    """
    spoken: list[str] = []
    for segment in path.split("/"):
        if not segment or segment == ".":
            continue
        if segment == "~":
            spoken.append("home")
        elif segment == "..":
            spoken.append("dot dot")
        elif segment.startswith("."):
            spoken.append("dot " + _speak_segment(segment[1:]))
        else:
            spoken.append(_speak_segment(segment))
    return " slash ".join(spoken)


def _is_filename(basename: str) -> bool:
    """True when the final dotted component is a known file extension."""
    if basename.lower() in _NOT_FILENAMES:
        return False
    return basename.rsplit(".", 1)[-1].lower() in _FILE_EXTENSIONS


# Directory segments, captured. The inner segment allows empty, so an absolute
# "/abs/path/" is consumed by the match rather than left stranded in the text.
_DIRS = r"((?:[\w.~-]*/)*)"
_BASENAME = r"([A-Za-z0-9_-]+(?:\.[A-Za-z][\w-]*)+)"  # name with >=1 letter-initial extension

_FILE_REF = re.compile(
    r"(?<!\w)"
    + _DIRS
    + _BASENAME
    + r":(\d+)(?:-(\d+))?(?::(\d+))?"  # :line, optional -end (range) or :column
    + r"(?!\w)"
)

# Same shape without the ":line" suffix. The trailing guard allows a following
# "." so a sentence-final "See README.md." still matches.
_FILE_NAME = re.compile(r"(?<![\w/~.-])" + _DIRS + _BASENAME + r"(?![\w-])")


def speak_file_refs(text: str) -> str:
    """Turn ``path/file.ext:line`` refs into spoken "line N of file dot ext in path".

    ``engine.py:42`` -> "line 42 of engine dot py"; ``src/cli.py:100-118`` ->
    "lines 100 to 118 of cli dot py in src"; ``foo.ts:666:10`` -> "line 666,
    column 10 of foo dot ts". The line number leads because spoken aloud it's
    the signal, and the directory trails as a prepositional phrase, which is how
    a person says it.

    The ``:line`` suffix is the trigger, so bare times/ratios/verses (``12:30``,
    ``3:1``, ``John 3:16``) and dotted versions (``1.2.3``) are left untouched --
    none of them carry a dotted-filename before the colon.
    """

    def repl(match: re.Match[str]) -> str:
        dirs, basename = match.group(1), match.group(2)
        start, end, column = match.group(3), match.group(4), match.group(5)
        if end:
            location = f"lines {start} to {end}"
        else:
            location = f"line {start}" + (f", column {column}" if column else "")
        spoken = f"{location} of {_speak_basename(basename)}"
        directories = _speak_dirs(dirs)
        return f"{spoken} in {directories}" if directories else spoken

    return _FILE_REF.sub(repl, text)


def speak_file_names(text: str) -> str:
    """Voice a bare filename or path, with no ``:line`` suffix needed.

    ``README.md`` -> "README dot md"; ``docs/speech-normalization.md`` -> "docs
    slash speech normalization dot md". Gated on a known extension
    (:data:`_FILE_EXTENSIONS`), so prose lookalikes espeak already reads
    correctly (``os.path.join``, ``e.g.``, ``example.com``) pass through.

    Run this AFTER :func:`speak_file_refs`: that stage rewrites its own matches
    into prose containing no dotted token, so the two never fight over one ref.
    """

    def repl(match: re.Match[str]) -> str:
        dirs, basename = match.group(1), match.group(2)
        if not _is_filename(basename):
            return match.group(0)
        spoken = _speak_basename(basename)
        directories = _speak_dirs(dirs)
        return f"{directories} slash {spoken}" if directories else spoken

    return _FILE_NAME.sub(repl, text)


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
    "aws": "A.W.S.",  # dotted: espeak reads a lone "A" as the article "uh"; dots force the letter name "ay"
    "uri": "U R I",
    "iam": "I.A.M.",  # ditto — the middle "A" needs the dot
    "saas": "sass",
    "paas": "pass",
    "tui": "T U I",
    "postgresql": "postgres",
    "kubectl": "kube control",
    "stackone": "stack one",  # org name espeak garbles as "stac kone"
    # "dedupe" glued reads as "de-dup" (short u); split + long "dee" gives "dee doop".
    # Inflections listed explicitly — apply_pronunciations matches whole words only.
    "dedupe": "dee dupe",
    "deduped": "dee duped",
    "deduping": "dee duping",
    "dedupes": "dee dupes",
    "dedup": "dee dupe",
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


def markdown_to_paragraphs(
    text: str,
    *,
    tables: str = "drop",
    strip_emoji_flag: bool = False,
    code_blocks: str = "drop",
    code_placeholder: str = "",
) -> list[str]:
    """Reduce Markdown to a list of speakable paragraphs. Headings and list
    items become their own paragraphs (so each gets its own pause). Tables are
    dropped or rendered comma-separated per ``tables``. Fenced code blocks are
    dropped, or (``code_blocks="placeholder"``) replaced with a spoken
    ``code_placeholder`` so a silently-skipped block doesn't sound disjointed."""
    speak_code = code_blocks == "placeholder" and bool(code_placeholder)
    # A function replacement keeps the placeholder literal (no backref parsing);
    # blank lines around it make it a standalone paragraph.
    fenced = f"\n\n{code_placeholder}\n\n" if speak_code else "\n"
    text = re.sub(r"(?ms)^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", lambda _: fenced, text)  # fenced code
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

    if speak_code:  # collapse runs of adjacent code blocks into one placeholder
        collapsed: list[str] = []
        for para in paragraphs:
            if para == code_placeholder and collapsed and collapsed[-1] == code_placeholder:
                continue
            collapsed.append(para)
        return collapsed
    return paragraphs


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #


def _shape_paragraph(
    text: str,
    *,
    pronunciations: dict[str, str] | None,
    filenames_flag: bool,
    expand_units_flag: bool,
    expand_numbers_flag: bool,
    pauses_flag: bool,
    locale: str,
) -> str:
    # Filenames first, for two reasons: the ref stage consumes the ":line" digits
    # (so the number stages see a plain "line 42", not a decimal), and spacing the
    # dot leaves the stem a standalone word, which is what lets the dev-term dict
    # below still fix "cli" -> "C L I" without gluing it to the extension.
    # Refs before bare names: the ref stage rewrites its matches into prose with
    # no dotted token left, so the two never fight over the same reference.
    if filenames_flag:
        text = speak_file_refs(text)
        text = speak_file_names(text)
    if expand_numbers_flag:
        # Version ranges before units: the "=" unit rule would split ">=".
        text = speak_versions(text)
        text = strip_thousands_separators(text)
    if pauses_flag:
        text = shape_pauses(text)
    if pronunciations:
        text = apply_pronunciations(text, pronunciations)
    if expand_units_flag:  # units BEFORE decimals (see note above)
        text = expand_units(text, locale)
    if expand_numbers_flag:
        text = versions_to_words(text)  # multi-dot versions BEFORE the decimal split
        text = decimals_to_words(text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def normalize_for_speech(
    text: str,
    *,
    markdown: bool = True,
    pronunciations: dict[str, str] | None = None,
    dev_terms: bool = True,
    filenames: bool = True,
    expand_units: bool = True,
    expand_numbers: bool = True,
    pauses: bool = True,
    tables: str = "drop",
    code_blocks: str = "drop",
    code_placeholder: str = "Code block.",
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
        paragraphs = markdown_to_paragraphs(
            text,
            tables=tables,
            strip_emoji_flag=strip_emoji,
            code_blocks=code_blocks,
            code_placeholder=code_placeholder,
        )
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
            filenames_flag=filenames,
            expand_units_flag=expand_units_flag,
            expand_numbers_flag=expand_numbers_flag,
            pauses_flag=pauses,
            locale=locale,
        )
        if not shaped:
            continue
        out.append(ensure_terminal_stop(shaped) if terminal_stops else shaped)
    return "\n".join(out)
