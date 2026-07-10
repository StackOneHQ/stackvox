# Proposal: speech-text normalization in StackVox

**Status:** draft for review · **Author:** Stu Behan (with Claude) · target StackVox `0.6.0`

## 1. Why

StackVox synthesizes whatever string it's handed. Two consumers now need the
same *pre*-step — turning real-world text into something that *sounds* right —
and they've already grown separate, diverging copies of it:

| Consumer | File | Input | Has | Lacks |
|---|---|---|---|---|
| behan.codes read-aloud | `blog/tools/read-aloud.py` | Markdown post files | pronunciation dict, unit/number expansion, decimals→"point", pause shaping, `say:` directive, title split | emoji stripping, tables-as-CSV |
| stackone-speaklast | `.../stackone-speaklast/scripts/extract-response.py` (`clean_markdown`) | Claude Code responses | emoji stripping, tables→comma, reference links, snake_case-safe emphasis | pronunciation dict, units/numbers, decimals, pauses |

Net effect: speaklast currently says "ree-dees", "stack-own", and "one, one
hundred ninety-eight point nine" — every bug the blog already fixed. Copying the
dict and rules into speaklast is the trap this proposal avoids.

**Key insight:** the part that keeps changing is the *pronunciation entries*
(`agy→antigravity`, `Behan→Bayan`) — and those are per-consumer config that was
never going to live in StackVox anyway. The *mechanism* (apply a dict, expand
units/numbers, markdown→prose, shape pauses) has been stable. So we can extract
the mechanism now and let each caller keep evolving its own dictionary.

## 2. Goals / non-goals

**Goals**
- One normalization mechanism in StackVox, usable as a library call and from the CLI.
- Configurable by flags + an injected pronunciation dict.
- Backward compatible — existing `speak`/`synthesize` behaviour unchanged unless opted in.

**Non-goals**
- Baking **domain/org** pronunciations (agy, StackOne, Redis…) into StackVox — those stay per-consumer. Generic fixes for dev acronyms espeak plain *mispronounces* (CLI→"kligh", AWS→"awz", URI→"yuri", …) **are** shipped in core, on by default via `dev_terms` / `_DEV_PRONUNCIATIONS`.
- Owning the blog's `say:` authoring directive (a Markdown-authoring convention; Claude responses never contain it).
- Changing default synthesis behaviour or the daemon protocol.

## 3. Proposed API

New module `stackvox/text.py`, one primary entry point:

```python
def normalize_for_speech(
    text: str,
    *,
    markdown: bool = True,          # strip Markdown structure to prose
    pronunciations: dict[str, str] | None = None,  # {written: spoken}, whole-word, case-insensitive
    expand_units: bool = True,      # £/p/kWh/MPG/kg/km, ÷ × =
    expand_numbers: bool = True,    # thousands commas removed; decimals → "X point d d"
    pauses: bool = True,            # dash → beat (…); comma before "("
    tables: str = "drop",           # "drop" | "csv"  (how Markdown tables are voiced)
    strip_emoji: bool = False,
    terminal_stops: bool = True,    # ensure each line ends in . / ? / : so a pause lands
) -> str:
    ...
```

Re-exported from `stackvox/__init__.py` (`from stackvox.text import normalize_for_speech`).
Individual stages also exposed for composability (e.g. `expand_numbers(text)`,
`apply_pronunciations(text, mapping)`), but `normalize_for_speech` is the one most callers use.

### Pipeline order (order matters — encodes real bugs we hit)

1. **Markdown** (if `markdown`): fenced/inline code, images, links→text, reference links, headings, blockquotes, horizontal rules, list markers, emphasis (leave lone `_` for snake_case), tables per `tables`.
2. **Emoji** (if `strip_emoji`).
3. **Numbers** (if `expand_numbers`): strip thousands commas (`1,198.9`→`1198.9`) *then* decimals→words (`1198.9`→`1198 point 9`).
4. **Units/symbols** (if `expand_units`): currency **before** decimals is impossible if decimals ran first, so units run here and the decimal pass must see `£1.63`→`1.63 pounds` first → **currency/units run before the decimal split**. (This is why `read-aloud.py` orders units → decimals.)
5. **Pauses** (if `pauses`): ` - `/`—`→` … `; word`(`→`word, (`.
6. **Pronunciations**: whole-word, case-insensitive, from `pronunciations`.
7. **Whitespace collapse**; **terminal stops** (if `terminal_stops`).

> Ordering note to preserve: **currency/unit expansion must precede the
> decimal-point split**, or `£1.63` becomes `£1 point 63`. See `read-aloud.py`.

## 4. CLI surface

Backward-compatible additions to the existing `speak`/`say`:

```
stackvox speak --normalize [--markdown] --file post.txt   # normalize, then synth
stackvox normalize --file resp.md                          # print normalized text only (pipe/debug)
```

`--normalize` off by default, so notification phrases synth exactly as today.

## 5. What moves vs. what stays

| | Moves into StackVox | Stays in the caller |
|---|---|---|
| **Core** | markdown→prose, numbers, units, pauses, pronunciation *mechanism*, emoji, terminal stops | — |
| **Blog** | — | reads `.md` + front matter; the `say:` directive; frontmatter-title split + **silence splice** in `generate-audio.sh`; its dict; `tables="drop"` |
| **speaklast** | — | transcript discovery + last-response extraction; its **Claude-tuned** dict; `tables="csv"`, `strip_emoji=True`; drops its own `clean_markdown()` |

### Illustrative dictionaries (stay per-consumer)

- **Blog:** `agy, 1M, 175K, xhigh, SessionStart, PermissionRequest, StackOne, OAuth, Behan, Redis`
- **speaklast (to tune by listening):** `StackOne, Redis, repo, …` — domain/product names Claude leans on. Generic dev acronyms (CLI, CI, AWS, URI, IAM…) now live in core's default `dev_terms` map, so they don't need repeating per-consumer.

## 6. Backward compatibility & rollout

- Additive + opt-in → no behaviour change for current users. Bump **minor → 0.6.0** (conventional commits), add `tests/test_text.py` to satisfy the coverage gate.
- **Distribution wrinkle:** both consumers currently run the StackVox **bundled in the Stack Nudge app** (`~/.stack-nudge/venv`, 0.5.0), *not* this clone. Until 0.6.0 ships in the bundle:
  - dev/local: `pip install -e ~/stackone/stackvox` and point the callers at that venv (blog's `generate-audio.sh` already resolves a `stackvox` binary; add a clone/venv candidate);
  - each caller keeps a thin local fallback (its current cleaner) if the installed StackVox predates `normalize_for_speech`, so nothing breaks on an old bundle.

## 7. Migration steps

1. **StackVox:** add `stackvox/text.py` + tests + `--normalize`/`normalize` CLI; export; release `0.6.0`.
2. **Blog:** replace the transform functions in `read-aloud.py` with a `normalize_for_speech(md, pronunciations=BLOG_DICT, tables="drop")` call; keep front-matter/`say:`/title-split; keep `generate-audio.sh`'s silence splice.
3. **speaklast:** replace `clean_markdown()` with `normalize_for_speech(resp, pronunciations=CLAUDE_DICT, tables="csv", strip_emoji=True)`.
4. Verify each by listening (blog posts + a few Claude responses).

## 8. Decisions (resolved 2026-07-07)

- **`terminal_stops`** — lives in the lib, **default on**. ✔
- **Default `tables`** — **`drop`** (blog default); speaklast passes `tables="csv"`. ✔
- **Locale** — add a **`locale` param, default `"en-GB"`**; currency/unit rules are keyed by locale (en-GB shipped first, others added later). ✔
- **Emoji** — **`strip_emoji=False`** by default; speaklast sets `True`. ✔
- **Distribution** — ship the updated StackVox **through the Stack Nudge app bundle** (both consumers run the bundled venv), so no editable-install fallback is needed in the callers once 0.6.0 is bundled. ✔
- **Frontmatter-title split** — stays caller-side (blog), alongside the silence splice.
