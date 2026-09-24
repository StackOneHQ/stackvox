"""User config file: per-user defaults for voice / speed / lang, and voice mixes.

Lives at `$XDG_CONFIG_HOME/stackvox/config.toml` (falling back to
`~/.config/stackvox/config.toml`), or wherever `STACKVOX_CONFIG` points if set.
A missing file is fine — defaults from `stackvox.engine` apply. A malformed
file logs a warning and is otherwise ignored.

File format::

    [defaults]
    voice = "bf_emma"
    speed = 1.1
    lang = "en-gb"

    [voices.narrator]
    mix = { am_michael = 0.6, bm_george = 0.4 }
    lang = "en-gb"
    speed = 1.0

Each ``[voices.<name>]`` table defines a Kokoro voice mix addressable by that
name, alongside the built-in ones in ``stackvox.voices``. A table with a
built-in's name replaces it.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - covered by 3.10 CI
    import tomli as tomllib

from stackvox.engine import DEFAULT_LANG, DEFAULT_SPEED, DEFAULT_VOICE
from stackvox.voices import BUILTIN_VOICES, VoiceMix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Defaults:
    """Resolved default values for synthesis parameters."""

    voice: str = DEFAULT_VOICE
    speed: float = DEFAULT_SPEED
    lang: str = DEFAULT_LANG


def config_path() -> Path:
    """Resolve where the config file lives.

    Honours `STACKVOX_CONFIG` first; otherwise XDG (`$XDG_CONFIG_HOME` →
    `~/.config`).
    """
    override = os.environ.get("STACKVOX_CONFIG")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "stackvox" / "config.toml"


def _read(path: Path) -> dict | None:
    """Parse the config file, or None when it is missing or malformed (the latter logged)."""
    if not path.is_file():
        return None
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("ignoring malformed stackvox config at %s: %s", path, exc)
        return None


def load_defaults(path: Path | None = None) -> Defaults:
    """Read the config file and return resolved defaults.

    Missing file → built-in defaults. Malformed file → warning logged,
    built-in defaults used. Per-key fallback so a config that only sets
    `voice` keeps the built-in `speed` and `lang`.
    """
    p = path or config_path()
    data = _read(p)
    if data is None:
        return Defaults()
    section = data.get("defaults", {})
    if not isinstance(section, dict):
        logger.warning("config %s: [defaults] must be a table; ignoring", p)
        return Defaults()
    return Defaults(
        voice=str(section.get("voice", DEFAULT_VOICE)),
        speed=float(section.get("speed", DEFAULT_SPEED)),
        lang=str(section.get("lang", DEFAULT_LANG)),
    )


def load_voices(path: Path | None = None) -> dict[str, VoiceMix]:
    """Built-in voice mixes, overlaid with any ``[voices.<name>]`` tables in the config.

    An invalid table is logged and skipped rather than failing the whole
    config, matching how ``load_defaults`` treats a bad ``[defaults]``.
    """
    voices = dict(BUILTIN_VOICES)
    p = path or config_path()
    data = _read(p)
    if data is None:
        return voices
    section = data.get("voices", {})
    if not isinstance(section, dict):
        logger.warning("config %s: [voices] must be a table; ignoring", p)
        return voices
    for name, table in section.items():
        mix = _parse_mix(table)
        if mix is None:
            logger.warning(
                "config %s: [voices.%s] needs mix = { <kokoro voice> = <weight>, ... } "
                "with positive weights, and optional string lang / numeric speed; ignoring",
                p,
                name,
            )
            continue
        voices[name] = mix
    return voices


def _parse_mix(table: object) -> VoiceMix | None:
    if not isinstance(table, dict):
        return None
    weights = table.get("mix")
    if not isinstance(weights, dict) or not weights:
        return None
    if not all(
        isinstance(weight, (int, float)) and not isinstance(weight, bool) and weight > 0
        for weight in weights.values()
    ):
        return None
    lang = table.get("lang")
    speed = table.get("speed")
    if lang is not None and not isinstance(lang, str):
        return None
    if speed is not None and (isinstance(speed, bool) or not isinstance(speed, (int, float))):
        return None
    return VoiceMix(
        mix=tuple((str(voice), float(weight)) for voice, weight in weights.items()),
        lang=lang,
        speed=float(speed) if speed is not None else None,
    )
