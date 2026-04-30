"""User config file: per-user defaults for voice / speed / lang.

Lives at `$XDG_CONFIG_HOME/stackvox/config.toml` (falling back to
`~/.config/stackvox/config.toml`), or wherever `STACKVOX_CONFIG` points if set.
A missing file is fine — defaults from `stackvox.engine` apply. A malformed
file logs a warning and is otherwise ignored.

File format::

    [defaults]
    voice = "bf_emma"
    speed = 1.1
    lang = "en-gb"
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


def load_defaults(path: Path | None = None) -> Defaults:
    """Read the config file and return resolved defaults.

    Missing file → built-in defaults. Malformed file → warning logged,
    built-in defaults used. Per-key fallback so a config that only sets
    `voice` keeps the built-in `speed` and `lang`.
    """
    p = path or config_path()
    if not p.is_file():
        return Defaults()
    try:
        data = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("ignoring malformed stackvox config at %s: %s", p, exc)
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
