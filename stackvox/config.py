"""User config file: per-user defaults for voice / speed / lang, voice mixes, and
the daemon's backend.

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

    [serve]
    backend = "mlx"
    model_dir = "~/models/my-model"
    model_adapter = "~/.config/stackvox/adapters/my_model.py"

Each ``[voices.<name>]`` table defines a Kokoro voice mix addressable by that
name, alongside the built-in ones in ``stackvox.voices``. A table with a
built-in's name replaces it.

``[serve]`` picks the backend ``stackvox serve`` starts when no ``--backend`` or
``--model-dir`` flag is given, plus that backend's model and generation defaults (the
``stackvox.engine.MLX_OPTIONS`` names). No other command acts on it. Relative
paths in it are relative to the config file's folder (a symlink's own folder,
not its target's).
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - covered by 3.10 CI
    import tomli as tomllib

from stackvox.engine import BACKENDS, DEFAULT_LANG, DEFAULT_SPEED, DEFAULT_VOICE, REFERENCE_PAIR
from stackvox.voices import BUILTIN_VOICES, VoiceMix

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Defaults:
    """Resolved default values for synthesis parameters."""

    voice: str = DEFAULT_VOICE
    speed: float = DEFAULT_SPEED
    lang: str = DEFAULT_LANG


@dataclass(frozen=True)
class ServeConfig:
    """The daemon's backend settings from ``[serve]``. A None backend leaves the choice to the flags."""

    backend: str | None = None
    model_dir: Path | None = None
    model_adapter: Path | None = None
    generation: dict[str, Any] = field(default_factory=dict)


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


def load_serve(path: Path | None = None) -> ServeConfig:
    """Read the daemon's backend settings from ``[serve]``.

    Only ``stackvox serve`` acts on this. The file is shared by every stackvox
    install on the machine, and a ``speak`` from one without the mlx extra
    would fail if it picked the mlx backend up from here.

    Lenient like ``[defaults]``: a missing table gives an empty ServeConfig and a
    bad value is logged and dropped. An mlx backend without a usable
    ``model_dir`` is dropped whole, since the daemon could not start with it.
    Unknown keys are warned about too, because a misspelt one (``top-p``)
    otherwise vanishes silently.

    Relative paths resolve against the config file's folder, not the working
    directory, which for a daemon started by a supervisor is often ``/``. For a
    symlinked config that is the link's folder, where the user sees the file.
    """
    p = path or config_path()
    data = _read(p)
    if data is None:
        return ServeConfig()
    section = data.get("serve", {})
    if not isinstance(section, dict):
        logger.warning("config %s: [serve] must be a table; ignoring", p)
        return ServeConfig()
    unknown = sorted(set(section) - _SERVE_KEYS)
    if unknown:
        logger.warning(
            "config %s: [serve] has unknown keys %s (names use underscores, like top_p); ignoring them",
            p,
            ", ".join(unknown),
        )
        section = {key: value for key, value in section.items() if key in _SERVE_KEYS}
    base = p.parent
    backend = section.get("backend")
    if backend is None:
        if section:
            logger.warning("config %s: [serve] settings need a backend; ignoring [serve]", p)
        return ServeConfig()
    if backend not in BACKENDS:
        logger.warning(
            "config %s: [serve] backend must be one of %s; ignoring [serve]", p, ", ".join(BACKENDS)
        )
        return ServeConfig()
    if backend != "mlx":
        ignored = sorted(set(section) - {"backend"})
        if ignored:
            logger.warning(
                "config %s: [serve] %s apply to the mlx backend only; ignoring", p, ", ".join(ignored)
            )
        return ServeConfig(backend=backend)
    model_dir = _anchored(_path(section.get("model_dir")), base)
    if model_dir is None:
        logger.warning('config %s: [serve] backend = "mlx" needs a usable model_dir; ignoring [serve]', p)
        return ServeConfig()
    model_adapter = _anchored(_path(section.get("model_adapter")), base)
    if model_adapter is None and "model_adapter" in section:
        logger.warning("config %s: [serve] model_adapter is not a usable path; ignoring it", p)
    generation: dict[str, Any] = {}
    for key, parse in _GENERATION_PARSERS.items():
        if key not in section:
            continue
        value = parse(section[key])
        if value is None:
            logger.warning("config %s: [serve] %s is invalid; ignoring it", p, key)
            continue
        generation[key] = _anchored(value, base) if isinstance(value, Path) else value
    # The engine refuses one without the other on every request, so a daemon
    # started with half a pair would come up and then fail everything it's asked.
    if sum(key in generation for key in REFERENCE_PAIR) == 1:
        logger.warning(
            "config %s: [serve] reference_audio and reference_text only work together; ignoring the one given",
            p,
        )
        for key in REFERENCE_PAIR:
            generation.pop(key, None)
    return ServeConfig(backend="mlx", model_dir=model_dir, model_adapter=model_adapter, generation=generation)


def _path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return Path(value).expanduser()
    except RuntimeError:  # "~name/..." for a user this machine doesn't have
        return None


def _anchored(path: Path | None, base: Path) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return base / path


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


# One entry per name in stackvox.engine.MLX_OPTIONS, typed as the engine takes it.
_GENERATION_PARSERS: dict[str, Callable[[object], Any]] = {
    "reference_audio": _path,
    "reference_text": _text,
    "instruct": _text,
    "temperature": _number,
    "top_p": _number,
    "top_k": _integer,
}
_SERVE_KEYS = {"backend", "model_dir", "model_adapter", *_GENERATION_PARSERS}
