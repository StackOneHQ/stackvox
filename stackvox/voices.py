"""Named voice mixes: Kokoro voices blended into a new voice.

Each Kokoro voice is a style vector, so a weighted average of several gives a
voice somewhere between them. A mix also carries the lang and speed it was
tuned with, because a blend of an American and a British voice only sounds the
way it was picked when it gets the same phonemes (``en-gb`` rather than
``en-us``) at the same pace.

Mixes are addressed by name anywhere a Kokoro voice id is accepted: the
library, ``--voice`` on the CLI, and the daemon's socket protocol.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoiceMix:
    """Kokoro voice ids and their weights, plus the lang and speed the blend was tuned with."""

    mix: tuple[tuple[str, float], ...]
    lang: str | None = None
    speed: float | None = None


# Picked by ear for the StackOne newscast explainers: American delivery with
# British vowels, the transatlantic newsreader sound.
BUILTIN_VOICES: dict[str, VoiceMix] = {
    "chip": VoiceMix(mix=(("am_liam", 0.6), ("bm_daniel", 0.4)), lang="en-gb", speed=1.0),
    "ramona": VoiceMix(mix=(("af_aoede", 0.5), ("af_bella", 0.2), ("bf_emma", 0.3)), lang="en-gb", speed=1.0),
}


class UnknownVoiceError(ValueError):
    """A voice mix names a Kokoro voice that isn't in the loaded voice pack."""


@dataclass(frozen=True)
class VoiceParams:
    """Fully resolved synthesis parameters."""

    voice: str
    speed: float
    lang: str


def resolve(
    voice: str | None,
    speed: float | None,
    lang: str | None,
    *,
    default_voice: str,
    default_speed: float,
    default_lang: str,
    custom: Mapping[str, VoiceMix],
) -> VoiceParams:
    """Fill in unset parameters: an explicit value first, then the mix's own, then the default.

    The mix beats the default so that ``--voice chip`` gets Chip's ``en-gb``
    even when the user's config sets ``lang = "en-us"`` for everything else.
    """
    name = voice or default_voice
    mix = custom.get(name)
    if speed is None:
        speed = mix.speed if mix is not None and mix.speed is not None else default_speed
    if not lang:
        lang = mix.lang if mix is not None and mix.lang else default_lang
    return VoiceParams(voice=name, speed=speed, lang=lang)


def blend(
    name: str,
    mix: VoiceMix,
    style_for: Callable[[str], np.ndarray],
    available: Collection[str],
) -> np.ndarray:
    """Weighted average of the mix's style vectors, normalised so weights needn't sum to 1."""
    missing = [voice for voice, _ in mix.mix if voice not in available]
    if missing:
        raise UnknownVoiceError(
            f"voice mix {name!r} uses voices not in the Kokoro voice pack: {', '.join(missing)}"
        )
    total = sum(weight for _, weight in mix.mix)
    blended = sum(weight * style_for(voice) for voice, weight in mix.mix) / total
    return np.asarray(blended, dtype=np.float32)
