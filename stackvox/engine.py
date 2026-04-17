"""Core engine: model loading, synthesis, and playback."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro

DEFAULT_VOICE = "af_sarah"
DEFAULT_SPEED = 1.0
DEFAULT_LANG = "en-us"

_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _cache_dir() -> Path:
    override = os.environ.get("STACKVOX_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "stackvox"


def _ensure_models(cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "kokoro-v1.0.onnx"
    voices_path = cache_dir / "voices-v1.0.bin"
    for path, url in [(model_path, _MODEL_URL), (voices_path, _VOICES_URL)]:
        if path.exists():
            continue
        print(f"[stackvox] downloading {path.name}...", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    return model_path, voices_path


class Stackvox:
    """Reusable TTS engine. Load the model once, speak many times.

    Example:
        tts = Stackvox(voice="af_bella")
        tts.speak("Hello world")
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_SPEED,
        lang: str = DEFAULT_LANG,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.lang = lang
        model_path, voices_path = _ensure_models(cache_dir or _cache_dir())
        self._kokoro = Kokoro(str(model_path), str(voices_path))

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lang: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """Return (samples, sample_rate) without playing."""
        samples, sample_rate = self._kokoro.create(
            text,
            voice=voice or self.voice,
            speed=speed if speed is not None else self.speed,
            lang=lang or self.lang,
        )
        return samples, sample_rate

    def speak(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        lang: Optional[str] = None,
        blocking: bool = True,
    ) -> None:
        """Synthesize and play through the system default output device."""
        samples, sample_rate = self.synthesize(text, voice=voice, speed=speed, lang=lang)
        sd.play(samples, sample_rate)
        if blocking:
            sd.wait()

    def stop(self) -> None:
        """Stop any in-progress playback started with blocking=False."""
        sd.stop()

    def voices(self) -> list[str]:
        return sorted(self._kokoro.get_voices())


_default: Optional[Stackvox] = None


def _get_default() -> Stackvox:
    global _default
    if _default is None:
        _default = Stackvox()
    return _default


def speak(
    text: str,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    lang: str = DEFAULT_LANG,
    blocking: bool = True,
) -> None:
    """One-shot: synthesize and play. Reuses a module-level engine across calls."""
    _get_default().speak(text, voice=voice, speed=speed, lang=lang, blocking=blocking)


def synthesize(
    text: str,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    lang: str = DEFAULT_LANG,
) -> tuple[np.ndarray, int]:
    """One-shot synthesis. Returns (samples, sample_rate)."""
    return _get_default().synthesize(text, voice=voice, speed=speed, lang=lang)
