"""Core engine: model loading, synthesis, and playback."""

from __future__ import annotations

import logging
import queue
import re
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro

from stackvox.paths import cache_dir as _default_cache_dir

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "af_sarah"
DEFAULT_SPEED = 1.0
DEFAULT_LANG = "en-us"

_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


def _download_with_progress(url: str, dest: Path) -> None:
    """Stream a URL to a file, printing percentage updates to stderr.

    The ~340 MB Kokoro model takes long enough on first run that a silent
    download looks like a hang; this gives users feedback without pulling
    in tqdm. Falls back to a single line when the server doesn't report
    Content-Length.
    """
    last_pct = -1
    label = dest.name

    def hook(blocks: int, blocksize: int, totalsize: int) -> None:
        nonlocal last_pct
        if totalsize <= 0:
            return
        pct = min(100, int(blocks * blocksize * 100 / totalsize))
        if pct != last_pct:
            mb_total = totalsize / 1_000_000
            print(
                f"\r[stackvox] downloading {label} {pct:3d}% ({mb_total:.0f} MB)",
                end="",
                file=sys.stderr,
                flush=True,
            )
            last_pct = pct

    urllib.request.urlretrieve(url, dest, reporthook=hook)
    if last_pct >= 0 and sys.stderr.isatty():
        # Finish the carriage-returned line so subsequent output starts on
        # its own line. On non-TTY (e.g. CI logs) stderr is line-buffered
        # and the \r writes already land on separate lines.
        print("", file=sys.stderr, flush=True)


def _ensure_models(cache_dir: Path) -> tuple[Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "kokoro-v1.0.onnx"
    voices_path = cache_dir / "voices-v1.0.bin"
    for path, url in [(model_path, _MODEL_URL), (voices_path, _VOICES_URL)]:
        if path.exists():
            continue
        _download_with_progress(url, path)
    return model_path, voices_path


# Split after ., !, or ? that is followed by whitespace, and on newlines. The
# whitespace requirement leaves decimals ("0.95") and mid-token dots
# ("file.ts") intact, which is good enough for speech chunking.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def _split_sentences(text: str) -> list[str]:
    """Break text into sentence-ish chunks for low-latency streaming playback."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text.strip()) if part.strip()]


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
        cache_dir: Path | None = None,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.lang = lang
        model_path, voices_path = _ensure_models(cache_dir or _default_cache_dir())
        self._kokoro = Kokoro(str(model_path), str(voices_path))
        self._stop_event = threading.Event()
        self._play_thread: threading.Thread | None = None

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        lang: str | None = None,
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
        voice: str | None = None,
        speed: float | None = None,
        lang: str | None = None,
        blocking: bool = True,
    ) -> None:
        """Synthesize and play through the system default output device.

        Streams sentence by sentence: the first sentence starts playing as soon
        as it is synthesized (~0.2s) while the rest synthesize in the background,
        rather than waiting for the whole text to synthesize first.
        """
        self._stop_event.clear()
        if blocking:
            self._stream_play(text, voice=voice, speed=speed, lang=lang)
        else:
            self._play_thread = threading.Thread(
                target=self._stream_play,
                kwargs={"text": text, "voice": voice, "speed": speed, "lang": lang},
                daemon=True,
            )
            self._play_thread.start()

    def _stream_play(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        lang: str | None = None,
    ) -> None:
        """Synthesize sentence by sentence and play each as it is ready.

        Kokoro batches by phoneme count (~510), so a whole paragraph synthesizes
        as one blob before any audio — the source of the lead-in lag. Splitting
        into sentences ourselves means the first sentence (~0.2s to synthesize)
        plays almost immediately. A producer thread synthesizes ahead into a
        bounded queue while this thread plays, so later sentences are usually
        ready by the time the previous one finishes.
        """
        sentences = _split_sentences(text)
        if not sentences:
            return

        chunks: queue.Queue = queue.Queue(maxsize=8)
        sentinel = object()

        def _produce() -> None:
            try:
                for sentence in sentences:
                    if self._stop_event.is_set():
                        break
                    chunks.put(self.synthesize(sentence, voice=voice, speed=speed, lang=lang))
            except Exception:
                logger.exception("synthesis error")
            finally:
                chunks.put(sentinel)

        producer = threading.Thread(target=_produce, daemon=True, name="stackvox-synth")
        producer.start()
        try:
            while not self._stop_event.is_set():
                item = chunks.get()
                if item is sentinel:
                    break
                samples, sample_rate = item
                sd.play(samples, sample_rate)
                sd.wait()
        finally:
            producer.join(timeout=1.0)

    def stop(self) -> None:
        """Stop any in-progress playback started with blocking=False."""
        self._stop_event.set()
        sd.stop()

    def voices(self) -> list[str]:
        return sorted(self._kokoro.get_voices())

    def speak_sequence(
        self,
        lines: list[dict],
        gap_seconds: float = 0.0,
        concurrent: bool = True,
    ) -> None:
        """Synthesize multiple lines, concatenate, play as one gapless buffer.

        Each line is a dict: {"text": str, "voice": str?, "speed": float?, "lang": str?}.
        With concurrent=True, synthesis happens in parallel threads (ONNX runtime
        releases the GIL) so the upfront wait is closer to the longest single
        synthesis rather than the sum of all of them.
        """
        if not lines:
            return

        def synth(line: dict) -> tuple[np.ndarray, int]:
            kwargs = {k: v for k, v in line.items() if k != "text"}
            return self.synthesize(line["text"], **kwargs)

        if concurrent and len(lines) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(lines)) as pool:
                results = list(pool.map(synth, lines))
        else:
            results = [synth(line) for line in lines]

        sample_rate = results[0][1]
        segments: list[np.ndarray] = []
        gap = np.zeros(int(sample_rate * gap_seconds), dtype=results[0][0].dtype) if gap_seconds > 0 else None
        for idx, (samples, _) in enumerate(results):
            segments.append(samples)
            if gap is not None and idx < len(results) - 1:
                segments.append(gap)

        audio = np.concatenate(segments)
        sd.play(audio, sample_rate)
        sd.wait()


_default: Stackvox | None = None
_default_lock = threading.Lock()


def _get_default() -> Stackvox:
    """Lazily build a module-level engine, safe under concurrent first calls.

    Without the lock, two threads racing to call speak() / synthesize() at
    process start could each instantiate Stackvox — meaning two 340 MB model
    loads. Double-checked locking keeps the fast path lock-free once
    initialised.
    """
    global _default
    if _default is None:
        with _default_lock:
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
