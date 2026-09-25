"""Optional backend for local MLX Audio text-to-speech models.

Any model MLX Audio can load natively works as-is. Models that need extra
handling (renamed weights, offline codec paths, tokenizer quirks) can be
adapted with a user-supplied adapter script, so stackvox itself carries no
model-specific code. See ``load_adapter`` for the adapter contract.
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, cast

import numpy as np

# Adapters typically swap class attributes on third-party modules, which is
# process-global; serialise loads so two engines can't interleave the swaps.
_LOAD_LOCK = Lock()
# Long inputs are generated in sentence-packed chunks this size or smaller:
# small enough to stay under typical per-call token limits and give prompt
# first audio when streaming, large enough to keep prosody natural.
CHUNK_CHARS = 400
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")

# Cap on how much of an underlying error is echoed; MLX Audio's messages can
# list hundreds of parameter names.
_ERROR_SUMMARY_CHARS = 200

LoadModel = Callable[[Path], Any]
Adapter = Callable[[Path, LoadModel], Any]


class ModelLoadError(RuntimeError):
    """A local MLX model, or the adapter meant to load it, could not be loaded."""


def _summarize(exc: BaseException) -> str:
    first_line = next((line.strip() for line in str(exc).splitlines() if line.strip()), "")
    if len(first_line) > _ERROR_SUMMARY_CHARS:
        first_line = first_line[: _ERROR_SUMMARY_CHARS - 1] + "…"
    return f"{type(exc).__name__}: {first_line}" if first_line else type(exc).__name__


@contextmanager
def swapped(owner: type, name: str, value: Any) -> Iterator[None]:
    """Replace ``owner.name`` for the duration of the block, restoring the
    original descriptor (or the inherited lookup) afterwards.

    Public so adapter scripts can patch third-party classes during a load
    without leaking the patch into the rest of the process.
    """
    own = vars(owner)
    had_own = name in own
    original = own.get(name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        if had_own:
            setattr(owner, name, original)
        else:
            delattr(owner, name)


def split_long_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Pack prose into sentence-aware chunks no longer than ``limit``."""
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        pieces = [sentence]
        if len(sentence) > limit:
            words = sentence.split()
            pieces = []
            piece = ""
            for word in words:
                candidate = f"{piece} {word}".strip()
                if piece and len(candidate) > limit:
                    pieces.append(piece)
                    piece = word
                else:
                    piece = candidate
            if piece:
                pieces.append(piece)
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > limit:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def load_adapter(path: Path) -> Adapter:
    """Import an adapter script and return its ``load`` function.

    An adapter is a Python file defining::

        def load(model_dir: Path, load_model: Callable[[Path], Any]) -> Any

    ``load_model`` is MLX Audio's own loader; the adapter may patch things
    around it (see ``swapped``), call it, or replace it outright, and must
    return the loaded model. The script runs with the caller's privileges, so
    only point this at code you trust — it is never discovered implicitly.
    """
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ModelLoadError(f"model adapter not found: {path}")
    spec = importlib.util.spec_from_file_location(f"stackvox_adapter_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ModelLoadError(f"model adapter is not an importable Python file: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ModelLoadError(f"model adapter {path} raised while importing ({_summarize(exc)})") from exc
    adapter = getattr(module, "load", None)
    if not callable(adapter):
        raise ModelLoadError(f"model adapter {path} must define load(model_dir, load_model)")
    return cast(Adapter, adapter)


def _load_model(model_dir: Path, adapter_path: Path | None) -> Any:
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ModelLoadError(
            "the mlx backend requires MLX Audio; install it with `pip install 'stackvox[mlx]'`"
        ) from exc
    if adapter_path is None:
        with _LOAD_LOCK:
            try:
                return load_model(model_dir)
            except Exception as exc:
                raise ModelLoadError(
                    f"MLX Audio could not load the model in {model_dir} ({_summarize(exc)}). "
                    "If this model needs custom loading, pass a model adapter with --model-adapter "
                    "(see 'Model adapters' in the README)."
                ) from exc
    adapter = load_adapter(adapter_path)
    with _LOAD_LOCK:
        try:
            return adapter(model_dir, load_model)
        except Exception as exc:
            raise ModelLoadError(
                f"model adapter {adapter_path} failed to load the model in {model_dir} ({_summarize(exc)})"
            ) from exc


def _load_audio(path: Path, sample_rate: int) -> Any:
    from mlx_audio.utils import load_audio

    return load_audio(str(path), sample_rate=sample_rate)


class MlxBackend:
    """Load one local MLX Audio model and expose stackvox's synthesis contract."""

    def __init__(self, model_dir: Path, adapter: Path | None = None) -> None:
        model_dir = model_dir.expanduser().resolve()
        if not model_dir.is_dir():
            raise ModelLoadError(f"model directory not found: {model_dir}")
        self.model_dir = model_dir
        self._model: Any = _load_model(model_dir, adapter)

    def synthesize(
        self,
        text: str,
        *,
        speed: float = 1.0,
        reference_audio: Path | str | None = None,
        reference_text: str | None = None,
        instruct: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Generate audio, forwarding only the options the caller set so each
        model's own defaults apply to the rest."""
        if (reference_audio is None) != (reference_text is None):
            raise ValueError("voice cloning requires both reference_audio and reference_text")

        kwargs: dict[str, Any] = {"speed": speed, "verbose": False}
        if reference_audio is not None:
            audio_path = Path(reference_audio).expanduser().resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"reference audio not found: {audio_path}")
            kwargs["ref_audio"] = _load_audio(audio_path, self._model.sample_rate)
            kwargs["ref_text"] = reference_text
        optional = {"instruct": instruct, "temperature": temperature, "top_p": top_p, "top_k": top_k}
        kwargs.update({key: value for key, value in optional.items() if value is not None})

        results = []
        for chunk in split_long_text(text):
            results.extend(self._model.generate(text=chunk, **kwargs))
        if not results:
            raise RuntimeError("model produced no audio")
        sample_rates = {result.sample_rate for result in results}
        if len(sample_rates) != 1:
            raise RuntimeError("model returned segments with different sample rates")
        segments = [np.asarray(result.audio, dtype=np.float32) for result in results]
        return np.ascontiguousarray(np.concatenate(segments)), sample_rates.pop()
