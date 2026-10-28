"""Optional backend for local MLX Audio text-to-speech models.

Any model MLX Audio can load natively works as-is. Models that need extra
handling (renamed weights, offline codec paths, tokenizer quirks) can be
adapted with a user-supplied adapter script, so stackvox itself carries no
model-specific code. See ``load_adapter`` for the adapter contract.
"""

from __future__ import annotations

import importlib.util
import queue
import re
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar, cast

import numpy as np

# Adapters typically swap class attributes on third-party modules, which is
# process-global; serialise loads so two engines can't interleave the swaps.
_LOAD_LOCK = Lock()
# Long inputs are generated in sentence-packed chunks this size or smaller,
# roughly two sentences. Autoregressive TTS models tend to drift on long
# prompts, skipping or inventing speech; in testing, 400-char chunks lost a
# quarter to a third of the text while ~170 kept all of it. Smaller chunks also
# bring the first streamed audio forward.
CHUNK_CHARS = 170
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")
# Over-long sentences, and the next sentence when a short chunk is topped up,
# are cut here before falling back to word boundaries, so the pause between
# chunks lands where a speaker would take a breath. A typed
# dash counts only with space either side, so hyphenated words stay whole.
_CLAUSE_BOUNDARY = re.compile(r"(?<=[,;:—–])\s+|(?<=\s-)\s+|(?<=\s--)\s+")
# Streaming leaves a gap whenever a chunk takes longer to generate than the one
# before it takes to play, so a short sentence sent alone ahead of a long one
# ("Yes." then 170 characters) is the worst case for models near real time.
# A chunk shorter than this that the next piece won't fit beside takes the
# opening of that piece, up to the chunk limit, instead.
_MIN_CHUNK_CHARS = 60

# MLX keeps freed GPU buffers in its own cache for reuse, and by default lets it
# grow to the device's recommended working set (tens of GB on a large Mac). A
# long-running daemon generating on and off would hold all of it, so cap it.
_BUFFER_CACHE_LIMIT_BYTES = 1 << 30

_MISSING_MLX = "the mlx backend requires MLX Audio; install it with `pip install 'stackvox[mlx]'`"

# Cap on how much of an underlying error is echoed; MLX Audio's messages can
# list hundreds of parameter names.
_ERROR_SUMMARY_CHARS = 200

LoadModel = Callable[[Path], Any]
Adapter = Callable[[Path, LoadModel], Any]
_Result = TypeVar("_Result")


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


def _split_words(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    piece = ""
    for word in text.split():
        candidate = f"{piece} {word}".strip()
        if piece and len(candidate) > limit:
            pieces.append(piece)
            piece = word
        else:
            piece = candidate
    if piece:
        pieces.append(piece)
    return pieces


def _sentence_pieces(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    pieces: list[str] = []
    for clause in (part.strip() for part in _CLAUSE_BOUNDARY.split(sentence)):
        if clause:
            pieces.extend([clause] if len(clause) <= limit else _split_words(clause, limit))
    return pieces


def _fill(chunk: str, parts: list[str], limit: int) -> tuple[str, str]:
    taken = 0
    while taken < len(parts) and len(" ".join([chunk, *parts[: taken + 1]])) <= limit:
        taken += 1
    return " ".join([chunk, *parts[:taken]]), " ".join(parts[taken:])


def _top_up(chunk: str, piece: str, limit: int) -> tuple[str, str]:
    """Move as much of ``piece``'s opening onto ``chunk`` as ``limit`` allows.

    Whole clauses are moved when they bring ``chunk`` up to ``_MIN_CHUNK_CHARS``,
    so the cut lands at punctuation; otherwise words are, and the cut lands
    mid clause.
    """
    topped, rest = _fill(chunk, _CLAUSE_BOUNDARY.split(piece), limit)
    if len(topped) >= _MIN_CHUNK_CHARS:
        return topped, rest
    return _fill(chunk, piece.split(), limit)


def split_long_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """Pack prose into chunks no longer than ``limit``.

    Chunks break between sentences where they can. A sentence longer than
    ``limit`` breaks at clause punctuation, and only a clause still longer than
    that breaks between words. A chunk under ``_MIN_CHUNK_CHARS`` that the next
    piece won't fit beside takes that piece's opening clauses, or words where no
    clause fits, rather than going out alone.
    """
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for piece in (piece for sentence in sentences for piece in _sentence_pieces(sentence, limit)):
        candidate = f"{current} {piece}".strip()
        if not current or len(candidate) <= limit:
            current = candidate
            continue
        if len(current) < _MIN_CHUNK_CHARS:
            current, piece = _top_up(current, piece, limit)
        chunks.append(current)
        current = piece
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
        raise ModelLoadError(_MISSING_MLX) from exc
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


def _limit_buffer_cache() -> None:
    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ModelLoadError(_MISSING_MLX) from exc

    mx.set_cache_limit(_BUFFER_CACHE_LIMIT_BYTES)


def _clear_buffer_cache() -> None:
    import mlx.core as mx

    mx.clear_cache()


def _load_audio(path: Path, sample_rate: int) -> Any:
    from mlx_audio.utils import load_audio

    return load_audio(str(path), sample_rate=sample_rate)


class _MlxThread:
    """A single daemon thread that runs every MLX call for one backend.

    MLX binds GPU streams to threads: a model loaded on one thread fails with
    "There is no Stream(gpu, 0) in current thread" when it first generates on
    another, which is exactly what streaming playback and the daemon worker do.
    Funnelling load and generation through one thread avoids that and also
    serialises access to a model that isn't safe to call concurrently. A daemon
    thread (rather than an executor) never holds up interpreter exit.
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue[tuple[Callable[[], Any], Future[Any]]] = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True, name="stackvox-mlx")
        self.thread.start()

    def _run(self) -> None:
        while True:
            job, future = self._jobs.get()
            try:
                future.set_result(job())
            except BaseException as exc:  # hand every failure back to the caller
                future.set_exception(exc)

    def call(self, job: Callable[[], _Result]) -> _Result:
        future: Future[_Result] = Future()
        self._jobs.put((job, future))
        return future.result()


class MlxBackend:
    """Load one local MLX Audio model and expose stackvox's synthesis contract.

    Safe to call from any thread: all MLX work runs on the backend's own thread.
    """

    def __init__(self, model_dir: Path, adapter: Path | None = None) -> None:
        model_dir = model_dir.expanduser().resolve()
        if not model_dir.is_dir():
            raise ModelLoadError(f"model directory not found: {model_dir}")
        self.model_dir = model_dir
        self._mlx_thread = _MlxThread()
        self._mlx_thread.call(_limit_buffer_cache)
        self._model: Any = self._mlx_thread.call(lambda: _load_model(model_dir, adapter))

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

        audio_path = None
        if reference_audio is not None:
            audio_path = Path(reference_audio).expanduser().resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"reference audio not found: {audio_path}")
        optional = {"instruct": instruct, "temperature": temperature, "top_p": top_p, "top_k": top_k}
        kwargs: dict[str, Any] = {"speed": speed, "verbose": False}
        kwargs.update({key: value for key, value in optional.items() if value is not None})
        return self._mlx_thread.call(lambda: self._generate(text, kwargs, audio_path, reference_text))

    def _generate(
        self, text: str, kwargs: dict[str, Any], audio_path: Path | None, reference_text: str | None
    ) -> tuple[np.ndarray, int]:
        """Runs on the MLX thread: decoding the reference and converting results
        to NumPy both evaluate MLX arrays, so they belong here too.

        Empties MLX's buffer cache before returning: models don't reliably do it
        themselves, and what one request leaves behind is mostly sized for that
        request rather than the next. The work lives in ``_render`` so its MLX
        arrays are released, and their buffers already in the cache, by the
        time it is cleared.
        """
        try:
            return self._render(text, kwargs, audio_path, reference_text)
        finally:
            _clear_buffer_cache()

    def _render(
        self, text: str, kwargs: dict[str, Any], audio_path: Path | None, reference_text: str | None
    ) -> tuple[np.ndarray, int]:
        if audio_path is not None:
            kwargs = {
                **kwargs,
                "ref_audio": _load_audio(audio_path, self._model.sample_rate),
                "ref_text": reference_text,
            }
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
