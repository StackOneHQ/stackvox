"""MLX backend tests with MLX Audio mocked at the package boundary."""

from __future__ import annotations

import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from stackvox import mlx_backend


@pytest.fixture
def backend(mocker, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = mocker.MagicMock()
    mocker.patch.object(mlx_backend, "_load_model", return_value=model)
    return mlx_backend.MlxBackend(model_dir), model


@pytest.fixture
def fake_mlx_audio(mocker):
    """Install a stub `mlx_audio.tts.utils.load_model` so loading needs no MLX."""
    load_model = mocker.MagicMock(name="load_model")
    utils = ModuleType("mlx_audio.tts.utils")
    utils.load_model = load_model  # type: ignore[attr-defined]
    mocker.patch.dict(
        sys.modules,
        {
            "mlx_audio": ModuleType("mlx_audio"),
            "mlx_audio.tts": ModuleType("mlx_audio.tts"),
            "mlx_audio.tts.utils": utils,
        },
    )
    return load_model


def _write_adapter(tmp_path, body):
    path = tmp_path / "adapter.py"
    path.write_text(body)
    return path


class TestLoading:
    def test_loads_resolved_local_model(self, mocker, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        load = mocker.patch.object(mlx_backend, "_load_model")
        mlx_backend.MlxBackend(model_dir)
        load.assert_called_once_with(model_dir.resolve(), None)

    def test_missing_model_directory_fails_before_import(self, tmp_path):
        with pytest.raises(mlx_backend.ModelLoadError, match="model directory"):
            mlx_backend.MlxBackend(tmp_path / "missing")

    def test_without_adapter_uses_mlx_audio_loader(self, fake_mlx_audio, tmp_path):
        actual = mlx_backend._load_model(tmp_path, None)
        fake_mlx_audio.assert_called_once_with(tmp_path)
        assert actual is fake_mlx_audio.return_value

    def test_adapter_receives_model_dir_and_default_loader(self, fake_mlx_audio, tmp_path):
        adapter = _write_adapter(
            tmp_path,
            "def load(model_dir, load_model):\n    return ('adapted', model_dir, load_model(model_dir))\n",
        )

        actual = mlx_backend._load_model(tmp_path, adapter)

        assert actual == ("adapted", tmp_path, fake_mlx_audio.return_value)


class TestLoadErrors:
    def test_native_failure_suggests_model_adapter(self, fake_mlx_audio, tmp_path):
        fake_mlx_audio.side_effect = ValueError("Missing 358 parameters: \nlayer.0.weight,\nlayer.1.weight")

        with pytest.raises(mlx_backend.ModelLoadError) as exc:
            mlx_backend._load_model(tmp_path, None)

        message = str(exc.value)
        assert f"could not load the model in {tmp_path}" in message
        assert "ValueError: Missing 358 parameters:" in message
        assert "layer.0.weight" not in message
        assert "--model-adapter" in message
        assert isinstance(exc.value.__cause__, ValueError)

    def test_long_underlying_message_is_truncated(self, fake_mlx_audio, tmp_path):
        fake_mlx_audio.side_effect = ValueError("x" * 1000)

        with pytest.raises(mlx_backend.ModelLoadError) as exc:
            mlx_backend._load_model(tmp_path, None)

        assert "x" * (mlx_backend._ERROR_SUMMARY_CHARS - 1) + "…" in str(exc.value)
        assert "x" * mlx_backend._ERROR_SUMMARY_CHARS not in str(exc.value)

    def test_adapter_failure_names_the_adapter_without_suggesting_one(self, fake_mlx_audio, tmp_path):
        adapter = _write_adapter(tmp_path, "def load(model_dir, load_model):\n    raise KeyError('codec')\n")

        with pytest.raises(mlx_backend.ModelLoadError) as exc:
            mlx_backend._load_model(tmp_path, adapter)

        message = str(exc.value)
        assert f"model adapter {adapter} failed to load" in message
        assert "KeyError: 'codec'" in message
        assert "--model-adapter" not in message

    def test_adapter_import_error_is_reported(self, tmp_path):
        adapter = _write_adapter(tmp_path, "import not_a_real_module_xyz\n")

        with pytest.raises(mlx_backend.ModelLoadError, match="raised while importing.*ModuleNotFoundError"):
            mlx_backend.load_adapter(adapter)


class TestLoadAdapter:
    def test_returns_load_function(self, tmp_path):
        adapter = _write_adapter(tmp_path, "def load(model_dir, load_model):\n    return 42\n")
        assert mlx_backend.load_adapter(adapter)(tmp_path, lambda model_dir: None) == 42

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(mlx_backend.ModelLoadError, match="model adapter not found"):
            mlx_backend.load_adapter(tmp_path / "nope.py")

    def test_script_without_load_is_an_error(self, tmp_path):
        adapter = _write_adapter(tmp_path, "value = 1\n")
        with pytest.raises(mlx_backend.ModelLoadError, match="must define load"):
            mlx_backend.load_adapter(adapter)


class TestSynthesize:
    def test_concatenates_generated_segments(self, backend):
        engine, model = backend
        model.generate.return_value = iter(
            [
                SimpleNamespace(audio=np.array([0.1, 0.2]), sample_rate=44100),
                SimpleNamespace(audio=np.array([0.3]), sample_rate=44100),
            ]
        )

        samples, sample_rate = engine.synthesize("hello")

        np.testing.assert_allclose(samples, [0.1, 0.2, 0.3])
        assert samples.dtype == np.float32
        assert sample_rate == 44100

    def test_passes_reference_and_sampling_options(self, backend, mocker, tmp_path):
        engine, model = backend
        loaded_audio = object()
        load_audio = mocker.patch.object(mlx_backend, "_load_audio", return_value=loaded_audio)
        reference = tmp_path / "voice.wav"
        reference.touch()
        model.generate.return_value = iter([SimpleNamespace(audio=np.zeros(2), sample_rate=44100)])

        engine.synthesize(
            "hello",
            reference_audio=str(reference),
            reference_text="Reference transcript.",
            instruct="Speak warmly.",
            temperature=0.8,
            top_p=0.9,
            top_k=20,
        )

        kwargs = model.generate.call_args.kwargs
        load_audio.assert_called_once_with(reference.resolve(), model.sample_rate)
        assert kwargs["ref_audio"] is loaded_audio
        assert kwargs["ref_text"] == "Reference transcript."
        assert kwargs["instruct"] == "Speak warmly."
        assert kwargs["temperature"] == 0.8
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 20

    def test_unset_options_are_left_to_model_defaults(self, backend):
        engine, model = backend
        model.generate.return_value = iter([SimpleNamespace(audio=np.zeros(2), sample_rate=44100)])

        engine.synthesize("hello")

        assert set(model.generate.call_args.kwargs) == {"text", "speed", "verbose"}

    @pytest.mark.parametrize(
        ("reference_audio", "reference_text"),
        [("voice.wav", None), (None, "Transcript")],
    )
    def test_reference_audio_and_text_are_required_together(self, backend, reference_audio, reference_text):
        engine, _ = backend
        with pytest.raises(ValueError, match="requires both"):
            engine.synthesize("hello", reference_audio=reference_audio, reference_text=reference_text)

    def test_missing_reference_audio_file_is_an_error(self, backend, tmp_path):
        engine, _ = backend
        with pytest.raises(FileNotFoundError, match="reference audio"):
            engine.synthesize("hello", reference_audio=tmp_path / "nope.wav", reference_text="Transcript")

    def test_empty_generation_is_an_error(self, backend):
        engine, model = backend
        model.generate.return_value = iter([])
        with pytest.raises(RuntimeError, match="produced no audio"):
            engine.synthesize("hello")

    def test_mixed_sample_rates_are_an_error(self, backend):
        engine, model = backend
        model.generate.return_value = iter(
            [
                SimpleNamespace(audio=np.zeros(1), sample_rate=24000),
                SimpleNamespace(audio=np.zeros(1), sample_rate=44100),
            ]
        )
        with pytest.raises(RuntimeError, match="different sample rates"):
            engine.synthesize("hello")

    def test_long_text_is_generated_in_multiple_sentence_chunks(self, backend):
        engine, model = backend
        model.generate.side_effect = lambda **kwargs: iter(
            [SimpleNamespace(audio=np.ones(1), sample_rate=44100)]
        )
        text = "First sentence. " * 80

        samples, _ = engine.synthesize(text)

        assert model.generate.call_count > 1
        assert len(samples) == model.generate.call_count
        assert all(
            len(call.kwargs["text"]) <= mlx_backend.CHUNK_CHARS for call in model.generate.call_args_list
        )


class TestSwapped:
    def test_restores_own_attribute(self):
        class Owner:
            @classmethod
            def build(cls):
                return "original"

        with mlx_backend.swapped(Owner, "build", lambda: "patched"):
            assert Owner.build() == "patched"

        assert Owner.build() == "original"
        assert isinstance(vars(Owner)["build"], classmethod)

    def test_removes_override_of_inherited_attribute(self):
        class Base:
            def run(self):
                return "base"

        class Child(Base):
            pass

        with mlx_backend.swapped(Child, "run", lambda self: "patched"):
            assert Child().run() == "patched"

        assert "run" not in vars(Child)
        assert Child().run() == "base"

    def test_restores_after_exception(self):
        class Owner:
            value = 1

        with pytest.raises(RuntimeError), mlx_backend.swapped(Owner, "value", 2):
            raise RuntimeError

        assert Owner.value == 1


class TestSplitLongText:
    def test_short_text_is_one_chunk(self):
        assert mlx_backend.split_long_text("One. Two.") == ["One. Two."]

    def test_blank_text_yields_no_chunks(self):
        assert mlx_backend.split_long_text("  \n ") == []

    def test_overlong_sentence_splits_on_words(self):
        chunks = mlx_backend.split_long_text("word " * 200, limit=50)
        assert all(len(chunk) <= 50 for chunk in chunks)
        assert " ".join(chunks).split() == ["word"] * 200


class TestMlxThread:
    def test_load_and_generation_share_one_thread_across_callers(self, mocker, tmp_path):
        seen = []
        model = mocker.MagicMock(sample_rate=44100)

        def record(name):
            def side_effect(*args, **kwargs):
                seen.append((name, threading.current_thread().name))
                return model if name == "load" else object()

            return side_effect

        def generate(**kwargs):
            seen.append(("generate", threading.current_thread().name))
            return iter([SimpleNamespace(audio=np.ones(1), sample_rate=44100)])

        model.generate.side_effect = generate
        mocker.patch.object(mlx_backend, "_load_model", side_effect=record("load"))
        mocker.patch.object(mlx_backend, "_load_audio", side_effect=record("load_audio"))
        reference = tmp_path / "voice.wav"
        reference.touch()
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        engine = mlx_backend.MlxBackend(model_dir)

        callers = [
            threading.Thread(
                target=engine.synthesize,
                args=("hello",),
                kwargs={"reference_audio": reference, "reference_text": "Ref."},
            )
            for _ in range(3)
        ]
        for caller in callers:
            caller.start()
        for caller in callers:
            caller.join()
        engine.synthesize("from the main thread")

        assert {thread for _, thread in seen} == {"stackvox-mlx"}
        assert [name for name, _ in seen].count("generate") == 4

    def test_generation_errors_reach_the_caller(self, backend):
        engine, model = backend
        model.generate.side_effect = RuntimeError("gpu fell over")
        with pytest.raises(RuntimeError, match="gpu fell over"):
            engine.synthesize("hello")

    def test_load_errors_reach_the_caller(self, mocker, tmp_path):
        mocker.patch.object(mlx_backend, "_load_model", side_effect=mlx_backend.ModelLoadError("bad model"))
        with pytest.raises(mlx_backend.ModelLoadError, match="bad model"):
            mlx_backend.MlxBackend(tmp_path)

    def test_chunk_limit_keeps_about_two_sentences(self):
        sentence = "This is sentence number one of a longer passage used to check chunking."
        chunks = mlx_backend.split_long_text(" ".join([sentence] * 6))
        assert all(chunk.count(".") <= 2 for chunk in chunks)
