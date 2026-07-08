"""Engine tests — no real model load, no audio device.

We mock at two boundaries: `Kokoro` (so `Stackvox.__init__` doesn't pull the
model) and `sounddevice` (so `speak()` doesn't try to talk to PortAudio).
This lets us verify call-shape and side effects without any of the heavy
runtime dependencies.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import pytest

from stackvox import engine


@pytest.fixture
def fake_ensure_models(mocker, tmp_path):
    """Short-circuit _ensure_models so Stackvox.__init__ skips the download."""
    model = tmp_path / "kokoro.onnx"
    voices = tmp_path / "voices.bin"
    model.touch()
    voices.touch()
    return mocker.patch.object(engine, "_ensure_models", return_value=(model, voices))


@pytest.fixture
def fake_kokoro(mocker, fake_ensure_models):
    """Mock Kokoro on top of fake_ensure_models — most tests want both."""
    return mocker.patch.object(engine, "Kokoro")


@pytest.fixture
def fake_audio(mocker):
    """Mock the sounddevice surface used by Stackvox."""
    mocker.patch.object(engine.sd, "play")
    mocker.patch.object(engine.sd, "wait")
    mocker.patch.object(engine.sd, "stop")


@pytest.fixture(autouse=True)
def reset_default():
    """Each test starts with no module-level singleton."""
    engine._default = None
    yield
    engine._default = None


class TestStackvoxInit:
    def test_stores_voice_speed_lang(self, fake_kokoro):
        tts = engine.Stackvox(voice="bf_emma", speed=1.2, lang="en-gb")
        assert tts.voice == "bf_emma"
        assert tts.speed == 1.2
        assert tts.lang == "en-gb"

    def test_uses_custom_cache_dir(self, fake_kokoro, fake_ensure_models, tmp_path):
        custom = tmp_path / "custom-cache"
        engine.Stackvox(cache_dir=custom)
        fake_ensure_models.assert_called_once_with(custom)

    def test_falls_back_to_default_cache_dir(self, fake_kokoro, fake_ensure_models, mocker, tmp_path):
        default = tmp_path / "default"
        mocker.patch.object(engine, "_default_cache_dir", return_value=default)
        engine.Stackvox()
        fake_ensure_models.assert_called_once_with(default)


class TestSynthesize:
    def test_passes_engine_defaults_to_kokoro(self, fake_kokoro):
        samples = np.zeros(10, dtype=np.float32)
        fake_kokoro.return_value.create.return_value = (samples, 24000)

        tts = engine.Stackvox(voice="af_sarah", speed=1.0, lang="en-us")
        result_samples, result_sr = tts.synthesize("hi")

        fake_kokoro.return_value.create.assert_called_once_with(
            "hi", voice="af_sarah", speed=1.0, lang="en-us"
        )
        assert result_sr == 24000
        assert np.array_equal(result_samples, samples)

    def test_per_call_overrides_take_priority(self, fake_kokoro):
        fake_kokoro.return_value.create.return_value = (np.zeros(10), 24000)
        tts = engine.Stackvox(voice="af_sarah", speed=1.0, lang="en-us")
        tts.synthesize("hi", voice="bf_emma", speed=1.5, lang="en-gb")
        fake_kokoro.return_value.create.assert_called_once_with(
            "hi", voice="bf_emma", speed=1.5, lang="en-gb"
        )

    def test_speed_zero_override_is_respected(self, fake_kokoro):
        """speed=0 is falsy but a valid override; ensure it's not silently dropped."""
        fake_kokoro.return_value.create.return_value = (np.zeros(10), 24000)
        tts = engine.Stackvox(speed=1.0)
        tts.synthesize("hi", speed=0.5)
        kwargs = fake_kokoro.return_value.create.call_args.kwargs
        assert kwargs["speed"] == 0.5


class TestSpeak:
    def test_blocking_synthesizes_and_plays_each_sentence(self, fake_kokoro, fake_audio):
        fake_kokoro.return_value.create.return_value = (np.zeros(10, dtype=np.float32), 24000)

        engine.Stackvox().speak("One sentence. Two sentence. Three here.")

        # One synth + play cycle per sentence — that's what enables early audio.
        assert fake_kokoro.return_value.create.call_count == 3
        assert engine.sd.play.call_count == 3
        assert engine.sd.wait.call_count == 3

    def test_single_sentence_plays_once(self, fake_kokoro, fake_audio):
        fake_kokoro.return_value.create.return_value = (np.zeros(10, dtype=np.float32), 24000)
        engine.Stackvox().speak("Just one sentence with no terminal punctuation")
        assert engine.sd.play.call_count == 1

    def test_non_blocking_plays_in_background_thread(self, fake_kokoro, fake_audio):
        fake_kokoro.return_value.create.return_value = (np.zeros(10, dtype=np.float32), 24000)
        tts = engine.Stackvox()

        tts.speak("hi there", blocking=False)

        assert tts._play_thread is not None
        tts._play_thread.join(timeout=5)
        engine.sd.play.assert_called()

    def test_passes_engine_defaults_to_synthesis(self, fake_kokoro, fake_audio):
        fake_kokoro.return_value.create.return_value = (np.zeros(4, dtype=np.float32), 24000)
        engine.Stackvox(voice="af_sarah", speed=1.0, lang="en-us").speak("hello")
        fake_kokoro.return_value.create.assert_called_once_with(
            "hello", voice="af_sarah", speed=1.0, lang="en-us"
        )

    def test_per_call_overrides_take_priority(self, fake_kokoro, fake_audio):
        fake_kokoro.return_value.create.return_value = (np.zeros(4, dtype=np.float32), 24000)
        engine.Stackvox(voice="af_sarah", speed=1.0, lang="en-us").speak(
            "hi", voice="bf_emma", speed=1.5, lang="en-gb"
        )
        fake_kokoro.return_value.create.assert_called_once_with(
            "hi", voice="bf_emma", speed=1.5, lang="en-gb"
        )

    def test_stop_sets_event_and_calls_sounddevice_stop(self, fake_kokoro, fake_audio):
        tts = engine.Stackvox()
        tts.stop()
        assert tts._stop_event.is_set()
        engine.sd.stop.assert_called_once()

    def test_blocking_surfaces_synthesis_errors(self, fake_kokoro, fake_audio):
        # Regression: a Kokoro failure must propagate, not report silent success.
        fake_kokoro.return_value.create.side_effect = RuntimeError("boom")
        tts = engine.Stackvox()
        with pytest.raises(RuntimeError, match="boom"):
            tts.speak("hello there")

    def test_non_blocking_logs_synthesis_errors_without_crashing(self, fake_kokoro, fake_audio, caplog):
        fake_kokoro.return_value.create.side_effect = RuntimeError("boom")
        tts = engine.Stackvox()
        with caplog.at_level(logging.ERROR):
            tts.speak("hi there", blocking=False)
            assert tts._play_thread is not None
            tts._play_thread.join(timeout=5)
        assert "playback failed" in caplog.text

    def test_each_speak_uses_a_fresh_stop_event(self, fake_kokoro, fake_audio):
        # A per-stream event means a later speak() can't clear an older stream's
        # cancellation (and stop() only affects the stream it was called on).
        fake_kokoro.return_value.create.return_value = (np.zeros(4, dtype=np.float32), 24000)
        tts = engine.Stackvox()
        original = tts._stop_event
        tts.speak("one sentence")
        assert tts._stop_event is not original

    def test_speak_after_stop_still_plays(self, fake_kokoro, fake_audio):
        fake_kokoro.return_value.create.return_value = (np.zeros(4, dtype=np.float32), 24000)
        tts = engine.Stackvox()
        tts.stop()
        tts.speak("hello world")
        engine.sd.play.assert_called()


class TestSplitSentences:
    def test_splits_on_sentence_punctuation(self):
        assert engine._split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_leaves_decimals_and_mid_token_dots_intact(self):
        assert engine._split_sentences("Set speed 0.95 in config.toml now.") == [
            "Set speed 0.95 in config.toml now."
        ]

    def test_splits_on_newlines(self):
        assert engine._split_sentences("Line one\nLine two\n\nLine three") == [
            "Line one",
            "Line two",
            "Line three",
        ]

    def test_blank_text_yields_no_sentences(self):
        assert engine._split_sentences("   \n  ") == []


class TestVoices:
    def test_returns_sorted_voice_ids(self, fake_kokoro):
        fake_kokoro.return_value.get_voices.return_value = ["bf_emma", "af_sarah", "am_adam"]
        tts = engine.Stackvox()
        assert tts.voices() == ["af_sarah", "am_adam", "bf_emma"]


class TestSpeakSequence:
    def test_empty_lines_returns_early(self, fake_kokoro, fake_audio):
        engine.Stackvox().speak_sequence([])
        engine.sd.play.assert_not_called()

    def test_concatenates_segments_with_optional_gap(self, fake_kokoro, fake_audio):
        a = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        b = np.array([2.0, 2.0], dtype=np.float32)
        fake_kokoro.return_value.create.side_effect = [(a, 100), (b, 100)]

        tts = engine.Stackvox()
        tts.speak_sequence([{"text": "first"}, {"text": "second"}], gap_seconds=0.05, concurrent=False)

        played = engine.sd.play.call_args.args[0]
        # Length: 3 + (100 * 0.05) silence + 2 = 10 samples.
        assert len(played) == 3 + 5 + 2
        # First 3 samples are `a`, last 2 are `b`, middle 5 are silence.
        np.testing.assert_array_equal(played[:3], a)
        np.testing.assert_array_equal(played[-2:], b)
        np.testing.assert_array_equal(played[3:8], np.zeros(5, dtype=np.float32))


class TestModuleLevelHelpers:
    def test_speak_reuses_module_singleton(self, fake_kokoro, fake_audio):
        fake_kokoro.return_value.create.return_value = (np.zeros(10), 24000)
        engine.speak("first")
        engine.speak("second")
        # Stackvox class only instantiated once → Kokoro only constructed once.
        assert fake_kokoro.call_count == 1

    def test_synthesize_reuses_module_singleton(self, fake_kokoro):
        fake_kokoro.return_value.create.return_value = (np.zeros(10), 24000)
        engine.synthesize("first")
        engine.synthesize("second")
        assert fake_kokoro.call_count == 1

    def test_get_default_is_thread_safe(self, fake_kokoro):
        """Concurrent first calls must not double-instantiate Stackvox.

        Without the lock around _get_default, multiple threads calling
        speak()/synthesize() at process start can both observe `_default is
        None` and each construct a Stackvox (each loading the 340 MB model).
        """
        fake_kokoro.return_value.create.return_value = (np.zeros(10), 24000)
        barrier = threading.Barrier(8)

        def race():
            barrier.wait()
            engine._get_default()

        threads = [threading.Thread(target=race) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert fake_kokoro.call_count == 1


class TestDownloadProgress:
    def test_reporthook_prints_percentage_when_size_known(self, mocker, capsys, tmp_path):
        """The hook should print updating progress lines while downloading."""
        captured_hook = {}

        def fake_urlretrieve(url, dest, reporthook=None):
            captured_hook["fn"] = reporthook
            # Simulate three chunks of a 1000-byte download.
            if reporthook:
                reporthook(0, 100, 1000)
                reporthook(5, 100, 1000)
                reporthook(10, 100, 1000)

        mocker.patch.object(engine.urllib.request, "urlretrieve", side_effect=fake_urlretrieve)
        engine._download_with_progress("http://example/m.bin", tmp_path / "m.bin")

        err = capsys.readouterr().err
        assert "downloading m.bin" in err
        assert "100%" in err

    def test_unknown_size_does_not_print(self, mocker, capsys, tmp_path):
        """totalsize <= 0 means Content-Length absent; hook should be a no-op."""

        def fake_urlretrieve(url, dest, reporthook=None):
            if reporthook:
                reporthook(0, 100, -1)

        mocker.patch.object(engine.urllib.request, "urlretrieve", side_effect=fake_urlretrieve)
        engine._download_with_progress("http://example/m.bin", tmp_path / "m.bin")

        # No percentage should have been emitted because total size was unknown.
        assert "downloading" not in capsys.readouterr().err


class TestEnsureModels:
    def test_skips_files_that_already_exist(self, mocker, tmp_path):
        download = mocker.patch.object(engine, "_download_with_progress")
        (tmp_path / "kokoro-v1.0.onnx").touch()
        (tmp_path / "voices-v1.0.bin").touch()

        engine._ensure_models(tmp_path)

        download.assert_not_called()

    def test_downloads_only_missing_files(self, mocker, tmp_path):
        download = mocker.patch.object(engine, "_download_with_progress")
        # Only the model exists; voices needs downloading.
        (tmp_path / "kokoro-v1.0.onnx").touch()

        engine._ensure_models(tmp_path)

        # Exactly one download, for the voices file.
        assert download.call_count == 1
        url_arg, dest_arg = download.call_args.args
        assert dest_arg.name == "voices-v1.0.bin"
