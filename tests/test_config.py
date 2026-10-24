"""Config loader tests — pure file/env logic, no engine touched."""

from __future__ import annotations

import logging
from pathlib import Path

from stackvox import config
from stackvox.engine import DEFAULT_LANG, DEFAULT_SPEED, DEFAULT_VOICE, MLX_OPTIONS
from stackvox.voices import BUILTIN_VOICES, VoiceMix


class TestConfigPath:
    def test_stackvox_config_env_takes_priority(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STACKVOX_CONFIG", str(tmp_path / "elsewhere.toml"))
        assert config.config_path() == tmp_path / "elsewhere.toml"

    def test_xdg_config_home_when_set(self, monkeypatch, tmp_path):
        monkeypatch.delenv("STACKVOX_CONFIG", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        assert config.config_path() == tmp_path / "xdg" / "stackvox" / "config.toml"

    def test_falls_back_to_home_dotconfig(self, monkeypatch):
        monkeypatch.delenv("STACKVOX_CONFIG", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        from pathlib import Path

        assert config.config_path() == Path.home() / ".config" / "stackvox" / "config.toml"


class TestLoadDefaults:
    def test_missing_file_returns_built_in_defaults(self, tmp_path):
        actual = config.load_defaults(tmp_path / "absent.toml")
        assert actual == config.Defaults()
        assert actual.voice == DEFAULT_VOICE
        assert actual.speed == DEFAULT_SPEED
        assert actual.lang == DEFAULT_LANG

    def test_full_config_overrides_all_three(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('[defaults]\nvoice = "bf_emma"\nspeed = 1.25\nlang = "en-gb"\n', encoding="utf-8")
        actual = config.load_defaults(path)
        assert actual.voice == "bf_emma"
        assert actual.speed == 1.25
        assert actual.lang == "en-gb"

    def test_partial_config_keeps_built_in_for_missing_keys(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('[defaults]\nvoice = "bf_emma"\n', encoding="utf-8")
        actual = config.load_defaults(path)
        assert actual.voice == "bf_emma"
        # speed and lang come from the engine defaults.
        assert actual.speed == DEFAULT_SPEED
        assert actual.lang == DEFAULT_LANG

    def test_empty_file_returns_built_in_defaults(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text("", encoding="utf-8")
        assert config.load_defaults(path) == config.Defaults()

    def test_malformed_toml_logs_warning_and_returns_defaults(self, tmp_path, caplog):
        path = tmp_path / "config.toml"
        path.write_text("this is not valid = toml\n[defaults\nvoice =", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            actual = config.load_defaults(path)
        assert actual == config.Defaults()
        assert any("malformed stackvox config" in r.message for r in caplog.records)

    def test_defaults_section_must_be_a_table(self, tmp_path, caplog):
        """`defaults = "string"` is parseable TOML but the wrong shape — log and ignore."""
        path = tmp_path / "config.toml"
        path.write_text('defaults = "not-a-table"\n', encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            actual = config.load_defaults(path)
        assert actual == config.Defaults()
        assert any("must be a table" in r.message for r in caplog.records)


class TestCLIPicksUpConfig:
    """Smoke test: argparse defaults reflect the user's config file."""

    def test_voice_default_comes_from_config(self, mocker, monkeypatch, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('[defaults]\nvoice = "bf_emma"\nspeed = 1.3\n', encoding="utf-8")
        monkeypatch.setenv("STACKVOX_CONFIG", str(path))

        from stackvox import cli

        speak = mocker.patch.object(cli, "_cmd_speak", return_value=0)
        mocker.patch.object(cli.sys, "argv", ["stackvox", "speak", "hello"])
        mocker.patch.object(cli.sys.stdin, "isatty", return_value=True)

        assert cli.main() == 0
        args = speak.call_args.args[0]
        assert args.voice == "bf_emma"
        # Speed and lang resolve per command, so a voice mix can sit between flag and config.
        params, _ = cli._voice_params(args)
        assert params.speed == 1.3
        # Lang wasn't in config; should fall through to built-in default.
        assert params.lang == DEFAULT_LANG

    def test_explicit_flag_overrides_config(self, mocker, monkeypatch, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('[defaults]\nvoice = "bf_emma"\n', encoding="utf-8")
        monkeypatch.setenv("STACKVOX_CONFIG", str(path))

        from stackvox import cli

        speak = mocker.patch.object(cli, "_cmd_speak", return_value=0)
        mocker.patch.object(cli.sys, "argv", ["stackvox", "speak", "--voice", "af_sarah", "hello"])
        mocker.patch.object(cli.sys.stdin, "isatty", return_value=True)

        assert cli.main() == 0
        assert speak.call_args.args[0].voice == "af_sarah"


class TestLoadVoices:
    def test_missing_file_returns_built_in_voices(self, tmp_path):
        assert config.load_voices(tmp_path / "absent.toml") == BUILTIN_VOICES

    def test_table_adds_a_named_mix(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text(
            '[voices.narrator]\nmix = { am_michael = 0.6, bm_george = 0.4 }\nlang = "en-gb"\nspeed = 1\n',
            encoding="utf-8",
        )
        voices = config.load_voices(path)
        assert voices["narrator"] == VoiceMix(
            mix=(("am_michael", 0.6), ("bm_george", 0.4)), lang="en-gb", speed=1.0
        )
        assert voices["chip"] == BUILTIN_VOICES["chip"]

    def test_table_with_a_built_in_name_replaces_it(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text("[voices.chip]\nmix = { am_adam = 1 }\n", encoding="utf-8")
        assert config.load_voices(path)["chip"] == VoiceMix(mix=(("am_adam", 1.0),))

    def test_invalid_table_is_skipped_with_a_warning(self, tmp_path, caplog):
        path = tmp_path / "config.toml"
        path.write_text(
            '[voices.noweights]\nlang = "en-gb"\n[voices.negative]\nmix = { am_adam = -1 }\n'
            "[voices.ok]\nmix = { am_adam = 1 }\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            voices = config.load_voices(path)
        assert "noweights" not in voices
        assert "negative" not in voices
        assert "ok" in voices
        assert "[voices.noweights]" in caplog.text

    def test_voices_section_must_be_a_table(self, tmp_path, caplog):
        path = tmp_path / "config.toml"
        path.write_text('voices = "chip"\n', encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            assert config.load_voices(path) == BUILTIN_VOICES
        assert "[voices] must be a table" in caplog.text


class TestLoadServe:
    def _write(self, tmp_path, body):
        path = tmp_path / "config.toml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_missing_file_leaves_the_backend_to_the_flags(self, tmp_path):
        assert config.load_serve(tmp_path / "absent.toml") == config.ServeConfig()

    def test_file_without_a_serve_table_leaves_the_backend_to_the_flags(self, tmp_path):
        path = self._write(tmp_path, '[defaults]\nvoice = "bf_emma"\n')
        assert config.load_serve(path) == config.ServeConfig()

    def test_mlx_backend_with_model_and_generation_defaults(self, tmp_path):
        path = self._write(
            tmp_path,
            "[serve]\n"
            'backend = "mlx"\n'
            'model_dir = "~/models/voice"\n'
            'model_adapter = "~/adapters/voice.py"\n'
            'reference_audio = "~/ref.wav"\n'
            'reference_text = "Ref."\n'
            "temperature = 1\n"
            "top_k = 20\n",
        )

        actual = config.load_serve(path)

        assert actual.backend == "mlx"
        assert actual.model_dir == Path("~/models/voice").expanduser()
        assert actual.model_adapter == Path("~/adapters/voice.py").expanduser()
        assert actual.generation == {
            "reference_audio": Path("~/ref.wav").expanduser(),
            "reference_text": "Ref.",
            "temperature": 1.0,
            "top_k": 20,
        }

    def test_mlx_backend_without_model_dir_is_dropped(self, tmp_path, caplog):
        path = self._write(tmp_path, '[serve]\nbackend = "mlx"\n')
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            assert config.load_serve(path) == config.ServeConfig()
        assert "needs a usable model_dir" in caplog.text

    def test_unknown_backend_is_dropped(self, tmp_path, caplog):
        path = self._write(tmp_path, '[serve]\nbackend = "espeak"\n')
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            assert config.load_serve(path) == config.ServeConfig()
        assert "backend must be one of" in caplog.text

    def test_settings_without_a_backend_are_dropped(self, tmp_path, caplog):
        path = self._write(tmp_path, '[serve]\nmodel_dir = "/models/voice"\n')
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            assert config.load_serve(path) == config.ServeConfig()
        assert "need a backend" in caplog.text

    def test_kokoro_backend_ignores_mlx_only_settings(self, tmp_path, caplog):
        path = self._write(tmp_path, '[serve]\nbackend = "kokoro"\nmodel_dir = "/models/voice"\ntop_k = 5\n')
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            assert config.load_serve(path) == config.ServeConfig(backend="kokoro")
        assert "model_dir, top_k apply to the mlx backend only" in caplog.text

    def test_a_badly_typed_value_drops_only_that_key(self, tmp_path, caplog):
        path = self._write(
            tmp_path,
            '[serve]\nbackend = "mlx"\nmodel_dir = "/models/voice"\nmodel_adapter = 3\ntop_k = 1.5\ntop_p = 0.9\n',
        )
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            actual = config.load_serve(path)
        assert actual == config.ServeConfig(
            backend="mlx", model_dir=Path("/models/voice"), generation={"top_p": 0.9}
        )
        assert "model_adapter is not a usable path" in caplog.text
        assert "top_k is invalid" in caplog.text

    def test_boolean_is_not_a_number(self, tmp_path):
        path = self._write(tmp_path, '[serve]\nbackend = "mlx"\nmodel_dir = "/m"\ntemperature = true\n')
        assert config.load_serve(path).generation == {}

    def test_serve_section_must_be_a_table(self, tmp_path, caplog):
        path = self._write(tmp_path, 'serve = "mlx"\n')
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            assert config.load_serve(path) == config.ServeConfig()
        assert "[serve] must be a table" in caplog.text

    def test_parses_every_generation_option_the_engine_takes(self):
        assert set(config._GENERATION_PARSERS) == set(MLX_OPTIONS)

    def test_half_a_reference_pair_is_dropped(self, tmp_path, caplog):
        path = self._write(
            tmp_path, '[serve]\nbackend = "mlx"\nmodel_dir = "/m"\nreference_audio = "/ref.wav"\ntop_k = 5\n'
        )
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            actual = config.load_serve(path)
        assert actual.generation == {"top_k": 5}
        assert "only work together" in caplog.text

    def test_unexpandable_model_dir_is_dropped_rather_than_raising(self, tmp_path, caplog):
        path = self._write(tmp_path, '[serve]\nbackend = "mlx"\nmodel_dir = "~no-such-user-xyz/model"\n')
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            assert config.load_serve(path) == config.ServeConfig()
        assert "needs a usable model_dir" in caplog.text

    def test_unexpandable_adapter_is_dropped_rather_than_raising(self, tmp_path, caplog):
        path = self._write(
            tmp_path, '[serve]\nbackend = "mlx"\nmodel_dir = "/m"\nmodel_adapter = "~no-such-user-xyz/a.py"\n'
        )
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            actual = config.load_serve(path)
        assert actual.model_adapter is None
        assert actual.model_dir == Path("/m")
        assert "model_adapter is not a usable path" in caplog.text

    def test_text_without_audio_is_dropped_too(self, tmp_path, caplog):
        path = self._write(tmp_path, '[serve]\nbackend = "mlx"\nmodel_dir = "/m"\nreference_text = "Ref."\n')
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            actual = config.load_serve(path)
        assert actual.generation == {}
        assert "only work together" in caplog.text

    def test_relative_paths_resolve_against_the_config_folder(self, tmp_path):
        path = self._write(
            tmp_path,
            "[serve]\n"
            'backend = "mlx"\n'
            'model_dir = "models/voice"\n'
            'model_adapter = "adapters/voice.py"\n'
            'reference_audio = "voices/ref.wav"\n'
            'reference_text = "Ref."\n',
        )

        actual = config.load_serve(path)

        assert actual.model_dir == tmp_path / "models/voice"
        assert actual.model_adapter == tmp_path / "adapters/voice.py"
        assert actual.generation["reference_audio"] == tmp_path / "voices/ref.wav"

    def test_unknown_keys_are_warned_about_and_ignored(self, tmp_path, caplog):
        path = self._write(
            tmp_path, '[serve]\nbackend = "mlx"\nmodel_dir = "/m"\ntop-p = 0.9\ntemprature = 1.0\ntop_k = 5\n'
        )
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            actual = config.load_serve(path)
        assert actual == config.ServeConfig(backend="mlx", model_dir=Path("/m"), generation={"top_k": 5})
        assert "unknown keys temprature, top-p" in caplog.text

    def test_only_unknown_keys_warns_once(self, tmp_path, caplog):
        path = self._write(tmp_path, '[serve]\nmodel-dir = "/m"\n')
        with caplog.at_level(logging.WARNING, logger="stackvox.config"):
            assert config.load_serve(path) == config.ServeConfig()
        assert "unknown keys model-dir" in caplog.text
        assert "need a backend" not in caplog.text
