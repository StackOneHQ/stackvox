"""Config loader tests — pure file/env logic, no engine touched."""

from __future__ import annotations

import logging

from stackvox import config
from stackvox.engine import DEFAULT_LANG, DEFAULT_SPEED, DEFAULT_VOICE
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
