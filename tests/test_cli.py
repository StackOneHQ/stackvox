"""CLI argument parsing — no engine loaded, no audio played."""

from __future__ import annotations

import io

import pytest

from stackvox import cli


@pytest.fixture(autouse=True)
def _stdin_is_a_tty(mocker):
    """Default tests to a TTY stdin so the new piped-stdin shortcut is off.

    Tests that exercise the piped path opt in by patching isatty to False
    and providing a real stdin StringIO.
    """
    mocker.patch.object(cli.sys.stdin, "isatty", return_value=True)


@pytest.fixture(autouse=True)
def _no_network_update_check(mocker):
    """Block the PyPI update check from making real HTTP requests in tests.

    Tests that care about update-notice behaviour opt in by patching
    `cli.updates.cached_update` / `cli.updates.check_for_update` themselves.
    """
    mocker.patch.object(cli.updates, "check_for_update", return_value=None)
    mocker.patch.object(cli.updates, "cached_update", return_value=None)


def test_bare_text_routes_to_speak(mocker):
    speak = mocker.patch.object(cli, "_cmd_speak", return_value=0)
    mocker.patch.object(cli.sys, "argv", ["stackvox", "hello world"])
    assert cli.main() == 0
    args = speak.call_args.args[0]
    assert args.cmd == "speak"
    assert args.text == "hello world"


def test_speak_subcommand_parses_voice_and_speed(mocker):
    speak = mocker.patch.object(cli, "_cmd_speak", return_value=0)
    mocker.patch.object(cli.sys, "argv", ["stackvox", "speak", "--voice", "bf_emma", "--speed", "1.2", "hi"])
    assert cli.main() == 0
    args = speak.call_args.args[0]
    assert args.voice == "bf_emma"
    assert args.speed == pytest.approx(1.2)
    assert args.text == "hi"


def test_no_subcommand_prints_help(mocker, capsys):
    mocker.patch.object(cli.sys, "argv", ["stackvox"])
    assert cli.main() == 1
    captured = capsys.readouterr()
    assert "usage: stackvox" in captured.out


def test_unknown_subcommand_treated_as_speak_text(mocker):
    """`stackvox foo` is treated as `stackvox speak foo`, not an error."""
    speak = mocker.patch.object(cli, "_cmd_speak", return_value=0)
    mocker.patch.object(cli.sys, "argv", ["stackvox", "hello"])
    cli.main()
    assert speak.call_args.args[0].text == "hello"


def test_piped_stdin_with_no_args_routes_to_speak(mocker):
    """`echo hi | stackvox` should default to speak."""
    speak = mocker.patch.object(cli, "_cmd_speak", return_value=0)
    mocker.patch.object(cli.sys, "argv", ["stackvox"])
    mocker.patch.object(cli.sys.stdin, "isatty", return_value=False)
    mocker.patch.object(cli.sys, "stdin", io.StringIO("hello from stdin\n"))
    assert cli.main() == 0
    assert speak.call_args.args[0].cmd == "speak"


def test_read_text_prefers_file_then_positional_then_stdin(mocker, tmp_path):
    """_read_text precedence: --file > positional > piped stdin."""
    file_arg = tmp_path / "src.txt"
    file_arg.write_text("from-file", encoding="utf-8")

    # Both --file and positional present → file wins.
    args = mocker.MagicMock(file=file_arg, text="from-positional")
    assert cli._read_text(args) == "from-file"

    # No --file, positional present → positional wins.
    args = mocker.MagicMock(file=None, text="from-positional")
    assert cli._read_text(args) == "from-positional"

    # No --file, no positional, stdin piped → stdin wins.
    mocker.patch.object(cli.sys.stdin, "isatty", return_value=False)
    mocker.patch.object(cli.sys, "stdin", io.StringIO("from-stdin"))
    args = mocker.MagicMock(file=None, text=None)
    assert cli._read_text(args) == "from-stdin"


def test_completion_bash_emits_complete_script(mocker, capsys):
    mocker.patch.object(cli.sys, "argv", ["stackvox", "completion", "bash"])
    assert cli.main() == 0
    captured = capsys.readouterr()
    assert "_stackvox_completion()" in captured.out
    assert "complete -F _stackvox_completion stackvox" in captured.out


# --- Subcommand handlers ----------------------------------------------------
#
# Each subcommand handler is exercised directly with a constructed Namespace
# so we don't have to reason about argparse behavior for every test. The
# Stackvox class and the daemon module's surface are mocked out.


@pytest.fixture
def fake_stackvox(mocker):
    return mocker.patch.object(cli, "Stackvox")


def _ns(**kwargs):
    """Build a minimal argparse.Namespace for handler tests."""
    import argparse

    return argparse.Namespace(**kwargs)


def _norm_ns(text=None, file=None, **overrides):
    """Namespace carrying the normalize flags at their library defaults."""
    base = dict(
        text=text,
        file=file,
        markdown=True,
        pronunciations=None,
        dev_terms=True,
        expand_units=True,
        expand_numbers=True,
        pauses=True,
        tables="drop",
        strip_emoji=False,
        terminal_stops=True,
        locale="en-GB",
    )
    base.update(overrides)
    return _ns(**base)


class TestCmdSpeak:
    def test_with_text_calls_engine_speak(self, fake_stackvox):
        rc = cli._cmd_speak(_ns(voice="af_sarah", speed=1.0, lang="en-us", text="hello", out=None))
        assert rc == 0
        fake_stackvox.return_value.speak.assert_called_once_with("hello")

    def test_with_out_writes_wav_instead_of_playing(self, fake_stackvox, mocker, tmp_path):
        import numpy as np

        fake_stackvox.return_value.synthesize.return_value = (
            np.zeros(10, dtype=np.float32),
            24000,
        )
        sf_write = mocker.patch.object(cli.sf, "write")
        out = tmp_path / "out.wav"

        rc = cli._cmd_speak(_ns(voice="af_sarah", speed=1.0, lang="en-us", text="hi", out=out))

        assert rc == 0
        sf_write.assert_called_once()
        fake_stackvox.return_value.speak.assert_not_called()

    def test_blank_input_returns_error(self, fake_stackvox, capsys):
        rc = cli._cmd_speak(_ns(voice="af_sarah", speed=1.0, lang="en-us", text="   ", out=None))
        assert rc == 1
        assert "provide text" in capsys.readouterr().err
        fake_stackvox.return_value.speak.assert_not_called()

    def test_normalize_transforms_text_before_engine(self, fake_stackvox):
        args = _norm_ns(text="# Heading", voice="af_sarah", speed=1.0, lang="en-us", out=None, normalize=True)
        rc = cli._cmd_speak(args)
        assert rc == 0
        fake_stackvox.return_value.speak.assert_called_once_with("Heading.")

    def test_no_normalize_leaves_text_untouched(self, fake_stackvox):
        # `normalize` absent from the namespace mirrors the flag being off.
        rc = cli._cmd_speak(_ns(voice="af_sarah", speed=1.0, lang="en-us", text="# Heading", out=None))
        assert rc == 0
        fake_stackvox.return_value.speak.assert_called_once_with("# Heading")


class TestCmdSay:
    def test_returns_zero_when_daemon_accepts(self, mocker):
        mocker.patch.object(cli.daemon, "say", return_value=(True, "ok"))
        rc = cli._cmd_say(_ns(voice="af_sarah", speed=1.0, lang="en-us", text="hi", fallback_say=False))
        assert rc == 0

    def test_returns_two_when_daemon_unreachable_and_no_fallback(self, mocker, capsys):
        mocker.patch.object(cli.daemon, "say", return_value=(False, "daemon not running"))
        rc = cli._cmd_say(_ns(voice="af_sarah", speed=1.0, lang="en-us", text="hi", fallback_say=False))
        assert rc == 2
        assert "daemon not running" in capsys.readouterr().err

    def test_fallback_say_shells_out_on_macos(self, mocker):
        mocker.patch.object(cli.daemon, "say", return_value=(False, "daemon not running"))
        # Must use which because cli imports inside the function body.
        mocker.patch("shutil.which", return_value="/usr/bin/say")
        run = mocker.patch("subprocess.run")
        rc = cli._cmd_say(_ns(voice="af_sarah", speed=1.0, lang="en-us", text="hi", fallback_say=True))
        assert rc == 0
        run.assert_called_once()
        assert run.call_args.args[0][0] == "say"

    def test_fallback_say_without_say_binary_returns_two(self, mocker, capsys):
        mocker.patch.object(cli.daemon, "say", return_value=(False, "daemon not running"))
        mocker.patch("shutil.which", return_value=None)
        rc = cli._cmd_say(_ns(voice="af_sarah", speed=1.0, lang="en-us", text="hi", fallback_say=True))
        assert rc == 2

    def test_blank_input_returns_one(self, capsys):
        rc = cli._cmd_say(_ns(voice="af_sarah", speed=1.0, lang="en-us", text="", fallback_say=False))
        assert rc == 1

    def test_normalize_transforms_text_before_daemon(self, mocker):
        say = mocker.patch.object(cli.daemon, "say", return_value=(True, "ok"))
        args = _norm_ns(
            text="# Heading", voice="af_sarah", speed=1.0, lang="en-us", fallback_say=False, normalize=True
        )
        rc = cli._cmd_say(args)
        assert rc == 0
        assert say.call_args.args[0] == "Heading."


class TestCmdServe:
    def test_propagates_serve_args(self, mocker):
        serve = mocker.patch.object(cli.daemon, "serve")
        rc = cli._cmd_serve(_ns(voice="bf_emma", speed=1.1, lang="en-gb"))
        assert rc == 0
        serve.assert_called_once_with(voice="bf_emma", speed=1.1, lang="en-gb")

    def test_returns_one_when_daemon_already_running(self, mocker, capsys):
        mocker.patch.object(cli.daemon, "serve", side_effect=RuntimeError("daemon already running"))
        rc = cli._cmd_serve(_ns(voice="af_sarah", speed=1.0, lang="en-us"))
        assert rc == 1
        assert "already running" in capsys.readouterr().err


class TestCmdStop:
    def test_returns_zero_when_daemon_already_stopped(self, mocker, capsys):
        mocker.patch.object(cli.daemon, "is_running", return_value=False)
        stop = mocker.patch.object(cli.daemon, "stop")
        rc = cli._cmd_stop(_ns())
        assert rc == 0
        stop.assert_not_called()
        assert "not running" in capsys.readouterr().err

    def test_calls_daemon_stop_when_running(self, mocker):
        mocker.patch.object(cli.daemon, "is_running", return_value=True)
        mocker.patch.object(cli.daemon, "stop", return_value=(True, "ok"))
        assert cli._cmd_stop(_ns()) == 0


class TestCmdStatus:
    def test_running_prints_pid_and_returns_zero(self, mocker, capsys):
        mocker.patch.object(cli.daemon, "is_running", return_value=True)
        # PID_PATH and SOCKET_PATH are read at module level — mock as needed.
        pid_path = mocker.MagicMock()
        pid_path.read_text.return_value = "12345\n"
        mocker.patch.object(cli.daemon, "PID_PATH", pid_path)
        mocker.patch.object(cli.daemon, "SOCKET_PATH", "/tmp/x.sock")
        # status now queries the running daemon's version; stub it so the test
        # doesn't reach a real socket.
        mocker.patch.object(cli.daemon, "version", return_value=(False, "n/a"))

        rc = cli._cmd_status(_ns())

        assert rc == 0
        out = capsys.readouterr().out
        assert "running" in out
        assert "12345" in out

    def test_running_flags_daemon_version_skew(self, mocker, capsys):
        mocker.patch.object(cli.daemon, "is_running", return_value=True)
        pid_path = mocker.MagicMock()
        pid_path.read_text.return_value = "1\n"
        mocker.patch.object(cli.daemon, "PID_PATH", pid_path)
        mocker.patch.object(cli.daemon, "SOCKET_PATH", "/tmp/x.sock")
        mocker.patch.object(cli.daemon, "version", return_value=(True, "0.5.0"))
        mocker.patch.object(cli.updates, "_current_version", return_value="0.8.0")
        mocker.patch.object(cli.updates, "check_for_update", return_value=None)

        rc = cli._cmd_status(_ns())

        assert rc == 0
        out = capsys.readouterr().out
        assert "0.5.0" in out and "0.8.0" in out  # the skew is surfaced
        assert "restart" in out.lower()

    def test_stopped_returns_one(self, mocker, capsys):
        mocker.patch.object(cli.daemon, "is_running", return_value=False)
        rc = cli._cmd_status(_ns())
        assert rc == 1
        assert "stopped" in capsys.readouterr().out

    def test_status_prints_update_notice_when_outdated(self, mocker, capsys):
        mocker.patch.object(cli.daemon, "is_running", return_value=False)
        mocker.patch.object(
            cli.updates,
            "check_for_update",
            return_value=cli.updates.UpdateInfo(current="0.3.1", latest="0.4.0"),
        )
        cli._cmd_status(_ns())
        out = capsys.readouterr().out
        assert "0.3.1" in out
        assert "0.4.0" in out
        assert "pipx upgrade" in out

    def test_status_prints_plain_version_when_up_to_date(self, mocker, capsys):
        mocker.patch.object(cli.daemon, "is_running", return_value=False)
        mocker.patch.object(cli.updates, "check_for_update", return_value=None)
        mocker.patch.object(cli.updates, "_current_version", return_value="0.4.0")
        cli._cmd_status(_ns())
        out = capsys.readouterr().out
        assert "version: 0.4.0" in out
        assert "pipx upgrade" not in out


class TestUpdateNotice:
    def test_silent_by_default(self, mocker, monkeypatch, capsys):
        monkeypatch.delenv("STACKVOX_UPDATE_NOTICE", raising=False)
        mocker.patch.object(
            cli.updates,
            "cached_update",
            return_value=cli.updates.UpdateInfo(current="0.3.1", latest="0.4.0"),
        )
        cli._maybe_print_update_notice()
        assert capsys.readouterr().err == ""

    def test_prints_to_stderr_when_opted_in(self, mocker, monkeypatch, capsys):
        monkeypatch.setenv("STACKVOX_UPDATE_NOTICE", "1")
        mocker.patch.object(
            cli.updates,
            "cached_update",
            return_value=cli.updates.UpdateInfo(current="0.3.1", latest="0.4.0"),
        )
        cli._maybe_print_update_notice()
        err = capsys.readouterr().err
        assert "0.3.1" in err
        assert "0.4.0" in err

    def test_silent_when_no_update_even_with_opt_in(self, mocker, monkeypatch, capsys):
        monkeypatch.setenv("STACKVOX_UPDATE_NOTICE", "1")
        mocker.patch.object(cli.updates, "cached_update", return_value=None)
        cli._maybe_print_update_notice()
        assert capsys.readouterr().err == ""


class TestCmdVoices:
    def test_prints_one_voice_per_line(self, fake_stackvox, capsys):
        fake_stackvox.return_value.voices.return_value = ["af_sarah", "bf_emma"]
        rc = cli._cmd_voices(_ns())
        assert rc == 0
        assert capsys.readouterr().out.split() == ["af_sarah", "bf_emma"]


class TestCmdWelcome:
    def test_calls_speak_sequence_with_welcome_lines(self, fake_stackvox):
        cli._cmd_welcome(_ns())
        fake_stackvox.return_value.speak_sequence.assert_called_once()
        lines = fake_stackvox.return_value.speak_sequence.call_args.args[0]
        # All WELCOME_LINES rows present.
        assert len(lines) == len(cli.WELCOME_LINES)
        # Every entry has text/voice/lang.
        for entry in lines:
            assert {"text", "voice", "lang"} <= entry.keys()


class TestCmdCompletion:
    def test_unsupported_shell_returns_one(self, capsys):
        # argparse choices=["bash"] means we'd never reach here normally;
        # exercise the defensive branch directly.
        rc = cli._cmd_completion(_ns(shell="fish"))
        assert rc == 1
        assert "unsupported shell" in capsys.readouterr().err


class TestCmdInstallHelper:
    def test_copies_helper_to_prefix_with_exec_bit(self, tmp_path):
        prefix = tmp_path / "bin"
        rc = cli._cmd_install_helper(_ns(prefix=prefix))

        assert rc == 0
        dest = prefix / "stackvox-say"
        assert dest.exists()
        # First line should be the bash shebang from the bundled script.
        assert dest.read_text(encoding="utf-8").startswith("#!/bin/bash")
        # Owner-execute bit is set.
        import stat as st

        assert dest.stat().st_mode & st.S_IXUSR

    def test_creates_prefix_dir_if_missing(self, tmp_path):
        prefix = tmp_path / "deep" / "nested" / "bin"
        assert not prefix.exists()
        rc = cli._cmd_install_helper(_ns(prefix=prefix))
        assert rc == 0
        assert (prefix / "stackvox-say").is_file()

    def test_warns_when_prefix_not_on_path(self, tmp_path, mocker, capsys):
        prefix = tmp_path / "out-of-path"
        # Force PATH to a value that definitely doesn't include `prefix`.
        mocker.patch.dict("os.environ", {"PATH": "/usr/bin:/bin"}, clear=False)
        cli._cmd_install_helper(_ns(prefix=prefix))
        err = capsys.readouterr().err
        assert "not on your PATH" in err

    def test_no_warning_when_prefix_on_path(self, tmp_path, mocker, capsys):
        prefix = tmp_path / "on-path"
        prefix.mkdir(parents=True)
        # Put `prefix` on PATH so the warning shouldn't fire.
        mocker.patch.dict("os.environ", {"PATH": f"{prefix}:/usr/bin"}, clear=False)
        cli._cmd_install_helper(_ns(prefix=prefix))
        assert "not on your PATH" not in capsys.readouterr().err


class TestCmdNormalize:
    def test_markdown_reduces_to_prose_with_terminal_stops(self, capsys):
        rc = cli._cmd_normalize(_norm_ns(text="# Heading\n\nBody"))
        assert rc == 0
        assert capsys.readouterr().out.splitlines() == ["Heading.", "Body."]

    def test_number_and_decimal_expansion(self, capsys):
        rc = cli._cmd_normalize(_norm_ns(text="It was 1,198.9 total"))
        assert rc == 0
        assert capsys.readouterr().out.strip() == "It was 1198 point 9 total."

    def test_pronunciations_file_is_applied(self, capsys, tmp_path):
        pron = tmp_path / "dict.json"
        pron.write_text('{"Redis": "ree-diss"}', encoding="utf-8")
        rc = cli._cmd_normalize(_norm_ns(text="Redis is fast", pronunciations=pron))
        assert rc == 0
        assert capsys.readouterr().out.strip() == "ree-diss is fast."

    def test_blank_input_returns_one(self, capsys):
        rc = cli._cmd_normalize(_norm_ns(text="   "))
        assert rc == 1
        assert "provide text" in capsys.readouterr().err

    def test_bad_pronunciations_file_errors_cleanly(self, capsys, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('["not", "a", "map"]', encoding="utf-8")
        rc = cli._cmd_normalize(_norm_ns(text="hi", pronunciations=bad))
        assert rc == 1
        assert "string -> string" in capsys.readouterr().err


class TestLoadPronunciations:
    def test_none_path_returns_none(self):
        assert cli._load_pronunciations(None) is None

    def test_valid_json_returns_mapping(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text('{"CLI": "see ell eye"}', encoding="utf-8")
        assert cli._load_pronunciations(path) == {"CLI": "see ell eye"}

    def test_non_object_json_raises(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text('["not", "a", "map"]', encoding="utf-8")
        with pytest.raises(ValueError, match="string -> string"):
            cli._load_pronunciations(path)

    def test_non_string_values_raise(self, tmp_path):
        path = tmp_path / "d.json"
        path.write_text('{"n": 3}', encoding="utf-8")
        with pytest.raises(ValueError, match="string -> string"):
            cli._load_pronunciations(path)


class TestCmdPaths:
    def test_prints_the_three_paths(self, capsys):
        rc = cli._cmd_paths(_ns())
        assert rc == 0
        out = capsys.readouterr().out
        assert "cache_dir:" in out
        assert "socket_path:" in out
        assert "pid_path:" in out


class TestCmdConfig:
    def test_prints_resolved_defaults_and_path(self, mocker, capsys, tmp_path):
        cfg = tmp_path / "config.toml"  # absent → "not present" note
        mocker.patch.object(cli.config, "config_path", return_value=cfg)
        mocker.patch.object(
            cli.config,
            "load_defaults",
            return_value=cli.config.Defaults(voice="bf_emma", speed=1.1, lang="en-gb"),
        )
        rc = cli._cmd_config(_ns())
        assert rc == 0
        out = capsys.readouterr().out
        assert "not present" in out
        assert "bf_emma" in out
        assert "1.1" in out
        assert "en-gb" in out


class TestNormalizeParsing:
    def test_normalize_subcommand_routes(self, mocker):
        norm = mocker.patch.object(cli, "_cmd_normalize", return_value=0)
        mocker.patch.object(cli.sys, "argv", ["stackvox", "normalize", "--tables", "csv", "hi"])
        assert cli.main() == 0
        args = norm.call_args.args[0]
        assert args.cmd == "normalize"
        assert args.tables == "csv"
        assert args.text == "hi"

    def test_speak_normalize_switch_parses(self, mocker):
        speak = mocker.patch.object(cli, "_cmd_speak", return_value=0)
        mocker.patch.object(
            cli.sys, "argv", ["stackvox", "speak", "--normalize", "--strip-emoji", "--no-pauses", "hi"]
        )
        assert cli.main() == 0
        args = speak.call_args.args[0]
        assert args.normalize is True
        assert args.strip_emoji is True
        assert args.pauses is False

    def test_paths_and_config_route(self, mocker):
        paths_handler = mocker.patch.object(cli, "_cmd_paths", return_value=0)
        mocker.patch.object(cli.sys, "argv", ["stackvox", "paths"])
        assert cli.main() == 0
        assert paths_handler.called


class TestCmdCancel:
    def test_running_calls_daemon_cancel(self, mocker):
        mocker.patch.object(cli.daemon, "is_running", return_value=True)
        cancel = mocker.patch.object(cli.daemon, "cancel", return_value=(True, "ok"))
        assert cli._cmd_cancel(_ns()) == 0
        cancel.assert_called_once()

    def test_no_daemon_is_a_noop(self, mocker):
        mocker.patch.object(cli.daemon, "is_running", return_value=False)
        cancel = mocker.patch.object(cli.daemon, "cancel")
        assert cli._cmd_cancel(_ns()) == 0
        cancel.assert_not_called()
