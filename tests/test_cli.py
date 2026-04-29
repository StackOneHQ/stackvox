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

        rc = cli._cmd_status(_ns())

        assert rc == 0
        out = capsys.readouterr().out
        assert "running" in out
        assert "12345" in out

    def test_stopped_returns_one(self, mocker, capsys):
        mocker.patch.object(cli.daemon, "is_running", return_value=False)
        rc = cli._cmd_status(_ns())
        assert rc == 1
        assert "stopped" in capsys.readouterr().out


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
