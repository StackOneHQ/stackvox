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
