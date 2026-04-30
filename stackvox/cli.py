"""CLI: `stackvox <subcommand>` or `stackvox "text"` (defaults to speak)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import soundfile as sf

from stackvox import daemon
from stackvox.engine import DEFAULT_LANG, DEFAULT_SPEED, DEFAULT_VOICE, Stackvox


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[stackvox] %(message)s",
        stream=sys.stderr,
    )


SUBCOMMANDS = {
    "serve",
    "stop",
    "status",
    "say",
    "speak",
    "voices",
    "welcome",
    "completion",
    "install-helper",
}

DEFAULT_HELPER_PREFIX = Path.home() / ".local" / "bin"

_BASH_COMPLETION = r"""# stackvox bash completion. Install with one of:
#   eval "$(stackvox completion bash)"          # current shell
#   stackvox completion bash > ~/.stackvox-completion.bash && \
#     echo 'source ~/.stackvox-completion.bash' >> ~/.bashrc
_stackvox_completion() {
    local cur prev subcommand
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    subcommand="${COMP_WORDS[1]:-}"

    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "speak say serve stop status voices welcome completion install-helper" -- "$cur") )
        return 0
    fi

    case "$prev" in
        --file|--out)
            COMPREPLY=( $(compgen -f -- "$cur") )
            return 0
            ;;
        --prefix)
            COMPREPLY=( $(compgen -d -- "$cur") )
            return 0
            ;;
        --speed)
            COMPREPLY=( $(compgen -W "0.8 0.9 1.0 1.1 1.2 1.5" -- "$cur") )
            return 0
            ;;
        --lang)
            COMPREPLY=( $(compgen -W "en-us en-gb fr-fr it hi pt-br es ja zh" -- "$cur") )
            return 0
            ;;
    esac

    case "$subcommand" in
        speak)
            COMPREPLY=( $(compgen -W "--voice --speed --lang --file --out --help" -- "$cur") )
            ;;
        say)
            COMPREPLY=( $(compgen -W "--voice --speed --lang --file --fallback-say --help" -- "$cur") )
            ;;
        serve)
            COMPREPLY=( $(compgen -W "--voice --speed --lang --help" -- "$cur") )
            ;;
        completion)
            COMPREPLY=( $(compgen -W "bash" -- "$cur") )
            ;;
        install-helper)
            COMPREPLY=( $(compgen -W "--prefix --help" -- "$cur") )
            ;;
        *)
            COMPREPLY=( $(compgen -W "--help" -- "$cur") )
            ;;
    esac
}
complete -F _stackvox_completion stackvox
"""

WELCOME_LINES = [
    ("bf_emma", "en-gb", "Welcome to stackvox."),
    ("af_heart", "en-us", "Welcome to stackvox."),
    ("ff_siwis", "fr-fr", "Bienvenue sur stackvox."),
    ("hf_alpha", "hi", "Stackvox mein aapka swagat hai."),
    ("if_sara", "it", "Benvenuti a stackvox."),
    ("pf_dora", "pt-br", "Bem-vindo ao stackvox."),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stackvox", description="Kokoro-82M TTS")
    sub = parser.add_subparsers(dest="cmd")

    p_speak = sub.add_parser("speak", help="Synthesize and play in-process (loads model each run)")
    _add_voice_args(p_speak)
    p_speak.add_argument("text", nargs="?")
    p_speak.add_argument("--file", type=Path)
    p_speak.add_argument("--out", type=Path, help="Write wav instead of playing")

    p_say = sub.add_parser("say", help="Send text to daemon (fast; fails if daemon not running)")
    _add_voice_args(p_say)
    p_say.add_argument("text", nargs="?")
    p_say.add_argument("--file", type=Path)
    p_say.add_argument(
        "--fallback-say", action="store_true", help="Shell out to macOS `say` if daemon unreachable"
    )

    p_serve = sub.add_parser("serve", help="Run the daemon in the foreground")
    _add_voice_args(p_serve)

    sub.add_parser("stop", help="Stop the running daemon")
    sub.add_parser("status", help="Print daemon status")
    sub.add_parser("voices", help="List available voices")
    sub.add_parser("welcome", help="Play a multilingual welcome message")

    p_completion = sub.add_parser("completion", help="Print a shell completion script")
    p_completion.add_argument("shell", choices=["bash"], help="Shell to generate completion for")

    p_install_helper = sub.add_parser(
        "install-helper",
        help="Copy the stackvox-say bash helper onto PATH (default: ~/.local/bin)",
    )
    p_install_helper.add_argument(
        "--prefix",
        type=Path,
        default=DEFAULT_HELPER_PREFIX,
        help=f"Install directory (default: {DEFAULT_HELPER_PREFIX})",
    )

    return parser


def _add_voice_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--lang", default=DEFAULT_LANG)


def _read_text(args: argparse.Namespace) -> str | None:
    """Resolve the text to speak from --file, the positional, or piped stdin.

    Precedence: --file > positional text > stdin (when piped, not a TTY).
    """
    file: Path | None = getattr(args, "file", None)
    if file is not None:
        return file.read_text(encoding="utf-8")
    text: str | None = getattr(args, "text", None)
    if text is not None:
        return text
    if not sys.stdin.isatty():
        piped = sys.stdin.read()
        return piped if piped.strip() else None
    return None


def _cmd_speak(args: argparse.Namespace) -> int:
    text = _read_text(args)
    if not text or not text.strip():
        print("Error: provide text or --file", file=sys.stderr)
        return 1
    tts = Stackvox(voice=args.voice, speed=args.speed, lang=args.lang)
    if args.out:
        samples, sr = tts.synthesize(text)
        sf.write(args.out, samples, sr)
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        tts.speak(text)
    return 0


def _cmd_say(args: argparse.Namespace) -> int:
    text = _read_text(args)
    if not text or not text.strip():
        print("Error: provide text or --file", file=sys.stderr)
        return 1
    ok, resp = daemon.say(text, voice=args.voice, speed=args.speed, lang=args.lang)
    if ok:
        return 0
    if args.fallback_say:
        import shutil
        import subprocess

        if shutil.which("say"):
            subprocess.run(["say", text], check=False)
            return 0
    print(f"[stackvox] {resp}", file=sys.stderr)
    return 2


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        daemon.serve(voice=args.voice, speed=args.speed, lang=args.lang)
    except RuntimeError as exc:
        print(f"[stackvox] {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_stop(_: argparse.Namespace) -> int:
    if not daemon.is_running():
        print("[stackvox] daemon not running", file=sys.stderr)
        return 0
    ok, resp = daemon.stop()
    print(resp, file=sys.stderr)
    return 0 if ok else 1


def _cmd_status(_: argparse.Namespace) -> int:
    if daemon.is_running():
        print(f"running (pid {daemon.PID_PATH.read_text().strip()}) on {daemon.SOCKET_PATH}")
        return 0
    print("stopped")
    return 1


def _cmd_voices(args: argparse.Namespace) -> int:
    tts = Stackvox()
    for name in tts.voices():
        print(name)
    return 0


def _cmd_welcome(_: argparse.Namespace) -> int:
    tts = Stackvox()
    tts.speak_sequence([{"text": text, "voice": voice, "lang": lang} for voice, lang, text in WELCOME_LINES])
    return 0


def _cmd_completion(args: argparse.Namespace) -> int:
    if args.shell == "bash":
        print(_BASH_COMPLETION)
        return 0
    # argparse `choices=` should prevent reaching here; defensive only.
    print(f"unsupported shell: {args.shell}", file=sys.stderr)
    return 1


def _cmd_install_helper(args: argparse.Namespace) -> int:
    """Copy the bundled `stackvox-say` bash helper onto PATH.

    The helper is shipped as package data (`stackvox/data/stackvox-say`) rather
    than installed automatically by setuptools — `script-files` is discouraged
    and doesn't round-trip through modern build backends. This subcommand is
    the explicit one-time install step.
    """
    import os
    import shutil
    import stat
    from importlib.resources import as_file, files

    prefix = Path(args.prefix).expanduser()
    prefix.mkdir(parents=True, exist_ok=True)
    dest = prefix / "stackvox-say"

    src = files("stackvox").joinpath("data/stackvox-say")
    with as_file(src) as src_path:
        shutil.copy2(src_path, dest)

    # Ensure executable bits regardless of how the wheel preserved them.
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Installed stackvox-say to {dest}", file=sys.stderr)

    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    if str(prefix) not in path_dirs:
        print(
            f"Note: {prefix} is not on your PATH. Either add it to PATH or invoke "
            f"the helper via its full path: {dest}",
            file=sys.stderr,
        )
    return 0


def main() -> int:
    _configure_logging()
    argv = sys.argv[1:]
    # Back-compat: `stackvox "text"` with no subcommand → speak.
    # Same shortcut for piped stdin: `echo hi | stackvox` → speak.
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv = ["speak", *argv]
    elif not argv and not sys.stdin.isatty():
        argv = ["speak"]

    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        return 1

    handlers = {
        "speak": _cmd_speak,
        "say": _cmd_say,
        "serve": _cmd_serve,
        "stop": _cmd_stop,
        "status": _cmd_status,
        "voices": _cmd_voices,
        "welcome": _cmd_welcome,
        "completion": _cmd_completion,
        "install-helper": _cmd_install_helper,
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
