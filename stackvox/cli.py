"""CLI: `stackvox <subcommand>` or `stackvox "text"` (defaults to speak)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf

from stackvox import daemon
from stackvox.engine import DEFAULT_LANG, DEFAULT_SPEED, DEFAULT_VOICE, Stackvox

SUBCOMMANDS = {"serve", "stop", "status", "say", "speak", "voices"}


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
    p_say.add_argument("--fallback-say", action="store_true", help="Shell out to macOS `say` if daemon unreachable")

    p_serve = sub.add_parser("serve", help="Run the daemon in the foreground")
    _add_voice_args(p_serve)

    sub.add_parser("stop", help="Stop the running daemon")
    sub.add_parser("status", help="Print daemon status")
    sub.add_parser("voices", help="List available voices")

    return parser


def _add_voice_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--lang", default=DEFAULT_LANG)


def _read_text(args: argparse.Namespace) -> str | None:
    if getattr(args, "file", None):
        return args.file.read_text()
    return getattr(args, "text", None)


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


def main() -> int:
    argv = sys.argv[1:]
    # Back-compat: `stackvox "text"` with no subcommand → speak.
    if argv and argv[0] not in SUBCOMMANDS and not argv[0].startswith("-"):
        argv = ["speak", *argv]

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
    }
    return handlers[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
