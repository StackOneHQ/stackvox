"""Long-running daemon that preloads Kokoro and plays speech on request.

Avoids the ~1-2s model load on every invocation. Listens on a unix socket.
Protocol: one JSON object per connection (line-terminated), reply is a status line.

Request shapes:
    {"text": "...", "voice": "af_sarah", "speed": 1.0, "lang": "en-us"}
    {"command": "stop"}
    {"command": "ping"}

Replies: "ok", "busy", "err: <msg>".
"""

from __future__ import annotations

import json
import os
import queue
import signal
import socket
import socketserver
import sys
import threading
from pathlib import Path
from typing import Optional

from stackvox.engine import DEFAULT_LANG, DEFAULT_SPEED, DEFAULT_VOICE, Stackvox, _cache_dir

SOCKET_PATH = _cache_dir() / "daemon.sock"
PID_PATH = _cache_dir() / "daemon.pid"
MAX_QUEUE = 2


class _DaemonState:
    def __init__(self, voice: str, speed: float, lang: str) -> None:
        self.tts = Stackvox(voice=voice, speed=speed, lang=lang)
        self.queue: queue.Queue[dict] = queue.Queue(maxsize=MAX_QUEUE)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                req = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.tts.speak(
                    req["text"],
                    voice=req.get("voice"),
                    speed=req.get("speed"),
                    lang=req.get("lang"),
                )
            except Exception as exc:
                print(f"[stackvox] playback error: {exc}", file=sys.stderr)

    def submit(self, req: dict) -> bool:
        try:
            self.queue.put_nowait(req)
            return True
        except queue.Full:
            return False

    def shutdown(self) -> None:
        self.stop_event.set()


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline().decode("utf-8", errors="replace").strip()
        if not line:
            return

        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            req = {"text": line}

        state: _DaemonState = self.server.state  # type: ignore[attr-defined]
        command = req.get("command")

        if command == "ping":
            self.wfile.write(b"ok\n")
            return

        if command == "stop":
            self.wfile.write(b"ok\n")
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return

        text = req.get("text")
        if not text:
            self.wfile.write(b"err: missing text\n")
            return

        if state.submit(req):
            self.wfile.write(b"ok\n")
        else:
            self.wfile.write(b"busy\n")


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    allow_reuse_address = True
    daemon_threads = True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_running() -> bool:
    if not PID_PATH.exists():
        return False
    try:
        pid = int(PID_PATH.read_text().strip())
    except ValueError:
        return False
    return _pid_alive(pid)


def serve(voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED, lang: str = DEFAULT_LANG) -> None:
    if is_running():
        raise RuntimeError(f"daemon already running (pid {PID_PATH.read_text().strip()})")

    SOCKET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()

    state = _DaemonState(voice=voice, speed=speed, lang=lang)
    server = _UnixServer(str(SOCKET_PATH), _Handler)
    server.state = state  # type: ignore[attr-defined]

    PID_PATH.write_text(str(os.getpid()))

    def handle_signal(signum: int, frame) -> None:  # type: ignore[no-untyped-def]
        state.shutdown()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    print(f"[stackvox] daemon listening on {SOCKET_PATH} (pid {os.getpid()})", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        SOCKET_PATH.unlink(missing_ok=True)
        PID_PATH.unlink(missing_ok=True)


def send(req: dict, timeout: float = 1.0) -> tuple[bool, str]:
    """Send a request to the daemon. Returns (ok, response)."""
    if not SOCKET_PATH.exists():
        return False, "daemon not running"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(SOCKET_PATH))
        sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
        resp = sock.recv(1024).decode("utf-8", errors="replace").strip()
        return resp == "ok", resp
    except (ConnectionRefusedError, FileNotFoundError, socket.timeout, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        sock.close()


def say(text: str, voice: Optional[str] = None, speed: Optional[float] = None, lang: Optional[str] = None) -> tuple[bool, str]:
    req: dict = {"text": text}
    if voice is not None:
        req["voice"] = voice
    if speed is not None:
        req["speed"] = speed
    if lang is not None:
        req["lang"] = lang
    return send(req)


def stop() -> tuple[bool, str]:
    return send({"command": "stop"})


def ping() -> tuple[bool, str]:
    return send({"command": "ping"}, timeout=0.5)
