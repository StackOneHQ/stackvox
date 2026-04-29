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

import ctypes
import ctypes.util
import json
import logging
import os
import queue
import signal
import socket
import socketserver
import sys
import threading

import sounddevice as sd

from stackvox.engine import DEFAULT_LANG, DEFAULT_SPEED, DEFAULT_VOICE, Stackvox
from stackvox.paths import pid_path, socket_path

logger = logging.getLogger(__name__)

SOCKET_PATH = socket_path()
PID_PATH = pid_path()
MAX_QUEUE = 2
WORKER_POLL_SECONDS = 0.5
CLIENT_TIMEOUT_SECONDS = 1.0
PING_TIMEOUT_SECONDS = 0.5
RECV_BYTES = 1024


# Set when PortAudio's cached default-output device is suspected stale and
# needs Pa_Terminate / Pa_Initialize before the next playback. Initial state
# is True so the first playback always refreshes; the macOS device watcher
# re-sets it on real device changes; the worker also re-sets on playback
# failure as a belt-and-suspenders retry path.
_audio_dirty = threading.Event()
_audio_dirty.set()

# Holds CoreAudio callback references so they aren't garbage collected.
_ca_refs: list = []


def _refresh_audio_devices() -> None:
    """Reset PortAudio so the next play picks up the current system default.

    PortAudio caches the default output device at init time; without this the
    daemon keeps playing to whatever was default when it started (e.g. the
    built-in speakers after the user swapped to Bluetooth). Terminating and
    re-initialising is the only portable way to refresh that cache. Costs
    ~10-50ms per call.
    """
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        logger.exception("failed to refresh audio devices")


def _start_device_watcher() -> None:
    """macOS only: mark `_audio_dirty` when the default output device changes.

    Avoids reinitialising PortAudio on every playback (the simpler approach,
    which adds 10-50ms of latency before each speech). macOS notifies the
    property listener on more than just real device changes — playback start,
    volume changes, and other side effects all fire it — so we compare the
    current default-output device ID against the last seen one and only mark
    dirty on actual changes.

    No-ops on non-macOS; the dirty flag stays at its initial state (set), so
    the first playback refreshes once and subsequent playbacks reuse the
    PortAudio context. Device changes on those platforms are handled by the
    worker's failure-retry path.
    """
    if sys.platform != "darwin":
        return
    try:
        ca = ctypes.CDLL(ctypes.util.find_library("CoreAudio") or "")
        cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation") or "")
    except Exception:
        logger.debug("CoreAudio unavailable; device watcher disabled")
        return

    class _PropAddr(ctypes.Structure):
        _fields_ = [
            ("mSelector", ctypes.c_uint32),
            ("mScope", ctypes.c_uint32),
            ("mElement", ctypes.c_uint32),
        ]

    _ListenerProc = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.c_uint32,  # inObjectID
        ctypes.c_uint32,  # inNumberAddresses
        ctypes.POINTER(_PropAddr),
        ctypes.c_void_p,
    )

    prop = _PropAddr(
        0x644F7574,  # kAudioHardwarePropertyDefaultOutputDevice  'dOut'
        0x676C6F62,  # kAudioObjectPropertyScopeGlobal             'glob'
        0,  # kAudioObjectPropertyElementMain
    )

    def _read_default_device() -> int:
        device = ctypes.c_uint32(0)
        size = ctypes.c_uint32(ctypes.sizeof(device))
        status = ca.AudioObjectGetPropertyData(
            1,  # kAudioObjectSystemObject
            ctypes.byref(prop),
            0,
            None,
            ctypes.byref(size),
            ctypes.byref(device),
        )
        return device.value if status == 0 else 0

    last_device = [_read_default_device()]

    def _on_device_change(obj_id: int, n: int, addrs, data) -> int:
        try:
            current = _read_default_device()
            if current and current != last_device[0]:
                last_device[0] = current
                _audio_dirty.set()
        except Exception:
            logger.debug("device-change callback error", exc_info=True)
        return 0

    cb = _ListenerProc(_on_device_change)
    _ca_refs.append(cb)  # prevent GC
    ca.AudioObjectAddPropertyListener(1, ctypes.byref(prop), cb, None)

    threading.Thread(
        target=cf.CFRunLoopRun,
        daemon=True,
        name="audio-device-watcher",
    ).start()
    logger.debug("audio device watcher started")


class _DaemonState:
    def __init__(self, voice: str, speed: float, lang: str) -> None:
        self.tts = Stackvox(voice=voice, speed=speed, lang=lang)
        self.queue: queue.Queue[dict] = queue.Queue(maxsize=MAX_QUEUE)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()
        _start_device_watcher()

    def _worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                req = self.queue.get(timeout=WORKER_POLL_SECONDS)
            except queue.Empty:
                continue
            if _audio_dirty.is_set():
                _refresh_audio_devices()
                _audio_dirty.clear()
            try:
                self.tts.speak(
                    req["text"],
                    voice=req.get("voice"),
                    speed=req.get("speed"),
                    lang=req.get("lang"),
                )
            except Exception:
                logger.exception("playback error")
                # Failed playback might be a stale audio context; mark dirty
                # so the next request refreshes before trying again.
                _audio_dirty.set()

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

    def handle_signal(signum: int, frame: object) -> None:
        state.shutdown()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    logger.info("daemon listening on %s (pid %d)", SOCKET_PATH, os.getpid())
    try:
        server.serve_forever()
    finally:
        SOCKET_PATH.unlink(missing_ok=True)
        PID_PATH.unlink(missing_ok=True)


def send(req: dict, timeout: float = CLIENT_TIMEOUT_SECONDS) -> tuple[bool, str]:
    """Send a request to the daemon. Returns (ok, response)."""
    if not SOCKET_PATH.exists():
        return False, "daemon not running"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(SOCKET_PATH))
        sock.sendall((json.dumps(req) + "\n").encode("utf-8"))
        resp = sock.recv(RECV_BYTES).decode("utf-8", errors="replace").strip()
        return resp == "ok", resp
    except (TimeoutError, ConnectionRefusedError, FileNotFoundError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        sock.close()


def say(
    text: str, voice: str | None = None, speed: float | None = None, lang: str | None = None
) -> tuple[bool, str]:
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
    return send({"command": "ping"}, timeout=PING_TIMEOUT_SECONDS)
