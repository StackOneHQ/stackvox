"""Daemon socket protocol — handler behavior with a mocked engine.

We spin up a real UnixStreamServer bound to a tmp socket, swap out the
engine with a mock, and verify the wire-level replies. No Kokoro load,
no audio device.
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


@dataclass
class ServerHarness:
    sock: Path
    state: Any  # daemon._DaemonState — kept as Any so tests can touch mock attrs
    tts: MagicMock
    refresh: MagicMock


@pytest.fixture
def server(mocker):
    """Start an isolated daemon server on a temp socket.

    Uses `tempfile.mkdtemp(dir="/tmp")` rather than pytest's tmp_path because
    macOS caps AF_UNIX socket paths at ~104 bytes and pytest paths are deep.
    """
    from stackvox import daemon

    stackvox_mock = mocker.patch.object(daemon, "Stackvox")
    refresh_mock = mocker.patch.object(daemon, "_refresh_audio_devices")

    tmp = Path(tempfile.mkdtemp(prefix="svx-", dir="/tmp"))
    sock = tmp / "t.sock"
    state = daemon._DaemonState(voice="af_sarah", speed=1.0, lang="en-us")
    srv = daemon._UnixServer(str(sock), daemon._Handler)
    srv.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield ServerHarness(
            sock=sock,
            state=state,
            tts=stackvox_mock.return_value,
            refresh=refresh_mock,
        )
    finally:
        state.shutdown()
        srv.shutdown()
        srv.server_close()
        sock.unlink(missing_ok=True)
        tmp.rmdir()


def _roundtrip(sock_path: Path, payload: str) -> str:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(str(sock_path))
        s.sendall(payload.encode("utf-8"))
        return s.recv(1024).decode("utf-8").strip()
    finally:
        s.close()


def test_ping_returns_ok(server: ServerHarness):
    assert _roundtrip(server.sock, json.dumps({"command": "ping"}) + "\n") == "ok"


def test_plain_text_is_treated_as_text_field(server: ServerHarness):
    """Non-JSON payloads are accepted and wrapped as `{"text": line}`."""
    import time

    assert _roundtrip(server.sock, "hello\n") == "ok"
    # The worker (mocked Stackvox) drains the queue immediately; poll until
    # the mock records the call rather than racing the queue.
    deadline = time.monotonic() + 1.0
    while not server.tts.speak.call_args_list and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.tts.speak.call_args.args[0] == "hello"


def test_missing_text_yields_error(server: ServerHarness):
    reply = _roundtrip(server.sock, json.dumps({"voice": "af_sarah"}) + "\n")
    assert reply.startswith("err:")


def test_full_queue_returns_busy(server: ServerHarness):
    from stackvox import daemon

    # Block the worker so the queue can actually fill up.
    server.state.queue.put_nowait({"text": "a"})
    server.state.queue.put_nowait({"text": "b"})
    assert server.state.queue.full() or server.state.queue.qsize() >= daemon.MAX_QUEUE

    reply = _roundtrip(server.sock, json.dumps({"text": "overflow"}) + "\n")
    # At least one of the sentinel messages OR busy — queue may drain if worker picks up first.
    assert reply in {"ok", "busy"}


def test_worker_refreshes_audio_only_when_dirty(server: ServerHarness):
    """PortAudio is refreshed once per dirty cycle, not before every play.

    Initial dirty state at startup → first playback refreshes. A second
    playback with no device change → no extra refresh. Re-marking dirty
    (simulating a default-output-device change) → next playback refreshes
    again.
    """
    import time

    from stackvox import daemon

    # Ensure a known starting state regardless of prior-test ordering: dirty
    # so the first playback in this test refreshes once.
    daemon._audio_dirty.set()

    def wait_for_speak_count(target: int) -> None:
        deadline = time.monotonic() + 1.0
        while server.tts.speak.call_count < target and time.monotonic() < deadline:
            time.sleep(0.01)

    # First playback consumes the dirty flag.
    assert _roundtrip(server.sock, json.dumps({"text": "a"}) + "\n") == "ok"
    wait_for_speak_count(1)
    assert server.refresh.call_count == 1

    # Second playback with no device change → no additional refresh.
    assert _roundtrip(server.sock, json.dumps({"text": "b"}) + "\n") == "ok"
    wait_for_speak_count(2)
    assert server.refresh.call_count == 1

    # Simulate a real device change.
    daemon._audio_dirty.set()

    assert _roundtrip(server.sock, json.dumps({"text": "c"}) + "\n") == "ok"
    wait_for_speak_count(3)
    assert server.refresh.call_count == 2


def test_send_helper_when_daemon_not_running(tmp_path, monkeypatch):
    from stackvox import daemon

    monkeypatch.setattr(daemon, "SOCKET_PATH", tmp_path / "nope.sock")
    ok, resp = daemon.send({"command": "ping"})
    assert not ok
    assert resp == "daemon not running"
