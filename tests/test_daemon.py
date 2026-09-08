"""Daemon tests for non-protocol surface — pid/socket helpers, send/say/stop/ping
client, _refresh_audio_devices, and the cross-platform watcher entry point.
The protocol/handler tests live in test_daemon_protocol.py.
"""

from __future__ import annotations

import os

import pytest

from stackvox import daemon


class TestPidAlive:
    def test_self_pid_is_alive(self):
        assert daemon._pid_alive(os.getpid()) is True

    def test_unused_pid_is_not_alive(self):
        # PID 0 is reserved on macOS/Linux and never owned by a real process.
        # `os.kill(0, 0)` is special-cased (signals the whole process group),
        # so use a high PID we can be confident is unused.
        assert daemon._pid_alive(2**31 - 1) is False


class TestIsRunning:
    def test_returns_false_when_pid_file_missing(self, mocker, tmp_path):
        mocker.patch.object(daemon, "PID_PATH", tmp_path / "missing.pid")
        assert daemon.is_running() is False

    def test_returns_false_when_pid_file_unreadable(self, mocker, tmp_path):
        pid = tmp_path / "garbage.pid"
        pid.write_text("not-a-number")
        mocker.patch.object(daemon, "PID_PATH", pid)
        assert daemon.is_running() is False

    def test_returns_true_when_pid_alive(self, mocker, tmp_path):
        pid = tmp_path / "live.pid"
        pid.write_text(str(os.getpid()))
        mocker.patch.object(daemon, "PID_PATH", pid)
        assert daemon.is_running() is True

    def test_returns_false_when_pid_dead(self, mocker, tmp_path):
        pid = tmp_path / "dead.pid"
        pid.write_text(str(2**31 - 1))
        mocker.patch.object(daemon, "PID_PATH", pid)
        assert daemon.is_running() is False


class TestSendHelpers:
    """The thin send/say/stop/ping wrappers around the unix-socket protocol."""

    def test_send_returns_failure_when_socket_missing(self, mocker, tmp_path):
        mocker.patch.object(daemon, "SOCKET_PATH", tmp_path / "nope.sock")
        ok, resp = daemon.send({"command": "ping"})
        assert ok is False
        assert resp == "daemon not running"

    def test_say_includes_only_supplied_overrides(self, mocker):
        send = mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        daemon.say("hello", voice="bf_emma")
        payload = send.call_args.args[0]
        assert payload == {"text": "hello", "voice": "bf_emma"}

    def test_say_with_all_overrides(self, mocker):
        send = mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        daemon.say("hi", voice="af_sarah", speed=1.5, lang="en-us")
        assert send.call_args.args[0] == {
            "text": "hi",
            "voice": "af_sarah",
            "speed": 1.5,
            "lang": "en-us",
        }

    def test_say_without_overrides_passes_only_text(self, mocker):
        send = mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        daemon.say("hi")
        assert send.call_args.args[0] == {"text": "hi"}

    def test_stop_sends_command_stop(self, mocker):
        # _read_pid is mocked out so the wait-for-exit loop can't reach a real
        # daemon's pid file and block on a live process.
        mocker.patch.object(daemon, "_read_pid", return_value=None)
        send = mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        daemon.stop()
        assert send.call_args.args[0] == {"command": "stop"}

    def test_ping_uses_short_timeout(self, mocker):
        send = mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        daemon.ping()
        assert send.call_args.kwargs["timeout"] == daemon.PING_TIMEOUT_SECONDS


class TestRefreshAudioDevices:
    def test_calls_terminate_then_initialize(self, mocker):
        terminate = mocker.patch.object(daemon.sd, "_terminate")
        initialize = mocker.patch.object(daemon.sd, "_initialize")

        daemon._refresh_audio_devices()

        terminate.assert_called_once()
        initialize.assert_called_once()

    def test_swallows_and_logs_errors(self, mocker, caplog):
        """A failure in the audio reset must not propagate out — the worker
        retries on the next request."""
        import logging

        mocker.patch.object(daemon.sd, "_terminate", side_effect=RuntimeError("portaudio gone"))
        mocker.patch.object(daemon.sd, "_initialize")

        with caplog.at_level(logging.ERROR, logger="stackvox.daemon"):
            daemon._refresh_audio_devices()  # must not raise

        assert any("failed to refresh audio devices" in r.message for r in caplog.records)


class TestStartDeviceWatcher:
    def test_no_op_on_non_darwin(self, mocker):
        """On Linux/Windows the function returns immediately; nothing should
        be loaded from CoreAudio and no thread should be started."""
        mocker.patch.object(daemon.sys, "platform", "linux")
        cdll = mocker.patch.object(daemon.ctypes, "CDLL")

        daemon._start_device_watcher()

        cdll.assert_not_called()

    @pytest.mark.skipif(
        not hasattr(daemon.sys, "platform") or daemon.sys.platform != "darwin",
        reason="macOS-only behaviour",
    )
    def test_disabled_when_coreaudio_unavailable(self, mocker, caplog):
        """If CDLL fails to load CoreAudio (unusual on real macOS but possible
        in stripped environments), the watcher should bail without raising."""
        import logging

        mocker.patch.object(daemon.ctypes, "CDLL", side_effect=OSError("no coreaudio"))
        with caplog.at_level(logging.DEBUG, logger="stackvox.daemon"):
            daemon._start_device_watcher()  # must not raise
        assert any("CoreAudio unavailable" in r.message for r in caplog.records)


class TestStopWaitsForExit:
    """`stop` reports success only once the process has actually gone.

    The daemon acknowledges the request before unwinding, so returning on the
    ack made `stackvox stop && stackvox serve` race: serve saw a live pid and
    refused, then the shutdown finished, leaving nothing running.
    """

    def test_returns_send_failure_unchanged(self, mocker):
        mocker.patch.object(daemon, "_read_pid", return_value=4242)
        mocker.patch.object(daemon, "send", return_value=(False, "daemon not running"))

        actual = daemon.stop()

        assert actual == (False, "daemon not running")

    def test_does_not_wait_when_no_pid_is_recorded(self, mocker):
        mocker.patch.object(daemon, "_read_pid", return_value=None)
        mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        alive = mocker.patch.object(daemon, "_pid_alive")

        actual = daemon.stop()

        assert actual == (True, "ok")
        alive.assert_not_called()

    def test_waits_until_the_process_has_exited(self, mocker):
        mocker.patch.object(daemon, "_read_pid", return_value=4242)
        mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        alive = mocker.patch.object(daemon, "_pid_alive", side_effect=[True, True, False])
        sleep = mocker.patch.object(daemon.time, "sleep")

        actual = daemon.stop()

        assert actual == (True, "ok")
        assert alive.call_count == 3
        assert sleep.call_count == 2

    def test_fails_when_the_process_outlives_the_timeout(self, mocker):
        mocker.patch.object(daemon, "_read_pid", return_value=4242)
        mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        mocker.patch.object(daemon, "_pid_alive", return_value=True)
        mocker.patch.object(daemon.time, "sleep")

        ok, resp = daemon.stop(timeout=0.0)

        assert ok is False
        assert "still alive" in resp
        assert "4242" in resp


class TestReadPid:
    def test_returns_none_when_missing(self, mocker, tmp_path):
        mocker.patch.object(daemon, "PID_PATH", tmp_path / "missing.pid")
        assert daemon._read_pid() is None

    def test_returns_none_when_unparseable(self, mocker, tmp_path):
        pid = tmp_path / "garbage.pid"
        pid.write_text("not-a-number")
        mocker.patch.object(daemon, "PID_PATH", pid)
        assert daemon._read_pid() is None

    def test_returns_the_recorded_pid(self, mocker, tmp_path):
        pid = tmp_path / "live.pid"
        pid.write_text("4242\n")
        mocker.patch.object(daemon, "PID_PATH", pid)
        assert daemon._read_pid() == 4242
