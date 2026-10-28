"""Daemon tests for non-protocol surface — pid/socket helpers, send/say/stop/ping
client, _refresh_audio_devices, the cross-platform watcher entry point, and the
wedge watchdog. The protocol/handler tests live in test_daemon_protocol.py.
"""

from __future__ import annotations

import logging
import os
import time

import pytest

from stackvox import daemon
from stackvox.engine import Playback


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

    def test_say_serializes_generation_options(self, mocker, tmp_path):
        send = mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        daemon.say("hi", reference_audio=tmp_path / "ref.wav", reference_text="Ref.", top_k=10)
        assert send.call_args.args[0] == {
            "text": "hi",
            "reference_audio": str(tmp_path / "ref.wav"),
            "reference_text": "Ref.",
            "top_k": 10,
        }

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


class TestStopKillsAHungDaemon:
    """A daemon that acknowledges `stop` but outlives the wait gets SIGKILL.

    A worker deadlocked in PortAudio blocks the normal shutdown path too, so
    without the kill `stop` reported failure and left the daemon running.
    """

    def test_kills_the_process_and_removes_its_files(self, mocker):
        mocker.patch.object(daemon, "_read_pid", return_value=4242)
        mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        mocker.patch.object(daemon, "_pid_alive", side_effect=[True, False])
        mocker.patch.object(daemon.time, "sleep")
        kill = mocker.patch.object(daemon.os, "kill")
        remove = mocker.patch.object(daemon, "_remove_runtime_files")

        ok, resp = daemon.stop(timeout=0.0)

        assert ok is True
        assert "killed pid 4242" in resp
        kill.assert_called_once_with(4242, daemon.signal.SIGKILL)
        remove.assert_called_once()

    def test_keeps_the_files_of_a_daemon_started_since(self, mocker):
        mocker.patch.object(daemon, "_read_pid", side_effect=[4242, 5151])
        mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        mocker.patch.object(daemon, "_pid_alive", side_effect=[True, False])
        mocker.patch.object(daemon.time, "sleep")
        mocker.patch.object(daemon.os, "kill")
        remove = mocker.patch.object(daemon, "_remove_runtime_files")

        ok, _resp = daemon.stop(timeout=0.0)

        assert ok is True
        remove.assert_not_called()

    def test_reports_success_when_the_process_exits_before_the_kill(self, mocker):
        mocker.patch.object(daemon, "_read_pid", return_value=4242)
        mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        mocker.patch.object(daemon, "_pid_alive", return_value=True)
        mocker.patch.object(daemon.time, "sleep")
        mocker.patch.object(daemon.os, "kill", side_effect=ProcessLookupError)
        remove = mocker.patch.object(daemon, "_remove_runtime_files")

        actual = daemon.stop(timeout=0.0)

        assert actual == (True, "ok")
        remove.assert_not_called()

    def test_fails_when_the_process_survives_sigkill(self, mocker):
        mocker.patch.object(daemon, "_read_pid", return_value=4242)
        mocker.patch.object(daemon, "send", return_value=(True, "ok"))
        mocker.patch.object(daemon, "_pid_alive", return_value=True)
        mocker.patch.object(daemon.time, "sleep")
        mocker.patch.object(daemon, "KILL_TIMEOUT_SECONDS", 0.0)
        mocker.patch.object(daemon.os, "kill")
        remove = mocker.patch.object(daemon, "_remove_runtime_files")

        ok, resp = daemon.stop(timeout=0.0)

        assert ok is False
        assert "still alive after SIGKILL" in resp
        assert "4242" in resp
        remove.assert_not_called()


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


class TestRemoveRuntimeFiles:
    def test_removes_the_socket_and_pid_files(self, mocker, tmp_path):
        sock = tmp_path / "daemon.sock"
        pid = tmp_path / "daemon.pid"
        sock.touch()
        pid.write_text("4242")
        mocker.patch.object(daemon, "SOCKET_PATH", sock)
        mocker.patch.object(daemon, "PID_PATH", pid)

        daemon._remove_runtime_files()

        assert not sock.exists()
        assert not pid.exists()

    def test_tolerates_files_already_gone(self, mocker, tmp_path):
        mocker.patch.object(daemon, "SOCKET_PATH", tmp_path / "daemon.sock")
        mocker.patch.object(daemon, "PID_PATH", tmp_path / "daemon.pid")

        daemon._remove_runtime_files()  # must not raise


class TestExitWedged:
    def test_removes_the_socket_before_exiting(self, mocker, tmp_path):
        """Clients start a fresh daemon when the socket is missing, so it must
        be gone by the time the process is."""
        sock = tmp_path / "daemon.sock"
        sock.touch()
        mocker.patch.object(daemon, "SOCKET_PATH", sock)
        mocker.patch.object(daemon, "PID_PATH", tmp_path / "daemon.pid")
        socket_present_at_exit = []
        hard_exit = mocker.patch.object(
            daemon.os, "_exit", side_effect=lambda code: socket_present_at_exit.append(sock.exists())
        )

        daemon._exit_wedged()

        hard_exit.assert_called_once_with(daemon.os.EX_SOFTWARE)
        assert socket_present_at_exit == [False]


class TestUtteranceBudget:
    def test_empty_text_gets_the_grace_allowance(self):
        assert daemon._utterance_budget("") == daemon.WATCHDOG_GRACE_SECONDS

    def test_grows_with_the_length_of_the_text(self):
        actual = daemon._utterance_budget("x" * 1000)

        assert actual == daemon.WATCHDOG_GRACE_SECONDS + 1000 / daemon.WATCHDOG_MIN_CHARS_PER_SECOND


@pytest.fixture
def exit_wedged(mocker):
    """The watchdog's hard exit, mocked: the real one would end the test run and
    delete the live daemon's socket and pid files."""
    return mocker.patch.object(daemon, "_exit_wedged")


@pytest.fixture
def state(mocker, exit_wedged):
    """A _DaemonState with the engine and the audio side mocked out, and a
    watchdog that polls fast enough to test. The poll interval is patched before
    the watchdog starts, because its first wait reads it once.

    Teardown waits for the watchdog to stop before the mocks are undone. A
    watchdog paused between spotting an overrun and calling _exit_wedged would
    otherwise wake in the next test and call its mock, or the real exit."""
    mocker.patch.object(daemon, "WATCHDOG_POLL_SECONDS", 0.01)
    mocker.patch.object(daemon, "Stackvox")
    mocker.patch.object(daemon, "_refresh_audio_devices")
    mocker.patch.object(daemon, "_start_device_watcher")
    daemon_state = daemon._DaemonState(voice="af_sarah", speed=1.0, lang="en-us")
    try:
        yield daemon_state
    finally:
        daemon_state.shutdown()
        daemon_state.watchdog.join()


def _wait_until(condition, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)


class TestWorkerDeadline:
    def test_is_armed_during_playback_and_cleared_after(self, state):
        deadlines_seen = []
        state.tts.speak.side_effect = lambda *args, **kwargs: deadlines_seen.append(state.deadline)
        before = time.monotonic()

        state.submit({"text": "hello"})
        _wait_until(lambda: deadlines_seen and state.deadline is None)

        assert len(deadlines_seen) == 1
        assert deadlines_seen[0] >= before + daemon._utterance_budget("hello")
        assert state.deadline is None

    def test_is_cleared_when_playback_fails(self, state):
        state.tts.speak.side_effect = RuntimeError("device gone")

        state.submit({"text": "hello"})
        _wait_until(lambda: state.tts.speak.called and state.deadline is None)

        assert state.tts.speak.called
        assert state.deadline is None


class TestUtteranceSummary:
    def test_times_count_from_when_the_request_was_accepted(self):
        playback = Playback(first_audio_at=107.0, audio_seconds=3.5)

        actual = daemon._utterance_summary(84, 100.0, 100.5, 111.25, playback)

        assert actual == (
            "spoke 84 chars: first audio after 7.0s (0.5s queued), finished after 11.2s, 3.5s of audio"
        )

    def test_says_when_nothing_played(self):
        playback = Playback(first_audio_at=None, audio_seconds=0.0)

        actual = daemon._utterance_summary(5, 100.0, 100.0, 100.5, playback)

        assert actual == "spoke 5 chars: no audio played, finished after 0.5s"


class TestWorkerLog:
    def test_logs_each_request_it_plays(self, state, caplog):
        state.tts.speak.side_effect = lambda *args, **kwargs: Playback(
            first_audio_at=time.monotonic(), audio_seconds=1.5
        )

        with caplog.at_level(logging.INFO, logger="stackvox.daemon"):
            state.submit({"text": "hello"})
            _wait_until(lambda: any("spoke" in message for message in caplog.messages))

        [actual] = [message for message in caplog.messages if "spoke" in message]
        assert actual.startswith("spoke 5 chars: first audio after ")
        assert actual.endswith(", 1.5s of audio")

    def test_logs_nothing_when_playback_fails(self, state, caplog):
        state.tts.speak.side_effect = RuntimeError("device gone")

        with caplog.at_level(logging.INFO, logger="stackvox.daemon"):
            state.submit({"text": "hello"})
            _wait_until(lambda: state.tts.speak.called and state.deadline is None)

        assert state.tts.speak.called
        assert state.deadline is None
        assert not any("spoke" in message for message in caplog.messages)


class TestOverrun:
    def test_is_none_while_idle(self, state):
        assert state.overrun(time.monotonic()) is None

    def test_is_none_within_the_deadline(self, state):
        state.deadline = 100.0

        assert state.overrun(99.0) is None

    def test_is_the_time_past_the_deadline(self, state):
        state.deadline = 100.0

        assert state.overrun(130.0) == 30.0


class TestWatchdog:
    def test_exits_once_the_worker_overruns_its_deadline(self, state, exit_wedged):
        state.deadline = time.monotonic() - 1.0
        _wait_until(lambda: exit_wedged.called)

        exit_wedged.assert_called()

    def test_leaves_a_worker_within_its_deadline_alone(self, state, exit_wedged):
        state.deadline = time.monotonic() + 60.0
        time.sleep(0.1)

        exit_wedged.assert_not_called()


def _hold_lock_elsewhere(path):
    """Take the startup lock through a separate open file description, as another process would."""
    import fcntl

    handle = open(path, "a")  # noqa: SIM115 - closed by the caller
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


@pytest.fixture
def private_paths(mocker, tmp_path):
    mocker.patch.object(daemon, "LOCK_PATH", tmp_path / "daemon.lock")
    mocker.patch.object(daemon, "PID_PATH", tmp_path / "daemon.pid")
    mocker.patch.object(daemon, "SOCKET_PATH", tmp_path / "daemon.sock")
    return tmp_path


class TestStartupLock:
    def test_refuses_while_another_holder_has_it(self, private_paths):
        other = _hold_lock_elsewhere(private_paths / "daemon.lock")
        try:
            with pytest.raises(RuntimeError, match="already starting or running"), daemon.startup_lock():
                pass
        finally:
            other.close()

    def test_is_reentrant_within_a_process(self, private_paths):
        with daemon.startup_lock(), daemon.startup_lock():
            pass

    def test_releases_on_exit(self, private_paths):
        with daemon.startup_lock():
            pass

        _hold_lock_elsewhere(private_paths / "daemon.lock").close()  # would raise if still held

    def test_serve_loads_nothing_while_another_daemon_holds_it(self, mocker, private_paths):
        state = mocker.patch.object(daemon, "_DaemonState")
        other = _hold_lock_elsewhere(private_paths / "daemon.lock")
        try:
            with pytest.raises(RuntimeError, match="already starting or running"):
                daemon.serve()
        finally:
            other.close()
        state.assert_not_called()

    def test_serve_releases_it_when_startup_fails(self, mocker, private_paths):
        mocker.patch.object(daemon, "_DaemonState", side_effect=RuntimeError("model failed to load"))

        with pytest.raises(RuntimeError, match="model failed to load"):
            daemon.serve()

        _hold_lock_elsewhere(private_paths / "daemon.lock").close()  # would raise if still held

    def test_bind_failure_is_a_clean_runtime_error(self, mocker, private_paths):
        mocker.patch.object(daemon, "_DaemonState")
        mocker.patch.object(daemon, "_UnixServer", side_effect=OSError(48, "Address already in use"))

        with pytest.raises(RuntimeError, match="could not listen on"):
            daemon.serve()

    def test_is_held_for_the_whole_block(self, private_paths):
        with daemon.startup_lock(), pytest.raises(BlockingIOError):
            _hold_lock_elsewhere(private_paths / "daemon.lock")

    def test_serve_holds_it_while_serving(self, mocker, private_paths):
        """Pins the lock to the daemon's life, not just its startup."""
        mocker.patch.object(daemon, "_DaemonState")
        mocker.patch.object(daemon, "_check_for_update_async")
        mocker.patch.object(daemon.signal, "signal")
        held_while_serving = []

        def serve_forever():
            try:
                _hold_lock_elsewhere(private_paths / "daemon.lock").close()
                held_while_serving.append(False)
            except BlockingIOError:
                held_while_serving.append(True)

        mocker.patch.object(daemon, "_UnixServer").return_value.serve_forever.side_effect = serve_forever

        daemon.serve()

        assert held_while_serving == [True]

    def test_unopenable_lock_file_is_a_clean_runtime_error(self, mocker, private_paths):
        mocker.patch.object(daemon, "open", create=True, side_effect=PermissionError(13, "Permission denied"))

        with pytest.raises(RuntimeError, match="could not open the startup lock"), daemon.startup_lock():
            pass

    def test_other_lock_failures_are_a_clean_runtime_error(self, mocker, private_paths):
        import errno

        mocker.patch.object(daemon.fcntl, "flock", side_effect=OSError(errno.EIO, "Input/output error"))

        with pytest.raises(RuntimeError, match="could not take the startup lock"), daemon.startup_lock():
            pass

    def test_unavailable_locking_starts_unlocked_with_one_warning(self, mocker, private_paths, caplog):
        import errno
        import logging

        mocker.patch.object(
            daemon.fcntl, "flock", side_effect=OSError(errno.ENOTSUP, "Operation not supported")
        )
        entered = []

        with (
            caplog.at_level(logging.WARNING, logger="stackvox.daemon"),
            daemon.startup_lock(),
            daemon.startup_lock(),
        ):
            entered.append(True)

        assert entered == [True]
        assert caplog.text.count("starting without the startup lock") == 1


class TestBackendClient:
    def test_reports_the_backend_the_daemon_names(self, mocker):
        mocker.patch.object(daemon, "send", return_value=(False, "mlx"))
        assert daemon.backend() == (True, "mlx")

    def test_an_old_daemons_error_is_not_a_backend(self, mocker):
        mocker.patch.object(daemon, "send", return_value=(False, "err: missing text"))
        assert daemon.backend() == (False, "err: missing text")
