from __future__ import annotations

import pytest

from stackvox import daemon


@pytest.fixture(autouse=True)
def isolated_runtime_files(mocker, tmp_path):
    """Point the daemon's socket and pid files into the test's temp directory.

    The defaults are the live daemon's own files, and a stray real
    `_exit_wedged` or startup during a test run would delete or replace them,
    leaving a running daemon that no client can reach."""
    mocker.patch.object(daemon, "SOCKET_PATH", tmp_path / "daemon.sock")
    mocker.patch.object(daemon, "PID_PATH", tmp_path / "daemon.pid")
