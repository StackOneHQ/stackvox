"""Update-check module tests — no real network, no real cache file."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from stackvox import updates


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    """Redirect the cache dir into tmp_path so we don't touch ~/.cache."""
    monkeypatch.setenv("STACKVOX_CACHE_DIR", str(tmp_path))
    # Some tests also need the disable-checks env vars unset.
    for var in (
        "STACKVOX_NO_UPDATE_CHECK",
        "CI",
        "GITHUB_ACTIONS",
        "BUILDKITE",
        "CIRCLECI",
        "GITLAB_CI",
        "TRAVIS",
    ):
        monkeypatch.delenv(var, raising=False)


# -----------------------------------------------------------------------------
# version comparison


class TestIsNewer:
    @pytest.mark.parametrize(
        "latest,current,expected",
        [
            ("0.4.0", "0.3.1", True),
            ("0.3.2", "0.3.1", True),
            ("1.0.0", "0.99.99", True),
            ("0.3.1", "0.3.1", False),
            ("0.3.0", "0.3.1", False),
            ("0.2.9", "0.3.0", False),
            # Local-version suffixes (`+local`) shouldn't fool the comparison.
            ("0.3.1+dev", "0.3.1", False),
        ],
    )
    def test_dotted(self, latest, current, expected):
        assert updates._is_newer(latest, current) is expected


# -----------------------------------------------------------------------------
# disable / opt-out behaviour


class TestIsDisabled:
    def test_off_when_no_env_vars_set(self):
        assert updates.is_disabled() is False

    def test_explicit_disable(self, monkeypatch):
        monkeypatch.setenv("STACKVOX_NO_UPDATE_CHECK", "1")
        assert updates.is_disabled() is True

    @pytest.mark.parametrize("var", ["CI", "GITHUB_ACTIONS", "BUILDKITE"])
    def test_skipped_in_common_ci(self, monkeypatch, var):
        monkeypatch.setenv(var, "true")
        assert updates.is_disabled() is True


# -----------------------------------------------------------------------------
# cache I/O


class TestCache:
    def test_read_returns_none_when_file_absent(self):
        assert updates.read_cache() is None

    def test_round_trip(self):
        when = datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)
        updates.write_cache("0.5.0", now=when)
        entry = updates.read_cache()
        assert entry is not None
        ts, latest = entry
        assert latest == "0.5.0"
        assert ts == when

    def test_corrupt_cache_yields_none(self, tmp_path):
        path = updates.cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not valid json", encoding="utf-8")
        assert updates.read_cache() is None

    def test_missing_keys_yield_none(self):
        path = updates.cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checked_at": "2026-04-30T00:00:00+00:00"}), encoding="utf-8")
        assert updates.read_cache() is None


# -----------------------------------------------------------------------------
# fetch


class TestFetchLatestVersion:
    def test_returns_version_from_pypi_json(self, mocker):
        body = json.dumps({"info": {"version": "0.5.0"}}).encode("utf-8")
        fake_resp = mocker.MagicMock()
        fake_resp.read.return_value = body
        fake_resp.__enter__ = mocker.MagicMock(return_value=fake_resp)
        fake_resp.__exit__ = mocker.MagicMock(return_value=False)
        mocker.patch.object(updates.urllib.request, "urlopen", return_value=fake_resp)

        assert updates.fetch_latest_version() == "0.5.0"

    def test_returns_none_on_network_error(self, mocker):
        mocker.patch.object(
            updates.urllib.request, "urlopen", side_effect=updates.urllib.error.URLError("nope")
        )
        assert updates.fetch_latest_version() is None

    def test_returns_none_on_malformed_json(self, mocker):
        fake_resp = mocker.MagicMock()
        fake_resp.read.return_value = b"not json at all"
        fake_resp.__enter__ = mocker.MagicMock(return_value=fake_resp)
        fake_resp.__exit__ = mocker.MagicMock(return_value=False)
        mocker.patch.object(updates.urllib.request, "urlopen", return_value=fake_resp)
        assert updates.fetch_latest_version() is None

    def test_skips_when_disabled(self, monkeypatch, mocker):
        monkeypatch.setenv("STACKVOX_NO_UPDATE_CHECK", "1")
        urlopen = mocker.patch.object(updates.urllib.request, "urlopen")
        assert updates.fetch_latest_version() is None
        urlopen.assert_not_called()


# -----------------------------------------------------------------------------
# higher-level cached_update / check_for_update


class TestCachedUpdate:
    def test_returns_none_when_no_cache(self):
        assert updates.cached_update() is None

    def test_returns_none_when_disabled(self, monkeypatch):
        updates.write_cache("99.0.0")
        monkeypatch.setenv("STACKVOX_NO_UPDATE_CHECK", "1")
        assert updates.cached_update() is None

    def test_returns_info_when_cache_says_newer(self, mocker):
        # Force a known "current" version regardless of installed metadata.
        mocker.patch.object(updates, "_current_version", return_value="0.3.1")
        updates.write_cache("0.4.0")
        info = updates.cached_update()
        assert info is not None
        assert info.current == "0.3.1"
        assert info.latest == "0.4.0"
        assert info.is_outdated is True

    def test_returns_none_when_already_on_latest(self, mocker):
        mocker.patch.object(updates, "_current_version", return_value="0.4.0")
        updates.write_cache("0.4.0")
        assert updates.cached_update() is None


class TestCheckForUpdate:
    def test_uses_cache_when_fresh(self, mocker):
        mocker.patch.object(updates, "_current_version", return_value="0.3.1")
        # Cache from 1 hour ago — well within the 24h TTL.
        when = datetime.now(timezone.utc) - timedelta(hours=1)
        updates.write_cache("0.5.0", now=when)
        fetch = mocker.patch.object(updates, "fetch_latest_version")

        info = updates.check_for_update()

        assert info is not None
        assert info.latest == "0.5.0"
        # Should not have fired the network call.
        fetch.assert_not_called()

    def test_refetches_when_cache_stale(self, mocker):
        mocker.patch.object(updates, "_current_version", return_value="0.3.1")
        when = datetime.now(timezone.utc) - timedelta(days=2)
        updates.write_cache("0.4.0", now=when)
        fetch = mocker.patch.object(updates, "fetch_latest_version", return_value="0.5.0")

        info = updates.check_for_update()

        assert info is not None
        assert info.latest == "0.5.0"
        fetch.assert_called_once()

    def test_falls_back_to_stale_cache_on_fetch_failure(self, mocker):
        mocker.patch.object(updates, "_current_version", return_value="0.3.1")
        when = datetime.now(timezone.utc) - timedelta(days=2)
        updates.write_cache("0.4.0", now=when)
        mocker.patch.object(updates, "fetch_latest_version", return_value=None)

        info = updates.check_for_update()

        assert info is not None
        assert info.latest == "0.4.0"

    def test_returns_none_when_no_cache_and_fetch_fails(self, mocker):
        mocker.patch.object(updates, "fetch_latest_version", return_value=None)
        assert updates.check_for_update() is None

    def test_returns_none_when_disabled(self, monkeypatch, mocker):
        monkeypatch.setenv("STACKVOX_NO_UPDATE_CHECK", "1")
        fetch = mocker.patch.object(updates, "fetch_latest_version")
        assert updates.check_for_update() is None
        fetch.assert_not_called()


class TestFormatNotice:
    def test_includes_versions_and_upgrade_command(self):
        info = updates.UpdateInfo(current="0.3.1", latest="0.4.0")
        msg = updates.format_notice(info)
        assert "0.3.1" in msg
        assert "0.4.0" in msg
        assert "pipx upgrade stackvox" in msg
