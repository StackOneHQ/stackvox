"""Best-effort PyPI update check.

Surfaces "a newer stackvox is on PyPI" without polluting the script paths
that make up most of stackvox's invocations. The actual fetch is opt-in
(only `stackvox status` and the daemon's startup trigger it); everywhere
else reads from a 24h cache.

Cache schema at `~/.cache/stackvox/update-check.json`::

    {"checked_at": "2026-04-30T13:42:00+00:00", "latest": "0.4.0"}

Disable entirely with `STACKVOX_NO_UPDATE_CHECK=1`. Auto-skipped when any
common CI env var is set so build logs stay clean.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - covered by 3.10 CI
    import tomli as tomllib

from stackvox.paths import cache_dir

logger = logging.getLogger(__name__)


def _source_tree_version(pyproject: Path | None = None) -> str | None:
    """Our version from `pyproject.toml`, when running from a source checkout.

    An editable install records its version in dist metadata at install time, so
    the moment a release bumps `pyproject.toml` that metadata goes stale. The
    developer then gets a phantom "update available: 0.9.0 -> 0.11.0" against
    their own tree, and `status` reads the running daemon as newer than the
    "installed" package and tells them to restart it for no reason.

    Returns None when there's no such file (a normal pip/pipx install) or when it
    isn't ours, so the metadata path stays authoritative for real installs.
    """
    path = pyproject or Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = data.get("project")
    if not isinstance(project, dict) or project.get("name") != "stackvox":
        return None
    version = project.get("version")
    return version if isinstance(version, str) else None


def _current_version() -> str:
    """Read our own installed version. Late-bound so importing this module
    early in the package init chain doesn't trip a circular import.

    A source checkout wins over dist metadata, which goes stale on an editable
    install after every release.
    """
    from_source = _source_tree_version()
    if from_source is not None:
        return from_source
    try:
        return _pkg_version("stackvox")
    except PackageNotFoundError:
        return "0.0.0+unknown"


PYPI_JSON_URL = "https://pypi.org/pypi/stackvox/json"
CACHE_TTL = timedelta(hours=24)
FETCH_TIMEOUT_SECONDS = 2.0

# Env vars set in common CI environments. Presence of any disables the check.
_CI_ENV_VARS = ("CI", "GITHUB_ACTIONS", "BUILDKITE", "CIRCLECI", "GITLAB_CI", "TRAVIS")


@dataclass(frozen=True)
class UpdateInfo:
    current: str
    latest: str

    @property
    def is_outdated(self) -> bool:
        return _is_newer(self.latest, self.current)


def cache_path() -> Path:
    return cache_dir() / "update-check.json"


def is_disabled() -> bool:
    """Whether the update check should be skipped at all."""
    if os.environ.get("STACKVOX_NO_UPDATE_CHECK"):
        return True
    return any(os.environ.get(v) for v in _CI_ENV_VARS)


def _is_newer(latest: str, current: str) -> bool:
    """Compare two PEP-440-ish dotted version strings.

    Handles the X.Y.Z and X.Y.Z+suffix forms stackvox actually publishes.
    Non-integer segments (e.g. an `rc1` tail) sort below numeric ones so
    `0.4.0` > `0.4.0rc1`, but two distinct pre-release labels compare
    equal — fine for stackvox since we don't ship alphas/betas.
    """

    def _key(v: str) -> tuple[int, ...]:
        head = v.split("+", 1)[0]  # drop +local
        parts: list[int] = []
        for piece in head.split("."):
            try:
                parts.append(int(piece))
            except ValueError:
                parts.append(-1)
        return tuple(parts)

    return _key(latest) > _key(current)


def fetch_latest_version(timeout: float = FETCH_TIMEOUT_SECONDS) -> str | None:
    """Hit PyPI's JSON API for the latest stackvox version.

    Returns None on any network or parse error — this is best-effort.
    """
    if is_disabled():
        return None
    try:
        ua = f"stackvox/{_current_version()} update-check"
        req = urllib.request.Request(PYPI_JSON_URL, headers={"User-Agent": ua})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - constant URL
            payload = json.loads(resp.read().decode("utf-8"))
        version = payload.get("info", {}).get("version")
        if isinstance(version, str):
            return version
        return None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        logger.debug("update check failed: %s", exc)
        return None


def write_cache(latest: str, *, now: datetime | None = None) -> None:
    """Persist the most recent successful check."""
    path = cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checked_at": (now or datetime.now(timezone.utc)).isoformat(),
        "latest": latest,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_cache() -> tuple[datetime, str] | None:
    """Return (timestamp, latest) from the cache file, or None if absent/broken."""
    path = cache_path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        when = datetime.fromisoformat(payload["checked_at"])
        latest = payload["latest"]
        if not isinstance(latest, str):
            return None
        return when, latest
    except (OSError, ValueError, KeyError) as exc:
        logger.debug("ignoring malformed update-check cache: %s", exc)
        return None


def cached_update(*, now: datetime | None = None) -> UpdateInfo | None:
    """Read the cache and return UpdateInfo for an unmet upgrade, else None."""
    if is_disabled():
        return None
    entry = read_cache()
    if entry is None:
        return None
    _checked_at, latest = entry
    info = UpdateInfo(current=_current_version(), latest=latest)
    return info if info.is_outdated else None


def check_for_update(*, now: datetime | None = None) -> UpdateInfo | None:
    """Fetch from PyPI if the cache is stale, then return any pending update.

    Synchronous; safe to call from the foreground only when ~2s of network
    latency is acceptable (e.g. `stackvox status`). For the daemon startup
    path, call this from a background thread.
    """
    if is_disabled():
        return None
    now = now or datetime.now(timezone.utc)
    entry = read_cache()
    if entry is None or (now - entry[0]) > CACHE_TTL:
        latest = fetch_latest_version()
        if latest is not None:
            write_cache(latest, now=now)
        else:
            # On fetch failure, fall back to whatever's already cached (which
            # may be None, in which case we just give up silently).
            if entry is None:
                return None
            latest = entry[1]
    else:
        latest = entry[1]
    info = UpdateInfo(current=_current_version(), latest=latest)
    return info if info.is_outdated else None


def format_notice(info: UpdateInfo) -> str:
    """Single-line user-facing message describing the available upgrade."""
    return f"update available: {info.current} → {info.latest} (run `pipx upgrade stackvox`)"
