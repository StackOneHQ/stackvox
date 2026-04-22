# Changelog

All notable changes to stackvox are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0](https://github.com/StackOneHQ/stackvox/compare/stackvox-v0.2.0...stackvox-v0.3.0) (2026-04-22)


### Features

* refresh audio devices before each daemon play ([1b15de7](https://github.com/StackOneHQ/stackvox/commit/1b15de7ed1c23946bdd3a9711b6d88dd5178a39d))
* **stackvox:** first commit ([0ad7de6](https://github.com/StackOneHQ/stackvox/commit/0ad7de60d30d2b88021726346c670361e059d482))

## [Unreleased]

### Added

- `CODE_OF_CONDUCT.md` adopting Contributor Covenant 2.1.
- GitHub issue templates (bug report, feature request) and a PR template.
- README badges for CI status, coverage, Python version, and license.
- `stackvox/py.typed` marker so downstream type checkers pick up the inline type hints.
- `.github/dependabot.yml` — weekly pip and GitHub Actions updates.
- Release automation via [release-please](https://github.com/googleapis/release-please): merging conventional-commit PRs rolls a release PR that, when merged, tags the commit and publishes to PyPI via Trusted Publishing (OIDC — no API tokens stored in the repo).
- PR-title validation workflow (`amannn/action-semantic-pull-request`) — important for squash-merge flows where the PR title becomes the commit message on main.
- mypy type-checking — configured in `[tool.mypy]` with pragmatic strictness, run as a dedicated CI job.
- Test coverage reporting via `pytest-cov`, uploaded to Codecov (public-repo tokenless, or via `CODECOV_TOKEN` if set).

### Changed

- Removed the tag-triggered `release.yml` workflow — release-please supersedes it.
- `[tool.commitizen]` trimmed to commit-message validation only; version bumps and CHANGELOG entries are now owned by release-please.

## [0.2.0] - 2026-04-22

First open-source release. Adds packaging/licensing hygiene, a test suite, CI, and a quality-of-life fix for audio device switches.

### Added

- Daemon automatically refreshes PortAudio before each utterance, so switching the system output device (e.g. plugging in Bluetooth headphones) no longer requires a daemon restart.
- `python -m stackvox` entry point.
- `stackvox.paths` module exposing `cache_dir()`, `socket_path()`, and `pid_path()` as public API.
- `LICENSE` (Apache 2.0), `NOTICE` with third-party attributions, `CONTRIBUTING.md`, and `SECURITY.md`.
- Test suite under `tests/` covering path resolution, CLI argument routing, and the daemon socket protocol.
- GitHub Actions CI running ruff lint/format and pytest across Python 3.10 – 3.13 on Ubuntu.
- `dev` optional dependency group (`pytest`, `pytest-mock`, `ruff`).
- Ruff configuration and pytest configuration in `pyproject.toml`.
- Project metadata: SPDX license expression, authors, keywords, project URLs, classifiers.
- README section documenting licenses and third-party attributions.

### Changed

- Library modules (`engine.py`, `daemon.py`) emit diagnostics via `logging` instead of `print()`. The CLI configures `basicConfig` at entry so output shape is unchanged for users.
- Example scripts moved from the repo root to `examples/`.
- `stackvox.__version__` is now read from installed package metadata, so `pyproject.toml` is the single source of truth for the version.
- Magic numbers in `daemon.py` (worker poll interval, client/ping timeouts, recv buffer size) promoted to named module constants.

### Internal

- Promoted `stackvox.engine._cache_dir` (private) to `stackvox.paths.cache_dir()` (public); `daemon.py` no longer reaches across module boundaries into a private helper.

## [0.1.0] - 2026-04-17

Initial commit. Offline TTS using Kokoro-82M via kokoro-onnx, with a Python library, CLI, and a long-running daemon driven over a unix socket for low-latency shell access.

[Unreleased]: https://github.com/StackOneHQ/stackvox/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/StackOneHQ/stackvox/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/StackOneHQ/stackvox/releases/tag/v0.1.0
