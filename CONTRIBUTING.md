# Contributing to stackvox

Thanks for your interest in making stackvox better. This doc covers local setup, how we run tests/lint, and conventions for commits and PRs.

## Development setup

```bash
git clone https://github.com/StackOneHQ/stackvox.git
cd stackvox
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install --install-hooks
pre-commit install --hook-type commit-msg
```

The two `pre-commit install` calls wire up (a) ruff lint/format on staged files and (b) commit-message validation via commitizen. CI runs the same checks — installing them locally just catches problems before you push.

System dependencies (for the audio and phonemizer stack):

- **macOS** — PortAudio ships with the system; `brew install espeak-ng` if you hit phonemizer errors.
- **Ubuntu/Debian** — `sudo apt-get install libportaudio2 espeak-ng`.

## Running tests and lint

```bash
pytest               # run the test suite
ruff check .         # lint
ruff format .        # auto-format
ruff format --check . # verify formatting without changing files
```

The tests mock out `Stackvox` so they don't download the 340 MB model — CI runs them across Python 3.10–3.13 on Ubuntu. Please keep that property: any new test that wants to touch real synthesis should be marked and skipped by default.

## Commit and branch style

- Branch names: `<short-topic>` (or `<ticket>/<topic>` if you're working against a StackOne ticket).
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/):
  - `feat: add X`
  - `fix: handle Y`
  - `docs: update README`
  - `refactor: extract paths module`
  - `test: cover socket protocol`
  - `chore: bump deps`
- The `commit-msg` hook validates each message against this format — if you'd rather use an interactive prompt, run `cz commit` instead of `git commit` (commitizen ships in the dev extras).
- Prefer small, focused commits over large omnibus ones.

## Pull request checklist

- [ ] `pytest` passes locally.
- [ ] `ruff check .` and `ruff format --check .` are clean.
- [ ] New public APIs have docstrings.
- [ ] README / CHANGELOG updated if user-visible behavior changed.
- [ ] No credentials, API keys, or PII in the diff.

## Code style notes

- Library code (`stackvox/`) uses `logging` for diagnostic output — please don't `print()` from there. CLI entry points (`cli.py`) may `print()` user-facing output.
- Filesystem paths go through `stackvox.paths`, not ad-hoc `~/.cache/...` construction.
- Constants that describe wire-protocol behavior (queue sizes, timeouts, buffer sizes) live as named module constants, not inline magic numbers.

## Release process

Releases are driven by [release-please](https://github.com/googleapis/release-please). You don't tag or publish manually.

1. Merge your PRs to `main` as usual, using conventional-commit messages/PR titles (`feat:`, `fix:`, `chore:`, etc.).
2. release-please opens a rolling "release PR" against `main` containing the version bump and the new `CHANGELOG.md` entries it derived from those commits.
3. When you're ready to cut a release, merge the release PR. release-please tags the commit (`vX.Y.Z`) and the `release-please.yml` workflow publishes the wheel and sdist to PyPI.

Breaking changes in pre-1.0 still bump the **minor** version (e.g. `0.2.0 → 0.3.0`) — configured via `bump-minor-pre-major` in `release-please-config.json`.

### PyPI auth — no token, no secret

We publish via **PyPI Trusted Publishing** (OIDC). There is no `PYPI_API_TOKEN` stored in this repo, and there shouldn't be. At publish time, GitHub mints a short-lived OIDC token containing claims like "StackOneHQ/stackvox, workflow release-please.yml, environment pypi"; PyPI matches it against a pre-configured Trusted Publisher and issues a short-lived upload token internally. Nothing to rotate, nothing to leak.

The `permissions: id-token: write` entry on the `publish` job in `release-please.yml` is what enables this — it's a GitHub capability, not a stored credential. If you're ever tempted to add a `PYPI_API_TOKEN` secret, stop and fix the Trusted Publisher config on PyPI instead.

### One-time maintainer setup (owner-only)

Secrets in repository settings:

| Secret | Purpose | Required? |
|---|---|---|
| `REPO_GH_PAT` | Fine-grained PAT with Contents + Pull-requests write on this repo. release-please uses it so the release PR auto-triggers CI — the default `GITHUB_TOKEN` cannot cascade workflows onto its own PRs. | Recommended. If missing we fall back to `GITHUB_TOKEN`; release-please still opens PRs and publishes on merge, but you'll need to close/reopen the release PR to run CI on it. |

PyPI side (no secret lands in the repo):

- Configure a Trusted Publisher at <https://pypi.org/manage/account/publishing/> with exactly these claims:
  - PyPI project name: `stackvox`
  - Owner: `StackOneHQ`
  - Repository name: `stackvox`
  - Workflow name: `release-please.yml`
  - Environment name: `pypi`

GitHub side:

- Create a `pypi` Environment under Settings → Environments. Just the name; a required-reviewer gate is optional but recommended.

## Reporting security issues

Do not file public GitHub issues for security vulnerabilities — see [`SECURITY.md`](./SECURITY.md).
