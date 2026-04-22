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

## Reporting security issues

Do not file public GitHub issues for security vulnerabilities — see [`SECURITY.md`](./SECURITY.md).
