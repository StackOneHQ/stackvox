# stackvox

[![ci](https://github.com/StackOneHQ/stackvox/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/StackOneHQ/stackvox/actions/workflows/ci.yml)
[![coverage](https://codecov.io/gh/StackOneHQ/stackvox/branch/main/graph/badge.svg)](https://codecov.io/gh/StackOneHQ/stackvox)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](./LICENSE)

Offline TTS using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx). Apache 2.0 model, ~340MB, CPU real-time, plays straight to system audio. Designed to be importable as a Python library, drivable as a CLI, or poked via a unix socket for ~13ms speech requests from shell scripts.

## Install

From PyPI — recommended for most users:

```bash
pipx install stackvox    # global CLI (`stackvox` and `stackvox-say` on PATH)
# or
pip install stackvox     # use as a library
```

From git, if you want an unreleased commit:

```bash
pipx install git+https://github.com/StackOneHQ/stackvox.git
# upgrade later with: pipx install --force git+https://github.com/StackOneHQ/stackvox.git
```

Dev install from a clone:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Model + voice files auto-download to `~/.cache/stackvox/` on first use. Override with `STACKVOX_CACHE_DIR`.

## CLI

```bash
stackvox "Hello world"              # synthesize and play in-process
stackvox speak "Hi" --voice bf_emma # same, explicit subcommand
stackvox speak "save" --out a.wav   # write wav instead of playing
stackvox welcome                    # multilingual welcome (6 languages)
stackvox voices                     # list all voice ids
```

Daemon mode (keeps the model resident so each subsequent call is instant):

```bash
stackvox serve         # foreground; run with `nohup stackvox serve &` to background
stackvox status        # is the daemon up?
stackvox say "Hello"   # send text to the daemon (fails if not running)
stackvox stop          # graceful shutdown
```

## `stackvox-say` (bash helper, ~13ms)

When you want minimum latency from shell scripts (hooks, CI steps, etc.), skip the Python client and use the bash helper — it talks directly to the daemon's unix socket via `nc`:

```bash
stackvox-say "back to you in 5"
stackvox-say --voice bf_emma --speed 1.1 "hello"
stackvox-say --fallback-say "text"     # shell out to macOS `say` if daemon is down
```

Exit codes: `0` ok, `2` daemon unreachable (unless `--fallback-say` was given).

## Python library

```python
from stackvox import Stackvox, speak, synthesize

# One-shot — model loads on first call, reused for subsequent calls.
speak("Hello world")

# Reusable engine.
tts = Stackvox(voice="af_bella")
tts.speak("First line")
tts.speak("Faster", speed=1.2)

# Non-blocking playback.
tts.speak("async", blocking=False)
tts.stop()

# Raw samples for custom processing.
samples, sr = tts.synthesize("give me the array")

# Gapless multi-line playback with concurrent synthesis.
tts.speak_sequence([
    {"text": "Hello", "voice": "af_heart", "lang": "en-us"},
    {"text": "Bonjour", "voice": "ff_siwis", "lang": "fr-fr"},
])
```

### Daemon client from Python

```python
from stackvox import daemon

ok, resp = daemon.say("queue this via the running daemon")
if daemon.is_running():
    daemon.stop()
```

## Voices

Kokoro ships voices across several languages. Voice prefix encodes gender + language:

| Prefix         | Language         | Example                  |
| -------------- | ---------------- | ------------------------ |
| `af_*`, `am_*` | American English | `af_heart`, `am_michael` |
| `bf_*`, `bm_*` | British English  | `bf_emma`, `bm_fable`    |
| `ff_*`         | French           | `ff_siwis`               |
| `hf_*`, `hm_*` | Hindi            | `hf_alpha`, `hm_omega`   |
| `if_*`, `im_*` | Italian          | `if_sara`, `im_nicola`   |
| `pf_*`, `pm_*` | Portuguese       | `pf_dora`, `pm_alex`     |
| `ef_*`, `em_*` | Spanish          | `ef_dora`, `em_alex`     |
| `jf_*`, `jm_*` | Japanese         | `jf_alpha`               |
| `zf_*`, `zm_*` | Mandarin Chinese | `zf_xiaoxiao`            |

Run `stackvox voices` for the authoritative list.

## Architecture

```
┌────────────────────┐      unix socket           ┌─────────────────────────┐
│  stackvox-say      │ ───────────────────────▶   │  stackvox daemon        │
│  (bash, ~13ms)     │   JSON line per request    │  (Python, long-lived)   │
└────────────────────┘                            │                         │
┌────────────────────┐      ~500ms (Py startup)   │  preloaded Kokoro ONNX  │
│  stackvox say      │ ───────────────────────▶   │  worker thread playback │
│  (Python client)   │                            │  → sounddevice → audio  │
└────────────────────┘                            └─────────────────────────┘
┌────────────────────┐
│  stackvox speak    │   loads model in-process, plays, exits
│  (one-shot CLI)    │
└────────────────────┘
```

Socket lives at `~/.cache/stackvox/daemon.sock` (override with `STACKVOX_SOCKET` for the client, `STACKVOX_CACHE_DIR` for the daemon). Protocol is one line of JSON per connection: `{"text":"...", "voice":"...", "speed":1.0, "lang":"en-us"}`; reply is `ok` / `busy` / `err: <msg>`. Plain text (no JSON) is accepted as a fallback and treated as `{"text": line}`.

Queue depth is 2 — rapid-fire requests beyond that get `busy` rather than piling up.

Before each utterance the daemon resets PortAudio so it picks up the current system default output device. Swap from speakers to Bluetooth headphones mid-session and the next `say` follows you — no daemon restart needed. The refresh costs ~10–50ms per play, which is invisible next to synthesis time.

## Requirements

- Python 3.10+
- macOS or Linux
- `nc` (BSD netcat — default on macOS, `netcat-openbsd` on Linux) for the bash helper

## Security considerations

stackvox doesn't open any network port. The daemon binds a unix socket under `~/.cache/stackvox/` (default file-mode `0600`, i.e. user-only per the OS defaults for files in `$HOME`). Any process running as the same local user can send text to the daemon — there's no per-message authentication on the socket itself. That's the trust boundary: stackvox assumes anything running as your UID is allowed to speak on your behalf.

If you're exposing stackvox through a different surface (HTTP server, shared system service, container), authentication and rate-limiting are your responsibility at that layer.

Model weights (`kokoro-v1.0.onnx`, ~340 MB) and voices are downloaded from the [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) GitHub release assets on first use and cached under `~/.cache/stackvox/`. If you operate in a restricted environment, pre-seed that directory offline.

Security issues themselves should not be filed as public GitHub issues — see [`SECURITY.md`](./SECURITY.md) for the disclosure process.

## License & attributions

stackvox itself is licensed under the **Apache License, Version 2.0** — see [`LICENSE`](./LICENSE). Third-party attributions are collected in [`NOTICE`](./NOTICE); the summary below is informational.

**Model.** Speech is generated by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (© hexgrad, Apache 2.0). The ONNX-converted weights (`kokoro-v1.0.onnx`) and voice pack (`voices-v1.0.bin`) are downloaded from the [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) release assets on first use and cached under `~/.cache/stackvox/`. stackvox does not modify or redistribute them.

**Runtime dependencies.** [`kokoro-onnx`](https://github.com/thewh1teagle/kokoro-onnx) (MIT, © thewh1teagle), [`onnxruntime`](https://github.com/microsoft/onnxruntime) (MIT, © Microsoft), [`sounddevice`](https://python-sounddevice.readthedocs.io/) (MIT, © Matthias Geier), [`soundfile`](https://github.com/bastibe/python-soundfile) (BSD-3, © Bastian Bechtold), [`numpy`](https://numpy.org/) (BSD-3).

**GPL note.** `kokoro-onnx` pulls in [`phonemizer-fork`](https://github.com/bootphon/phonemizer) as a transitive runtime dependency; it is licensed under **GPL-3.0**. stackvox does not bundle, modify, or statically link it — pip installs it alongside stackvox and the two communicate through phonemizer's published Python API at runtime. If you redistribute a combined work (e.g. a frozen binary, container image, or vendored wheel set) that includes phonemizer-fork, review GPL-3.0 obligations for that distribution.
