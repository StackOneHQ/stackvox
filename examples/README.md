# Examples

Runnable snippets that demonstrate stackvox's library surface. Run them from the repo root after `pip install -e .`.

## `basic.py`

The Quickstart. Shows:

- module-level `speak(...)` — loads the model on first call and reuses it after,
- a reusable `Stackvox` engine with a custom voice,
- non-blocking playback and `stop()`,
- raw sample access via `synthesize(...)` for custom processing.

```bash
python examples/basic.py
```

## `voice_demo.py`

Iterates through a French, Italian, Hindi, Portuguese, and British-English voice, speaking a short line in each language so you can pick one by ear. Useful when you want to see (well, hear) what the voice prefixes in the README table actually sound like.

```bash
python examples/voice_demo.py
```
