# stackvox

Offline TTS library using [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx). Apache 2.0, ~340MB, CPU real-time, plays straight to system audio.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Model + voices auto-download to `~/.cache/stackvox/` on first use (override with `STACKVOX_CACHE_DIR`).

## Library usage

```python
from stackvox import Stackvox, speak

# One-shot — model loads on first call, cached for subsequent calls
speak("Hello world")

# Reusable engine
tts = Stackvox(voice="af_bella")
tts.speak("First line")
tts.speak("Faster", speed=1.2)

# Non-blocking
tts.speak("async", blocking=False)
tts.stop()

# Raw samples (numpy array + sample rate)
samples, sr = tts.synthesize("give me the array")
```

## CLI

Installed as `stackvox` when you `pip install -e .`:

```bash
stackvox "Hello world"                 # play
stackvox "Hi" --voice bf_emma          # different voice
stackvox "Save" --out out.wav          # write instead of play
stackvox --file script.txt
stackvox --list-voices
```

## Voices

`af_*` / `am_*` = American female/male, `bf_*` / `bm_*` = British. Run `--list-voices` for the full set.
