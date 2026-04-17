"""Quickstart examples for stackvox."""

from stackvox import Stackvox, speak

# One-shot (loads model on first call, reuses it after)
speak("Hello from stackvox.")

# Reusable engine with a different voice
tts = Stackvox(voice="af_bella")
tts.speak("This is Bella speaking.")
tts.speak("And slightly faster.", speed=1.2)

# Non-blocking playback
tts.speak("Fire and forget.", blocking=False)
tts.stop()

# Raw samples for custom handling
samples, sr = tts.synthesize("Give me the array.")
print(f"Got {len(samples)} samples at {sr}Hz")
