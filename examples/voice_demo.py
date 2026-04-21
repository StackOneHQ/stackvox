"""Play sample lines across languages."""

from stackvox import Stackvox

SAMPLES = [
    ("ff_siwis", "fr-fr", "Bonjour, je suis Siwis — une voix française."),
    ("if_sara", "it", "Ciao, sono Sara — una voce italiana."),
    ("im_nicola", "it", "Ciao, sono Nicola — una voce maschile italiana."),
    ("hf_alpha", "hi", "Namaste, main Alpha hoon — ek Hindi awaaz."),
    ("hm_omega", "hi", "Namaste, main Omega hoon — ek Hindi purush awaaz."),
    ("pf_dora", "pt-br", "Olá, eu sou Dora — uma voz em português."),
    ("pm_alex", "pt-br", "Olá, eu sou Alex — uma voz masculina em português."),
    ("bm_lewis", "en-gb", "Hello, I'm Lewis — closest UK voice; Kokoro has no Scottish option."),
    ("bm_daniel", "en-gb", "And I'm Daniel — another British male voice."),
]

tts = Stackvox()
for voice, lang, line in SAMPLES:
    print(f"→ {voice} ({lang})")
    tts.speak(line, voice=voice, lang=lang)
