"""stackvox — offline TTS using Kokoro-82M."""

from stackvox import daemon
from stackvox.engine import Stackvox, speak, synthesize

__all__ = ["Stackvox", "speak", "synthesize", "daemon"]
__version__ = "0.1.0"
