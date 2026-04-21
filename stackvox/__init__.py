"""stackvox — offline TTS using Kokoro-82M."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from stackvox import daemon
from stackvox.engine import Stackvox, speak, synthesize

try:
    __version__ = _pkg_version("stackvox")
except PackageNotFoundError:
    # Running from a source tree without the package installed (e.g. editable
    # bootstrap before `pip install -e .` has completed).
    __version__ = "0.0.0+unknown"

__all__ = ["Stackvox", "daemon", "speak", "synthesize"]
