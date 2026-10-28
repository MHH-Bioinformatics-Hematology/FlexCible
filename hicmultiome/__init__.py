"""Compatibility alias: the package was renamed to ``flexcible`` (2026-10-28).

``import hicmultiome.<module>`` keeps working for old scripts, configs and run
logs by loading the modules of ``flexcible`` under this name.
"""
import flexcible as _f

__path__ = _f.__path__          # submodules resolve to flexcible/<module>.py
__version__ = getattr(_f, "__version__", "0.1.0")
from flexcible import *          # noqa: F401,F403
