"""Deprecated compatibility shim for the ``genbio.gbsp`` package.

``genbio.aidosp`` was renamed to ``genbio.gbsp`` in the AIDO.* -> GB.* rebrand.
Importing ``genbio.aidosp`` (or any ``genbio.aidosp.*`` submodule) emits a
``DeprecationWarning`` and transparently resolves to ``genbio.gbsp``. Use
``genbio.gbsp`` directly in new code.
"""

import importlib
import sys
import warnings

warnings.warn(
    "'genbio.aidosp' is deprecated and will be removed in a future release; "
    "use 'genbio.gbsp' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Resolve this package name to the renamed ``genbio.gbsp`` package so that both
# ``import genbio.aidosp`` and ``import genbio.aidosp.<submodule>`` keep working.
_target = importlib.import_module("genbio.gbsp")
sys.modules[__name__] = _target
