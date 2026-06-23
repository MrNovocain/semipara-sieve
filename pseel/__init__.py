"""Source-tree import shim for running ``python -m pseel.run`` before installation.

The installable package lives under ``src/pseel``. This shim extends the package
search path when working directly from the repository root.
"""

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "pseel"
if _SRC_PACKAGE.exists():
    __path__.append(str(_SRC_PACKAGE))  # type: ignore[name-defined]
