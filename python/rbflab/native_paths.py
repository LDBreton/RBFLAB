"""Locate optional native sources in a source checkout."""
from pathlib import Path
import os


def source_root():
    """Return native-source root, optionally selected by RBFLAB_SOURCE_ROOT.

    Importing RBFLAB never requires native sources. Compilation validates the
    files later; installed-wheel users can point to a matching source checkout.
    """
    override = os.environ.get("RBFLAB_SOURCE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "cpp" / "real.hpp").is_file():
            return parent
    return Path(__file__).resolve().parents[3]
