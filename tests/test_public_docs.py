"""Guard the published API reference against undocumented new exports."""

from pathlib import Path

import rbflab


def test_public_exports_have_docstrings_and_reference_entries():
    root = Path(__file__).resolve().parents[1]
    reference = root / "docs" / "api"
    listed = {
        line.removeprefix("::: rbflab.").strip()
        for page in reference.glob("*.md")
        for line in page.read_text(encoding="utf-8").splitlines()
        if line.startswith("::: rbflab.")
    }
    exported = set(rbflab.__all__)

    assert listed == exported
    assert all(getattr(rbflab, name).__doc__ for name in exported)
