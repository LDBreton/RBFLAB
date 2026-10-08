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

    assert {name for name in listed if "." not in name} == exported
    assert all(getattr(rbflab, name).__doc__ for name in exported)


def test_geometry_reference_exports():
    from rbflab import geometry, meshgen
    text = (Path(__file__).resolve().parents[1]/"docs/api/mesh-generation.md").read_text(encoding="utf-8")
    for module in (geometry,meshgen):
        for name in module.__all__:
            assert f"::: {module.__name__}.{name}" in text
            assert getattr(module,name).__doc__,name
