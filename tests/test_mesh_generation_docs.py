"""Execute the construction guide's Python blocks in reading order."""
from pathlib import Path
import re
import pytest

PAGES = ["index", "planar-domains", "parametric-borders", "custom-domains",
         "labels-normals", "sampling", "volumes-3d", "surfaces-3d", "staggered", "gmsh"]


@pytest.mark.parametrize("page", PAGES)
def test_mesh_generation_tutorial(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if page == "gmsh":
        pytest.importorskip("gmsh")
    source = Path(__file__).resolve().parents[1] / "docs" / "geometry" / (page + ".md")
    blocks = re.findall(r"^```python\s*\n(.*?)^```", source.read_text(encoding="utf-8"), re.M | re.S)
    assert blocks, f"No executable construction example in {source}"
    namespace = {"__name__": "__tutorial__"}
    try:
        for index, code in enumerate(blocks, 1):
            exec(compile(code, f"{source}:block{index}", "exec"), namespace)
    finally:
        plt.close("all")
