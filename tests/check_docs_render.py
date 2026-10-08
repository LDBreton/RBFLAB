"""Check expanded snippets and math in the generated site."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
for relative in (
    "tutorials/one-stencil/index.html",
    "tutorials/heat-equation/index.html",
    "tutorials/stokes/index.html",
    "theory/rbf-fd/index.html",
):
    html = (root / "site" / relative).read_text(encoding="utf-8")
    assert "--8<--" not in html, relative
    assert 'class="arithmatex"' in html, relative
    assert '<em>i)' not in html, relative

heat = (root / "site/tutorials/heat-equation/index.html").read_text(encoding="utf-8")
for name in ("five_point_laplacian", "rbf_fd_laplacian", "symbolic_solution"):
    assert name in heat, name
print("Rendered documentation includes math and runnable source.")
