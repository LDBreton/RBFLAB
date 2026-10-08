# Maintaining the documentation

Public API pages are generated from Python docstrings with MkDocs and
mkdocstrings. The reference includes every name exported through
`rbflab.__all__`; private implementation helpers are not part of it.

## Write a public docstring

Add a triple-quoted docstring immediately inside a public function or class.
Describe what the object does, the mathematical convention when relevant,
parameter types and shapes, the returned object, and numerical limitations.
Use Google-style `Args:` and `Returns:` sections for nontrivial functions.
Keep explanations and complete examples in guides rather than in long
constructor docstrings.

When adding a public export, put it on the appropriate page in `docs/api/`.
The `test_public_docs.py` check compares reference entries with
`rbflab.__all__` and requires nonempty docstrings.

## Preview and validate

From a source checkout:

```sh
python -m pip install -e ".[docs,test]"
python -m pytest tests/test_public_docs.py tests/test_tutorials.py -q
python -m mkdocs build --strict
python tests/check_docs_render.py
python -m mkdocs serve
```

`mkdocs serve` prints a local preview URL. The generated `site/` directory
is ignored by Git; only source docstrings and Markdown pages are committed.
The documentation workflow runs the checks on pull requests and on `main`.

Documentation dependencies are optional and are not installed by
`pip install rbflab`.

## Keep examples and equations synchronized

Tutorial pages embed their complete source from examples/tutorials with the
snippets extension. Edit the Python script first, run it, and then update the
explanation and recorded result. Add a small numerical assertion to
tests/test_tutorials.py for a meaningful invariant. Keep expensive refinement
runs behind an explicit script option.

Use inline math delimiters for symbols and display math blocks with a blank
line before and after each pair of dollar-sign delimiters. Inspect the built
HTML when changing math syntax or snippets. The post-build check catches
unexpanded snippets and several broken-math patterns.

For numerical results state cloud and node count, kernel and shape parameter,
polynomial degree, stencil size, boundary treatment, time scheme and step,
precision/backend, and whether an error is nodal, sampled off-node, or
semidiscrete. Report sampled RMS as RMS, not as a continuous L2 norm without
quadrature. Prefer links to reproducible script commands over screenshots
of numbers.

## Manual design

The chapter navigation is configured in mkdocs.yml. The visual design lives
in docs/stylesheets/manual.css, and the radial-node mark is an original SVG
in docs/assets/rbflab-mark.svg. Keep layout changes in that stylesheet so
theme upgrades remain simple. The home page uses relative site links.

Complete tutorial scripts are embedded in expandable example panels. Keep
short teaching fragments and equations visible in the main text. Test light
and dark themes, mobile chapter navigation, search, and equation rendering
in a browser after changing the theme or MathJax configuration.
MathJax uses full-page startup typesetting, including the boldsymbol extension;
enabling instant navigation would also require a navigation-aware callback.
