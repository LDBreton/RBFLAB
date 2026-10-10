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

## Assets, references, and browser caches

The MkDocs hook in scripts/docs_hooks.py adds content hashes to custom CSS and
MathJax configuration URLs on every page, including nested tutorials. This
makes browsers request updated assets when their content changes. Previously
opened HTML pages may still need a reload after a deployment. The generated-site
check verifies shared styling and local links across all navigation pages.

Regenerate the original stencil figures with
`python -m examples.make_stencil_figures` after installing the examples extra.
The RBF-FD figure computes actual weights and checks constant and quadratic
reproduction; the LHI figure is explicitly a schematic. SVGs remain sharp
when zoomed, and PNG copies are available for reuse. Do not copy figures from
papers without appropriate permission.

Keep paper metadata and DOI links in theory/references.md, and explain each
citation's relevance on the corresponding theory page. Distinguish mathematical
background from algorithms that the public package actually implements.


## Geometry integration

The submodules `geometry` and `meshgen` expose their own `__all__` lists. Document
those names in api/mesh-generation.md; test_public_docs checks them. Run the
ported geometry regression suite in tests/meshgen and the new geometry/PDE tests.
Generate assets with `python -m examples.make_geometry_gallery --backend cpp`.
Curved examples embed their source directly; common measurement/backend helpers
live in examples/_curved.py. Keep claims synchronized with curved_validation.py.

## Foundations and method cards

Keep notation consistent with theory/notation.md. H_i denotes a local augmented
interpolation system; A_h denotes the assembled PDE system. Global and LHI have
separate chapters; theory/global-lhi.md remains a compatibility landing page.
The transpose-solve derivation lives in theory/local-weights.md.

Generate the homepage method cards with `python -m examples.make_method_gallery`
(or `--backend cpp` for native cavity assembly). The images are numerical fields,
not substitutes for validation plots. method_gallery.json records the recipes
and diagnostics. Keep their captions and tutorial links aligned with the actual
example; no image should imply a benchmark that has not been checked.

## Curated learning route

Keep a single Examples chapter. Select one leading geometry/problem for each
concept, with foundational operator lessons before coupled flow. Supplementary
regression pages can remain linked without expanding the main navigation.
Heat teaches RBF matrices; its source no longer includes classical five-point FD.
Cavity equations must match the maintained solver, including the compatibility
multiplier and the difference between algebraic residual and divergence defect.
Named snippet markers expose the actual implementation, not a duplicate pseudocode
solver. Run the first-problem, heat and cavity tests after changing these sections.

## Mathematics-to-code teaching route

The Examples chapter serves RBF researchers adapting methods and instructors
teaching students. Its paths cover data approximation, PDE construction, and
custom algorithms. Keep the main route focused on writing mathematics with the
API; diagnostics and convergence studies remain supporting guides.

For each main lesson:

1. State the mathematical task and prerequisites.
2. Introduce arrays, symbols and their shapes before using them.
3. Pair each mathematical construction with a short visible code snippet.
4. Explain the object returned and how to use or modify it.
5. Include a brief useful check and a complete runnable source at the end.

Use named source snippets rather than duplicate teaching implementations in
Markdown. Tests in test_tutorials check matrix meaning and agreement between
API routes. Do not imply that inspection records are solver-editing hooks.
Keep source/target ordering and coefficient-versus-value distinctions explicit.

The new teaching illustrations are regenerated with
`python -m examples.make_teaching_figures`. It executes the same named snippet
sections embedded in the manual, so figures show the actual sample memberships,
operators and solutions. The generated `docs/assets/tutorial_figures.json`
records normalized source hashes; the render check rejects stale figures.
Use square domains in the main route, and keep other applications in the gallery. Course
sequences and modification exercises live in tutorials/teaching.md. Avoid exposing
private planning notes or historical implementation discussions in the manual.
