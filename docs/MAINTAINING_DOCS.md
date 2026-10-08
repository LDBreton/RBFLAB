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
python -m pytest tests/test_public_docs.py -q
python -m mkdocs build --strict
python -m mkdocs serve
```

`mkdocs serve` prints a local preview URL. The generated `site/` directory
is ignored by Git; only source docstrings and Markdown pages are committed.
The documentation workflow runs the checks on pull requests and on `main`.

Documentation dependencies are optional and are not installed by
`pip install rbflab`.
