# RBFLAB 0.1.0 release record

RBFLAB 0.1.0 was published on 2026-10-08 from a fresh public source snapshot.
The private research repository and its history were not published.

- Source: [LDBreton/RBFLAB](https://github.com/LDBreton/RBFLAB), tag
  [`v0.1.0`](https://github.com/LDBreton/RBFLAB/releases/tag/v0.1.0), commit
  `7406f59b7fa6c9b68bcc97a94a038eb17fccfe21`.
- Package: [rbflab 0.1.0](https://pypi.org/project/rbflab/0.1.0/), MIT license
  with Louis Breton as copyright holder.
- The user confirmed rights to release the selected code and generated images
  under MIT. The repository includes focused source, examples, and regenerated
  README media; it excludes legacy reference code, experiments, and results.
- [Release smoke CI](https://github.com/LDBreton/RBFLAB/actions/runs/37828612836)
  passed six jobs: installed-wheel examples on Windows and Ubuntu with Python
  3.11/3.12, native C++ checks, and gallery generation.
- [PyPI publishing CI](https://github.com/LDBreton/RBFLAB/actions/runs/37829569165)
  passed the version/tag check, focused tests, distribution build, `twine check`,
  and Trusted Publisher upload. The `pypi` GitHub environment required a
  reviewer; no API token was stored in the repository.
- PyPI wheel SHA-256:
  `cf4551fbefdf9a29dfe6af313a8e11fa69c7cb315757198cc9b384f9863f6ec0`.
- PyPI source archive SHA-256:
  `cd730981456e000e79e77a1fe83ef843279e7a558994f5cfcb907f933ef38e00`.
- A fresh Windows Python 3.12 installation from the public index ran the
  symbolic Poisson and divergence-free Stokes examples; `pip check` passed.

The C++ double and MPFR backends remain source-build options. The cavity
example is a showcase with a documented divergence defect, not validation of
a general Navier–Stokes solver.
