# RBFLAB 0.2.0 integration validation

The RBFMeshGen integration was validated on 2026-10-08 before publication.

- Standard suite: 158 passed, 20 opt-in native skips, 42 unittest subtests.
- Native-enabled targeted suite: 49 passed initially; three stale private-fixture
  imports were replaced with the public Stokes tutorial and all three reruns passed.
  Coverage includes C++ Float64/MPFR symbolic compilation, 3D vector operators,
  and matched-cloud C++/PyTorch scalar results.
- Both owned Eigen/OpenMP executables built successfully on Ubuntu WSL.
- All four curved scalar examples ran with Python, C++ Float64 and PyTorch CPU
  Float64 using identical clouds and numerical settings.
- Annulus refinement improved across seeds 17, 42 and 73 with fixed stencil/kernel
  settings. Recorded errors and timings are in docs/assets/curved_validation.json.
- Strict MkDocs build and local-link/style checks passed on 42 pages. Browser
  preview confirmed the new homepage and rendered annulus mathematics.
- Wheel and sdist passed twine checks. An isolated Windows Python 3.12 environment
  installed the wheel with core dependencies only and ran the curved PDE examples,
  existing Poisson/Stokes checks, and pip check. Optional imports remain lazy.
- The wheel includes both MIT license notices. All six imported source-file hashes
  still match the original RBFMeshGen checkout; its local files were not modified.

Release CI additionally checks Windows/Linux wheels, geometry/Gmsh, optional
C++ and PyTorch, generated figures, and the documentation deployment.

---

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

### Retrying a failed upload workflow

If a release build fails before upload, fix CI on `main` and use the manual
`publish to PyPI` workflow with the **existing release tag**. It checks out and
builds that immutable tag, verifies its version, and retains the protected
`pypi` environment. Never move a published tag or replace a PyPI version.
The installed-package test step sets `RBFLAB_SOURCE_ROOT` to its matching
checkout because native sources are intentionally absent from the wheel.
