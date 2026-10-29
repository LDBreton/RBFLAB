# Changelog

## 0.6.0 — mixed local approximation and shared Stokes weights (unreleased)

- Add `MixedSpace` and field-block `FieldOperator` to `LocalApproximation`, with independent source/target arities, named sparse maps, degree-generated scalar/solenoidal polynomial tails, and inspected local algebra.
- Support Python Float64/mpmath and native C++ Float64/MPFR mixed Hermite construction, with real MPI scheduling and exact identical-request reuse through `LocalExecution`.
- Route existing LU Stokes LHI weight construction through `LocalApproximation`, preserving stencil memberships, kernel policies, quotient-pressure augmentation, compact matrices and local pressure-gradient reconstruction. Global PDE assembly and time integration retain their existing roles.
- Add `LHI(..., execution=LocalExecution(...))` for supported Stokes LU weight scheduling. MPI requires C++; the global sparse solve remains separate. SVD and experimental backends retain their explicit specialized paths.
- Correct direct `PressureSpace` handling: omit constant polynomial columns and require constant-invariant local samples/trials/targets; absolute values use `ScalarSpace`. Permit quotient-pressure factors in mixed Hermite requests with the same invariant-functional checks.
- Preserve the historical normalized streamfunction-curl basis for generic 2D solenoidal polynomials; 3D uses the solenoidal nullspace basis.
- Add saved-weight, compact-matrix, native/MPI parity checks and a runnable generic coupled-field example. These checks do not establish cavity stability or a globally reconciled pressure field.
- Fix mixed quality selection to grow eligible sample groups against the joint functional polynomial matrix; preserve explicit memberships and nearest-stencil behavior.
- Enable target-centered local kernel scaling for standard C++ LU Stokes weights and MPI, preserving the Python method's radius and derivative chain factors. Native SVD/custom-executable paths and geometry-dependent shape rules retain physical-coordinate restrictions.
- Document the existing radius conventions for single-field and mixed/Stokes local scaling. Add mixed/Stokes contracts to Python CI, native/MPI parity to the Linux native job, and the generic mixed example to installed-wheel smoke checks.
- Add short, numbered public examples for coupled scalar maps, one 37-point Stokes stencil, and a complete local Stokes solve. Keep the checkpointed measured driver in the research tree.

This source-version change does not publish a package or create a release tag.

## 0.5.0 — functional approximation and matrix algorithms

- Support unsteady divergence-free global Stokes with `CppGlobalBackend`: native stiffness/mass assembly, backward Euler/BDF2 shifted LU caching and mass products in double/MPFR, with public time-dependent data and field APIs.
- Extend `CppGlobalBackend` to stationary symmetric divergence-free Stokes fields in 2D/3D, with double/MPFR arithmetic, separate velocity/pressure kernels, solenoidal polynomial tails and native field evaluation.
- Add `CppGlobalBackend` for scalar stationary global collocation in 2D/3D, with double/MPFR assembly, polynomial augmentation, persistent dense LU solves and native evaluation for both schemes.
- Optimize native global double kernels and symmetric assembly; reuse validated artifacts and bounded idle workers (`reuse_workers=False` opts out). Add reproducible timing/accuracy benchmarks with saved-run comparisons.
- Add `Samples` and `LocalApproximation`, with explicit space representers and translates.
- Expose named source blocks, target operators, local weights and on-demand local algebra.
- Add executable tutorials constructing scalar LHI, its stationary system and heat mass matrix with ordinary SciPy code.
- Migrate the differentiation, custom-kernel, one-stencil, custom-assembly, independent LHI centers, reconstruction and heat-matrix lessons to the functional API.
- Test the new examples against installed wheels; distinguish matrix construction from optional symbolic PDE adapters in the manual.
- Route standard `RBFFD.operators`/`.weights` and scalar LHI weight assembly through the shared functional engine; preserve convenience assembly/reconstruction and the RBF-FD SciPy result contract.
- Support scalar and componentwise divergence-free functional construction in 2D/3D, with Python, C++ Float64/MPFR and Torch backend checks.
- Retain specialized coupled-block and Stokes compatibility implementations; general mixed-field symbolic assembly is not added.
- Document the shared numerical architecture, backend limits, target exclusion and independent-center evolution maps.

### Research API consolidation (breaking changes from 0.4)

- Canonical `interpolate`, `GlobalCollocation`, `RBFFD`, and `LHI` entry points.
- `RBFFD.weights` replaces the root `rbf_fd_weights`; explicit centers retain order.
- `SymbolicStokes` explicitly identifies the specialized vector Stokes compiler.
- Numerical shape policy belongs to `StencilPolicy`; local factorization belongs to `LocalSolver`.
- `CenterGroup` exposes independent scalar stationary LHI solution, PDE, boundary and derivative-data centers.
- Shared preflight, prepared copies, system recipes/DOF descriptions and matrix-cache safeguards.
- Root compatibility/experimental exports removed; geometry and diagnostics use named modules.
- Stokes LHI matrix edits are rejected because dependent row maps require reassembly.

See the [research API guide](docs/guides/research-api.md) and
[API decision record](docs/development/API_CONSOLIDATION.md).


## 0.4.0

- Unify generation of primitive domains, oriented borders, RBFMesh constructions, implicit volumes and parametric surfaces under `meshgen.generate`.
- Add shared symbolic curve derivatives, optional explicit tangents, endpoint-safe numerical differentiation, and explicit normal overrides for `Border` and `ParametricBoundary`.
- Preserve boundary/interface labels and automatically orient border-derived normals without changing input samples.
- Replace the geometry chapter with illustrated construction sections for 2D, labels, normals, sampling, 3D volumes, surfaces and mesh topology.
- Document primitive arguments, partial-boundary labeling and staggered clouds; keep older importers and inbound documentation URLs compatible.


## 0.3.0

- Add labeled `geometry.Polygon`, rotated `Rectangle`, and validated `with_holes` composition.
- Add a geometry cookbook, generated node gallery, and a symbolic Poisson example on an ellipse with two holes.
- Redesign the README around geometry, equations, numerical results and backend choices.
- Remove the RBFMeshGen migration entry from primary navigation; retain its historical URL.
- Serialize documentation deployments to avoid overlapping GitHub Pages jobs.


## 0.2.0

- Integrated RBFMeshGen geometry, random/Halton/Sobol generation, 3D implicit
  regions, parametric surfaces, and explicit 2D staggered layouts.
- Added a shared PointCloud contract with interfaces and region metadata,
  labeled curved domains, analytic normals, and geometric quality diagnostics.
- Added annulus, flower heat, mixed ellipse, annular Stokes and 3D ball tutorials;
  local scalar examples select Python, C++ or PyTorch backends.
- Added domain-masked plots/animations, node/stencil views and 3D slices.
- Fixed C++ source discovery in the public python/ package layout.
- Preserved old RBFLAB imports and the legacy mesh adapter; Gmsh, visualization
  and tensor dependencies remain optional. Shapely joins core dependencies.

## 0.1.0

- Curated six symbolic and numerical starting examples.
- Added `unit_box_grid` for small 2D/3D problems without Gmsh.
- Documented installation, optional backends, precision stages, and current
  capability limits.
- Added optional 2D field plots and GIF animation, with reproducible README
  images generated from numerical heat and Stokes solutions.
- Added a standalone Re=100 staggered RBF-FD Navier–Stokes cavity example and
  an optional nodal-velocity GIF/streamline plotting helper.
