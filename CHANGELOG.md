# Changelog

## Unreleased — research API consolidation (breaking)

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
