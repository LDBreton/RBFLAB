# Changelog

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
