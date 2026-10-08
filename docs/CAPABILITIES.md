# Verified capabilities for RBFLAB 0.2.0

The method and backend combinations below are intentionally precise. A local
backend choice does not automatically change a separate global sparse solve.

| Capability | Python | C++ double | C++ MPFR | PyTorch |
|---|---|---|---|---|
| Scalar global collocation | Yes | — | — | — |
| Scalar LHI | Yes | — | — | — |
| Scalar RBF-FD operators, 2D/3D | Yes | Yes, from source checkout | Supported; installed archive test pending | Yes, CPU Float64 |
| Divergence-free steady Stokes, global | Yes | — | — | — |
| Divergence-free Stokes LHI, 2D | Yes | Yes, from source archive | 50 digit local weights verified from source archive | Unaugmented stencils only; not the curated example |
| Custom symbolic kernel evaluation | Yes | Compilation from source checkout | Compilation from source checkout | Supported where the local backend accepts the kernel |
| Full MPFR global sparse solve | Supported by selected Python methods | Separate from local C++ backend | Separate setting | — |

The curated [3D operator example](https://github.com/LDBreton/RBFLAB/blob/main/examples/operators_3d.py) ran with Python,
C++ double, and PyTorch Float64 on Windows/WSL. The
[Stokes example](https://github.com/LDBreton/RBFLAB/blob/main/examples/stokes_spaces.py) ran with Python, C++ double,
and C++ MPFR local weights.
The six default Python examples ran from an installed wheel and source archive
outside the checkout.

RBFMeshGen generation is integrated into `rbflab.geometry` and `rbflab.meshgen`; Gmsh remains optional. The built-in
`unit_box_grid` supplies a small 2D/3D deterministic cloud for tutorials.


Curved-domain scalar examples were compared on identical clouds with Python,
C++ Float64 and PyTorch CPU Float64. See [the validation record](guides/curved-validation.md).
Core geometry includes 2D parametric domains/holes, 3D implicit regions and
parametric surfaces. Staggering remains explicit 2D triangle-based geometry.
