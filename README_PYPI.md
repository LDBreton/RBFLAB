# RBFLAB

### From geometry to equations, with radial basis functions.

Build labeled **2D and 3D point clouds**, write equations with **SymPy**, and choose
**global collocation**, **local Hermite interpolation** or **RBF-FD**. Inspect local
weights and sparse matrices when you want to go beyond the high-level API.

[**Read the manual**](https://ldbreton.github.io/RBFLAB/) ·
[2D shapes and holes](https://ldbreton.github.io/RBFLAB/geometry/planar-domains/) ·
[Mathematical foundations](https://ldbreton.github.io/RBFLAB/theory/) ·
[API reference](https://ldbreton.github.io/RBFLAB/api/discretizations/)

![Actual generated clouds: obstacles, curved holes, a concave polygon and a 3D volume](https://raw.githubusercontent.com/LDBreton/RBFLAB/v0.4.0/docs/assets/geometry_gallery.png)

## Install and start

```sh
python -m pip install rbflab
# Optional plots and animations:
python -m pip install "rbflab[examples]"
```

The Python core needs no compiler, Gmsh or tensor framework. Geometry generation
is built in. See the [installation guide](https://ldbreton.github.io/RBFLAB/INSTALL/)
for optional backends and exact dependencies. The new polygon and hole-composition
examples below require **RBFLAB 0.4 or newer**.

## One domain. One equation. A numerical solution.

Solve Poisson's equation on an ellipse with two holes. Here a known solution,
`sin(x)*cos(y)`, supplies boundary data and lets us check the numerical error.

```python
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen

domain = geometry.with_holes(
    geometry.Ellipse(1.6, 1., labels=("wall",)),
    {"round_hole": geometry.Disk(.28, center=(-.65, 0)),
     "slot": geometry.Ellipse(.24, .43, center=(.6, 0), angle=-.3)},
)
cloud = meshgen.generate(domain, interior=500,
    boundary={"wall": 120, "round_hole": 32, "slot": 40}, seed=42)

model = rbf.SymbolicScalar(2)
u = model.field
x, y = model.coordinates
exact = sp.sin(x)*sp.cos(y)
problem = model.stationary(
    sp.Eq(-model.laplacian(u), 2*exact),
    boundary=[model.bc(label, sp.Eq(u, exact)) for label in domain.boundaries],
)
method = rbf.RBFFD(rbf.PHS(5), 35, polynomial_degree=3,
    stencil_policy=rbf.StencilPolicy(scaling="local"),
    local_backend=rbf.PythonBackend(compute_condition=False))
solution = problem.solve(cloud, method)
print(solution.evaluate([[0., 0.], [0., .5]]))
```

![Generated labeled nodes and the computed solution on a perforated ellipse](https://raw.githubusercontent.com/LDBreton/RBFLAB/v0.4.0/docs/assets/perforated_poisson.png)

With this 692-node recipe, the Float64 Python run gives maximum nodal error
**3.77 × 10⁻⁵** and independent off-node error **3.86 × 10⁻⁵**. Read the
[step-by-step tutorial](https://ldbreton.github.io/RBFLAB/tutorials/perforated-poisson/)
for the mathematics, plotting, and checks on independent clouds.

Plotting stays short:

```python
from rbflab import viz
fig, ax = viz.plot_scalar(solution, domain=domain, title="Poisson / two holes")
fig.savefig("solution.png", dpi=180)
```

## Construct your own numerical method

The development API exposes **sampled functionals, trial functions and named
sparse blocks** through `LocalApproximation`. Ordinary RBF-FD and local Hermite
interpolation use the same mathematical construction. Assemble stiffness, mass,
forcing and boundary contributions yourself, then write a SciPy solve or time
loop. Existing PDE convenience interfaces remain available.

- [From functionals to operators](https://ldbreton.github.io/RBFLAB/tutorials/local-approximation/)
- [LHI and heat from matrices](https://ldbreton.github.io/RBFLAB/tutorials/lhi-matrices/)
- [Architecture and research API](https://ldbreton.github.io/RBFLAB/guides/research-api/)

These additions are unreleased; use the current source checkout. The installed
PyPI release may not yet provide these names.

## See the methods at work

<table>
<tr>
<td width="50%"><a href="https://ldbreton.github.io/RBFLAB/tutorials/flower-heat/"><img src="https://raw.githubusercontent.com/LDBreton/RBFLAB/v0.4.0/docs/assets/flower_heat.gif" alt="Computed heat evolution on a flower-shaped domain"></a></td>
<td width="50%"><a href="https://ldbreton.github.io/RBFLAB/tutorials/annular-stokes/"><img src="https://raw.githubusercontent.com/LDBreton/RBFLAB/v0.4.0/docs/assets/method_stokes.png" alt="Computed velocity streamlines between rotating cylinders"></a></td>
</tr>
<tr>
<td><b>Heat on a flower</b><br>Symbolic forcing, BDF2 time stepping and animation.</td>
<td><b>Stokes between cylinders</b><br>Divergence-free velocity spaces on a curved domain.</td>
</tr>
</table>

| Explore | Learn to use |
|---|---|
| [2D shapes and holes](https://ldbreton.github.io/RBFLAB/geometry/planar-domains/) | Named walls, holes, concave polygons and 3D volumes |
| [Interpolate data](https://ldbreton.github.io/RBFLAB/tutorials/interpolation/) | Arrays, interpolants and field derivatives |
| [Your own PDE assembly](https://ldbreton.github.io/RBFLAB/tutorials/custom-assembly/) | Combine RBF maps with ordinary SciPy algebra |
| [Teach with RBFLAB](https://ldbreton.github.io/RBFLAB/tutorials/teaching/) | Lesson sequences and mathematical modification exercises |
| [One local stencil](https://ldbreton.github.io/RBFLAB/tutorials/one-stencil/) | Approximation, differentiation weights and sparse rows |
| [Heat from matrices](https://ldbreton.github.io/RBFLAB/tutorials/heat-equation/) | RBF spatial matrices and explicit time-integration code |
| [Mixed conditions](https://ldbreton.github.io/RBFLAB/tutorials/ellipse/) | Dirichlet, Neumann and Robin data on curved boundaries |
| [Compare methods](https://ldbreton.github.io/RBFLAB/tutorials/global-lhi/) | Global collocation and local Hermite interpolation |
| [Poisson in a ball](https://ldbreton.github.io/RBFLAB/tutorials/ball/) | 3D scalar operators and independent error checks |
| [Custom kernels](https://ldbreton.github.io/RBFLAB/tutorials/custom-kernel/) | Symbolic kernels, derivatives and optional C++ compilation |
| [Navier–Stokes cavity](https://ldbreton.github.io/RBFLAB/tutorials/cavity/) | An experimental staggered-flow showcase with documented limitations |

## Build a domain from labeled curves

Symbolic curves derive tangents and outward normals automatically:

```python
import sympy as sp
from rbflab import geometry, meshgen

t = sp.symbols("t", real=True)
outer = geometry.Border((sp.cos(t), sp.sin(t)), "outer", 0, 2*sp.pi)
hole = geometry.Border((sp.cos(t)/3, sp.sin(t)/3), "hole", 0, 2*sp.pi)
cloud = meshgen.generate([outer(96), hole(-48)], interior=240, seed=42)
```

Ordinary callables accept optional tangents and explicit normals, with a documented
numerical derivative fallback. The [mesh-generation chapter](https://ldbreton.github.io/RBFLAB/geometry/)
explains both curve APIs, boundary labels, sampling, 3D volumes/surfaces and staggered
clouds, with images of the constructions. These unified curve features require
**RBFLAB 0.4 or newer**.

## Choose a backend; keep the problem

| Backend | Installation | Role |
|---|---|---|
| Python / NumPy / SciPy | `pip install rbflab` | Default numerical implementation |
| C++ / Eigen | [Source-build setup](https://ldbreton.github.io/RBFLAB/INSTALL/) | Supported local assembly in Float64 or MPFR |
| PyTorch | `pip install "rbflab[torch]"` | Supported local tensor assembly; documented CPU Float64 path |
| Gmsh (optional geometry) | `pip install "rbflab[mesh]"` | Tagged meshes and explicit staggered layouts |

From a source checkout, run the same example with a different local backend:

```sh
python -m examples.poisson_perforated --backend python --plot solution.png
python -m examples.poisson_perforated --backend cpp
python -m examples.poisson_perforated --backend torch
```

The example prepares compiled/tensor kernels before assembly. Its global sparse
solve uses SciPy Float64. Backend coverage and precision differ by method; see
[capabilities](https://ldbreton.github.io/RBFLAB/CAPABILITIES/). Small problems do
not necessarily run faster with extra threads or a tensor backend.

## Research tools with explicit limits

- Use `PointCloud` with your own coordinates, or generate labeled nodes in RBFLAB.
  The sampler does not enforce minimum inter-node separation.
- Access discrete operators, sparse matrices, local weights and reconstruction
  diagnostics. Keep algebraic residuals separate from solution and PDE errors.
- Divergence-free Stokes spaces build incompressibility into the velocity kernel.
  LHI pressure-gradient reconstruction and nearest-stencil off-node evaluation
  have their own accuracy limits.
- Extended-precision local weights do not automatically make a global sparse
  solve extended precision. The cavity showcase is not a validated general
  Navier–Stokes solver.

## Contribute and reproduce

Source code lives in `python/rbflab`, runnable examples in `examples`, and focused
regression tests in `tests`. Gallery figures are reproducible from
[geometry_gallery.py](https://github.com/LDBreton/RBFLAB/blob/v0.4.0/examples/geometry_gallery.py) and
[make_method_gallery.py](https://github.com/LDBreton/RBFLAB/blob/v0.4.0/examples/make_method_gallery.py).

[Maintaining the documentation](https://ldbreton.github.io/RBFLAB/MAINTAINING_DOCS/) · [Changelog](https://github.com/LDBreton/RBFLAB/blob/v0.4.0/CHANGELOG.md) · [MIT license](https://github.com/LDBreton/RBFLAB/blob/v0.4.0/LICENSE)
