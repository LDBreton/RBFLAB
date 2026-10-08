# RBFLAB

RBFLAB is a Python library for radial basis function interpolation and PDEs on
point clouds. It supports global collocation, local Hermite interpolation
(LHI), RBF-FD, symbolic linear operators and boundary conditions, and
divergence-free velocity spaces. Scalar operators work in 2D and 3D.

Source and examples: https://github.com/LDBreton/RBFLAB

Install the Python core with:

```sh
python -m pip install rbflab
```

Optional extras include `rbflab[examples]` for Matplotlib plots and GIFs,
`rbflab[mesh]` for Gmsh, and `rbflab[torch]` for the documented CPU Float64
tensor backend. The C++ double/MPFR backends are source-build options.

Here is a symbolic Poisson problem:

```python
import sympy as sp
import rbflab as rbf

model = rbf.SymbolicScalar(2)
u = model.field
x, y = model.coordinates
exact = sp.sin(sp.pi*x) * sp.sin(sp.pi*y)
lhs = -model.laplacian(u)
problem = model.stationary(
    sp.Eq(lhs, lhs.subs(u, exact).doit()),
    boundary=[model.bc("boundary", sp.Eq(u, 0))],
)
cloud = rbf.unit_box_grid(5)
solution = problem.solve(
    cloud, rbf.GlobalCollocation(rbf.IMQ(2), scheme="asymmetric")
)
print(solution.evaluate([[0.3, 0.4]]))
```

The source repository also contains short heat, Stokes, 3D operator, custom
kernel, and Re=100 lid-driven cavity examples. The cavity example is a
showcase with a documented nonzero discrete divergence defect; it does not
validate a general Navier–Stokes solver. LHI off-node reconstruction has its
own accuracy limitations, and MPFR local weights do not automatically make a
global sparse solve extended precision.

RBFLAB is licensed under MIT.
