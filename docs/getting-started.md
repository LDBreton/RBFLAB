# Getting started

Install the core package:

```sh
python -m pip install rbflab
```

This example defines a Poisson problem from its exact solution, assembles a global RBF collocation system, and evaluates the result:

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
method = rbf.GlobalCollocation(rbf.IMQ(2), scheme="asymmetric")
system = method.assemble(problem, cloud)
solution = system.solve()
print(solution.evaluate([[0.3, 0.4]]))
```

A method assembles a **system** containing the numerical matrix and right-hand side. Solving it returns a **solution** with evaluation and diagnostics. To compare discretizations, keep the problem and cloud and change the method:

```python
method = rbf.LHI(rbf.PHS(5), 20, polynomial_degree=2)
# or: method = rbf.RBFFD(rbf.PHS(5), 20, polynomial_degree=2)
```

Stencil size must fit the cloud. RBF-FD and LHI produce different local systems, so compare their errors as well as their algebraic residuals. The runnable [Poisson example](https://github.com/LDBreton/RBFLAB/blob/main/examples/symbolic_poisson.py) reports errors against the known solution.

See the [equations and spaces](api/equations-spaces.md) and [discretizations](api/discretizations.md) references for other choices. Current 2D Stokes examples use a divergence-free velocity kernel; [capabilities](CAPABILITIES.md) describes the available scope.
