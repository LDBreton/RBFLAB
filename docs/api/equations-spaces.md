# Equations, operators, and spaces

The scalar API accepts linear differential operators and tagged boundary conditions. The symbolic interfaces compile SymPy expressions into these numerical problems.

`SymbolicSystem` declares coupled scalar stationary fields. `SymbolicStokes` is
the specialized constant-viscosity momentum compiler with prescribed velocity
boundaries; it does not claim arbitrary mixed vector PDE support.

::: rbflab.SymbolicScalar

::: rbflab.SymbolicSystem

::: rbflab.LinearPDE

::: rbflab.BoundaryCondition

::: rbflab.Dirichlet

::: rbflab.Identity

::: rbflab.Derivative

::: rbflab.Laplacian

::: rbflab.NormalDerivative

::: rbflab.Robin

::: rbflab.SpatialOperator

Spaces also describe explicit local trials through `.representers(source)` and
`.translates(points)`. See [functional approximation](../tutorials/local-approximation.md).

Polynomial augmentation is selected independently on each space. The default
`polynomial_degree=None` omits polynomial columns; an explicit `None` does the
same. An integer requests exactly that total degree, including degrees below a
kernel's conventional CPD minimum. Such a choice may make the augmented system
singular or less accurate; RBFLAB reports diagnostics but does not reject the
degree solely for being below that minimum. Use `polynomial_degree="auto"` to
select the kernel minimum. This controls basis construction, not approximation
quality or stability.

For example, constant-only solenoidal velocity polynomials and no pressure
polynomial tail can be selected independently:

```python
import rbflab as rbf

model = rbf.SymbolicStokes(2)
spaces = {
    model.velocity: rbf.DivergenceFreeSpace(rbf.PHS(5), polynomial_degree=0),
    model.pressure: rbf.PressureSpace(rbf.PHS(5), polynomial_degree=None),
}
method = rbf.GlobalCollocation(spaces=spaces)
```

This space choice is supported by Python assembly and the C++ MPFR global
backend. For C++ MPFR, pass `backend=rbf.CppGlobalBackend(...)` and
`precision=rbf.Precision(global_digits=...)` to `GlobalCollocation`. The pressure
space is modulo additive constants, so `None` means no additional pressure
polynomial columns; the pressure constant remains omitted as the gauge mode.
These options do not imply a general stability guarantee.

::: rbflab.ScalarSpace

::: rbflab.DivergenceFreeSpace

::: rbflab.PressureSpace

::: rbflab.EvolutionPDE

::: rbflab.SymbolicStokes

::: rbflab.CenterGroup
