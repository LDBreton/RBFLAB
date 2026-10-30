# Equations, operators, and spaces

The scalar API accepts linear differential operators and tagged boundary conditions. The symbolic interfaces compile SymPy expressions into these numerical problems.

`SymbolicSystem` declares coupled scalar stationary fields. `SymbolicStokes` is
the specialized constant-viscosity momentum compiler with prescribed velocity
boundaries. Both models expose named symbolic fields for their PDE expressions;
those same fields can key the approximation spaces used by the solver.

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
u, p = model.fields
equation = -model.laplacian(u) + model.gradient(p)
space = rbf.MixedSpace({
    u: rbf.DivergenceFreeSpace(rbf.PHS(5), polynomial_degree=0),
    p: rbf.PressureSpace(rbf.PHS(5), polynomial_degree=None),
})
method = rbf.GlobalCollocation(spaces=space)
```

`SymbolicSystem` works the same way for scalar fields: unpack `model.fields`,
use those symbols in its stationary equations, and use them as keys in a
`MixedSpace`. `SymbolicStokes` fields retain vector velocity and scalar pressure
roles. The same fields also describe local linear spatial functionals:

```python
samples = {
    "momentum": rbf.Samples(points, operator=-nu * rbf.laplacian(u)
                            + rbf.gradient(p)),
    "pressure_gradient": rbf.Samples(points, operator=rbf.gradient(p)),
}
trial = space.representers(samples)
approximation = rbf.LocalApproximation(
    trial=trial, operators={"momentum": -nu * rbf.laplacian(u)
                            + rbf.gradient(p)},
)
```

These local operators are linear spatial functionals. Use the symbolic model
compiler for nonlinear or time-dependent PDE expressions; the local functional
API does not compile those PDEs.

This space choice is supported by Python assembly and the C++ MPFR global
backend. For C++ MPFR, pass `backend=rbf.CppGlobalBackend(...)` and
`precision=rbf.Precision(global_digits=...)` to `GlobalCollocation`. The pressure
space is modulo additive constants, so `None` means no additional pressure
polynomial columns; the pressure constant remains omitted as the gauge mode.
These options do not imply a general stability guarantee.

::: rbflab.ScalarSpace

::: rbflab.DivergenceFreeSpace

::: rbflab.PressureSpace

::: rbflab.MixedSpace

::: rbflab.FieldOperator

::: rbflab.Stencil

::: rbflab.SharedStencil

`Stencil` is the canonical fixed geometric selector. `SharedStencil` remains an
alias for compatibility. Use `Stencil(points, size=37)` to select a shared set
of physical sites, or `Stencil(points, indices=memberships)` for explicit
physical-site index rows (one row per target). `Samples(indices=...)` remains
available for explicit group-local memberships.

::: rbflab.laplacian

::: rbflab.gradient

::: rbflab.vector

::: rbflab.derivative

::: rbflab.dx

::: rbflab.dy

::: rbflab.dz

`u, p = space.fields` returns field expressions in the `MixedSpace` insertion
order. Write `-nu * laplacian(u) + gradient(p)` and pass it directly to
`Samples(operator=...)` or `LocalApproximation.operators`. Geometry supplies
the spatial dimension (2 or 3). Scalar and vector output shapes must agree
under addition. Coefficients are finite real constants; nonlinear products
and tensor-valued gradients are not supported. Expressions compile to the
existing `FieldOperator` blocks, whose explicit syntax remains available.

For direct local approximation, `PressureSpace` admits only constant-invariant
source, trial and target functionals, such as pressure derivatives. It never
inserts a constant polynomial, including when the requested degree is `None`.
Absolute nodal pressure data or value targets use `ScalarSpace`; any global
gauge belongs to the assembled equations. This resolves the earlier local-engine
behavior that silently added a constant and accepted absolute observations.

::: rbflab.EvolutionPDE

::: rbflab.SymbolicStokes

::: rbflab.CenterGroup
