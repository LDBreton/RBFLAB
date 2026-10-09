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

::: rbflab.ScalarSpace

::: rbflab.DivergenceFreeSpace

::: rbflab.PressureSpace

::: rbflab.EvolutionPDE

::: rbflab.SymbolicStokes

::: rbflab.CenterGroup
