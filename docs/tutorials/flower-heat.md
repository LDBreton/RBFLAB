# Heat diffusion on a flower

The flower example is now the opening worked case of
[Heat diffusion with RBF-FD](heat-equation.md). That chapter connects its symbolic
PDE to local weights, sparse Laplacian blocks, boundary elimination and time stepping.

![Computed heat diffusion on a flower domain](../assets/flower_heat.gif)

The verification field is $u=e^{-2t}(\cos x\cos y+\tfrac14\sin(2x))$.
Run `python -m examples.heat_flower --animation flower.gif` from a source checkout.
