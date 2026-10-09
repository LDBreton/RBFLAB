# Heat on a flower-shaped domain

Apply the symbolic workflow to a curved domain with nonzero boundary data.
Read [the heat construction](heat-equation.md) first for the time discretization.

## 1. Start with the equation and a curved domain

For a diffusivity $\kappa>0$,

$$u_t-\kappa\Delta u=f,\qquad u|_{\partial\Omega}=g(t),\qquad u(\cdot,0)=u_0.$$

The flower example uses $\kappa=0.15$ and the known solution

$$u_{\rm exact}(x,y,t)=e^{-2t}\left(\cos x\cos y+\tfrac14\sin(2x)\right).$$

The forcing is $f=\partial_tu_{\rm exact}-\kappa\Delta u_{\rm exact}$, and the boundary
values are its trace. This is a forced verification case, not an insulated heat pulse.
Here is the complete core calculation, without example-helper functions:

```python
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen

domain = geometry.Flower()
cloud = meshgen.generate(domain, interior=250, boundary=120, seed=42)
model = rbf.SymbolicScalar(2, transient=True)
u, t = model.field, model.time
x, y = model.coordinates
exact = sp.exp(-2*t)*(sp.cos(x)*sp.cos(y) + sp.sin(2*x)/4)
lhs = sp.diff(u,t) - sp.Rational(15,100)*model.laplacian(u)
problem = model.evolution(
    sp.Eq(lhs, lhs.subs(u,exact).doit()), initial=exact.subs(t,0),
    boundary=[model.bc("boundary", sp.Eq(u,exact))],
)
method = rbf.RBFFD(rbf.PHS(5), 35, polynomial_degree=3,
    stencil_policy=rbf.StencilPolicy(scaling="local"))
trajectory = problem.solve(cloud, method, "0.025", 20, scheme="bdf2")
```

The 370-node recipe uses PHS5, degree-three polynomials, 35-node stencils and
Float64. Here `scaling="local"` divides kernel distances by each stencil's
radius; RBFLAB includes the inverse-square factor for the physical Laplacian.
See [local stencil scaling](../theory/conditioning.md#local-stencil-scaling).
Backward Euler starts BDF2. To animate the computed solution:

```python
from rbflab import viz
viz.animate_scalar(trajectory, "flower.gif", domain=domain)
```

The display grid does not enter the PDE discretization. The recorded final
sampled off-node error at $t=0.5$ is about $1.1\times10^{-3}$; it contains both
space and time error. Reproduce the error report with
`python -m examples.heat_flower` from a source checkout.


![The flower cloud and computed heat field](../assets/flower_heat.gif)

The standalone code above uses the same recipe as `python -m examples.heat_flower`.
The original maintained example also supports `--backend cpp`, `--backend torch`,
and `--animation flower.gif`.

??? example "Complete example with plotting and backend options"

    ```python
    --8<-- "examples/heat_flower.py"
    ```
