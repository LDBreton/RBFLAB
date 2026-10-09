# Learn through examples

Choose a problem, see the equations and the computed result, then inspect the
operators that produced it. The sequence below moves from local approximation to
scalar PDEs, time stepping and coupled flow. Geometry is part of each problem;
curved domains do not need a separate tutorial collection.

Start with [your first RBF problem](../getting-started.md) if the problem/method/system
interface is new. Use the [mesh-generation chapter](../geometry/index.md) when you
want to construct a domain without solving a PDE.

## From a field to derivative weights

| Tutorial | What you build | What you check |
|---|---|---|
| [Interpolation and derivatives](interpolation.md) | A kernel interpolant and derivatives | Error against a known field |
| [One RBF-FD stencil](one-stencil.md) | Weights from a local interpolation system | Polynomial reproduction and the local residual |

## Solve, time-step, and compare

<div class="rbf-examples">
<article class="rbf-example"><a href="perforated-poisson/"><img src="../assets/perforated_poisson.png" alt="Poisson solution on an ellipse with two holes" loading="lazy"></a><div class="rbf-example-body"><h3><a href="perforated-poisson/">Poisson with holes</a></h3><p>Named boundaries, symbolic forcing and sampled error.</p></div></article>
<article class="rbf-example"><a href="heat-equation/"><img src="../assets/flower_heat.gif" alt="Computed transient temperature on a flower domain" loading="lazy"></a><div class="rbf-example-body"><h3><a href="heat-equation/">Heat diffusion</a></h3><p>From a symbolic evolution problem to RBF-FD matrices and BDF2.</p></div></article>
<article class="rbf-example"><a href="ellipse/"><img src="../assets/ellipse_boundary.png" alt="Mixed boundary conditions on a labeled ellipse" loading="lazy"></a><div class="rbf-example-body"><h3><a href="ellipse/">Mixed boundaries</a></h3><p>Dirichlet, Neumann and Robin data on different arcs.</p></div></article>
</div>

Then [compare global collocation, LHI and RBF-FD](global-lhi.md) on the same equation,
or [define a symbolic kernel](custom-kernel.md). These tutorials make the numerical
choice explicit instead of hiding it inside a problem-specific solver.

## Coupled fields and three dimensions

| Tutorial | New idea | Scope |
|---|---|---|
| [Stokes between rotating cylinders](annular-stokes.md) | Divergence-free velocity and pressure spaces | Steady annular flow with known velocity |
| [Poisson inside a ball](ball.md) | A 3D domain, derivatives and independent error checks | Scalar elliptic PDE |
| [Navier–Stokes cavity](cavity.md) | Staggered operators, explicit convection and a coupled sparse solve | Explained experimental flow algorithm with visible divergence defect |

## How to run the examples

- Short snippets use an installed `rbflab` package. Plotting needs `rbflab[examples]`.
- `python -m examples...` commands run from a source checkout, with the whole
  `examples` folder available. Follow [installation](../INSTALL.md).
- Python is the default. Where a tutorial offers C++ or PyTorch, it changes
  supported local assembly; it does not automatically change every solver stage.
- Report both the numerical recipe and the error measure. Attractive fields
  alone do not validate pressure, conservation or convergence.

Older focused checks remain available as supplementary pages: [annular Laplace](annulus.md),
[manufactured unsteady Stokes](stokes.md), [3D polynomial reproduction](scalar-3d.md),
and [polynomial mixed-boundary verification](mixed-boundary.md). The main route
above selects one leading example for each concept.
