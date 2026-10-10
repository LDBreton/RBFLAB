# From mathematical ideas to working code

Choose a question below, then follow the equation or interpolation formula into
Python objects and a result you can inspect. The lessons target **RBFLAB 0.5.0
or newer**. Start with [installation](../INSTALL.md) and [your first PDE on an
ellipse](../getting-started.md) if you have not used the library before.

<div class="rbf-examples">
<article class="rbf-example"><a href="interpolation/"><img src="../assets/teaching_data_card.png" alt="Illustration of scattered samples over a field"></a><div class="rbf-example-body"><h3><a href="interpolation/">A field from data</a></h3><p>Fit values, choose a kernel, then evaluate and differentiate.</p></div></article>
<article class="rbf-example"><a href="symbolic-pde/"><img src="../assets/teaching_pde_card.png" alt="Illustration of an elliptical PDE domain and marked wall nodes"></a><div class="rbf-example-body"><h3><a href="symbolic-pde/">A PDE from equations</a></h3><p>Name the boundary, write the equations, choose a method, and solve.</p></div></article>
<article class="rbf-example"><a href="custom-assembly/"><img src="../assets/teaching_algorithm_card.png" alt="Illustration of a sparse local derivative map"></a><div class="rbf-example-body"><h3><a href="custom-assembly/">Your own algorithm</a></h3><p>Build local weights and sparse maps, then assemble the equation yourself.</p></div></article>
</div>

The cards illustrate the three routes; each lesson identifies its actual cloud,
operator, and numerical check. For geometry without a PDE, see
[mesh generation](../geometry/index.md).

## Start from data

1. [Interpolate scattered data](interpolation.md): values at points determine an
   RBF expansion; evaluate the fitted field at new points.
2. [Differentiate a field](differentiation.md): apply derivative functionals to
   an interpolant or a local map.
3. [Define a custom kernel](custom-kernel.md): turn a symbolic radial formula
   into a kernel with the derivatives your calculation needs.

This route needs no PDE. It distinguishes measured values, expansion
coefficients, and derivative weights as they arise.

## Start from a PDE

1. [Solve the first ellipse problem](../getting-started.md): generate nodes,
   choose RBF-FD, solve, and measure error at unused points.
2. [Write a symbolic PDE](symbolic-pde.md): express the interior and boundary
   equations independently of the discretization.
3. [Compare global collocation, LHI, and RBF-FD](global-lhi.md): see which
   unknowns each method solves for. The individual [global](global-collocation.md),
   [LHI](lhi.md), and [one-stencil RBF-FD](one-stencil.md) lessons derive the
   constructions.
4. [Advance heat in time](heat-equation.md): see how a spatial sparse matrix
   enters a time-stepping formula.

The symbolic problem is an equation-driven interface. `RBFFD` and `LHI` can
assemble supported scalar problems from it. For a local construction you can
control term by term, follow the next route.

## Build a numerical method

The local vocabulary is **space → sampled functionals and trial functions →
target operators → sparse blocks → your algorithm**. [Functionals to
operators](local-approximation.md) introduces `Samples` and
`LocalApproximation`. Then [one stencil](one-stencil.md) derives RBF-FD weights,
while [local Hermite interpolation](lhi.md) adds value, PDE, and boundary
functionals. [Choose LHI centers](lhi-centers.md) explains their distinct roles.

Use [LHI and heat from matrices](lhi-matrices.md) to assemble a stationary solve
and time loop from named blocks. [Assemble your own PDE](custom-assembly.md)
combines derivative maps with SciPy. These lessons show where RBFLAB supplies
weights and where your code supplies forcing, boundary values, matrix algebra,
and time integration. The [cavity algorithm](cavity.md) is an experimental
coupled-flow construction, with its own stated limitations.

Global interpolation and collocation use dense coefficient systems.
Specialized [Stokes fields](annular-stokes.md) and the cavity use additional
coupled-system machinery; the shared local functional interface does not yet
replace every coupled solver.

## Apply the ideas on curved and 3D domains

- [Poisson with holes](perforated-poisson.md): label an obstacle boundary and solve.
- [Mixed boundaries on an ellipse](ellipse.md): assign Dirichlet, Neumann, and
  Robin equations to arcs.
- [Heat on a flower](flower-heat.md): prescribe nonzero transient boundary data.
- [Poisson inside a 3D ball](ball.md): carry the geometry and operator into 3D.

Each application includes its domain or node image and a numerical recipe. The
[teaching guide](teaching.md) offers lesson sequences and exercises. For
conditioning, error interpretation, and refinement, use the
[mathematical foundations](../theory/index.md) and
[validation guide](../guides/curved-validation.md).

Short code fragments need an installed `rbflab` package. Commands of the form
`python -m examples...` run from a source checkout; examples are not installed
as a Python subpackage. Plots require `rbflab[examples]`. See
[installation](../INSTALL.md) and [capabilities](../CAPABILITIES.md) for the
optional C++ and PyTorch backends and their supported methods.
