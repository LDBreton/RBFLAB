# Teach and adapt the examples

The lessons are designed for readers who know the mathematics of RBFs and want
to implement it, including students learning scientific Python alongside it.
Use the visible snippets as a guided construction; the complete script is a
reference to return to, not the first thing to explain.

## A lesson built around one mathematical change

1. State the data, unknown, approximation and equation on the board.
2. Identify their code objects before running anything.
3. Construct the example one step at a time and inspect shapes.
4. Ask students to predict the effect of changing one ingredient.
5. Make the change and explain the resulting field or matrix.

For example, switching $\partial_x$ to $\Delta$ changes the target functional
and weights, while switching the sampled data keeps a reusable operator fixed.
Changing from RBF-FD to LHI changes the local information, not just a solver name.

## Three suggested lesson sequences

### Interpolation and approximation

[Scattered data](interpolation.md) → [derivatives](differentiation.md) → [custom kernels](custom-kernel.md).

Students supply `(N, 2)` points and `(N,)` values, fit a field, differentiate it,
and modify a radial family. An exercise can keep the same data and change only
the kernel, or keep the kernel and change only the target functional.

### From a PDE to a discrete equation

[Symbolic PDE](symbolic-pde.md) → [global collocation](global-collocation.md) →
[one stencil](one-stencil.md) → [custom assembly](custom-assembly.md) → [LHI](lhi.md).

Ask students to identify the meaning of an unknown vector before looking at its
length. Have them construct a PDE row, then a boundary row. In LHI, ask why known
forcing values contribute to the right-hand side even though they were used in
the local approximation.

### Evolution and coupled fields

[Heat](heat-equation.md) → [Stokes spaces](annular-stokes.md) → [cavity algorithm](cavity.md).

Students replace a time formula while retaining a spatial matrix, then distinguish
incompressibility built into a trial space from incompressibility imposed through
block equations. Use the small cavity startup configuration for code discussion;
a long animation is not required to understand the update.

## Modification exercises

| Starting lesson | Ask students to implement | Concept being exercised |
|---|---|---|
| Interpolation | Replace synthetic values with a supplied data array | Data and coefficient roles |
| Differentiation | Evaluate a gradient at a new target cloud | Rectangular source-to-target maps |
| Custom kernel | Change a parameter, then change the formula | Bound kernel versus symbolic family |
| Symbolic PDE | Add a reaction term or a Robin wall | Equation versus approximation |
| Global collocation | Write one row directly with `kernel.matrix` | Target and source functionals |
| RBF-FD | Choose one neighborhood explicitly | Weight construction and scattering |
| LHI | Change the PDE-center count | Data functionals versus physical locations |
| Heat | Supply nonzero time-dependent boundary data | Boundary contribution to the update |
| Stokes | Change the inner wall angular speed | Vector boundary data and spaces |

Keep one simple check, such as polynomial reproduction or comparison at a few
known points, to reveal indexing/sign mistakes. Detailed convergence experiments
are optional follow-up work, not prerequisites for these lessons.

## Explain the layers before changing them

| Layer | Responsibility |
|---|---|
| Geometry and cloud | Locations, subsets, labels and normals |
| Kernel and space | Local/global trial functions and polynomial modes |
| Problem | Interior and boundary equations |
| Method | How trial functions produce discrete equations |
| Operator/system | Reusable weights, matrices and right-hand sides |
| Solution | Values or coefficients and their field reconstruction |

The same symbol `@` may mean differential-operator composition or application of
a discrete matrix; name the objects involved. Likewise, a local interpolation
matrix used to build weights is not the matrix enforcing the global PDE.

## Prepare a teaching session

Install the core package first; plotting needs the examples extra. A source
checkout is needed for module commands and the complete example scripts.
Use Python for the baseline lessons; introduce C++ or PyTorch only after the
mathematical construction is clear and the supported backend setup is complete.

Figures are generated from maintained numerical recipes. They may be reused with
the repository's MIT-licensed material; preserve its license notice when copying
substantial portions. The [maintenance guide](../MAINTAINING_DOCS.md) explains how
to update source snippets and regenerate figures together.
