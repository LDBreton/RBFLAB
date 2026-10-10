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

## One route with optional extensions

Use the [six-tutorial sequence](index.md) as the course spine:
[interpolation](interpolation.md) →
[local operators](local-approximation.md) →
[stationary assembly](custom-assembly.md) →
[global collocation](global-collocation.md) →
[LHI](lhi.md) →
[heat](heat-equation.md).
The primary examples use the unit square. Ask students to identify the
unknown vector before looking at its length: global collocation solves for
expansion coefficients, while stationary RBF-FD and LHI solve for nodal values.

Choose one extension only when the topic calls for it:
[differentiation](differentiation.md) or
[custom kernels](custom-kernel.md) after interpolation;
[one stencil](one-stencil.md) after local operators;
[independent LHI centers](lhi-centers.md) or
[LHI heat matrices](lhi-matrices.md) after LHI.
Use the [gallery](../gallery/index.md) for curved domains, 3D, and coupled fields.

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
