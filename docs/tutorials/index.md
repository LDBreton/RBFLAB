# Six tutorials, one route

These lessons start with scattered values and end with a heat-equation time loop.
The introductory PDE and the principal examples use points in the unit square,
so changes in the **numerical construction** are easier to see. Read in order or
enter at the step you need. Each lesson gives the mathematical idea, the API
construction, a concrete result, and something to change.

Install **RBFLAB 0.5.0 or newer** with
`python -m pip install --upgrade "rbflab>=0.5"`.
The short code fragments use the installed library. Module commands for complete
scripts run from a [source checkout](../INSTALL.md); those scripts are not
installed as a Python package.

| Step | Build | Key distinction |
|---|---|---|
| **1. [Interpolate scattered data](interpolation.md)** | Fit a field from sample values and evaluate it. | Coefficients are not sample values. |
| **2. [From samples to operators](local-approximation.md)** | Declare a space, source samples, a trial, and target functionals. | Local square solves produce rows of a potentially rectangular sparse map. |
| **3. [Assemble a stationary PDE](custom-assembly.md)** | Combine local derivative maps, apply boundary data, and solve. | The PDE matrix is assembled from weights; it is not a local interpolation matrix. |
| **4. [Global collocation](global-collocation.md)** | Impose PDE and boundary rows on a global trial expansion. | Its dense unknown vector contains expansion coefficients. |
| **5. [Local Hermite interpolation](lhi.md)** | Include value, PDE, and boundary functionals in each patch. | Known forcing enters the right-hand side; solution values remain the unknowns. |
| **6. [Time-step heat](heat-equation.md)** | Reuse a spatial operator in backward Euler and BDF2. | Time integration is ordinary matrix algebra after spatial assembly. |

The local steps use the release 0.5 `Samples` and `LocalApproximation` interface.
[One stencil](one-stencil.md) opens the weight solve in detail. Global
[interpolation](interpolation.md) and [collocation](global-collocation.md) use
their own dense, coefficient-based interfaces. The optional
[symbolic PDE](symbolic-pde.md) lesson shows an equation-driven `RBFFD`
adapter; it does not replace the explicit operator construction.

## Go deeper where it helps

- After step 1: [differentiate a field](differentiation.md) and
  [define a custom kernel](custom-kernel.md).
- After step 2: [inspect one RBF-FD stencil](one-stencil.md) and read the
  [weight derivation](../theory/local-weights.md).
- After step 5: [choose independent LHI centers](lhi-centers.md) and
  [build LHI heat matrices](lhi-matrices.md).
- For classes: use the [teaching guide](teaching.md).
- For domains beyond the square and coupled equations: open the
  [gallery and advanced applications](../gallery/index.md).

The baseline uses the Python backend. C++ and PyTorch support depends on the
construction; check [installation](../INSTALL.md) and
[capabilities](../CAPABILITIES.md) before changing backends. Reported checks
in these lessons are examples, not convergence guarantees.
