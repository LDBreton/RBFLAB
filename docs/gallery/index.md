# Gallery and advanced applications

The [six-lesson route](../tutorials/index.md) develops the numerical ideas on
square point clouds. Come here when you want to change the geometry, boundary
model, dimension, or coupled physics while retaining those ideas.

| Application | What changes |
|---|---|
| [Poisson with holes](../tutorials/perforated-poisson.md) | An obstacle adds boundary labels and constraints. |
| [Mixed boundaries on an ellipse](../tutorials/ellipse.md) | Different arcs receive Dirichlet, Neumann, and Robin conditions. |
| [Heat on a flower](../tutorials/flower-heat.md) | A curved domain and nonzero transient boundary data. |
| [Poisson in a 3D ball](../tutorials/ball.md) | Coordinates, geometry, and differential operators become three-dimensional. |
| [Coupled Stokes fields](../tutorials/annular-stokes.md) | Velocity and pressure spaces form a coupled system. |
| [Cavity flow algorithm](../tutorials/cavity.md) | A full experimental pressure–velocity update is written out. |

For method comparisons and older specialized examples, see
[global versus local methods](../tutorials/global-lhi.md),
[Stokes velocity](../tutorials/stokes.md),
[mixed-boundary application](../tutorials/mixed-boundary.md), and
[scalar 3D application](../tutorials/scalar-3d.md).
The [mesh-generation guide](../geometry/index.md) explains how to build and
label point clouds before applying a PDE method.

The cavity chapter is a coarse-cloud numerical demonstration, not a benchmark
result. Each application states its own recipe and checks; transfer a recipe to
a new domain only after checking accuracy and conditioning there.
