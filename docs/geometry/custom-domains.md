# Custom curves, holes and 3D regions

For labeled polygons, rectangles and named holes, start with the
[geometry cookbook](cookbook.md). This page explains the lower-level curve
and implicit-region interfaces.

Prefer the original `Border(...)` and `curve(n)` syntax? Use the
[parametric-borders tutorial](parametric-borders.md); that interface is also built in.

## A domain from a curve

Supply coordinates **and their analytic tangent**. This avoids noisy finite
differences when constructing normals for Neumann or Robin conditions.

```python
import numpy as np
from rbflab import geometry, meshgen

wall = geometry.ParametricBoundary(
    curve=lambda t: (1.4*np.cos(t), .8*np.sin(t)),
    tangent=lambda t: (-1.4*np.sin(t), .8*np.cos(t)),
    label="wall",
)
domain = geometry.ParametricDomain(wall)
cloud = meshgen.generate(domain, interior=200, boundary={"wall": 90}, seed=7)
```

A contour can be a sequence of connected arcs with distinct labels. Pass hole
contours through `holes=[...]`. Normals are corrected for contour orientation:
outer normals point away from the material; hole normals point into the hole.
Open, self-intersecting or intersecting-hole contours are rejected.

Each arc excludes its final endpoint. A junction therefore belongs to the arc
that starts there. For smooth mixed-condition junctions this gives a deterministic
single row. If a corner needs membership in several groups, construct that
membership explicitly in `PointCloud` or use a tagged triangular mesh.

Generic parametric membership uses 2048 polygon segments per arc. It is a geometric
approximation, not exact CAD. The built-in curved primitives use analytic membership.
Geometric refinement must be considered separately from PDE refinement.

## An implicit 3D region

```python
region = geometry.Sphere(radius=1, boundary_label="wall")
cloud = meshgen.generate(region, interior=300, boundary=180, seed=42)
```

For a general region, pass a function negative inside, finite bounding limits,
and preferably its gradient:

```python
region = geometry.ImplicitRegion(
    lambda x, y, z: x*x + y*y + z*z - 1,
    bounds=[(-1, 1)]*3,
    gradient=lambda x, y, z: (2*x, 2*y, 2*z),
    volume=4*np.pi/3,
    surface_area=4*np.pi,
)
```

Boundary generation projects candidates to the zero level set with a finite
iteration/candidate budget. A difficult field can fail; this is not a general
CAD mesher. Physical `boundary_distance` needs a supplied clearance function.
The unified generator accepts a single implicit region and a total boundary
count. The retained `RBFMesh3D` API supports region collections; it does not
resolve arbitrary overlaps into a conforming volume mesh.

## From your own arrays

```python
from rbflab import PointCloud
cloud = PointCloud(points, boundary={"wall": wall_ids}, normals={"wall": wall_normals})
```

No generator is required. Optional triangle connectivity is kept separate from
coordinates. Generation does not fabricate triangles or tetrahedra.
