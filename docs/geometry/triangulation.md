# Create a boundary-refined triangle mesh

This lesson follows the boundary-first workflow familiar from FreeFEM++: sample
the labeled contours, request an interior size field, then inspect the connected
triangles. RBFLAB uses Triangle through MeshPy here; the mesh is not produced by
FreeFEM++/BAMG. Its vertices and edge midpoints can then be exposed as point clouds.

## Step 1 — Sample and triangulate the boundary

For an unstructured distribution with gradual refinement near densely sampled
curves, use `meshgen.triangulate`. This follows the boundary-count workflow of
[FreeFEM's buildmesh](https://doc.freefem.org/documentation/mesh-generation.html)
using Triangle through MeshPy, rather than the FreeFEM/BAMG implementation.
Install the optional dependency with `python -m pip install "rbflab[triangulation]"`.
MeshPy and Triangle retain their own [upstream licenses](https://documen.tician.de/meshpy/faq.html).

```python
from rbflab import geometry, meshgen

domain = geometry.Annulus(inner_radius=.35, outer_radius=1.)
mesh = meshgen.triangulate(domain, boundary={"inner": 96, "outer": 48}, growth=.25)
cloud = geometry.staggered_clouds(mesh).vertices

# Or use the existing signed Border(n) syntax:
# mesh = meshgen.triangulate([outer(48), hole(-96)], growth=.25)
```

`mesh` is a `TriangleMesh2D`: `vertices`, `triangles`, and labeled `boundary_edges`
are available directly. The converted `cloud` includes connectivity and boundary
normals. These normals come from the straight mesh edges; use the existing
`staggered_clouds` normal overrides when analytic curve normals are needed.
Corners may belong to multiple edge labels.

The target spacing at an interior position is the minimum, over all boundary
segments, of `segment_length + growth * distance_to_segment`. Adding more
samples on one curve shortens its segments and refines the nearby interior.
Smaller `growth` extends that fine resolution farther into the domain.
`max_spacing=.1` optionally caps the target interior spacing. Refinement uses
triangle area at the centroid, so this is a sizing target, not a strict bound
on every edge length.

`fixed_boundary=True` retains the supplied boundary nodes. Set it to `False`
to allow additional points on the sampled boundary segments. The default
`min_angle=25` is a quality target; fixed boundaries and geometric corners can
prevent reaching it. Geometry is the polygon formed by the sampled curves:
increase boundary counts to resolve curvature and narrow gaps. One exterior
with disjoint holes is supported, including non-star-shaped domains; internal
material interfaces and overlapping positive contours are not yet supported.

Run `python examples/triangulated_curved_holes.py` to compare 24 versus 96
samples on one elliptical hole, with the other curves unchanged. It plots
both the triangulation and its vertices as an RBF point cloud, and reports
near-hole counts and local nearest-neighbor spacing.

## Step 2 — Plot the connected mesh and its cloud

```python
import matplotlib.pyplot as plt
from rbflab import viz

fig, ax = plt.subplots(figsize=(5, 4))
ax.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles,
               linewidth=.5)
ax.set_aspect("equal")
ax.set_title("Connected triangles")
# Use the standard cloud viewer in its own figure.
fig, ax = viz.plot_cloud(cloud, normals=True)
print("vertices:", mesh.vertices.shape, "triangles:", mesh.triangles.shape)
```

The triangle rows contain three vertex indices. The cloud boundary groups contain
vertex indices too; they are not triangle indices.

## Step 3 — Change one refinement control

| Option | Default | What to try | What changes |
|---|---|---|---|
| `boundary` | `100` for a domain | `{"inner": 96, "outer": 48}` | More inner samples refine the hole and adjacent material |
| `growth` | `.25` | `.15` | Fine boundary spacing extends farther into the interior |
| `max_spacing` | `None` | `.12` | Caps the target interior spacing; does not bound every edge |
| `min_angle` | `25.` | `20.` | Changes the quality target in degrees; valid range is `[0, 28]` |
| `fixed_boundary` | `True` | `False` | Allows extra nodes along the sampled straight boundary segments |

```python
finer = meshgen.triangulate(domain, boundary={"inner": 96, "outer": 48},
                            growth=.15, max_spacing=.12)
finer_cloud = geometry.staggered_clouds(finer).vertices
fig, ax = viz.plot_cloud(finer_cloud)
print("baseline / refined vertex counts:", len(mesh.vertices), len(finer.vertices))
```

There is no `interior` count here: the sizing and quality controls determine the
number of vertices. Smaller targets usually cost more nodes and work.

## Step 4 — Create two staggered node sets

```python
layout = geometry.staggered_clouds(mesh)
vertex_cloud = layout.vertices
midpoint_cloud = layout.edge_midpoints
fig, ax = viz.plot_cloud(midpoint_cloud)
print("vertex / midpoint nodes:", len(vertex_cloud.points), len(midpoint_cloud.points))
```

Each unique edge contributes one midpoint, including interior edges. The two
clouds are different sampling locations on the same mesh, not independent random
samples. Continue with [staggered layouts](staggered.md) for connectivity maps,
normal overrides and saving a layout.

**Try it:** change only the inner count from 96 to 48. Compare curvature near the
hole, the vertex count and the spacing nearby. Then restore it and change only
`growth`. This separates geometric resolution from interior refinement.

**If construction fails:** check that holes are disjoint and strictly enclosed,
that boundary counts resolve narrow gaps, and that MeshPy is installed. A fixed
boundary may limit triangle quality; adding boundary samples or allowing splitting
can help, but inserted nodes still lie on the polygonal segments.
