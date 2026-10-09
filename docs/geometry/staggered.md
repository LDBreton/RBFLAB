# What is a staggered cloud?

A staggered layout uses **different spatial locations for different fields**.
Instead of storing every unknown at the same nodes, an algorithm might place one
field at mesh vertices and another at edge midpoints. The locations interleave,
which is the meaning of *staggered*.

RBFLAB's current construction starts from an explicit **2D triangle mesh** and
produces two point clouds: its vertices and the midpoint of every unique edge.
It does not decide which PDE field belongs to which cloud.

![Four vertices, five unique edge midpoints, and their interlaced layout](../assets/meshes/staggered.png)

## Build the geometry shown above

```python
from rbflab.geometry import TriangleMesh2D, staggered_clouds

mesh = TriangleMesh2D(
    vertices=[[0,0], [2,0], [1.6,1.2], [-.2,1]],
    triangles=[[0,1,2], [0,2,3]],
    boundary_edges={
        "bottom": [[0,1]], "right": [[1,2]],
        "top": [[2,3]], "left": [[3,0]],
    },
)
layout = staggered_clouds(mesh)
vertex_cloud = layout.vertices
midpoint_cloud = layout.edge_midpoints
```

There are four vertices and two triangles, but **five unique edges**. The shared
diagonal contributes one midpoint, not one copy per incident triangle. Its midpoint
is an interior node. The four exterior-edge midpoints belong to their named boundaries.

## When is it useful?

A flow algorithm may assign pressure to one location set and velocity to another,
or evaluate gradients at locations different from their input data. The geometry
API uses neutral names `vertices` and `edge_midpoints` because other algorithms
may assign different quantities to the same sets.

A staggered layout alone does not produce a stable pressure–velocity coupling,
a divergence-free discretization or a PDE solution. You still choose operators,
source/target clouds, boundary rows and the numerical scheme. Ordinary RBF-FD
also works without staggering; do not construct a mesh solely because the library
has this option.

## How do labels and normals transfer?

`TriangleMesh2D` boundary labels refer to **edges**, expressed as pairs of vertex
indices. They must cover all exterior edges when supplied explicitly. If omitted,
exterior edges receive a single `boundary` label.

- Each boundary edge midpoint inherits its edge's label and outward normal.
- Each endpoint vertex belongs to every boundary label incident to it.
- Vertex normals are length-weighted averages **within each label**, not across all labels.
- A vertex at the meeting of `bottom` and `left` can therefore have two labels and
  two different per-label normals.
- Tagged internal edges become `interfaces`; their nodes remain interior unless
  they also belong to an exterior boundary. Interface normals are not guessed.

This corner convention differs from the single-owner endpoint sampling of planar
`Border`/`ParametricBoundary` arcs. Your boundary assembly must choose compatible
rows at a point with multiple labels.

## What connectivity is retained?

| Attribute | Meaning |
|---|---|
| `layout.vertices` | PointCloud in the original vertex order |
| `layout.edge_midpoints` | PointCloud at unique indexed edges |
| `layout.edges` | Vertex pair defining each midpoint |
| `layout.triangle_edges` | Three edge indices for each triangle |
| `layout.edge_triangles` | Incident triangles for each edge |

Both point clouds can be passed to the discrete-operator API. The connectivity
lets you inspect or implement your own source/target relationships.

Save and load with `save_staggered_layout(path, layout)` and
`load_staggered_layout(path)`. The format uses numeric NPZ and JSON metadata,
without pickle.

## From a larger mesh

The [Gmsh example](gmsh.md) constructs a plate with holes and transfers its physical
curve labels to the two point clouds. Random/QMC clouds have no supplied edges,
so they are **not silently triangulated** into a staggered layout. Current scope
is conforming planar triangles; 3D staggering is not implemented.
