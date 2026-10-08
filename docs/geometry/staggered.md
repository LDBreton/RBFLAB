# Gmsh and staggered clouds

Use staggering when an algorithm needs values at triangle vertices and at
unique edge midpoints. These names describe geometry; the PDE decides which
field belongs to each cloud.

```python
from rbflab.geometry import TriangleMesh2D, staggered_clouds

mesh = TriangleMesh2D(
    vertices=[[0, 0], [1, 0], [1, 1], [0, 1]],
    triangles=[[0, 1, 2], [0, 2, 3]],
    boundary_edges={"wall": [[0, 1], [1, 2], [2, 3], [3, 0]]},
    interface_edges={"diagonal": [[0, 2]]},
)
layout = staggered_clouds(mesh)
vertex_cloud = layout.vertices
midpoint_cloud = layout.edge_midpoints
```

Both clouds are ordinary RBFLAB `PointCloud` objects. Vertex order is preserved;
edge ordering is deterministic. `layout.edges`, `triangle_edges`, and `edge_triangles`
retain connectivity. Interface normals are not guessed: their direction depends
on the chosen side of an interface. Exterior normals follow the incident triangle;
vertex normals are length-weighted averages within each boundary label.

Save/load with `save_staggered_layout(path, layout)` and
`load_staggered_layout(path)`. The numeric NPZ plus JSON format does not use pickle.
It remains compatible with RBFMeshGen's version-2 layout archives.

## Gmsh perforated plate

```sh
python -m pip install "rbflab[mesh,examples]"
python -m examples.perforated_plate --output outputs/perforated_plate.png
```

This example builds a plate containing three circular holes, preserves physical
curve labels and maps noncontiguous Gmsh node tags to array indices. It is a
geometry demonstration, not a validated flow simulation.

For your own initialized Gmsh model:

```python
from rbflab.meshgen import from_gmsh
mesh, original_node_tags = from_gmsh()
layout = staggered_clouds(mesh)
```

The adapter reads an existing first-order XY-planar triangular mesh and exterior
physical curves. It does not initialize, clear or finalize the model. For tagged
internal interfaces, construct `TriangleMesh2D` explicitly. Existing
`rbflab.gmsh_square` and `rbflab.gmsh_cube` remain available.

**Current scope:** explicit 2D triangular staggering. Random clouds are not
silently triangulated, and 3D staggering is not implemented.
