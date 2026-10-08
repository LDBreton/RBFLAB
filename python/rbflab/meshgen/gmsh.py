"""Optional adapters for an existing Gmsh model; no Gmsh import at module load."""
import numpy as np
from ..geometry import TriangleMesh2D


def from_gmsh():
    """Read the current planar first-order triangular Gmsh mesh.

    Returns (TriangleMesh2D, original_node_tags). Physical curve names define
    boundary labels (or a single boundary group when none are supplied).
    Construction-only points are discarded. The caller owns
    initialization/finalization; this function does not alter the model.
    Supports exterior curves only; use TriangleMesh2D explicitly for interfaces.
    """
    import gmsh
    if not gmsh.isInitialized():
        raise RuntimeError("Initialize Gmsh and generate a 2D mesh first")
    types, _, blocks = gmsh.model.mesh.getElements(2)
    if list(types) != [2]:
        raise ValueError('Example requires first-order, three-node triangles')
    triangle_tags = np.asarray(blocks[0]).reshape(-1, 3)
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    coordinates = np.asarray(coordinates).reshape(-1, 3)
    # Ignore construction-only points (e.g. circle centers), retain tag map.
    used_tags = np.unique(triangle_tags)
    coordinates_by_tag = dict(zip(map(int, tags), coordinates))
    xyz = np.array([coordinates_by_tag[int(tag)] for tag in used_tags])
    if not np.allclose(xyz[:, 2], xyz[0, 2], atol=1e-12, rtol=0):
        raise ValueError('Only planar XY meshes are supported')
    vertices = xyz[:, :2]
    triangles = np.searchsorted(used_tags, triangle_tags)
    boundaries = {}
    for _, tag in gmsh.model.getPhysicalGroups(1):
        label = gmsh.model.getPhysicalName(1, tag) or f'boundary_{tag}'
        line_blocks = []
        for curve in gmsh.model.getEntitiesForPhysicalGroup(1, tag):
            line_types, _, nodes = gmsh.model.mesh.getElements(1, int(curve))
            if list(line_types) != [1]:
                raise ValueError('Example requires first-order, two-node boundary lines')
            line_blocks.append(np.asarray(nodes[0]).reshape(-1, 2))
        line_tags = np.concatenate(line_blocks)
        if not np.isin(line_tags, used_tags).all():
            raise ValueError('Boundary nodes must belong to the triangle mesh')
        mapped = np.searchsorted(used_tags, line_tags)
        if label in boundaries:
            mapped = np.unique(np.sort(np.vstack((boundaries[label], mapped)),axis=1),axis=0)
        boundaries[label] = mapped
    return TriangleMesh2D(vertices, triangles, boundaries or None), used_tags.copy()
