"""Mesher-independent point cloud plus an optional Gmsh square adapter."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class PointCloud:
    """Validated 2D or 3D coordinates with tagged boundary nodes.

    Args:
        points: Finite, unique coordinates with shape `(N, dimension)`.
        boundary (dict): Mapping from boundary label to arrays of point indices.
        normals: Mapping from labels to unit-normal arrays matching their nodes.
        triangles: Optional `(T, 3)` connectivity for visualization.
        interfaces: Named internal-interface node indices (remain interior).
        regions: Named region node indices, which may overlap."""
    points: np.ndarray
    boundary: dict
    normals: dict
    triangles: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=int))

    interfaces: dict = field(default_factory=dict)
    regions: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = np.array(self.points, dtype=float, copy=True)
        p = self.points
        if p.ndim != 2 or p.shape[1] not in (2, 3) or len(p) == 0 or not np.isfinite(p).all():
            raise ValueError("Expected nonempty finite (N, 2) or (N, 3) points")
        if len(np.unique(p, axis=0)) != len(p):
            raise ValueError("Duplicate points are not supported")
        for name in ("boundary", "interfaces", "regions"):
            groups = {}
            for label, values in getattr(self, name).items():
                raw = np.asarray(values)
                if raw.ndim != 1 or (raw.size and (raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw >= len(p)))):
                    raise ValueError(f"Invalid {name} indices")
                indices = raw.astype(int, copy=True)
                if len(np.unique(indices)) != len(indices):
                    raise ValueError(f"Repeated {name} indices")
                groups[label] = indices
            setattr(self, name, groups)
        self.normals = {k: np.asarray(v, dtype=float).copy() for k, v in self.normals.items()}
        for label, normals in self.normals.items():
            if label not in self.boundary or normals.shape != (len(self.boundary[label]), self.dimension):
                raise ValueError("Boundary normals must match (N_boundary, dimension)")
            if not np.isfinite(normals).all() or not np.allclose(np.linalg.norm(normals, axis=1), 1):
                raise ValueError("Normals must be finite unit vectors")
        triangles = np.asarray(self.triangles)
        if triangles.size and triangles.dtype.kind not in "iu":
            raise ValueError("Triangle connectivity must contain integers")
        self.triangles = np.array(triangles, dtype=int, copy=True)
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ValueError("Triangles must have shape (N, 3)")
        if np.any(self.triangles < 0) or np.any(self.triangles >= len(p)):
            raise ValueError("Invalid triangle connectivity")

    @property
    def dimension(self):
        return self.points.shape[1]

    @property
    def boundary_indices(self):
        if not self.boundary:
            return np.empty(0, dtype=int)
        return np.unique(np.concatenate(list(self.boundary.values())))

    @property
    def interior_indices(self):
        return np.setdiff1d(np.arange(len(self.points)), self.boundary_indices)

    @property
    def interior(self):
        return self.points[self.interior_indices]


def gmsh_square(cells=8, filename=None):
    """Create a deterministic transfinite triangular unit-square mesh.

    Physical curve labels are bottom/right/top/left. Connectivity is for
    visualization; collocation uses only coordinates, labels and normals.
    Owns a short Gmsh session; does not clear a caller's existing Gmsh model.
    """
    if type(cells) is not int or cells < 2:
        raise ValueError("cells must be an integer >= 2")
    import gmsh
    if gmsh.isInitialized():
        raise RuntimeError("gmsh_square requires an uninitialized Gmsh session")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("unit_square")
        points = [gmsh.model.geo.addPoint(x, y, 0) for x, y in
                  [(0, 0), (1, 0), (1, 1), (0, 1)]]
        curves = [gmsh.model.geo.addLine(points[i], points[(i+1) % 4]) for i in range(4)]
        loop = gmsh.model.geo.addCurveLoop(curves)
        surface = gmsh.model.geo.addPlaneSurface([loop])
        gmsh.model.geo.synchronize()
        labels = ["bottom", "right", "top", "left"]
        vectors = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for label, curve in zip(labels, curves):
            tag = gmsh.model.addPhysicalGroup(1, [curve])
            gmsh.model.setPhysicalName(1, tag, label)
            gmsh.model.mesh.setTransfiniteCurve(curve, cells+1)
        tag = gmsh.model.addPhysicalGroup(2, [surface])
        gmsh.model.setPhysicalName(2, tag, "domain")
        gmsh.model.mesh.setTransfiniteSurface(surface)
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        tags = tags[order]
        xy = np.asarray(coordinates).reshape(-1, 3)[order, :2]
        lookup = {int(tag): i for i, tag in enumerate(tags)}
        boundary, normals = {}, {}
        for label, curve, normal in zip(labels, curves, vectors):
            bt, _, _ = gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)
            ids = np.array(sorted({lookup[int(t)] for t in bt}), dtype=int)
            boundary[label] = ids
            normals[label] = np.tile(normal, (len(ids), 1))
        _, element_nodes = gmsh.model.mesh.getElementsByType(2)
        triangles = np.array([lookup[int(t)] for t in element_nodes], dtype=int).reshape(-1, 3)
        if filename is not None:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(path))
        return PointCloud(xy, boundary, normals, triangles)
    finally:
        gmsh.finalize()


def unit_box_grid(cells=5, dimension=2):
    """Small structured unit-box cloud for examples; Gmsh is not required.

    Face labels and outward normals are supplied for boundary operators.
    Corners belong to each adjacent face; boundary conditions take list order.
    """
    if type(cells) is not int or cells < 2:
        raise ValueError("cells must be an integer >= 2")
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    axes = [np.linspace(0.0, 1.0, cells + 1)] * dimension
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dimension)
    labels = (("left", "right"), ("bottom", "top"), ("front", "back"))
    boundary, normals = {}, {}
    for axis in range(dimension):
        for side in (0, 1):
            label = labels[axis][side]
            ids = np.flatnonzero(points[:, axis] == float(side))
            vector = np.zeros(dimension)
            vector[axis] = 2 * side - 1
            boundary[label] = ids
            normals[label] = np.tile(vector, (len(ids), 1))
    return PointCloud(points, boundary, normals)



def gmsh_cube(cells=4, filename=None):
    """Transfinite unit cube with six tagged faces and outward unit normals.

    Gmsh generates hexahedra; collocation uses only the points. Volume
    connectivity is not stored in PointCloud. An optional .msh retains it.
    """
    if type(cells) is not int or cells < 2:
        raise ValueError("cells must be an integer >= 2")
    import gmsh
    if gmsh.isInitialized():
        raise RuntimeError("gmsh_cube requires an uninitialized Gmsh session")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("unit_cube")
        volume = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        for _, curve in gmsh.model.getEntities(1):
            gmsh.model.mesh.setTransfiniteCurve(curve, cells+1)
        faces = {}
        names = (("left", "right"), ("front", "back"), ("bottom", "top"))
        for _, face in gmsh.model.getEntities(2):
            center = np.asarray(gmsh.model.occ.getCenterOfMass(2, face))
            axis = int(np.argmax(abs(center-.5)))
            side = int(center[axis] > .5)
            normal = np.zeros(3); normal[axis] = 2*side-1
            label = names[axis][side]
            faces[label] = (face, normal)
            physical = gmsh.model.addPhysicalGroup(2, [face])
            gmsh.model.setPhysicalName(2, physical, label)
            gmsh.model.mesh.setTransfiniteSurface(face)
            gmsh.model.mesh.setRecombine(2, face)
        gmsh.model.mesh.setTransfiniteVolume(volume)
        physical = gmsh.model.addPhysicalGroup(3, [volume])
        gmsh.model.setPhysicalName(3, physical, "domain")
        gmsh.model.mesh.generate(3)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        xyz = np.asarray(coordinates).reshape(-1, 3)[order]
        lookup = {int(tag): i for i, tag in enumerate(tags[order])}
        boundary, normals = {}, {}
        for label, (face, normal) in faces.items():
            bt, _, _ = gmsh.model.mesh.getNodes(2, face, includeBoundary=True)
            ids = np.array(sorted({lookup[int(t)] for t in bt}), dtype=int)
            boundary[label] = ids
            normals[label] = np.tile(normal, (len(ids), 1))
        if filename is not None:
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(path))
        return PointCloud(xyz, boundary, normals)
    finally:
        gmsh.finalize()
