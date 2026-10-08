"""Optional, duck-typed RBFMeshGen adapter; no generator dependency at import."""
import numpy as np
from .geometry import PointCloud


def from_rbfmeshgen(mesh, *, boundary_labels, normals=None):
    """Import a 2D or 3D mesh using explicitly selected exterior boundary labels.

    Other labeled interface samples remain interior nodes. Exact duplicate
    coordinates are merged, preserving membership in multiple boundary labels.
    normals maps labels to arrays, or callbacks receiving that label's points.
    No normals are guessed; Dirichlet conditions do not require them.
    """
    labels=list(boundary_labels)
    if len(set(labels))!=len(labels):
        raise ValueError("Repeated boundary labels")
    coordinates=[];lookup={};boundary={label:[] for label in labels}
    supplied={};normal_sources=normals or {}
    if set(normal_sources)-set(labels):
        raise ValueError("Normals supplied for an unselected boundary label")
    def add(point):
        key=(float(point.x),float(point.y))
        if hasattr(point,"z"):
            key += (float(point.z),)
        if coordinates and len(key) != len(coordinates[0]):
            raise ValueError("Mesh points must have a consistent dimension")
        if key not in lookup:
            lookup[key]=len(coordinates);coordinates.append(key)
        return lookup[key]
    for point in mesh.Points:
        add(point)
    for point in mesh.Boundary_Points:
        index=add(point)
        label=getattr(point,"boundary_label",getattr(point,"label",None))
        if label in boundary and index not in boundary[label]:
            boundary[label].append(index)
    if any(not ids for ids in boundary.values()):
        raise ValueError("Each selected boundary label must have samples")
    points=np.asarray(coordinates,dtype=float)
    for label,value in normal_sources.items():
        locations=points[boundary[label]]
        supplied[label]=value(locations) if callable(value) else value
    return PointCloud(points,boundary,supplied)
