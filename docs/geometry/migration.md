# Migrating from RBFMeshGen

RBFMeshGen's maintained geometry and sampling source is integrated into RBFLAB
0.2. You no longer need a separate generator package for these workflows.
The original repository and its history are retained.

## Preferred API

```python
from rbflab import geometry, meshgen
cloud = meshgen.generate(geometry.Annulus(), interior=240, boundary=144, seed=42)
# Pass cloud directly to problem.solve(cloud, method).
```

## Existing scripts

| Earlier import | Integrated import |
|---|---|
| `RBFMeshGen.Border` | `rbflab.geometry.Border` |
| `RBFMeshGen.RBFMesh` | `rbflab.meshgen.RBFMesh` |
| `RBFMeshGen.RBFMesh3D` | `rbflab.meshgen.RBFMesh3D` |
| `Sphere`, `Box`, `Cylinder`, `ImplicitRegion` | `rbflab.geometry` |
| `ParametricSurface3D` | `rbflab.meshgen.ParametricSurface3D` |
| `TriangleMesh2D`, `staggered_clouds`, save/load | `rbflab.geometry` |
| `PointCloud2D` | `rbflab.PointCloud` (shared validated contract) |
| Old plotting helpers | `rbflab.viz.plot_cloud` after conversion |

The retained `RBFMesh` and `RBFMesh3D` classes keep `Points` and `Boundary_Points`
for existing scripts. Convert with:

```python
from rbflab import from_rbfmeshgen
cloud = from_rbfmeshgen(mesh, boundary_labels=["outer", "hole"],
                       interface_labels=["material_interface"], normals=normal_callbacks)
```

Only list interfaces actually present in your mesh. Explicit normal arrays or
callbacks remain necessary for this legacy adapter; it does not infer normals.
Region point labels are preserved. Existing `rbflab.PointCloud`, grid helpers,
and solver imports have not changed. Installation requires Python 3.11+, matching
RBFLAB, rather than RBFMeshGen's older Python requirement.

## Provenance

Imported source came from the RBFMeshGen working tree, including its local
parametric-surface and staggered implementations. The [source manifest](../provenance/rbfmeshgen-source.json)
records original hashes and the base commit. The [original MIT license text](../provenance/RBFMeshGen-LICENSE.txt)
is retained alongside RBFLAB's MIT license. Numerical research outputs, virtual
environments, binaries and historical plots were not imported.
