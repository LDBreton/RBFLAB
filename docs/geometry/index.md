# Geometry and mesh generation

RBFLAB geometry follows the same useful order as a FreeFEM++ mesh script: write the boundary curves, decide how each directed contour encloses material, assign labels, then choose discretization and inspect the result. Here the output is usually a labeled point cloud; it is not a finite-element triangulation.

![Planar domains built from curves and generated nodes](../assets/meshes/manual-planar-atlas.png)

## The learning path

1. [Built-in planar domains](planar-domains.md): choose a disk, ellipse, polygon, flower, annulus, or cut-out domain.
2. [Parametric borders](parametric-borders.md): write curve formulas and assign signed segment counts, as with FreeFEM++ `border` definitions.
3. [Custom contours](custom-domains.md): construct explicit outer and hole contours with `ParametricBoundary` and `ParametricDomain`.
4. [Labels and normals](labels-normals.md): connect names and outward directions to boundary conditions.
5. [Sampling](sampling.md): adjust boundary and interior resolution and inspect point quality.

The [visual mesh atlas](gallery.md) compares the complete construction range:
primitive and custom 2D shapes, point distributions, boundary-refined triangles,
3D volume and surface clouds, and staggered locations.

## First cloud

```python
from rbflab import geometry as g, meshgen, viz

domain = g.Annulus(inner_radius=.35, outer_radius=1.)
cloud = meshgen.generate(domain, interior=200,
                         boundary={"inner": 40, "outer": 80}, seed=42)
fig, ax = viz.plot_cloud(cloud, normals=True)
```

![Annulus nodes and outward normals](../assets/meshes/annulus.png)

There are 200 interior points and 120 boundary points. The hole has boundary samples but no material points; its outward-from-material normals point into the void. Labels are metadata until a PDE model assigns equations to them.

## Geometry is not connectivity

`meshgen.generate` returns unique coordinates, label-to-index boundary groups, and outward unit normals. Ordinary generation does not create triangle connectivity. Use [`meshgen.triangulate`](sampling.md#boundary-driven-triangulation) or an imported mesh when your method requires cells. A point-cloud plot must not be mistaken for a triangulation.

| Construction | Set boundary resolution with | Material and holes |
|---|---|---|
| Built-in or `ParametricDomain` | `boundary=total` or a label-count dictionary | Explicit built-in semantics or `outer`/`holes` |
| `Border` collection | `piece(count)`; a negative count reverses traversal | Directed contour orientation |

For direct `Border` construction, omit `boundary=` in `generate`; the signed count already fixes each piece's sample count. For domains, `boundary` is independent of `interior`.

## Inspect labels and coordinates

```python
print(cloud.points.shape)
print(len(cloud.interior))
for name, indices in cloud.boundary.items():
    print(name, len(indices), cloud.normals[name].shape)
```

Use `cloud.points[cloud.boundary["inner"]]` for boundary coordinates. The normals array is aligned row for row with those indices. See [labels and normals](labels-normals.md) for corner ownership, derivative accuracy, and explicit overrides.

## Next

Start with [built-in 2D domains](planar-domains.md), then follow the border and
contour lessons in order. Choose [Padua distributions](padua.md) when degree
controls your node recipe, or [boundary-refined triangulation](triangulation.md)
when labeled curves should determine interior triangle density. The
[3D volume](volumes-3d.md), [surface](surfaces-3d.md), [Gmsh](gmsh.md), and
[staggered cloud](staggered.md) pages cover their respective constructions.
