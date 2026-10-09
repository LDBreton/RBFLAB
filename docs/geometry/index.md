# Geometry and node generation

Define the shape, name its boundaries, and generate the nodes on which your
PDE will be solved. Geometry and numerical methods share the same `PointCloud`:
no conversion step and no mandatory mesh generator.

![Generated clouds on a channel with obstacles, a perforated flower, a concave polygon and a cutaway sphere](../assets/geometry_gallery.png)

## Start with a domain

```python
from rbflab import geometry, meshgen, viz

domain = geometry.Annulus(inner_radius=0.4, outer_radius=1.0)
cloud = meshgen.generate(
    domain, interior=240, boundary={"inner": 48, "outer": 96}, seed=42,
)
print(meshgen.quality(cloud, stencil_size=35, polynomial_degree=3))
fig, ax = viz.plot_cloud(cloud, normals=True)
```

The geometry and solver coordinates are Float64. Selecting MPFR for local
weights increases arithmetic precision, not the precision of input coordinates.

## What a point cloud contains

| Attribute | Meaning |
|---|---|
| `points` | Unique `(N, 2)` or `(N, 3)` coordinates |
| `boundary` | Label to exterior boundary indices; corners may have several labels |
| `normals` | Outward unit vectors aligned with each label's indices |
| `interfaces` | Named internal-interface indices; these remain interior unknowns |
| `regions` | Named groups of region node indices, possibly overlapping |
| `triangles` | Optional supplied connectivity; never invented by sampling |

For generator outputs, `regions` identifies interior samples. Region and interface
metadata do **not** automatically impose transmission equations. Set those equations
explicitly in your discretization.

## Choose a geometry

- `geometry.Disk`, `Ellipse`, `Annulus`, and `Flower`: smooth planar domains.
- `geometry.Polygon` and `Rectangle`: labeled straight edges, including concave shapes.
- `geometry.with_holes`: add named circular, elliptical or polygonal holes.
- `geometry.Border` + `meshgen.RBFMesh`: FreeFEM-inspired labeled curves with signed segment counts.
- `geometry.ParametricBoundary` and `ParametricDomain`: user-defined curves with analytic tangents.
- `geometry.Sphere`, `Box`, `Cylinder`, and `ImplicitRegion`: 3D volumes.
- `meshgen.ParametricSurface3D`: separately sampled parametric surfaces.
- `geometry.TriangleMesh2D`: an explicit conforming triangular mesh for staggering.

Start with the [geometry cookbook](cookbook.md) or the
[FreeFEM-inspired border syntax](parametric-borders.md), then explore
[custom boundaries](custom-domains.md) or [staggered clouds](staggered.md).
The [Poisson example with holes](../tutorials/perforated-poisson.md) carries one
composed domain all the way through symbolic assembly and error measurement.

## Sampling and quality

`method="random"`, `"halton"`, and `"sobol"` produce reproducible interior samples
when a seed is given. They do **not** enforce a minimum inter-node distance.
Boundary points in 2D are approximately uniform in arc length; 3D boundary
sampling follows each implicit region's implementation. General implicit
projection does not promise a uniform surface distribution.

`boundary_distance` rejects interior nodes too close to a boundary. It is not
an inter-node separation rule. `max_candidates` bounds rejection sampling;
failure raises an error instead of silently returning fewer nodes. Rejection
and truncation weaken the balance properties of a Sobol sequence.

`meshgen.quality` reports spacing and normalized stencil polynomial-rank
indicators. Small values identify geometry needing inspection; good values
do not prove PDE stability. Use independent seeds and refinement, not one cloud.

## Four different length scales

1. **Point spacing:** distances between neighboring physical nodes.
2. **Stencil radius:** extent of the nodes used for one local approximation.
3. **Shape parameter:** controls a smooth kernel; its units follow the kernel convention.
4. **Support radius:** distance beyond which a compact kernel is exactly zero.

These are different choices. Local coordinate scaling makes kernel parameters
dimensionless relative to a stencil radius; it does not improve the input cloud.
See [conditioning](../theory/conditioning.md).

## What is validated

The [annulus](../tutorials/annulus.md) compares three spatial methods and independent
clouds. The [flower heat](../tutorials/flower-heat.md), [mixed ellipse](../tutorials/ellipse.md),
[annular Stokes](../tutorials/annular-stokes.md), and [3D ball](../tutorials/ball.md)
tutorials check computed fields against known solutions. Existing square/cube tests
remain regression baselines.
