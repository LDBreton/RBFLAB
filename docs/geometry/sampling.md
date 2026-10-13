# Sampling, resolution and geometric quality

After fixing the geometry, choose how many samples to generate and how to
place them. Sampling choices affect spacing and the quality of local stencils;
they do not change the meaning of boundary labels.

## Compare interior samplers

```python
from rbflab import geometry as g, meshgen, viz

domain = g.Disk()
cloud = meshgen.generate(domain, interior=180, boundary=60,
                         method="halton", seed=42)
fig, ax = viz.plot_cloud(cloud)
```

![Random, Halton and Sobol interior clouds on the same disk](../assets/meshes/sampling.png)

Use `method="random"`, `"halton"` or `"sobol"` to reproduce the three panels.
They keep the same domain and boundary count. Halton and Sobol are low-discrepancy
sequences; rejection at curved boundaries and truncation weaken their original
balance properties. **None of these methods enforces a minimum inter-node distance.**

A fixed seed reproduces a recipe without changing Python/NumPy global random
state through the unified generator. Use several independent seeds when assessing
a numerical method; one favorable cloud is not a stability result.

## Counts mean different things on different inputs

| Geometry input | Interior resolution | Boundary resolution |
|---|---|---|
| Planar primitive or `ParametricDomain` | Exact requested `interior` count | Total `boundary`, or exact counts per arc label |
| `Border` or a list of borders | Requested material samples | Counts already assigned by `border(n)`; omit `boundary` |
| 3D volume | Exact requested `interior` count | Positive total surface count, distributed by the region sampler |
| Parametric surface | Must use `interior=0` | `boundary` is the surface sample count |

Internal-interface samples are retained in addition to the randomly/QMC-sampled
material points and count as interior unknowns. Boundary and interface metadata
are separate. Boundary pieces omitted with `is_border=False` are not sampled
into these groups, although they still define the border geometry.

## Clearance is not separation

`boundary_distance=.02` rejects interior samples closer than `.02` physical units
to the boundary. It does **not** enforce `.02` between every pair of interior nodes.
Generic planar clearance uses the polygon representation. A custom 3D implicit
region needs a suitable clearance function for physical-distance rejection.

`max_candidates` bounds the sampling search. If a thin domain or a large clearance
makes the request impossible within that budget, generation raises an error
rather than quietly returning fewer nodes.

## Inspect the cloud before a large solve

```python
print(meshgen.quality(cloud, stencil_size=25, polynomial_degree=2))
```

The report includes nearest-node spacing, a minimum/maximum spacing ratio,
local separation indicators and normalized polynomial-rank indicators.
Small ratios warn of close nodes or poorly shaped neighborhoods. These diagnostics
are geometric checks, not error bounds or guarantees of a stable PDE discretization.

## Keep four length scales separate

1. Physical **point spacing** between neighboring nodes.
2. **Stencil radius**, the extent of one local approximation.
3. Kernel **shape parameter**, whose units depend on the kernel convention.
4. Compact-kernel **support radius**, beyond which its value is zero.

Local coordinate scaling normalizes a stencil; it does not repair the input
cloud. See [conditioning and precision](../theory/conditioning.md).

## Refine geometry as well as nodes

Border samples define their polygonal geometry, so increasing `border(n)` refines
both boundary nodes and that approximation. Generic `ParametricDomain` uses a
separate fixed 2048-segment-per-arc table; increasing its boundary count does not
change the table. Built-in smooth primitives have analytic membership.

For a refinement study, increase interior and boundary resolution together,
resolve narrow gaps and curvature, test independent clouds, and measure geometry
error separately from PDE error when it can become significant.

## Padua interior points with existing boundaries

Use `method="padua"` and a positive polynomial `degree` with either sampled
`Border` objects or a `ParametricDomain`. Omit `interior`: the degree determines
the candidate grid and the domain determines how many interior nodes remain.

```python
from rbflab import geometry, meshgen

domain = geometry.Annulus(inner_radius=.35, outer_radius=1.)
cloud = meshgen.generate(domain, method="padua", degree=28,
                         boundary={"inner": 40, "outer": 80})
# For sampled borders, keep wall(n) and omit boundary:
# cloud = meshgen.generate(wall(80), method="padua", degree=28)
```

The first-family reference grid has `(degree+1)*(degree+2)//2` nodes on
`[-1, 1]^2` (Koushki et al., 2020, section 3.1,
DOI: `10.1016/j.aej.2020.04.047`). `meshgen.padua_points(degree)` returns
this complete grid. Generation scales it to the domain's axis-aligned bounding
box and retains strictly interior nodes, excluding holes and honoring
`boundary_distance`. Reference-edge nodes are excluded; boundary nodes come
from the existing sampler: `Border(n)` retains its parameter sampling and
parametric domains retain their arc-length sampling, labels, and normals.
Internal border interfaces and region groups remain available.

This clipped interior grid plus separately sampled boundaries is a Padua-based
point cloud, not the complete polynomial interpolation set. It is different
from the paper's square-to-disk mapping; no Padua interpolation optimality or
PDE accuracy guarantee is claimed on general domains. It is deterministic,
ignores `seed`, creates no triangle connectivity, and supports only 2D.
`max_candidates` limits the full reference grid size before allocation.

## Experimental mapped Padua points concentrated near holes

Use `meshgen.mapped_padua` to map nodes into the material instead of clipping a
reference grid. It accepts the same `ParametricDomain`, sampled `Border`
collections, and `RBFMesh` geometry, with one outer contour and one or more
separated holes:

This radial construction forms visible rays and can leave unevenly spaced
interior nodes. Prefer the boundary-driven triangulation below when the goal
is an unstructured distribution with smooth local refinement. Clipped Padua
generation remains available for geometry that the radial map rejects.

```python
domain = geometry.Annulus(inner_radius=.35, outer_radius=1.)
cloud = meshgen.mapped_padua(
    domain, degree=20, clustering=2.,
    boundary={"inner": 40, "outer": 80},
)
# For oriented Border geometry:
# cloud = meshgen.mapped_padua([outer(80), hole(-40)], degree=20, clustering=2.)
```

`degree` is the reference Padua degree **per hole patch**. Physical boundary
samples retain their existing coordinates, labels and normals. In particular,
`Border(n)` still samples the supplied parameter and `ParametricBoundary`
samples by arc length. The **interior map uses arc length for either input**.

The generator partitions the exterior into nearest-center cells, one for each
hole. Each cell must contain its complete hole. Hole centers default to their
polygon centroids; `centers=[(x0, y0), ...]` overrides them in hole-contour order.
Both a hole and its cell must be strictly star-shaped about that center: every
ray must encounter each contour once. Arbitrary non-star-shaped geometry,
overlapping material contours, and internal material interfaces are unsupported
and raise an error; there is no fallback to clipping.

For a hole curve `gamma(s)` in normalized arc length, let `B(s)` be the point
where the ray from its center through `gamma(s)` meets the cell boundary.
For reference Padua coordinates `(xi, eta)`, use
`s = arccos(xi)/pi` and `r = (1-eta)/2`. The inverse cosine removes artificial
clustering at the periodic seam, giving uniformly spaced arc-length columns
around each hole while keeping the staggered Padua radial coordinates:

\[
F(s,r) = \gamma(s) + g_\beta(r)\,[B(s)-\gamma(s)],\qquad
g_\beta(r)=\frac{\exp(\beta r)-1}{\exp(\beta)-1}.
\]

`clustering=0` uses `g(r)=r`; increasing it toward the supported maximum of 8
compresses interior points toward holes. The directions are radial from each
hole's center, not generally perpendicular to its curve. The layer spans the
entire surrounding cell. Shared cell dividers receive a single set of samples;
their junctions are merged. The angular seam is half-open, and physical boundary
samples replace the radial endpoint nodes. These choices produce a mapped
Padua-based cloud, not a complete polynomial interpolation set on the material.

`cloud.regions["patch_0"]`, etc., record cell membership, including shared nodes.
Dividers are interior points, not boundary-condition groups or transmission
interfaces. No triangle connectivity is produced.

The map checks star-shapedness, cell coverage, overlap, hole containment and
node separation. `min_spacing=.001` optionally rejects a cloud with closer
nodes; it never thins the cloud to achieve the threshold. Curves are represented
by the existing dense tables (2048 segments per arc), including for border
inputs. These checks apply at that geometric resolution; they cannot certify
unresolved features. Strong clustering can worsen RBF conditioning, and mapped
Padua points do not automatically inherit polynomial interpolation guarantees.

Run `python examples/padua_mapped_holes.py` from the repository for a curved
exterior with three elliptical holes, showing clipped, linearly mapped, and
graded clouds at comparable total node counts. Its printed metric is the
fraction of interior nodes within 0.12 physical units of a hole; it measures
concentration, not PDE accuracy.

## Boundary-driven triangulation

For an unstructured distribution with gradual refinement near densely sampled
curves, use `meshgen.triangulate`. This follows the boundary-count workflow of
[FreeFEM's buildmesh](https://doc.freefem.org/documentation/mesh-generation.html)
using Triangle through MeshPy, rather than the FreeFEM/BAMG implementation.
Install the optional dependency with `python -m pip install "rbflab[triangulation]"`.
MeshPy and Triangle retain their own [upstream licenses](https://documen.tician.de/meshpy/faq.html).

```python
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
