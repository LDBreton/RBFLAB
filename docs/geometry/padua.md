# Padua node distributions

Follow the blocks in order. Start with clipped points; use the experimental mapped recipe when you deliberately want concentration around holes.

## Step 1 — Generate clipped Padua nodes

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

## Step 2 — Plot and count the retained nodes

```python
from rbflab import viz

fig, ax = viz.plot_cloud(cloud)
print("clipped interior:", len(cloud.interior), "total:", len(cloud.points))
```

Increase `degree` to 36 and rerun the first block. Expect a denser reference
pattern, but inspect the retained count instead of treating degree as an exact
material-node count. The boundary counts still total 120.

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


## Step 3 — Tune the mapped recipe

| `mapped_padua` option | Default | Meaning |
|---|---|---|
| `degree` | Required | Positive integer per hole patch; determines candidate resolution |
| `boundary` | `100` for domains | Total or per-label counts; omit for sampled borders |
| `clustering` | `2.` | In `[0, 8]`; zero gives linear radial spacing, larger values concentrate near holes |
| `centers` | Hole centroids | One `(x, y)` center per hole, in contour order; must satisfy star-shapedness |
| `min_spacing` | `0.` | Rejects too-close nodes; does not move or remove them |
| `max_candidates` | `1000000` | Limits reference candidates across patches |

```python
for clustering in (0., 2., 4.):
    graded = meshgen.mapped_padua(domain, degree=20, clustering=clustering,
                                  boundary={"inner": 40, "outer": 80})
    fig, ax = viz.plot_cloud(graded)
    ax.set_title(f"Mapped Padua; clustering={clustering}")
    print(clustering, len(graded.points), meshgen.quality(graded))
```

**Try it:** compare the strip next to the inner circle across the three figures.
Larger clustering should concentrate nodes there. Check the spacing diagnostics
as well: concentration can introduce close neighbors and worsen conditioning.
If the map rejects a concave or off-center hole, verify its geometric restrictions
or use clipped generation / triangulation for that geometry.

