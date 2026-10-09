# Geometry cookbook

These recipes use `from rbflab import geometry, meshgen, viz`.
They produce labeled point clouds directly; Gmsh is optional.
The `Polygon`, `Rectangle` and `with_holes` helpers require RBFLAB 0.3 or newer.

![Four generated geometries and their nodes](../assets/geometry_gallery.png)

## A channel with two obstacles

Use a separate label for each wall and obstacle. The names later identify
boundary conditions; a name such as `inlet` alone imposes no physics.

```python
from rbflab import geometry, meshgen, viz

domain = geometry.with_holes(
    geometry.Rectangle(4., 1.8, labels=("bottom", "outlet", "top", "inlet")),
    {
        "cylinder": geometry.Disk(.32, center=(-.8, 0)),
        "ellipse": geometry.Ellipse(.4, .2, center=(.7, 0), angle=.45),
    },
)
cloud = meshgen.generate(domain, interior=380, boundary=180, seed=42)
fig, ax = viz.plot_cloud(cloud, normals=True)
```

`boundary=180` allocates the total approximately in proportion to arc length.
For explicit resolution on each boundary, pass all labels:

```python
cloud = meshgen.generate(domain, interior=500, boundary={
    "bottom": 48, "outlet": 24, "top": 48, "inlet": 24,
    "cylinder": 40, "ellipse": 40,
}, seed=42)
```

The resulting normals point **out of the fluid**, including into each obstacle.
`with_holes` leaves its input domains unchanged and rejects touching, overlapping
or exterior holes. It preserves any existing holes. Curved-contour topology and
membership are checked using polygon approximations, with 2048 segments per arc;
the sampled curves and tangents remain analytic. This is not arbitrary CAD Boolean
geometry. Narrow gaps still need enough nodes to resolve the PDE.

## A perforated flower

```python
domain = geometry.with_holes(
    geometry.Flower(petals=6, amplitude=.16),
    {"hole": geometry.Disk(.38)},
)
cloud = meshgen.generate(domain, interior=380,
                         boundary={"boundary": 132, "hole": 48}, seed=42)
```

A single-arc hole takes its dictionary key as its boundary label. A polygonal
hole has several edges and receives labels such as `slot/edge_0`, `slot/edge_1`.
The exterior labels are preserved. Use `list(domain.boundaries)` to inspect them.

## A concave L-shaped domain

```python
domain = geometry.Polygon(
    [(0,0), (2,0), (2,.7), (.8,.7), (.8,2), (0,2)],
    labels=("base", "exit", "inner_floor", "inner_wall", "entry", "outer_wall"),
)
cloud = meshgen.generate(domain, interior=380, boundary=180, seed=42)
```

Vertices follow the contour in either orientation. Self-intersections and repeated
vertices are rejected; a repeated closing vertex is allowed. Edge `j` runs from
vertex `j` to the next vertex. A corner belongs to the edge **starting** there,
and its stored normal is that edge's outward normal. It is not an averaged corner
normal. Choose corner boundary rows deliberately when prescribing fluxes.
A re-entrant corner can reduce PDE regularity; a visually good cloud does not
remove that mathematical difficulty.

## Rotate a rectangle

```python
domain = geometry.Rectangle(2., 1., center=(.3, -.2), angle=.4)
```

The default labels are `bottom`, `right`, `top`, `left` in the rectangle's local
coordinates; they rotate with it. Angles are counterclockwise, in radians.

## Generate a 3D volume

```python
domain = geometry.Sphere(radius=1., boundary_label="surface")
cloud = meshgen.generate(domain, interior=550, boundary=500, seed=42)
fig, ax = viz.plot_cloud(cloud)
```

The gallery hides the front of the sphere **for display only** so that interior
nodes can be seen. The computational cloud is the full ball. See the
[3D Poisson tutorial](../tutorials/ball.md) for a numerical example and
[custom domains](custom-domains.md) for implicit regions and parametric surfaces.

## From these nodes to an equation

The [Poisson tutorial](../tutorials/perforated-poisson.md) solves on an ellipse with
two holes, with Python, C++ or PyTorch local assembly. It reports nodal and
independent off-node errors against a known nonpolynomial solution.

![A generated cloud and its computed Poisson solution](../assets/perforated_poisson.png)

The gallery generator is `examples/geometry_gallery.py`; run
`python -m examples.geometry_gallery` from a source checkout with the plotting
extra installed. These are real generated clouds and a computed solution.
The sampler does not promise minimum inter-node separation. Inspect
`meshgen.quality`, independent seeds and refinement before trusting a PDE result.
