# Plotting numerical results

Install the optional graphics packages with
`python -m pip install "rbflab[examples]"`. RBFLAB loads Matplotlib only when a
plotting function runs, so the ordinary solver install stays small.

```python
from rbflab import viz

fig, ax = viz.plot_scalar(solution, title="Temperature")
fig.savefig("temperature.png", dpi=160)

fig, ax = viz.plot_velocity(stokes_solution, title="Flow")
fig.savefig("flow.png", dpi=160)

viz.animate_scalar(trajectory, "heat.gif", fps=9, every=2)

# A time sequence of 2D nodal velocity fields:
viz.animate_velocity_samples(points, times, velocities, "flow.gif")
```

`plot_scalar` accepts a 2D solution with `evaluate(points)`;
`plot_velocity` accepts a 2D velocity solution with `velocity(points)`;
`animate_scalar` accepts an evolution trajectory and writes a GIF using Pillow.
`animate_velocity_samples` accepts arrays of point coordinates, times, and
velocities; it can also write a final PNG with `poster="flow.png"`.
Each plot can set `bounds`, `resolution`, and `title`. `plot_scalar` returns
Matplotlib figure and axes so callers can customize them.

The regular plotting grid only samples a computed solution. It does not add
nodes to the collocation or RBF-FD discretization. An image is a view of the
numerical field, not evidence of error convergence; use the numerical checks
in the example scripts for that.

From a source checkout, regenerate the README figures with:

```sh
python -m examples.make_gallery
```

The generator solves a localized heat pulse and the steady Stokes example,
checks that the heat peak decreases without significant negative undershoot,
then writes `docs/assets/heat_diffusion.png`, `heat_diffusion.gif`, and
`stokes_velocity.png`.

From a source checkout, the [Re=100 cavity example](https://github.com/LDBreton/RBFLAB/blob/main/examples/navier_stokes_cavity.py)
makes the README's flow GIF and PNG with `python -m examples.navier_stokes_cavity`.
Its default backend is Python; `--backend cpp` uses the optional native backend.


## Curved domains, holes and 3D slices

```python
fig, ax = viz.plot_cloud(cloud, normals=True)
fig, ax = viz.plot_scalar(solution, domain=domain)
fig, ax = viz.plot_velocity(flow_solution, domain=domain)
viz.animate_scalar(trajectory, "heat.gif", domain=domain)
# For a scalar 3D solution:
fig, ax = viz.plot_slice(solution_3d, ball_domain, axis=2, coordinate=0)
```

The field helpers mask the exterior and holes **before evaluation**. They sample
the numerical reconstruction without inventing triangulation. The older
`animate_velocity_samples` helper interpolates nodal arrays over a rectangular
window; use it only when that display domain is appropriate (such as the cavity).
It is not a hole-aware curved-domain renderer.

Regenerate the curved gallery with `python -m examples.make_geometry_gallery`;
add `--backend cpp` or `--backend torch` after installing the relevant backend.
