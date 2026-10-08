# Re=100 lid-driven cavity

Run the [standalone example](https://github.com/LDBreton/RBFLAB/blob/main/examples/navier_stokes_cavity.py) from a
source checkout after installing the plotting extra:

```sh
python -m pip install ".[examples]"
python -m examples.navier_stokes_cavity
```

The command writes `docs/assets/navier_stokes_cavity.gif` and a final-frame
PNG. Use `--backend cpp` after building the optional native backend. The
default Python backend needs no mesher or compiler.

The unit square has a lid speed of one and viscosity 0.01. Pressure lives at
grid vertices; velocity lives at triangle edge midpoints. RBF-FD uses PHS7,
degree-three polynomials, and 28-point stencils. Convection uses AB2 after an
Euler first step; diffusion uses Crank–Nicolson. A coupled sparse system solves
velocity and pressure together with a zero-mean pressure gauge.

The 16-cell example has 800 velocity points and 289 pressure points. Its final
maximum discrete divergence at t=20 is about 0.025. The numerical scheme uses
a compatibility multiplier, so this is a visible incompressibility defect, not
roundoff. Its final velocity differs from an archived research run on almost
identical points by RMS 0.00210. The animation demonstrates the workflow; it
does not validate pressure or general Navier–Stokes accuracy.

The GIF linearly interpolates nodal velocities onto a display grid. That grid
does not participate in the RBF-FD solve.
