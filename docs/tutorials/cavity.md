# Lid-driven cavity: a custom Navier–Stokes algorithm

**You will learn:** to reuse RBF-FD operators inside an algorithm that is larger than a single linear PDE solve. Prerequisites: [heat matrices](heat-equation.md), [mixed boundaries](mixed-boundary.md), and [time stepping](../theory/time-discretization.md).

The example advances a small lid-driven cavity using the maintained solver in `examples/navier_stokes_cavity.py`. The tutorial wrapper chooses cloud resolution, Reynolds number, final time, and an optional animation path; it then records velocity, pressure, and discrete divergence. The full solver shows the intermediate velocity and pressure-correction steps. This is a **numerical showcase**, not a validated general-purpose Navier–Stokes solver or a reproduction of a published high-Reynolds benchmark.

```python
--8<-- "examples/tutorials/cavity.py"
```

[Download the runnable script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/cavity.py).

Run a bounded case with `python -m examples.tutorials.cavity --cells 6 --end 0.005`. The verified example stored five time levels, 120 two-component velocity values per state, and 49 pressure nodes. Its reported maximum discrete divergence was \(7.925\times10^{-4}\), with zero pressure mean. These diagnostics are specific to the case and do not establish mesh convergence, a divergence-free continuum field, or correct high-Reynolds dynamics.

For a visual result install the optional examples dependencies and pass `--output outputs/cavity_tutorial.gif`. The [visualization guide](../VISUALIZATION.md) describes plotting choices. Repeat with more nodes and a shorter time step; compare kinetic energy, divergence, and velocity profiles. See the [full maintained algorithm](https://github.com/LDBreton/RBFLAB/blob/main/examples/navier_stokes_cavity.py) and [error measures](../theory/errors.md).
