# Lid-driven cavity implementation

The complete explanation now lives in [Build a lid-driven cavity solver](../tutorials/cavity.md).
It derives the three source/target maps, CN/AB2 time step, boundary elimination,
coupled velocity/pressure matrix, pressure mean and compatibility multiplier.

The recorded Re=100 demonstration has 800 velocity nodes and 289 pressure nodes.
Its maximum discrete divergence at time 20 is about 0.025; the tutorial explains
why this is an incompressibility defect rather than a linear-solver residual.

```sh
python -m examples.navier_stokes_cavity --cells 16 --end 20 --output outputs/cavity.gif
```

Use the [short startup check](../tutorials/cavity.md#7-run-a-short-check-before-a-long-animation)
before a longer run.
