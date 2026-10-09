# Curved-domain validation

The examples report maximum absolute **solution errors** at nodes and at 120
independent interior query points (seed 981). These sampled values are neither
continuous $L^2$ norms nor the sparse-system residual. The Stokes example uses
100 independent queries and reports velocity, divergence and pressure gradient
separately.

Default scalar examples use Float64, Halton seed 42, PHS5 plus degree-three
polynomials, [local coordinate scaling](../theory/conditioning.md#local-stencil-scaling), and 35-node stencils (55 in 3D).

| Example | Total nodes | Sampled off-node maximum error |
|---|---:|---:|
| Annulus, RBF-FD | 384 | 0.003061 |
| Mixed-boundary ellipse | 360 | 0.000271 |
| Flower heat, final t=0.5 | 370 | 0.001076 |
| Ball Poisson, 3D | 480 | 0.000198 |
| Annular Stokes, velocity | 150 | 0.004017 |
| Annular Stokes, pressure gradient | 150 | 0.045711 |

The [complete matched-backend and refinement record](../assets/curved_validation.json)
contains the measured errors and stage timings.

These are reproducible example results, not universal tolerances or method rankings.
C++ and PyTorch runs use identical clouds and parameters; their agreement is
checked separately from accuracy against the analytic solution.

## Independent clouds and refinement

For the annulus, retain the same 35-node policy and kernel while increasing
interior nodes from 120 to 480. Boundary counts increase proportionally to the
square root of the interior count: 222 to 684 total nodes.

| Independent seed | Coarse sampled error | Fine sampled error |
|---:|---:|---:|
| 17 | 0.01904 | 0.00257 |
| 42 | 0.01589 | 0.00359 |
| 73 | 0.00932 | 0.00347 |

All three improve in this test without cloud-specific parameter tuning. Two
levels and three seeds do not establish an asymptotic convergence order or
stability for arbitrary irregular clouds. Halton samples have no enforced
minimum separation. The reference query set is fixed across runs.

Reproduce the study and measured stage timings:

```sh
python -m examples.curved_validation --backends python cpp torch \
    --output outputs/curved_validation.json
```

Generate the displayed figures directly from solved fields:

```sh
python -m examples.make_geometry_gallery --backend cpp
```

The [gallery record](../assets/curved_gallery.json) records the illustrated runs.
Native setup requirements and timing limits are described in
[backend selection](curved-backends.md). The gallery is not a performance benchmark.
