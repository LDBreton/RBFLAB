# Steady and unsteady Stokes in a divergence-free space

**You will learn:** to declare coupled velocity and pressure fields, assign a divergence-free velocity space, and inspect pressure-gradient error. Prerequisites: [divergence-free kernels](../theory/divergence-free.md) and [time discretization](../theory/time-discretization.md).

Steady Stokes momentum is

$$
-\mu\Delta\boldsymbol u+\nabla p=\boldsymbol f.
$$

Unsteady momentum adds \(\partial_t\boldsymbol u\). The manufactured example uses \(\mu=1\), \(\boldsymbol u=(e^{-t}y(1-y),0)\), and \(p=-2x\); the steady version omits the exponential. Boundary values and forcing are derived from these fields.

??? example "Complete runnable script"

    ```python
    --8<-- "examples/tutorials/stokes_spaces.py"
    ```

[Download the runnable script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/stokes_spaces.py).

Run `python -m examples.tutorials.stokes_spaces`. Velocity uses the matrix kernel \(\nabla\nabla^{\mathsf T}\phi-I\Delta\phi\), which is divergence-free by construction. The symbolic system supplies momentum equations but no redundant divergence equation. Pressure is defined up to a constant, so the example checks \(\nabla p\), not an absolute pressure value.

For the 25-node cloud, both global systems were \(61\times61\). A verified steady query had velocity error \(1.482\times10^{-17}\), pressure-gradient error \(4.521\times10^{-16}\), and reported divergence zero. At \(t=0.03\), the short BDF2 run had velocity error \(1.271\times10^{-7}\), pressure-gradient error \(3.296\times10^{-6}\), and reported divergence zero. These are **single-query** errors, not domain norms.

BDF2 starts with one backward Euler step. The initial condition sets velocity, not pressure. Try changing \(\mu\) and rederive forcing. See [equations and spaces](../api/equations-spaces.md).
