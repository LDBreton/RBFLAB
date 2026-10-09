"""Build steady annular Stokes flow through symbolic fields and kernel spaces."""
import argparse
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen


def run(plot=None):
    # --8<-- [start:geometry]
    domain = geometry.Annulus(.5, 1.)
    cloud = meshgen.generate(domain, interior=70,
                             boundary={"inner": 32, "outer": 48}, seed=42)
    # --8<-- [end:geometry]
    # --8<-- [start:fields]
    model = rbf.SymbolicSystem(2, vector_fields=("U",), scalar_fields=("p",))
    U, p = model.fields
    x, y = model.coordinates
    mu = 1
    momentum = -mu*model.laplacian(U) + model.gradient(p)
    # --8<-- [end:fields]
    # --8<-- [start:boundary]
    problem = model.stationary([sp.Eq(momentum, sp.zeros(2, 1))], boundary=[
        model.bc("inner", sp.Eq(U, sp.ImmutableMatrix([-y, x]))),
        model.bc("outer", sp.Eq(U, sp.zeros(2, 1))),
    ])
    # --8<-- [end:boundary]
    # --8<-- [start:spaces]
    spaces = {U: rbf.DivergenceFreeSpace(rbf.IMQ(3), 2),
              p: rbf.PressureSpace(rbf.IMQ(3), 1)}
    method = rbf.GlobalCollocation(spaces=spaces)
    system = method.assemble(problem, cloud)
    solution = system.solve()
    # --8<-- [end:spaces]
    # --8<-- [start:evaluate]
    query = np.array([[.75, 0.], [0., .75], [-.6, .2]])
    velocity = solution.velocity(query)  # shape (M, 2)
    pressure_gradient = solution.pressure_gradient(query)  # shape (M, 2)
    pressure = solution.pressure(query, reference=([.75, 0.], 0.))  # shape (M,)
    # --8<-- [end:evaluate]
    factor = (1/np.sum(query*query, axis=1)-1)/3
    truth = np.column_stack((-query[:, 1]*factor, query[:, 0]*factor))
    error = float(np.max(np.abs(velocity-truth)))
    print(f"velocity={velocity.shape}, pressure={pressure.shape}, pressure gradient={pressure_gradient.shape}")
    print(f"sampled velocity error={error:.3e}")
    if plot:
        from rbflab import viz
        fig, ax = viz.plot_velocity(solution, domain=domain, title="Stokes / rotating inner wall")
        fig.savefig(plot, dpi=160)
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot")
    run(**vars(parser.parse_args()))
