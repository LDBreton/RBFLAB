"""Translate reaction-diffusion and Dirichlet/Robin equations into the scalar API."""
import argparse
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry, meshgen


def run(robin=False):
    # --8<-- [start:geometry]
    domain = geometry.Ellipse(a=1.3, b=.8, labels=("wall",))
    cloud = meshgen.generate(domain, interior=200, boundary=80, seed=42)
    # --8<-- [end:geometry]
    # --8<-- [start:field]
    model = rbf.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    alpha = 2
    source = (2+alpha)*sp.sin(x)*sp.cos(y)
    # --8<-- [end:field]
    # --8<-- [start:boundary]
    wall = model.bc("wall", sp.Eq(u, sp.sin(x)*sp.cos(y)))
    # --8<-- [end:boundary]
    if robin:
        # --8<-- [start:robin]
        known_field = sp.sin(x)*sp.cos(y)
        wall = model.bc("wall", sp.Eq(model.normal_derivative(u)+u,
                         model.normal_derivative(known_field)+known_field))
        # --8<-- [end:robin]
    # --8<-- [start:problem]
    problem = model.stationary(sp.Eq(-model.laplacian(u)+alpha*u, source),
                               boundary=[wall])
    # --8<-- [end:problem]
    # --8<-- [start:solve]
    method = rbf.RBFFD(rbf.PHS(5), stencil_size=35, polynomial_degree=3,
                       stencil_policy=rbf.StencilPolicy(scaling="local"))
    system = method.assemble(problem, cloud)
    solution = system.solve()
    # --8<-- [end:solve]
    # --8<-- [start:evaluate]
    query = np.array([[0., 0.], [.3, .2], [-.4, .1]])
    values = solution.evaluate(query)
    dx = solution.evaluate(query, rbf.Derivative(0))
    # --8<-- [end:evaluate]
    error = float(np.max(np.abs(values-np.sin(query[:, 0])*np.cos(query[:, 1]))))
    print(f"boundary={'Robin' if robin else 'Dirichlet'}, system={system.matrix.shape}")
    print(f"query values={values}, derivative shape={dx.shape}, sampled error={error:.3e}")
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robin", action="store_true")
    run(**vars(parser.parse_args()))
