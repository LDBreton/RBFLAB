"""A first symbolic Poisson problem on the unit square; no helper module required."""
import argparse
from pathlib import Path
import numpy as np
import sympy as sp
import rbflab as rbf
from rbflab import geometry


def run(plot=None):
    # --8<-- [start:geometry]
    cloud = geometry.unit_box_grid(10)  # 121 nodes: 81 interior, 40 boundary
    # --8<-- [end:geometry]

    # --8<-- [start:equation]
    model = rbf.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    exact = sp.sin(x)*sp.cos(y)
    problem = model.stationary(
        sp.Eq(-model.laplacian(u), 2*exact),
        boundary=[model.bc("boundary", sp.Eq(u, exact))],
    )
    # --8<-- [end:equation]

    # --8<-- [start:solve]
    method = rbf.RBFFD(rbf.PHS(5), stencil_size=35, polynomial_degree=3,
        stencil_policy=rbf.StencilPolicy(scaling="local"))
    system = method.assemble(problem, cloud)
    solution = system.solve()
    # --8<-- [end:solve]

    # --8<-- [start:check]
    query = np.random.default_rng(17).uniform(.05, .95, (100, 2))
    reference = np.sin(query[:, 0])*np.cos(query[:, 1])
    error = float(np.max(np.abs(solution.evaluate(query)-reference)))
    print(f"nodes={len(cloud.points)}, sampled maximum error={error:.6e}")
    # --8<-- [end:check]
    if plot:
        from rbflab import viz
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#0b1220",
                                 constrained_layout=True)
        viz.plot_cloud(cloud, ax=axes[0], title="121 nodes / square boundary")
        viz.plot_scalar(solution, bounds=((0, 1), (0, 1)), resolution=140, ax=axes[1],
                        title="Computed Poisson solution")
        output = Path(plot)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160)
        plt.close(fig)
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot")
    run(**vars(parser.parse_args()))
