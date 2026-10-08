"""Reusable 3D RBF-FD operators, sparse matrices and local reconstruction.

Run: python examples/operators_3d.py --backend python
Optional: --backend torch or --backend cpp
"""
import argparse
import numpy as np
import rbflab as r


def run(backend_name="python"):
    rng = np.random.default_rng(31)
    source = rng.uniform(-1, 1, (55, 3))
    targets = rng.uniform(-.5, .5, (12, 3))
    backends = {
        "python": lambda: r.PythonBackend(compute_condition=False),
        "cpp": lambda: r.CppBackend(threads=2, compute_condition=False),
        "torch": lambda: r.TorchBackend(threads=2, compute_condition=False),
    }
    method = r.RBFFD(
        spaces={"u": r.ScalarSpace(r.PHS(7), 3)},
        stencil_size=30, local_backend=backends[backend_name](),
    )
    ops = method.operators(source=source, targets=targets, space="u",
                           operators={"dx": r.Derivative(0, 3), "lap": r.Laplacian(3)})
    values = source[:, 0]**2 + source[:, 1]**2 + source[:, 2]**2
    dx_error = float(np.max(np.abs(ops.dx @ values - 2*targets[:, 0])))
    lap_error = float(np.max(np.abs(ops.lap @ values - 6)))
    local = ops.dx.reconstruct_local(0)
    print(f"{backend_name}: matrix {ops.dx.matrix.shape}; dx error {dx_error:.3e}; "
          f"Laplacian error {lap_error:.3e}; first local matrix {local.matrix.shape}")
    return dx_error, lap_error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "cpp", "torch"), default="python")
    args = parser.parse_args()
    run(args.backend)
