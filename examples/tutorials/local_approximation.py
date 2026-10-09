"""Build interpolation and derivative maps from a declared approximation space."""
import argparse
import numpy as np
import rbflab as rbf


def run(backend="python"):
    implementation = {"python": rbf.PythonBackend, "cpp": rbf.CppBackend,
                      "torch": rbf.TorchBackend}[backend](compute_condition=False)
    # --8<-- [start:construction]
    cloud = rbf.geometry.unit_box_grid(6)
    X = cloud.points
    Y = np.array([[.3, .4], [.6, .7], [.45, .55]])
    space = rbf.ScalarSpace(rbf.PHS(5), polynomial_degree=2)
    source = {"u": rbf.Samples(X, size=20)}
    local = rbf.LocalApproximation(
        source=source,
        trial=space.representers(source),
        backend=implementation,
        stencil_policy=rbf.StencilPolicy(scaling="local"),
    )
    ops = local.operators(
        targets=Y,
        operators={"value": rbf.Identity(2), "lap": rbf.Laplacian(2)},
    )
    # --8<-- [end:construction]
    # --8<-- [start:apply]
    U = 1 + np.sum(X**2, axis=1)
    reconstructed = ops.value @ {"u": U}
    laplacian = ops.lap["u"] @ U
    D = ops.lap["u"].matrix  # rows: targets, columns: source values
    row = ops.lap.local(0)
    algebra = ops.lap.reconstruct_local(0)
    # --8<-- [end:apply]
    # Explicitly leave Torch arithmetic only for this NumPy error report.
    if backend == "torch":
        reconstructed = reconstructed.detach().cpu().numpy()
        laplacian = laplacian.detach().cpu().numpy()
    error = float(np.max(np.abs(laplacian - 4)))
    interpolation_error = float(np.max(np.abs(reconstructed - (1 + np.sum(Y**2, axis=1)))))
    print(f"source={len(X)}, targets={len(Y)}, sparse map={D.shape}")
    print(f"quadratic interpolation error={interpolation_error:.3e}, Laplacian error={error:.3e}")
    return {"interpolation_error": interpolation_error, "laplacian_error": error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "cpp", "torch"), default="python")
    run(**vars(parser.parse_args()))
