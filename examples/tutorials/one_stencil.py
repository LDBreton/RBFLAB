"""Inspect and verify one augmented RBF-FD Laplacian stencil.

Run: python -m examples.tutorials.one_stencil
"""
import numpy as np
import rbflab as rbf


def run(cells=5, stencil_size=20):
    # --8<-- [start:geometry]
    cloud = rbf.unit_box_grid(cells)
    target_index = int(np.argmin(np.linalg.norm(cloud.points - [0.5, 0.5], axis=1)))
    target = cloud.points[[target_index]]
    # --8<-- [end:geometry]
    # --8<-- [start:operator]
    method = rbf.RBFFD(
        spaces={"u": rbf.ScalarSpace(rbf.PHS(5), 2)},
        stencil_size=min(stencil_size, len(cloud.points)),
        local_backend=rbf.PythonBackend(compute_condition=False),
    )
    op = method.operators(
        source=cloud.points, targets=target, space="u",
        operators={"lap": rbf.Laplacian(2)},
    ).lap
    # --8<-- [end:operator]
    # --8<-- [start:inspect]
    local = op.local(0)
    reconstructed = op.reconstruct_local(0)
    # --8<-- [end:inspect]
    # --8<-- [start:weights]
    points = cloud.points[local.indices]
    weights = local.weights[:, 0]
    augmented = np.r_[weights, local.multipliers[:, 0]]
    gram_residual = float(np.linalg.norm(
        reconstructed.matrix.T @ augmented - reconstructed.rhs[:, 0],
        ord=np.inf,
    ))
    constant_response = float(weights.sum())
    quadratic_response = float(weights @ np.sum(points**2, axis=1))
    # --8<-- [end:weights]
    # --8<-- [start:direct]
    direct = rbf.rbf_fd_weights(
        rbf.PHS(5), points, target[0], rbf.Laplacian(2),
        polynomial_degree=2,
    )
    # --8<-- [end:direct]
    direct_difference = float(np.max(np.abs(weights - np.asarray(direct).reshape(-1))))
    radius = float(np.max(np.linalg.norm(points - target[0], axis=1)))

    print(f"target={target[0]}, stencil_nodes={len(points)}, "
          f"polynomial_multipliers={len(local.multipliers)}, radius={radius:.3f}")
    print(f"local_matrix={reconstructed.matrix.shape}, "
          f"global_row={op.matrix.shape}, nonzeros={op.matrix.nnz}")
    print(f"constant={constant_response:.3e} (expected 0), "
          f"lap(x^2+y^2)={quadratic_response:.12g} (expected 4)")
    print(f"local_equation_residual={gram_residual:.3e}, "
          f"direct_weight_difference={direct_difference:.3e}")
    return dict(constant=constant_response, quadratic=quadratic_response,
                gram_residual=gram_residual, direct_difference=direct_difference)


if __name__ == "__main__":
    run()
