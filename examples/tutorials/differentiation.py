"""Differentiate an interpolant, then construct reusable local derivative maps."""
import numpy as np
import rbflab as rbf


def run():
    # --8<-- [start:data]
    rng = np.random.default_rng(7)
    centers = rng.uniform(-1., 1., (64, 2))
    values = np.sin(centers[:, 0]) + np.cos(centers[:, 1])
    query = np.array([[.1, .2], [-.3, .4], [.5, -.2]])
    interpolant = rbf.interpolate(rbf.PHS(5), centers, values, polynomial_degree=2)
    # --8<-- [end:data]
    # --8<-- [start:derivatives]
    dx = rbf.Derivative(0, dimension=2)
    dy = rbf.Derivative(1, dimension=2)
    gradient = np.column_stack((interpolant.evaluate(query, dx),
                               interpolant.evaluate(query, dy)))
    laplacian = interpolant.evaluate(query, rbf.Laplacian(2))
    # --8<-- [end:derivatives]
    # --8<-- [start:maps]
    method = rbf.RBFFD(rbf.PHS(5), stencil_size=20, polynomial_degree=2)
    ops = method.operators(source=centers, targets=query,
                           operators={"dx": dx, "dy": dy, "lap": rbf.Laplacian(2)})
    local_gradient = np.column_stack((ops.dx @ values, ops["dy"] @ values))
    local_laplacian = ops.lap @ values
    # --8<-- [end:maps]
    truth = np.column_stack((np.cos(query[:, 0]), -np.sin(query[:, 1])))
    polynomial = 1+centers[:, 0]+centers[:, 1]**2
    reproduction = float(np.max(np.abs(ops.lap @ polynomial-2)))
    result = {"gradient_error": float(np.max(np.abs(gradient-truth))),
              "local_gradient_error": float(np.max(np.abs(local_gradient-truth))),
              "polynomial_laplacian_error": reproduction}
    print(f"gradient={gradient.shape}, laplacian={laplacian.shape}, local matrix={ops.lap.shape}")
    print(result)
    return result


if __name__ == "__main__":
    run()
