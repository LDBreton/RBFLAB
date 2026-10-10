"""Interpolate the same scattered data with IMQ and polynomial-augmented PHS.

Run: python -m examples.tutorials.interpolation
"""
import numpy as np
import rbflab as rbf


def run():
    # --8<-- [start:data]
    rng = np.random.default_rng(7)
    centers = rng.uniform(-1., 1., (64, 2))  # shape (N, dimension)
    values = np.sin(centers[:, 0]) + np.cos(centers[:, 1])  # shape (N,)
    # --8<-- [end:data]
    # --8<-- [start:interpolants]
    imq = rbf.interpolate(rbf.IMQ(2), centers, values)
    phs = rbf.interpolate(rbf.PHS(5), centers, values, polynomial_degree=2)
    # --8<-- [end:interpolants]
    # --8<-- [start:evaluate]
    query = np.array([[.1, .2], [-.3, .4], [.5, -.2]])  # shape (M, 2)
    predicted = phs.evaluate(query)  # shape (M,)
    kernel_matrix = rbf.IMQ(2).matrix(centers, centers)  # shape (N, N)
    # --8<-- [end:evaluate]
    expected = np.sin(query[:, 0]) + np.cos(query[:, 1])
    imq_error = float(np.max(np.abs(imq.evaluate(query)-expected)))
    phs_error = float(np.max(np.abs(predicted-expected)))
    # --8<-- [start:polynomial]
    quadratic = 1 + centers[:, 0] + centers[:, 1]**2
    polynomial_fit = rbf.interpolate(rbf.PHS(5), centers, quadratic,
                                     polynomial_degree=2)
    reproduced = polynomial_fit.evaluate(query)
    # --8<-- [end:polynomial]
    polynomial_error = float(np.max(np.abs(reproduced-(1+query[:, 0]+query[:, 1]**2))))
    print(f"data={values.shape}, query values={predicted.shape}, kernel matrix={kernel_matrix.shape}")
    print(f"Same-field sampled error: IMQ={imq_error:.3e}, PHS={phs_error:.3e}")
    print(f"Separate polynomial reproduction check: {polynomial_error:.3e}")
    return imq_error, polynomial_error


if __name__ == "__main__":
    run()
