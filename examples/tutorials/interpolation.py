"""Interpolate scalar data and check polynomial reproduction.

Run: python -m examples.tutorials.interpolation
"""
import numpy as np
import rbflab as rbf


def run():
    cloud = rbf.unit_box_grid(4)
    centers = cloud.points
    query = np.random.default_rng(7).uniform(0.1, 0.9, (30, 2))
    smooth = lambda pts: np.sin(np.pi*pts[:, 0])*np.cos(np.pi*pts[:, 1])
    quadratic = lambda pts: 1 + pts[:, 0] + pts[:, 1]**2
    imq_solution = rbf.interpolate(rbf.IMQ(2), centers, smooth(centers))
    phs_solution = rbf.interpolate(rbf.PHS(5), centers, quadratic(centers),
                                   polynomial_degree=2)
    imq_error = float(np.max(np.abs(imq_solution.evaluate(query)-smooth(query))))
    phs_error = float(np.max(np.abs(phs_solution.evaluate(query)-quadratic(query))))
    gram = rbf.IMQ(2).matrix(centers, centers)
    print(f"centers={len(centers)}, Gram={gram.shape}, IMQ_error={imq_error:.3e}, "
          f"PHS5_quadratic_error={phs_error:.3e}")
    return imq_error, phs_error


if __name__ == "__main__":
    run()
