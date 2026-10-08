"""Bind and evaluate an IMQ-like symbolic kernel at runtime.

Run: python -m examples.tutorials.custom_kernel
"""
import numpy as np
import sympy as sp
import rbflab as rbf


def run():
    s = sp.Symbol("s", nonnegative=True)  # s is squared Euclidean distance.
    c = sp.Symbol("c", positive=True)
    family = rbf.Kernel((1+c*s)**sp.Rational(-1, 2), s, (c,))
    kernel = family(c="2")
    centers = rbf.unit_box_grid(4).points
    values = np.sin(centers[:, 0]) + np.cos(centers[:, 1])
    solution = rbf.interpolate(kernel, centers, values)
    query = np.array([[0.27, 0.42], [0.61, 0.35]])
    expected = np.sin(query[:, 0]) + np.cos(query[:, 1])
    error = float(np.max(np.abs(solution.evaluate(query)-expected)))
    value_at_origin = float(kernel.derivative([[0., 0.]], (0, 0))[0])
    print(f"parameters={kernel.values}, kernel_at_origin={value_at_origin:g}, "
          f"off_node_max_error={error:.3e}")
    return error


if __name__ == "__main__":
    run()
