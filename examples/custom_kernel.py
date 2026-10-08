"""Symbolic kernel with runtime parameters and ordinary interpolation.

Run: python examples/custom_kernel.py
Optional source-build C++ compilation: python examples/custom_kernel.py --compile
"""
import argparse
import numpy as np
import sympy as sp
import rbflab as r


def run(compile_kernel=False):
    s = sp.Symbol("s", nonnegative=True)  # squared distance
    c = sp.Symbol("c", positive=True)
    family = r.Kernel((1 + c*s)**sp.Rational(-1, 2), s, (c,))
    kernel = family(c="2")
    rng = np.random.default_rng(13)
    centers = rng.uniform(-1, 1, (30, 2))
    query = rng.uniform(-.8, .8, (10, 2))
    truth = lambda points: np.sin(points[:, 0]) + np.cos(points[:, 1])
    solution = r.interpolate(kernel, centers, truth(centers))
    error = float(np.max(np.abs(solution.evaluate(query)-truth(query))))
    origin = float(kernel.derivative([[0., 0.]], (0, 0))[0])
    print(f"custom IMQ: held-out error {error:.3e}; K(0)={origin:.1f}")
    if compile_kernel:
        first = family.compile(dimension=2, derivative_order=2)
        second = family.compile(dimension=2, derivative_order=2)
        assert first.directory == second.directory
        print(f"C++ artifact reused from {second.directory}")
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true")
    run(parser.parse_args().compile)
