"""Accuracy checks for the six copyable release tutorials."""
from examples import (symbolic_poisson, symbolic_boundary, symbolic_heat,
                      stokes_spaces, operators_3d, custom_kernel)


def test_symbolic_poisson_methods():
    for method in ("global", "lhi", "rbffd"):
        assert symbolic_poisson.run(method) < .05


def test_symbolic_boundary_and_heat():
    assert symbolic_boundary.run() < .1
    assert symbolic_heat.run() < .02


def test_divergence_free_stokes():
    assert max(stokes_spaces.run()) < 1e-6
    assert max(stokes_spaces.run("lhi")) < 1e-6


def test_3d_operators_and_custom_kernel():
    assert max(operators_3d.run()) < 1e-8
    assert custom_kernel.run() < .05
