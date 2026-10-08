"""Small numerical checks for the public teaching examples."""

import numpy as np

from examples.tutorials import one_stencil, heat_matrices, mixed_boundary
from examples.tutorials import interpolation, scalar_3d, stokes_spaces
from examples.tutorials import compare_methods, custom_kernel


def test_one_stencil_reproduces_quadratic():
    result = one_stencil.run()
    assert abs(result["constant"]) < 1e-11
    assert abs(result["quadratic"] - 4) < 1e-11
    assert result["gram_residual"] < 1e-10
    assert result["direct_difference"] < 1e-11


def test_heat_matrix_matches_symbolic_and_includes_boundary_data():
    errors = heat_matrices.run(cells=4, dt=0.01, steps=2)
    assert abs(errors["rbf_fd_matrix"] - errors["symbolic_rbffd"]) < 1e-11
    assert errors["rbf_fd_matrix"] < 0.04
    boundary = heat_matrices.run(cells=4, dt=0.01, steps=2,
                                 scheme="bdf2", nonzero_boundary=True)
    assert abs(boundary["rbf_fd_matrix"] - boundary["symbolic_rbffd"]) < 1e-10
    assert boundary["rbf_fd_matrix"] < 0.002


def test_manufactured_polynomials_and_mixed_boundary():
    assert mixed_boundary.run() < 1e-10
    _, phs_error = interpolation.run()
    assert phs_error < 1e-10
    nodal_error, lap_reproduction = scalar_3d.run()
    assert nodal_error < 1e-10
    assert lap_reproduction < 1e-10


def test_divergence_free_stokes_reference_point():
    result = stokes_spaces.run()
    for velocity, gradient, divergence in result.values():
        assert np.isfinite([velocity, gradient, divergence]).all()
        assert velocity < 1e-5
        assert gradient < 1e-4
        assert divergence < 1e-10



def test_global_local_comparison_and_custom_kernel():
    errors = compare_methods.run()
    assert set(errors) == {"global_symmetric", "global_asymmetric", "lhi", "rbffd"}
    assert all(np.isfinite(value) and value < 0.05 for value in errors.values())
    assert custom_kernel.run() < 0.01
