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


def test_first_problem_on_ellipse():
    from examples.tutorials.first_problem import run
    assert run() < 2e-4


def test_heat_page_standalone_symbolic_recipe():
    from pathlib import Path
    import re
    page = Path(__file__).resolve().parents[1] / "docs/tutorials/flower-heat.md"
    code = re.search(r"```python\n(.*?)\n```", page.read_text(encoding="utf-8"), re.S).group(1)
    namespace = {}
    exec(compile(code, str(page), "exec"), namespace)
    cloud, trajectory = namespace["cloud"], namespace["trajectory"]
    x, y = cloud.points.T
    truth = np.exp(-1)*(np.cos(x)*np.cos(y)+np.sin(2*x)/4)
    assert np.max(np.abs(trajectory.final.evaluate(cloud.points)-truth)) < .004


def test_teaching_derivative_maps_reproduce_polynomials():
    from examples.tutorials.differentiation import run
    result = run()
    assert result["polynomial_laplacian_error"] < 1e-10
    assert result["gradient_error"] < .01


def test_teaching_global_and_lhi_rows_match_assembly():
    from examples.tutorials.global_collocation import run as global_run
    from examples.tutorials.lhi_construction import run as lhi_run
    global_result, local_result = global_run(), lhi_run()
    assert global_result["assembly_difference"] < 1e-11
    assert global_result["evaluation_difference"] < 1e-11
    assert local_result["center_excluded"]
    assert local_result["row_difference"] < 1e-11
    assert local_result["rhs_difference"] < 1e-11


def test_teaching_custom_matrix_matches_symbolic_equation():
    from examples.tutorials.custom_assembly import run
    result = run()
    assert result["route_difference"] < 1e-9
    assert result["error"] < .002


def test_teaching_heat_has_identical_symbolic_and_matrix_recipe():
    from examples.tutorials.heat_equation import run
    result = run()
    assert result["route_difference"] < 1e-10
    assert result["error"] < .02


def test_teaching_symbolic_boundary_variations():
    from examples.tutorials.symbolic_pde import run
    assert run() < .001
    assert run(robin=True) < .001


def test_teaching_coupled_spaces_and_pressure_reference():
    from examples.tutorials.stokes_construction import run
    assert run() < .02


def test_functional_api_tutorials_and_independent_centers():
    from examples.tutorials import local_approximation, lhi_matrices, lhi_centers
    maps = local_approximation.run()
    assert maps["interpolation_error"] < 1e-10
    assert maps["laplacian_error"] < 1e-10
    lhi = lhi_matrices.run()
    assert lhi["stationary_error"] < 1e-10
    assert lhi["heat_error"] < .004
    assert lhi_centers.run() < 1e-10
