"""Independent polynomial tails for divergence-free LHI Stokes spaces."""
import os
import numpy as np
import pytest
import sympy as sp
import tempfile
import rbflab as r


def _constant_stokes_problem():
    model = r.SymbolicStokes(2)
    velocity_symbol, pressure_symbol = model.fields
    x, y = model.coordinates
    constant = sp.ImmutableMatrix([sp.Rational(5, 4), sp.Rational(-3, 4)])
    momentum = -model.laplacian(velocity_symbol) + model.gradient(pressure_symbol)
    problem = model.stationary(
        [sp.Eq(momentum, sp.zeros(2, 1))],
        boundary=[model.bc("boundary", sp.Eq(velocity_symbol, constant))],
    )
    cloud = r.geometry.unit_box_grid(3)
    return model, problem, cloud, np.asarray(constant, dtype=float).reshape(2)


def _solve(spaces, backend, tmp_path):
    _, problem, cloud, constant = _constant_stokes_problem()
    method = r.LHI(
        spaces=spaces,
        stencil_size=9,
        precision=r.Precision(local_digits=80, global_dtype="mpmath"),
        local_backend=backend(tmp_path),
    )
    return method.assemble(problem, cloud).solve(), cloud, constant


def _hybrid_velocity():
    return r.Hybrid(r.Gaussian("0.5"), r.PHS(7), "1e-5")


def _pressure():
    return r.Hybrid(r.Gaussian("0.0005"), r.PHS(3), "1e-6")


def _smooth_pressure():
    return r.Gaussian("0.0005")


def _assert_mp_close(solution, name, points, expected, tolerance="2e-7"):
    values = getattr(solution, name)(points, extended=True)
    arithmetic = solution.state.system.local_arithmetic
    expected = np.asarray(expected, dtype=object)
    error = max((abs(values[i, j] - arithmetic.number(expected[i, j]))
                 for i in range(values.rows) for j in range(values.cols)),
                default=arithmetic.number(0))
    assert error < arithmetic.number(tolerance)


def test_lhi_independent_velocity_augmentation_reproduces_constants_python(tmp_path):
    model, _, _, _ = _constant_stokes_problem()
    spaces = {
        model.fields[0]: r.DivergenceFreeSpace(_hybrid_velocity(), polynomial_degree=0),
        model.fields[1]: r.PressureSpace(_pressure(), polynomial_degree=None),
    }
    solution, cloud, constant = _solve(
        spaces, lambda _: r.PythonBackend(compute_condition=False), tmp_path
    )

    # Degree-zero solenoidal velocity tails are exactly the two translations;
    # an unaugmented pressure contributes no extra local polynomial columns.
    assert all(len(stencil.polynomial_basis.columns) == 2
               for stencil in solution.state.system.stencils)
    query = np.array([[0.17, 0.23], [0.5, 0.5], [0.82, 0.61]])
    all_points = np.vstack([cloud.points, query])
    _assert_mp_close(solution, "velocity", all_points,
                     np.broadcast_to(constant, all_points.shape))
    _assert_mp_close(solution, "pressure_gradient", query, np.zeros_like(query))

    # Momentum applied to a constant velocity vanishes in every assembled row.
    system = solution.state.system
    ni, nb = len(cloud.interior_indices), len(cloud.boundary_indices)
    for row in range(2 * ni):
        component = row // ni
        total = sum(value * constant[col // ni]
                    for col, value in system.sy_rows[row].items())
        total += sum(value * constant[col // nb]
                     for col, value in system.sb_rows[row].items())
        assert abs(float(total)) < 2e-7


def test_lhi_pressure_only_augmentation_and_degree_two_pressure_one_python(tmp_path):
    model, _, _, _ = _constant_stokes_problem()
    velocity_kernel = r.Gaussian("0.5")
    pressure_kernel = _smooth_pressure()
    cases = [
        (None, 1),
        (2, 1),
    ]
    for velocity_degree, pressure_degree in cases:
        spaces = {
            model.fields[0]: r.DivergenceFreeSpace(velocity_kernel, velocity_degree),
            model.fields[1]: r.PressureSpace(pressure_kernel, pressure_degree),
        }
        solution, cloud, constant = _solve(
            spaces, lambda _: r.PythonBackend(compute_condition=False), tmp_path
        )
        basis = solution.state.system.stencils[0].polynomial_basis
        if velocity_degree is None:
            assert len(basis.columns) == 2
            assert all(all(term[0] == 2 for term in column) for column in basis.columns)
        else:
            query = np.vstack([cloud.points[cloud.boundary_indices], [[0.31, 0.46]]])
            _assert_mp_close(solution, "velocity", query,
                             np.broadcast_to(constant, query.shape))
            _assert_mp_close(solution, "pressure_gradient", query, np.zeros_like(query))


@pytest.mark.skipif(os.environ.get("RBFLAB_CPP_TESTS") != "1", reason="Opt-in native integration")
def test_lhi_independent_velocity_augmentation_cpp_mpfr(tmp_path, monkeypatch):
    # Keep native worker input/output under pytest's writable workspace on Windows.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    model, _, _, _ = _constant_stokes_problem()
    spaces = {
        model.fields[0]: r.DivergenceFreeSpace(_hybrid_velocity(), polynomial_degree=0),
        model.fields[1]: r.PressureSpace(_pressure(), polynomial_degree=None),
    }
    solution, cloud, constant = _solve(
        spaces,
        lambda cache: r.CppBackend(threads=2, compute_condition=False, cache_dir=str(cache)),
        tmp_path,
    )
    query = np.vstack([cloud.points[cloud.boundary_indices], [[0.27, 0.68]]])
    _assert_mp_close(solution, "velocity", query, np.broadcast_to(constant, query.shape))
    _assert_mp_close(solution, "pressure_gradient", query, np.zeros_like(query))
