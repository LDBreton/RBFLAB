import numpy as np
import pytest
import sympy as sp

import rbflab as r


@pytest.mark.parametrize('method_type', [r.GlobalCollocation, r.LHI])
def test_mixed_space_matches_mapping_for_symbolic_scalar_blocks(method_type):
    cloud = r.geometry.unit_box_grid(3)
    model = r.SymbolicSystem(fields=('a', 'b'))
    a, b = model.fields
    x, y = model.coordinates
    exact_a, exact_b = 1 + x*x, 1 + y*y
    problem = model.stationary(
        [sp.Eq(-model.laplacian(a) + b, -2 + exact_b),
         sp.Eq(-model.laplacian(b) + a, -2 + exact_a)],
        boundary=[model.bc('boundary', [sp.Eq(a, exact_a), sp.Eq(b, exact_b)])])
    spaces = {a: r.ScalarSpace(r.PHS(5), 2), b: r.ScalarSpace(r.PHS(5), 2)}
    method_options = {'stencil_size': 12} if method_type is r.LHI else {}
    mapping = method_type(spaces=spaces, **method_options).assemble(problem, cloud)
    mixed = method_type(spaces=r.MixedSpace(spaces), **method_options).assemble(problem, cloud)
    left = mixed.matrix.toarray() if hasattr(mixed.matrix, 'toarray') else np.asarray(mixed.matrix)
    right = mapping.matrix.toarray() if hasattr(mapping.matrix, 'toarray') else np.asarray(mapping.matrix)
    np.testing.assert_allclose(left, right, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(mixed.rhs, mapping.rhs, rtol=1e-13, atol=1e-13)
    assert mixed.solve().diagnostics['relative_residual'] < 1e-9


@pytest.mark.parametrize('method_type', [r.GlobalCollocation, r.LHI])
def test_mixed_space_matches_mapping_for_symbolic_stokes(method_type):
    cloud = r.geometry.unit_box_grid(3)
    model = r.SymbolicStokes(2)
    x, y = model.coordinates
    velocity, pressure = model.velocity, model.pressure
    problem = model.stationary(
        -model.laplacian(velocity) + model.gradient(pressure),
        boundary=[model.bc('boundary', sp.Eq(velocity, sp.ImmutableMatrix([0, 0])))])
    spaces = {velocity: r.DivergenceFreeSpace(r.Gaussian(2), None),
              pressure: r.ScalarSpace(r.Gaussian(2), None)}
    method_options = {'stencil_size': 12} if method_type is r.LHI else {}
    mapping = method_type(spaces=spaces, **method_options).assemble(problem, cloud)
    mixed = method_type(spaces=r.MixedSpace(spaces), **method_options).assemble(problem, cloud)
    left = mixed.matrix.toarray() if hasattr(mixed.matrix, 'toarray') else np.asarray(mixed.matrix)
    right = mapping.matrix.toarray() if hasattr(mapping.matrix, 'toarray') else np.asarray(mapping.matrix)
    np.testing.assert_allclose(left, right,
                               rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(mixed.rhs, dtype=float), np.asarray(mapping.rhs, dtype=float),
                               rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize('method_type', [r.GlobalCollocation, r.LHI, r.RBFFD])
def test_mixed_space_matches_mapping_for_scalar_symbolic_pde(method_type):
    cloud = r.geometry.unit_box_grid(4)
    model = r.SymbolicScalar(2)
    u = model.field
    x, y = model.coordinates
    exact = 1 + x*x + y*y
    problem = model.stationary(
        sp.Eq(-model.laplacian(u), -4),
        boundary=[model.bc('boundary', sp.Eq(u, exact))])
    space = r.ScalarSpace(r.PHS(5), 2)
    mapping = {u: space}
    method_options = {'stencil_size': 12} if method_type is not r.GlobalCollocation else {}
    old_system = method_type(spaces=mapping, **method_options).assemble(problem, cloud)
    mixed_system = method_type(spaces=r.MixedSpace(mapping), **method_options).assemble(problem, cloud)
    old_matrix = old_system.matrix.toarray() if hasattr(old_system.matrix, 'toarray') else np.asarray(old_system.matrix)
    mixed_matrix = mixed_system.matrix.toarray() if hasattr(mixed_system.matrix, 'toarray') else np.asarray(mixed_system.matrix)
    np.testing.assert_allclose(mixed_matrix, old_matrix, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(mixed_system.rhs, dtype=float),
                               np.asarray(old_system.rhs, dtype=float), rtol=1e-13, atol=1e-13)


def test_mixed_space_selects_symbolic_field_for_rbffd_operators():
    cloud = r.geometry.unit_box_grid(4)
    model = r.SymbolicSystem(fields=('a', 'b'))
    a, b = model.fields
    spaces = {a: r.ScalarSpace(r.PHS(5), 2), b: r.ScalarSpace(r.Gaussian(2), 2)}
    mixed = r.RBFFD(spaces=r.MixedSpace(spaces), stencil_size=12)
    legacy = r.RBFFD(spaces=spaces, stencil_size=12)
    operator = {'lap': r.Laplacian(2)}
    actual = mixed.operators(source=cloud, operators=operator, space=a)['lap'].matrix
    expected = legacy.operators(source=cloud, operators=operator, space=a)['lap'].matrix
    np.testing.assert_allclose(actual.toarray(), expected.toarray(), rtol=1e-13, atol=1e-13)
