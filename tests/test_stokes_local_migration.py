"""Legacy compact Stokes weights and matrices through the common local API."""
from fractions import Fraction
import os

import numpy as np
import pytest
import sympy as sp

import rbflab as r
from rbflab.lhi_backends import _prepare, _python_job, _finish
from rbflab.lhi_stokes import LHIUnsteadyStokes
from rbflab.unsteady_stokes import UnsteadyStokesProblem
from rbflab.sparse_precision import MPSparseMatrix


def _recipe(digits=None, shape='fixed', scaling='physical', backend=None, execution=None):
    vk, pk = r.IMQ(2), r.Gaussian('1.5')
    degree, pdegree = (None, None) if shape == 'hardy' else (2, 1)
    method = LHIUnsteadyStokes(vk, stencil_size=9, pressure_kernel=pk,
        polynomial_degree=degree, pressure_polynomial_degree=pdegree,
        precision=r.Precision(local_digits=digits,
                              global_dtype='mpmath' if digits else 'float64'),
        stencil_policy=r.StencilPolicy(scaling=scaling, shape_rule=shape),
        local_backend=backend or r.PythonBackend(compute_condition=False))
    object.__setattr__(method.local_backend, '_shape_rule', shape)
    if execution is not None:
        method.execution = execution
    problem = UnsteadyStokesProblem((2, -3), ('1.25', '-.75'),
                                    ('1.25', '-.75'), viscosity=Fraction(1, 10))
    return method, problem, r.geometry.unit_box_grid(3)


def _legacy(method, problem, cloud):
    # This retained reference assembles the original Hermite Gram directly;
    # production migration must call LocalApproximation instead.
    a, tasks, stencils = _prepare(method, problem, cloud, method.stencil_policy.shape_rule)
    for task in tasks:
        task['condition'] = False
    results = [_python_job(task) for task in tasks]
    return _finish(method, problem, cloud, a, tasks, stencils, results, {'reference': True})


def _dense(matrix):
    if isinstance(matrix, MPSparseMatrix):
        return [[row.get(j, 0) for j in range(matrix.shape[1])] for row in matrix.rows]
    return matrix.toarray().tolist()


def _relative(a, b, ctx=None):
    a, b = np.asarray(a, object).reshape(-1), np.asarray(b, object).reshape(-1)
    number = ctx.mpf if ctx else float
    difference = max((abs(number(x)-number(y)) for x, y in zip(a, b)), default=0)
    scale = max((abs(number(x)) for x in a), default=0)
    return difference/(scale or 1)


def _assert_same(reference, migrated, tolerance):
    a = reference.local_arithmetic
    number = a.ctx.mpf if a.ctx else float
    assert len(reference.stencils) == len(migrated.stencils)
    for left, right in zip(reference.stencils, migrated.stencils):
        np.testing.assert_array_equal(left.points, right.points)
        assert left.slots == right.slots
        assert left.functionals == right.functionals
        assert left.scale == right.scale
    assert _relative(reference.reconstruction_weights, migrated.reconstruction_weights, a.ctx) < number(tolerance)
    for name in ('SY', 'SL', 'mass'):
        assert _relative(_dense(getattr(reference, name)), _dense(getattr(migrated, name)), a.ctx) < number(tolerance)
    boundary_columns = 2*len(reference.cloud.boundary_indices)
    dense = lambda system: [[row.get(j, 0) for j in range(boundary_columns)] for row in system.sb_rows]
    assert _relative(dense(reference), dense(migrated), a.ctx) < number(tolerance)


def _assert_nodal_time_parity(reference, migrated, tolerance):
    """Check both time schemes and step sizes using actual cloud nodes only."""
    ctx = reference.local_arithmetic.ctx
    number = ctx.mpf if ctx else float
    for scheme in ('backward_euler', 'bdf2'):
        for dt in ('.001', '.0005'):
            left = reference.solve(dt, 2, scheme=scheme).states[-1]
            right = migrated.solve(dt, 2, scheme=scheme).states[-1]
            for name, points in [('evaluate_velocity', reference.cloud.points),
                                 ('pressure_gradient', reference.cloud.interior)]:
                a = getattr(left, name)(points, extended=bool(ctx))
                b = getattr(right, name)(points, extended=bool(ctx))
                assert _relative(a.tolist(), b.tolist(), ctx) < number(tolerance)


@pytest.mark.parametrize('digits,shape,scaling', [
    (None, 'fixed', 'physical'), (40, 'fixed', 'physical'),
    (40, 'fixed', 'local'), (None, 'hardy', 'physical'),
    (40, 'radius_scaled', 'physical'), (40, 'separation_scaled', 'physical'),
])
def test_python_stokes_uses_common_weights_and_preserves_compact_matrices(monkeypatch, digits, shape, scaling):
    method, problem, cloud = _recipe(digits, shape, scaling)
    reference = _legacy(method, problem, cloud)
    calls = []
    original = r.LocalApproximation.operators
    def spy(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)
    monkeypatch.setattr(r.LocalApproximation, 'operators', spy)
    # A migrated assembly must not merely retain its previous weight solver.
    monkeypatch.setattr('rbflab.lhi_backends._python_job',
                        lambda *args: (_ for _ in ()).throw(AssertionError('legacy weight path used')))
    migrated = method.assemble(problem, cloud)
    assert calls
    _assert_same(reference, migrated, '2e-7' if digits is None else '1e-25')


@pytest.mark.parametrize('transient', [False, True])
@pytest.mark.parametrize('scaling', ['physical', 'local'])
def test_public_stokes_wrapper_nodal_velocity_and_pressure_gradient(transient, scaling, monkeypatch):
    model = r.SymbolicStokes(2, transient=transient)
    u, p = model.fields
    constant = sp.ImmutableMatrix([sp.Rational(5, 4), -sp.Rational(3, 4)])
    lhs = -sp.Rational(1, 10)*model.laplacian(u)+model.gradient(p)
    boundary = [model.bc('boundary', sp.Eq(u, constant))]
    if transient:
        problem = model.evolution([sp.Eq(sp.diff(u, model.time)+lhs, sp.ImmutableMatrix([2, -3]))],
                                  initial={u: constant}, boundary=boundary)
    else:
        problem = model.stationary([sp.Eq(lhs, sp.ImmutableMatrix([2, -3]))], boundary=boundary)
    cloud = r.geometry.unit_box_grid(3)
    calls, original = [], r.LocalApproximation.operators
    def spy(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)
    monkeypatch.setattr(r.LocalApproximation, 'operators', spy)
    system = r.LHI(spaces={u: r.DivergenceFreeSpace(r.IMQ(2), 2),
                           p: r.PressureSpace(r.Gaussian('1.5'), 1)}, stencil_size=9,
                   precision=r.Precision(local_digits=40, global_dtype='mpmath'),
                   stencil_policy=r.StencilPolicy(scaling=scaling),
                   local_backend=r.PythonBackend(compute_condition=False)).assemble(problem, cloud)
    assert calls
    result = system.solve(dt='.001', steps=2).states[-1] if transient else system.solve()
    velocity = result.velocity(cloud.points, extended=True)
    gradient = result.pressure_gradient(cloud.interior, extended=True)
    assert max(abs(velocity[i, c]-constant[c]) for i in range(velocity.rows) for c in range(2)) < 1e-25
    assert max(abs(gradient[i, c]-(2, -3)[c]) for i in range(gradient.rows) for c in range(2)) < 1e-25


@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS') != '1', reason='Opt-in native integration')
@pytest.mark.parametrize('digits', [None, 80])
@pytest.mark.parametrize('scaling', ['physical', 'local'])
def test_cpp_stokes_weights_and_matrix_parity(digits, scaling, tmp_path):
    method, problem, cloud = _recipe(digits, scaling=scaling)
    reference = _legacy(method, problem, cloud)
    method.local_backend = r.CppBackend(compute_condition=False, cache_dir=tmp_path)
    migrated = method.assemble(problem, cloud)
    _assert_same(reference, migrated, '2e-7' if digits is None else '1e-60')
    if scaling == 'local':
        _assert_nodal_time_parity(reference, migrated, '2e-7' if digits is None else '1e-60')


def test_python_process_workers_preserve_stokes_maps():
    method, problem, cloud = _recipe(40, backend=r.PythonBackend(workers=2, compute_condition=False))
    reference = _legacy(method, problem, cloud)
    migrated = method.assemble(problem, cloud)
    _assert_same(reference, migrated, '1e-25')


@pytest.mark.skipif(os.environ.get('RBFLAB_MPI_TESTS') != '1', reason='Opt-in real MPI integration')
@pytest.mark.parametrize('digits,scaling', [(40, 'physical'), (None, 'local'), (80, 'local')])
def test_mpi_stokes_weights_and_matrix_parity(digits, scaling, tmp_path):
    method, problem, cloud = _recipe(digits, scaling=scaling)
    reference = _legacy(method, problem, cloud)
    method.local_backend = r.CppBackend(compute_condition=False, cache_dir=tmp_path)
    method.execution = r.LocalExecution(scheduler='mpi', ranks=3)
    migrated = method.assemble(problem, cloud)
    tolerance = '2e-7' if digits is None else '1e-60' if digits == 80 else '1e-25'
    _assert_same(reference, migrated, tolerance)
    assert migrated.backend_diagnostics['local_execution'][0]['scheduler'] == 'mpi'
    assert migrated.backend_diagnostics['local_execution'][0]['ranks'] == 3
    if scaling == 'local':
        _assert_nodal_time_parity(reference, migrated, tolerance)


@pytest.mark.parametrize('route', ['svd', 'custom'])
def test_internal_specialized_cpp_still_rejects_local_scaling(route):
    backend = r.CppBackend(executable='unused-legacy-executable' if route == 'custom' else None)
    if route == 'svd':
        object.__setattr__(backend, '_solver', r.LocalSolver('svd'))
    method, problem, cloud = _recipe(scaling='local', backend=backend)
    with pytest.raises(NotImplementedError, match='require physical kernel coordinates'):
        method.assemble(problem, cloud)
