"""37-site 6/1 shared geometry against the original explicit memberships."""
import os
from fractions import Fraction
import numpy as np
import pytest
import sympy as sp
from scipy.spatial import cKDTree
import rbflab as r


def build(backend, shared, symbolic=False):
    cloud = r.geometry.unit_box_grid(8)
    points, inside = cloud.points, cloud.interior
    wall = points[cloud.boundary_indices]
    targets = inside[[0, 24]]  # near-wall and centered stencils
    q = Fraction(8, 44)
    s = sp.Symbol('s', nonnegative=True)
    keys = r.SymbolicStokes(2, transient=True).fields if symbolic else ('u', 'p')
    space = r.MixedSpace({
        keys[0]: r.DivergenceFreeSpace(r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7), 6),
        keys[1]: r.PressureSpace(r.Kernel(q**4*sp.exp(-q*q*s), s)(), 1),
    })
    u, p = space.fields
    L, P = -r.laplacian(u)+r.gradient(p), r.gradient(p)
    if shared:
        stencil = r.SharedStencil(points, size=37)
        source = {'velocity': r.Samples(inside, operator=u, within=stencil),
                  'boundary': r.Samples(wall, operator=u, within=stencil),
                  'momentum': r.Samples(inside, operator=L, within=stencil, target='exclude')}
    else:
        distances, neighbors = cKDTree(points).query(targets, k=37)
        ids = [j[np.lexsort((j, d))] for j, d in zip(neighbors, distances)]
        imap = {tuple(point): i for i, point in enumerate(inside)}
        momentum = [np.array([imap[tuple(points[j])] for j in row
                              if tuple(points[j]) in imap and not np.array_equal(points[j], target)])
                    for row, target in zip(ids, targets)]
        source = {'velocity': r.Samples(points, operator=u, indices=ids),
                  'momentum': r.Samples(inside, operator=L, indices=momentum)}
    maps = r.LocalApproximation(source=source, trial=space.representers(source),
        precision=r.Precision(local_digits=80),
        backend=r.PythonBackend() if backend == 'python' else r.CppBackend(timeout=3600),
        execution=r.LocalExecution(scheduler='mpi', ranks=2) if backend == 'mpi' else None,
    ).operators(targets=targets, operators={'L': L, 'P': P})
    matrices = []
    for label in ('L', 'P'):
        operator = maps[label]
        if shared:
            matrices.extend(operator[group].to_scipy().toarray()
                            for group in ('velocity', 'boundary', 'momentum'))
        else:
            full = operator['velocity'].to_scipy()
            for ids in (cloud.interior_indices, cloud.boundary_indices):
                columns = (2*ids[:, None]+np.arange(2)).ravel()
                matrices.append(full[:, columns].toarray())
            matrices.append(operator['momentum'].to_scipy().toarray())
    return matrices, maps


def test_shared_stencil_37_point_python_reference():
    shared, _ = build('python', True)
    reference, _ = build('python', False)
    for actual, expected in zip(shared, reference):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=2e-12)


@pytest.mark.skipif(os.environ.get('RBFLAB_CPP_TESTS') != '1' or
                    os.environ.get('RBFLAB_MPI_TESTS') != '1', reason='Opt-in C++ and real MPI parity')
def test_shared_stencil_python_cpp_mpi_parity():
    reference, _ = build('python', False)
    for backend in ('cpp', 'mpi'):
        actual, maps = build(backend, True)
        for a, b in zip(actual, reference):
            np.testing.assert_allclose(a, b, atol=1e-12, rtol=2e-12)
        # Nonzero constant velocity and p=x-y reproduce L and grad(p).
        source = maps.L.input_layout
        data = {name: np.tile([1., -1.], (len(entry['points']), 1))
                for name, entry in source.items()}
        for name in ('L', 'P'):
            result = np.asarray((maps[name] @ data).tolist(), dtype=float)
            np.testing.assert_allclose(result, np.tile([1., -1.], (2, 1)), atol=1e-10)
        if backend == 'mpi':
            assert maps.diagnostics['execution']['scheduler'] == 'mpi'
