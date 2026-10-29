"""Mixed quotient fields and canonical solenoidal polynomial coordinates."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

import rbflab as r


def recipe(pressure_degree=1, digits=None):
    x = r.geometry.unit_box_grid(2).points
    I, Dx, Dy, Lap = r.Identity(), r.Derivative(0), r.Derivative(1), r.Laplacian()
    nu = Fraction(1, 10)
    fields = r.MixedSpace({
        'u': r.DivergenceFreeSpace(r.IMQ(2), polynomial_degree=1),
        'p': r.PressureSpace(r.Gaussian('0.7'), polynomial_degree=pressure_degree),
    })
    U = r.FieldOperator({'u': [[I, 0], [0, I]]})
    L = r.FieldOperator({'u': [[-nu*Lap, 0], [0, -nu*Lap]],
                         'p': [[Dx], [Dy]]})
    GP = r.FieldOperator({'p': [[Dx], [Dy]]})
    source = {'u': r.Samples(x, operator=U),
              'L': r.Samples(x, operator=L, target='exclude')}
    local = r.LocalApproximation(source=source, trial=fields.representers(source),
        precision=r.Precision(local_digits=digits,
                              global_dtype='mpmath' if digits else 'float64'),
        backend=r.PythonBackend(compute_condition=False))
    return local, x, L, GP


@pytest.mark.parametrize('degree,pressure_columns', [(None, 0), (0, 0), (1, 2)])
@pytest.mark.parametrize('digits', [None, 40])
def test_mixed_quotient_omits_constant_and_reproduces_gradient(degree, pressure_columns, digits):
    local, x, L, GP = recipe(degree, digits)
    maps = local.operators(targets=[[.5, .5]], operators={'L': L, 'gradient': GP})
    count = maps.L.local(0).multipliers
    assert (count.rows if digits else count.shape[0]) == 5+pressure_columns
    if degree == 1:
        data = {'u': np.c_[1+x[:, 0], 2-x[:, 1]],
                'L': np.tile([2., -3.], (len(x), 1))}
        for target in (maps.L, maps.gradient):
            values = target @ data
            values = np.asarray(values.tolist() if digits else values, float)
            np.testing.assert_allclose(values, [[2, -3]], atol=3e-9)


@pytest.mark.parametrize('source_value', [False, True])
def test_mixed_quotient_rejects_absolute_pressure(source_value):
    local, x, L, GP = recipe()
    P = r.FieldOperator({'p': [[r.Identity()+r.Derivative(0)]]})
    if source_value:
        source = dict(local.source, pressure=r.Samples(x, operator=P))
        space = local.trial.space
        local = r.LocalApproximation(source=source, trial=space.representers(source))
        targets = {'gradient': GP}
    else:
        targets = {'p': P}
    with pytest.raises(ValueError, match='annihilate constants'):
        local.operators(targets=[[.5, .5]], operators=targets)


def test_zero_quotient_target_is_valid():
    local, x, L, GP = recipe()
    zero = local.operators(targets=[[.5, .5]],
                          operators={'zero': r.FieldOperator({'p': [[0]]})}).zero
    assert np.count_nonzero(zero.matrix.toarray()) == 0


def test_2d_stream_polynomial_coordinates_match_historical_basis():
    from rbflab.mixed_local import _columns, polynomial
    from rbflab.nodal import Arithmetic
    from rbflab.stokes_polynomials import StokesPolynomialBasis
    x = r.geometry.unit_box_grid(2).points
    a = Arithmetic(r.IMQ(2), 50)
    fields = [SimpleNamespace(vector=True, degree=6, quotient_pressure=False),
              SimpleNamespace(vector=False, degree=3, quotient_pressure=True)]
    columns = _columns(fields, 2)
    I, Dx, Dy, Lap = r.Identity(), r.Derivative(0), r.Derivative(1), r.Laplacian()
    old_ops = [{0: I}, {1: I}, {0: -Lap, 2: Dx}, {1: -Lap, 2: Dy}, {2: Dx}, {2: Dy}]
    rows = [{(0, c) if c < 2 else (1, 0): op for c, op in item.items()}
            for item in old_ops]
    points = np.array([x[i] for i in (0, 1, 3, 4, 7, 8)])
    origin, radius = [.5, .5], a.number('0.7')
    actual = polynomial(a, points, rows, columns, origin, radius)
    old = StokesPolynomialBasis(a, origin, radius, 6, 3).matrix(points, old_ops)
    assert actual.rows == old.rows and actual.cols == old.cols
    assert max(abs(actual[i, j]-old[i, j])
               for i in range(old.rows) for j in range(old.cols)) == 0
