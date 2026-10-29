"""Quotient pressure observations must not select an implicit absolute gauge."""
import numpy as np
import pytest

import rbflab as r


@pytest.mark.parametrize('degree,columns', [(None, 0), (0, 0), (1, 2)])
@pytest.mark.parametrize('digits', [None, 40])
def test_pressure_quotient_polynomials_and_gradient_reproduction(degree, columns, digits):
    x = r.geometry.unit_box_grid(2).points
    source = {'dx': r.Samples(x, r.Derivative(0)),
              'dy': r.Samples(x, r.Derivative(1))}
    space = r.PressureSpace(r.Gaussian('0.7'), polynomial_degree=degree)
    maps = r.LocalApproximation(
        source=source, trial=space.representers(source),
        backend=r.PythonBackend(compute_condition=False),
        precision=r.Precision(local_digits=digits,
                              global_dtype='mpmath' if digits else 'float64'),
    ).operators(targets=[[.31, .42]], operators={'dx': r.Derivative(0),
                                               'dy': r.Derivative(1)})
    multipliers = maps.dx.local(0).multipliers
    assert (multipliers.rows if digits else multipliers.shape[0]) == columns
    if degree == 1:
        data = {'dx': np.full(len(x), 2), 'dy': np.full(len(x), -3)}
        assert abs(float((maps.dx @ data)[0]) - 2) < 2e-9
        assert abs(float((maps.dy @ data)[0]) + 3) < 2e-9


def test_pressure_absolute_source_target_and_translate_are_rejected():
    x = r.geometry.unit_box_grid(2).points
    space = r.PressureSpace(r.IMQ(2), polynomial_degree=1)
    with pytest.raises(ValueError, match='modulo constants'):
        r.LocalApproximation(source=x, space=space).operators(
            targets=x[:1], operators={'dx': r.Derivative(0)})
    source = {'dx': r.Samples(x, r.Derivative(0)),
              'dy': r.Samples(x, r.Derivative(1))}
    local = r.LocalApproximation(source=source, trial=space.representers(source))
    with pytest.raises(ValueError, match='modulo constants'):
        local.operators(targets=x[:1], operators={'p': r.Identity()})
    # An identity contribution remains gauge-dependent even with derivatives.
    with pytest.raises(ValueError, match='modulo constants'):
        local.operators(targets=x[:1], operators={'p': r.Identity()+r.Derivative(0)})
    with pytest.raises(ValueError, match='modulo constants'):
        r.LocalApproximation(source={'dx': r.Samples(x, r.Derivative(0))},
                             trial=space.translates(x)).operators(
            targets=x[:1], operators={'dx': r.Derivative(0)})


def test_absolute_pressure_remains_explicit_scalar_space():
    x = r.geometry.unit_box_grid(2).points
    space = r.ScalarSpace(r.IMQ(2), polynomial_degree=1)
    value = r.LocalApproximation(source=x, space=space, stencil_size=9).operators(
        targets=[[.31, .42]], operators={'p': r.Identity()}).p
    assert abs(float((value @ np.full(len(x), 7))[0]) - 7) < 1e-12


def test_quotient_descriptor_cannot_be_silently_stripped_by_conveniences():
    cloud = r.geometry.unit_box_grid(2)
    space = r.PressureSpace(r.IMQ(2), polynomial_degree=1)
    with pytest.raises(ValueError, match='modulo constants'):
        r.interpolate(space=space, centers=cloud.points, data=np.zeros(len(cloud.points)))
    problem = r.LinearPDE(-r.Laplacian(), 0, [r.Dirichlet(0)])
    with pytest.raises(NotImplementedError, match='absolute field'):
        r.RBFFD(spaces={'p': space}, stencil_size=9).assemble(problem, cloud)
