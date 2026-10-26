from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace
import sympy as sp
from rbflab.symbolic import SymbolicData
from rbflab.global_data_codegen import native_data_plan, generate_data, native_data_bundle

x, y, t = sp.symbols('x y t')


def problem(value, other=0):
    return SimpleNamespace(forcing=(value, other), boundary=(('boundary', (other, value)),))


def data(expression, **kwargs):
    return SymbolicData(expression, (x, y), time_symbol=t, **kwargs)


def test_canonical_dedup_and_boundary_mapping():
    plan = native_data_plan(problem(data(sp.sin(x+t))), 2)
    assert plan.expressions == (sp.sin(sp.Symbol('z0')+sp.Symbol('tau')), sp.Integer(0))
    assert plan.forcing_ids == (0, 1)
    assert plan.boundary_groups == (('boundary', (1, 0)),)


def test_time_is_dynamic_and_bound_time_does_not_change_program():
    first = native_data_plan(problem(data(x*t, time=1)), 2)
    second = native_data_plan(problem(data(x*t, time=2)), 2)
    assert first == second
    assert 'const Real& tau' in generate_data(first, 'mpfr')
    assert native_data_plan(problem(data(x*(t+1))), 2).expressions != first.expressions


def test_float_fraction_and_decimal_exact():
    for value, expected in [(0.1, sp.Rational(0.1)), (Fraction(1, 10), sp.Rational(1, 10)), (Decimal('0.1'), sp.Rational(1, 10)), (sp.Float('0.1', 40), sp.Rational(sp.Float('0.1', 40)))]:
        assert native_data_plan(problem(value), 2).expressions[0] == expected


def test_unsupported_data_falls_back():
    n = sp.Symbol('n')
    unsupported = [lambda points: points[:, 0], data(sp.Piecewise((x, x>0), (y, True))), data(x+n),
        SymbolicData(x, (x, y), normal_symbols=(n,)), data(sp.tan(x)), float('inf'), data(x, normals=[[1, 0]])]
    assert all(native_data_plan(problem(value), 2) is None for value in unsupported)


def test_mp_constants_and_precision_cache():
    plan = native_data_plan(problem(data(sp.pi*x+sp.E*t+sp.sin(x)+sp.sqrt(y))), 2)
    source = generate_data(plan, 'mpfr')
    assert 'mpfr_const_pi(value.v,MPFR_RNDN)' in source
    assert 'Real::precision()' in source
    assert 'value=exp(Real(1))' in source
    assert 'M_PI' not in source and '3.14159' not in source
    assert '#define RBFLAB_GLOBAL_DATA' in source
    assert 'constexpr int DATA_PROGRAMS=2' in source


def test_canonical_coordinate_names_and_selection_order():
    a, b, q = sp.symbols('a b q')
    first = native_data_plan(problem(data(x+t)), 2)
    second = native_data_plan(problem(SymbolicData(a+q, (a, b), time_symbol=q)), 2)
    assert first == second
    assert native_data_plan(problem(data(x+t+1)), 2) != first


def test_abs_and_rational_power_use_real_overloads():
    plan = native_data_plan(problem(data(sp.Abs(x)+y**sp.Rational(1, 3)+x**3)), 2)
    for arithmetic in ('mpfr', 'float64'):
        source = generate_data(plan, arithmetic)
        assert 'abs(z0)' in source
        assert 'pow(z0,3)' in source
        assert 'pow(z1,' in source
        assert 'cbrt(' not in source and 'fabs(' not in source


def test_double_constants_avoid_platform_pi_macro():
    source = generate_data(native_data_plan(problem(data(sp.pi*x+sp.E*t)), 2), 'float64')
    assert 'Real(std::acos(-1.0))' in source
    assert 'exp(Real(1))' in source
    assert 'M_PI' not in source


def test_cse_is_scoped_to_selected_program():
    expression = sp.sin(x+t)+sp.cos(x+t)
    source = generate_data(native_data_plan(problem(data(expression)), 2), 'mpfr')
    assert 'case 0: {' in source
    assert 'const Real data_tmp_0=' in source
    assert 'sin(data_tmp_0)' in source and 'cos(data_tmp_0)' in source


def test_stationary_time_bound_data_preserves_python_fallback():
    stationary = problem(data(x+t, time=.5))
    stationary.transient = False
    assert native_data_plan(stationary, 2) is None
    stationary.transient = True
    assert native_data_plan(stationary, 2) is not None


def test_neutral_scalar_bundle_deduplicates_and_preserves_ids():
    plan, ids = native_data_bundle((SymbolicData(x, (x, y)), 0, SymbolicData(x, (x, y))), 2)
    assert ids == (0, 1, 0)
    assert plan.forcing_ids == () and plan.boundary_groups == ()
    assert native_data_bundle((data(x+t, time=.5),), 2) is None
    assert native_data_bundle((data(x+t),), 2, transient=True) is not None
    assert native_data_bundle((lambda points: points[:, 0],), 2) is None
