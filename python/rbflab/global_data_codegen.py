"""Conservative generated evaluators for symbolic global flow data."""
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import math
import sympy as sp
from .symbolic import SymbolicData
from .kernel_compiler import MPConstantPrinter, DoublePrinter


@dataclass(frozen=True)
class NativeDataPlan:
    expressions: tuple
    forcing_ids: tuple
    boundary_groups: tuple
    dimension: int


def native_data_bundle(values, dimension, transient=False):
    """Classify a problem-neutral scalar expression bundle and program IDs."""
    if dimension not in (2, 3):
        return None
    expressions, ids = [], []
    try:
        for value in values:
            if not transient and isinstance(value, SymbolicData) and value.time_symbol is not None:
                return None
            expression = _expression(value, dimension)
            if expression is None:
                return None
            if expression not in expressions:
                expressions.append(expression)
            ids.append(expressions.index(expression))
    except (TypeError, ValueError, AttributeError):
        return None
    return NativeDataPlan(tuple(expressions), (), (), dimension), tuple(ids)


def _expression(value, dimension):
    coordinates = tuple(sp.Symbol(f'z{i}') for i in range(dimension))
    tau = sp.Symbol('tau')
    if isinstance(value, SymbolicData):
        if len(value.coordinates) != dimension or value.requires_normals or value.normals is not None:
            return None
        expression = sp.sympify(value.expression)
        allowed = set(value.coordinates)
        replacements = dict(zip(value.coordinates, coordinates))
        if value.time_symbol is not None:
            allowed.add(value.time_symbol)
            replacements[value.time_symbol] = tau
        if expression.free_symbols - allowed:
            return None
        expression = expression.xreplace(replacements)
    elif isinstance(value, (int, float, Fraction, Decimal, sp.Expr)):
        if isinstance(value, float):
            if not math.isfinite(value):
                return None
            expression = sp.Rational(value)
        elif isinstance(value, (Fraction, Decimal)):
            expression = sp.Rational(str(value))
        else:
            expression = sp.sympify(value)
        if expression.free_symbols or expression.is_real is not True or expression.is_finite is not True:
            return None
    else:
        return None
    # Preserve the represented value of SymPy Float, rather than rendering it
    # through a binary64 C literal or a truncated decimal printer.
    expression = expression.xreplace({number: sp.Rational(number) for number in expression.atoms(sp.Float)})
    for node in sp.preorder_traversal(expression):
        if node in coordinates or node == tau or node in (sp.pi, sp.E):
            continue
        if isinstance(node, sp.Rational):
            continue
        if node.func in (sp.Add, sp.Mul, sp.exp, sp.sin, sp.cos, sp.log, sp.Abs):
            continue
        if node.func == sp.Pow and isinstance(node.exp, sp.Rational):
            continue
        return None
    return expression


def native_data_plan(problem, dimension):
    """Return a canonical bundle, or None for existing Python evaluation."""
    if dimension not in (2, 3):
        return None
    expressions = []
    def program(value):
        if getattr(problem, 'transient', True) is False and isinstance(value, SymbolicData) and value.time_symbol is not None:
            # Stationary Python data may already carry an explicitly bound
            # time; a runtime native data(0) request would change its meaning.
            raise ValueError('unsupported native data')
        expression = _expression(value, dimension)
        if expression is None:
            raise ValueError('unsupported native data')
        if expression not in expressions:
            expressions.append(expression)
        return expressions.index(expression)
    try:
        if len(problem.forcing) != dimension:
            return None
        forcing = tuple(program(value) for value in problem.forcing)
        boundary = []
        for on, components in problem.boundary:
            if len(components) != dimension:
                return None
            if not isinstance(on, str):
                on = tuple(on)
                if any(not isinstance(label, str) for label in on):
                    return None
            boundary.append((on, tuple(program(value) for value in components)))
    except (TypeError, ValueError, AttributeError):
        return None
    return NativeDataPlan(tuple(expressions), forcing, tuple(boundary), dimension)


class _DataOperations:
    def _print_Abs(self, expression):
        return 'abs(' + self._print(expression.args[0]) + ')'

    def _print_Pow(self, expression):
        base = self._print(expression.base)
        exponent = expression.exp
        if exponent.is_Integer and -(2**31) <= int(exponent) < 2**31:
            power = str(int(exponent))
        else:
            power = self._print(exponent)
        return 'pow(' + base + ',' + power + ')'


class _DoubleDataPrinter(_DataOperations, DoublePrinter):
    def _print_Pi(self, expression):
        return 'Real(std::acos(-1.0))'

    def _print_Exp1(self, expression):
        return 'exp(Real(1))'


class _MPDataPrinter(_DataOperations, MPConstantPrinter):
    def _print_Pi(self, expression):
        return self._constant(expression)

    def _print_Exp1(self, expression):
        return self._constant(expression)

    def declarations(self):
        lines = []
        for value, name in self.constants.items():
            lines += [f'inline const Real& {name}(){{', 'static thread_local Real value;',
                'static thread_local mpfr_prec_t precision=0;', 'if(precision!=Real::precision()){']
            if value == sp.pi:
                lines.append('mpfr_set_prec(value.v,Real::precision()); mpfr_const_pi(value.v,MPFR_RNDN);')
            elif value == sp.E:
                lines.append('value=exp(Real(1));')
            else:
                from .kernel_compiler import Printer
                lines.append('value=' + Printer().doprint(value) + ';')
            lines += ['precision=Real::precision(); } return value; }']
        return lines


def generate_data(plan, arithmetic):
    """Emit runtime-time evaluators; expressions determine cache identity."""
    if not isinstance(plan, NativeDataPlan):
        raise TypeError('Expected NativeDataPlan')
    if arithmetic not in ('mpfr', 'float64'):
        raise ValueError('Native symbolic data supports mpfr or float64')
    printer = _MPDataPrinter() if arithmetic == 'mpfr' else _DoubleDataPrinter()
    bodies = []
    for index, expression in enumerate(plan.expressions):
        substitutions, reduced = sp.cse(expression, symbols=sp.numbered_symbols('data_tmp_'))
        bodies.append(f'case {index}: {{')
        bodies.extend(f'const Real {name}={printer.doprint(value)};' for name, value in substitutions)
        bodies += [f'return {printer.doprint(reduced[0])};', '}']
    lines = ['#define RBFLAB_GLOBAL_DATA', f'constexpr int DATA_PROGRAMS={len(plan.expressions)};', 'namespace native_data {']
    if arithmetic == 'mpfr':
        lines.extend(printer.declarations())
    lines += ['inline Real value(int index,const std::vector<Real>& point,const Real& tau){']
    lines.extend(f'const Real& z{i}=point.at({i});' for i in range(plan.dimension))
    lines += ['switch(index){', *bodies, 'default: throw std::runtime_error("Unknown native data program");', '}}', '}',
        'inline Real data_value(int index,const std::vector<Real>& point,const Real& time){ return native_data::value(index,point,time); }']
    return '\n'.join(lines) + '\n'
