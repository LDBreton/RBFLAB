"""Shared 2D curve evaluation and differentiation, without topology assumptions."""
import numpy as np
import sympy as sp


def vector(value, name):
    """Validate a finite real planar vector."""
    try:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raise ValueError
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must return two finite coordinates (real-valued)") from exc
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must return two finite coordinates (real-valued)")
    return result


def unit_vectors(values, name="normal"):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite 2D vectors")
    lengths = np.linalg.norm(values, axis=1)
    if np.any(lengths == 0):
        raise ValueError(f"{name} must be nonzero; split nonsmooth curves or supply normal explicitly")
    return values/lengths[:, None]


class Curve2D:
    """Internal compiled evaluator shared by Border and ParametricBoundary."""
    def __init__(self, curve, interval, *, tangent=None, normal=None,
                 parameter=None, difference_step=None):
        limits = np.asarray(interval, dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] == limits[1]:
            raise ValueError("curve interval must have two distinct finite endpoints")
        self.lower, self.upper = sorted(limits)
        symbolic = not callable(curve)
        if parameter is not None and not isinstance(parameter, sp.Symbol):
            raise TypeError("parameter must be a SymPy Symbol")
        specs = [spec for spec in (curve, tangent, normal) if spec is not None and not callable(spec)]
        expressions = []
        for spec in specs:
            try:
                pair = tuple(sp.sympify(v) for v in spec)
            except (TypeError, ValueError, sp.SympifyError) as exc:
                raise TypeError("symbolic curve/tangent/normal must have two coordinate expressions") from exc
            if len(pair) != 2:
                raise ValueError("symbolic curve/tangent/normal must have two coordinates")
            expressions.extend(pair)
        symbols = set().union(*(expr.free_symbols for expr in expressions))
        if parameter is None:
            if len(symbols) > 1:
                raise ValueError("bind extra symbols with subs() and supply the curve parameter")
            parameter = next(iter(symbols), sp.Symbol("_curve_t", real=True))
        if symbols - {parameter}:
            raise ValueError("unbound symbolic parameters; bind them with subs() before constructing a curve")
        self.parameter = parameter

        def compile_vector(spec, name):
            function = spec if callable(spec) else sp.lambdify(parameter, tuple(spec), modules="numpy", cse=True)
            return lambda t: vector(function(float(t)), name)

        self.point = compile_vector(curve, "curve")
        if difference_step is not None and (not np.isfinite(difference_step) or difference_step <= 0):
            raise ValueError("difference_step must be finite and positive")
        span = self.upper-self.lower
        step = span*np.cbrt(np.finfo(float).eps) if difference_step is None else float(difference_step)
        if not 0 < step <= span/4:
            raise ValueError("difference_step must not exceed one quarter of the parameter interval")
        self.step = step
        if tangent is not None:
            self.tangent = compile_vector(tangent, "tangent")
            self.derivative_source = "explicit"
        elif symbolic:
            derivative = tuple(sp.diff(expr, parameter) for expr in curve)
            self.tangent = compile_vector(derivative, "tangent")
            self.derivative_source = "symbolic"
        else:
            self.tangent = self._difference
            self.derivative_source = "finite_difference"
        self.normal = None if normal is None else compile_vector(normal, "normal")

    def _difference(self, t):
        a, b, h = self.lower, self.upper, self.step
        if not a <= t <= b:
            raise ValueError("curve parameter lies outside its interval")
        if t+h == t or t-h == t:
            raise ValueError("parameter interval is too small at this offset; rescale the parameter")
        if t-h >= a and t+h <= b:
            return (self.point(t+h)-self.point(t-h))/(2*h)
        if t-h < a:
            return (-3*self.point(t)+4*self.point(t+h)-self.point(t+2*h))/(2*h)
        return (3*self.point(t)-4*self.point(t-h)+self.point(t-2*h))/(2*h)

    def normals(self, parameters):
        if self.normal is not None:
            return unit_vectors([self.normal(t) for t in parameters])
        tangent = np.array([self.tangent(t) for t in parameters])
        return unit_vectors(np.column_stack((tangent[:, 1], -tangent[:, 0])), "tangent")
