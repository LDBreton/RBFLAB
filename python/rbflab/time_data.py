"""Dimension-independent time-dependent data shared by scalar and Stokes APIs."""
from dataclasses import dataclass
from .precision import PrecisionData, mp_number


@dataclass(frozen=True)
class TimeData:
    """Callbacks: float(points,time), extended(ctx,*coordinates,time)."""
    float_callback: object
    extended_callback: object

    def at(self, time):
        return PrecisionData(lambda points: self.float_callback(points,float(time)),
            lambda ctx,*coordinates: self.extended_callback(ctx,*coordinates,mp_number(ctx,time)))


def at_time(data,time):
    if hasattr(data, "at_time"):
        return data.at_time(time)
    if isinstance(data,TimeData):
        return data.at(time)
    if callable(data):
        return lambda points:data(points,float(time))
    return data


@dataclass(frozen=True)
class InitialData:
    """Initial values plus analytic derivative data keyed by Cartesian multi-index.

    Each entry is a scalar, NumPy callback, or PrecisionData for MP evaluation.
    Supply every derivative requested by the boundary operator; missing entries
    are errors, not assumed zeros.
    """
    values: object
    derivatives: dict

    def __call__(self,points):
        from .problems import values
        return values(self.values,points)

    def mp_values(self,ctx,points):
        from .precision import mp_values
        if callable(self.values) and not hasattr(self.values,"mp_values"):
            raise TypeError("Extended initial data require PrecisionData callbacks")
        return mp_values(self.values,points,ctx)

    def operator_values(self,operator,points,ctx=None):
        import numpy as np
        from .problems import values
        from .precision import mp_values
        out=ctx.matrix(len(points),1) if ctx else np.zeros(len(points))
        for alpha,coefficient in operator.terms:
            if not any(alpha):data=self.values
            elif alpha in self.derivatives:data=self.derivatives[alpha]
            else:raise ValueError(f"Missing initial derivative {alpha}")
            if ctx and callable(data) and not hasattr(data,"mp_values"):
                raise TypeError("Extended initial derivatives require PrecisionData callbacks")
            vals=mp_values(data,points,ctx) if ctx else values(data,points)
            c=mp_number(ctx,coefficient) if ctx else float(coefficient)
            for i,v in enumerate(vals):out[i]+=c*v
        return out
