import numpy as np

from rbflab.assembly import Factor
from rbflab.kernels import IMQ
from rbflab.nodal import Arithmetic
from rbflab.precision import MPBackend, MPFactor


def test_float64_factor_skips_condition_by_default(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("condition estimate should be opt-in")

    monkeypatch.setattr(np.linalg, "cond", unexpected)
    factor = Factor([[4.0, 1.0], [2.0, 3.0]])
    arithmetic = Arithmetic(IMQ())
    arithmetic_factor = arithmetic.factor([[4.0, 1.0], [2.0, 3.0]])

    assert factor.condition is None
    assert arithmetic_factor.condition is None


def test_float64_factor_condition_can_be_requested():
    factor = Factor([[4.0, 1.0], [2.0, 3.0]], compute_condition=True)
    assert factor.condition is not None
    assert factor.condition >= 1.0


def test_mp_factor_condition_is_opt_in_and_can_be_requested():
    backend = MPBackend(IMQ(), 40)
    matrix = backend.ctx.matrix([[4, 1], [2, 3]])

    factor = MPFactor(backend, matrix)
    estimated = MPFactor(backend, matrix, compute_condition=True)

    assert factor.condition is None
    assert factor.condition_mp is None
    assert estimated.condition is not None
    assert estimated.condition_mp is not None
    assert estimated.condition >= 1.0

