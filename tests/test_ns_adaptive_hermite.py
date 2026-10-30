"""Algebraic safeguards for the experimental target-frozen Oseen closure."""
import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

from research.navier_stokes_schemes.adaptive_hermite import corrected_source, system


def test_target_frozen_observations_include_variable_advector_correction():
    velocity = np.array([[1., 2.], [3., -1.]])
    dx = np.array([2., 4., 6., 8.])
    dy = np.array([1., 3., 5., 7.])
    stokes = np.array([10., 20., 30., 40.])
    global_oseen = stokes + np.repeat(velocity[:, 0], 2)*dx + np.repeat(velocity[:, 1], 2)*dy
    frozen = corrected_source(global_oseen, dx, dy, velocity, velocity[0])
    np.testing.assert_allclose(frozen, stokes+dx+2*dy)
    assert not np.allclose(frozen, global_oseen)


def test_coupled_system_reproduces_all_four_reconstruction_equations():
    n = 4
    identity = eye(n, format="csr")
    zero = csr_matrix((n, n))
    # Distinct corrections catch source-sign and unknown-order mistakes.
    maps = {"L": (.2*identity, .1*identity, .05*identity, .03*identity, -.02*identity),
            "X": (.3*identity, .2*identity, .01*identity, .02*identity, .01*identity),
            "Y": (-.1*identity, .3*identity, .02*identity, -.01*identity, .03*identity)}
    g, rhs_time = np.arange(n, dtype=float), np.arange(n, dtype=float)+1
    matrix, rhs = system(maps, 2., rhs_time, g)
    U, L, X, Y = np.split(spsolve(matrix, rhs), 4)
    np.testing.assert_allclose(2*U+L, rhs_time)
    for name, target in (("L", L), ("X", X), ("Y", Y)):
        a, b, s, cx, cy = maps[name]
        np.testing.assert_allclose(target, a@U+b@g+s@L+cx@X+cy@Y)


def test_constant_advector_needs_no_source_correction():
    momentum = np.arange(4, dtype=float)
    velocity = np.tile([1., -2.], (2, 1))
    np.testing.assert_array_equal(
        corrected_source(momentum, np.ones(4), np.ones(4), velocity, velocity[0]), momentum)
