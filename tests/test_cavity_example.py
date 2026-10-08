"""Small physical and algebraic checks for the standalone cavity example."""
import numpy as np

from examples.navier_stokes_cavity import clouds, solve


def test_cavity_example_constraints():
    velocity, pressure = clouds(6)
    points, times, frames, diagnostics = solve(cells=6, end=.005)
    np.testing.assert_array_equal(points, velocity.points)
    assert len(pressure.points) == 49
    assert times[0] == 0 and times[-1] == .005
    assert np.isfinite(frames).all()
    top = velocity.boundary["top"]
    still = np.setdiff1d(velocity.boundary_indices, top)
    np.testing.assert_allclose(frames[:, top, 0], 1, atol=1e-12)
    np.testing.assert_allclose(frames[:, top, 1], 0, atol=1e-12)
    np.testing.assert_allclose(frames[:, still], 0, atol=1e-12)
    assert abs(diagnostics["pressure_mean"]) < 1e-10
    assert abs(diagnostics["max_divergence"] - abs(diagnostics["compatibility"])) < 1e-10
    assert diagnostics["max_speed"] < 1.1
