"""Manufactured-solution checks; optional backends have explicit opt-in gates."""
import importlib.util
import os
import numpy as np
import pytest
from examples import annulus,ellipse_boundary,heat_flower,ball_poisson,stokes_annulus


def test_curved_scalar_examples():
    for example,tolerance in ((annulus,.015),(ellipse_boundary,.003),(ball_poisson,.002)):
        metrics=example.run()[3]
        assert metrics["nodal_max"]<tolerance and metrics["off_node_max"]<tolerance
    assert heat_flower.run(steps=4)[3]["off_node_max"]<.004


def test_annulus_methods_and_refinement():
    for method in ("global","lhi"):
        assert annulus.run(interior=120,method=method)[3]["off_node_max"]<.03
    for seed in (17,73):
        coarse=annulus.run(interior=120,seed=seed)[3]["off_node_max"]
        fine=annulus.run(interior=480,seed=seed)[3]["off_node_max"]
        assert fine < coarse


def test_annular_stokes():
    m=stokes_annulus.run()[3]
    assert m["off_node_velocity_max"]<.006
    assert m["pressure_gradient_max"]<.08
    assert m["divergence_max"]<1e-10


@pytest.mark.parametrize("backend",["cpp","torch"])
def test_optional_curved_backend_parity(backend):
    if backend=="cpp" and os.environ.get("RBFLAB_CPP_TESTS")!="1":
        pytest.skip("Set RBFLAB_CPP_TESTS=1 after preparing native dependencies")
    if backend=="torch" and importlib.util.find_spec("torch") is None:
        pytest.skip("Install rbflab[torch]")
    _,cloud,python,_=annulus.run(interior=80)
    _,_,other,_=annulus.run(interior=80,backend=backend)
    np.testing.assert_allclose(python.evaluate(cloud.points),other.evaluate(cloud.points),rtol=2e-8,atol=2e-9)
