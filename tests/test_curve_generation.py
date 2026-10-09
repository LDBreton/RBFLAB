"""Curve differentiation, outward normals and unified generation contracts."""
import numpy as np
import sympy as sp
import pytest
from rbflab import geometry as g, meshgen as m


def test_symbolic_callable_and_explicit_tangents_agree():
    t = sp.symbols("t", real=True)
    curves = [g.ParametricBoundary((2*sp.cos(t), sp.sin(t)), parameter=t, interval=(0, 2*sp.pi)),
              g.ParametricBoundary(lambda q: (2*np.cos(q), np.sin(q))),
              g.ParametricBoundary(lambda q: (2*np.cos(q), np.sin(q)),
                                   tangent=lambda q: (-2*np.sin(q), np.cos(q)))]
    assert [c.derivative_source for c in curves] == ["symbolic", "finite_difference", "explicit"]
    clouds = [m.generate(g.ParametricDomain(c), interior=30, boundary=80) for c in curves]
    for cloud in clouds:
        p = cloud.points[cloud.boundary["boundary"]]
        expected = p/[4, 1]
        expected /= np.linalg.norm(expected, axis=1)[:, None]
        np.testing.assert_allclose(cloud.normals["boundary"], expected, atol=2e-9)
    np.testing.assert_allclose(clouds[0].points, clouds[1].points, atol=1e-14)


def test_endpoint_difference_does_not_evaluate_outside_interval():
    def bounded(t):
        assert 0 <= t <= 1
        return t, t*t
    arc = g.ParametricBoundary(bounded, interval=(0, 1))
    for t in (0, 1e-7, .5, 1-1e-7, 1):
        np.testing.assert_allclose(arc._geometry.tangent(t), (1, 2*t), atol=1e-9)


def test_border_generation_is_nonmutating_and_orients_holes():
    t = sp.symbols("t", real=True)
    outer = g.Border((sp.cos(t), sp.sin(t)), "outer", 0, 2*sp.pi)(64)
    hole = g.Border((.4*sp.cos(t), .4*sp.sin(t)), "hole", 0, 2*sp.pi)(-32)
    mesh = m.RBFMesh(outer, hole)
    mesh.generate_points(10, seed=8)
    old_points = list(mesh.Points)
    cloud = m.generate(mesh, interior=70, seed=9)
    again = m.generate([outer, hole], interior=70, seed=9)
    assert mesh.Points == old_points and hole.n_segments == -32
    np.testing.assert_array_equal(cloud.points, again.points)
    assert len(cloud.points) == 166
    for label, sign in (("outer", 1), ("hole", -1)):
        p = cloud.points[cloud.boundary[label]]
        expected = sign*p/np.linalg.norm(p, axis=1)[:, None]
        np.testing.assert_allclose(cloud.normals[label], expected, atol=1e-13)
    with pytest.raises(ValueError, match="omit boundary"):
        m.generate(mesh, boundary=50)
    with pytest.raises(ValueError, match="max_candidates"):
        m.generate(mesh, interior=20, boundary_distance=2, max_candidates=50)


def test_parametric_hole_orientation_and_explicit_normal_override():
    t = sp.symbols("t", real=True)
    outer = g.ParametricBoundary((sp.cos(t), -sp.sin(t)), label="outer")
    hole = g.ParametricBoundary((.3*sp.cos(t), .3*sp.sin(t)), label="hole",
                                normal=(-2*sp.cos(t), -2*sp.sin(t)))
    cloud = m.generate(g.ParametricDomain(outer, holes=[hole]), interior=10,
                       boundary={"outer": 32, "hole": 16})
    for label, sign in (("outer", 1), ("hole", -1)):
        p = cloud.points[cloud.boundary[label]]
        np.testing.assert_allclose(cloud.normals[label], sign*p/np.linalg.norm(p,axis=1)[:,None])
    assert hole.derivative_source == "symbolic"
    clone = g.with_holes(g.Disk(), {"hole": g.ParametricDomain(hole)})
    assert clone.boundaries["hole"].derivative_source == "symbolic"


def test_explicit_border_normal_is_authoritative_and_normalized():
    border = g.Border(lambda t: (np.cos(t), np.sin(t)), "wall", 0, 2*np.pi,
                      normal=lambda t: (-2*np.cos(t), -2*np.sin(t)))(32)
    # Deliberately inward override: callers own the direction; do not silently flip.
    cloud = m.generate(border, interior=5)
    p = cloud.points[cloud.boundary["wall"]]
    np.testing.assert_allclose(cloud.normals["wall"], -p, atol=1e-14)
    cloud = m.generate(border, interior=5, normals={"wall": lambda p: 3*p})
    np.testing.assert_allclose(cloud.normals["wall"], p, atol=1e-14)
    with pytest.raises(ValueError, match="exterior boundary"):
        m.generate(border, normals={"missing": lambda p: p})
    with pytest.raises(ValueError, match="nonzero"):
        m.generate(border, normals={"wall": lambda p: p*0})
    with pytest.raises(ValueError, match="real-valued"):
        m.generate(border, normals={"wall": lambda p: p*(1+1j)})


def test_joined_arcs_group_labels_and_detect_internal_interfaces():
    borders = [g.Border(lambda t: (t, 0), "wall", 0, 1)(8),
               g.Border(lambda t: (1, t), "outlet", 0, 1)(8),
               g.Border(lambda t: (1-t, 1), "wall", 0, 1)(8),
               g.Border(lambda t: (0, 1-t), "inlet", 0, 1)(8)]
    cloud = m.generate(borders, interior=30)
    assert len(cloud.boundary["wall"]) == 16
    np.testing.assert_allclose(cloud.normals["wall"][:8], np.tile((0,-1),(8,1)), atol=1e-10)
    np.testing.assert_allclose(cloud.normals["wall"][8:], np.tile((0,1),(8,1)), atol=1e-10)
    inside = g.Border(lambda t: (.5+.2*np.cos(t), .5+.2*np.sin(t)), "interface", 0, 2*np.pi)(32)
    cloud = m.generate([*borders, inside], interior=30)
    assert len(cloud.interfaces["interface"]) == 32
    assert "interface" not in cloud.boundary and "interface" not in cloud.normals
    assert len(cloud.interior_indices) == 62


def test_reversed_parameter_interval_and_numerical_border_normals():
    circle = g.Border(lambda t: (np.cos(t), np.sin(t)), "wall", 2*np.pi, 0)(-32)
    cloud = m.generate(circle, interior=10)
    p = cloud.points[cloud.boundary["wall"]]
    np.testing.assert_allclose(cloud.normals["wall"], p, atol=2e-9)


def test_invalid_curve_specs_and_degenerate_tangent():
    t, a = sp.symbols("t a", real=True)
    with pytest.raises(ValueError, match="unbound"):
        g.ParametricBoundary((a*sp.cos(t), sp.sin(t)), parameter=t)
    with pytest.raises(ValueError, match="bind extra"):
        g.Border((a*sp.cos(t), sp.sin(t)), "x", 0, 1)
    with pytest.raises(ValueError, match="quarter"):
        g.ParametricBoundary(lambda t: (t,t*t), interval=(0,1), difference_step=1)
    with pytest.raises(ValueError, match="nonzero"):
        g.ParametricBoundary((t*t, t*t*t), interval=(0,1)).sample(10)
    with pytest.raises(ValueError, match="nonzero"):
        g.ParametricBoundary((sp.cos(t),sp.sin(t)), normal=(0,0)).sample(10)


def test_surface_uses_unified_generation_without_mutating_samples():
    surface = m.ParametricSurface3D(lambda u,v: (u,v,0*u), (0,1),(0,1), label="sheet")
    cloud = m.generate(surface, interior=0, boundary=50, seed=7)
    assert surface.Boundary_Points == []
    assert cloud.points.shape == (50,3) and len(cloud.interior_indices)==0
    np.testing.assert_allclose(cloud.normals["sheet"], np.tile((0,0,1),(50,1)))
    with pytest.raises(ValueError, match="no volume interior"):
        m.generate(surface)
