"""Quadratic pressure must be reproduced without global pressure samples."""
import json
import numpy as np
import pytest
import sympy as sp

import rbflab as r
from scipy.sparse import csr_matrix, eye, save_npz
from research.navier_stokes_schemes import control_maps, control_cavity


def synthetic_map_folder(tmp_path, augmentation=(6,2), amplitude='nu^2*q^4'):
    cloud=r.geometry.unit_box_grid(6)
    inside, boundary=cloud.interior,cloud.points[cloud.boundary_indices]
    folder=tmp_path/'maps';folder.mkdir()
    recipe=dict(cells=6,nu=.01,augmentation=list(augmentation),pressure_amplitude=amplitude)
    (folder/'configuration.json').write_text(json.dumps(recipe))
    np.savez(folder/'nodes.npz',points=inside,boundary_points=boundary)
    n,nb=2*len(inside),2*len(boundary)
    for name in control_maps.NAMES:
        shape=(n,nb if name in ('B','PB','DxB','DyB') else n)
        matrix=eye(n,format='csr') if name in ('A','M') else csr_matrix(shape)
        save_npz(folder/f'{name}.npz',matrix)
    return folder


def test_saved_quadratic_maps_load_the_actual_trial_space(tmp_path):
    folder=synthetic_map_folder(tmp_path)
    problem=control_maps.load_maps(folder)
    assert problem.space.spaces['u'].polynomial_degree==6
    assert problem.space.spaces['p'].polynomial_degree==2
    assert isinstance(problem.space.spaces['p'],r.PressureSpace)
    assert problem.map_configuration['augmentation']==[6,2]
    assert set(problem.source)=={'velocity','boundary','momentum'}


def test_cavity_configuration_records_quadratic_pressure(tmp_path):
    folder=synthetic_map_folder(tmp_path)
    control_cavity.run(folder,tmp_path/'run',.00025,end=.0005,method='newton_krylov')
    config=json.loads((tmp_path/'run/configuration.json').read_text())
    assert config['augmentation']=='6/2'
    assert config['local_pressure_elimination'] is True
    assert config['stencil']==37


def test_quadratic_loader_requires_explicit_normalized_pressure(tmp_path):
    folder=synthetic_map_folder(tmp_path,amplitude='q^4')
    with pytest.raises(ValueError,match='viscosity-normalized'):
        control_maps.load_maps(folder)


@pytest.fixture(scope="module")
def quadratic_local_maps():
    cloud = r.geometry.unit_box_grid(6)
    interior = cloud.interior
    boundary = cloud.points[cloud.boundary_indices]
    targets = interior[[0, len(interior)//2]]
    q, nu = sp.Rational(6, 44), sp.Rational(1, 100)
    s = sp.Symbol("s", nonnegative=True)
    space = r.MixedSpace({
        "u": r.DivergenceFreeSpace(r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7), 6),
        "p": r.PressureSpace(r.Kernel(nu**2*q**4*sp.exp(-q*q*s), s)(), 2),
    })
    u, p = space.fields
    momentum = -nu*r.laplacian(u)+r.gradient(p)
    stencil = r.Stencil(cloud.points, size=37)
    source = {
        "velocity": r.Samples(interior, operator=u, within=stencil),
        "boundary": r.Samples(boundary, operator=u, within=stencil),
        "momentum": r.Samples(interior, operator=momentum, within=stencil, target="exclude"),
    }
    local = r.LocalApproximation(source=source, trial=space.representers(source),
        precision=r.Precision(local_digits=60), backend=r.PythonBackend())
    maps = local.operators(targets=targets, operators={
        "velocity":u, "momentum":momentum, "gradient":r.gradient(p),
        "dx":r.dx(u), "dy":r.dy(u)})
    return maps, interior, boundary, targets


@pytest.mark.parametrize("pressure", ["x-y", "x2", "xy", "y2"])
def test_pressure_balance_has_zero_velocity_and_exact_gradient(quadratic_local_maps, pressure):
    maps, interior, boundary, targets = quadratic_local_maps
    def gradient(points):
        x, y = points.T
        return {"x-y":np.column_stack((np.ones_like(x), -np.ones_like(y))),
                "x2":np.column_stack((2*x, 0*y)),
                "xy":np.column_stack((y, x)),
                "y2":np.column_stack((0*x, 2*y))}[pressure]
    data = {"velocity":np.zeros_like(interior), "boundary":np.zeros_like(boundary),
            "momentum":gradient(interior)}
    for name in ("velocity", "dx", "dy"):
        actual = np.asarray((maps[name]@data).tolist(), dtype=float)
        np.testing.assert_allclose(actual, np.zeros_like(targets), atol=1e-11, rtol=0)
    for name in ("momentum", "gradient"):
        actual = np.asarray((maps[name]@data).tolist(), dtype=float)
        np.testing.assert_allclose(actual, gradient(targets), atol=1e-11, rtol=0)


def test_quadratic_velocity_and_pressure_share_the_same_reconstruction(quadratic_local_maps):
    maps, interior, boundary, targets = quadratic_local_maps
    def velocity(points):
        x, y = points.T
        return np.column_stack((x*x, -2*x*y))
    x, y = interior.T
    data = {"velocity":velocity(interior), "boundary":velocity(boundary),
            "momentum":np.column_stack((y-.02, x))}
    x, y = targets.T
    exact = {"velocity":velocity(targets), "momentum":np.column_stack((y-.02,x)),
             "gradient":np.column_stack((y,x)), "dx":np.column_stack((2*x,-2*y)),
             "dy":np.column_stack((0*x,-2*x))}
    for name, target in exact.items():
        actual = np.asarray((maps[name]@data).tolist(),dtype=float)
        np.testing.assert_allclose(actual,target,atol=1e-11,rtol=0)
