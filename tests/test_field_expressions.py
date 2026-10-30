import os
import numpy as np
import pytest
import rbflab as r


def ingredients():
    space = r.MixedSpace({'u': r.DivergenceFreeSpace(r.Gaussian(1), 2),
                          'p': r.PressureSpace(r.Gaussian(1), 1)})
    u, p = space.fields
    I, L, dx, dy = r.Identity(), r.Laplacian(), r.Derivative(0), r.Derivative(1)
    explicit = (r.FieldOperator({'u': [[I, 0], [0, I]]}),
                r.FieldOperator({'u': [[-L, 0], [0, -L]], 'p': [[dx], [dy]]}),
                r.FieldOperator({'p': [[dx], [dy]]}))
    return space, (u, -r.laplacian(u) + r.gradient(p), r.gradient(p)), explicit


def test_expression_shapes_and_three_dimensions():
    space, expressions, explicit = ingredients()
    for expression, block in zip(expressions, explicit):
        assert expression.compile(2) == block
    u, p = space.fields
    assert expressions[1].compile(3).arity == 3
    assert r.laplacian(r.gradient(p)).compile(3).arity == 3
    with pytest.raises(ValueError):
        u + p
    with pytest.raises(ValueError):
        r.gradient(u)
    with pytest.raises(TypeError):
        u * p
    with pytest.raises(ValueError):
        float('inf') * u


@pytest.mark.parametrize('backend', ['python', 'cpp', 'mpi'])
def test_expression_weights_and_evolution_match_explicit(backend):
    if backend != 'python' and os.environ.get('RBFLAB_CPP_TESTS') != '1':
        pytest.skip('Opt-in native tests')
    if backend == 'mpi' and os.environ.get('RBFLAB_MPI_TESTS') != '1':
        pytest.skip('Opt-in MPI tests')
    from scipy.sparse import eye
    from scipy.sparse.linalg import spsolve
    space, expressions, explicit = ingredients()
    cloud = r.geometry.unit_box_grid(2)
    outputs = []
    u, p = space.fields
    componentwise = (r.vector(u[0], u[1]),
                     r.vector(-r.laplacian(u[0])+r.dx(p), -r.laplacian(u[1])+r.dy(p)),
                     r.vector(r.derivative(p, axis=0), r.derivative(p, axis=1)))
    for value, momentum, gradient in (expressions, explicit, componentwise):
        source = {'velocity': r.Samples(cloud.points, operator=value, size=9),
                  'momentum': r.Samples(cloud.points, operator=momentum,
                                        within='velocity', target='exclude')}
        local = r.LocalApproximation(source=source, trial=space.representers(source),
            precision=r.Precision(local_digits=40),
            backend=r.PythonBackend(compute_condition=False) if backend == 'python'
                    else r.CppBackend(compute_condition=False),
            execution=r.LocalExecution(scheduler='mpi', ranks=2) if backend == 'mpi' else None)
        maps = local.operators(targets=cloud.points, operators={'L': momentum, 'P': gradient})
        A, SL = [maps.L[name].to_scipy() for name in source]
        PY, PL = [maps.P[name].to_scipy() for name in source]
        M = eye(18) - SL
        # Compare two BE/BDF2 steps with identical synthetic nodal data.
        previous = np.arange(18.) / 18
        fields = []
        for dt in (.00025, .000125):
            U = previous.copy()
            for step in (1, 2):
                history = U/dt if step == 1 else (2*U-.5*old)/dt
                alpha = 1/dt if step == 1 else 1.5/dt
                F = np.ones(18)
                new = spsolve((alpha*M+A).tocsc(), M @ (history+F))
                G = PY @ new + PL @ (F-alpha*new+history)
                old, U = U, new
            fields.extend([U, G])
        outputs.append(([a.toarray() for a in (M, A, PY, PL)], fields))
    for output in outputs[1:]:
        for left, right in zip(outputs[0][0]+outputs[0][1], output[0]+output[1]):
            np.testing.assert_array_equal(left, right)


def test_component_derivatives_and_shape_validation():
    space, _, _ = ingredients()
    u, p = space.fields
    for d in (2, 3):
        rebuilt = r.vector(*(u[i] for i in range(d)))
        assert rebuilt.compile(d) == u.compile(d)
        assert r.derivative(u, 0, order=0).compile(d) == u.compile(d)
        expected = r.Derivative(0, d) @ r.Derivative(1, d)
        assert r.dx(r.dy(p)).compile(d) == r.FieldOperator({'p': [[expected]]})
    assert r.dz(p).compile(3) == r.FieldOperator({'p': [[r.Derivative(2, 3)]]})
    assert r.vector(p, p, p, p).compile(2).arity == 4
    with pytest.raises(ValueError):
        (r.vector(p, p, p) + u).compile(2)
    with pytest.raises(ValueError):
        u[2].compile(2)
    with pytest.raises(ValueError):
        p[0]
    with pytest.raises(ValueError):
        u[-1]
    with pytest.raises(ValueError):
        r.dz(p).compile(2)
    with pytest.raises(ValueError):
        r.derivative(p, axis=0, order=-1)
    with pytest.raises(ValueError):
        r.vector(u)
    with pytest.raises(TypeError):
        r.vector()


@pytest.mark.parametrize('backend', ['python', 'cpp', 'mpi'])
def test_component_example_weights_and_polynomial_values(backend):
    if backend != 'python' and os.environ.get('RBFLAB_CPP_TESTS') != '1':
        pytest.skip('Opt-in native tests')
    if backend == 'mpi' and os.environ.get('RBFLAB_MPI_TESTS') != '1':
        pytest.skip('Opt-in MPI tests')
    from examples.component_operators import run
    maps = run('python' if backend == 'python' else 'cpp', 2 if backend == 'mpi' else 0)
    np.testing.assert_array_equal(maps.expression.to_scipy().toarray(), maps.blocks.to_scipy().toarray())
    points = maps.expression.input_layout['values']['points']
    x, y = points.T
    values = maps.expression @ {'values': np.c_[x*x+y, x*y]}
    np.testing.assert_allclose(np.asarray(values.tolist(), dtype=float), np.c_[4*x, np.ones(len(x))], atol=1e-12)
    if backend == 'mpi':
        assert maps.diagnostics['execution']['scheduler'] == 'mpi'
