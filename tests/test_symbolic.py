"""Symbolic front end checked against analytic data and independent numeric APIs."""
from fractions import Fraction
import numpy as np
import pytest
import sympy as sp
import mpmath
from numpy.testing import assert_allclose
import rbflab as r
from rbflab.precision import mp_number
from rbflab.problems import boundary_data


def model_problem(dimension):
    model=r.SymbolicScalar(dimension);u=model.field;x=model.coordinates[0]
    exact=1+sum((i+1)*v*v for i,v in enumerate(model.coordinates))
    expression=-model.laplacian(u)+sp.pi*u+sp.diff(u,model.coordinates[-1])
    # Coordinate-dependent Robin plus an oblique Cartesian contribution.
    operator=(2+sp.sin(x))*u+model.normal_derivative(u)+sp.Rational(1,3)*sp.diff(u,x)
    others=['left','right','bottom'] if dimension==2 else ['left','right','front','back','bottom']
    problem=model.stationary(sp.Eq(expression,expression.subs(u,exact).doit()),boundary=[
        model.bc(others,sp.Eq(u,exact)),
        model.bc('top',sp.Eq(operator,operator.subs(u,exact).doit()))])
    return model,problem,exact


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('name',['symmetric','asymmetric','rbffd','lhi'])
@pytest.mark.parametrize('digits',[None,40])
def test_custom_boundary_and_irrational_pde_all_methods(dimension,name,digits):
    model,problem,exact=model_problem(dimension)
    cloud=r.geometry.unit_box_grid(2, dimension)
    if name in ('symmetric','asymmetric'):
        method=r.GlobalCollocation(r.PHS(5),scheme=name,polynomial_degree=2,
                                    precision=r.Precision(global_digits=digits))
    else:
        method=(r.RBFFD if name=='rbffd' else r.LHI)(r.PHS(5),len(cloud.points),polynomial_degree=2,
            precision=r.Precision(local_digits=digits,global_dtype='mpmath' if digits else 'float64'))
    solution=problem.solve(cloud,method)
    q=np.random.default_rng(18).uniform(.1,.9,(5,dimension))
    expected=model.data(exact)
    assert_allclose(solution.evaluate(q),expected(q),atol=2e-10)
    for node,(operator,target) in boundary_data(problem,cloud).items():
        assert_allclose(solution.evaluate(cloud.points[[node]],operator),target,atol=2e-9)
    if digits:
        values=solution.evaluate_mp(q);ctx=values.ctx
        assert max(abs(v-w) for v,w in zip(values,expected.mp_values(ctx,q)))<ctx.mpf('1e-30')


@pytest.mark.parametrize('dimension',[2,3])
def test_parameter_binding_derivative_expansion_and_source_operator(dimension):
    m=r.SymbolicScalar(dimension);u=m.field;x=m.coordinates[0];z=m.coordinates[-1]
    nu=sp.Symbol('nu',positive=True)
    expression=-nu*m.laplacian(u)+sp.diff(x*u,x)-x*sp.diff(u,x)+sp.diff(u,x,z)
    problem=m.stationary(sp.Eq(expression,0),boundary=[m.bc('boundary',sp.Eq(u,0))],parameters={nu:'0.125'})
    expected=-Fraction(1,8)*r.Laplacian(dimension)+r.Identity(dimension)+r.Derivative(0,dimension)@r.Derivative(dimension-1,dimension)
    assert problem.operator==expected
    variable=m.stationary(sp.Eq(x*sp.diff(u,x),0),boundary=[])
    assert isinstance(variable.operator,r.SpatialOperator)
    with pytest.raises(ValueError,match='Unbound parameters'):
        m.stationary(sp.Eq(nu*m.laplacian(u),0),boundary=[])


def test_private_context_data_and_constants():
    m=r.SymbolicScalar(3);x,y,z=m.coordinates
    data=m.data(sp.sin(x)+sp.exp(y)+sp.log(1+z)+sp.pi/7+sp.sqrt(2))
    points=np.array([[.125,.25,.5]])
    contexts=[mpmath.mp.clone(),mpmath.mp.clone()]
    for ctx,digits in zip(contexts,[60,35]):
        ctx.dps=digits
        expected=ctx.sin(ctx.mpf('.125'))+ctx.exp(ctx.mpf('.25'))+ctx.log(ctx.mpf('1.5'))+ctx.pi/7+ctx.sqrt(2)
        assert abs(data.mp_values(ctx,points)[0]-expected)<ctx.mpf(10)**(-digits+5)
        assert abs(mp_number(ctx,sp.pi)-ctx.pi)<ctx.mpf(10)**(-digits+5)
    op=sp.pi*r.Laplacian(3)+sp.sqrt(2)*r.Identity(3)
    assert op.dimension==3
    with pytest.raises(ValueError,match='constant'):r.Identity()*sp.Symbol('a')


@pytest.mark.parametrize('dimension',[2,3])
@pytest.mark.parametrize('name',['symmetric','asymmetric','rbffd','lhi'])
def test_symbolic_heat_normalization_and_extended_data(dimension,name):
    m=r.SymbolicScalar(dimension,transient=True);u=m.field;t=m.time
    exact=(1+t)*(1+sum(v*v for v in m.coordinates))
    lhs=2*sp.diff(u,t)-m.laplacian(u)
    problem=m.evolution(sp.Eq(lhs,lhs.subs(u,exact).doit()),initial=exact.subs(t,0),
        boundary=[m.bc('boundary',sp.Eq(3*u,3*exact))])
    assert problem.operator==-Fraction(1,2)*r.Laplacian(dimension)
    cloud=r.geometry.unit_box_grid(2, dimension)
    if name in ('symmetric','asymmetric'):
        method=r.GlobalCollocation(r.PHS(5),scheme=name,polynomial_degree=2,precision=r.Precision(global_digits=35))
    else:
        method=(r.RBFFD if name=='rbffd' else r.LHI)(r.PHS(5),len(cloud.points),polynomial_degree=2,
            precision=r.Precision(local_digits=35,global_dtype='mpmath'))
    state=problem.solve(cloud,method,'.05',2).final
    points=np.full((1,dimension),.25);actual=state.evaluate_mp(points);ctx=actual.ctx
    expected=(1+ctx.mpf('.1'))*(1+dimension*ctx.mpf('.25')**2)
    assert abs(actual[0]-expected)<ctx.mpf('1e-25')
    assert ctx.norm(state.residuals(points,extended=True),'inf')<ctx.mpf('1e-24')


@pytest.mark.parametrize('kind',['square','product','sine','other_field'])
def test_reject_nonlinear_or_coupled_expression(kind):
    m=r.SymbolicScalar();u=m.field;x,y=m.coordinates
    expr={'square':u**2,'product':sp.diff(u,x)*sp.diff(u,y),'sine':sp.sin(u),
          'other_field':sp.Function('v')(x,y)+u}[kind]
    with pytest.raises((ValueError,NotImplementedError)):
        m.stationary(sp.Eq(expr,0),boundary=[])


def test_boundary_data_errors_and_priority():
    m=r.SymbolicScalar();u=m.field;x,y=m.coordinates
    cloud=r.geometry.unit_box_grid(2)
    problem=m.stationary(sp.Eq(-m.laplacian(u),0),boundary=[
        m.bc('left',sp.Eq(u,1)),m.bc('boundary',sp.Eq(u,2))])
    bd=boundary_data(problem,cloud)
    assert all(bd[int(j)][1]==1 for j in cloud.boundary['left'])
    expr=m.normal_derivative(u)+u
    problem=m.stationary(sp.Eq(-m.laplacian(u),0),boundary=[m.bc('boundary',sp.Eq(expr,m.normal[0]))])
    with pytest.raises(ValueError,match='Normals'):
        boundary_data(problem,r.PointCloud(cloud.points,cloud.boundary,{}))
    with pytest.raises(ValueError,match='Unbound'):
        m.stationary(sp.Eq(-m.laplacian(u),sp.Symbol('missing')),boundary=[])
    with pytest.raises(TypeError,match='SymPy expressions'):
        m.stationary('laplacian(u)=0',boundary=[])
    with pytest.raises(ValueError,match='Parameter keys'):
        m.stationary(sp.Eq(u,0),boundary=[],parameters={x:2})
    with pytest.raises(ValueError,match='finite real'):
        m.stationary(sp.Eq(u,0),boundary=[],parameters={sp.Symbol('a'):sp.oo})
    with pytest.raises(ValueError,match='no nonzero'):
        m.stationary(sp.Eq(u,0),boundary=[m.bc('boundary',sp.Eq(0,0,evaluate=False))])


def test_evolution_capability_errors():
    m=r.SymbolicScalar(transient=True);u=m.field;t=m.time;x=m.coordinates[0]
    bc=[m.bc('boundary',sp.Eq(u,0))]
    for derivative in [sp.diff(u,t,2),sp.diff(u,x,t)]:
        with pytest.raises(NotImplementedError,match='first time'):
            m.evolution(sp.Eq(derivative-m.laplacian(u),0),initial=0,boundary=bc)
    with pytest.raises(ValueError,match='time-dependent'):
        m.evolution(sp.Eq(sp.diff(u,t)-m.laplacian(u),0),initial=0,
                    boundary=[m.bc('boundary',sp.Eq((1+t)*u+m.normal_derivative(u),0))])
    with pytest.raises(ValueError,match='first time'):
        m.evolution(sp.Eq(-m.laplacian(u),0),initial=0,boundary=bc)
    with pytest.raises(ValueError,match='non-transient'):
        m.stationary(sp.Eq(u,0),boundary=bc)



def test_standalone_operator_and_boundary_without_normals():
    m=r.SymbolicScalar();x,y=m.coordinates;u=m.field
    op=m.operator(-m.laplacian(u)+sp.Rational(1,7)*sp.diff(u,x,y))
    assert op==-r.Laplacian()+Fraction(1,7)*(r.Derivative(0)@r.Derivative(1))
    with pytest.raises(ValueError,match='independent'):
        m.operator(-m.laplacian(u)+1)
    cloud=r.geometry.unit_box_grid(2)
    exact=1+x*x+y*y
    boundary_operator=(1+x)*u+sp.diff(u,y)
    problem=m.stationary(sp.Eq(-m.laplacian(u),-4),boundary=[
        m.bc(['left','right','bottom'],sp.Eq(u,exact)),
        m.bc('top',sp.Eq(boundary_operator,boundary_operator.subs(u,exact).doit()))])
    without_normals=r.PointCloud(cloud.points,cloud.boundary,{})
    sol=problem.solve(without_normals,r.GlobalCollocation(r.PHS(5),polynomial_degree=2))
    query=np.array([[.25,.375]])
    assert_allclose(sol.evaluate(query),m.data(exact)(query),atol=1e-10)
