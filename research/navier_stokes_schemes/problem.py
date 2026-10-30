"""One readable public-API mixed LHI setup shared by the time schemes."""
from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.sparse import eye

import rbflab as r


@dataclass
class Problem:
    cells: int = 6
    nu: float = 1.0
    ranks: int = 2
    digits: int = 80
    offset: float = 0.0
    native_timeout: float = 3600
    linear_solver: str = 'direct'
    global_ordering: str = 'COLAMD'
    reuse_stencils: bool = False
    weight_folder: object = None
    build_maps: bool = True

    def __post_init__(self):
        cloud = r.geometry.unit_box_grid(self.cells)
        self.points = cloud.interior
        self.boundary_points = cloud.points[cloud.boundary_indices]
        self.bounds = np.array([[0., 1.], [0., 1.]])
        self.stencil = r.Stencil(cloud.points, size=37)
        q = sp.Rational(self.cells, 44)
        velocity_kernel = r.Hybrid(r.Gaussian(10*q*q), r.PHS(7), q**7)
        s = sp.Symbol('s', nonnegative=True)
        pressure_kernel = r.Kernel(q**4*sp.exp(-q*q*s), s)()
        self.space = r.MixedSpace({
            'u': r.DivergenceFreeSpace(velocity_kernel, polynomial_degree=6),
            'p': r.PressureSpace(pressure_kernel, polynomial_degree=1),
        })
        self.u, self.p = self.space.fields
        momentum = -self.nu*r.laplacian(self.u) + r.gradient(self.p)
        self.source = {
            'velocity': r.Samples(self.points, operator=self.u, within=self.stencil),
            'boundary': r.Samples(self.boundary_points, operator=self.u, within=self.stencil),
            'momentum': r.Samples(self.points, operator=momentum,
                                  within=self.stencil, target='exclude'),
        }
        self.options = dict(backend=r.CppBackend(threads=1, timeout=self.native_timeout),
                            execution=r.LocalExecution(scheduler='mpi', ranks=self.ranks),
                            precision=r.Precision(local_digits=self.digits))
        self._manufactured_fields()
        self._evaluator = None
        self.maps = self.local = None
        if not self.build_maps:
            return
        if self.reuse_stencils:
            from .reused import assemble
            blocks, self.reuse_info = assemble(self, cloud)
            for name, matrix in blocks.items():
                setattr(self, name, matrix)
            return
        local = r.LocalApproximation(source=self.source,
                                     trial=self.space.representers(self.source), **self.options)
        self.local = local
        self.maps = local.operators(targets=self.points, operators={
            'L': momentum, 'P': r.gradient(self.p),
            'Dx': r.dx(self.u), 'Dy': r.dy(self.u), 'value': self.u})
        self.A, self.B, self.SL = self.blocks(self.maps.L)
        self.M = eye(self.A.shape[0], format='csr') - self.SL
        self.PY, self.PB, self.PL = self.blocks(self.maps.P)
        self.DxU, self.DxB, self.DxL = self.blocks(self.maps.Dx)
        self.DyU, self.DyB, self.DyL = self.blocks(self.maps.Dy)

    @staticmethod
    def blocks(operator):
        return tuple(operator[name].to_scipy()
                     for name in ('velocity', 'boundary', 'momentum'))

    def _manufactured_fields(self):
        x, y = sp.symbols('x y')
        psi = 256*x**2*(1-x)**2*y**2*(1-y)**2
        v = sp.Matrix([sp.diff(psi, y)+self.offset, -sp.diff(psi, x)-self.offset])
        grad_p = sp.Matrix([1, -1])
        ell = -self.nu*(v.diff(x, 2)+v.diff(y, 2)) + grad_p
        convection = v.jacobian([x, y])*v
        kinetic_gradient = sp.Matrix([sp.diff(v.dot(v)/2, z) for z in (x, y)])

        def sample(expression, points):
            return np.column_stack([np.broadcast_to(
                sp.lambdify((x, y), e, 'numpy')(*points.T), len(points))
                for e in expression]).ravel()

        self.U0 = sample(v, self.points)
        self.L0 = sample(ell, self.points)
        self.G0 = sample(grad_p, self.points)
        self.K0 = sample(kinetic_gradient, self.points)
        self.C0 = sample(convection, self.points)
        self.g0 = sample(v, self.boundary_points)

    def initial(self):
        return self.U0.copy(), self.L0.copy()

    def initial_rotational(self):
        return self.U0.copy(), self.L0+self.K0

    def force(self, t):
        return np.exp(-t)*(self.L0-self.U0) + np.exp(-2*t)*self.C0

    def boundary(self, t):
        return np.exp(-t)*self.g0

    def exact(self, t):
        return np.exp(-t)*self.U0, np.exp(-t)*self.G0

    def evaluate(self, U, L, t, points):
        if np.array_equal(points, self.points):
            return U.reshape(-1, 2)
        if self.maps is None or self.local is None:
            raise NotImplementedError('Moving-point evaluation requires retained, unreused local fits')
        if self._evaluator is None:
            from .reconstruction import FixedStencilReconstruction
            self._evaluator = FixedStencilReconstruction(self.local, self.maps.value, self.u)
        return self._evaluator.evaluate(U, L, self.boundary(t), points)
