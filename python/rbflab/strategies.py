"""Experimental fixed recipes, separate from discretization defaults."""
import math
from .geometry import PointCloud
from .kernels import Gaussian,Hybrid,PHS
from .lhi_stokes import LHIUnsteadyStokes
from .precision import Precision
from .stencils import StencilPolicy


def growing_stencil_size(point_count,factor=5):
    """ceil(factor*sqrt(N)), capped at N; not a 3D scaling prescription."""
    if type(point_count) is not int or point_count<3:
        raise ValueError('point_count must be an integer >=3')
    if not math.isfinite(float(factor)) or float(factor)<=0:
        raise ValueError('factor must be finite and positive')
    return min(point_count,max(3,math.ceil(float(factor)*math.sqrt(point_count))))


def growing_hybrid_stokes(cloud,precision=None,*,fields=None):
    """Fixed experimental 2D Stokes refinement recipe.

    Validated on unit-square perturbed grids, N=49..289, viscosity one.
    Physical kernel parameters are not invariant under rescaling the domain.
    No parameters depend on exact solutions or cloud-specific error searches.
    Pass fields=(U,p) for the symbolic space API, with the identical recipe.
    Larger stencils increase cost; nonzero-pressure convergence is not certified.
    """
    if not isinstance(cloud,PointCloud):raise TypeError('Expected PointCloud')
    method = LHIUnsteadyStokes(Hybrid(Gaussian('.5'),PHS(7),'1e-5'),
        stencil_size=growing_stencil_size(len(cloud.points)),
        pressure_kernel=Hybrid(Gaussian('.0005'),PHS(3),'1e-6'),
        legacy_unaugmented_hybrid=True,min_boundary_centers=min(5,len(cloud.boundary_indices)),
        stencil_policy=StencilPolicy(scaling='physical'),
        precision=precision if precision is not None else Precision())

    if fields is None:return method
    if len(fields)!=2:raise ValueError("Expected fields=(velocity, pressure)")
    from .methods import LHI
    from .spaces import DivergenceFreeSpace,PressureSpace
    velocity,pressure=fields
    return LHI(stencil_size=method.stencil_size,precision=method.precision,
        stencil_policy=method.stencil_policy,
        spaces={velocity:DivergenceFreeSpace(method.kernel,polynomial_degree=None),
                pressure:PressureSpace(method.pressure_kernel,polynomial_degree=None)},
        legacy_unaugmented_hybrid=True,min_boundary_centers=method.min_boundary_centers)
