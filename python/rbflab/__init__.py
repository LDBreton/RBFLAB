"""RBFLAB: approximation spaces, differential operators and RBF methods.

Geometry/generation and diagnostics live in their named submodules.
Historical/experimental algorithms are not part of this stable namespace.
"""
from . import geometry, meshgen
from .geometry import PointCloud
from .precision import Precision, PrecisionData
from .configuration import LocalSolver
from .kernels import IMQ, Gaussian, PHS, Hybrid, DivergenceFree
from .symbolic_kernel import Kernel, Wendland
from .operators import Identity, Derivative, Laplacian, NormalDerivative, Robin, SpatialOperator
from .problems import LinearPDE, BoundaryCondition, Dirichlet
from .evolution import EvolutionPDE
from .symbolic import SymbolicScalar
from .symbolic_system import SymbolicSystem
from .spaces import ScalarSpace, DivergenceFreeSpace, PressureSpace, MixedSpace, SymbolicStokes
from .mixed import FieldOperator
from .field_expressions import laplacian, gradient, vector, derivative, dx, dy, dz
from .mixed_backends import LocalExecution
from .stencils import StencilPolicy
from .samples import Samples, Stencil, SharedStencil
from .local_approximation import LocalApproximation
from .centers import CenterGroup
from .methods import GlobalCollocation, LHI, interpolate
from .patches import PatchCover
from .rbf_fd import RBFFD
from .discrete_operators import DiscreteOperator, OperatorSet
from .lhi_backends import PythonBackend, CppBackend
from .global_backend import CppGlobalBackend
from .torch_backend import TorchBackend
from .execution import execution_info

__all__ = [
    'geometry', 'meshgen', 'PointCloud', 'Precision', 'PrecisionData', 'LocalSolver',
    'IMQ', 'Gaussian', 'PHS', 'Hybrid', 'DivergenceFree', 'Kernel', 'Wendland',
    'Identity', 'Derivative', 'Laplacian', 'NormalDerivative', 'Robin', 'SpatialOperator',
    'laplacian', 'gradient',
    'vector', 'derivative', 'dx', 'dy', 'dz',
    'SharedStencil',
    'Stencil',
    'LinearPDE', 'BoundaryCondition', 'Dirichlet', 'EvolutionPDE',
    'SymbolicScalar', 'SymbolicSystem', 'SymbolicStokes',
    'ScalarSpace', 'DivergenceFreeSpace', 'PressureSpace', 'MixedSpace', 'FieldOperator', 'StencilPolicy', 'Samples', 'LocalApproximation', 'CenterGroup',
    'GlobalCollocation', 'LHI', 'RBFFD', 'interpolate', 'PatchCover', 'DiscreteOperator', 'OperatorSet',
    'PythonBackend', 'CppBackend', 'CppGlobalBackend', 'TorchBackend', 'LocalExecution', 'execution_info',
]
