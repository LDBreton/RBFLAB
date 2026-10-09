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
from .spaces import ScalarSpace, DivergenceFreeSpace, PressureSpace, SymbolicStokes
from .stencils import StencilPolicy
from .samples import Samples
from .local_approximation import LocalApproximation
from .centers import CenterGroup
from .methods import GlobalCollocation, LHI, interpolate
from .rbf_fd import RBFFD
from .discrete_operators import DiscreteOperator, OperatorSet
from .lhi_backends import PythonBackend, CppBackend
from .torch_backend import TorchBackend

__all__ = [
    'geometry', 'meshgen', 'PointCloud', 'Precision', 'PrecisionData', 'LocalSolver',
    'IMQ', 'Gaussian', 'PHS', 'Hybrid', 'DivergenceFree', 'Kernel', 'Wendland',
    'Identity', 'Derivative', 'Laplacian', 'NormalDerivative', 'Robin', 'SpatialOperator',
    'LinearPDE', 'BoundaryCondition', 'Dirichlet', 'EvolutionPDE',
    'SymbolicScalar', 'SymbolicSystem', 'SymbolicStokes',
    'ScalarSpace', 'DivergenceFreeSpace', 'PressureSpace', 'StencilPolicy', 'Samples', 'LocalApproximation', 'CenterGroup',
    'GlobalCollocation', 'LHI', 'RBFFD', 'interpolate', 'DiscreteOperator', 'OperatorSet',
    'PythonBackend', 'CppBackend', 'TorchBackend',
]
