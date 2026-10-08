"""RBFLAB: radial basis functions for interpolation and PDEs."""
from .precision import Precision, PrecisionData
from .kernels import IMQ, Gaussian, PHS, Hybrid, DivergenceFree
from .operators import Identity, Derivative, Laplacian, NormalDerivative, Robin
from .geometry import PointCloud, gmsh_square, gmsh_cube, unit_box_grid
from .problems import LinearPDE, BoundaryCondition, Dirichlet
from .nodal import rbf_fd_weights
from .rbf_fd import RBFFD
from .methods import GlobalCollocation, LHI, interpolate

__all__ = [
    "from_rbfmeshgen", "nodal_diagnostics", "TimeData", "UnsteadyStokesProblem", "LHIUnsteadyStokes", "GlobalUnsteadyStokes", "StokesProblem", "GlobalStokes", "StencilPolicy", "Precision", "PrecisionData", "IMQ", "Gaussian", "PHS", "Hybrid", "DivergenceFree", "Identity", "Derivative", "Laplacian",
    "NormalDerivative", "Robin", "PointCloud", "gmsh_square", "gmsh_cube", "unit_box_grid", "LinearPDE",
    "BoundaryCondition", "Dirichlet", "GlobalCollocation", "LHI", "RBFFD", "rbf_fd_weights", "interpolate",
]


from .stencils import StencilPolicy

from .mesh_adapters import from_rbfmeshgen
from .diagnostics import nodal_diagnostics

from .stokes import StokesProblem, GlobalStokes

from .unsteady_stokes import TimeData, UnsteadyStokesProblem, GlobalUnsteadyStokes

from .lhi_stokes import LHIUnsteadyStokes

from .strategies import growing_hybrid_stokes, growing_stencil_size
__all__ += ["growing_hybrid_stokes", "growing_stencil_size"]

from .evolution import EvolutionPDE, EvolutionSystem, EvolutionTrajectory
__all__ += ["EvolutionPDE", "EvolutionSystem", "EvolutionTrajectory"]

from .symbolic import SymbolicScalar
__all__ += ["SymbolicScalar"]

from .operators import SpatialOperator
__all__ += ["SpatialOperator"]

from .time_data import InitialData
__all__ += ["InitialData"]

from .symbolic_system import SymbolicSystem, BlockPDE
from .block_methods import BlockGlobal, BlockLHI
__all__ += ["SymbolicSystem", "BlockPDE", "BlockGlobal", "BlockLHI"]

from .spaces import ScalarSpace, DivergenceFreeSpace, PressureSpace
__all__ += ["ScalarSpace", "DivergenceFreeSpace", "PressureSpace"]

from .legacy_cpp import LegacyCppLHIBackend
__all__ += ["LegacyCppLHIBackend"]

from .lhi_backends import PythonBackend, CppBackend
__all__ += ['PythonBackend', 'CppBackend']

from .symbolic_kernel import Kernel, Wendland
__all__ += ["Kernel", "Wendland"]

from .cuda_backend import CudaLHIBackend
__all__ += ["CudaLHIBackend"]

from .torch_backend import TorchBackend, TorchKernel
__all__ += ['TorchBackend', 'TorchKernel']

from .differentiable_lhi import DifferentiableLHI
__all__ += ['DifferentiableLHI']

from .discrete_operators import DiscreteOperator, OperatorSet
__all__ += ["DiscreteOperator", "OperatorSet"]
