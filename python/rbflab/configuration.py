"""Backend-independent local linear algebra choices."""
from dataclasses import dataclass

@dataclass(frozen=True)
class LocalSolver:
    """Local factorization. SVD is currently supported by C++ Float64 Stokes LHI.

    rcond is a relative singular-value cutoff; it changes the numerical method.
    """
    method: str = "lu"
    rcond: float | None = None

    def __post_init__(self):
        import math
        if self.method not in ("lu", "svd"):
            raise ValueError("LocalSolver.method must be lu or svd")
        if self.rcond is not None and (not math.isfinite(self.rcond) or not 0 <= self.rcond < 1):
            raise ValueError("rcond must be finite in [0, 1)")
        if self.rcond is not None and self.method != "svd":
            raise ValueError("rcond requires SVD")

def configured_backend(method):
    """Private execution adapter; numerical choices remain on the method."""
    import copy
    from .lhi_backends import PythonBackend
    backend = copy.deepcopy(method.local_backend or PythonBackend())
    object.__setattr__(backend, '_shape_rule', method.stencil_policy.shape_rule)
    object.__setattr__(backend, '_solver', method.local_solver)
    return backend

class BackendSettings:
    @property
    def shape_rule(self): return getattr(self, '_shape_rule', 'fixed')
    @property
    def local_solver(self): return getattr(self, '_solver', LocalSolver()).method
    @property
    def svd_rcond(self): return getattr(self, '_solver', LocalSolver()).rcond
