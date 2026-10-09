"""Research-only algorithms; no stable compatibility or backend guarantees.

These imports are lazy so ordinary RBFLAB use has no experimental dependency.
"""
import importlib
_NAMES = {
    'LegacyCppLHIBackend': 'legacy_cpp', 'CudaLHIBackend': 'cuda_backend',
    'DifferentiableLHI': 'differentiable_lhi', 'TorchKernel': 'torch_backend',
    'growing_hybrid_stokes': 'strategies', 'growing_stencil_size': 'strategies',
}
def __getattr__(name):
    if name not in _NAMES: raise AttributeError(name)
    return getattr(importlib.import_module('.'+_NAMES[name], __package__), name)
