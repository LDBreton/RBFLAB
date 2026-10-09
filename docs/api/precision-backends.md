# Precision and backends

Extended precision can be selected for local weight construction, global dense solves, or the supported sparse LHI path. The optional backends have method-specific limits; see their docstrings and [capabilities](../CAPABILITIES.md).

::: rbflab.Precision

::: rbflab.PrecisionData

For a worked explanation of `scaling="local"`, its default, and its effect on
shape parameters, see [local stencil scaling](../theory/conditioning.md#local-stencil-scaling).

::: rbflab.StencilPolicy

::: rbflab.PythonBackend

::: rbflab.CppBackend

::: rbflab.TorchBackend

::: rbflab.TorchKernel

::: rbflab.CudaLHIBackend

::: rbflab.DifferentiableLHI

::: rbflab.LegacyCppLHIBackend
