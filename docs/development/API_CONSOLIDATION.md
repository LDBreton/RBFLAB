# API consolidation decision record

Goal: a small mathematical API with explicit capabilities and reusable local algebra.
Breaking changes are intentional; no compatibility aliases are required.

## Decisions

- Keep interpolate, GlobalCollocation, RBFFD, LHI; retain problem.solve convenience.
- RBFFD.weights owns explicit single-stencil work; remove root rbf_fd_weights.
- Keep SymbolicScalar and SymbolicSystem for scalar equations; expose SymbolicStokes explicitly for the specialized vector compiler.
- Keep spaces, operators, precision and three supported backends.
- Specialized Stokes and block assemblers remain internal implementation routes.
- Backend-independent stencil policy owns shape rules. Solver choice is a separate LocalSolver configuration.
- LHI gains named functional CenterGroup objects, including separate PDE clouds, explicit membership and target rules. Initial independent-group support is scalar stationary Python; other paths reject early.
- Default scalar LHI and explicit groups share weight computation and assembly.
- Mutable scalar matrix access invalidates cached factors before reuse. Stokes LHI rejects edits because its row maps also require reassembly.
- prepare returns a new method and leaves the original unchanged.
- Operators expose both local weights and reconstruction; dense matrices are rebuilt on demand.

## Public symbol decisions

| Symbol | Decision | Replacement / reason |
|---|---|---|
| from_rbfmeshgen | Remove | meshgen.generate |
| nodal_diagnostics | Move to submodule | diagnostics |
| TimeData | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| UnsteadyStokesProblem | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| LHIUnsteadyStokes | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| GlobalUnsteadyStokes | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| StokesProblem | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| GlobalStokes | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| StencilPolicy | Keep public | Canonical mathematical building block |
| Precision | Keep public | Canonical mathematical building block |
| PrecisionData | Keep public | Canonical mathematical building block |
| IMQ | Keep public | Canonical mathematical building block |
| Gaussian | Keep public | Canonical mathematical building block |
| PHS | Keep public | Canonical mathematical building block |
| Hybrid | Keep public | Canonical mathematical building block |
| DivergenceFree | Keep public | Canonical mathematical building block |
| Identity | Keep public | Canonical mathematical building block |
| Derivative | Keep public | Canonical mathematical building block |
| Laplacian | Keep public | Canonical mathematical building block |
| NormalDerivative | Keep public | Canonical mathematical building block |
| Robin | Keep public | Canonical mathematical building block |
| PointCloud | Keep public | Canonical mathematical building block |
| gmsh_square | Move to submodule | geometry |
| gmsh_cube | Move to submodule | geometry |
| unit_box_grid | Move to submodule | geometry |
| LinearPDE | Keep public | Canonical mathematical building block |
| BoundaryCondition | Keep public | Canonical mathematical building block |
| Dirichlet | Keep public | Canonical mathematical building block |
| GlobalCollocation | Keep public | Canonical mathematical building block |
| LHI | Keep public | Canonical mathematical building block |
| RBFFD | Keep public | Canonical mathematical building block |
| rbf_fd_weights | Consolidate | RBFFD.weights |
| interpolate | Keep public | Canonical mathematical building block |
| growing_hybrid_stokes | Move to submodule | Explicit experimental namespace; no stable root export |
| growing_stencil_size | Move to submodule | Explicit experimental namespace; no stable root export |
| EvolutionPDE | Keep public | Canonical mathematical building block |
| EvolutionSystem | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| EvolutionTrajectory | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| SymbolicScalar | Keep public | Canonical mathematical building block |
| SpatialOperator | Keep public | Canonical mathematical building block |
| InitialData | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| SymbolicSystem | Keep public | Canonical mathematical building block |
| BlockPDE | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| BlockGlobal | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| BlockLHI | Make internal | Use canonical methods with explicit problems/spaces; retain numerical implementation |
| ScalarSpace | Keep public | Canonical mathematical building block |
| DivergenceFreeSpace | Keep public | Canonical mathematical building block |
| PressureSpace | Keep public | Canonical mathematical building block |
| LegacyCppLHIBackend | Move to submodule | Explicit experimental namespace; no stable root export |
| PythonBackend | Keep public | Canonical mathematical building block |
| CppBackend | Keep public | Canonical mathematical building block |
| Kernel | Keep public | Canonical mathematical building block |
| Wendland | Keep public | Canonical mathematical building block |
| CudaLHIBackend | Move to submodule | Explicit experimental namespace; no stable root export |
| TorchBackend | Keep public | Canonical mathematical building block |
| TorchKernel | Move to submodule | Explicit experimental namespace; no stable root export |
| DifferentiableLHI | Move to submodule | Explicit experimental namespace; no stable root export |
| DiscreteOperator | Keep public | Canonical mathematical building block |
| OperatorSet | Keep public | Canonical mathematical building block |

## New canonical controls

| Symbol | Purpose |
|---|---|
| SymbolicStokes | Explicit constant-viscosity vector momentum compiler |
| CenterGroup | Named solution/PDE/boundary/observation functional centers |
| LocalSolver | Method-level LU/SVD policy with explicit backend support |
| method.preflight | Shared configuration support checks before assembly |
| system.recipe / dof_map / reconstruction | Effective setup and representation metadata |

The core generic functional matrix implementation remains shared internal code;
a general public trial/data DSL is deferred. Independent groups are scalar
stationary Python in this iteration, including 3D and extended arithmetic.
