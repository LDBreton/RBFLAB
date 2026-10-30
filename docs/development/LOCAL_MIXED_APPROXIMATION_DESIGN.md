# Mixed local Hermite approximation design

Design record, reviewed with Sol medium and Astra high on 2026-10-28. The
implementation update below describes delivered functionality. The subsequent
original staged proposal records the design audit and includes deferred
interfaces; use the current workflow and capability pages for supported calls.
Neither the design nor successful algebraic parity establishes stability.

Implementation update: the first generic milestone is now implemented with
`MixedSpace`, field-block `FieldOperator`, matching mixed representers, Python
and native C++ Float64/MPFR weights, and real MPI scheduling through
`LocalExecution`. Direct and mixed quotient-pressure behavior is corrected.
Existing Python/C++ LU Stokes LHI weight construction now uses this common
engine, with historical polynomial coordinates and compact equation assembly
preserved. SVD/experimental paths remain explicit specialized exceptions. See the
[current workflow](../guides/research-api.md#mixed-field-hermite-maps) and
[capability limits](../CAPABILITIES.md#generic-mixed-local-approximation).
The design below retains future proposals: `StencilSet`, explicit custom
polynomial bases, mixed translates, translated-task reuse and migration of
scalar block convenience assembly are not implemented. Actual
request reuse is `reuse="exact"`, not the proposed `"translated_exact"`.

## Original staged proposal

Extend `LocalApproximation` to approximate named scalar and vector fields together, using linear functionals that can combine those fields. Preserve the existing separation between spaces, sampled quantities, trial construction, target operators, numerical execution, and user-owned equations. The motivating case is mixed divergence-free velocity/pressure compact LHI, but the extension should not encode a particular pressure closure, PDE, boundary treatment, or time integrator.

## Capabilities and gaps at the original design audit

The public foundation already exists. `Samples` describes observation points and operators; `space.representers(source)` describes kernel representers; `LocalApproximation.operators` returns target maps with named source blocks. `.local()` exposes memberships and weights; `.reconstruct_local()` exposes the physical augmented matrix and right-hand side. Local maps do not classify data as known or unknown.

| Requirement | Current capability | Required extension |
| --- | --- | --- |
| Scalar Hermite value/derivative/PDE samples | Supported | Reuse |
| Componentwise derivatives of one DF vector field | Supported in 2D/3D | Reuse |
| Mixed DF velocity and scalar pressure kernels | Not in common local engine | Named product space |
| Momentum observations combining velocity and pressure | Not supported by `Samples.operator` | Field-aware linear functional |
| Velocity/momentum groups with two values per point and pressure groups with one | One common component count is assumed | Per-group input arity and per-target output arity |
| Exact group memberships, target exclusion and normals | Supported | Reuse and expose physical-site derivation |
| Select 37 physical sites once, then derive observation groups | Possible through manually prepared `Samples.indices` | Reusable indexed stencil description |
| Independent polynomial degrees per field | Specialized Stokes path only | Direct-sum polynomial construction |
| Exact historical polynomial basis and normalization | Not configurable in common local engine | Explicit basis/normalization descriptor |
| Multiple targets sharing one factorization | Supported | Retain with varying target arities |
| MP local weights and optional MP sparse storage | Supported combinations exist | Extend coverage to mixed requests |
| Native local construction | Scalar/homogeneous DF functional paths | Mixed kernel/functionals job protocol |
| MPI and exact translated-task reuse | Research drivers only | Execution policy over the same jobs |
| Shared global pressure, gauge and compact time equations | User equations/specialized research code | Examples and explicit layouts; no automatic closure |

The principal source restrictions are in `local_approximation.py`: a trial has one `ScalarSpace`-derived descriptor; `operators()` selects one `components` count; every source block and every target uses that count. `samples.py` binds scalar differential operators. `functional_backends.py` and `cpp/functional_local.hpp` inherit the same single-kernel, homogeneous-component assumptions. Separate velocity and pressure approximations cannot replace a jointly conditioned Hermite solve.

`CenterGroup` belongs to the packaged LHI compatibility path and assigns roles such as solution/PDE/boundary/data. Independent groups there remain scalar stationary Python functionality. Extend `Samples` and the common engine instead of introducing another role-based assembly path. Some earlier API decision records describe functionality as deferred that the common engine now supports; current implementation and capability documentation take precedence over those historical statements.

### Which implementations currently share LocalApproximation

| Public operation | Actual route | Uses common local weight engine? |
| --- | --- | --- |
| Scalar stationary `LHI.assemble` | `methods.LHI` -> `_compat_lhi.assemble_scalar_lhi` -> `LocalApproximation.operators` | Yes |
| Supported scalar transient `LHI.assemble` | `evolution.assemble_evolution` -> stationary template above, followed by compact mass/data assembly | Yes, for local weights |
| DF Stokes stationary/transient `LHI(spaces=...)` | `space_stokes.LocalFlowSystem` -> `LHIUnsteadyStokes` -> specialized Python/native backend | No |
| Coupled scalar `LHI(spaces=...)` | `space_stokes.assemble_spaces` -> `BlockLHI` -> `BlockBasis` | No |
| Direct DF `LocalApproximation` | Common functional engine with componentwise operators on one DF field | Yes, but not coupled velocity-pressure |

A runtime spy on a 16-node unit-square cloud observed one `LocalApproximation.operators` call for scalar stationary assembly, one for scalar transient assembly, and zero for each corresponding DF Stokes assembly. Both Stokes assemblies returned `LocalFlowSystem` wrapping `LHIStokesSystem`. This checks routing only, not numerical accuracy or stability. Existing Stokes LHI exposes velocity unknowns and local pressure gradients, not the research study's shared global scalar pressure.

Therefore this is a generalization of the existing local engine followed by consolidation of specialized weight paths. It is not merely exposing an existing mixed `LocalApproximation` implementation through a prettier signature.

## Preserve the exact mathematical construction

For mixed fields `u` and `p`, use the block kernel

\[
K=\operatorname{diag}(K_u,K_p),\qquad
K_u=(\nabla\nabla^T-I\Delta)\phi_u.
\]

A scalar functional row is a finite sum over field components,

\[
\lambda z=\sum_{f,c}D_{f,c}z_{f,c}.
\]

A vector-valued functional is an ordered list of these rows. For example, 2D momentum has rows `u[0]: -nu*Lap, p[0]: Dx` and `u[1]: -nu*Lap, p[0]: Dy`. Its representer applies the whole row to the source argument of the block kernel. Coupling must occur before factorization, not by concatenating independently computed velocity/pressure weights.

For source rows `lambda`, trial rows `mu`, and polynomial columns `pi`, form

\[
A=\begin{pmatrix}G&P_s\\P_t^T&0\end{pmatrix},\quad
G_{ab}=\lambda_a^x\mu_b^yK(x,y),\quad
(P_s)_{a\ell}=\lambda_a\pi_\ell,\quad
(P_t)_{b\ell}=\mu_b\pi_\ell.
\]

Weights for a target functional solve `A.T @ augmented_weights = Q`. The upper RHS block is `Q[b,k] = tau_k^x mu_b^y K(x,y)` evaluated at the target and trial sites; the lower block is `Q[n+ell,k] = tau_k pi_ell`. The RHS uses trial representers, including when source and trial differ. Matching Hermite observations/trials give a symmetric matrix; arbitrary supported source/trial pairings need not. Retain the transpose solve as the general contract.

The exact research configuration to reproduce is:

- 37 physical sites selected once per owner; all velocity observations, interior momentum neighbors excluding the owner, and noncorner pressure observations including the owner.
- Scalar observation count `74 + 2*n_momentum_sites + n_pressure_sites`; 183 for a fully interior stencil, with a 221 by 221 augmented system.
- Velocity degree 6 with 35 solenoidal columns; pressure degree 1 with three visible columns, historically ordered `y,x,1`.
- Owner-centered polynomial coordinates normalized by maximum Euclidean stencil radius. This differs from the common engine's current first-source-point origin and coordinatewise maximum polynomial scale.
- Physical kernels `phi_u=exp(-10*q*q*r*r)+q**7*r**7` and `phi_p=nu**2*q**4*exp(-q*q*r*r)`, with `q=(1/44)/h`. For the saved uniform unit-square grids, `h=1/cells`, hence `q=cells/44`; do not substitute fill distance or stencil radius. For cavity, `nu=0.01`.
- Targets `Lx,Ly,P,Px,Py,LapP`; a vector momentum target plus scalar pressure, vector pressure gradient and scalar pressure Laplacian give the same six scalar RHS.
- MP80 local arithmetic; Float64 geometry and global matrices in the recorded production runs. MP sparse storage remains a separate option.

## Proposed public objects

### MixedSpace

Add `MixedSpace({name: space, ...})` with an ordered field layout. Initially support a direct sum of existing scalar and divergence-free field spaces, including independent kernels and polynomial degrees. Implement matching `.representers(source)` first. Preserve existing single-field `.translates(...)` and asymmetric trial support, but reject mixed `.translates(...)` until per-field/component trial multiplicities are designed and tested. A mixed field with two velocity components and one scalar pressure does not have one translate per physical site by default.

Use `ScalarSpace(Kp, polynomial_degree=1)` for a pressure field with a visible constant. `PressureSpace` is documented as pressure modulo constants, but that contract is not implemented uniformly: the direct common-engine route forces degree 0 when degree is `None`, and its scalar polynomial builder retains the constant for degree 0 or 1. A runtime probe observed polynomial column counts 1, 1 and 3 for declared degrees `None`, 0 and 1, respectively. Specialized Stokes polynomial construction omits the constant. Do not rely on this inconsistency or silently propagate it into the mixed API. Reject quotient `PressureSpace` as a mixed factor initially until its admissible functionals/reference representation are reconciled. A future explicit quotient mode must reject absolute pressure observations without a defined reference. Global pressure gauge equations remain outside `MixedSpace`.

### FieldOperator

Add one explicit field-functional descriptor rather than a new symbolic PDE language. After independent Sol medium/Astra high review, the preferred public constructor maps field names to rectangular matrices of existing scalar differential operators. This displays the block operator directly. For example:

```python
# PROPOSED API; not executable with the current release.
I = r.Identity(2)
Dx, Dy, Lap = r.Derivative(0, 2), r.Derivative(1, 2), r.Laplacian(2)

U = r.FieldOperator({"u": [[I, 0], [0, I]]})
P = r.FieldOperator({"p": [[I]]})
L = r.FieldOperator({
    "u": [[-nu * Lap, 0], [0, -nu * Lap]],
    "p": [[Dx], [Dy]],
})
GradP = r.FieldOperator({"p": [[Dx], [Dy]]})
LapP = r.FieldOperator({"p": [[Lap]]})
```

Each block has shape `(output_arity, field_components)`. All blocks share the same positive row count; omitted fields contribute zero. Literal `0` is dimension-neutral; other bare numbers are rejected initially, so use `Identity` explicitly. Field component counts and spatial dimensions are checked when bound to the mixed space and geometry. Normalize this constructor into sparse `(field, component, derivative, coefficient)` scalar rows internally; do not expose two equivalent public row/block constructors in the first release.

`Samples.operator` and target `operators` accept this descriptor. Each group has a fixed output arity; operators may vary by point if arity remains fixed. Split groups when arity differs. Reject zero/dependent duplicate source or trial rows at the same point as appropriate; repeated operator patterns at different points are ordinary samples. Zero and duplicate target rows are valid map requests. Same-point pressure and velocity observations are distinct and valid. Duplicate screening is not a general linear-independence certificate; retain local rank/factorization diagnostics.

This is generic operator-block algebra, not a momentum API. For two scalar fields `a` and `b`, `FieldOperator({"a": [[Lap - I]], "b": [[2 * I]]})` represents `(Delta-I)a + 2b` using the same construction. No pressure, divergence or fluid role is attached to a field name.

Mixed requests must bind fields explicitly. Reject `Samples(operator=None)` or a bare scalar differential operator on `MixedSpace` initially rather than guessing which fields it applies to. Preserve existing single-field implicit-identity and componentwise operator semantics. A field-value convenience constructor can follow, but a dimension-free `space.value("u")` requires a new binding rule because existing DF descriptors do not store spatial dimension. Explicit identity blocks avoid that machinery in the initial API.

Do not reinterpret existing scalar `Operator.@` composition. Expression shortcuts such as `gradient(p)` can follow after the block form is established; they are not needed to implement the core. Reuse `SymbolicSystem`/`SymbolicStokes` compilation concepts internally where applicable rather than adding an unrelated symbolic equation language, but mixed field functionals do not require a compiled PDE problem.

### StencilSet

As a later convenience, add a reusable geometry-only description of physical-site memberships. It is not required to validate the mixed core: existing `Samples.indices` already suffices. `StencilSet(points, targets=..., size=37, policy=...)` selects once; an alternative `indices=saved_rows` imports exact parent-node memberships without reselection. `size` and explicit `indices` are mutually exclusive. Own copied Float64 geometry, stable parent IDs and ordered target IDs. The target need not coincide with a source point.

`stencils.samples(nodes=parent_ids, operator=..., target=...)` intersects each selected physical stencil with eligible parent IDs, removes excluded targets, translates to group-local indices, and returns ordinary `Samples`. The helper performs filtering before constructing `Samples.indices`; current `Samples` only validates explicit rows and does not remove excluded IDs for the caller. Omit `nodes` to use all parent sites. Filtering does not replace excluded nodes with farther neighbors. Empty intersections are allowed when the combined local system is well posed.

Derived samples retain the stencil target coordinates/IDs and their ordering. `LocalApproximation.operators(targets=...)` must validate that they match exactly; changed or reordered targets require explicit rebinding/reselection. Equal target count alone is insufficient. Legacy manually supplied `Samples.indices` retains its positional contract; only stencil-derived groups add this stronger provenance guarantee.

Existing `Samples.target="exclude"` means exclusion of all geometrically coincident points, not just a matching owner ID. The helper retains that meaning; an ID-only filter, if needed for coincident distinct entities, must have a different explicit name. IDs preserve provenance and avoid accidental node merging, but do not silently change exclusion semantics.

This helper does not impose PDE roles, boundary conditions or unknowns. `Samples(indices=...)` stays the low-level route for independent clouds or externally defined layouts. Do not merge distinct nodes just because their coordinates match. For unstructured nodes, require explicit memberships/subset IDs or selection policy; do not depend on grid indexing.

## Proposed user workflow

The following is the proposed complete workflow, including the later optional stencil/basis conveniences, not a current runnable example or the first implementation milestone. `Ku` and `Kp` are existing supported kernel objects bound to the physical parameters above; `interior_ids`, `boundary_ids` and `pressure_ids` belong to the same parent cloud. Pressure IDs omit corners. Owner points contain interior and open-wall pressure centers.

```python
space = r.MixedSpace({
    "u": r.DivergenceFreeSpace(Ku, polynomial_degree=6),
    "p": r.ScalarSpace(Kp, polynomial_degree=1),
})

# Import exact memberships for parity; use size=37 for a new cloud.
stencils = r.StencilSet(cloud.points, targets=owners, indices=saved_rows)
source = {
    "velocity": stencils.samples(nodes=interior_ids, operator=U),
    "boundary": stencils.samples(nodes=boundary_ids, operator=U),
    "momentum": stencils.samples(nodes=interior_ids, operator=L,
                                 target="exclude"),
    "pressure": stencils.samples(nodes=pressure_ids, operator=P),
}

local = r.LocalApproximation(
    source=source,
    trial=space.representers(source),
    backend=r.PythonBackend(),
    precision=r.Precision(local_digits=80, global_dtype="float64"),
    stencil_policy=r.StencilPolicy(scaling="physical"),
    polynomial_basis=study_basis,  # proposed explicit basis descriptor below
)
maps = local.operators(
    targets=owners,
    operators={"momentum": L, "pressure": P,
               "grad_pressure": GradP, "lap_pressure": LapP},
)

A  = maps.momentum["velocity"].matrix
SB = maps.momentum["boundary"].matrix
G  = maps.momentum["pressure"].matrix
SL = maps.momentum["momentum"].matrix
check = maps.momentum.reconstruct_local(0)
```

The smaller first milestone uses the same `MixedSpace`, `FieldOperator` and `LocalApproximation` syntax with ordinary `Samples` and already available exact group memberships:

```python
# PROPOSED mixed support; Samples/indices syntax already exists.
source = {
    "velocity": r.Samples(Xi, operator=U, indices=velocity_rows),
    "boundary": r.Samples(Xb, operator=U, indices=boundary_rows),
    "momentum": r.Samples(Xi, operator=L, indices=momentum_rows,
                          target="exclude"),
    "pressure": r.Samples(Xp, operator=P, indices=pressure_rows),
}
local = r.LocalApproximation(source=source, trial=space.representers(source))
maps = local.operators(targets=owners,
                       operators={"momentum": L, "pressure": P})
```

These rows must already implement the common physical stencil filtering. This initial route separates the new mathematical capability from the later neighborhood ergonomics.

In this example `maps.momentum` has two output rows per owner; its pressure block has one input column per global pressure site. A separate scalar-pressure target has one output row per owner and can still act on two-component velocity source groups. This requires fixing the existing `SampleMap` reshape assumption: source and target components are independent.

Owners include walls for pressure closure, so the example maps contain momentum evaluation rows on walls too. Global interior momentum matrices select interior owner rows. For matching interior owner/source ordering, construct `M = I - SL_interior`. In the general independent-cloud case use explicit restriction/embedding maps for the time derivative instead of assuming an identity correspondence. No implicit interpolation of time-dependent unknown data.

## Polynomial basis and scaling

Default mixed augmentation is the direct sum of the degree-generated field bases. For exact study parity, add an optional `polynomial_basis` descriptor to `LocalApproximation`, with ordered field-valued polynomial columns, center policy and scale policy. A normalized `PolynomialBasis` descriptor should support `origin="target"`, `scale="radius"` and exact symbolic polynomial columns in reference coordinates. Substitution and physical derivatives are applied using the configured arithmetic. Validate that the velocity columns are solenoidal when advertised as such. Check source and trial polynomial rank separately.

Default degree-based behavior remains unchanged. Different nonsingular polynomial bases spanning the same space should produce equivalent exact-arithmetic physical weights, but raw augmented matrices, scaling and finite-precision results need not match. Direct matrix parity therefore imports the historical ordered columns and normalization; it must not claim equality merely from matching degrees.

An optional later explicit basis replaces the generated columns, not the declared space contract. Its span must equal the field-degree augmentation declared by `MixedSpace` and respect the DF constraint. Validate these conditions algebraically from polynomial coefficients before the separate stencil rank check; a locally full-rank matrix is insufficient. Preserve `resolve_polynomial_degree` semantics: `None` disables augmentation, `"auto"` requests the kernel minimum, and an explicit below-minimum degree is retained with the existing warning, rather than silently upgraded or newly rejected. Equality with the declared span must not become a hidden stricter CPD policy. General enrichment beyond that span requires an explicit space capability. Record the effective ordered basis, transform and normalization in the recipe. Generic arbitrary polynomial customization is not a prerequisite for initial mixed weights; generated direct-sum bases and explicit owner/radius normalization cover the first validation milestone.

Keep three separate choices visible:

1. Physical kernel parameters, including pressure amplitude and PHS scaling.
2. Optional coordinate normalization in kernel evaluation, with every chain-rule derivative factor.
3. Algebraic equilibration before local factorization.

For these uniform-mesh studies, existing `Kernel`/`Hybrid` objects can be bound explicitly once per mesh using exact strings/rationals where appropriate. Use physical kernel coordinates to avoid applying a second unintended scale. A later per-stencil parameter binding rule should be serializable, inspectable and included in job fingerprints; it is not required for the first uniform-mesh reproduction. Avoid arbitrary Python callbacks as the only native parameter specification.

## Internal request and map layout

Normalize all supported requests to an internal field/block description:

- Ordered fields: name, spatial dimension, component count, kernel family and values, scalar or DF transformation.
- Scalar observation rows: parent/group point identity, component-operator terms and group offset.
- Scalar trial rows with independent membership and ordering.
- Named target functionals with explicit row slices and per-target point identity.
- Polynomial source/trial/target matrices and basis normalization metadata.
- Precision, physical kernel parameters and algebraic solver policy.

Generalize jobs from `n * owner.components` to explicit scalar row counts. Equality of source and trial sizes means equality of total scalar observations `sum(group_sites * group_arity)`, not equality of physical site counts. Kernel contraction sums `D_left * K_field * D_right` over components of each field. The existing Stokes `_blocks` implementation is a reference for signs and mixed contractions, not a second public backend. Keep existing scalar/DF shortcuts as adapters into the common request model, retaining their public behavior.

Maps retain `maps.target["source_group"]`, `.matrix`, `.to_scipy()`, `.local()` and `.reconstruct_local()`. Add explicit `input_layout` per source group and `output_layout` per target: arity, node-major order, labels, parent IDs and scalar row/column slices. A group of `n` velocity samples has `2*n` columns and a group of `n` pressure samples has `n` columns. Do not make users infer offsets from geometry or flattening conventions. Group values have shape `(n, arity)` or the documented flat shape, and output reshaping follows output arity.

Mixed map application uses one deterministic output convention, independent of the shapes supplied to individual source blocks. Proposed mixed convention: `(n_targets, output_arity)`, including `(n_targets, 1)` for scalar outputs; flattened algebra remains available through `.matrix`. Normalize every block result to this convention before summing a mapping of source values, avoiding accidental broadcasting between `(n,)` and `(n,1)`. Do not change the existing single-field output convention silently.

The saved research worker uses component-major scalar observation ordering within its vector groups. Production parity must import that order or apply explicit row, column and target permutations before comparing matrices and weights. Preserve public node-major semantics rather than silently adopting the research file layout.

Shared pressure is achieved by one pressure source group with one global indexed column layout. Each local pressure observation selects columns from that group. The approximation engine does not allocate a different global pressure unknown for every local reconstruction.

## Native and MPI execution

Extend the common native functional worker, not the experiment's integer operator-code protocol. Compile independent supported radial families per field and serialize the typed functionals and mixed polynomial blocks. Pressure identity, pressure gradient and pressure Laplacian must be ordinary target functionals, without Gaussian-only target codes.

Initial mixed support should be Python Float64/MP reference, followed by C++ Float64/MPFR. Torch mixed support remains explicitly unsupported until implemented and tested; existing Torch behavior is retained. Local factorization remains LU initially.

Add execution configuration independently from mathematical backend selection. Proposed spelling: `execution=LocalExecution(scheduler="mpi", ranks=8, reuse="translated_exact")` on `LocalApproximation`, with `CppBackend(threads=1)` selecting arithmetic implementation per worker. A default serial policy preserves existing calls; existing backend worker controls remain honored unless an explicit execution policy conflicts, in which case raise rather than silently oversubscribe.

All ranks consume the same normalized job schema. Use stable job IDs, bounded batches, deterministic permutation restoration, rank exit-code checks and failure records. Exact deduplication keys include relative coordinates, functional coefficients, trial construction, polynomial normalization/basis, target functionals, kernel values/scales, precision and solver controls. Absolute-dependent data invalidate translation equivalence unless resolved identically. Do not round coordinates to manufacture cache hits. Same kernel family permits compilation reuse, not numerical weight reuse after parameters change.

Preserve MP strings/values until requested storage conversion. Record computed unique jobs, requested jobs, LU count, cache hits, per-worker timings, failures, effective parameters, source/target layouts and exported dtype through the existing execution-report conventions. MPI local weights do not imply an MPI global solve.

## Equation assembly remains explicit

The public maps can reproduce the research momentum and pressure closure matrices, including `M=I-SL`. Interior pressure closure uses the pressure-Laplacian target; vertical/horizontal open walls use the corresponding normal momentum component. Corner singular fields and their derivatives supply known data outside the local approximation engine. Convection reconstruction can be a separate DF-only `LocalApproximation`.

The user chooses closure equations, time scheme, gauge, row replacement or compatibility border. Preserve original rows when replacing a gauge row so their residuals remain auditable. A compatibility multiplier that changes the PPE source must be reported as a physical source modification. No convenience API may silently introduce this multiplier or Q2/Q1, or claim that divergence-free local bases establish a globally divergence-free interpolant or stable evolution.

This design makes the tested method expressible; it does not repair the recorded cavity instability.

## Implementation sequence and acceptance criteria

1. **Typed mixed reference path.** Implement `MixedSpace`, field-block `FieldOperator`, varying source/target arities, direct-sum degree-generated polynomials and normalization controls in the common Python engine. Matching mixed representers only; no general mixed translates. Preserve scalar/DF tests and explicit degree/warning semantics. Accept exact `Samples.indices` first; no node helper, arbitrary basis language or MPI is needed to validate the algebra.
2. **Reproducible local nodes.** Implement `StencilSet` and derived `Samples`; compare memberships against saved interior, straight-wall, corner-adjacent and tied-distance stencils. Support external unstructured clouds and independent source/target sets. Demonstrate 37 sites without silently taking 37 from each group.
3. **Exact study parity.** Import representative saved 37-site requests. Compare Gram blocks, polynomial blocks, target RHS, all six weight columns and assembled named maps against the research worker. Compare degree-equivalent bases using physical weights, not raw matrix entries; use explicit historical basis for raw parity.
4. **Shared native request.** Add C++ mixed contraction/code generation, Float64 and MPFR tests, local matrix inspection and explicit storage export. No special PDE target IDs. Benchmark bounded representative workloads against the research path with recorded timings and matching precision.
5. **MPI scheduling and reuse.** Exercise one/multiple ranks, repeated/nonrepeated layouts, stable restoration, error propagation and cache invalidation. Compare numerical results before accepting performance claims.
6. **User-facing examples and capability promotion.** Add a compact mixed Stokes weight example, a shared-pressure `p=xy` static control, and explicit compact time-matrix assembly. Retain failing/positive-mode cases as diagnostic examples when relevant; do not publish a successful cavity claim from static accuracy.

7. **Consolidate existing packaged weight routes.** After parity, make scalar block and specialized Stokes assembly adapters consume common mixed local maps rather than maintaining separate kernel/weight solvers. Preserve their PDE/data/solution semantics outside the engine. Stokes migration requires explicit resolution of quotient-pressure augmentation and compatible reconstruction/precision diagnostics; do not switch it onto a visible constant with no pressure observations and create a singular local system. Pressure-quotient reconciliation is a dependency for that migration, not justification for silently changing the first shared-pressure example.

Meaningful checks include constant and affine pressure reproduction, the included-owner pressure delta row, DF velocity polynomial reproduction, coupled momentum with independently computed analytic data, asymmetric source/trial transpose solves, six-target multi-RHS factorization reuse, MP storage preservation, global pressure column sharing, physical derivative/amplitude scaling and both successful and rejected rank configurations. Run nodal manufactured field errors and finite-mode/time-evolution diagnostics separately from algebraic parity. No expensive fine-grid PDE rerun is necessary to approve this API design.

## Audit evidence

Public contracts and owning implementation:

- Root exports: `python/rbflab/__init__.py`; [capabilities](../CAPABILITIES.md), [research API guide](../guides/research-api.md).
- Local maps: `python/rbflab/local_approximation.py`; Samples/trials: `python/rbflab/samples.py`; spaces: `python/rbflab/spaces.py`.
- Functional backends: `python/rbflab/functional_backends.py`; native runtime: `cpp/functional_local.hpp`; scalar operators: `python/rbflab/operators.py`.
- Specialized contractions: `python/rbflab/stokes.py`; historical basis: `python/rbflab/stokes_polynomials.py`; compatibility centers: `python/rbflab/centers.py`.
- Focused tests: `tests/test_local_approximation.py`, `tests/test_discrete_operators.py`, `tests/test_functional_backends.py`.

Verification for this audit: `.venv/Scripts/python.exe -m pytest tests/test_local_approximation.py tests/test_discrete_operators.py -q`: **42 passed, 13 skipped**; `tests/test_functional_backends.py -q`: **15 passed, 8 skipped**. Total: **57 passed, 21 skipped**. Skips were opt-in native integration; no new native validation, MPI benchmark or cavity solve was run for this design. Three independent agent reviews inspected the API, exact study requirements and architectural fit. Runtime code was not changed.

Follow-up design discussion used Sol 6.1 at medium effort and Astra at high effort, including direct reviewer discussion. The revisions adopt their agreed field-block public syntax and smaller initial scope. The routing and PressureSpace probes above were executed during this follow-up. The earlier 57-test result is retained evidence from the preceding audit, not a claim that mixed support now exists or that these tests were rerun after a runtime change. Only this design document was edited.

The exact study is owned by local research files `research/divfree_lhi_energy/cavity_global_pressure_ppe_mpi.py`, `cavity_global_pressure_ppe_native.cpp`, `cavity_global_pressure_ppe_native_parity.py`, `shared_pressure_system.py` and `shared_pressure_bordered_system.py`. These are research evidence, not public capabilities. The existing six-target parity fixture has fewer physical sites than production; full 37-site interior/wall acceptance is still required for a public replacement.
