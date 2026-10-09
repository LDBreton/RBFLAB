# Implementation checklist

- [x] 1. Inventory and decisions
- [x] 2. Canonical methods and explicit local weights
- [x] 3. Stokes/block entry point consolidation
- [x] 4. Explicit capabilities and preflight
- [x] 5. Numerical/execution configuration separation
- [x] 6. Public namespace cleanup
- [x] 7. Shared local functional construction
- [x] 8. System metadata, mutation and preparation
- [x] 9. Effective numerical recipes
- [x] 10. Examples, documentation and validation
- [x] 11. Explicit LHI functional center groups

No release or publication is part of this task.

## Verified outcome (2026-10-09)

- Full maintained suite: 206 passed, 20 optional native skips, 42 subtests passed.
- Focused consolidation suite after additional block-route checks: 17 passed.
- Operator suite with RBFLAB_CPP_TESTS=1: 40 passed (Python/PyTorch/C++ double/MPFR, 2D/3D).
- Symbolic-kernel suite with native tests enabled: 10 passed.
- Strict MkDocs build passed.
- Local review wheel built and installed into an isolated target. From outside
  the checkout, explicit RBF-FD weights and the independent-center LHI example
  passed (polynomial error 8.88e-16). Not a published release.
- C++ steady Stokes: velocity error 2.72e-16 (Float64) and 4.61e-15
  (35-digit local weights / Float64 global system) on the polynomial example.
- C++ SVD through LocalSolver: velocity error 3.58e-13.
- Relocated Hardy rule: relative Python/C++ sparse-matrix difference 4.05e-15.
- Explicit/default scalar LHI groups produce matching matrices. Independent PDE
  and derivative-data groups reproduce polynomials in Float64 and full extended
  precision; separate tests cover Neumann data and 3D groups.

Scope boundaries are in API_DEFERRED.md. In particular, independent groups for
transient/coupled LHI and a universal public trial/data DSL were not implemented
or advertised. The existing default transient and Stokes routes remain available.

The first full run found a test still importing the removed root mesh adapter;
it was migrated to the retained internal adapter and the final full run passed.
