# Functional local approximation architecture

## Decision

The canonical local research API describes a space, trial functions, source
functionals and targets. It returns named sparse maps. Users assemble equations,
boundary elimination, constraints and time integration with these maps.

`LocalApproximation`, `Samples`, and space trial descriptors share local kernel,
polynomial, factorization and reconstruction machinery. Ordinary RBF-FD is the
value-sample construction. LHI is the Hermite value/PDE/boundary construction;
PDE data become forcing or derivative contributions only during equation assembly.

Global collocation retains its coefficient-based representation. Standard
`RBFFD.operators` and `.weights` now delegate to the shared engine; their adapter
preserves the historical SciPy matrix interface even when Torch constructs the
weights. Direct `LocalApproximation` retains native backend matrix storage.

Scalar `LHI.assemble` also delegates weight construction to this engine through
`_compat_lhi.py`. Its adapter preserves historical known-data assembly and
solution/reconstruction semantics. Specialized coupled-block and divergence-free
Stokes convenience paths remain separate compatibility implementations. General
mixed-field symbolic assembly has not been added by this change.

This separates completed numerical sharing from future migration: existing
formulations are not silently deleted or replaced by a different mathematical
rule. In particular, componentwise functionals on a vector kernel do not by
themselves provide a mixed velocity-pressure equation assembler.

## Invariants

- Source group order determines concatenated matrix column order.
- Point/value/operator meanings stay inspectable through named blocks.
- Trial and data functionals are explicit; differentiated trials alone do not
  imply symmetry.
- PDE-target exclusion is an explicit neighborhood choice.
- Independent transient PDE centers need an explicit unknown-data map.
- Backends execute supported algebra; they do not select the PDE formulation.
- Local precision and global arithmetic are separate choices.
- Compiled expression reuse does not imply numerical factorization reuse.
- Multiple target operators should reuse the same local factorization.

## Extension rule

Extend kernel/operator evaluation, trial/data descriptions, neighborhood
selection, or backend execution at the shared layer. Add examples for equations
and time schemes before adding equation-specific public solver classes. Preserve
optimized kernels only when they agree with the reference mathematical request.

The tutorials `local-approximation.md` and `lhi-matrices.md` are the executable
teaching contract. Capability claims must match tested combinations; unsupported
spaces and precision choices must fail explicitly rather than change method.
