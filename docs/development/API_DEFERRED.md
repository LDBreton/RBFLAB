# Deferred research/API work

- General mixed vector symbolic compiler beyond constant-viscosity Stokes.
- Independent LHI functional clouds for transient and coupled systems: require explicit unknown-dependent data/mass maps. Reject rather than treating f-u_t as prescribed.
- A public arbitrary trial/data pairing language beyond supported Hermite construction.
- Additional backend combinations, anisotropic and arbitrary matrix-valued kernel protocols.
- Globally reconciled LHI pressure and smooth reconstruction where not already implemented.

These are not advertised as supported capabilities.

## Explicit maintenance boundaries

- Scalar LHI accepts PythonBackend with one worker and condition estimation;
  worker/diagnostic-control support is not silently implied by the backend name.
- Stokes LHI matrices have dependent row maps: edits are rejected, while exported
  matrices remain usable with external solvers.
- The retained mesh object-list importer is internal historical validation code.
- Experiment implementations remain in their modules and are exposed lazily only
  through rbflab.experimental. Their native helpers are still used internally.
- Kernel extensibility remains the validated symbolic radial expression route;
  a duck-typed arbitrary matrix-valued evaluator protocol is deferred.
- Preflight validates configured support; local numerical rank still requires
  constructing a stencil. No preflight can certify stability or conditioning.
