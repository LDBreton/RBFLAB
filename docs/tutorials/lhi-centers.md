# Choose independent functional centers

An LHI patch combines solution values and PDE/boundary data. These functionals
need not share one point cloud. For $Lu=f$ with Dirichlet data:

$$Lu(X_u)\approx S_uU+S_bg(X_b)+S_ff(X_f),$$

$$S_uU=f(X_u)-S_bg(X_b)-S_ff(X_f).$$

Only $U$ is unknown. Additional PDE samples increase the local functional system
without adding global unknowns. Read [the LHI construction](lhi.md) first.

![Independent solution, boundary and PDE center clouds](../assets/figures/lhi-centers.png)

## Declare the groups, then assign their numerical meaning

The complete example below uses `Samples` and `LocalApproximation` to reproduce
$u=1+x^2+y^2$. Five independent PDE locations supply $Lu=-4$.
The source dictionary defines available data; the right-hand side determines
which data are prescribed.

```python
--8<-- "examples/tutorials/lhi_centers.py"
```

`ops.L["u"].matrix` has four columns, one per interior unknown. The boundary
and forcing blocks have their own column counts and are multiplied by arrays
on their own sample sets. Source groups never need a shared global point index.

## Choose memberships deliberately

- `size=n`: select $n$ eligible nearest points in that group.
- `size=None` (the default): use all eligible points.
- `indices=rows`: exact group-local indices for each target, in the given order.
  Do not also specify a size or selection policy.
- `target="exclude"`: remove exact coordinate matches.
- `target="require"`: require the target among the selected points; useful at
  solution nodes but inappropriate for off-node reconstruction.
- `policy=StencilPolicy(selection="quality", ...)`: choose the group's selection
  rule. Kernel scaling belongs to `LocalApproximation.stencil_policy`.

Coincident locations with different functionals are valid; identical duplicated
functionals are rejected. The polynomial functionals must have full column rank.
A larger nominal stencil does not by itself guarantee independent information.

## Add derivative or boundary observations

For measured $u_x$ values at $X_d$, add
`"dx_data": rbf.Samples(Xd, operator=rbf.Derivative(0), size=4)` before constructing
the trial. The resulting map has a corresponding `"dx_data"` block, which you
multiply by the observed derivative values. The field being interpolated is
still the same scalar function.

For boundary normals, pass `normals=...` with `NormalDerivative` or a
normal-dependent Robin operator. The arrays must follow the group's point
ordering. A `Samples` object has no hidden PDE role or automatic forcing.

## Inspect and reuse

`ops.L.local(i).groups` gives the selected indices in each named group.
`ops.L.reconstruct_local(i)` reconstructs the local augmented equation.
New data on unchanged samples reuse the same maps. New query locations need
new target maps, as the example shows.

The functional engine supports Python, C++ Float64/MPFR and Torch Float64 for
its documented scalar and componentwise divergence-free paths. Native matrices
retain their backend storage; `to_scipy()` explicitly converts to Float64.

## Independent centers in an evolution problem

For $u_t+Lu=f$, PDE samples are $f(X_f)-u_t(X_f)$. If $X_f=X_u$ in the same order,

$$(I-S_f)\dot U+S_uU=f_u-S_ff_f-S_bg.$$

If the point sets differ, you must provide the map from unknown time derivatives
to $X_f$, including derivatives of prescribed terms when appropriate. A stationary
forcing array cannot supply those unknown time derivatives. See
[LHI and heat matrices](lhi-matrices.md) for the precise construction.

Run `python -m examples.tutorials.lhi_centers` from the source checkout.
