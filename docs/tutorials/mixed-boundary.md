# Poisson equation with mixed boundary data

**You will learn:** to label boundary rows, define a Robin condition with its outward normal, and inspect the resulting global system. Prerequisites: the [interpolation](interpolation.md) and [RBF-FD stencil](one-stencil.md) tutorials.

The example solves a manufactured scalar Poisson problem with Dirichlet data on three edges and a Robin condition on the top edge. The Robin operator combines the value with the outward normal derivative; at the top edge \(n=(0,1)\). Corner labels follow the geometry's tag priority, so each corner gets one enforced boundary row.

The manufactured solution is \(u=x^2+y^2\), hence
\(-\Delta u=-4\). On the top edge, the outward normal is \((0,1)\) and the
Robin functional is \(B u=2u+\partial_nu\), so its prescribed value there
is \(2(x^2+1)+2\). Other edges prescribe \(u\) itself. Interior rows of
the sparse system approximate \(-\Delta\); top boundary rows approximate
\(B\); remaining boundary rows are value interpolation. The script
declares these rows symbolically, assembles them on one cloud, solves,
and checks the returned field at every node.

```python
--8<-- "examples/tutorials/mixed_boundary.py"
```

[Download the runnable script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/mixed_boundary.py).

Run `python -m examples.tutorials.mixed_boundary`. A verified 36-node case assembled a \(36\times36\) sparse matrix with 720 nonzeros and a nodal maximum error of \(2.953\times10^{-14}\) for its manufactured polynomial. This checks the boundary signs and polynomial reproduction on this small cloud; it does not measure a general convergence rate.

Change the Robin coefficient and manufacture matching data again. Inspect the boundary tags and row indices before comparing errors. See the [boundary/equation API](../api/equations-spaces.md) and [error measures](../theory/errors.md).
