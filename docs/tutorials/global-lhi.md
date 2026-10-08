# Compare global collocation, LHI, and RBF-FD

**You will learn:** how one Poisson PDE yields distinct ansatz functions, unknown vectors, and evaluation rules. Prerequisites: [interpolation](interpolation.md) and [RBF-FD](one-stencil.md).

Solve \(-\Delta u=2\pi^2\sin(\pi x)\sin(\pi y)\) with zero Dirichlet data on the unit square. Symmetric global collocation applies PDE/boundary functionals to both kernel arguments. Asymmetric global collocation uses ordinary kernel translates as its trial basis. LHI forms local Hermite weight systems with solution, boundary, and PDE centers. RBF-FD uses local value-interpolation weights. See the [derivation](../theory/global-lhi.md) before interpreting their matrices.

If \(F_i\) is a PDE or boundary row, asymmetric global
collocation uses \(A_{ij}=F_i^xK(x_i,x_j)\), while symmetric Hermite
collocation uses \(G_{ij}=F_i^xF_j^yK(x_i,x_j)\). Their solved vectors
are coefficients of different trial functions. LHI instead writes
one local functional identity at each interior center and keeps
only solution-center weights in its sparse global row. RBF-FD keeps
weights obtained from local value interpolation. The script uses one
PDE and cloud, constructs each method object, solves its system, and
samples all four results at the same seeded off-node points.

```python
--8<-- "examples/tutorials/compare_methods.py"
```

[Download the runnable script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/compare_methods.py).

Run `python -m examples.tutorials.compare_methods`. With 36 cloud nodes, both global matrices and the RBF-FD matrix are \(36\times36\); the LHI matrix is \(16\times16\) because it solves for interior solution values. LHI also pays for local Hermite systems. At 40 independent off-node points, verified maximum errors were \(2.459\times10^{-2}\) (symmetric global), \(1.952\times10^{-2}\) (asymmetric global), \(1.764\times10^{-2}\) (LHI), and \(9.923\times10^{-3}\) (RBF-FD).

The global methods use IMQ(2); the local methods use PHS5 with degree-two polynomials. Therefore these numbers compare complete example recipes, **not** algorithms at matched parameters. Repeat with denser clouds and report both nodal and off-node errors, stencil size, and conditioning. See [discretizations](../api/discretizations.md).
