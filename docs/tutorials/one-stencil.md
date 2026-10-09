# Build one RBF-FD stencil

**You will learn:** to turn local kernel interpolation into one sparse Laplacian row, inspect its weights, and check polynomial reproduction. Prerequisite: basic NumPy.


<figure class="stencil-figure">
<a href="../../assets/rbf_fd_stencil.svg" aria-label="Open the stencil diagram at full size"><img src="../../assets/rbf_fd_stencil.svg" alt="An irregular 121-node cloud with 20 selected neighbors around a target, followed by their signed Laplacian weights in a sparse matrix row"></a>
<figcaption>One computed RBF-FD row on an irregular cloud. Blue nodes form the 20-point stencil; the orange star is the target. The dashed circle marks its radius R, not compact kernel support. The right panel shows R<sup>2</sup> times the Laplacian weights in global node order; unselected columns are zero. The figure uses PHS5, degree-two polynomials, and a seeded 121-node cloud; the runnable one-stencil tutorial defaults to 36 nodes.</figcaption>
</figure>

For the complete coefficient-elimination argument, read [from approximation to weights](../theory/local-weights.md).

## The local approximation

For a target \(\boldsymbol{x}_i\) and nearby nodes \(S_i\), RBF-FD approximates

$$
\Delta u(\boldsymbol{x}_i)\approx\sum_{j\in S_i}w_{ij}u(\boldsymbol{x}_j).
$$

The example uses the signed PHS5 kernel \(\phi(r)=-r^5\), degree-two polynomials \(p=(1,x,y,x^2,xy,y^2)\), and 20 stencil nodes. The local augmented interpolation matrix and weight equation are

$$
G=\begin{bmatrix}\Phi&P\\P^{\mathsf T}&0\end{bmatrix},\qquad
G^{\mathsf T}\begin{bmatrix}w\\\lambda\end{bmatrix}
=\begin{bmatrix}\Delta_x\phi(\lVert x-x_j\rVert)|_{x=x_i}\\\Delta p_m(x_i)\end{bmatrix}.
$$

Here \(\Phi_{jk}=\phi(\lVert x_j-x_k\rVert)\). Polynomial multipliers \(\lambda\) enforce reproduction and are not nodal solution values. The transpose records the functional convention even when this scalar matrix is symmetric.

## Compute and inspect the row

Run `python -m examples.tutorials.one_stencil`. The full script is included below:

??? example "Complete runnable script"

    ```python
    --8<-- "examples/tutorials/one_stencil.py"
    ```

[Download the runnable script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/one_stencil.py).

## Verify consistency

`op.local(0)` gives selected indices, weights, multipliers, and diagnostics. `op.reconstruct_local(0)` rebuilds the local matrix and right-hand side on demand. `op.matrix` is the assembled sparse differentiation row. With 36 cloud nodes, the example forms a \(26\times26\) local matrix (20 values and 6 polynomial constraints) and a \(1\times36\) global row. It reports \(\sum_jw_{ij}\approx0\), \(\sum_jw_{ij}(x_j^2+y_j^2)\approx4\), and a local equation residual near \(6.1\times10^{-14}\).

These checks establish local algebraic consistency, not PDE stability. Change the stencil size and inspect the selected radius and residual; then try the [heat equation](heat-equation.md). The [RBF-FD derivation](../theory/rbf-fd.md) and [discretization API](../api/discretizations.md) explain the conventions.
