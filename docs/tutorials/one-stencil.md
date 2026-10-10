# Build one RBF-FD stencil

This lesson uses the explicit local API introduced in
[functionals to operators](local-approximation.md): samples describe the data,
a space describes the trial functions, and targets describe what to evaluate.

**You will learn:** to turn local kernel interpolation into one sparse Laplacian row, inspect its weights, and check polynomial reproduction. Prerequisite: basic NumPy.


![The 36-node square cloud, selected 20-point stencil, and its sparse Laplacian row](../assets/tutorial_stencil.png)

The highlighted target has one 20-node neighborhood. Its weights occupy
20 of the 36 columns in the assembled row; other columns are zero.

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

## 1. Select source and target locations

```python
import numpy as np
import rbflab as rbf
cells, stencil_size = 5, 20
```

```python
--8<-- "examples/tutorials/one_stencil.py:geometry"
```

`cloud.points` has 36 rows. `target` has shape `(1, 2)`: one evaluation location
near $(0.5,0.5)$. The figure uses the same 36-node construction.

## 2. Describe the local space and target functional

```python
--8<-- "examples/tutorials/one_stencil.py:operator"
```

`ScalarSpace(PHS(5), 2)` selects the trial family and quadratic polynomials.
The name `u` labels the source-value block; it is not a symbolic field.
`space.representers(source)` constructs the basis from those value functionals.
Requesting `Laplacian(2)` produces a `FunctionalOperator` with a `(1, 36)`
sparse matrix. `op["u"]` selects its only source block.

This example retains physical kernel scaling. Other lessons choose
[local scaling](../theory/conditioning.md#local-stencil-scaling); scaling is a
separate decision from polynomial degree and neighbor count.

## 3. Read the local system and its weights

```python
--8<-- "examples/tutorials/one_stencil.py:inspect"
```

`local.indices` selects 20 cloud rows. Scalar `local.weights` has shape `(20, 1)`;
its polynomial multipliers have shape `(6, 1)`. `reconstructed.matrix` is the
$26\times26$ augmented interpolation system and `.rhs` is its target-functional
column. These are not the global PDE matrix and forcing.

```python
--8<-- "examples/tutorials/one_stencil.py:weights"
```

This directly expresses the transpose equation in the derivation. Only the first
20 entries become nodal differentiation weights; polynomial multipliers are not
columns of the sparse row. `op @ values` applies it to a `(36,)` source vector.

## 4. Supply your own neighborhood

If your method selects neighbors another way, supply them explicitly:

```python
--8<-- "examples/tutorials/one_stencil.py:direct"
```

This returns nodal weights. You own the point selection and their scatter into
global columns. Modifying an inspection copy from `op.local(0)` does not rebuild
`op.matrix`.

### A quick check

The constant response is zero; the response to $x^2+y^2$ is four, up to roundoff.
This makes polynomial reproduction tangible. Further diagnostic interpretation
belongs in [errors and residuals](../theory/errors.md).

## Complete example

Run `python -m examples.tutorials.one_stencil` from a source checkout.

??? example "Complete runnable script"

    ```python
    --8<-- "examples/tutorials/one_stencil.py"
    ```

[Download the script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/one_stencil.py).

**Next:** [Assemble a PDE](custom-assembly.md) or return to the
[main route](index.md).
