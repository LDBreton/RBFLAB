# Interpolate scattered data

**Goal:** turn measured values at irregular locations into a field you can query.
You need NumPy arrays and the idea of an RBF expansion; no PDE or mesh generator
is needed. This lesson uses Float64 and the Python backend.

## 1. Represent locations and measurements

Let $X=\{x_i\}_{i=1}^N\subset\mathbb R^2$ and let $d_i$ be the measured value at
$x_i$. Locations and values are separate arrays. Import the two packages:

```python
import numpy as np
import rbflab as rbf
```

```python
--8<-- "examples/tutorials/interpolation.py:data"
```

`centers` has shape `(64, 2)`; row `i` is $(x_i,y_i)$. `values` has shape `(64,)`;
entry `i` belongs to that row. We generate values from $\sin x+\cos y$ so the
example is self-contained. Replace these two arrays with your measurements.
There are no boundary labels or interior/boundary distinctions in this problem.

## 2. Choose the interpolating space

An augmented interpolant has the form

$$s(x)=\sum_{j=1}^N a_j\phi(\|x-x_j\|)+\sum_{m=1}^Q b_m p_m(x).$$

The coefficients satisfy $s(x_i)=d_i$ and $P^Ta=0$. In matrix form,

$$
\begin{bmatrix}\Phi&P\\P^T&0\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
=\begin{bmatrix}d\\0\end{bmatrix}.
$$

Here $\Phi_{ij}=\phi(\|x_i-x_j\|)$ and $P_{im}=p_m(x_i)$. Without polynomial
augmentation, the system is simply $\Phi a=d$.

```python
--8<-- "examples/tutorials/interpolation.py:interpolants"
```

Both calls fit **the same data**. `IMQ(2)` uses $(1+2r^2)^{-1/2}$ without
polynomials here. `PHS(5)` uses the signed kernel $-r^5$ and requires at least
degree-two polynomials. In 2D that space is spanned by $1,x,y,x^2,xy,y^2$.
The implementation normalizes its polynomial basis internally without changing
this space. `interpolate` assembles and solves the dense coefficient system.

The returned objects represent functions. Their coefficients are generally
**not** the supplied measurements: the measurements are values of the sum.

## 3. Evaluate at new locations

```python
--8<-- "examples/tutorials/interpolation.py:evaluate"
```

A query is another `(M, 2)` coordinate array. `predicted[k]` approximates the
field at `query[k]`; the query locations need not be sample centers.
`kernel_matrix` exposes the unaugmented IMQ block $\Phi$, not the PHS augmented
matrix. Building it separately is optional when using `interpolate`.

![Scattered samples, the PHS reconstruction, and its x derivative](../assets/teaching_interpolation.png)

The picture uses the same 64 samples and PHS recipe. The display grid covers
$[-0.7,0.7]^2$; it adds no interpolation constraints. The derivative panel is
constructed in the [next lesson](differentiation.md).

## 4. Use polynomials deliberately

Polynomial augmentation also guarantees reproduction of the chosen polynomial
space, provided the interpolation system is nonsingular. For example:

```python
--8<-- "examples/tutorials/interpolation.py:polynomial"
```

This is a separate illustration using the quadratic $1+x+y^2$. We change the
**data**, while keeping the same kernel and nodes.

## Adapt this to your data

- Replace `centers` and `values`; preserve their row correspondence.
- Change `query` to the locations where you need predictions.
- Choose the kernel and polynomial degree together. See [kernel conventions](../theory/interpolation.md).
- For 3D data, use `(N, 3)` centers and `(M, 3)` queries.

This API performs exact interpolation, not noise-aware smoothing or regression.
Repeated centers can make the system singular. For large data sets, a global
dense interpolant has different memory costs from local methods. Extrapolation
beyond the sampled region needs care; the API does not supply an uncertainty bound.

### A quick check

Evaluate at `centers` and compare with `values`. For the quadratic example,
compare `reproduced` with `1 + query[:, 0] + query[:, 1]**2`. These checks help
catch data-order mistakes; they are not a substitute for knowing your data.

## Complete example

Run `python -m examples.tutorials.interpolation` from a [source checkout](../INSTALL.md).
The short API fragments above also work with an installed package when combined
with their imports and preceding steps.

??? example "Complete runnable script"

    ```python
    --8<-- "examples/tutorials/interpolation.py"
    ```

[Download the script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/interpolation.py).

**Next:** [Differentiate a field](differentiation.md) or [define a kernel](custom-kernel.md).
