# A 3D scalar Poisson problem

**You will learn:** to pass three coordinates, identify boundary nodes, and request the three-dimensional Laplacian. Prerequisites: the [mixed-boundary](mixed-boundary.md) tutorial.

The example uses a structured \(4\times4\times4\) point cloud in the unit cube and a manufactured quadratic solution. A 3D Laplacian is \(\Delta u=u_{xx}+u_{yy}+u_{zz}\); the operator dimension and cloud coordinate dimension must agree.

The manufactured field is \(u=x^2+y^2+z^2\), so
\(-\Delta u=-6\) throughout the cube and Dirichlet data are the field's
trace on its six tagged faces. With three coordinates, degree-two
polynomials contain ten monomials, and a 30-point local stencil has
enough nodes to represent them. The script solves the scalar PDE and
separately builds a 3D Laplacian matrix at interior targets; applying
it to exact nodal quadratic values should produce six.

??? example "Complete runnable script"

    ```python
    --8<-- "examples/tutorials/scalar_3d.py"
    ```

[Download the runnable script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/scalar_3d.py).

Run `python -m examples.tutorials.scalar_3d`. A verified 64-node run had nodal maximum error \(4.705\times10^{-15}\) and quadratic Laplacian reproduction error \(2.487\times10^{-14}\). It is a polynomial consistency check on a small cloud, not proof of accuracy for arbitrary 3D geometry. Try more nodes and visualize an interior slice or scatter plot with labeled coordinates. See [geometry/operators](../api/kernels-geometry.md) and [error measures](../theory/errors.md).
