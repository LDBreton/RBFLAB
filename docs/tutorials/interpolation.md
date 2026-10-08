# Interpolation and derivatives

**You will learn:** to distinguish interpolation coefficients from nodal data, build a kernel matrix, and use polynomial augmentation. Prerequisite: NumPy arrays.

For values \(d_i\) at centers \(x_i\), solve the [augmented interpolation system](../theory/interpolation.md) for kernel coefficients \(a_j\) and polynomial coefficients \(b_m\). The complete example compares IMQ with signed PHS5 plus degree-two polynomials and evaluates at independent query points.

The approximation is \(s(x)=\sum_j a_j\phi(\|x-x_j\|)+\sum_m b_m p_m(x)\).
At the centers, \(s(x_i)=d_i\); for augmented PHS, the side condition
\(P^{\mathsf T}a=0\) fixes the polynomial part. The script first constructs
25 centers and two sets of data, then calls interpolate for each kernel.
It queries a seeded set of 30 off-node points, and constructs an explicit
\(25\times25\) IMQ Gram matrix to show which matrix the coefficient solve
starts from. The polynomial target is \(1+x+y^2\), so degree two should
reproduce it up to roundoff.

```python
--8<-- "examples/tutorials/interpolation.py"
```

[Download the runnable script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/interpolation.py).

Run `python -m examples.tutorials.interpolation`. With 25 centers, the verified IMQ example had sampled maximum error \(1.060\times10^{-2}\). The polynomially augmented PHS example reproduces its quadratic target to \(4.441\times10^{-16}\). These are different target functions, so they are consistency demonstrations rather than an accuracy ranking.

Try evaluating a derivative or varying center count. Check the shape of the kernel matrix and ask whether a coefficient is a value at a center; generally it is not. See the [kernel API](../api/kernels-geometry.md) and [conditioning](../theory/conditioning.md).
