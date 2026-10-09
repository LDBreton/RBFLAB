# Papers and further reading

These sources explain the mathematical ideas behind the manual. A citation
does not mean that RBFLAB implements every algorithm or reproduces every
experiment in that paper. The tutorial pages state the actual tested recipes.

## A route through the literature

| Topic | Start here | Read alongside |
|---|---|---|
| RBFs for PDEs | [Fornberg & Flyer (2015)](#fornberg-flyer-2015) | [Interpolation](interpolation.md) |
| PHS and polynomial reproduction | [Flyer et al. (2016)](#flyer-2016), [Bayona et al. (2017)](#bayona-2017) | [RBF-FD weights](rbf-fd.md) |
| Local Hermite data | [Stevens et al. (2009)](#stevens-2009) | [Local Hermite interpolation](lhi.md) |
| Divergence-free spaces | [Narcowich & Ward (1994)](#narcowich-ward-1994), [Wendland (2009)](#wendland-2009) | [Divergence-free kernels](divergence-free.md) |
| Flat kernels and numerical conditioning | [Fornberg et al. (2011)](#fornberg-2011), [Wright & Fornberg (2017)](#wright-fornberg-2017) | [Conditioning](conditioning.md) |
| Compact support | [Wendland (1995)](#wendland-1995) | [Kernel conventions](interpolation.md) |
| Implicit time integration | [Hairer & Wanner (1996)](#hairer-wanner-1996) | [Time discretization](time-discretization.md) |

## RBF approximation and local differentiation

### Fornberg & Flyer (2015) {#fornberg-flyer-2015}

B. Fornberg and N. Flyer.
[Solving PDEs with radial basis functions](https://doi.org/10.1017/S0962492914000130).
*Acta Numerica* **24**, 215–258 (2015).
[Author-hosted manuscript](https://www.colorado.edu/amath/sites/default/files/attached-files/2015_ff_solving_pdes_with_rbfs_acta_numerica_0.pdf).

A broad entry point connecting global RBF methods, flat kernels, and local
RBF-FD differentiation. Read sections 2 and 5 with the interpolation and
single-stencil tutorials.

### Flyer et al. (2016) {#flyer-2016}

N. Flyer, B. Fornberg, V. Bayona, and G. A. Barnett.
[On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy](https://doi.org/10.1016/j.jcp.2016.05.026).
*Journal of Computational Physics* **321**, 21–38 (2016).

Explains polynomial augmentation and stagnation errors in local approximation.
It motivates testing polynomial reproduction and varying polynomial degree,
rather than assigning a convergence rate from the PHS exponent alone.

### Bayona et al. (2017) {#bayona-2017}

V. Bayona, N. Flyer, B. Fornberg, and G. A. Barnett.
[On the role of polynomials in RBF-FD approximations: II. Numerical solution of elliptic PDEs](https://doi.org/10.1016/j.jcp.2016.12.008).
*Journal of Computational Physics* **332**, 257–273 (2017).

Connects polynomially augmented PHS stencils to elliptic PDE solves and boundary
behavior. Useful background for the Poisson and heat examples; its results do
not establish stability for every cloud or stencil choice in this library.

### Wendland (1995) {#wendland-1995}

H. Wendland.
[Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree](https://doi.org/10.1007/BF02123482).
*Advances in Computational Mathematics* **4**, 389–396 (1995).

The construction behind the Wendland family. Smoothness and positive
definiteness depend on the dimension and the chosen member; a compact-support
radius is a kernel parameter, distinct from a nearest-neighbor stencil radius.

## Hermite interpolation and incompressible flow

### Stevens et al. (2009) {#stevens-2009}

D. Stevens, H. Power, M. Lees, and H. Morvan.
[A Meshless Solution Technique for the Solution of 3D Unsaturated Zone Problems, Based on Local Hermitian Interpolation with Radial Basis Functions](https://doi.org/10.1007/s11242-008-9303-z).
*Transport in Porous Media* **79**, 149–169 (2009; online publication 2008).

An application of local Hermite interpolation combining local data and
differential information. It motivates the distinction between value, PDE,
and boundary functionals. The manual's LHI diagram explains RBFLAB's convention;
it is an original schematic, not a reproduced figure or porous-media benchmark.

### Narcowich & Ward (1994) {#narcowich-ward-1994}

F. J. Narcowich and J. D. Ward.
[Generalized Hermite interpolation via matrix-valued conditionally positive definite functions](https://doi.org/10.1090/S0025-5718-1994-1254147-6).
*Mathematics of Computation* **63**, 661–688 (1994).

Mathematical foundations for generalized Hermite interpolation and
matrix-valued kernels, including constrained vector fields.

### Wendland (2009) {#wendland-2009}

H. Wendland.
[Divergence-free kernel methods for approximating the Stokes problem](https://doi.org/10.1137/080730299).
*SIAM Journal on Numerical Analysis* **47**(4), 3158–3179 (2009).

Develops and analyzes Stokes collocation in an analytically divergence-free
velocity space. This is the main reading for the velocity/pressure kernel
construction. Global collocation error results should not be transferred
unchanged to local LHI assembly or off-node reconstruction.

## Conditioning and time integration

### Fornberg et al. (2011) {#fornberg-2011}

B. Fornberg, E. Larsson, and N. Flyer.
[Stable computations with Gaussian radial basis functions](https://doi.org/10.1137/09076756X).
*SIAM Journal on Scientific Computing* **33**(2), 869–892 (2011).
[Author-hosted manuscript](https://www.colorado.edu/amath/sites/default/files/attached-files/siscmanuscript.pdf).

RBF-QR illustrates why an ill-conditioned direct kernel basis does not by
itself imply a poor approximation space. RBFLAB's direct local solvers and
extended precision are not implementations of RBF-QR.

### Wright & Fornberg (2017) {#wright-fornberg-2017}

G. B. Wright and B. Fornberg.
[Stable computations with flat radial basis functions using vector-valued rational approximations](https://doi.org/10.1016/j.jcp.2016.11.030).
*Journal of Computational Physics* **331**, 137–156 (2017).
[Author preprint](https://arxiv.org/abs/1610.05374).

Introduces RBF-RA, including applications to Hermite finite-difference weights.
It is further reading for flat-kernel computations, not a currently documented
RBFLAB backend.

### Hairer & Wanner (1996) {#hairer-wanner-1996}

E. Hairer and G. Wanner.
[Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems](https://doi.org/10.1007/978-3-642-05221-7),
second edition. Springer (1996).

A book reference for stiff systems, multistep stability, and differential-algebraic
equations. Read chapter V for multistep methods and chapter VI when the
semidiscrete mass matrix is singular.
