# Compare global collocation, LHI, and RBF-FD

This supplementary comparison brings together [global collocation](global-collocation.md),
[RBF-FD](one-stencil.md), and [LHI](lhi.md). Work through their individual
constructions first; then use this page to compare the objects they produce.

| Method | Global unknowns | Local information | Field evaluation |
|---|---|---|---|
| Global asymmetric | Ordinary trial coefficients | All centers | Global radial expansion |
| Global symmetric | Source-functional trial coefficients | All functional centers | Global Hermite expansion |
| RBF-FD | Nodal solution values | Local value interpolants | Local reconstruction |
| LHI | Interior solution-center values | Values, boundary data, PDE data | Local Hermite reconstruction |

Solve \(-\Delta u=2\pi^2\sin(\pi x)\sin(\pi y)\) with zero Dirichlet data on the unit square. Symmetric global collocation applies PDE/boundary functionals to both kernel arguments. Asymmetric global collocation uses ordinary kernel translates as its trial basis. LHI forms local Hermite weight systems with solution, boundary, and PDE centers. RBF-FD uses local value-interpolation weights. See the separate [global](../theory/global.md) and [LHI](../theory/lhi.md) derivations before interpreting their matrices.


<figure class="stencil-figure">
<a href="../../assets/lhi_centers.svg" aria-label="Open the LHI center diagram at full size"><img src="../../assets/lhi_centers.svg" alt="LHI schematic with blue circles for solution values, teal triangles for PDE data, orange squares on the boundary, and a target that has no PDE-data triangle"></a>
<figcaption>An illustrative LHI neighborhood. A circle and triangle can occupy the same location while representing different functionals. The target belongs to the solution centers and is excluded from the PDE-data centers. Marker counts are schematic, not a recommended stencil size.</figcaption>
</figure>

## What each matrix represents

If \(F_i\) is a PDE or boundary row, asymmetric global
collocation uses \(A_{ij}=F_i^xK(x_i,x_j)\), while symmetric Hermite
collocation uses \(G_{ij}=F_i^xF_j^yK(x_i,x_j)\). Their solved vectors
are coefficients of different trial functions. LHI instead writes
one local functional identity at each interior center and keeps
only solution-center weights in its sparse global row. RBF-FD keeps
weights obtained from local value interpolation. The script uses one
PDE and cloud, constructs each method object, solves its system, and
samples all four results at the same seeded off-node points.

## Run the comparison

??? example "Complete runnable script"

    ```python
    --8<-- "examples/tutorials/compare_methods.py"
    ```

[Download the runnable script](https://raw.githubusercontent.com/LDBreton/RBFLAB/main/examples/tutorials/compare_methods.py).

Run `python -m examples.tutorials.compare_methods`. All methods solve the same
Poisson equation on the same 36-node cloud. The global and RBF-FD matrices have
36 rows; LHI has 16 interior equations because boundary data are eliminated
through the local identities. Different matrix dimensions do not by themselves
compare the total work: LHI also builds local Hermite systems.

The global examples use IMQ(2); local examples use PHS5 with quadratic augmentation.
The script's brief solution checks describe these recipes, not a matched method
ranking. To adapt it, first choose which local information and which unknown vector
your mathematical idea requires, then select the corresponding construction.
