# Kernels and geometry

Kernel shape parameters use the conventions in their docstrings. Polynomial
augmentation is selected on the approximation space: `None` omits polynomial
columns, an integer requests that exact degree, and `"auto"` selects the
kernel's minimum degree. An integer below the conventional CPD minimum is
accepted, though solvability and accuracy can be affected; see
[equations and spaces](equations-spaces.md).

::: rbflab.IMQ

::: rbflab.Gaussian

::: rbflab.PHS

::: rbflab.Hybrid

::: rbflab.DivergenceFree

::: rbflab.Kernel

::: rbflab.Wendland

::: rbflab.PointCloud

::: rbflab.geometry
    options:
      members: false

::: rbflab.meshgen
    options:
      members: false
