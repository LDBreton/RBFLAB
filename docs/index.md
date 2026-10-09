---
title: User manual
hide:
  - toc
---

<div class="rbf-home">
<section class="rbf-hero" aria-labelledby="manual-title">
<p class="rbf-eyebrow">RBFLAB · Scientific computing in Python</p>
<h1 id="manual-title">Radial basis functions.<br>From equations to solutions.</h1>
<p class="rbf-lead">A practical manual for interpolation and partial differential equations on point clouds. Write equations symbolically, choose a numerical method, and work directly with the matrices behind the solution.</p>
<div class="rbf-actions">
<a class="rbf-button rbf-button--primary" href="getting-started/">Solve your first PDE <span aria-hidden="true">→</span></a>
<a class="rbf-button" href="tutorials/heat-equation/">Learn through the heat equation</a>
</div>
<div class="rbf-install"><code>pip install rbflab</code><span>Python 3.11+ · Open source · Version 0.3.0</span></div>
</section>

<figure class="rbf-feature">
<a href="geometry/" aria-label="Explore curved domains"><img src="assets/geometry_gallery.png" alt="Generated nodes on domains with obstacles, curved holes, concave edges and a 3D volume" width="2145" height="1402"></a>
<figcaption class="rbf-caption">Named boundaries, curved holes, concave polygons and 3D volumes. <a href="geometry/cookbook/">Build your geometry →</a></figcaption>
</figure>

<h2 id="explore">Explore the manual</h2>
<div class="rbf-chapters">
<section class="rbf-chapter">
<span class="rbf-chapter-number" aria-hidden="true">01</span>
<div><h3><a href="getting-started/">Getting started</a></h3><p>Install the library and solve a Poisson problem. Connect the equation, point cloud, discretization, and solution.</p></div>
</section>
<section class="rbf-chapter">
<span class="rbf-chapter-number" aria-hidden="true">02</span>
<div><h3><a href="tutorials/one-stencil/">Learning by examples</a></h3><p>Build one stencil, assemble heat matrices, impose boundary conditions, and move on to Stokes and 3D problems.</p></div>
</section>
<section class="rbf-chapter">
<span class="rbf-chapter-number" aria-hidden="true">03</span>
<div><h3><a href="theory/">Mathematical foundations</a></h3><p>Understand kernel conventions, polynomial constraints, local Hermite systems, conditioning, and error measures.</p></div>
</section>
<section class="rbf-chapter">
<span class="rbf-chapter-number" aria-hidden="true">04</span>
<div><h3><a href="api/discretizations/">Python API reference</a></h3><p>Inspect operators, local weights, sparse matrices, spaces, precision settings, and the available backends.</p></div>
</section>
</div>

<h2 id="methods">One problem, several numerical methods</h2>
<p>Keep the equation and cloud, then change the approximation. The manual explains what each system solves for and how to assess its result.</p>

<table>
<thead><tr><th>Method</th><th>Global unknowns</th><th>Matrix</th></tr></thead>
<tbody>
<tr><td><a href="tutorials/global-lhi/">Global collocation</a></td><td>Kernel expansion coefficients</td><td>Dense</td></tr>
<tr><td><a href="tutorials/global-lhi/">Local Hermite interpolation</a></td><td>Interior solution-center values</td><td>Sparse</td></tr>
<tr><td><a href="tutorials/one-stencil/">RBF finite differences</a></td><td>Nodal values</td><td>Sparse</td></tr>
<tr><td><a href="tutorials/stokes/">Divergence-free Stokes</a></td><td>Velocity and pressure-space coefficients</td><td>Coupled</td></tr>
</tbody>
</table>

<h2 id="curved">Beyond the square</h2>
<div class="rbf-examples">
<article class="rbf-example"><a href="tutorials/flower-heat/"><img src="assets/flower_heat.gif" alt="Computed heat diffusion on a flower-shaped domain" loading="lazy"></a><div class="rbf-example-body"><h3><a href="tutorials/flower-heat/">Heat on a flower</a></h3><p>Symbolic forcing, evolving boundary data and a reusable animation call.</p></div></article>
<article class="rbf-example"><a href="tutorials/ellipse/"><img src="assets/ellipse_boundary.png" alt="Mixed ellipse boundaries and numerical field" loading="lazy"></a><div class="rbf-example-body"><h3><a href="tutorials/ellipse/">Curved boundary conditions</a></h3><p>Dirichlet, Neumann and Robin conditions on one labeled ellipse.</p></div></article>
<article class="rbf-example"><a href="tutorials/ball/"><img src="assets/ball_poisson.png" alt="3D cloud and numerical cross-section" loading="lazy"></a><div class="rbf-example-body"><h3><a href="tutorials/ball/">Into three dimensions</a></h3><p>A nonpolynomial manufactured solution inside a ball.</p></div></article>
</div>
<h2 id="examples">See the methods at work</h2>
<div class="rbf-examples rbf-method-gallery">
<article class="rbf-example">
<a href="tutorials/heat-equation/"><img src="assets/method_heat.png" alt="Computed RBF-FD heat surface at t = 0.02" loading="lazy"></a>
<div class="rbf-example-body"><h3><a href="tutorials/heat-equation/">Heat equation</a></h3><p>Five-point FD, RBF-FD matrices, and symbolic assembly for the same problem.</p></div>
</article>
<article class="rbf-example">
<a href="tutorials/annular-stokes/"><img src="assets/method_stokes.png" alt="Computed annular Stokes speed and streamlines" loading="lazy"></a>
<div class="rbf-example-body"><h3><a href="tutorials/annular-stokes/">Divergence-free Stokes</a></h3><p>Rotating-cylinder flow in an annulus, with measured velocity and pressure-gradient errors.</p></div>
</article>
<article class="rbf-example">
<a href="tutorials/cavity/"><img src="assets/method_cavity.png" alt="Computed Re = 100 cavity speed and streamlines at t = 20" loading="lazy"></a>
<div class="rbf-example-body"><h3><a href="tutorials/cavity/">Lid-driven cavity</a></h3><p>A custom Navier–Stokes algorithm, with diagnostics and explicit validation limits.</p></div>
</article>
</div>

<p class="rbf-caption">Computed fields, rendered for a closer look: RBF-FD heat at t = 0.02, steady annular Stokes, and the coarse Re = 100 cavity at t = 20. The cavity is an illustrative run, not a grid-converged benchmark.</p>

<aside class="rbf-note"><p><strong>Designed for exploration.</strong> The core runs with Python, NumPy, and SciPy. Optional C++ and PyTorch backends accelerate supported local methods. Check the <a href="CAPABILITIES/">capability table</a> for dimensions, precision, and method support, or follow the <a href="INSTALL/">installation guide</a>.</p></aside>
<p class="rbf-caption">Tutorial module commands run from a source checkout. The curved tutorials use a shared helper in the examples folder; keep that folder together. The cavity wrapper also uses its maintained example solver.</p>
</div>
