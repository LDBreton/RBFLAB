---
title: User manual
hide:
  - toc
---

<div class="rbf-home">
<section class="rbf-hero" aria-labelledby="manual-title">
<p class="rbf-eyebrow">RBFLAB · Scientific computing in Python</p>
<h1 id="manual-title">Radial basis functions.<br>From equations to solutions.</h1>
<p class="rbf-lead">A Python library for interpolation and PDEs on scattered nodes. Start with an equation and a curved domain, or build local weights and sparse operators for your own algorithm. The tutorials connect each mathematical choice to the Python object that represents it.</p>
<div class="rbf-actions">
<a class="rbf-button rbf-button--primary" href="getting-started/">Solve your first PDE <span aria-hidden="true">→</span></a>
<a class="rbf-button" href="tutorials/local-approximation/">Build your operators</a>
<a class="rbf-button" href="INSTALL/">Install Python or C++</a>
</div>
<div class="rbf-install"><code>pip install rbflab</code><span>Python 3.11+ · Open source · Version 0.5.0</span></div>
</section>

<figure class="rbf-feature">
<a href="geometry/" aria-label="Explore curved domains"><img src="assets/geometry_gallery.png" alt="Generated nodes on domains with obstacles, curved holes, concave edges and a 3D volume" width="2145" height="1402"></a>
<figcaption class="rbf-caption">Named boundaries, curved holes, concave polygons and 3D volumes. <a href="geometry/planar-domains/">Build your geometry →</a></figcaption>
</figure>

<h2 id="explore">Explore the manual</h2>
<div class="rbf-chapters">
<section class="rbf-chapter">
<span class="rbf-chapter-number" aria-hidden="true">01</span>
<div><h3><a href="getting-started/">Getting started</a></h3><p>Install the library and solve a Poisson problem. Connect the equation, point cloud, discretization, and solution.</p></div>
</section>
<section class="rbf-chapter">
<span class="rbf-chapter-number" aria-hidden="true">02</span>
<div><h3><a href="tutorials/">Tutorials</a></h3><p>Follow the mathematics into code: interpolate data, solve PDEs on curved domains, and assemble your own sparse algorithm.</p></div>
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

<h2 id="local-api">From approximation to your algorithm</h2>
<p>RBFLAB 0.5 makes ordinary RBF-FD and Hermite constructions explicit through <code>LocalApproximation</code>. <a href="tutorials/local-approximation/">Declare source functionals and target operators</a>, then <a href="tutorials/lhi-matrices/">assemble LHI and a heat time loop with sparse matrices</a>. Install or upgrade to <code>rbflab&gt;=0.5</code> to use these interfaces.</p>
<figure class="rbf-feature"><a href="guides/research-api/"><img src="assets/local-architecture.svg" alt="Spaces and sampled functionals become local systems, named sparse blocks and user-written algorithms"></a></figure>

<h2 id="methods">One problem, several numerical methods</h2>
<p>Keep the equation and cloud, then change the approximation. The manual explains what each system solves for and how to assess its result.</p>

<table>
<thead><tr><th>Method</th><th>Global unknowns</th><th>Matrix</th></tr></thead>
<tbody>
<tr><td><a href="tutorials/global-collocation/">Global collocation</a></td><td>Kernel expansion coefficients</td><td>Dense</td></tr>
<tr><td><a href="tutorials/lhi/">Local Hermite interpolation</a></td><td>Interior solution-center values</td><td>Sparse</td></tr>
<tr><td><a href="tutorials/one-stencil/">RBF finite differences</a></td><td>Nodal values</td><td>Sparse</td></tr>
<tr><td><a href="tutorials/annular-stokes/">Divergence-free Stokes</a></td><td>Velocity and pressure-space coefficients</td><td>Coupled</td></tr>
</tbody>
</table>

<h2 id="examples">Choose a problem and see how it works</h2>
<p>For a guided learning sequence, choose <a href="tutorials/">interpolation, PDEs, or custom algorithms</a>. Instructors can use the <a href="tutorials/teaching/">teaching guide and modification exercises</a>.</p>
<div class="rbf-examples">
<article class="rbf-example"><a href="tutorials/perforated-poisson/"><img src="assets/perforated_poisson.png" alt="Computed Poisson field on an ellipse with two holes" loading="lazy"></a><div class="rbf-example-body"><h3><a href="tutorials/perforated-poisson/">Poisson with holes</a></h3><p>Geometry, labels, symbolic forcing and a measured solution error.</p></div></article>
<article class="rbf-example"><a href="tutorials/flower-heat/"><img src="assets/flower_heat.gif" alt="Computed heat diffusion on a flower-shaped domain" loading="lazy"></a><div class="rbf-example-body"><h3><a href="tutorials/flower-heat/">Heat on a flower</a></h3><p>Solve a forced transient equation with nonzero boundary data on a curved domain.</p></div></article>
<article class="rbf-example"><a href="tutorials/cavity/"><img src="assets/method_cavity.png" alt="Computed Re 100 cavity velocity on a coarse cloud" loading="lazy"></a><div class="rbf-example-body"><h3><a href="tutorials/cavity/">Build a flow algorithm</a></h3><p>Staggered maps, wall values, convection and the full pressure–velocity block solve.</p></div></article>
</div>
<p class="rbf-caption">Each tutorial states its equations, numerical recipe and limitations. The cavity image illustrates an experimental coarse-cloud demonstration with a measured divergence defect; it is not a benchmark result. <a href="tutorials/">Explore the full learning route →</a></p>

<aside class="rbf-note"><p><strong>Designed for exploration.</strong> The core runs with Python, NumPy, and SciPy. Optional C++ and PyTorch backends accelerate supported local methods. Check the <a href="CAPABILITIES/">capability table</a> for dimensions, precision, and method support, or follow the <a href="INSTALL/">installation guide</a>.</p></aside>
<p class="rbf-caption">Tutorial module commands run from a source checkout. Some tutorials use a shared helper in the examples folder; keep that folder together. The cavity chapter shows the maintained solver itself, including assembly and the time loop.</p>
</div>
