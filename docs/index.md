---
title: RBFLAB manual
hide:
  - toc
---

<div class="rbf-home">
<section class="rbf-hero" aria-labelledby="manual-title">
<p class="rbf-eyebrow">RBFLAB · Scientific computing in Python</p>
<h1 id="manual-title">From scattered nodes to a numerical solution.</h1>
<p class="rbf-lead">RBFLAB is a Python library for radial basis function interpolation and PDEs on point clouds. The six-tutorial route uses the unit square to connect mathematical ideas, library objects, and computed results.</p>
<div class="rbf-actions">
<a class="rbf-button rbf-button--primary" href="tutorials/">Start the tutorials <span aria-hidden="true">→</span></a>
<a class="rbf-button" href="getting-started/">Solve a first problem</a>
<a class="rbf-button" href="INSTALL/">Install RBFLAB</a>
</div>
<div class="rbf-install"><code>python -m pip install "rbflab>=0.5"</code><span>Python 3.11+ · Version 0.5.0 learning path</span></div>
</section>

<figure class="rbf-feature">
<a href="tutorials/" aria-label="Follow the six tutorial lessons"><img src="assets/tutorial_overview.png" alt="A square point cloud, a sparse differential operator, and its computed solution"></a>
<figcaption class="rbf-caption">The same basic workflow runs through the lessons: nodes → operators → solution. <a href="tutorials/">Follow the route →</a></figcaption>
</figure>

<h2>Choose where to begin</h2>
<div class="rbf-chapters">
<section class="rbf-chapter"><span class="rbf-chapter-number" aria-hidden="true">01</span><div><h3><a href="getting-started/">First problem</a></h3><p>Declare a Poisson equation and inspect its solution on a square cloud.</p></div></section>
<section class="rbf-chapter"><span class="rbf-chapter-number" aria-hidden="true">02</span><div><h3><a href="tutorials/">Six tutorials</a></h3><p>Move from interpolation to local operators, stationary equations, global and Hermite methods, then heat evolution.</p></div></section>
<section class="rbf-chapter"><span class="rbf-chapter-number" aria-hidden="true">03</span><div><h3><a href="gallery/">Gallery and advanced</a></h3><p>Apply the same ideas to curved boundaries, holes, 3D domains, and coupled flow.</p></div></section>
<section class="rbf-chapter"><span class="rbf-chapter-number" aria-hidden="true">04</span><div><h3><a href="theory/">Mathematics and API</a></h3><p>Use the derivations, capability table, and reference while changing a numerical method.</p></div></section>
</div>

<p class="rbf-caption">The core lessons use Python and the release 0.5 local interface. Global interpolation and collocation have separate coefficient-based APIs. See <a href="CAPABILITIES/">capabilities</a> for backend and method support.</p>
</div>
