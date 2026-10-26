"""Generate the mesh-style comparison shown in the construction overview.

Run python -m examples.mesh_styles; requires examples and triangulation extras.
"""
from rbflab import geometry as g, meshgen as m


def main():
    domain = g.Annulus(inner_radius=.35, outer_radius=1.)
    boundary = {"inner": 40, "outer": 80}
    items = []
    for method in ("random", "halton", "sobol"):
        cloud = m.generate(domain, interior=200, boundary=boundary,
                           method=method, seed=42)
        items.append((method, cloud))
    items.append(("Clipped Padua / degree 24", m.generate(
        domain, method="padua", degree=24, boundary=boundary)))
    items.append(("Mapped Padua / clustering 2", m.mapped_padua(
        domain, degree=20, clustering=2., boundary=boundary)))
    mesh = m.triangulate(domain, boundary=boundary, growth=.25)
    items.append(("Boundary-refined triangles", g.staggered_clouds(mesh).vertices))
    # Keep actual connectivity visible on the triangulation panel.
    import matplotlib.pyplot as plt
    from examples.mesh_construction import panel, save, BG
    fig, axes = plt.subplots(2, 3, figsize=(18.6, 10.2), facecolor=BG)
    for ax, (title, cloud) in zip(axes.flat, items):
        if title == "Boundary-refined triangles":
            ax.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles,
                       color="#334c64", linewidth=.5)
        panel(ax, cloud, f"{title}\n{len(cloud.points)} total nodes")
        print(title, "interior", len(cloud.interior), "total", len(cloud.points))
    fig.subplots_adjust(wspace=.24, hspace=.65, bottom=.15)
    save(fig, "styles.png")


if __name__ == "__main__":
    main()
