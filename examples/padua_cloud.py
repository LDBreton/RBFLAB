"""Padua interiors with existing arc-length and Border boundary samplers."""
import numpy as np
from rbflab import geometry, meshgen, viz


def main():
    import matplotlib.pyplot as plt
    arc_domain = geometry.Annulus(inner_radius=.35, outer_radius=1)
    border = geometry.Border(lambda t: (np.cos(t), np.sin(t)),
                             "wall", 0, 2*np.pi)(80)
    for domain, options in [(arc_domain, {"boundary": {"inner": 40, "outer": 80}}),
                            (border, {})]:
        cloud = meshgen.generate(domain, method="padua", degree=28, **options)
        viz.plot_cloud(cloud, normals=True)
    plt.show()


if __name__ == "__main__":
    main()
