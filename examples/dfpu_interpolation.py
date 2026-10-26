"""Planar divergence-free partition of unity on a triangulated disk.

Run with ``python examples/dfpu_interpolation.py`` from a source install.
Requires the optional triangulation dependencies.
"""
import numpy as np
import rbflab as r


def main():
    mesh = r.meshgen.triangulate(r.geometry.Disk(), boundary=20, max_spacing=.4)
    nodes = mesh.vertices
    cover = r.PatchCover.from_mesh(mesh, n_patches=4, min_points=8, seed=0)

    # Streamfunction psi=sin(x) cos(y)+0.2xy, u=(-psi_y,psi_x).
    x, y = nodes.T
    velocity = np.column_stack((np.sin(x)*np.sin(y)-.2*x,
                                np.cos(x)*np.cos(y)+.2*y))
    space = r.DivergenceFreeSpace(r.IMQ(), polynomial_degree=1)
    with r.interpolate(space, nodes, velocity, patches=cover) as fit:
        query = nodes[mesh.triangles].mean(axis=1)
        reconstructed = fit.evaluate(query)
        print(f'{len(nodes)} mesh vertices, {len(query)} triangle centroids, {len(cover.radii)} patches')
        print('maximum centroid divergence:', np.max(np.abs(fit.divergence(query))))
        print('maximum nodal velocity discrepancy:', np.max(np.abs(fit.evaluate(nodes)-velocity)))
        print('sample streamfunction:', fit.potential(query[:3]))
        print('sample velocity:', reconstructed[:3])


if __name__ == '__main__':
    main()
