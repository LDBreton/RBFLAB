"""Gmsh perforated plate: explicit triangles to geometry-only staggered clouds.

Install the optional dependency: python -m pip install gmsh
Run: python examples/perforated_plate.py --output outputs/perforated_plate.png
Gmsh API reference: https://gmsh.info/doc/texinfo/#Gmsh-API
"""
import argparse

import numpy as np

from rbflab.geometry import TriangleMesh2D, staggered_clouds, save_staggered_layout


HOLES = [(-1.35, -0.15, 0.38), (0., 0.4, 0.32), (1.3, -0.2, 0.48)]


def build_perforated_plate(mesh_size=0.24):
    """Return (TriangleMesh2D, original_node_tags) from a fresh Gmsh session.

    Node tags need not be contiguous or zero-based. The returned tag array maps
    each vertex row back to its Gmsh tag. Physical curve groups supply labels.
    This example owns its Gmsh session; it refuses to replace an active one.
    """
    import gmsh

    if gmsh.isInitialized():
        raise RuntimeError('Run this standalone example without an active Gmsh session')
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.add('perforated_plate')
        geo = gmsh.model.geo
        corners = [geo.addPoint(x, y, 0, mesh_size)
                   for x, y in [(-2.5, -1.3), (2.5, -1.3), (2.5, 1.3), (-2.5, 1.3)]]
        sides = [geo.addLine(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        loops = [geo.addCurveLoop(sides)]
        groups = dict(zip(['bottom', 'right', 'top', 'left'], [[side] for side in sides]))
        for i, (x, y, radius) in enumerate(HOLES):
            center = geo.addPoint(x, y, 0)
            points = [geo.addPoint(x + radius * dx, y + radius * dy, 0, mesh_size / 2)
                      for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
            arcs = [geo.addCircleArc(points[j], center, points[(j + 1) % 4]) for j in range(4)]
            loops.append(geo.addCurveLoop(arcs))
            groups[f'hole_{i + 1}'] = arcs
        surface = geo.addPlaneSurface(loops)
        geo.synchronize()
        for label, curves in groups.items():
            tag = gmsh.model.addPhysicalGroup(1, curves)
            gmsh.model.setPhysicalName(1, tag, label)
        gmsh.model.addPhysicalGroup(2, [surface])
        gmsh.option.setNumber('Mesh.ElementOrder', 1)
        gmsh.model.mesh.generate(2)
        from rbflab.meshgen import from_gmsh
        return from_gmsh()
    finally:
        gmsh.finalize()


def plot_layout(mesh, layout, output=None):
    import matplotlib.pyplot as plt

    if output:
        plt.switch_backend('Agg')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    colors = {'bottom': '#73808c', 'right': '#73808c', 'top': '#73808c', 'left': '#73808c',
              'hole_1': '#aa3684', 'hole_2': '#208e80', 'hole_3': '#cb7723'}
    v = mesh.vertices
    axes[0].triplot(*v.T, mesh.triangles, color='#b1bac4', linewidth=.45)
    axes[0].set_title(f'Gmsh triangle mesh · {len(mesh.triangles):,} triangles', loc='left')
    for ax, cloud, marker, color, title in [
        (axes[1], layout.vertices, 'o', '#2763ad', 'Vertex cloud'),
        (axes[2], layout.edge_midpoints, 's', '#d16c22', 'Edge-midpoint cloud'),
    ]:
        ax.scatter(*cloud.points.T, s=7, marker=marker, c=color, linewidths=0)
        ax.set_title(f'{title} · {len(cloud.points):,} nodes', loc='left')
        for label, indices in cloud.boundary.items():
            stride = max(1, len(indices) // 12)
            selected = np.arange(0, len(indices), stride)
            ax.quiver(*cloud.points[indices[selected]].T, *cloud.normals[label][selected].T,
                      color=colors[label], angles='xy', scale_units='xy', scale=8, width=.002)
    for ax in axes:
        for label, pairs in mesh.boundary_edges.items():
            segments = v[pairs]
            for segment in segments:
                ax.plot(*segment.T, color=colors[label], lw=1)
        ax.set(xlim=(-2.7, 2.7), ylim=(-1.5, 1.5), aspect='equal', xlabel='x', ylabel='y')
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('One mesh, two geometric point clouds\nPerforated plate with three labeled holes', fontsize=16)
    if output:
        fig.savefig(output, dpi=170)
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--export')
    args = parser.parse_args()
    mesh, tags = build_perforated_plate()
    layout = staggered_clouds(mesh)
    if args.export:
        save_staggered_layout(args.export, layout)
    plot_layout(mesh, layout, args.output)
    print(f'{len(tags)} vertices, {len(mesh.triangles)} triangles, '
          f'{len(layout.edge_midpoints.points)} edge midpoints')


if __name__ == '__main__':
    main()
