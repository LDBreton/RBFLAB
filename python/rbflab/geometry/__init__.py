"""Geometry containers and domains, independent of numerical PDE methods."""
from .cloud import PointCloud, unit_box_grid, gmsh_square, gmsh_cube
from ._borders import MeshPoint, Border, is_close, find_polygons
from ._regions3d import MeshPoint3D, ImplicitRegion, Sphere, Box, Cylinder
from .staggered import TriangleMesh2D, StaggeredLayout, staggered_clouds, save_staggered_layout, load_staggered_layout

from .domains import ParametricBoundary, ParametricDomain, Disk, Ellipse, Annulus, Flower

__all__ = ['PointCloud', 'unit_box_grid', 'gmsh_square', 'gmsh_cube', 'MeshPoint', 'Border', 'is_close', 'find_polygons', 'MeshPoint3D', 'ImplicitRegion', 'Sphere', 'Box', 'Cylinder', 'TriangleMesh2D', 'StaggeredLayout', 'staggered_clouds', 'save_staggered_layout', 'load_staggered_layout', 'ParametricBoundary', 'ParametricDomain', 'Disk', 'Ellipse', 'Annulus', 'Flower']
