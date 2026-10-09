"""Point generation; imports no plotting library or PDE solver.

Use generate() for an array PointCloud. RBFMesh and RBFMesh3D retain the
object-list APIs, including FreeFEM-inspired Border(n) geometry construction.
"""
from ..geometry import Border, MeshPoint, MeshPoint3D, ImplicitRegion, Sphere, Box, Cylinder, find_polygons
from .polygons import (RBFMesh, exclude_nested_polygons, calculate_point_allocation,
                       generate_regions, generate_points_within_polygons)
from .volumes import (RBFMesh3D, calculate_volume_allocation, calculate_boundary_allocation,
                      estimate_region_volume, generate_points_within_regions, generate_boundary_points)
from .surfaces import ParametricSurface3D

from .generate import generate, quality
from .gmsh import from_gmsh

__all__ = ['generate', 'quality', 'from_gmsh', 'RBFMesh', 'RBFMesh3D', 'ParametricSurface3D', 'calculate_point_allocation', 'generate_points_within_polygons', 'generate_regions', 'exclude_nested_polygons', 'calculate_volume_allocation', 'calculate_boundary_allocation', 'estimate_region_volume', 'generate_points_within_regions', 'generate_boundary_points']
