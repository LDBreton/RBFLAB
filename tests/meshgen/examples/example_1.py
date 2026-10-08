import numpy as np
from rbflab.meshgen import RBFMesh, Border


# Define a parametric function for a circle
def circle_parametric_function(radius, t):
    return radius * np.cos(t), radius * np.sin(t)


# Define the borders of the mesh
outer_radius = 1.0
inner_radius = 0.5

border_outer1 = Border(parametric_function=lambda t: circle_parametric_function(outer_radius, t), label=1, t_start=0,
                       t_end=np.pi)
border_outer2 = Border(parametric_function=lambda t: circle_parametric_function(outer_radius, t), label=1,
                       t_start=np.pi, t_end=2 * np.pi)
border_inner1 = Border(parametric_function=lambda t: circle_parametric_function(inner_radius, t), label=1, t_start=0,
                       t_end=2 * np.pi)

# Generate a random mesh
random_mesh = RBFMesh(border_outer1(100), border_outer2(200), border_inner1(-100))

# Generate points
num_points = 10000
random_mesh.generate_points(num_points)

# Plot the points
