import numpy as np
from rbflab.meshgen import RBFMesh, Border


# Define a parametric function for a circle
def circle_parametric_function(radius, center, t):
    return radius * np.cos(t) - center[0], radius * np.sin(t) - center[1]


# Define the borders of the mesh
outer_radius = 1.0

Circle1 = Border(parametric_function=lambda t: circle_parametric_function(1, [0, 0], t), label=1,
                 t_start=0,
                 t_end=2.0 * np.pi)

Circle2 = Border(parametric_function=lambda t: circle_parametric_function(0.8, [0, 0], t), label=2,
                 t_start=0,
                 t_end=2.0 * np.pi)

Circle3 = Border(parametric_function=lambda t: circle_parametric_function(0.3, [0, 0], t), label=1,
                 t_start=0,
                 t_end=2.0 * np.pi)

# Generate a random mesh
random_mesh = RBFMesh(Circle1(100),Circle2(100),Circle3(-100))

# Generate points
num_points = 10000
random_mesh.generate_points(num_points)

# Plot the points
