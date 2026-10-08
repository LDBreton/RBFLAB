import numpy as np
from rbflab.meshgen import RBFMesh, Border


def upper_border_func(t):
    return t, 0.17735 * np.sqrt(t) - 0.075597 * t - 0.212836 * (t ** 2) + 0.17363 * (t ** 3) - 0.06254 * (t ** 4)


def lower_border_func(t):
    return t, -(0.17735 * np.sqrt(t) - 0.075597 * t - 0.212836 * (t ** 2) + 0.17363 * (t ** 3) - 0.06254 * (t ** 4))


def circular_border_func(t):
    return 0.8 * np.cos(t) + 0.5, 0.8 * np.sin(t)


# Creating borders
upper = Border(parametric_function=upper_border_func, label='upper', t_start=0, t_end=1)

# Here, notice the reversed t_start and t_end
lower = Border(parametric_function=lower_border_func, label='lower', t_start=1, t_end=0)

circle_s = Border(parametric_function=circular_border_func, label='circle', t_start=0, t_end=2 * np.pi)

random_mesh = RBFMesh(circle_s(300), upper(305), lower(305), abs_tol=1e-04)

num_points = 10000
random_mesh.generate_points(num_points)
