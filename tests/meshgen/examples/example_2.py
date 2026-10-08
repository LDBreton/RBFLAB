from rbflab.meshgen import RBFMesh, Border

# Define each border using lambda functions as in your provided example:
C01 = Border(parametric_function=lambda t: (0, -1 + t), label='upper', t_start=0, t_end=1)
C02 = Border(parametric_function=lambda t: (1.5 - 1.5 * t, -1), label='upper', t_start=0, t_end=1)
C03 = Border(parametric_function=lambda t: (1.5, -t), label='upper', t_start=0, t_end=1)
C04 = Border(parametric_function=lambda t: (1 + 0.5 * t, 0), label='others', t_start=0, t_end=1)
C05 = Border(parametric_function=lambda t: (0.5 + 0.5 * t, 0), label='others', t_start=0, t_end=1)
C06 = Border(parametric_function=lambda t: (0.5 * t, 0), label='others', t_start=0, t_end=1)
C11 = Border(parametric_function=lambda t: (0.5, -0.5 * t), label='inner', t_start=0, t_end=1)
C12 = Border(parametric_function=lambda t: (0.5 + 0.5 * t, -0.5), label='inner', t_start=0, t_end=1)
C13 = Border(parametric_function=lambda t: (1, -0.5 + 0.5 * t), label='inner', t_start=0, t_end=1)

# List all borders for possible use or inspection
n = 100
random_mesh = RBFMesh(C01(-n), C02(-n), C03(-n), C04(-n), C05(-n), C06(-n), C11(n), C12(n), C13(n))

num_points = 10000
random_mesh.generate_points(num_points)
