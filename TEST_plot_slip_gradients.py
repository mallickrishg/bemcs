import matplotlib.pyplot as plt
import numpy as np

# Slip functions
def f_1(x, a):
    return (x / a) * (9 * (x / a) / 8 - 3 / 4)

def f_2(x, a):
    return (1 - 3 * (x / a) / 2) * (1 + 3 * (x / a) / 2)

def f_3(x, a):
    return (x / a) * (9 * (x / a) / 8 + 3 / 4)

# Slip gradient functions
def df_1_dx(x, a):
    return (9 * x) / (4 * a**2) - 3 / (4 * a)

def df_2_dx(x, a):
    return -(9 * x) / (2 * a**2)

def df_3_dx(x, a):
    return (9 * x) / (4 * a**2) + 3 / (4 * a)

# compute slip from 3qn coefficients
def get_slip(x, a, phi_1, phi_2, phi_3):
    return phi_1 * f_1(x, a) + phi_2 * f_2(x, a) + phi_3 * f_3(x, a)
# compute slipgradient from 3qn coefficients
def get_slipgradient(x, a, phi_1, phi_2, phi_3):
    return phi_1 * df_1_dx(x, a) + phi_2 * df_2_dx(x, a) + phi_3 * df_3_dx(x, a)

# Define element locations and sizes
x_a = -1.0
x_b = 0.0
x_c = 1.0
x_ab_centroid = 0.5 * (x_a + x_b)
x_bc_centroid = 0.5 * (x_b + x_c)
a = 0.5

# Test vector of weights for 3qn
quadratic_weights = np.array((0,1,0,1,0,1))

# evluation points
n_pts = 100
x_ab = np.linspace(x_a, x_b, n_pts)
x_bc = np.linspace(x_b, x_c, n_pts)

ab_slip = get_slip(x_ab - x_ab_centroid,a,quadratic_weights[0],quadratic_weights[1],quadratic_weights[2])
ab_slipgradient = get_slipgradient(x_ab - x_ab_centroid,a,quadratic_weights[0],quadratic_weights[1],quadratic_weights[2])

bc_slip = get_slip(x_bc - x_bc_centroid,a,quadratic_weights[3],quadratic_weights[4],quadratic_weights[5])
bc_slipgradient = get_slipgradient(x_bc - x_bc_centroid,a,quadratic_weights[3],quadratic_weights[4],quadratic_weights[5])

# plot figure
fig,axs = plt.subplots(2,1, figsize=(6,8))
axs[0].plot(x_ab, ab_slip, "-r", label="ab")
axs[0].plot(x_bc, bc_slip, "-b", label="bc")
axs[0].set_xlabel("x")
axs[0].set_ylabel("slip")
axs[0].grid(True)

axs[1].plot(x_ab, ab_slipgradient, "-r")
axs[1].plot(x_bc, bc_slipgradient, "-b")
axs[1].set_xlabel("x")
axs[1].set_ylabel("slip gradient")
axs[1].grid(True)

plt.show()