import numpy as np
import bemcs


def get_kernels_linforce(x_obs, y_obs, els, connect_matrix, mu=1):

    n_obs = len(x_obs)
    n_els = len(connect_matrix[:, 0])

    kernel_u = np.zeros((n_obs, n_els))
    kernel_sx = np.zeros((n_obs, n_els))
    kernel_sy = np.zeros((n_obs, n_els))

    # provide coefficients for forces
    lincoefs = np.array([[1.0, 0.0], [0.0, 1.0]])

    for i in range(0, n_els):
        # define new els for GF calculation
        els_mod = bemcs.initialize_els()
        els_mod.x1 = els.x1[connect_matrix[i, :].astype(int)]
        els_mod.y1 = els.y1[connect_matrix[i, :].astype(int)]
        els_mod.x2 = els.x2[connect_matrix[i, :].astype(int)]
        els_mod.y2 = els.y2[connect_matrix[i, :].astype(int)]
        bemcs.standardize_els_geometry(els_mod, reorder=False)

        K_sx, K_sy, K_u = bemcs.get_displacement_stress_kernel_force_antiplane(
            x_obs, y_obs, els_mod, mu
        )

        # compute displacements and stress components
        kernel_u[:, i] = np.tensordot(K_u, lincoefs, axes=([2, 1], [0, 1]))
        kernel_sx[:, i] = np.tensordot(K_sx, lincoefs, axes=([2, 1], [0, 1]))
        kernel_sy[:, i] = np.tensordot(K_sy, lincoefs, axes=([2, 1], [0, 1]))

    return kernel_sx, kernel_sy, kernel_u
