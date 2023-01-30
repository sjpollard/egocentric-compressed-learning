import numpy as np
import tensorly as tl


def random_gaussian_matrix(shape):

    mu = 0.0
    sigma = 1.0 / shape[0]
    M = np.random.default_rng().normal(mu, sigma, shape)

    return M


def compress_tensor(X, phi_matrices, modes):
    return tl.tenalg.multi_mode_dot(X, phi_matrices, modes)


def expand_tensor(Y, theta_matrices, modes):
    return tl.tenalg.multi_mode_dot(Y, theta_matrices, modes)
