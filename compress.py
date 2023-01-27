import numpy as np
import tensorly.tenalg as t


def random_gaussian_matrix(shape):

    mu = 0.0
    sigma = 1.0 / shape[0]
    M = np.random.default_rng().normal(mu, sigma, shape)

    return M

def compressed_features(X, phi_matrices, theta_matrices, modes):
    Y = t.multi_mode_dot(X, phi_matrices, modes)
    return t.multi_mode_dot(Y, theta_matrices, modes)

def main():

    X = np.random.default_rng().integers(0, 255, (3, 10, 5, 5))

    M1 = random_gaussian_matrix((1, 3))
    M2 = random_gaussian_matrix((5, 10))
    M3 = random_gaussian_matrix((3, 5))
    M4 = random_gaussian_matrix((3, 5))

    X_hat = compressed_features(X, [M1, M2, M3, M4], [M1.T, M2.T, M3.T, M4.T], [0, 1, 2, 3])

if __name__ == "__main__":
    main()
