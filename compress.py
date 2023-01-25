import numpy as np
import tensorly.tenalg as t


def random_gaussian_matrix(shape):

    mu = 0.0
    sigma = 1.0 / shape[0]
    M = np.random.default_rng().normal(mu, sigma, shape)

    return M


def main():

    """ X = np.array([[[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]],
                  [[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]],
                  [[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]]]) """

    X = np.random.default_rng().integers(0, 255, (3, 50, 50))

    print(X)
    
    M1 = random_gaussian_matrix((3, 3))
    M2 = random_gaussian_matrix((25, 50))
    M3 = random_gaussian_matrix((25, 50))

    Y = t.multi_mode_dot(X, [M1, M2, M3], [0, 1, 2])

    print(Y)

    X_hat = t.multi_mode_dot(Y, [M1.T, M2.T, M3.T], [0, 1, 2])

    print(X_hat)


if __name__ == "__main__":
    main()
