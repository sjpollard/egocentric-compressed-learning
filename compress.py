import numpy as np
import tensorly as tl
import torch
import math

tl.set_backend('pytorch')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

def random_gaussian_matrix(shape):
    m = shape[0]
    mu = 0.0
    sigma = 1.0 / m
    M = torch.normal(mu, sigma, size=shape)
    return M


def random_bernoulli_matrix(shape):
    root_m = math.sqrt(shape[0])
    M = np.random.default_rng().choice([-1.0 / root_m, 1.0 / root_m], shape, p=[0.5, 0.5])
    return torch.from_numpy(M).float()


def process_batch(batch, measurements, modes):
    for i in range(batch.size(0)):
        phi_matrices = list(map(lambda x, y: random_bernoulli_matrix((x, batch[i].size(y))).to(DEVICE), measurements, modes))
        compressed = tl.tenalg.multi_mode_dot(batch[i], phi_matrices, modes)
        batch[i] = tl.tenalg.multi_mode_dot(compressed, phi_matrices, modes, transpose=True)


def compress_tensor(X, phi_matrices, modes):
    return tl.tenalg.multi_mode_dot(X, phi_matrices, modes)


def expand_tensor(Y, theta_matrices, modes):
    return tl.tenalg.multi_mode_dot(Y, theta_matrices, modes)
