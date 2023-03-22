import numpy as np
import tensorly as tl
import torch
import math

tl.set_backend('pytorch')


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


def process_batch(batch, phi_matrices, theta_matrices, modes):
    for i in range(batch.size(0)):
        compressed = tl.tenalg.multi_mode_dot(batch[i].clone(), phi_matrices, modes)
        batch[i] = tl.tenalg.multi_mode_dot(compressed, theta_matrices, modes, transpose=True)
