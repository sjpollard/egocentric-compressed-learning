import numpy as np
import tensorly as tl
import torch
import math

tl.set_backend('pytorch')

def random_bernoulli_matrix(shape):
    root_m = math.sqrt(shape[0])
    M = np.random.default_rng().choice([-1.0 / root_m, 1.0 / root_m], shape, p=[0.5, 0.5])
    return torch.from_numpy(M).float()

def random_gaussian_matrix(shape):
    root_m = math.sqrt(shape[0])
    mu = 0.0
    sigma = 1.0 / root_m
    M = torch.normal(mu, sigma, size=shape)
    return M

def mutual_coherence(M):
    M = M / np.linalg.norm(M, axis=0)
    G = np.abs(M.T @ M)
    np.fill_diagonal(G, 0)
    mu = np.max(G)
    return mu

def mutual_coherence_experiment():
    matrix_paths = ['P01_P02_bernoulli_158_158_2_3_20_v1',
                    'P01_P02_gaussian_158_158_2_3_20_v1',
                    'P01_P02_bernoulli_learnt_phi_theta_158_158_2_3_20_v1',
                    'P01_P02_gaussian_learnt_phi_theta_158_158_2_3_20_v1',
                    'P01_P02_bernoulli_112_112_2_3_20_v1',
                    'P01_P02_gaussian_112_112_2_3_20_v1',
                    'P01_P02_bernoulli_learnt_phi_theta_112_112_2_3_20_v1',
                    'P01_P02_gaussian_learnt_phi_theta_112_112_2_3_20_v1',
                    'P01_P02_bernoulli_71_71_2_3_20_v1',
                    'P01_P02_gaussian_71_71_2_3_20_v1',
                    'P01_P02_bernoulli_learnt_phi_theta_71_71_2_3_20_v1',
                    'P01_P02_gaussian_learnt_phi_theta_71_71_2_3_20_v1',
                    'P01_P02_bernoulli_22_22_2_3_20_v1',
                    'P01_P02_gaussian_22_22_2_3_20_v1',
                    'P01_P02_bernoulli_learnt_phi_theta_22_22_2_3_20_v1',
                    'P01_P02_gaussian_learnt_phi_theta_22_22_2_3_20_v1']
    matrices = list(map(lambda x: torch.load(f'checkpoints/{x}/phi_{x}.pt', map_location='cpu'), matrix_paths))
    for i in range(len(matrices)):
        print(f'{matrix_paths[i]} height {mutual_coherence(matrices[i][0].detach().numpy()):.4f} width {mutual_coherence(matrices[i][1].detach().numpy()):.4f}')

def process_batch(batch, phi_matrices, theta_matrices, modes):
    for i in range(batch.size(0)):
        compressed = tl.tenalg.multi_mode_dot(batch[i].clone(), phi_matrices, modes)
        batch[i] = tl.tenalg.multi_mode_dot(compressed, theta_matrices, modes, transpose=True)
