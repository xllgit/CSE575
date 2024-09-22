import math

import numpy as np
import torch
from torch import sqrt, pow, cat, zeros, Tensor
from scipy.integrate import solve_ivp
from sklearn.neighbors import KernelDensity
from torch.distributions import Normal 

class ToyDataset:
    """Handles the generation of classification toy datasets"""
    def generate(self, n_samples:int, dataset_type:str, **kwargs):
        """Handles the generation of classification toy datasets
        :param n_samples: number of datasets points in the generated dataset
        :type n_samples: int
        :param dataset_type: {'moons', 'spirals', 'spheres', 'gaussians', 'gaussians_spiral', diffeqml'}
        :type dataset_type: str
        :param dim: if 'spheres': dimension of the spheres
        :type dim: float
        :param inner_radius: if 'spheres': radius of the inner sphere
        :type inner_radius: float
        :param outer_radius: if 'spheres': radius of the outer sphere
        :type outer_radius: float
        """
        if dataset_type == 'moons':
            return generate_moons(n_samples=n_samples, **kwargs)
        elif dataset_type == 'spirals':
            return generate_spirals(n_samples=n_samples, **kwargs)
        elif dataset_type == 'spheres':
            return generate_concentric_spheres(n_samples=n_samples, **kwargs)
        elif dataset_type == 'gaussians':
            return generate_gaussians(n_samples=n_samples, **kwargs)
        elif dataset_type == 'gaussians_spiral':
            return generate_gaussians_spiral(n_samples=n_samples, **kwargs)
        elif dataset_type == 'diffeqml':
            return generate_diffeqml(n_samples=n_samples, **kwargs)

def randnsphere(dim:int, radius:float) -> Tensor:
    """Uniform sampling on a sphere of `dim` and `radius`

    :param dim: dimension of the sphere
    :type dim: int
    :param radius: radius of the sphere
    :type radius: float
    """
    v = torch.randn(dim)
    inv_len = radius / sqrt(pow(v, 2).sum())
    return v * inv_len

def generate_gaussians(n_samples=100, n_gaussians=7, dim=2,
                       radius=0.5, std_gaussians=0.1, noise=1e-3): 
    """Creates `dim`-dimensional `n_gaussians` on a ring of radius `radius`.

    :param n_samples: number of datasets points in the generated dataset
    :type n_samples: int
    :param n_gaussians: number of gaussians distributions placed on the circle of radius `radius`
    :type n_gaussians: int
    :param dim: dimension of the dataset. The distributions are placed on the hyperplane (x1, x2, 0, 0..) if dim > 2
    :type dim: int
    :param radius: radius of the circle on which the distributions lie
    :type radius: int
    :param std_gaussians: standard deviation of the gaussians.
    :type std_gaussians: int
    :param noise: standard deviation of noise magnitude added to each datasets point
    :type noise: float
    """
    X = torch.zeros(n_samples * n_gaussians, dim) ; y = torch.zeros(n_samples * n_gaussians).long()
    angle = torch.zeros(1)
    if dim > 2: loc = torch.cat([radius*torch.cos(angle), radius*torch.sin(angle), torch.zeros(dim-2)])
    else: loc = torch.cat([radius*torch.cos(angle), radius*torch.sin(angle)])
    dist = Normal(loc, scale=std_gaussians)

    for i in range(n_gaussians):
        angle += 2*math.pi / n_gaussians
        if dim > 2: dist.loc = torch.Tensor([radius*torch.cos(angle), torch.sin(angle), radius*torch.zeros(dim-2)])
        else: dist.loc = torch.Tensor([radius*torch.cos(angle), radius*torch.sin(angle)])
        X[i*n_samples:(i+1)*n_samples] = dist.sample(sample_shape=(n_samples,)) + torch.randn(dim)*noise
        y[i*n_samples:(i+1)*n_samples] = i
    return X, y

def generate_concentric_spheres(n_samples:int=100, noise:float=1e-4, dim:int=3,
                                inner_radius:float=0.5, outer_radius:int=1): 
    """Creates a *concentric spheres* dataset of `n_samples` datasets points.

    :param n_samples: number of datasets points in the generated dataset
    :type n_samples: int
    :param noise: standard deviation of noise magnitude added to each datasets point
    :type noise: float
    :param dim: dimension of the spheres
    :type dim: float
    :param inner_radius: radius of the inner sphere
    :type inner_radius: float
    :param outer_radius: radius of the outer sphere
    :type outer_radius: float
    """
    X, y = zeros((n_samples, dim)), torch.zeros(n_samples)
    y[:n_samples // 2] = 1
    samples = []
    for i in range(n_samples // 2):
        samples.append(randnsphere(dim, inner_radius)[None, :])
    X[:n_samples // 2] = cat(samples)
    X[:n_samples // 2] += zeros((n_samples // 2, dim)).normal_(0, std=noise)
    samples = []
    for i in range(n_samples // 2):
        samples.append(randnsphere(dim, outer_radius)[None, :])
    X[n_samples // 2:] = cat(samples)
    X[n_samples // 2:] += zeros((n_samples // 2, dim)).normal_(0, std=noise)
    return X, y


def generate_spirals(n_samples=100, noise=1e-4, **kwargs):
    """Creates a *spirals* dataset of `n_samples` datasets points.

    :param n_samples: number of datasets points in the generated dataset
    :type n_samples: int
    :param noise: standard deviation of noise magnitude added to each datasets point
    :type noise: float
    """
    n = np.sqrt(np.random.rand(n_samples, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_samples, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_samples, 1) * noise
    X, y = (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
            np.hstack((np.zeros(n_samples), np.ones(n_samples))))
    X, y = torch.Tensor(X), torch.Tensor(y).long()
    return X, y

def sample_gaussian(n_samples=1<<10, device=torch.device('cpu')):
    X, y = ToyDataset().generate(n_samples, 'gaussians', n_gaussians=2, dim=2, noise=.05)
    return 2*X.to(device), y.long().to(device)

def sample_annuli(n_samples=1<<10, device=torch.device('cpu')):
    X, y = ToyDataset().generate(n_samples, 'spheres', dim=2, noise=.05)
    return 2*X.to(device), y.long().to(device)

def sample_spiral(n_samples=1<<10, device=torch.device('cpu')):
    X, y = ToyDataset().generate(n_samples, 'spirals', n_gaussians=2, dim=2, noise=.05)
    return 2*X.to(device), y.long().to(device)

def plot_scatter(ax, X, y):
	colors = ['blue', 'orange']
	ax.scatter(X[:,0], X[:,1], c=[colors[int(yi)] for yi in y], alpha=0.2, s=10.)
    

