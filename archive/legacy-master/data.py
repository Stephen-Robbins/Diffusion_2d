# data.py
import numpy as np
import torch

def generate_gaussian_mixture(n_samples, n_components=8, dim=2, std_scale=0.1):
    """
    Generates a mixture of Gaussians on a circle around the origin.
    """
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    means = 1.5*np.array([[np.cos(a), np.sin(a)] for a in angles])
    stds = np.random.uniform(0.1, std_scale, size=(n_components, dim))

    component_choices = np.random.choice(n_components, size=n_samples)
    samples = np.empty((n_samples, dim))

    for i in range(n_components):
        mask = (component_choices == i)
        samples[mask] = np.random.normal(means[i], stds[i], size=(mask.sum(), dim))

    return torch.tensor(samples, dtype=torch.float32)

def generate_rectangle_data(n_samples, width=1, height=4):
    """
    Generates a rectangle of uniformly distributed data.
    """
    x = np.random.uniform(-width/2, width/2, n_samples)
    y = np.random.uniform(-height/2, height/2, n_samples)
    samples = np.column_stack((x, y))
    
    return torch.tensor(samples, dtype=torch.float32)

def generate_all_datasets(n_samples):
    return {
        "Gaussian Mixture": generate_gaussian_mixture(n_samples),
        "Rectangle": generate_rectangle_data(n_samples),

    }