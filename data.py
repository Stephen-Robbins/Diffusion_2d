"""Dataset generators for training toy diffusion models."""

from typing import Dict

import numpy as np
import torch

def generate_gaussian_mixture(
    n_samples: int,
    n_components: int = 8,
    dim: int = 2,
    std_scale: float = 0.1,
) -> torch.Tensor:
    """Generate a mixture of Gaussians arranged on a circle.

    Args:
        n_samples: Number of samples to draw.
        n_components: Number of Gaussian components.
        dim: Dimensionality of the data.
        std_scale: Maximum standard deviation for each component.

    Returns:
        Tensor containing the generated samples with shape ``(n_samples, dim)``.
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

def generate_rectangle_data(
    n_samples: int, width: float = 1, height: float = 4
) -> torch.Tensor:
    """Generate uniformly distributed points in a rectangle.

    Args:
        n_samples: Number of points to generate.
        width: Width of the rectangle.
        height: Height of the rectangle.

    Returns:
        Tensor of shape ``(n_samples, 2)`` containing the samples.
    """
    x = np.random.uniform(-width/2, width/2, n_samples)
    y = np.random.uniform(-height/2, height/2, n_samples)
    samples = np.column_stack((x, y))
    
    return torch.tensor(samples, dtype=torch.float32)

def generate_all_datasets(n_samples: int) -> Dict[str, torch.Tensor]:
    """Return all predefined toy datasets."""
    return {
        "Gaussian Mixture": generate_gaussian_mixture(n_samples),
        "Rectangle": generate_rectangle_data(n_samples),    }
