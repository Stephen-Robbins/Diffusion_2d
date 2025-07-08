# sampler.py
"""Utilities for sampling from a trained diffusion model."""

import torch
import numpy as np
from tqdm import tqdm
from sde import VPSDE
# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

sde=VPSDE()
def reverse_sde_sampling(
    score_net1: torch.nn.Module,
    num_samples: int,
    num_diffusion_timesteps: int,
    device: torch.device,
    x_shape: tuple[int, ...] = (2,),
) -> torch.Tensor:
    """Sample from the model by integrating the reverse SDE.

    Args:
        score_net1: Trained score network.
        num_samples: Number of points to generate.
        num_diffusion_timesteps: Number of discretization steps.
        device: Execution device.
        x_shape: Shape of each generated sample.

    Returns:
        Tensor of generated samples in data space.
    """
    # Initialize x from the prior (typically a standard normal)
    x = torch.randn((num_samples, *x_shape), device=device)
    dt = -1.0 / num_diffusion_timesteps  # negative because we reverse time
  
    with torch.no_grad():
        for i in tqdm(range(num_diffusion_timesteps - 1, -1, -1), desc="Sampling"):
            t = torch.full((num_samples, 1), i / num_diffusion_timesteps, device=device)
            
            score=score_net1(x, t)
            
            # Compute the drift: f(x, t) - g(t)^2 * (combined score)
            drift = sde.f(x, t) - (sde.g(t) ** 2) * score
            
            # Sample standard Gaussian noise
            z = torch.randn_like(x)
            
            # Euler–Maruyama update
            x = x + drift * dt + sde.g(t) * np.sqrt(-dt) * z
               
        
    return x


