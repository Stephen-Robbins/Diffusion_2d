# sampler.py
import torch
import numpy as np
from tqdm import tqdm

# Device selection
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Define beta and alpha schedules (same as in trainer)
bmin = 0.1
bmax = 20

def beta(t):
    return bmin + t * (bmax - bmin)

def alpha(t):
    x = bmin * t + ((bmax - bmin) * t**2) / 2
    return torch.exp(-x / 2)

def p(x, t):
    a = alpha(t).view(-1, 1)
    mu = x * a
    std = torch.sqrt(1 - a**2)
    return mu, std

def f(x, t):
    """
    Drift function f(x, t). (Placeholder: adjust as needed.)
    """
    return -0.5 * beta(t) * x

def g(t):
    """
    Diffusion coefficient g(t).
    """
    return torch.sqrt(beta(t))

def reverse_sde_sampling(score_net1, 
                         num_samples, 
                         num_diffusion_timesteps, 
                         device, 
                         x_shape=(2,), 
                         eps=1e-3):
    """
    Samples from the learned distribution using the reverse-time SDE in x-space,
    with the score defined as the average of two trained score networks:
    
        score(x, t) = 1/2 * (score_net1(x, t) + score_net2(x, t))
    
    The SDE is solved using Euler–Maruyama discretization.
    
    Arguments:
        score_net1: The first trained score model.
        score_net2: The second trained score model.
        num_samples: Number of samples to generate.
        num_diffusion_timesteps: Number of timesteps in the reverse SDE.
        device: Torch device.
        x_shape: The shape of each sample (default is (2,) for 2D data).
        eps: A small constant (unused here, but sometimes used for numerical stability).
    
    Returns:
        A tensor of generated samples in x-space.
    """
    # Initialize x from the prior (typically a standard normal)
    x = torch.randn((num_samples, *x_shape), device=device)
    dt = -1.0 / num_diffusion_timesteps  # negative because we reverse time
  
    with torch.no_grad():
        for i in tqdm(range(num_diffusion_timesteps - 1, -1, -1), desc="Sampling"):
            t = torch.full((num_samples, 1), i / num_diffusion_timesteps, device=device)
            
            score=score_net1(x, t)
            
            # Compute the drift: f(x, t) - g(t)^2 * (combined score)
            drift = f(x, t) - (g(t) ** 2) * score
            
            # Sample standard Gaussian noise
            z = torch.randn_like(x)
            
            # Euler–Maruyama update
            x = x + drift * dt + g(t) * np.sqrt(-dt) * z
               
        
    return x


