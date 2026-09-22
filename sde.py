"""Variance-preserving SDE utilities."""

import torch

class VPSDE:
    """Simple variance preserving SDE."""

    def __init__(self, bmin: float = 0.1, bmax: float = 20) -> None:
        self.bmin = bmin
        self.bmax = bmax

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Linear schedule for beta."""
        return self.bmin + t * (self.bmax - self.bmin)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Exponentially decaying alpha coefficient."""
        x = self.bmin * t + ((self.bmax - self.bmin) * t**2) / 2
        return torch.exp(-x / 2)

    def p(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean and std of the forward process."""
        a = self.alpha(t).view(-1, 1)
        mu = x * a
        std = torch.sqrt(1 - a**2)
        return mu, std

    def f(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Drift function for the reverse-time SDE."""
        return -0.5 * self.beta(t) * x
    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient."""
        return torch.sqrt(self.beta(t))
