"""Simple score network used for 2D diffusion experiments."""

import torch
import torch.nn as nn

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class ScoreNet(nn.Module):
    """Multi-layer perceptron estimating the score of noisy data."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 256, time_dim: int = 256):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the network at data ``x`` and time ``t``."""

        time_embedding = self.time_mlp(t)
        x_t = torch.cat([x, time_embedding], dim=-1)
        return self.net(x_t)
