# model.py
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
    """A simple Multi-Layer Perceptron to estimate the score."""
    def __init__(self, input_dim=2, hidden_dim=256, time_dim=256):
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

    def forward(self, x, t):
        # Time embedding
        time_embedding = self.time_mlp(t)
        # Concatenate time embedding with input
        x_t = torch.cat([x, time_embedding], dim=-1)
        return self.net(x_t)
