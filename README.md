# Diffusion 2D

A minimal playground for score-based diffusion models in two dimensions. The project provides simple PyTorch implementations for training a score network on toy datasets and sampling from the learned distribution via reverse-time stochastic differential equations.

## Key Idea

Diffusion models corrupt data with gradually increasing Gaussian noise and learn a "score" function that predicts the gradient of the log-density at each noise level. Sampling is achieved by running the diffusion process in reverse using the estimated score.

## Installation

```bash
pip install -e .
```

## Usage

```python
import torch
from diffusion_2d import data, model, trainer

# Generate a toy dataset
samples = data.generate_gaussian_mixture(10000)

# Instantiate model and optimizer
net = model.ScoreNet()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# Train
trainer.train_diffusion_model(samples, net, opt,
                              num_diffusion_timesteps=1000,
                              batch_size=128,
                              num_epochs=10,
                              device=torch.device('cpu'))
```

## Related Work

This repository is inspired by [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) by Song et al.

## Folder Guide

| Path | Description |
|------|-------------|
| `src/` | Python source code for datasets, models and training |
| `notebooks/` | Example Jupyter notebooks |
| `data/` | Small sample datasets |
| `checkpoints/` | Saved model checkpoints |

## Contact

For questions please contact `your.email@example.com`.
