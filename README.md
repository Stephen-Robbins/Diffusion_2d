# 2D Diffusion Models

Implementation of score-based diffusion models for 2D data distributions.

## Overview

This project implements diffusion models for learning and sampling from 2D data distributions. The model learns to reverse a diffusion process that gradually adds noise to data, allowing generation of new samples from the learned distribution.

## Architecture

- **Score Network**: MLP-based network that estimates the score function (gradient of log probability)
- **SDE Framework**: Variance Preserving (VP) stochastic differential equation
- **Sampler**: Implements both Euler-Maruyama and Predictor-Corrector sampling methods

## Files

- `model.py`: Neural network architecture for score estimation
- `sde.py`: Stochastic differential equation definitions
- `sampler.py`: Sampling algorithms for generation
- `trainer.py`: Training loop and loss functions
- `data.py`: 2D dataset generators (rectangles, circles, spirals, etc.)

## Notebooks

- `train_models.ipynb`: Training experiments on different 2D distributions
- `sample_models.ipynb`: Sampling and visualization of trained models

## Usage

```python
from trainer import Trainer
from data import get_rectangle_dataset

# Load data
data = get_rectangle_dataset(n_samples=10000)

# Train model
trainer = Trainer(data, n_epochs=1000)
trainer.train()

# Generate samples
samples = trainer.sample(n_samples=1000)
```

## Checkpoints

Trained model checkpoints are saved in `checkpoints/` directory.