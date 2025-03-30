import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# ---------------------------
# 1. Data Acquisition and Preprocessing
# ---------------------------
# Load the CSV file. Ensure your CSV has "Date" and "Rate" columns.
df = pd.read_csv('fed_funds_rate.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Normalize the Fed funds rate for stable training.
scaler = MinMaxScaler()
df['Rate_norm'] = scaler.fit_transform(df[['Rate']])

# Define lengths: history window and forecast horizon.
history_length = 10    # number of past time steps as condition
forecast_length = 12   # number of future steps to forecast

# Create a dataset of (history, future) pairs using a sliding window.
class FedFundsForecastDataset(Dataset):
    def __init__(self, series, history_length, forecast_length):
        self.series = series
        self.history_length = history_length
        self.forecast_length = forecast_length
        
    def __len__(self):
        return len(self.series) - self.history_length - self.forecast_length + 1
    
    def __getitem__(self, idx):
        history = self.series[idx : idx + self.history_length]
        future = self.series[idx + self.history_length : idx + self.history_length + self.forecast_length]
        # Return both as tensors. Condition (history) and target (future) are 1D vectors.
        return (torch.tensor(history, dtype=torch.float32),
                torch.tensor(future, dtype=torch.float32))

data_series = df['Rate_norm'].values
dataset = FedFundsForecastDataset(data_series, history_length, forecast_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---------------------------
# 2. Define the Conditional Score Network
# ---------------------------
# The network will take as input:
#   - x: the target forecast sequence (of length forecast_length) undergoing diffusion
#   - t: the diffusion time (scalar per example)
#   - cond: the conditioning history (of length history_length)
#
# We embed t and cond separately then concatenate with x.
class ConditionalScoreNet(nn.Module):
    def __init__(self, target_dim=forecast_length, cond_dim=history_length, hidden_dim=256, time_dim=256):
        super().__init__()
        # Time embedding network
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # Conditioning (history) embedding network
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Main network: input is [target + time embedding + condition embedding]
        self.net = nn.Sequential(
            nn.Linear(target_dim + time_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim)
        )
    
    def forward(self, x, t, cond):
        # x: [batch_size, forecast_length]
        # t: [batch_size, 1]
        # cond: [batch_size, history_length]
        time_embed = self.time_mlp(t)
        cond_embed = self.cond_mlp(cond)
        # Concatenate x with time and condition embeddings
        x_input = torch.cat([x, time_embed, cond_embed], dim=-1)
        return self.net(x_input)

# ---------------------------
# 3. Set Up the Diffusion Components and Training Functions
# ---------------------------
# We'll use the VPSDE class from your sde.py. For this example, we assume the same diffusion dynamics.
from sde import VPSDE
sde_instance = VPSDE()

def score_matching_loss(score_net, x, t, cond):
    """
    Computes the score matching loss.
    x: [batch_size, forecast_length]
    t: [batch_size, 1]
    cond: [batch_size, history_length] conditioning information
    """
    # Get the mean and std from the forward diffusion process.
    mean, std = sde_instance.p(x, t)
    # Sample noise and diffuse the target.
    z = torch.randn_like(x)
    x_t = mean + std * z
    score = score_net(x_t, t, cond)
    loss = torch.mean(torch.sum((std * score + z) ** 2, dim=1))
    return loss

def train_diffusion_model(dataloader, score_net, optimizer, num_diffusion_timesteps, num_epochs, device):
    score_net.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        losses = []
        for history, future in dataloader:
            # Move tensors to device.
            cond = history.to(device)         # shape: [batch_size, history_length]
            target = future.to(device)          # shape: [batch_size, forecast_length]
            current_batch_size = target.size(0)
            t = torch.rand((current_batch_size, 1), device=device)
            loss = score_matching_loss(score_net, target, t, cond)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch+1}/{num_epochs} loss: {avg_loss:.4f}")
        epoch_losses.append(avg_loss)
    return epoch_losses

# ---------------------------
# 4. Instantiate the Model and Train
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalScoreNet(target_dim=forecast_length, cond_dim=history_length).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
num_diffusion_timesteps = 100  # Adjust as needed
train_diffusion_model(dataloader, model, optimizer, num_diffusion_timesteps, num_epochs, device)

# ---------------------------
# 5. Reverse SDE Sampling to Forecast the Next 12 Rates
# ---------------------------
def reverse_sde_sampling(score_net, cond, num_diffusion_timesteps, device, target_dim):
    """
    Samples a forecast from the learned model using reverse-time SDE.
    cond: [num_samples, history_length] conditioning history (last known values)
    """
    num_samples = cond.shape[0]
    # Initialize target from a standard normal.
    x = torch.randn((num_samples, target_dim), device=device)
    dt = -1.0 / num_diffusion_timesteps  # Negative for reverse integration
  
    with torch.no_grad():
        for i in tqdm(range(num_diffusion_timesteps - 1, -1, -1), desc="Sampling"):
            t = torch.full((num_samples, 1), i / num_diffusion_timesteps, device=device)
            score = score_net(x, t, cond)
            drift = sde_instance.f(x, t) - (sde_instance.g(t) ** 2) * score
            z = torch.randn_like(x)
            x = x + drift * dt + sde_instance.g(t) * np.sqrt(-dt) * z
    return x

# ---------------------------
# 6. Forecasting Example
# ---------------------------
# Let’s assume you want to forecast the next 12 rates for the latest available history.
# We extract the most recent history from the normalized data.
latest_history = torch.tensor(data_series[-history_length:], dtype=torch.float32).unsqueeze(0).to(device)
# Sample a forecast
forecast_samples = reverse_sde_sampling(model, latest_history, num_diffusion_timesteps, device, forecast_length)
forecast_samples = forecast_samples.cpu().numpy()

# Plot the forecasted trajectories.
for sample in forecast_samples:
    plt.plot(range(history_length, history_length + forecast_length), sample, alpha=0.7)
plt.xlabel("Time step")
plt.ylabel("Normalized Fed Funds Rate")
plt.title("Forecasted Next 12 Fed Target Rates")
plt.show()

# Optionally, inverse-transform the forecasts to the original scale.
forecast_original_scale = scaler.inverse_transform(forecast_samples)
print("Forecasted Fed Funds Rates (original scale):")
print(forecast_original_scale)
