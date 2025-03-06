import torch
import numpy as np
from tqdm.auto import tqdm  # Import tqdm for progress bars
import os
from sde import VPSDE

# Set up the device (MPS, CUDA, or CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

sde=VPSDE()



def score_matching_loss(score_net1,  x,  t):
    """
    Returns:
        The score matching loss.
    """
    # Get mean and standard deviation for the forward diffusion on each input
    mean, std = sde.p(x, t)

   
    # Sample the noise z (shared between the two processes)
    z = torch.randn_like(x)

    # Diffuse x and y with the same noise
    x_t = mean + std * z

    score_1 =  score_net1(x_t, t)
    
    # Combine the losses
    loss = torch.mean(torch.sum((std*score_1 + z)**2, dim=1))
    
    return loss

def train_diffusion_model(data_x, 
                              score_net1, 
                              optimizer, 
                              num_diffusion_timesteps,
                              batch_size, 
                              num_epochs, 
                              device, 
                              checkpoint_path=None,
                              save_every=100):
    """

    """
    # Set both networks to training mode
    score_net1.train()
   

    epoch_losses = []

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")
    
    # For each epoch...
    for epoch in epoch_pbar:
        losses = []
        # Shuffle indices (assuming data_x and data_y have the same length)
        perm = torch.randperm(len(data_x))
        
        # Iterate over mini-batches (without a separate progress bar)
        for i in range(0, len(data_x), batch_size):
            # Get indices for current batch
            end_idx = min(i + batch_size, len(data_x))
            idx = perm[i:end_idx]
            
            batch_x = data_x[idx].to(device)
        
            current_batch_size = batch_x.shape[0]
            
            # Sample a random time t for each example in the batch
            t = torch.rand((current_batch_size, 1), device=device)
            
            # Compute the loss using the configured loss functions
            loss = score_matching_loss(score_net1, batch_x, t)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record the loss
            losses.append(loss.item())
        
        # Calculate and log the average loss for this epoch
        avg_loss = np.mean(losses)
        epoch_losses.append(avg_loss)
        
        # Update progress bar with the current loss
        epoch_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        
        # Optionally, save a checkpoint
        if checkpoint_path is not None and ((epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs):
            # Create directory if it doesn't exist
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir and not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            torch.save({
                'score_net1_state_dict': score_net1.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    return epoch_losses
