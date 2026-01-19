import torch
import torch.nn as nn
import numpy as np
from models import StandardFeedForwardNet
# from models.ensemble import ModelWithHead, ReadoutAdapter # Uncomment only if you actually have this file

def build_model(params, input_size, output_size, device, is_regression=True):
    """
    Constructs a model based on a parameter dictionary.
    """
    hidden_sizes = params['hidden_sizes']
    activation = params['activation']
    dropout = params.get('dropout', 0.0)  # Changed key to match Optuna script ('dropout' vs 'dropout_rate')
    
    # --- FIX 1: Pass dropout to the constructor ---
    model = StandardFeedForwardNet(
        input_size=input_size, 
        hidden_sizes=hidden_sizes, 
        output_size=output_size, 
        activation=activation, 
        dropout=dropout
    )
    
    return model.to(device)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Optional: Add gradient clipping here for safety
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # --- FIX 3: Weighted average for accurate epoch loss ---
        batch_size = data.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        
    return running_loss / total_samples

def evaluate_mee(model, loader, device, target_scaler=None):
    """
    Evaluates the model using Mean Euclidean Error (MEE).
    Always calculates on the ORIGINAL scale (Real MEE).
    """
    model.eval()
    total_mee = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Move to CPU/Numpy for calculation
            out_np = output.cpu().numpy()
            tgt_np = target.cpu().numpy()
            
            if target_scaler is not None:
                out_real = target_scaler.inverse_transform(out_np)
                tgt_real = target_scaler.inverse_transform(tgt_np)
            else:
                out_real = out_np
                tgt_real = tgt_np
                
            # Calculate Euclidean distance for each sample
            diff = out_real - tgt_real
            euclidean_dists = (diff**2).sum(axis=1)**0.5
            
            total_mee += euclidean_dists.sum()
            total_samples += data.size(0)
            
    return total_mee / total_samples

def evaluate(model, loader, criterion, device, target_scaler=None):
    """
    Evaluates the model. 
    If target_scaler is provided, calculates metrics on the ORIGINAL scale (Real MSE).
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # --- FIX 2: Handle Unscaling efficiently (Pure Numpy) ---
            if target_scaler is not None:
                # Move to CPU/Numpy
                out_np = output.cpu().numpy()
                tgt_np = target.cpu().numpy()
                
                # Inverse Transform to Real Scale
                out_real = target_scaler.inverse_transform(out_np)
                tgt_real = target_scaler.inverse_transform(tgt_np)
                
                # Calculate Squared Error in Numpy (Avoid moving back to GPU)
                # This assumes criterion is MSE. If it's something else, this logic needs adapting.
                batch_loss = np.mean((out_real - tgt_real) ** 2)
            else:
                # Standard PyTorch loss on scaled data
                batch_loss = criterion(output, target).item()

            # --- FIX 3: Weighted average ---
            batch_size = data.size(0)
            running_loss += batch_loss * batch_size
            total_samples += batch_size
            
    return running_loss / total_samples