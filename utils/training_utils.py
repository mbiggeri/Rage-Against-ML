import torch
import torch.nn as nn
import torch.optim as optim
from models import StandardFeedForwardNet
from models.ensemble import ModelWithHead, ReadoutAdapter

def build_model(params, input_size, output_size, device, is_regression=True):
    """
    Constructs a model based on a parameter dictionary.
    """
    hidden_sizes = params['hidden_sizes']
    activation = params['activation']
    model_type = params.get('model', 'standard') # Default to standard if not specified
    dropout = params.get('dropout_rate', 0.0)    # Support dropout if added later

    if model_type == 'standard':
        base = StandardFeedForwardNet(input_size, hidden_sizes, output_size, activation)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Wrap in ModelWithHead for consistency (useful for Ensembles later)
    # For regression, readout is Identity. For classification, it's a Linear layer if needed.
    mode = 'regression' if is_regression else 'classification'
    model = ModelWithHead(base, ReadoutAdapter(output_size, output_size, mode))
    
    return model.to(device)

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch and returns the average loss.
    """
    model.train()
    total_loss = 0.0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

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
    If target_scaler is None, calculates metrics on the SCALED scale (Loss).
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            if target_scaler is not None:
                # 1. Move to CPU to use sklearn
                out_np = output.cpu().numpy()
                tgt_np = target.cpu().numpy()
                
                # 2. Inverse Transform
                out_real = target_scaler.inverse_transform(out_np)
                tgt_real = target_scaler.inverse_transform(tgt_np)
                
                # 3. Convert back to tensor for the criterion (or use numpy directly)
                # Using tensor ensures compatibility if criterion is a torch function
                output = torch.tensor(out_real, device=device)
                target = torch.tensor(tgt_real, device=device)

            loss = criterion(output, target)
            total_loss += loss.item()
            
    return total_loss / len(loader)