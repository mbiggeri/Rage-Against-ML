import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import argparse

# Ensure local modules can be found
sys.path.append(os.getcwd())

from utils.data_loader import get_ml_cup_data
from models.standard import StandardFeedForwardNet
from models.ensemble import ModelWithHead, ReadoutAdapter
import training_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- CONFIGURATION ---
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading data on {DEVICE}...")
train_loader, val_loader, test_loader, blind_test_loader, INPUT_SIZE, OUTPUT_SIZE, target_scaler = get_ml_cup_data(
    BATCH_SIZE, 
    validation_ratio=0.20,
    scale_target=True,
    num_workers=0 
)

def objective(trial, epochs=100):
    # --- 1. SEARCH SPACE (Shallow & Wide Regime) ---
    
    # Shallower: Only 1 to 3 hidden layers
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    # Largest: Very wide layers (up to 2048 neurons)
    hidden_size = trial.suggest_int("hidden_size", 4, 2048, log=True)
    
    # Construct hidden_sizes list (Rectangular MLP)
    hidden_sizes = [hidden_size] * n_layers
    
    # Regularization is crucial for wide nets
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Activation
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu", "mish"])
    
    # Optimization
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # --- 2. BUILD MODEL ---
    base = StandardFeedForwardNet(
        input_size=INPUT_SIZE,
        hidden_sizes=hidden_sizes,
        output_size=OUTPUT_SIZE,
        activation=activation,
        dropout=dropout
    )
    
    model = ModelWithHead(base, ReadoutAdapter(OUTPUT_SIZE, OUTPUT_SIZE, 'regression')).to(DEVICE)
    
    # --- 3. OPTIMIZER & SCHEDULER ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    # --- 4. TRAINING LOOP ---
    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
        # Validation
        val_mse = training_utils.evaluate(model, val_loader, criterion, DEVICE, target_scaler=target_scaler)
        
        # Update Scheduler
        scheduler.step(val_mse)
        
        # Pruning
        trial.report(val_mse, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Search")
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    print(f"Starting Wide-MLP Search: Trials={args.n_trials}, Epochs={args.epochs}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # --- DB LOCK FIX: Create Storage Explicitly ---
    db_url = "sqlite:///optuna.db"
    storage = optuna.storages.RDBStorage(url=db_url)

    try:
        study = optuna.create_study(
            study_name="mlcup_search",
            storage=storage,
            load_if_exists=True,
            direction="minimize"
        )
        
        study.optimize(lambda trial: objective(trial, epochs=args.epochs), n_trials=args.n_trials)
        print("Search complete.")
        
    finally:
        # --- DB LOCK FIX: Dispose Engine ---
        print("Releasing database lock...")
        storage.engine.dispose()