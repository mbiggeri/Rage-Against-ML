import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import argparse
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from utils.data_loader import MLCupDataset, GaussianNoise

# --- CONTROL FLAGS ---
USE_KFOLD = True    
K_FOLDS = 5
USE_NOISE = False    
NOISE_STD = 0.05     

# Ensure local modules can be found
sys.path.append(os.getcwd())

try:
    from models.standard import StandardFeedForwardNet
except ImportError:
    from models import StandardFeedForwardNet

from utils.data_loader import get_ml_cup_data
import training_utils

# --- CONFIGURATION ---
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading RAW data on {DEVICE} for dynamic scaling...")

# 1. LOAD RAW DATA
train_loader_raw, val_loader_raw, internal_test_raw, INPUT_SIZE, OUTPUT_SIZE, _ = get_ml_cup_data(
    BATCH_SIZE, validation_ratio=0.15, test_ratio=0.10, scaler=None, scale_target=False, num_workers=0
)

# 2. PRE-PROCESSING & PCA LOGIC
from sklearn.decomposition import PCA

N_COMPONENTS = 0

def extract_numpy(loader):
    return loader.dataset.X.numpy(), loader.dataset.y.numpy()

X_train_np, y_train_np = extract_numpy(train_loader_raw)
X_val_np, y_val_np     = extract_numpy(val_loader_raw)

if N_COMPONENTS > 0:
    print(f"--- PCA ENABLED ({N_COMPONENTS} comps) ---")
    print("Forcing StandardScaler before PCA...")
    std_scaler = StandardScaler()
    X_train_np = std_scaler.fit_transform(X_train_np)
    X_val_np   = std_scaler.transform(X_val_np)
    
    pca = PCA(n_components=N_COMPONENTS)
    X_train_np = pca.fit_transform(X_train_np)
    X_val_np   = pca.transform(X_val_np)
    
    INPUT_SIZE = N_COMPONENTS
    print(f"New Input Size: {INPUT_SIZE}")
else:
    print("--- PCA DISABLED ---")

def objective(trial, epochs=100):
    # ==========================================
    # 1. HYPERPARAMETERS
    # ==========================================
    if N_COMPONENTS > 0:
        scaler_type = trial.suggest_categorical("scaler_post_pca", ["none", "minmax"])
    else:
        scaler_type = trial.suggest_categorical("scaler", ["standard", "minmax", "robust", "none"])

    def get_scaler(s_type):
        if s_type == "standard": return StandardScaler()
        elif s_type == "minmax": return MinMaxScaler()
        elif s_type == "robust": return RobustScaler()
        return None

    use_target_scaling = trial.suggest_categorical("scale_target", [True, False])

    noise_transform = GaussianNoise(std=NOISE_STD, active=USE_NOISE)

    n_layers = trial.suggest_int("n_layers", 1, 2)
    hidden_size = trial.suggest_int("hidden_size", 4, 32, log=True)
    hidden_sizes = [hidden_size] * n_layers
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    activation = trial.suggest_categorical("activation", ["relu", "gelu", "silu", "mish"])
    
    # --- CRITICAL FIX: Safe Learning Rates ---
    if not use_target_scaling:
        # If targets are raw (large values), LR must be tiny to prevent explosion
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True) 
    else:
        # If targets are scaled (approx 0-1), standard LRs work
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)

    # Helper to create and train a model for one split
    def train_and_evaluate(X_t, y_t, X_v, y_v, trial_obj=None):
        # 1. Apply Input Scaler
        scaler = get_scaler(scaler_type)
        if scaler:
            X_t_in = scaler.fit_transform(X_t)
            X_v_in = scaler.transform(X_v)
        else:
            X_t_in, X_v_in = X_t, X_v

        # 2. Apply Target Scaler
        t_scaler = None
        if use_target_scaling:
            t_scaler = StandardScaler()
            y_t_scaled = t_scaler.fit_transform(y_t)
            y_v_scaled = t_scaler.transform(y_v)
        else:
            y_t_scaled, y_v_scaled = y_t, y_v

        # 3. Create Tensors
        t_X_train = torch.tensor(X_t_in, dtype=torch.float32)
        t_y_train = torch.tensor(y_t_scaled, dtype=torch.float32)
        t_X_val = torch.tensor(X_v_in, dtype=torch.float32)
        t_y_val = torch.tensor(y_v_scaled, dtype=torch.float32)

        # 4. Create Datasets
        train_ds = MLCupDataset(t_X_train, t_y_train, transform=noise_transform)
        val_ds = MLCupDataset(t_X_val, t_y_val, transform=None)
        
        loader_t = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        loader_v = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # 5. Build Model
        model = StandardFeedForwardNet(
            input_size=INPUT_SIZE,
            hidden_sizes=hidden_sizes,
            output_size=OUTPUT_SIZE,
            activation=activation,
            dropout=dropout
        ).to(DEVICE)

        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        criterion = nn.MSELoss()

        # 6. Training Loop
        final_val_mse = 1e9 # Default high value

        for epoch in range(epochs):
            model.train()
            for data, target in loader_t:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                out = model(data)
                loss = criterion(out, target)
                
                # --- SAFEGUARD: Check for NaN/Inf Loss ---
                if torch.isnan(loss) or torch.isinf(loss):
                    return 1e9 # Return a finite failing score

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Evaluate
            val_mse = training_utils.evaluate(model, loader_v, criterion, DEVICE, target_scaler=t_scaler)
            
            # --- SAFEGUARD: Check for NaN/Inf Validation ---
            if np.isnan(val_mse) or np.isinf(val_mse):
                return 1e9

            scheduler.step(val_mse)
            final_val_mse = val_mse
            
            # Pruning (Only for non-CV to save time)
            if not USE_KFOLD and trial_obj:
                trial_obj.report(val_mse, epoch)
                if trial_obj.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return final_val_mse

    # ==========================================
    # 2. EXECUTION (K-Fold vs Hold-Out)
    # ==========================================
    if USE_KFOLD:
        X_dev = np.concatenate([X_train_np, X_val_np], axis=0)
        y_dev = np.concatenate([y_train_np, y_val_np], axis=0)
        
        kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        scores = []
        
        for i, (train_idx, val_idx) in enumerate(kf.split(X_dev)):
            X_f_train, X_f_val = X_dev[train_idx], X_dev[val_idx]
            y_f_train, y_f_val = y_dev[train_idx], y_dev[val_idx]
            
            score = train_and_evaluate(X_f_train, y_f_train, X_f_val, y_f_val)
            scores.append(score)
            
        # Robust Mean: If any score is huge (failed), the mean will be huge, but NOT NaN.
        return np.mean(scores)
        
    else:
        score = train_and_evaluate(X_train_np, y_train_np, X_val_np, y_val_np, trial_obj=trial)
        return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna Hyperparameter Search")
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    print(f"Starting Search: Trials={args.n_trials}, Epochs={args.epochs}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    db_url = "sqlite:///optuna_mlcup_nn.db"
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
        storage.engine.dispose()