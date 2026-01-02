import os
import torch
import requests
import sys
import json
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split, Subset

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# --- Integrated Classes from dataloaders/ ---
class GaussianNoise:
    """Applies Gaussian noise to the input tensor."""
    def __init__(self, std=0.05, active=False):
        self.std = std
        self.active = active

    def __call__(self, x):
        if not self.active:
            return x
        # Generate noise with the same shape as the input x
        noise = torch.randn_like(x) * self.std
        return x + noise

class MLCupDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform # Store the transform (e.g., GaussianNoise)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]
        
        # Apply transform if it exists
        if self.transform:
            x_sample = self.transform(x_sample)
            
        return x_sample, y_sample

class MLCupDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True, **kwargs):
        super(MLCupDataLoader, self).__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            **kwargs
        )

# --- Configuration Loading ---

try:
    with open('./config/keras_nn.json') as keras_nn_config:
        CONFIG = json.load(keras_nn_config)
except FileNotFoundError:
    CONFIG = {"seed": 42}

# --- 2. Data Loading Functions ---

def get_monk1_data(batch_size, data_root='./data'):
    monk_dir = os.path.join(data_root, 'monk')
    os.makedirs(monk_dir, exist_ok=True)
    
    train_file = os.path.join(monk_dir, 'monks-1.train')
    test_file = os.path.join(monk_dir, 'monks-1.test')
    
    # Download if files don't exist
    if not os.path.exists(train_file):
        print("Downloading MONK-1 train data...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.train"
        r = requests.get(url)
        with open(train_file, 'w') as f:
            f.write(r.text)
            
    if not os.path.exists(test_file):
        print("Downloading MONK-1 test data...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test"
        r = requests.get(url)
        with open(test_file, 'w') as f:
            f.write(r.text)

    # One-hot encoding definitions for 6 attributes (Total: 17 features)
    attr_dims = [3, 3, 2, 3, 4, 2]
    
    def parse_monk_file(file_path):
        features = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                label = int(parts[0])
                attrs = [int(a) for a in parts[1:-1]] # last part is ID
                
                # One-hot encode features
                one_hot_features = []
                for i, attr_val in enumerate(attrs):
                    one_hot = torch.zeros(attr_dims[i])
                    one_hot[attr_val - 1] = 1.0 # Values are 1-based
                    one_hot_features.append(one_hot)
                
                features.append(torch.cat(one_hot_features))
                labels.append(label)
                
        return torch.stack(features), torch.tensor(labels, dtype=torch.long)

    print("Parsing MONK-1 data...")
    train_x, train_y = parse_monk_file(train_file)
    test_x, test_y = parse_monk_file(test_file)
    
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = 17
    output_size = 2
    
    return train_loader, test_loader, input_size, output_size

def get_ml_cup_data(batch_size, data_root='./data', validation_ratio=0.20, test_ratio=0.10, scaler: TransformerMixin=StandardScaler(), scale_target=True, num_workers=4):
    """
    Splits ML-CUP25-TR.csv into Train, Validation, and Internal Test.
    """
    # Check if a GPU is available
    use_pin_memory = torch.cuda.is_available()
    
    # data will be in ./MLC25/
    ml_cup_dir = os.path.join(data_root, 'MLC25')
    os.makedirs(ml_cup_dir, exist_ok=True)
    
    train_file = os.path.join(ml_cup_dir, 'ML-CUP25-TR.csv')

    if not os.path.exists(train_file):
        print("Downloading ML-CUP25-TR train data...")
        url = "https://gist.githubusercontent.com/FlavRomano/a19771d5c67f71dad557e5fa384db38b/raw/7290bff843b8a5c3a650457281c93c1d54e55f51/ML-CUP25-TR.csv"
        r = requests.get(url)
        with open(train_file, 'w') as f:
            f.write(r.text)
    
    if not os.path.exists(train_file):
        print(f"ERROR: ML-CUP dataset file not found.")
        sys.exit(1)

    # --- Parser for ML-CUP data ---
    def parse_ml_cup_file(file_path):
        features = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                # Check for sufficient columns (ID + 12 Inputs + 4 Targets = 17 columns)
                if len(parts) < 17: 
                    continue
                
                try:
                    # Read columns 1 to 12 (Indices 1 to 13) as Features
                    feature_values = [float(p) for p in parts[1:13]]
                    
                    # Read columns 13 to 16 as Targets
                    label_values = [float(p) for p in parts[13:17]]
                    
                    features.append(torch.tensor(feature_values, dtype=torch.float32))
                    labels.append(torch.tensor(label_values, dtype=torch.float32))
                except ValueError as e:
                    pass

        return torch.stack(features), torch.stack(labels)

    print("Parsing ML-CUP data (TR only)...")
    # 1. Parse Labeled Data (TR)
    full_x, full_y = parse_ml_cup_file(train_file)
    
    # --- Splitting Strategy ---
    # Split TR into (Train+Val) and Internal Test
    train_val_x, test_int_x, train_val_y, test_int_y = train_test_split(
        full_x, full_y, test_size=test_ratio, random_state=CONFIG["seed"]
    )
    
    # Split (Train+Val) into Train and Validation
    if validation_ratio > 0.0:
        adjusted_val_ratio = validation_ratio / (1.0 - test_ratio)
        train_x, val_x, train_y, val_y = train_test_split(
            train_val_x, train_val_y, test_size=adjusted_val_ratio, random_state=CONFIG["seed"]
        )
    else:
        train_x, train_y = train_val_x, train_val_y
        val_x = torch.empty((0, train_x.shape[1]), dtype=torch.float32)
        val_y = torch.empty((0, train_y.shape[1]), dtype=torch.float32)

    print(f"Data Split: Train={len(train_x)}, Val={len(val_x)}, Internal Test={len(test_int_x)}")

    # --- 1. Scale Inputs (X) ---
    if scaler:
        print(f"Applying scaling {scaler.__class__.__name__} to Inputs...")
        scaler.fit(train_x)
        train_x = torch.tensor(scaler.transform(train_x), dtype=torch.float32)
        if len(val_x) > 0:
            val_x = torch.tensor(scaler.transform(val_x), dtype=torch.float32)
        test_int_x = torch.tensor(scaler.transform(test_int_x), dtype=torch.float32)

    # --- 2. Scale Targets (Y) ---
    target_scaler = None
    if scale_target:
        print("Applying StandardScaler to Targets...")
        target_scaler = StandardScaler()
        
        # Fit ONLY on training targets
        target_scaler.fit(train_y.numpy())
        
        # Transform all sets
        train_y = torch.tensor(target_scaler.transform(train_y.numpy()), dtype=torch.float32)
        if len(val_y) > 0:
            val_y = torch.tensor(target_scaler.transform(val_y.numpy()), dtype=torch.float32)
        test_int_y = torch.tensor(target_scaler.transform(test_int_y.numpy()), dtype=torch.float32)
    
    # Create Datasets
    train_dataset = MLCupDataset(train_x, train_y)
    val_dataset = MLCupDataset(val_x, val_y)
    test_int_dataset = MLCupDataset(test_int_x, test_int_y)
    
    # Create Loaders
    train_loader = MLCupDataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader = MLCupDataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    internal_test_loader = MLCupDataLoader(dataset=test_int_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    
    input_size = 12
    output_size = 4
    
    # Returns: Train, Val, Internal_Test, metadata...
    return train_loader, val_loader, internal_test_loader, input_size, output_size, target_scaler

# --- 3. Helper Functions (Merged from data_loader_2.py) ---

def split_dataloader(dataloader, val_fraction=0.2, batch_size=32, seed=42):
    """
    Splits a PyTorch dataset into train and validation DataLoaders.
    """
    dataset = dataloader.dataset
    total_samples = len(dataset)
    n_val = int(total_samples * val_fraction)
    n_train = total_samples - n_val

    # reproducible split
    generator = torch.Generator().manual_seed(seed)

    train_subset, val_subset = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader

def cv_fold_split(dataset, train_idx, valid_idx, batch_size):
    """
    Creates train and validation DataLoaders for a specific Cross-Validation fold.
    """
    train_data = Subset(dataset, train_idx)
    valid_data = Subset(dataset, valid_idx)

    train_loader_cv = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader_cv = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader_cv, valid_loader_cv

def apply_pca_on_X(dataset, n_components, standardize=True):
    """
    Returns a new Dataset with PCA applied to dataset.X
    """
    # Ensure X is numpy for sklearn PCA
    X = dataset.X
    if isinstance(X, torch.Tensor):
        X = X.numpy()
        
    X = np.asarray(X)

    if standardize:
        X = StandardScaler().fit_transform(X)

    X_pca = PCA(n_components=n_components).fit_transform(X)

    # Create new dataset with transformed X
    new_dataset = deepcopy(dataset)
    new_dataset.X = torch.tensor(X_pca, dtype=torch.float32) # Convert back to Tensor
    return new_dataset