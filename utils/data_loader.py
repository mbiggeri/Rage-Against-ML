import os
import torch
import requests
import sys
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from dataloaders import MLCupDataLoader, MLCupDataset

import numpy as np
from copy import deepcopy
from sklearn.decomposition import PCA

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin

import json

with open('./config/keras_nn.json') as keras_nn_config:
    CONFIG = json.load(keras_nn_config)
    print("config loaded")

# --- 2. Data Loading ---

# --- Funzione Helper per evitare duplicazione di codice ---
def load_monk(dataset_id, batch_size, data_root='./data'):
    """
    Generic MONK dataset loader for MONK-1, MONK-2, MONK-3.
    """
    monk_dir = os.path.join(data_root, 'monk')
    os.makedirs(monk_dir, exist_ok=True)
    
    filename_train = f'monks-{dataset_id}.train'
    filename_test = f'monks-{dataset_id}.test'
    
    train_file = os.path.join(monk_dir, filename_train)
    test_file = os.path.join(monk_dir, filename_test)
    
    # Base URL UCI Repository
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/"
    
    # Download if files don't exist
    if not os.path.exists(train_file):
        print(f"Downloading MONK-{dataset_id} train data...")
        r = requests.get(base_url + filename_train)
        with open(train_file, 'w') as f:
            f.write(r.text)
            
    if not os.path.exists(test_file):
        print(f"Downloading MONK-{dataset_id} test data...")
        r = requests.get(base_url + filename_test)
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

    print(f"Parsing MONK-{dataset_id} data...")
    train_x, train_y = parse_monk_file(train_file)
    test_x, test_y = parse_monk_file(test_file)
    
    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = 17
    output_size = 2
    
    return train_loader, test_loader, input_size, output_size



def get_ml_cup_data(batch_size, data_root='./data', test_ratio=0.20, mps=False, scaler: TransformerMixin=None) -> tuple[MLCupDataLoader, MLCupDataLoader]:
    # data will be in ./MLC25/
    ml_cup_dir = os.path.join(data_root, 'MLC25')
    os.makedirs(ml_cup_dir, exist_ok=True)
    
    train_file = os.path.join(ml_cup_dir, 'ML-CUP25-TR.csv')
    test_file = os.path.join(ml_cup_dir, 'ML-CUP25-TS.csv')

    if not os.path.exists(train_file):
        print("Downloading ML-CUP25-TR train data...")
        url = "https://gist.githubusercontent.com/FlavRomano/a19771d5c67f71dad557e5fa384db38b/raw/7290bff843b8a5c3a650457281c93c1d54e55f51/ML-CUP25-TR.csv"
        r = requests.get(url)
        with open(train_file, 'w') as f:
            f.write(r.text)
            
    if not os.path.exists(test_file):
        print("Downloading ML-CUP25-TS test data...")
        url = "https://gist.githubusercontent.com/FlavRomano/453dc2affc584028cb122d6b52cec295/raw/1cb1e84b26f8efd2ac081701d610c94498f988e1/ML-CUP25-TS.csv"
        r = requests.get(url)
        with open(test_file, 'w') as f:
            f.write(r.text)
    
    # --- Check if files exist ---
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("---" * 20)
        print(f"ERROR: ML-CUP dataset files not found.")
        print(f"This script cannot download the ML-CUP dataset automatically.")
        print(f"Please manually place your dataset files at these locations:")
        print(f"Training data: {os.path.abspath(train_file)}")
        print(f"Test data:     {os.path.abspath(test_file)}")
        print("---" * 20)
        sys.exit(1) # Stop the script

    N_INPUTS = 12
    N_TARGETS = 4

    ## Training set: ID, INPUTS, TARGET_1, TARGET_2, TARGET_3, TARGET_4 (last 4 columns)
    columns = (
        ["ID"] +
        [f"INPUT_{i}" for i in range(N_INPUTS)] +
        [f"TARGET_{i}" for i in range(N_TARGETS)]
    )
    ml_cup_tr = pd.read_csv("./data/MLC25/ML-CUP25-TR.csv", skiprows=7, names=columns)
    train_x = ml_cup_tr[[f"INPUT_{i}" for i in range(N_INPUTS)]].values
    train_y = ml_cup_tr[[f"TARGET_{i}" for i in range(N_TARGETS)]].values
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=test_ratio, random_state=CONFIG["seed"]) 

    if scaler:
        print(f"applying scaling {scaler.__class__.__name__} on train_X and test_X")
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

    if mps:
        train_x = train_x.astype("float32")
        train_y = train_y.astype("float32")
        test_x = test_x.astype("float32")
        test_y = test_y.astype("float32")

    train_dataset = MLCupDataset(train_x, train_y)
    train_loader = MLCupDataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MLCupDataset(test_x, test_y)
    test_loader = MLCupDataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def split_dataloader(dataloader, val_fraction=0.2, batch_size=32, seed=42):
    """
    Splits a PyTorch dataset into train and validation DataLoaders.
    
    Args:
        dataloader (torch.utils.data.DataLoader): original dataloader
        val_fraction (float): fraction of samples for the validation set (0 < val_fraction < 1)
        batch_size (int): batch size for loaders
        seed (int): random seed for reproducible split
    
    Returns:
        train_loader (DataLoader)
        val_loader   (DataLoader)
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
    train_data = torch.utils.data.Subset(dataset, train_idx)
    valid_data = torch.utils.data.Subset(dataset, valid_idx)

    train_loader_cv = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader_cv = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader_cv, valid_loader_cv

def apply_pca_on_X(dataset, n_components, standardize=True):
    """
    Returns a new Dataset with PCA applied to dataset.X
    """
    X = np.asarray(dataset.X)

    if standardize:
        X = StandardScaler().fit_transform(X)

    X_pca = PCA(n_components=n_components).fit_transform(X)

    new_dataset = deepcopy(dataset)
    new_dataset.X = X_pca
    return new_dataset


